#!/usr/bin/env python3
"""NIMINI — NVIDIA NIM chat proxy.

Persistent conversations + live translation + capability-aware routing with
recovery recipes, health probes, BYOK auth, and a tool ecosystem (image gen,
embed, rerank, OCR, MolMIM, web search, sandboxed code exec, math, TTS).

Single Python file, stdlib + `requests` + `cryptography` (+ optional `pypdf`).
Conversations and accounts persist in SQLite at ./chat.db. Tool-generated
images/audio go to ./artifacts/ and are served via /v1/artifact/{id}.

Routes
------
  GET  /                                -> nvidia_chatbot.html (the SPA)
  GET  /health                          -> "ok"
  GET  /models                          -> {items, aliases, categories}
  GET  /v1/tools                        -> tool metadata
  GET  /v1/tts/health  /voices          -> VibeVoice worker liveness
  GET  /v1/auth/me                      -> {user{id,username,is_admin,key{present,tail}}}
  GET  /v1/artifact/{id}                -> serves a stored artifact (auth-gated)
  GET  /conversations                   -> list user's conversations
  GET  /conversations/{id}              -> conversation + messages
  POST /conversations                   -> create
  PUT  /conversations/{id}              -> update title/model/system_prompt/params
  DEL  /conversations/{id}              -> delete
  POST /conversations/{id}/messages     -> append a message
  POST /v1/chat/completions             -> proxied chat (with recovery + tool loop)
  POST /v1/translate                    -> proxied translation (streaming)
  POST /v1/route                        -> classify + pick model
  POST /v1/compact                      -> summarize a transcript
  POST /v1/extract                      -> file content extraction (pdf/text)
  POST /v1/tts                          -> proxy to VibeVoice
  POST /v1/auth/register                -> {username, password, nvidia_api_key}
  POST /v1/auth/login                   -> {username, password}
  POST /v1/auth/logout                  -> drop session
  POST /v1/auth/key                     -> {nvidia_api_key} update stored key

Code layout (look for these markers to navigate):
  KEY LOADING · EVENTS · MODEL REGISTRY · HEALTH PROBES · ROUTING POLICY ·
  PRE-FLIGHT + TRUNCATION · RECOVERY RECIPES · DB · ARTIFACT STORE ·
  FILE EXTRACTION · AUTH · HTTP HANDLER · TOOL REGISTRY ·
  CODE_EXECUTE SANDBOX · MAIN

Usage
-----
    python serve.py                          # 127.0.0.1:8765, open registration
    python serve.py --host 0.0.0.0           # LAN, auto-generated reg token
    python serve.py --allowed-origins '*'    # disable CORS allowlist (dev only)
"""
import sys
import os
import io
import re
import json
import time
import base64
import secrets
import hashlib
import datetime
import sqlite3
import pathlib
import threading
import itertools
import argparse
from http.cookies import SimpleCookie
from urllib.parse import urlparse, parse_qs
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

try:
    import requests
except ImportError:
    sys.exit("[fatal] 'requests' is required: pip install requests")

try:
    import pypdf  # noqa: F401
    _PYPDF_OK = True
except ImportError:
    _PYPDF_OK = False

ROOT = pathlib.Path(__file__).parent.resolve()
ENV_PATH = ROOT / "a.env"
HTML_PATH = ROOT / "nvidia_chatbot.html"
DB_PATH = ROOT / "chat.db"
ARTIFACTS_DIR = ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)
NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
MAX_BODY = 32 * 1024 * 1024  # 32 MiB request cap (PDFs can be heavy)
MAX_UPSTREAM_BYTES = 5 * 1024 * 1024  # claw US-021: pre-flight body size limit

# ============================ KEY LOADING ============================

def normalize_key(raw: str) -> str:
    """Strip 'KEY=' prefix, 'Bearer ' prefix, and surrounding quotes/whitespace.
    Returns "" if nothing usable remains."""
    if not raw:
        return ""
    line = raw.strip().splitlines()[0].strip() if raw.strip() else ""
    if "=" in line and not line.lower().startswith(("nvapi-", "bearer ")):
        line = line.split("=", 1)[1].strip()
    line = line.strip('"').strip("'").strip()
    if line.lower().startswith("bearer "):
        line = line[7:].strip()
    return line


def load_key() -> str:
    """Load the OPTIONAL server-side NVIDIA key from a.env.
    BYOK mode: each user supplies their own key, so a.env is no longer required.
    When present, the server-side key is used only for background health probes
    (so /models can render without any user logged in). Returns "" if absent."""
    if not ENV_PATH.exists():
        return ""
    return normalize_key(ENV_PATH.read_text(encoding="utf-8"))


SERVER_KEY = load_key()
if SERVER_KEY:
    _mask = f"{SERVER_KEY[:8]}...{SERVER_KEY[-4:]}" if len(SERVER_KEY) > 12 else "****"
    print(f"[init] server-side health-probe key: {_mask}")
else:
    print(f"[init] no server-side key in {ENV_PATH} — health probes disabled "
          "until at least one user registers (their key is borrowed once).")
print(f"[init] db:  {DB_PATH}")
if not _PYPDF_OK:
    print("[init] pypdf not installed — PDF extraction disabled (pip install pypdf)")

# ============================ EVENTS, SEQ, REQ-ID ============================

_seq = itertools.count(1)
def next_seq() -> int:
    return next(_seq)


def make_req_id() -> str:
    return "req_" + secrets.token_hex(4)


def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")


def make_id() -> str:
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{secrets.token_hex(2)}"


def emit_event(event: str, *, req_id: str = "", source: str = "user", **fields):
    """Turn-lifecycle event with provenance (claw US-002 / US-014).

    All events go to stderr as JSON lines for ingestion. Fields beyond the
    standard keys are included as-is.
    """
    rec = {
        "seq": next_seq(),
        "ts": now_iso(),
        "req_id": req_id,
        "source": source,
        "event": event,
        **fields,
    }
    sys.stderr.write(f"[event] {json.dumps(rec, ensure_ascii=False)}\n")
    return rec

# ============================ MODEL REGISTRY ============================

# Capability registry. Used by select_model() and apply_quirks().
MODEL_REGISTRY = {
    "meta/llama-4-maverick-17b-128e-instruct": {
        "ctx_tokens": 1_000_000, "max_output": 8192,
        "vision": True,  "reasoning": False,
        # Maverick exhibits a strong "must call a tool or refuse" bias when tools
        # are bound. The tool_refusal_bias quirk activates a stronger no-tool-OK
        # prefix in apply_quirks(); without it, simple greetings (especially in
        # non-English languages, e.g. "안녕") return "I am not able to complete
        # this task..." even with tool_choice=auto.
        "quirks": ["tool_refusal_bias"],
    },
    "meta/llama-3.3-70b-instruct": {
        "ctx_tokens": 128_000, "max_output": 8192,
        "vision": False, "reasoning": False, "quirks": [],
    },
    # NOTE: meta/llama-3.1-405b-instruct removed — NVIDIA returned HTTP 410 Gone
    # for this model on 2026-05. If they bring it back, restore the entry and
    # re-add 405b to MODEL_ALIASES + CATEGORY_PREFERENCE['reasoning'].
    "deepseek-ai/deepseek-v4-flash": {
        "ctx_tokens": 1_000_000, "max_output": 8192,
        "vision": False, "reasoning": True,
        "quirks": ["dsv4_chat_template_required"],
    },
    "deepseek-ai/deepseek-v4-pro": {
        "ctx_tokens": 1_000_000, "max_output": 8192,
        "vision": False, "reasoning": True,
        "quirks": ["dsv4_chat_template_required", "nim_unstable"],
    },
}

# claw US-023: short aliases that resolve to canonical model ids.
MODEL_ALIASES = {
    "flash":    "deepseek-ai/deepseek-v4-flash",
    "pro":      "deepseek-ai/deepseek-v4-pro",
    "maverick": "meta/llama-4-maverick-17b-128e-instruct",
    "70b":      "meta/llama-3.3-70b-instruct",
}


def resolve_alias(name: str) -> str:
    if not name:
        return name
    if name in MODEL_REGISTRY:
        return name
    return MODEL_ALIASES.get(name, name)


# Per-category preference (primary first; remainder is fallback chain).
CATEGORY_PREFERENCE = {
    "general":   ["meta/llama-4-maverick-17b-128e-instruct",
                  "meta/llama-3.3-70b-instruct"],
    "reasoning": ["deepseek-ai/deepseek-v4-flash",
                  "deepseek-ai/deepseek-v4-pro"],
    "code":      ["deepseek-ai/deepseek-v4-flash",
                  "meta/llama-3.3-70b-instruct",
                  "deepseek-ai/deepseek-v4-pro"],
    "creative":  ["meta/llama-3.3-70b-instruct",
                  "meta/llama-4-maverick-17b-128e-instruct"],
    "vision":    ["meta/llama-4-maverick-17b-128e-instruct"],
}

# ============================ HEALTH PROBES ============================

_health: dict = {}     # model -> {ok: bool, ts: float, error: str|None, latency_ms: int|None}
_health_lock = threading.Lock()
_HEALTH_TTL = 30 * 60   # 30 minutes
_HEALTH_PROBE_TIMEOUT = 12


def model_health(model: str) -> dict:
    """Returns {ok, ts, error, latency_ms} or {} if unknown/stale."""
    with _health_lock:
        rec = dict(_health.get(model) or {})
    if not rec:
        return {}
    if time.time() - rec.get("ts", 0) > _HEALTH_TTL:
        return {}
    return rec


def health_snapshot() -> list:
    out = []
    for m, meta in MODEL_REGISTRY.items():
        h = model_health(m)
        out.append({
            "id": m,
            "ctx_tokens": meta["ctx_tokens"],
            "vision": meta["vision"],
            "reasoning": meta["reasoning"],
            "quirks": list(meta.get("quirks") or []),
            "healthy": (None if not h else bool(h.get("ok"))),
            "last_check": h.get("ts"),
            "last_error": h.get("error"),
            "latency_ms": h.get("latency_ms"),
        })
    return out


def probe_model(model: str):
    """Background health probe. Uses SERVER_KEY (from a.env) if available,
    otherwise borrows the most-recently-active user's key. If neither exists,
    skips the probe (the model entry will report healthy=None until a key
    becomes available)."""
    key = SERVER_KEY or _borrow_any_user_key()
    if not key:
        return  # nothing to probe with; UI will show "unknown"
    started = time.time()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 4, "temperature": 0, "stream": False,
    }
    if "dsv4_chat_template_required" in MODEL_REGISTRY.get(model, {}).get("quirks", []):
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    err = None
    ok = False
    try:
        r = requests.post(NVIDIA_URL, headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }, json=payload, timeout=_HEALTH_PROBE_TIMEOUT)
        ok = r.ok
        if not ok:
            err = f"HTTP {r.status_code}"
            body = (r.text or "")[:120]
            if body:
                err += f" {body!r}"
        r.close()
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:80]}"
    latency_ms = int((time.time() - started) * 1000)
    with _health_lock:
        _health[model] = {"ok": ok, "ts": time.time(),
                          "error": err, "latency_ms": latency_ms}
    emit_event("health.probe", source="healthcheck",
               model=model, ok=ok, latency_ms=latency_ms, error=err)


def probe_all_async():
    for m in MODEL_REGISTRY:
        threading.Thread(target=probe_model, args=(m,), daemon=True).start()


def health_loop():
    """Background re-probe every TTL/2."""
    while True:
        time.sleep(_HEALTH_TTL // 2)
        for m in list(MODEL_REGISTRY.keys()):
            try:
                probe_model(m)
            except Exception as e:
                emit_event("health.probe.failed", source="healthcheck",
                           model=m, error=str(e)[:80])

# ============================ ROUTING POLICY ============================

ROUTER_MODEL = "meta/llama-3.3-70b-instruct"

ROUTER_SYSTEM = (
    "You are a routing classifier. Read the user's message and classify it into "
    "EXACTLY ONE category. Output a JSON object only — no preamble, no markdown fences.\n\n"
    "Categories:\n"
    "- \"reasoning\": math, logic, multi-step proofs, complex analysis, scientific problems\n"
    "- \"code\": programming, code review, debugging, software engineering\n"
    "- \"creative\": creative writing, fiction, poetry, brainstorming, storytelling\n"
    "- \"general\": casual chat, factual Q&A, definitions, opinions, everyday requests\n\n"
    "Output: {\"category\":\"<one>\",\"reason\":\"<<=40 char phrase>\",\"confidence\":<0..1>}"
)


def estimate_message_tokens(messages) -> int:
    """Rough estimate. ~3 chars/token; +1500 per image."""
    total = 0
    for m in messages or []:
        c = m.get("content")
        if isinstance(c, str):
            total += len(c) // 3
        elif isinstance(c, list):
            for p in c:
                t = p.get("type")
                if t == "text":
                    total += len(p.get("text") or "") // 3
                elif t == "image_url":
                    total += 1500
    return total


def health_aware_filter(candidates: list) -> tuple:
    """Returns (healthy_first, unknown_or_unhealthy). Caller can prefer healthy."""
    healthy, others = [], []
    for m in candidates:
        h = model_health(m)
        if h.get("ok") is True:
            healthy.append(m)
        else:
            others.append(m)
    return healthy, others


def select_model(category: str, has_image: bool, est_input_tokens: int,
                 requested_max_output: int):
    """Returns (model, fallback_chain, notes)."""
    notes = []
    cat = category if category in CATEGORY_PREFERENCE else "general"
    if cat != category:
        notes.append(f"unknown category {category!r} → general")

    if has_image and cat != "vision":
        notes.append("image attached → forced vision lane")
        cat = "vision"

    candidates = list(CATEGORY_PREFERENCE[cat])

    # vision capability filter
    if has_image:
        kept = [m for m in candidates if MODEL_REGISTRY.get(m, {}).get("vision")]
        if not kept:
            kept = [m for m in MODEL_REGISTRY if MODEL_REGISTRY[m].get("vision")]
            notes.append("category had no vision-capable model; widened")
        candidates = kept

    # ctx capacity filter
    needed = est_input_tokens + max(0, requested_max_output)
    fits = [m for m in candidates if MODEL_REGISTRY.get(m, {}).get("ctx_tokens", 0) >= needed]
    if not fits:
        fits = sorted(
            [m for m in MODEL_REGISTRY if MODEL_REGISTRY[m].get("ctx_tokens", 0) >= needed],
            key=lambda m: -MODEL_REGISTRY[m]["ctx_tokens"],
        )
        if fits:
            notes.append(f"ctx need {needed} → widened beyond category")
        else:
            largest = max(MODEL_REGISTRY, key=lambda m: MODEL_REGISTRY[m]["ctx_tokens"])
            fits = [largest]
            notes.append(f"ctx need {needed} exceeds all registered models")
    candidates = fits

    # health filter — prefer healthy, but don't outright drop unknowns (TTL stale)
    healthy, others = health_aware_filter(candidates)
    if healthy and others:
        notes.append(f"reordered by health: {len(healthy)} healthy first")
        candidates = healthy + others
    elif not healthy and others:
        # All unhealthy or unknown — proceed but flag.
        if any(model_health(m).get("ok") is False for m in others):
            notes.append("all candidates currently flagged unhealthy")

    chosen = candidates[0]
    fallback = candidates[1:]
    return chosen, fallback, notes


_TOOL_PREFIX_TAG = "[[tool_use_default_v1]]"


def _build_tool_use_prefix(model: str) -> str:
    """Build the system-message prefix that tells the model 'tools are optional'.
    Stronger language for models tagged with tool_refusal_bias."""
    quirks = MODEL_REGISTRY.get(model, {}).get("quirks", [])
    base = (
        "You have access to tools, but using them is OPTIONAL. "
        "Call a tool only when it directly helps answer the user's request. "
        "When no tool is appropriate — including greetings, casual conversation, "
        "or anything answerable from your own knowledge — respond directly in "
        "natural language. Not calling a tool is a normal, expected response and "
        "is NEVER grounds for refusal."
    )
    if "tool_refusal_bias" in quirks:
        base += (
            " Specifically, do NOT respond with phrases like \"I cannot complete "
            "this task\", \"this falls outside the scope of my functions\", or "
            "\"I am not able to help with this\" merely because no listed tool "
            "was applicable. Such phrasing is incorrect — answer the user's "
            "actual question instead."
            " When you DO call a tool, you MUST use the structured tool_calls "
            "API field. NEVER emit a JSON object describing the call (such as "
            "{\"type\":\"function\",\"name\":...,\"parameters\":...}) inside your "
            "text response — that is not a tool call, it is just text the user "
            "will see as raw JSON."
            " Do NOT proactively call media-generation tools (image generation, "
            "etc.) for greetings, smalltalk, or any input where the user did "
            "not explicitly request that output medium. For \"hi\", \"hello\", "
            "or similar casual greetings in any language, respond in plain text only."
            " Respond in the same language as the user's most recent message. "
            "If their message mixes languages, mirror that mix; do not unilaterally "
            "switch to English (or any other language) unless explicitly asked."
        )
    return f"{_TOOL_PREFIX_TAG}\n{base}"


def _strip_tool_prefix(payload: dict) -> None:
    """Remove any prior tool-use prefix system message (idempotency)."""
    msgs = payload.get("messages") or []
    payload["messages"] = [
        m for m in msgs
        if not (m.get("role") == "system"
                and isinstance(m.get("content"), str)
                and m["content"].startswith(_TOOL_PREFIX_TAG))
    ]


def _inject_tool_use_prefix(payload: dict, model: str) -> None:
    """If tools are bound, prepend a system message permitting no-tool responses.
    Idempotent: removes any prior injection before re-adding for the current model
    (so fallback to a different model gets the right strictness level)."""
    if not payload.get("tools"):
        return
    _strip_tool_prefix(payload)
    prefix_msg = {"role": "system", "content": _build_tool_use_prefix(model)}
    payload["messages"] = [prefix_msg] + (payload.get("messages") or [])


_TEXT_TOOLCALL_COUNTER = itertools.count(1)


def _try_parse_text_tool_calls(content) -> list | None:
    """Recover tool calls that the model emitted as JSON text in the content
    field instead of via the structured tool_calls field. This is a known
    Llama 4 Maverick quirk: the model writes
        {"type": "function", "name": "tts", "parameters": {...}}
    as plain content, which the tool loop never sees. Returns a list of
    OpenAI-format tool_calls entries on success, or None if content is not
    a recognizable text-format tool call.

    Recognized JSON shapes (single object or array of objects):
      {"type": "function", "name": "X", "parameters": {...}}
      {"name": "X", "parameters": {...}}
      {"name": "X", "arguments": {...}}
    """
    if not isinstance(content, str):
        return None
    s = content.strip()
    # Strip markdown code fences if the model wrapped the JSON
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
        s = s.strip()
    if not (s.startswith("{") or s.startswith("[")):
        return None
    try:
        obj = json.loads(s)
    except (ValueError, json.JSONDecodeError):
        return None

    items = obj if isinstance(obj, list) else [obj]
    out = []
    for item in items:
        if not isinstance(item, dict):
            return None
        # Accept the "type=function" wrapper or a bare {name, parameters}
        if item.get("type") == "function" and isinstance(item.get("function"), dict):
            inner = item["function"]
            name = inner.get("name")
            args = inner.get("arguments") or inner.get("parameters")
        else:
            name = item.get("name")
            args = item.get("parameters") or item.get("arguments")
        if not isinstance(name, str) or not name:
            return None
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return None
        out.append({
            "id": f"call_text_{next(_TEXT_TOOLCALL_COUNTER)}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        })
    return out or None


def apply_quirks(payload: dict, model: str) -> dict:
    quirks = MODEL_REGISTRY.get(model, {}).get("quirks", [])
    if "dsv4_chat_template_required" in quirks:
        if "chat_template_kwargs" not in payload:
            payload["chat_template_kwargs"] = {"enable_thinking": True, "thinking": True}
    cap = MODEL_REGISTRY.get(model, {}).get("max_output")
    if cap and isinstance(payload.get("max_tokens"), int) and payload["max_tokens"] > cap:
        payload["max_tokens"] = cap
    # Tool-use behavioral default: when tools are bound, explicitly permit no-tool
    # responses. Some models (e.g. Llama 4 Maverick) treat tool-bound contexts as
    # "must call tool or refuse" and reject simple greetings without this. Safe
    # for all models — non-quirky models get the base permissive language only.
    _inject_tool_use_prefix(payload, model)
    return payload


def explain_http_error(status: int, body_text: str, req_id: str = "") -> str:
    snippet = (body_text or "").strip()[:240]
    base = {
        400: "Bad request. Check model compatibility.",
        401: "API key rejected. Verify the key in a.env (or your stored BYOK).",
        403: "Permission denied. The key may not have access to this model.",
        404: "Model not found.",
        408: "Request timeout.",
        413: "Request body too large. Reduce attachments or context.",
        422: "Input format rejected.",
        429: "Rate limit exceeded.",
        500: "NVIDIA internal error.",
        502: "NVIDIA gateway error.",
        503: "Model temporarily unavailable.",
        504: "Response timed out.",
    }.get(status, f"HTTP {status}")
    parts = [f"[{req_id}] " + base if req_id else base]
    if snippet:
        parts.append(f"(server response: {snippet})")
    return " ".join(parts)

# ============================ PRE-FLIGHT + TRUNCATION ============================

def payload_byte_size(payload: dict) -> int:
    try:
        return len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    except Exception:
        return 0


def truncate_history_in_place(payload: dict, target_bytes: int, notes: list) -> bool:
    """Drop oldest non-system messages (keeping last user message) until under target.
    Returns True if any truncation happened."""
    msgs = payload.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 3:
        return False
    sys_msgs = [m for m in msgs if m.get("role") == "system"]
    other = [m for m in msgs if m.get("role") != "system"]
    if len(other) < 2:
        return False
    truncated = 0
    while len(other) > 1:
        candidate = sys_msgs + other
        size = len(json.dumps({**payload, "messages": candidate}, ensure_ascii=False).encode("utf-8"))
        if size <= target_bytes:
            break
        # drop oldest non-system
        other.pop(0)
        truncated += 1
    if truncated == 0:
        return False
    payload["messages"] = sys_msgs + other
    notes.append(f"truncated {truncated} oldest message(s) to fit {target_bytes}B")
    return True

# ============================ RECOVERY RECIPES ============================

class RecoveryLedger:
    """Records each recovery step taken for one upstream call.

    Each entry: {recipe, attempt, state, ts, summary}
    state ∈ {"started", "succeeded", "failed", "escalated"}

    The full ledger is returned in error response bodies as `error.ledger`.
    Headers carry only the count (`X-Recovery-Steps`) — base64 JSON in a
    header risks blowing past the 8 KiB nginx default header buffer when a
    deep fallback chain accumulates many entries.
    """
    def __init__(self):
        self.entries = []
    def add(self, **kwargs):
        kwargs.setdefault("ts", now_iso())
        self.entries.append(kwargs)
    def step_count(self) -> int:
        return len(self.entries)


def plan_recovery(status_or_exc, payload: dict):
    """
    Decides what recipe to apply for a given failure.
    Returns (recipe_id, action, sleep_seconds) where action ∈
        "retry_same"     — retry with same model after sleep
        "fallback"       — move to next model in chain
        "truncate_retry" — shrink payload, retry same model
        None             — no recipe; escalate immediately
    """
    if isinstance(status_or_exc, int):
        s = status_or_exc
        if s in (502, 503, 504):
            return ("upstream_5xx_immediate_fallback", "fallback", 0.0)
        if s == 500:
            return ("upstream_500_short_backoff_then_fallback", "fallback", 1.5)
        if s == 429:
            return ("rate_limit_backoff_then_retry", "retry_same", 3.0)
        if s == 408:
            return ("timeout_short_backoff", "retry_same", 1.0)
        if s == 413:
            return ("payload_too_large_truncate", "truncate_retry", 0.0)
        return None
    # exception path (network)
    return ("upstream_network_backoff_then_fallback", "fallback", 1.0)

# ============================ DB ============================

_db_lock = threading.RLock()  # reentrant: callers may hold it across nested helpers
_db_local = threading.local()
import atexit


def _close_thread_local_conn() -> None:
    """Close the current thread's sqlite connection if any. Called from
    Handler.finish() at the end of each request. Without this, ThreadingHTTPServer
    leaks one connection (and its fd) per request thread; over hours that
    exhausts the process's fd table."""
    conn = getattr(_db_local, "conn", None)
    if conn is None:
        return
    try:
        conn.close()
    except Exception:
        pass
    _db_local.conn = None


def db_conn():
    """Return a thread-local SQLite connection, creating it on first use.

    ThreadingHTTPServer spawns a new thread per request, so each request reuses
    the connection across all of its DB ops (typical request: 2–5 ops). This
    avoids the open + 3×PRAGMA overhead that the previous "new connection per
    call" pattern paid on every single query.

    The connection is closed at the end of each request via Handler.finish()
    calling _close_thread_local_conn — without that, request threads die but
    their connections live on, leaking fds."""
    conn = getattr(_db_local, "conn", None)
    if conn is not None:
        return conn
    conn = sqlite3.connect(DB_PATH, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    _db_local.conn = conn
    return conn


@atexit.register
def _atexit_close_main_thread_conn() -> None:
    """Close the main thread's connection (used by init_db / health probe loop)
    at process shutdown so WAL has a chance to checkpoint cleanly. Worker
    thread connections are closed individually by Handler.finish."""
    _close_thread_local_conn()


def init_db():
    with _db_lock, db_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY, title TEXT, model TEXT, system_prompt TEXT,
                params TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL, role TEXT NOT NULL,
                content TEXT, reasoning_content TEXT, created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_msg_conv     ON messages(conversation_id);
            CREATE INDEX IF NOT EXISTS idx_msg_created  ON messages(conversation_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_conv_updated ON conversations(updated_at DESC);

            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                is_admin INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_exp  ON sessions(expires_at);
        """)
        # Migration: add user_id column to conversations if missing.
        # Existing rows are left with NULL user_id; the first user to register
        # as admin will adopt them via _adopt_orphan_conversations().
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(conversations)").fetchall()]
        if "user_id" not in cols:
            conn.execute("ALTER TABLE conversations ADD COLUMN user_id TEXT")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id, updated_at DESC)")
        # Migration: BYOK — every user supplies their own NVIDIA NIM key at
        # registration, used for all upstream calls made on their behalf.
        # Encrypted at rest with a KEK derived from the user's password
        # (envelope encryption); see _envelope_* helpers below. The legacy
        # `nvidia_api_key` plaintext column is kept for backward compat with
        # accounts that pre-date envelope encryption — it is migrated into
        # `nvidia_api_key_enc` opportunistically on the next login and then
        # NULLed out.
        ucols = [r["name"] for r in conn.execute("PRAGMA table_info(users)").fetchall()]
        if "nvidia_api_key" not in ucols:
            conn.execute("ALTER TABLE users ADD COLUMN nvidia_api_key TEXT")
        if "nvidia_api_key_enc" not in ucols:
            conn.execute("ALTER TABLE users ADD COLUMN nvidia_api_key_enc BLOB")
        # Artifacts (sandbox-generated images, image-gen results, TTS audio).
        # Stored on disk under ARTIFACTS_DIR; this row is the owner / mime
        # index used by GET /v1/artifact/{id} to enforce per-user access.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS artifacts (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                mime TEXT NOT NULL,
                size INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_user ON artifacts(user_id, created_at DESC)")


init_db()


def conv_get(cid: str, user_id: str):
    if not ID_RE.match(cid):
        return None
    with db_conn() as conn:
        row = conn.execute(
            "SELECT * FROM conversations WHERE id = ? AND user_id = ?",
            (cid, user_id),
        ).fetchone()
        if not row:
            return None
        msgs = conn.execute(
            "SELECT id, role, content, reasoning_content, created_at "
            "FROM messages WHERE conversation_id = ? ORDER BY id ASC", (cid,),
        ).fetchall()
    out = dict(row)
    out["params"] = json.loads(out["params"]) if out["params"] else None
    out["messages"] = [dict(m) for m in msgs]
    return out


def conv_meta(cid: str, user_id: str):
    if not ID_RE.match(cid):
        return None
    with db_conn() as conn:
        row = conn.execute(
            "SELECT c.id, c.title, c.model, c.created_at, c.updated_at, "
            "       (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS n_messages "
            "FROM conversations c WHERE c.id = ? AND c.user_id = ?", (cid, user_id)
        ).fetchone()
    return dict(row) if row else None


def conv_list(user_id: str):
    with db_conn() as conn:
        rows = conn.execute("""
            SELECT c.id, c.title, c.model, c.created_at, c.updated_at,
                   (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS n_messages
            FROM conversations c WHERE c.user_id = ?
            ORDER BY c.updated_at DESC
        """, (user_id,)).fetchall()
    return [dict(r) for r in rows]


def conv_create(payload: dict, user_id: str) -> dict:
    cid = make_id()
    ts = now_iso()
    # claw US-015: identity completeness — populate immediately, use placeholders if unknown
    title = (payload.get("title") or "").strip() or None
    model = payload.get("model") or None
    system_prompt = payload.get("system_prompt") or None
    params = payload.get("params")
    params_json = json.dumps(params, ensure_ascii=False) if params is not None else None
    with _db_lock, db_conn() as conn:
        conn.execute(
            "INSERT INTO conversations (id, title, model, system_prompt, params, "
            "created_at, updated_at, user_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (cid, title, model, system_prompt, params_json, ts, ts, user_id),
        )
    emit_event("conversation.created", source="user", conversation_id=cid,
               title=title, model=model, user_id=user_id)
    return conv_meta(cid, user_id)


def conv_update(cid: str, payload: dict, user_id: str) -> bool:
    if not ID_RE.match(cid):
        return False
    fields, values = [], []
    for col in ("title", "model", "system_prompt"):
        if col in payload:
            fields.append(f"{col} = ?")
            v = payload[col]
            values.append(v if v is None else str(v))
    if "params" in payload:
        fields.append("params = ?")
        values.append(json.dumps(payload["params"], ensure_ascii=False) if payload["params"] is not None else None)
    fields.append("updated_at = ?")
    values.append(now_iso())
    values.extend([cid, user_id])
    with _db_lock, db_conn() as conn:
        cur = conn.execute(
            f"UPDATE conversations SET {', '.join(fields)} "
            "WHERE id = ? AND user_id = ?", values,
        )
    return cur.rowcount > 0


def conv_delete(cid: str, user_id: str) -> bool:
    if not ID_RE.match(cid):
        return False
    with _db_lock, db_conn() as conn:
        cur = conn.execute(
            "DELETE FROM conversations WHERE id = ? AND user_id = ?",
            (cid, user_id),
        )
    if cur.rowcount > 0:
        emit_event("conversation.deleted", source="user",
                   conversation_id=cid, user_id=user_id)
    return cur.rowcount > 0


def message_append(cid: str, role: str, content: str, user_id: str,
                   reasoning_content: str = None) -> int:
    if not ID_RE.match(cid):
        raise ValueError("invalid conversation id")
    if role not in ("user", "assistant", "system"):
        raise ValueError(f"invalid role: {role!r}")
    ts = now_iso()
    with _db_lock, db_conn() as conn:
        # Verify ownership before any write — prevents cross-user message injection
        row = conn.execute(
            "SELECT id, title FROM conversations WHERE id = ? AND user_id = ?",
            (cid, user_id),
        ).fetchone()
        if not row:
            raise LookupError("conversation not found")
        cur = conn.execute(
            "INSERT INTO messages (conversation_id, role, content, reasoning_content, created_at) "
            "VALUES (?, ?, ?, ?, ?)", (cid, role, content, reasoning_content, ts),
        )
        msg_id = cur.lastrowid
        new_title = row["title"]
        if not new_title and role == "user" and content:
            first = content.strip().splitlines()[0]
            new_title = first[:60] + ("…" if len(first) > 60 else "")
            conn.execute("UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                         (new_title, ts, cid))
        else:
            conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (ts, cid))
    return msg_id

# ============================ ARTIFACT STORE ============================

# Tool executors append artifacts to ctx["artifacts"]. The previous design
# carried the binary as base64 inside the ctx dict (and through the response
# JSON, and into chat.db saved messages) — a 5 MB image became ~6.6 MB of
# base64 ballooning every layer. Now we write bytes to disk under
# ARTIFACTS_DIR/{id}.{ext} and emit a URL the client fetches separately
# (auth-gated by /v1/artifact/{id}).

_EXT_FROM_MIME = {
    "image/png":  "png",  "image/jpeg": "jpg", "image/jpg": "jpg",
    "image/webp": "webp", "image/gif":  "gif",
    "audio/wav":  "wav",  "audio/mpeg": "mp3", "audio/ogg": "ogg",
}


def save_artifact(ctx: dict, kind: str, mime: str, data: bytes,
                  **extra) -> dict:
    """Persist `data` to disk + register in DB; return the metadata dict to
    append to ctx["artifacts"]. Required ctx key: "user_id" (set by the
    caller before invoking any tool executor)."""
    user_id = ctx.get("user_id") or ""
    if not user_id:
        raise RuntimeError("save_artifact: ctx missing user_id")
    aid = secrets.token_urlsafe(16)
    ext = _EXT_FROM_MIME.get((mime or "").lower(), "bin")
    path = ARTIFACTS_DIR / f"{aid}.{ext}"
    path.write_bytes(data)
    ts = now_iso()
    with _db_lock, db_conn() as conn:
        conn.execute(
            "INSERT INTO artifacts (id, user_id, kind, mime, size, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (aid, user_id, kind, mime, len(data), ts),
        )
    out = {
        "id": aid,
        "kind": kind,
        "format": mime,
        "url": f"/v1/artifact/{aid}",
        "size": len(data),
    }
    out.update(extra)
    return out


def _artifact_lookup(aid: str, user_id: str) -> tuple:
    """Return (path, mime) for an artifact owned by user_id, or (None, None)
    on miss / cross-user attempt."""
    if not aid or not user_id:
        return None, None
    # Path traversal guard: ids are token_urlsafe so they never contain "/" or
    # "..". Reject anything that looks otherwise.
    if "/" in aid or ".." in aid or len(aid) > 64:
        return None, None
    with db_conn() as conn:
        row = conn.execute(
            "SELECT mime FROM artifacts WHERE id = ? AND user_id = ?",
            (aid, user_id),
        ).fetchone()
    if not row:
        return None, None
    mime = row["mime"]
    ext = _EXT_FROM_MIME.get((mime or "").lower(), "bin")
    path = ARTIFACTS_DIR / f"{aid}.{ext}"
    if not path.is_file():
        return None, None
    return path, mime


# ============================ FILE EXTRACTION ============================

TEXT_EXTENSIONS = frozenset({
    "txt", "md", "markdown", "rst", "tex", "ltx", "bib", "cls", "sty",
    "json", "yaml", "yml", "toml", "ini", "cfg", "conf", "env", "properties",
    "py", "pyi", "ipynb", "js", "mjs", "cjs", "jsx", "ts", "tsx",
    "rs", "go", "java", "kt", "swift", "scala", "groovy", "clj",
    "c", "cc", "cpp", "cxx", "h", "hh", "hpp", "hxx", "cs", "fs",
    "rb", "php", "pl", "pm", "lua", "tcl", "r", "jl", "m", "f", "f90",
    "sh", "bash", "zsh", "fish", "ps1", "bat", "cmd",
    "html", "htm", "xhtml", "css", "scss", "sass", "less",
    "xml", "svg", "vue", "svelte", "astro",
    "csv", "tsv", "log", "gitignore", "gitattributes",
    "sql", "graphql", "gql", "proto", "thrift",
    "dockerfile", "containerfile", "makefile", "cmake",
    "vim", "el", "lisp", "scm", "rkt", "hs", "ml", "mli", "elm",
    "asm", "s", "wat", "wgsl", "glsl", "hlsl",
    "diff", "patch",
})


def _decode_text_bytes(data: bytes) -> tuple:
    """Try a few encodings. Returns (text, encoding) or raises."""
    for enc in ("utf-8", "utf-8-sig", "utf-16", "cp949", "euc-kr", "latin-1"):
        try:
            return data.decode(enc), enc
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("text", data, 0, len(data), "no encoding worked")


def extract_pdf(data: bytes) -> dict:
    if not _PYPDF_OK:
        raise RuntimeError("pypdf not installed")
    import pypdf as _pp
    reader = _pp.PdfReader(io.BytesIO(data))
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception as e:
            pages.append(f"[page extract error: {e}]")
    return {"text": "\n\n".join(pages).strip(), "n_pages": len(pages)}

# ============================ AUTH ============================
# REGISTRATION_TOKEN (set in main from CLI/env): if non-empty, required as
# X-Registration-Token header on /v1/auth/register. This prevents random
# external signups while the server is exposed; leave empty for open signup
# on a trusted network.
REGISTRATION_TOKEN: str = ""

# CORS allowlist. Reflecting any Origin with Allow-Credentials: true (the
# previous behavior) lets a malicious page on http://evil.example call this
# proxy from the user's browser with the user's session cookie. The default
# list is Capacitor / Ionic mobile webview origins (the original motivation
# for cross-origin support); add others via --allowed-origins or the
# NIMINI_ALLOWED_ORIGINS env var (comma-separated). The literal "*" disables
# the allowlist entirely (insecure — only for local dev).
ALLOWED_ORIGINS: list = [
    "https://localhost",
    "capacitor://localhost",
    "ionic://localhost",
]
SESSION_COOKIE_NAME = "nim_session"
SESSION_TTL_SECONDS = 30 * 24 * 3600   # 30 days
SESSION_SLIDE_THRESHOLD = 3600          # refresh expiry if <1h since last refresh
PUBLIC_PATHS = {
    "/health",
    # HTML page itself must be reachable so the frontend can run checkAuth()
    # and show the login modal. The HTML is static; all data APIs remain gated.
    "/", "/index.html", "/chat",
    "/v1/auth/register", "/v1/auth/login",
    # TTS metadata endpoints: probed on page load (before login). Worker liveness
    # and the static voice list are not sensitive — actual synthesis (POST /v1/tts)
    # remains session-gated.
    "/v1/tts/health", "/v1/tts/voices",
}

# Password hashing parameters (scrypt — stdlib, no external deps).
# N=2^17 (~128 MiB per hash) follows OWASP 2024 guidance for scrypt. RFC 7914
# defaults (N=2^14) are too low for current hardware. The cost is paid only on
# register / login / key change, so an extra ~250 ms per such call is fine; a
# single VPS handles ~8 concurrent registrations comfortably at this setting.
# Argon2id would be preferable but is not in the stdlib — out of scope for the
# zero-dep philosophy. If a user upgrades, set _SCRYPT_N higher and bump the
# stored "version" once we add one (TODO: per-user param record for migration).
_SCRYPT_N = 2 ** 17
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 64


def _scrypt_derive(password: str, salt_hex: str | None = None) -> tuple:
    """Run scrypt(password, salt) once. Returns (full_64_byte_output, salt_hex).
    Caller splits: first 32 bytes = stored password hash, last 32 bytes = KEK
    (envelope-encryption key derivation). Doing both from one scrypt call
    keeps login cheap; ~2.5 s on a free-tier VPS at N=2^17."""
    if salt_hex is None:
        salt = secrets.token_bytes(16)
        salt_hex = salt.hex()
    else:
        salt = bytes.fromhex(salt_hex)
    # N=2^17 with r=8 needs ~128 MiB; the stdlib default maxmem=32 MiB rejects
    # it. Allocate up to 256 MiB. On hosts that can't honor that, login fails
    # fast at this point with a clear error rather than deeper down.
    derived = hashlib.scrypt(
        password.encode("utf-8"), salt=salt,
        n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=_SCRYPT_DKLEN,
        maxmem=256 * 1024 * 1024,
    )
    return derived, salt_hex


def _hash_password(password: str, salt_hex: str | None = None) -> tuple:
    """Return (hash_hex, salt_hex) for storage. Stores ONLY the first 32 bytes
    of the scrypt output as the password hash; the last 32 bytes are reserved
    for KEK derivation at login time (and never stored)."""
    derived, salt_hex = _scrypt_derive(password, salt_hex)
    return derived[:32].hex(), salt_hex


def _verify_password(password: str, hash_hex: str, salt_hex: str) -> tuple:
    """Return (ok, scrypt_output_64_bytes_or_None).

    Supports two stored-hash formats:
      - 64-char hex (32 bytes): new format, just the password-verification half
      - 128-char hex (64 bytes): legacy format, full scrypt output

    The legacy format predates envelope encryption; opportunistic migration
    happens at login (caller switches the stored hash to the new 32-byte form
    once they verify with it)."""
    try:
        derived, _ = _scrypt_derive(password, salt_hex)
    except (ValueError, TypeError):
        return False, None
    if len(hash_hex) == 64:
        ok = secrets.compare_digest(derived[:32].hex(), hash_hex)
    elif len(hash_hex) == 128:
        ok = secrets.compare_digest(derived.hex(), hash_hex)
    else:
        ok = False
    return (True, derived) if ok else (False, None)


# Envelope-encryption helpers. Fernet wraps AES-128-CBC + HMAC-SHA256 with a
# version byte and timestamp; we only need the AEAD primitive but Fernet's
# stable, well-tested API is fine for ~70-byte NVIDIA keys. The KEK is the
# last 32 bytes of the scrypt password derivation — never stored, only
# present in memory for the lifetime of a session.
def _envelope_encrypt(plaintext: str, kek: bytes) -> bytes:
    from cryptography.fernet import Fernet
    f = Fernet(base64.urlsafe_b64encode(kek))
    return f.encrypt(plaintext.encode("utf-8"))


def _envelope_decrypt(ciphertext: bytes, kek: bytes) -> str:
    from cryptography.fernet import Fernet, InvalidToken
    f = Fernet(base64.urlsafe_b64encode(kek))
    return f.decrypt(ciphertext).decode("utf-8")


# Username policy: 3–32 chars, alphanumeric + underscore + hyphen + dot.
# Password policy: at least 8 chars, no upper bound enforced here (scrypt is fine).
_USERNAME_RE = re.compile(r"^[A-Za-z0-9_.\-]{3,32}$")
_MIN_PASSWORD_LEN = 8

# NVIDIA NIM keys observed in the wild start with "nvapi-" followed by a long
# token. We don't enforce the prefix strictly (NVIDIA could change it), but we
# do require something that looks like an opaque secret of reasonable length.
_NVIDIA_KEY_RE = re.compile(r"^[A-Za-z0-9_.\-]{20,200}$")


def validate_nvidia_key(key: str, timeout: int = 10) -> tuple[bool, str]:
    """Verify a NIM API key by making a tiny chat completion call.
    Returns (ok, detail). detail is a short human-readable error on failure."""
    if not key or not isinstance(key, str):
        return False, "empty key"
    key = normalize_key(key)
    if not _NVIDIA_KEY_RE.match(key):
        return False, "key format looks wrong (expected ~20+ char token)"
    try:
        # 4-token ping against the smallest fast model. /v1/models on this
        # endpoint requires auth too, but a chat completion is the actual
        # auth path the user will exercise — better signal.
        r = requests.post(NVIDIA_URL, headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }, json={
            "model": "meta/llama-3.3-70b-instruct",
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 4, "temperature": 0, "stream": False,
        }, timeout=timeout)
    except requests.RequestException as e:
        return False, f"network error reaching NVIDIA: {type(e).__name__}"
    if r.status_code == 200:
        return True, "ok"
    if r.status_code == 401:
        return False, "NVIDIA rejected the key (401 Unauthorized)"
    if r.status_code == 403:
        return False, "key has no permission for this model (403)"
    return False, f"validation call returned HTTP {r.status_code}"


def _user_create(username: str, password: str, nvidia_api_key: str,
                 is_admin: bool = False) -> dict:
    """Create a new user. Raises ValueError on validation failure,
    sqlite3.IntegrityError on username collision. Caller is responsible for
    validating nvidia_api_key against NVIDIA before invoking this function.

    The NVIDIA key is envelope-encrypted with a KEK derived from the password
    in the same scrypt call as the password hash (one scrypt = both halves)."""
    if not isinstance(username, str) or not _USERNAME_RE.match(username):
        raise ValueError("invalid username (3–32 chars, [A-Za-z0-9_.-])")
    if not isinstance(password, str) or len(password) < _MIN_PASSWORD_LEN:
        raise ValueError(f"password must be at least {_MIN_PASSWORD_LEN} characters")
    nvidia_api_key = normalize_key(nvidia_api_key or "")
    if not _NVIDIA_KEY_RE.match(nvidia_api_key):
        raise ValueError("invalid NVIDIA API key format")
    uid = secrets.token_hex(16)
    derived, pw_salt = _scrypt_derive(password)
    pw_hash = derived[:32].hex()
    kek = derived[32:]
    enc = _envelope_encrypt(nvidia_api_key, kek)
    ts = now_iso()
    with _db_lock, db_conn() as conn:
        conn.execute(
            "INSERT INTO users (id, username, password_hash, password_salt, "
            "is_admin, created_at, nvidia_api_key_enc) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (uid, username, pw_hash, pw_salt, 1 if is_admin else 0, ts, enc),
        )
    emit_event("user.created", source="user", user_id=uid,
               username=username, is_admin=is_admin)
    return {"id": uid, "username": username, "is_admin": is_admin,
            "created_at": ts, "_kek": kek, "_api_key": nvidia_api_key}


class KeyUpdateError(Exception):
    """Raised by _user_set_key with a specific reason so the HTTP layer can
    surface it instead of returning a generic 'could not update key'."""


def _user_set_key(user_id: str, nvidia_api_key: str, kek: bytes) -> None:
    """Re-encrypt and store a new NVIDIA key for a user. Caller must validate
    nvidia_api_key against NVIDIA first AND supply the user's current KEK
    (from the session cache). Raises KeyUpdateError on every failure mode
    so the caller can produce an actionable error message."""
    nvidia_api_key = normalize_key(nvidia_api_key or "")
    if not _NVIDIA_KEY_RE.match(nvidia_api_key):
        raise KeyUpdateError("invalid NVIDIA key format")
    if not kek or len(kek) != 32:
        raise KeyUpdateError(
            "session decryption key unavailable — log in again to refresh it")
    try:
        enc = _envelope_encrypt(nvidia_api_key, kek)
    except Exception as e:
        raise KeyUpdateError(f"encryption failed: {type(e).__name__}") from e
    with _db_lock, db_conn() as conn:
        cur = conn.execute(
            "UPDATE users SET nvidia_api_key_enc = ?, nvidia_api_key = NULL "
            "WHERE id = ?",
            (enc, user_id),
        )
    if cur.rowcount == 0:
        raise KeyUpdateError(f"no user row for id={user_id!r}")
    emit_event("user.key_updated", source="user", user_id=user_id)


# In-memory map: session token -> {"api_key": <plaintext>, "kek": <32 bytes>}.
# Populated on login (when we have the password and can derive the KEK) and
# read by per-request handlers. Cleared on logout. After a server restart the
# cache is empty, so users with valid session cookies will need to log in
# again to access NVIDIA-backed endpoints (other endpoints still work). The
# tradeoff is documented in docs/SECURITY.md.
SESSION_KEY_CACHE: dict = {}
_session_cache_lock = threading.Lock()


def _session_cache_set(token: str, api_key: str, kek: bytes) -> None:
    with _session_cache_lock:
        SESSION_KEY_CACHE[token] = {"api_key": api_key, "kek": kek}


def _session_cache_get(token: str) -> dict | None:
    with _session_cache_lock:
        return SESSION_KEY_CACHE.get(token)


def _session_cache_drop(token: str) -> None:
    with _session_cache_lock:
        SESSION_KEY_CACHE.pop(token, None)


def _user_count() -> int:
    with db_conn() as conn:
        return conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"]


def _borrow_any_user_key() -> str:
    """Return any user's NVIDIA key for background tasks (health probes) when
    no SERVER_KEY is configured. Three sources, in order:

    1. Any session in the in-memory cache — these are decrypted, free to use.
    2. Any user's legacy plaintext column (rare after migration).
    3. Encrypted-only users contribute nothing here: we don't have their KEK
       once they're logged out.

    Returns "" if nothing usable is available; the caller (probe_model) skips
    the probe rather than failing loudly."""
    with _session_cache_lock:
        for entry in SESSION_KEY_CACHE.values():
            if entry.get("api_key"):
                return entry["api_key"]
    with db_conn() as conn:
        row = conn.execute(
            "SELECT nvidia_api_key FROM users "
            "WHERE nvidia_api_key IS NOT NULL AND nvidia_api_key != '' "
            "ORDER BY is_admin DESC, created_at ASC LIMIT 1"
        ).fetchone()
    return (row["nvidia_api_key"] if row else "") or ""


def _user_get_by_username(username: str):
    with db_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
    return dict(row) if row else None


def _user_get_by_id(user_id: str):
    with db_conn() as conn:
        row = conn.execute(
            "SELECT id, username, is_admin, created_at, nvidia_api_key, "
            "       nvidia_api_key_enc "
            "FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
    return dict(row) if row else None


def _adopt_orphan_conversations(user_id: str) -> int:
    """Reassign conversations with NULL user_id to the given user.
    Run once when the first admin registers — picks up data from the
    pre-multi-user era. Returns the number of rows updated."""
    with _db_lock, db_conn() as conn:
        cur = conn.execute(
            "UPDATE conversations SET user_id = ? WHERE user_id IS NULL",
            (user_id,),
        )
    if cur.rowcount > 0:
        emit_event("conversation.orphans_adopted", source="user",
                   user_id=user_id, count=cur.rowcount)
    return cur.rowcount


def _session_create(user_id: str) -> tuple:
    """Create a new session. Returns (token, expires_at_iso)."""
    token = secrets.token_urlsafe(32)
    now = time.time()
    expires = now + SESSION_TTL_SECONDS
    expires_iso = datetime.datetime.fromtimestamp(expires, tz=datetime.timezone.utc) \
                                   .strftime("%Y-%m-%dT%H:%M:%SZ")
    created_iso = now_iso()
    with _db_lock, db_conn() as conn:
        conn.execute(
            "INSERT INTO sessions (token, user_id, created_at, expires_at) "
            "VALUES (?, ?, ?, ?)",
            (token, user_id, created_iso, expires_iso),
        )
    return token, expires_iso


def _session_get(token: str):
    """Look up a session token. Returns dict with user_id/expires_at, or None
    if missing / expired. Expired sessions are eagerly deleted."""
    if not token:
        return None
    with db_conn() as conn:
        row = conn.execute(
            "SELECT token, user_id, created_at, expires_at FROM sessions "
            "WHERE token = ?", (token,),
        ).fetchone()
    if not row:
        return None
    # Compare ISO strings — both are UTC, lex-comparable
    if row["expires_at"] <= now_iso():
        with _db_lock, db_conn() as conn:
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
        _session_cache_drop(token)
        return None
    return dict(row)


def _session_slide(token: str) -> str | None:
    """If the session is close to expiry refresh window, extend it.
    Returns the new expires_at_iso if extended, else None."""
    sess = _session_get(token)
    if not sess:
        return None
    # Parse current expiry, check if we're inside the slide window
    try:
        exp_dt = datetime.datetime.strptime(sess["expires_at"], "%Y-%m-%dT%H:%M:%SZ") \
                                  .replace(tzinfo=datetime.timezone.utc)
    except ValueError:
        return None
    remaining = (exp_dt.timestamp() - time.time())
    if remaining > SESSION_TTL_SECONDS - SESSION_SLIDE_THRESHOLD:
        # Recently created/refreshed — don't churn DB
        return None
    new_expires = time.time() + SESSION_TTL_SECONDS
    new_exp_iso = datetime.datetime.fromtimestamp(new_expires, tz=datetime.timezone.utc) \
                                   .strftime("%Y-%m-%dT%H:%M:%SZ")
    with _db_lock, db_conn() as conn:
        conn.execute("UPDATE sessions SET expires_at = ? WHERE token = ?",
                     (new_exp_iso, token))
    return new_exp_iso


def _session_delete(token: str) -> bool:
    if not token:
        return False
    with _db_lock, db_conn() as conn:
        cur = conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
    return cur.rowcount > 0


def _session_purge_expired() -> int:
    with _db_lock, db_conn() as conn:
        cur = conn.execute("DELETE FROM sessions WHERE expires_at <= ?", (now_iso(),))
    return cur.rowcount


def _build_session_cookie(token: str, max_age: int = SESSION_TTL_SECONDS) -> str:
    # SameSite=None + Secure is required for the cookie to flow when the
    # frontend (Capacitor APK at https://localhost) and backend (DDNS host)
    # are on different origins. Browsers reject SameSite=None without Secure,
    # which also means same-origin HTTP setups MUST be served behind HTTPS
    # if they want sessions to work — practically: use a tunnel / DDNS+TLS.
    # For pure-LAN HTTP, browsers still accept the cookie because they
    # apply Lax-by-default to first-party requests.
    return (f"{SESSION_COOKIE_NAME}={token}; "
            f"Path=/; SameSite=None; Secure; Max-Age={max_age}; HttpOnly")


def _expire_session_cookie() -> str:
    return (f"{SESSION_COOKIE_NAME}=; "
            "Path=/; SameSite=None; Secure; Max-Age=0; HttpOnly")


# Match every <script>...</script> WITHOUT a `src=` attribute, i.e. inline
# script blocks whose content needs a CSP hash to be allowed. The negative
# lookahead inside the opening tag rejects <script src="...">.
_INLINE_SCRIPT_RE = re.compile(rb'<script(?![^>]*\ssrc=)[^>]*>(.*?)</script>',
                               re.DOTALL | re.IGNORECASE)
_HTML_COMMENT_RE = re.compile(rb'<!--.*?-->', re.DOTALL)


def _inline_script_sha256s(html_bytes: bytes) -> list:
    """Return base64-encoded SHA-256 hashes of every inline script block in
    `html_bytes`, suitable for inclusion in a `script-src 'sha256-...'` CSP.

    HTML comments are stripped first so a literal `<script>` mention inside a
    comment doesn't generate a spurious hash. The hash is computed against
    the script body only (between `<script>` and `</script>`), matching what
    the browser computes when checking the CSP."""
    no_comments = _HTML_COMMENT_RE.sub(b"", html_bytes)
    out = []
    for m in _INLINE_SCRIPT_RE.finditer(no_comments):
        digest = hashlib.sha256(m.group(1)).digest()
        out.append(base64.b64encode(digest).decode("ascii"))
    return out


def _origin_allowed(origin: str) -> bool:
    """True if `origin` is on the CORS allowlist (or the wildcard '*' is set)."""
    if not origin:
        return False
    if "*" in ALLOWED_ORIGINS:
        return True
    return origin in ALLOWED_ORIGINS


def auth_check(handler):
    """Return one of:
       ("ok", user_dict)        - valid session, user attached
       ("public", None)         - public path, no auth needed
       ("fail", None)           - not authenticated; caller should send 401
    """
    path = handler.path.split("?", 1)[0]
    if path in PUBLIC_PATHS:
        return ("public", None)

    # Read session token from cookie
    token = ""
    cookie_header = handler.headers.get("Cookie", "")
    if cookie_header:
        try:
            jar = SimpleCookie()
            jar.load(cookie_header)
            if SESSION_COOKIE_NAME in jar:
                token = jar[SESSION_COOKIE_NAME].value
        except Exception:
            pass

    if not token:
        return ("fail", None)
    sess = _session_get(token)
    if not sess:
        return ("fail", None)
    user = _user_get_by_id(sess["user_id"])
    if not user:
        # Orphaned session (user deleted) — clean up
        _session_delete(token)
        return ("fail", None)
    user["_session_token"] = token
    return ("ok", user)


# ============================ HTTP HANDLER ============================

class Handler(BaseHTTPRequestHandler):
    # Don't advertise the implementation in the Server: header. BaseHTTPRequestHandler
    # also sends a sys_version (Python/X.Y); blank both so we don't fingerprint.
    server_version = ""
    sys_version = ""

    def log_message(self, fmt, *args):
        sys.stderr.write(f"[{self.log_date_time_string()}] {fmt % args}\n")

    def finish(self):
        """End-of-request hook. Close the thread-local sqlite connection so
        worker threads don't leak fds. The base class's finish() flushes and
        closes the wfile/rfile streams."""
        try:
            return super().finish()
        finally:
            _close_thread_local_conn()

    def _add_cors_headers(self):
        """Mirror the Origin header back ONLY for origins on the allowlist
        (`ALLOWED_ORIGINS`). The Vary: Origin header is always emitted so
        caches don't reuse responses meant for a different origin."""
        origin = self.headers.get("Origin", "")
        self.send_header("Vary", "Origin")
        if origin and _origin_allowed(origin):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Credentials", "true")

    def _send_bytes(self, code, data, ctype, extra_headers=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        # HTML responses get a CSP header (frame-ancestors needs the HTTP
        # form — meta http-equiv silently ignores it). Hashes for inline
        # <script> blocks are computed from the bytes we're about to send,
        # so JS edits Just Work without anyone manually updating a hash.
        if ctype.startswith("text/html"):
            hashes = _inline_script_sha256s(data)
            script_src = "'self' https://cdn.jsdelivr.net" + "".join(
                f" 'sha256-{h}'" for h in hashes)
            self.send_header("Content-Security-Policy",
                "default-src 'self'; "
                f"script-src {script_src}; "
                "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com; "
                # KaTeX's CSS pulls its math fonts from cdn.jsdelivr.net via
                # @font-face, so jsDelivr has to be on font-src too. Google
                # Fonts and inline data: URIs (some emoji fallbacks) round it out.
                "font-src 'self' https://fonts.gstatic.com https://cdn.jsdelivr.net data:; "
                "img-src 'self' data: blob:; "
                "media-src 'self' data: blob:; "
                "connect-src 'self'; "
                "frame-ancestors 'none'; "
                "base-uri 'self'; "
                "form-action 'self'")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
        self._add_cors_headers()
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_OPTIONS(self):
        """CORS preflight. Only honored for origins on `ALLOWED_ORIGINS`."""
        origin = self.headers.get("Origin", "")
        req_method = self.headers.get("Access-Control-Request-Method", "")
        req_headers = self.headers.get("Access-Control-Request-Headers", "")
        self.send_response(204)
        self.send_header("Vary", "Origin")
        if origin and _origin_allowed(origin):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Credentials", "true")
            if req_method:
                self.send_header("Access-Control-Allow-Methods",
                                 "GET, POST, PUT, DELETE, OPTIONS")
            if req_headers:
                self.send_header("Access-Control-Allow-Headers", req_headers)
            else:
                self.send_header("Access-Control-Allow-Headers",
                                 "Content-Type, X-Registration-Token")
            self.send_header("Access-Control-Max-Age", "86400")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _send_text(self, code, body, ctype="text/plain; charset=utf-8", extra_headers=None):
        self._send_bytes(code, body.encode("utf-8"), ctype, extra_headers)

    def _send_json(self, code, obj, extra_headers=None):
        self._send_bytes(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                         "application/json; charset=utf-8", extra_headers)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length < 0 or length > MAX_BODY:
            raise ValueError(f"body length {length} out of range")
        raw = self.rfile.read(length) if length > 0 else b""
        if not raw:
            return {}
        return json.loads(raw)

    def _require_user_key(self) -> str | None:
        """Return the logged-in user's plaintext NVIDIA key from the session
        cache (populated at login). Sends an appropriate error and returns
        None if the user isn't logged in / has no key set / the cache is
        empty (e.g. server restarted after the user logged in long ago)."""
        if not self.user:
            self._send_text(401, "auth required"); return None
        token = self.user.get("_session_token")
        cached = _session_cache_get(token) if token else None
        if cached and cached.get("api_key"):
            return cached["api_key"]
        # No cache entry. Two cases worth distinguishing for a useful message:
        if self.user.get("nvidia_api_key_enc") or self.user.get("nvidia_api_key"):
            # Key exists in DB but we can't decrypt — the cache evaporated
            # (server restart). The user needs to re-login to repopulate.
            self._send_json(401, {
                "error": {
                    "message": ("Session lost the decryption key (server "
                                "restarted?). Please log in again."),
                    "code": "session_key_unavailable",
                },
            })
        else:
            self._send_json(400, {
                "error": {
                    "message": ("Set your NVIDIA API key first via "
                                "POST /v1/auth/key (or in the UI's profile menu). "
                                "Get a free key at https://build.nvidia.com."),
                    "code": "no_api_key",
                },
            })
        return None

    def _auth_gate(self) -> bool:
        """Run auth check. On success, attach the authenticated user
        (or None for public paths) to self.user / self.user_id and return True.
        On failure, send 401 and return False."""
        status, user = auth_check(self)
        if status == "public":
            self.user = None
            self.user_id = None
            return True
        if status == "ok":
            self.user = user
            self.user_id = user["id"]
            return True
        # fail
        msg = ("auth required.\n"
               "POST /v1/auth/register or /v1/auth/login to obtain a session.\n")
        self._send_text(401, msg, extra_headers={"WWW-Authenticate": "Cookie"})
        return False

    # -------- GET --------
    def do_GET(self):
        if not self._auth_gate():
            return
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html", "/chat"):
            if not HTML_PATH.exists():
                self._send_text(500, f"HTML not found at {HTML_PATH}"); return
            self._send_bytes(200, HTML_PATH.read_bytes(), "text/html; charset=utf-8")
            return
        if path == "/health":
            self._send_text(200, "ok"); return
        if path == "/models":
            self._send_json(200, {
                "items": health_snapshot(),
                "aliases": MODEL_ALIASES,
                "categories": CATEGORY_PREFERENCE,
                "pdf_supported": _PYPDF_OK,
            })
            return
        if path == "/v1/tools":
            self._send_json(200, {
                "items": list_tools_meta(),
                "image_models": list(IMG_MODEL_MAP.keys()),
                "embed_models": list(EMBED_MODEL_ALIASES.keys()),
            })
            return
        if path == "/v1/tts/voices":
            self._do_tts_voices(); return
        if path == "/v1/tts/health":
            self._do_tts_health(); return
        if path == "/v1/auth/me":
            self._do_auth_me(); return
        if path == "/conversations":
            self._send_json(200, {"items": conv_list(self.user_id)}); return
        m = re.fullmatch(r"/conversations/([A-Za-z0-9_-]+)", path)
        if m:
            data = conv_get(m.group(1), self.user_id)
            if not data:
                self._send_text(404, "not found"); return
            self._send_json(200, data); return
        m = re.fullmatch(r"/v1/artifact/([A-Za-z0-9_\-]+)", path)
        if m:
            self._do_artifact_get(m.group(1)); return
        self._send_text(404, "not found")

    def _do_artifact_get(self, aid: str):
        """Stream an artifact's bytes to the requesting user, only if they own
        it. The auth gate ran before us; self.user_id is set."""
        path, mime = _artifact_lookup(aid, self.user_id)
        if not path:
            self._send_text(404, "not found"); return
        try:
            data = path.read_bytes()
        except OSError:
            self._send_text(404, "not found"); return
        # Long cache: artifacts are immutable (id is content-addressed by
        # token). Browser can cache aggressively; cookie still gates access.
        self._send_bytes(200, data, mime,
                         extra_headers={"Cache-Control": "private, max-age=86400"})

    # -------- POST --------
    def do_POST(self):
        if not self._auth_gate():
            return
        path = self.path.split("?", 1)[0]
        if path == "/v1/chat/completions":
            self._proxy_completions(); return
        if path == "/v1/translate":
            self._proxy_translate(); return
        if path == "/v1/route":
            self._do_route(); return
        if path == "/v1/compact":
            self._do_compact(); return
        if path == "/v1/extract":
            self._do_extract(); return
        if path == "/v1/tts":
            self._do_tts_proxy(); return
        if path == "/v1/auth/register":
            self._do_auth_register(); return
        if path == "/v1/auth/login":
            self._do_auth_login(); return
        if path == "/v1/auth/logout":
            self._do_auth_logout(); return
        if path == "/v1/auth/key":
            self._do_auth_set_key(); return
        if path == "/conversations":
            try:
                body = self._read_json()
            except (ValueError, json.JSONDecodeError) as e:
                self._send_text(400, f"bad request: {e}"); return
            try:
                meta = conv_create(body if isinstance(body, dict) else {},
                                   self.user_id)
            except sqlite3.Error as e:
                self._send_text(500, f"db error: {e}"); return
            self._send_json(200, meta); return
        m = re.fullmatch(r"/conversations/([A-Za-z0-9_-]+)/messages", path)
        if m:
            cid = m.group(1)
            try:
                body = self._read_json()
            except (ValueError, json.JSONDecodeError) as e:
                self._send_text(400, f"bad request: {e}"); return
            if not isinstance(body, dict):
                self._send_text(400, "expected object"); return
            try:
                msg_id = message_append(
                    cid, body.get("role"),
                    body.get("content") or "",
                    self.user_id,
                    reasoning_content=body.get("reasoning_content") or None,
                )
            except LookupError:
                self._send_text(404, "conversation not found"); return
            except ValueError as e:
                self._send_text(400, str(e)); return
            except sqlite3.Error as e:
                self._send_text(500, f"db error: {e}"); return
            self._send_json(200, {"ok": True, "message_id": msg_id,
                                  "conversation": conv_meta(cid, self.user_id)})
            return
        self._send_text(404, "not found")

    def do_PUT(self):
        if not self._auth_gate():
            return
        path = self.path.split("?", 1)[0]
        m = re.fullmatch(r"/conversations/([A-Za-z0-9_-]+)", path)
        if m:
            cid = m.group(1)
            try:
                body = self._read_json()
            except (ValueError, json.JSONDecodeError) as e:
                self._send_text(400, f"bad request: {e}"); return
            if not isinstance(body, dict):
                self._send_text(400, "expected object"); return
            ok = conv_update(cid, body, self.user_id)
            if not ok:
                self._send_text(404, "not found"); return
            self._send_json(200, {"ok": True,
                                  "conversation": conv_meta(cid, self.user_id)})
            return
        self._send_text(404, "not found")

    def do_DELETE(self):
        if not self._auth_gate():
            return
        path = self.path.split("?", 1)[0]
        m = re.fullmatch(r"/conversations/([A-Za-z0-9_-]+)", path)
        if m:
            ok = conv_delete(m.group(1), self.user_id)
            if not ok:
                self._send_text(404, "not found"); return
            self._send_json(200, {"ok": True}); return
        self._send_text(404, "not found")

    # -------- /v1/auth/* --------
    def _do_auth_register(self):
        try:
            body = self._read_json()
        except (ValueError, json.JSONDecodeError) as e:
            self._send_text(400, f"bad json: {e}"); return
        if not isinstance(body, dict):
            self._send_text(400, "expected object"); return

        # Optional registration token gate (set via env/CLI)
        if REGISTRATION_TOKEN:
            supplied = self.headers.get("X-Registration-Token", "") \
                       or body.get("registration_token", "")
            if supplied != REGISTRATION_TOKEN:
                self._send_text(403, "registration token required or invalid")
                return

        username = (body.get("username") or "").strip()
        password = body.get("password") or ""
        nvidia_api_key = normalize_key(body.get("nvidia_api_key") or "")
        if not nvidia_api_key:
            self._send_text(400, "nvidia_api_key required (get one free at "
                                 "https://build.nvidia.com)"); return
        # Validate the key against NVIDIA before creating the account.
        # Saves the user from registering only to get errors on first chat.
        ok, detail = validate_nvidia_key(nvidia_api_key)
        if not ok:
            self._send_text(400, f"NVIDIA key validation failed: {detail}")
            return

        # First user becomes admin and inherits orphan conversations.
        # Serialize the count→create→adopt sequence under _db_lock (reentrant)
        # so two concurrent first-time registrations can't both observe
        # count==0 and both become admin.
        with _db_lock:
            is_first = (_user_count() == 0)
            try:
                user = _user_create(username, password, nvidia_api_key,
                                    is_admin=is_first)
            except ValueError as e:
                self._send_text(400, str(e)); return
            except sqlite3.IntegrityError:
                self._send_text(409, "username already taken"); return
            except sqlite3.Error as e:
                self._send_text(500, f"db error: {e}"); return

            adopted = 0
            if is_first:
                try:
                    adopted = _adopt_orphan_conversations(user["id"])
                except sqlite3.Error:
                    pass  # non-fatal

        # Auto-login the newly registered user. _user_create handed back the
        # KEK from its scrypt derivation; reuse it to seed the session cache
        # so the user's first turn doesn't 400 on a missing cache entry.
        token, expires = _session_create(user["id"])
        _session_cache_set(token, user["_api_key"], user["_kek"])
        self._send_json(200, {
            "ok": True,
            "user": {"id": user["id"], "username": user["username"],
                     "is_admin": bool(user["is_admin"])},
            "session": {"expires_at": expires},
            "adopted_orphan_conversations": adopted,
            "is_first_user": is_first,
        }, extra_headers={"Set-Cookie": _build_session_cookie(token)})

    def _do_auth_set_key(self):
        """POST /v1/auth/key {nvidia_api_key} — update the logged-in user's key.
        Validates against NVIDIA before persisting, same as registration.
        The KEK from the current session is reused to re-encrypt; if the
        cache lost it (server restart), the user must log in again first."""
        try:
            body = self._read_json()
        except (ValueError, json.JSONDecodeError) as e:
            self._send_text(400, f"bad json: {e}"); return
        if not isinstance(body, dict):
            self._send_text(400, "expected object"); return
        new_key = normalize_key(body.get("nvidia_api_key") or "")
        if not new_key:
            self._send_text(400, "nvidia_api_key required"); return
        token = self.user.get("_session_token")
        cached = _session_cache_get(token) if token else None
        if not cached or not cached.get("kek"):
            self._send_json(401, {"error": {
                "message": "session decryption key unavailable — please log in again",
                "code": "session_key_unavailable",
            }}); return
        ok, detail = validate_nvidia_key(new_key)
        if not ok:
            self._send_text(400, f"key validation failed: {detail}"); return
        try:
            _user_set_key(self.user_id, new_key, cached["kek"])
        except KeyUpdateError as e:
            # 500 by default; specific not-found / no-row cases are rare
            # (user was deleted between login and this call). Caller gets
            # the exact reason instead of "db error?".
            self._send_text(500, f"key update failed: {e}"); return
        # Refresh cache with the new plaintext
        _session_cache_set(token, new_key, cached["kek"])
        self._send_json(200, {"ok": True, "detail": "key updated"})

    def _do_auth_login(self):
        try:
            body = self._read_json()
        except (ValueError, json.JSONDecodeError) as e:
            self._send_text(400, f"bad json: {e}"); return
        if not isinstance(body, dict):
            self._send_text(400, "expected object"); return

        username = (body.get("username") or "").strip()
        password = body.get("password") or ""
        if not username or not password:
            self._send_text(400, "username and password required"); return

        user = _user_get_by_username(username)
        # Always run hashing even on missing user to even out timing —
        # not perfect but raises the bar against username enumeration.
        if not user:
            try:
                _scrypt_derive(password)  # discard
            except Exception:
                pass
            self._send_text(401, "invalid credentials"); return

        ok, scrypt_out = _verify_password(
            password, user["password_hash"], user["password_salt"])
        if not ok:
            self._send_text(401, "invalid credentials"); return
        kek = scrypt_out[32:]

        # Decrypt the user's NVIDIA key. Three paths:
        #   (a) nvidia_api_key_enc present  → decrypt with KEK
        #   (b) legacy plaintext only       → migrate (encrypt + null plaintext +
        #                                      shorten password_hash to 32 bytes)
        #   (c) neither                     → user has no key set yet (rare —
        #                                      pre-BYOK accounts before this
        #                                      release shipped); login still
        #                                      succeeds, but NVIDIA endpoints
        #                                      will 400 until /v1/auth/key sets one
        api_key = ""
        try:
            if user.get("nvidia_api_key_enc"):
                api_key = _envelope_decrypt(user["nvidia_api_key_enc"], kek)
            elif user.get("nvidia_api_key"):
                # Legacy: plaintext was stored. Migrate now while we have the KEK.
                api_key = user["nvidia_api_key"]
                enc = _envelope_encrypt(api_key, kek)
                new_hash = scrypt_out[:32].hex()
                with _db_lock, db_conn() as conn:
                    conn.execute(
                        "UPDATE users SET nvidia_api_key_enc = ?, "
                        "       password_hash = ?, nvidia_api_key = NULL "
                        "WHERE id = ?",
                        (enc, new_hash, user["id"]),
                    )
                emit_event("user.envelope_migrated", source="user",
                           user_id=user["id"])
            elif len(user["password_hash"]) == 128:
                # Account predates split hash AND has no key. Just shorten the
                # stored hash so the next pass doesn't keep treating it as legacy.
                new_hash = scrypt_out[:32].hex()
                with _db_lock, db_conn() as conn:
                    conn.execute(
                        "UPDATE users SET password_hash = ? WHERE id = ?",
                        (new_hash, user["id"]),
                    )
        except Exception as e:
            emit_event("auth.envelope_failed", source="user",
                       user_id=user["id"], error=f"{type(e).__name__}: {str(e)[:80]}")
            self._send_text(500, "could not decrypt stored key — see logs")
            return

        token, expires = _session_create(user["id"])
        if api_key:
            _session_cache_set(token, api_key, kek)
        emit_event("auth.login", source="user", user_id=user["id"],
                   username=user["username"])
        self._send_json(200, {
            "ok": True,
            "user": {"id": user["id"], "username": user["username"],
                     "is_admin": bool(user["is_admin"])},
            "session": {"expires_at": expires},
        }, extra_headers={"Set-Cookie": _build_session_cookie(token)})

    def _do_auth_logout(self):
        # Read session cookie and delete the corresponding row
        token = ""
        cookie_header = self.headers.get("Cookie", "")
        if cookie_header:
            try:
                jar = SimpleCookie()
                jar.load(cookie_header)
                if SESSION_COOKIE_NAME in jar:
                    token = jar[SESSION_COOKIE_NAME].value
            except Exception:
                pass
        if token:
            _session_delete(token)
            _session_cache_drop(token)
            emit_event("auth.logout", source="user", user_id=self.user_id)
        self._send_json(200, {"ok": True},
                        extra_headers={"Set-Cookie": _expire_session_cookie()})

    def _do_auth_me(self):
        # _auth_gate already attached self.user
        u = self.user
        # Surface the masked tail of the user's NVIDIA key. Three states:
        #   present + tail : decrypted in cache, can show last 4 chars
        #   present (no tail): stored encrypted but cache empty (must re-login)
        #   not present    : no key set
        token = u.get("_session_token")
        cached = _session_cache_get(token) if token else None
        if cached and cached.get("api_key"):
            ak = cached["api_key"]
            key_status = {"present": True,
                          "tail": (ak[-4:] if len(ak) >= 8 else "****")}
        elif u.get("nvidia_api_key_enc") or u.get("nvidia_api_key"):
            key_status = {"present": True, "tail": "", "needs_login": True}
        else:
            key_status = {"present": False, "tail": ""}
        self._send_json(200, {
            "user": {"id": u["id"], "username": u["username"],
                     "is_admin": bool(u["is_admin"]),
                     "created_at": u.get("created_at"),
                     "key": key_status},
        })


    def _do_extract(self):
        try:
            data = self._read_json()
        except (ValueError, json.JSONDecodeError) as e:
            self._send_text(400, f"bad json: {e}"); return
        if not isinstance(data, dict):
            self._send_text(400, "expected object"); return
        fname = (data.get("filename") or "file").strip()
        b64 = data.get("data_base64") or ""
        mime = (data.get("mime") or "").lower()
        try:
            raw = base64.b64decode(b64, validate=False)
        except Exception as e:
            self._send_text(400, f"bad base64: {e}"); return
        ext = fname.lower().rsplit(".", 1)[-1] if "." in fname else ""

        emit_event("extract.requested", source="user",
                   filename=fname, size=len(raw), ext=ext, mime=mime)

        # PDF
        if ext == "pdf" or "pdf" in mime:
            if not _PYPDF_OK:
                self._send_text(501, "pypdf not installed on server"); return
            try:
                out = extract_pdf(raw)
            except Exception as e:
                self._send_text(500, f"pdf parse error: {e}"); return
            self._send_json(200, {
                "type": "pdf", "filename": fname, "size": len(raw),
                "text": out["text"], "n_pages": out["n_pages"],
            })
            return

        # text-like
        if mime.startswith("text/") or ext in TEXT_EXTENSIONS:
            try:
                text, enc = _decode_text_bytes(raw)
            except UnicodeDecodeError:
                self._send_text(415, "could not decode as text"); return
            self._send_json(200, {
                "type": "text", "filename": fname, "size": len(raw),
                "text": text, "encoding": enc,
            })
            return

        self._send_text(415, f"unsupported file type: {fname!r}")

    # -------- /v1/route --------
    def _do_route(self):
        req_id = make_req_id()
        try:
            data = self._read_json()
        except (ValueError, json.JSONDecodeError) as e:
            self._send_text(400, f"bad json: {e}"); return
        if not isinstance(data, dict):
            self._send_text(400, "expected object"); return

        text = data.get("text") or ""
        has_image = bool(data.get("has_image"))
        est_tokens = int(data.get("est_tokens") or 0)
        max_output = int(data.get("max_output") or 512)

        emit_event("route.started", req_id=req_id, source="user",
                   has_image=has_image, est_tokens=est_tokens)

        if has_image:
            cls = {"category": "vision", "reason": "image attached", "confidence": 1.0}
        elif not text.strip():
            cls = {"category": "general", "reason": "empty input", "confidence": 0.5}
        else:
            user_key = self._require_user_key()
            if user_key is None:
                return  # 400 already sent
            cls = self._classify_via_llm(text, req_id, user_key)

        chosen, fallback, notes = select_model(
            cls["category"], has_image, est_tokens, max_output
        )
        result = {
            "req_id": req_id,
            "category": cls["category"],
            "model": chosen,
            "fallback": fallback,
            "reason": cls.get("reason", ""),
            "confidence": cls.get("confidence", 0.0),
            "notes": notes,
            "health": {m: model_health(m) for m in [chosen] + fallback if m},
        }
        emit_event("route.decided", req_id=req_id, source="user",
                   category=result["category"], model=chosen,
                   fallback_count=len(fallback), notes=notes)
        self._send_json(200, result, extra_headers={"X-Req-Id": req_id})

    def _classify_via_llm(self, text: str, req_id: str, api_key: str) -> dict:
        payload = {
            "model": ROUTER_MODEL,
            "messages": [
                {"role": "system", "content": ROUTER_SYSTEM},
                {"role": "user", "content": text[:1500]},
            ],
            "max_tokens": 100, "temperature": 0.1, "top_p": 0.9, "stream": False,
        }
        try:
            r = requests.post(NVIDIA_URL, headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json", "Accept": "application/json",
            }, json=payload, timeout=20)
        except requests.RequestException as e:
            emit_event("route.classify.failed", req_id=req_id, source="user", error=str(e)[:80])
            return {"category": "general", "reason": "router unreachable", "confidence": 0.0}
        if r.status_code != 200:
            return {"category": "general", "reason": f"router HTTP {r.status_code}", "confidence": 0.0}
        try:
            content = r.json()["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, ValueError):
            return {"category": "general", "reason": "router parse failed", "confidence": 0.0}
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content,
                         flags=re.IGNORECASE | re.MULTILINE).strip()
        m = re.search(r"\{[^{}]*\}", content)
        if m:
            content = m.group(0)
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return {"category": "general", "reason": "router json invalid", "confidence": 0.0}
        cat = parsed.get("category", "general")
        if cat not in CATEGORY_PREFERENCE or cat == "vision":
            cat = "general"
        return {
            "category": cat,
            "reason": (parsed.get("reason") or "").strip()[:60],
            "confidence": float(parsed.get("confidence") or 0.0),
        }

    # -------- /v1/compact --------
    # Compresses an array of messages into a single summary message. Triggered
    # by the /compact slash command in the UI, and automatically when the
    # client detects the 80% context-limit threshold. The whole transcript
    # is passed in; the returned summary becomes the new first message in
    # subsequent /v1/chat/completions calls, and new turns build on top of it.
    #
    # The client picks the budget via target_max_tokens (see below). Two modes
    # exist on the UI side: `rel N%` (target = N% of current transcript tokens)
    # and `fix N%` (target = N% of the model's context window). The server
    # treats target_max_tokens as an opaque integer — the modes are pure UI.
    #
    # Default model: Maverick (1M ctx) — Llama 3.3 70B's 128K is often smaller
    # than the transcript we're compressing. Maverick is non-reasoning so it
    # doesn't waste tokens on what is essentially summarization.
    def _do_compact(self):
        req_id = make_req_id()
        api_key = self._require_user_key()
        if api_key is None:
            return
        try:
            data = self._read_json()
        except (ValueError, json.JSONDecodeError) as e:
            self._send_text(400, f"bad json: {e}"); return
        if not isinstance(data, dict):
            self._send_text(400, "expected object"); return

        messages = data.get("messages") or []
        if not isinstance(messages, list) or not messages:
            self._send_text(400, "messages array required and non-empty"); return

        # Compaction default: Maverick (1M context). 70B has 128K which is
        # often smaller than the transcript we're trying to compact (e.g. a
        # 200K-token PDF history hits 70B's own ctx limit, paradoxically).
        # Maverick is non-reasoning so it doesn't waste reasoning tokens on
        # what is essentially summarization.
        model = resolve_alias(data.get("model") or "meta/llama-4-maverick-17b-128e-instruct")
        if model not in MODEL_REGISTRY:
            self._send_text(400, f"unknown model: {model}"); return

        # Build a compact-friendly transcript. Roles are kept so the summarizer
        # understands turn structure. Long messages are NOT truncated here —
        # the whole point is to compress them; truncating first would lose info.
        # Messages must already be plain strings (the frontend stringifies
        # multipart content before sending).
        transcript_parts = []
        for m in messages:
            role = m.get("role") or "user"
            content = m.get("content") or ""
            if isinstance(content, list):
                # Multipart (image attachments etc.) — extract text parts and
                # note non-text parts. We can't summarize raw image bytes.
                text_bits = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            text_bits.append(part.get("text") or "")
                        elif part.get("type") == "image_url":
                            text_bits.append("[image attached]")
                content = " ".join(text_bits)
            content = str(content)
            transcript_parts.append(f"[{role}]\n{content}")
        transcript = "\n\n".join(transcript_parts)

        emit_event("compact.started", req_id=req_id, source="user",
                   model=model, message_count=len(messages),
                   transcript_chars=len(transcript))

        # Caller specifies target_max_tokens explicitly. This drives both:
        # (a) the system prompt's "≤N tokens" instruction, so the LLM knows
        #     the budget, and (b) the upstream max_tokens cap, so even if the
        #     LLM ignores the budget the response can't exceed it.
        # If not provided, fall back to a generous default (4096) — that's
        # the legacy behavior for backward compatibility, but any sensible
        # client should pass an explicit budget.
        target_max_tokens = data.get("target_max_tokens")
        if not isinstance(target_max_tokens, int) or target_max_tokens < 64:
            target_max_tokens = 4096
        # Cap at 8192 — Maverick's model-card max_output. Higher values would
        # be silently truncated by the upstream anyway.
        target_max_tokens = min(target_max_tokens, 8192)

        system = (
            "You are a conversation summarizer. The user provides a transcript "
            "of a chat between a human and an AI assistant. Your job is to "
            "produce a faithful, dense summary that the assistant can use as "
            "context to continue the conversation coherently.\n\n"
            "REQUIREMENTS:\n"
            "1. Preserve every concrete fact, decision, code change, name, "
            "number, and identifier that has been established. These are the "
            "load-bearing details — losing them breaks continuity.\n"
            "2. Preserve the user's stated preferences, goals, and constraints.\n"
            "3. Note any unresolved threads or pending questions.\n"
            "4. Drop pleasantries, redundant repetition, and meta-commentary "
            "about the conversation itself.\n"
            "5. Write in the same language as the original transcript. "
            "If the transcript is mixed-language, keep that mix.\n"
            "6. Use a structured form: short paragraphs or bullet points, "
            "grouped by topic.\n"
            "7. Output ONLY the summary text. No preamble, no 'Here is the "
            "summary', no closing remarks.\n"
            f"8. CRITICAL: The summary MUST be ≤ {target_max_tokens} tokens. "
            "Prioritize the most load-bearing details if you must cut. "
            "A useful 80%-of-budget summary is better than a complete summary "
            "that gets truncated mid-sentence."
        )

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content":
                    f"Transcript to summarize:\n\n{transcript}"},
            ],
            "max_tokens": target_max_tokens,
            "temperature": 0.3,
            "top_p": 0.9,
            "stream": False,
        }
        # Apply model quirks (e.g. dsv4 chat template) so we don't hit
        # template-required errors.
        payload = apply_quirks(payload, model)

        t0 = time.time()
        try:
            r = requests.post(NVIDIA_URL, headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json", "Accept": "application/json",
            }, json=payload, timeout=120)
        except requests.RequestException as e:
            emit_event("compact.failed", req_id=req_id, source="user",
                       error=str(e)[:120])
            self._send_json(502, {"error": f"upstream unreachable: {e}",
                                  "req_id": req_id}); return

        latency_ms = int((time.time() - t0) * 1000)

        if r.status_code != 200:
            # Extract NVIDIA's error text so we can see WHY the call was
            # rejected (token limit? wrong model? rate limit?). Without this
            # the user just sees "400" with no actionable info.
            detail = r.text[:500]
            emit_event("compact.failed", req_id=req_id, source="user",
                       status=r.status_code, latency_ms=latency_ms,
                       detail=detail)
            self._send_json(r.status_code,
                            {"error": f"upstream HTTP {r.status_code}",
                             "detail": detail, "req_id": req_id}); return

        try:
            summary = r.json()["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, ValueError) as e:
            emit_event("compact.failed", req_id=req_id, source="user",
                       error=f"parse: {e}", latency_ms=latency_ms)
            self._send_json(502, {"error": "upstream parse failed",
                                  "req_id": req_id}); return

        emit_event("compact.completed", req_id=req_id, source="user",
                   model=model, latency_ms=latency_ms,
                   summary_chars=len(summary))

        self._send_json(200, {
            "req_id": req_id,
            "summary": summary,
            "model": model,
            "latency_ms": latency_ms,
            "input_message_count": len(messages),
            "input_chars": len(transcript),
            "output_chars": len(summary),
        }, extra_headers={"X-Req-Id": req_id})

    # -------- /v1/translate --------
    def _proxy_translate(self):
        api_key = self._require_user_key()
        if api_key is None:
            return
        try:
            data = self._read_json()
        except (ValueError, json.JSONDecodeError) as e:
            self._send_text(400, f"bad json: {e}"); return
        if not isinstance(data, dict):
            self._send_text(400, "expected object"); return
        text = (data.get("text") or "").strip()
        if not text:
            self._send_text(400, "empty text"); return
        target = data.get("target_lang") or "English"
        model = resolve_alias(data.get("model") or "meta/llama-3.3-70b-instruct")
        want_stream = bool(data.get("stream", True))

        system = (
            f"You are a precise translator. Translate the user's text into {target}. "
            "Output ONLY the translation. Do not add explanations, quotation marks, "
            "headers, or commentary. Preserve technical terms, code, and equations as-is."
        )
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ],
            "max_tokens": min(int(data.get("max_tokens") or 4096), 8192),
            "temperature": 0.2, "top_p": 0.9, "stream": want_stream,
        }
        # Translation goes through a single attempt — no recovery recipe.
        self._forward_simple(payload, want_stream, api_key)

    def _forward_simple(self, payload, want_stream, api_key):
        try:
            r = requests.post(NVIDIA_URL, headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream" if want_stream else "application/json",
            }, json=payload, stream=want_stream, timeout=300)
        except requests.RequestException as e:
            self._send_text(502, f"upstream error: {e}"); return

        self.send_response(r.status_code)
        ctype = r.headers.get("Content-Type", "application/json")
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        if want_stream:
            self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            if want_stream:
                for chunk in r.iter_content(chunk_size=None):
                    if not chunk:
                        continue
                    try:
                        self.wfile.write(chunk)
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError,
                            ConnectionAbortedError, OSError):
                        # Client disconnected mid-stream (Stop button, browser
                        # close, network drop). Close the upstream so we don't
                        # keep paying for tokens we'll never deliver.
                        emit_event("turn.client_disconnect", req_id=req_id,
                                   source="user", model=payload.get("model"))
                        break
            else:
                try:
                    self.wfile.write(r.content)
                except (BrokenPipeError, ConnectionResetError,
                        ConnectionAbortedError, OSError):
                    pass
        finally:
            r.close()

    # -------- /v1/tts (proxy to local VibeVoice worker) --------
    def _do_tts_proxy(self):
        try:
            body = self._read_json()
        except (ValueError, json.JSONDecodeError) as e:
            self._send_text(400, f"bad json: {e}"); return
        if not isinstance(body, dict):
            self._send_text(400, "expected object"); return
        text = (body.get("text") or "").strip()
        voice = body.get("voice") or "Emma"
        if not text:
            self._send_text(400, "empty text"); return
        if len(text) > 4000:
            text = text[:4000]
        try:
            r = requests.post(f"{VIBEVOICE_WORKER}/tts",
                              json={"text": text, "voice": voice},
                              timeout=180)
        except requests.RequestException as e:
            self._send_json(502, {
                "error": "vibevoice_unreachable",
                "worker": VIBEVOICE_WORKER,
                "detail": f"{type(e).__name__}: {str(e)[:200]}",
                "hint": "Start the worker with: python vibevoice_worker.py",
            }); return
        if not r.ok:
            self._send_json(r.status_code, {
                "error": "vibevoice_failed",
                "status": r.status_code,
                "detail": r.text[:500],
            }); return
        # Forward audio bytes verbatim
        self._send_bytes(200, r.content,
                         r.headers.get("Content-Type", "audio/wav"))

    def _do_tts_voices(self):
        try:
            r = requests.get(f"{VIBEVOICE_WORKER}/voices", timeout=10)
        except requests.RequestException as e:
            self._send_json(200, {
                "available": False,
                "voices": [],
                "worker": VIBEVOICE_WORKER,
                "detail": f"{type(e).__name__}: {str(e)[:200]}",
            }); return
        if not r.ok:
            self._send_json(200, {
                "available": False,
                "voices": [],
                "worker": VIBEVOICE_WORKER,
                "detail": f"HTTP {r.status_code}: {r.text[:200]}",
            }); return
        try:
            data = r.json()
        except ValueError:
            self._send_json(200, {"available": False, "voices": [],
                                  "detail": "non-JSON response"}); return
        data["available"] = True
        data["worker"] = VIBEVOICE_WORKER
        self._send_json(200, data)

    def _do_tts_health(self):
        try:
            r = requests.get(f"{VIBEVOICE_WORKER}/health", timeout=5)
            ok = r.ok
            detail = r.text[:200] if r.ok else f"HTTP {r.status_code}"
        except requests.RequestException as e:
            ok = False
            detail = f"{type(e).__name__}: {str(e)[:200]}"
        self._send_json(200, {
            "ok": ok, "worker": VIBEVOICE_WORKER, "detail": detail,
        })

    # -------- /v1/chat/completions with full recovery -----------
    def _proxy_completions(self):
        req_id = make_req_id()
        api_key = self._require_user_key()
        if api_key is None:
            return
        try:
            payload = self._read_json()
        except (ValueError, json.JSONDecodeError) as e:
            self._send_text(400, f"bad json: {e}"); return
        if not isinstance(payload, dict):
            self._send_text(400, "expected object"); return

        # alias resolution
        if payload.get("model"):
            payload["model"] = resolve_alias(payload["model"])

        # fallback chain (claw US-004 escalation chain)
        fallback_chain = payload.pop("_fallback", None) or []
        if not isinstance(fallback_chain, list):
            fallback_chain = []
        fallback_chain = [resolve_alias(m) for m in fallback_chain if isinstance(m, str)]

        ledger = RecoveryLedger()
        notes = []

        # Apply quirks for the primary model
        apply_quirks(payload, payload.get("model", ""))

        # Pre-flight body size + auto-truncation (claw US-021)
        size = payload_byte_size(payload)
        if size > MAX_UPSTREAM_BYTES:
            ledger.add(recipe="preflight_size_truncate", attempt=0, state="started",
                       summary=f"size={size}B > {MAX_UPSTREAM_BYTES}B")
            ok = truncate_history_in_place(payload, MAX_UPSTREAM_BYTES, notes)
            if ok:
                ledger.add(recipe="preflight_size_truncate", attempt=0, state="succeeded",
                           summary=f"truncated to {payload_byte_size(payload)}B; {notes[-1]}")
            else:
                ledger.add(recipe="preflight_size_truncate", attempt=0, state="failed",
                           summary="cannot truncate further")
                self._send_json(413, {
                    "error": {"message": explain_http_error(413, "client-side preflight", req_id),
                              "status": 413, "req_id": req_id, "ledger": ledger.entries}
                }, extra_headers={"X-Req-Id": req_id,
                                  "X-Recovery-Steps": str(ledger.step_count())})
                return

        want_stream = bool(payload.get("stream"))
        wants_tools = bool(payload.pop("_use_tools", False))
        emit_event("turn.started", req_id=req_id, source="user",
                   model=payload.get("model"),
                   stream=want_stream, tools=wants_tools,
                   fallback_count=len(fallback_chain))

        if wants_tools:
            self._run_with_tools(payload, want_stream, fallback_chain, ledger,
                                 req_id, api_key)
        else:
            self._run_with_recipes(payload, want_stream, fallback_chain, ledger,
                                   req_id, api_key)

    def _run_with_recipes(self, payload, want_stream, fallback_chain, ledger,
                          req_id, api_key, attempt=1, max_attempts=4):
        """Tries upstream; on retriable error, applies the matching recipe.
        Each recipe execution is recorded in the ledger."""
        if attempt > max_attempts:
            ledger.add(recipe="escalation", attempt=attempt, state="escalated",
                       summary=f"reached max attempts {max_attempts}")
            emit_event("turn.failed", req_id=req_id, source="fallback",
                       reason="max attempts reached")
            self._send_json(502, {
                "error": {"message": f"[{req_id}] max retry attempts reached", "status": 502,
                          "req_id": req_id, "ledger": ledger.entries}
            }, extra_headers={"X-Req-Id": req_id,
                              "X-Recovery-Steps": str(ledger.step_count())})
            return

        model_now = payload.get("model")
        emit_event("turn.upstream.started", req_id=req_id,
                   source=("user" if attempt == 1 else "fallback"),
                   model=model_now, attempt=attempt)

        try:
            r = requests.post(NVIDIA_URL, headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream" if want_stream else "application/json",
            }, json=payload, stream=want_stream, timeout=300)
        except requests.RequestException as e:
            return self._handle_failure(
                payload, want_stream, fallback_chain, ledger, req_id, api_key,
                attempt, max_attempts, status_or_exc=e, body_text="")

        # Decide based on status
        if r.ok:
            # Stream/forward — once headers are committed there's no rolling back.
            self.send_response(r.status_code)
            ctype = r.headers.get("Content-Type", "application/json")
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            if want_stream:
                self.send_header("X-Accel-Buffering", "no")
            self.send_header("X-Req-Id", req_id)
            self.send_header("X-NIM-Model", model_now or "")
            self.send_header("X-NIM-Attempt", str(attempt))
            self.send_header("X-Recovery-Steps", str(ledger.step_count()))
            self.end_headers()
            try:
                if want_stream:
                    for chunk in r.iter_content(chunk_size=None):
                        if not chunk:
                            continue
                        try:
                            self.wfile.write(chunk)
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError,
                                ConnectionAbortedError, OSError):
                            # Client disconnected mid-stream — close upstream
                            emit_event("turn.client_disconnect", req_id=req_id,
                                       source=("user" if attempt == 1 else "fallback"),
                                       model=model_now, attempt=attempt)
                            break
                else:
                    try:
                        self.wfile.write(r.content)
                    except (BrokenPipeError, ConnectionResetError,
                            ConnectionAbortedError, OSError):
                        pass
            finally:
                r.close()
            emit_event("turn.completed", req_id=req_id,
                       source=("user" if attempt == 1 else "fallback"),
                       model=model_now, attempt=attempt)
            return

        # Failure — capture body before deciding recipe
        try:
            body_text = r.text[:500]
        except Exception:
            body_text = ""
        status = r.status_code
        r.close()
        return self._handle_failure(
            payload, want_stream, fallback_chain, ledger, req_id, api_key,
            attempt, max_attempts, status_or_exc=status, body_text=body_text)

    def _handle_failure(self, payload, want_stream, fallback_chain, ledger,
                        req_id, api_key, attempt, max_attempts,
                        status_or_exc, body_text):
        plan = plan_recovery(status_or_exc, payload)
        if plan is None:
            # Non-retriable — escalate immediately
            ledger.add(recipe="non_retriable", attempt=attempt, state="escalated",
                       summary=f"status_or_exc={status_or_exc}; body={body_text!r}")
            status = status_or_exc if isinstance(status_or_exc, int) else 502
            msg = explain_http_error(status, body_text, req_id)
            emit_event("turn.failed", req_id=req_id, source="user",
                       status=status, reason="non-retriable")
            self._send_json(status, {
                "error": {"message": msg, "status": status,
                          "req_id": req_id, "ledger": ledger.entries}
            }, extra_headers={"X-Req-Id": req_id,
                              "X-Recovery-Steps": str(ledger.step_count())})
            return

        recipe_id, action, sleep_s = plan
        ledger.add(recipe=recipe_id, attempt=attempt, state="started",
                   summary=f"status_or_exc={status_or_exc}; sleep={sleep_s}s; action={action}")
        emit_event("turn.recovery.started", req_id=req_id, source="fallback",
                   recipe=recipe_id, action=action, attempt=attempt)
        if sleep_s > 0:
            time.sleep(sleep_s)

        if action == "retry_same":
            ledger.add(recipe=recipe_id, attempt=attempt, state="succeeded",
                       summary="retrying same model after backoff")
            return self._run_with_recipes(payload, want_stream, fallback_chain,
                                          ledger, req_id, api_key,
                                          attempt=attempt + 1, max_attempts=max_attempts)
        if action == "truncate_retry":
            notes = []
            ok = truncate_history_in_place(payload, MAX_UPSTREAM_BYTES // 2, notes)
            if ok:
                ledger.add(recipe=recipe_id, attempt=attempt, state="succeeded",
                           summary=notes[-1] if notes else "truncated")
                return self._run_with_recipes(payload, want_stream, fallback_chain,
                                              ledger, req_id, api_key,
                                              attempt=attempt + 1, max_attempts=max_attempts)
            ledger.add(recipe=recipe_id, attempt=attempt, state="failed",
                       summary="cannot truncate further")
            # fall through to fallback if available
            action = "fallback"

        if action == "fallback":
            if fallback_chain:
                next_model = fallback_chain[0]
                rest = fallback_chain[1:]
                payload["model"] = next_model
                apply_quirks(payload, next_model)
                ledger.add(recipe=recipe_id, attempt=attempt, state="succeeded",
                           summary=f"switched to {next_model}")
                return self._run_with_recipes(payload, want_stream, rest, ledger,
                                              req_id, api_key,
                                              attempt=attempt + 1, max_attempts=max_attempts)
            ledger.add(recipe=recipe_id, attempt=attempt, state="escalated",
                       summary="no fallback available")
            status = status_or_exc if isinstance(status_or_exc, int) else 502
            msg = explain_http_error(status, body_text, req_id)
            emit_event("turn.failed", req_id=req_id, source="fallback",
                       status=status, reason="no fallback available")
            self._send_json(status, {
                "error": {"message": msg, "status": status,
                          "req_id": req_id, "ledger": ledger.entries}
            }, extra_headers={"X-Req-Id": req_id,
                              "X-Recovery-Steps": str(ledger.step_count())})
            return

    # ---------- tool-use loop ----------
    def _extract_user_images(self, payload):
        """Pull image data URLs out of user messages so tools can reference them by index."""
        images = []
        for m in payload.get("messages") or []:
            if m.get("role") != "user":
                continue
            c = m.get("content")
            if isinstance(c, list):
                for part in c:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "image_url":
                        url = (part.get("image_url") or {}).get("url")
                        if isinstance(url, str) and url.startswith("data:image/"):
                            images.append(url)
        return images

    def _tool_iter_upstream(self, payload, fallback_chain, ledger, req_id,
                            iter_n, api_key):
        """One non-streaming upstream call inside the tool loop. Tries primary
        then up to one fallback on retriable failure. Returns (served_model, message)
        or None on hard failure (in which case an error response was already sent)."""
        primary = payload["model"]
        candidates = [primary] + (list(fallback_chain[:1]) if fallback_chain else [])
        last_status, last_body = None, ""
        for attempt, model in enumerate(candidates, 1):
            attempt_payload = dict(payload, model=model)
            apply_quirks(attempt_payload, model)
            emit_event("tool.iter.upstream", req_id=req_id, source="user",
                       iter=iter_n, attempt=attempt, model=model)
            try:
                r = requests.post(NVIDIA_URL, headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                }, json=attempt_payload, timeout=180)
            except requests.RequestException as e:
                ledger.add(recipe=f"tool_iter_{iter_n}_network",
                           attempt=attempt, state="failed",
                           summary=f"model={model} {type(e).__name__}: {str(e)[:80]}")
                continue
            if r.ok:
                try:
                    data = r.json()
                    msg = data["choices"][0]["message"]
                except (ValueError, KeyError, IndexError, TypeError) as e:
                    ledger.add(recipe=f"tool_iter_{iter_n}_parse",
                               attempt=attempt, state="failed",
                               summary=f"model={model} bad response: {e}")
                    continue
                ledger.add(recipe=f"tool_iter_{iter_n}_ok",
                           attempt=attempt, state="succeeded",
                           summary=f"model={model}")
                return model, msg
            last_status = r.status_code
            last_body = (r.text or "")[:300]
            ledger.add(recipe=f"tool_iter_{iter_n}_http",
                       attempt=attempt, state="failed",
                       summary=f"model={model} HTTP {r.status_code}")
            if r.status_code in (400, 401, 403, 404):
                break  # non-retriable

        msg = explain_http_error(last_status or 502, last_body, req_id)
        emit_event("turn.failed", req_id=req_id, source="user",
                   status=last_status, reason="tool_iter_failed")
        self._send_json(last_status or 502, {
            "error": {"message": msg, "status": last_status or 502,
                      "req_id": req_id, "ledger": ledger.entries}
        }, extra_headers={"X-Req-Id": req_id,
                          "X-Recovery-Steps": str(ledger.step_count())})
        return None

    def _forward_assistant_msg(self, msg, want_stream, ctx, req_id, ledger,
                                served_model, attempt):
        """Forward a non-streamed assistant message to the client. If the client
        requested streaming, synthesize SSE; otherwise send JSON. Always trails
        artifacts and tool log so the client can render images and the tool list."""
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        artifacts_payload = {
            "_nim_artifacts": ctx.get("artifacts") or [],
            "_nim_tool_log":  ctx.get("tool_log") or [],
        }
        common_headers = {
            "X-Req-Id": req_id,
            "X-NIM-Model": served_model or "",
            "X-NIM-Attempt": str(attempt),
            "X-Recovery-Steps": str(ledger.step_count()),
        }
        emit_event("turn.completed", req_id=req_id, source="user",
                   model=served_model, attempt=attempt,
                   tool_calls=len(ctx.get("tool_log") or []),
                   artifacts=len(ctx.get("artifacts") or []))

        if want_stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Accel-Buffering", "no")
            for k, v in common_headers.items():
                self.send_header(k, v)
            self.end_headers()
            try:
                if reasoning:
                    evt = {"choices": [{"delta": {"reasoning_content": reasoning}, "index": 0}]}
                    self.wfile.write(("data: " + json.dumps(evt, ensure_ascii=False) + "\n\n").encode("utf-8"))
                    self.wfile.flush()
                if content:
                    evt = {"choices": [{"delta": {"content": content}, "index": 0}]}
                    self.wfile.write(("data: " + json.dumps(evt, ensure_ascii=False) + "\n\n").encode("utf-8"))
                    self.wfile.flush()
                # artifacts + tool log trailer (custom data line, not [DONE])
                self.wfile.write(("data: " + json.dumps(artifacts_payload, ensure_ascii=False) + "\n\n").encode("utf-8"))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            body = {
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "reasoning_content": reasoning if reasoning else None,
                    },
                    "finish_reason": "stop",
                }],
                "model": served_model,
                **artifacts_payload,
            }
            self._send_json(200, body, extra_headers=common_headers)

    def _run_with_tools(self, payload, want_stream, fallback_chain, ledger,
                        req_id, api_key):
        """Tool-use loop. Each iteration is non-streaming. Final response is
        forwarded to client as streaming (synthetic SSE) or JSON depending on
        the original want_stream. Recovery is per-iteration; the tool loop itself
        is bounded by MAX_TOOL_ITERS."""
        # Inject tool definitions
        payload["tools"] = [t["schema"] for t in TOOL_REGISTRY.values()]
        payload["tool_choice"] = "auto"

        ctx = {
            "artifacts": [],
            "tool_log": [],
            "user_images": self._extract_user_images(payload),
            # Tool executors that hit NVIDIA endpoints (image_generate,
            # embed_similarity, rerank, ocr, molmim_generate) read this key
            # from ctx — keeping it scoped to the per-turn context dict means
            # we never leak it across users.
            "api_key": api_key,
            # save_artifact() needs this to record ownership; /v1/artifact/{id}
            # uses it to gate access by session user.
            "user_id": self.user_id,
        }

        MAX_TOOL_ITERS = 6
        iteration = 0
        served_model = payload["model"]
        while iteration < MAX_TOOL_ITERS:
            iteration += 1
            iter_payload = dict(payload)
            iter_payload["stream"] = False

            result = self._tool_iter_upstream(iter_payload, fallback_chain,
                                              ledger, req_id, iteration, api_key)
            if result is None:
                return  # error already sent
            served_model, msg = result

            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                # Recovery: some models (notably Llama 4 Maverick) emit tool calls
                # as JSON text in content instead of via the structured tool_calls
                # field. Try to parse and convert before treating as final answer.
                recovered = _try_parse_text_tool_calls(msg.get("content"))
                if recovered:
                    emit_event("tool.text_format_recovered", req_id=req_id,
                               source="user", model=served_model,
                               iter=iteration, count=len(recovered))
                    ledger.add(recipe=f"tool_iter_{iteration}_text_recovery",
                               attempt=1, state="succeeded",
                               summary=f"recovered {len(recovered)} text-format call(s)")
                    msg["tool_calls"] = recovered
                    msg["content"] = ""  # clear raw JSON; it's now a structured call
                    tool_calls = recovered
            if not tool_calls:
                # Final answer
                self._forward_assistant_msg(msg, want_stream, ctx, req_id, ledger,
                                            served_model=served_model, attempt=iteration)
                return

            # Append assistant turn (with tool_calls) to history
            payload["messages"].append({
                "role": "assistant",
                "content": msg.get("content") or "",
                "tool_calls": tool_calls,
            })

            # Execute each tool call
            for tc in tool_calls:
                fn = tc.get("function") or {}
                name = fn.get("name") or ""
                args_raw = fn.get("arguments") or "{}"
                if isinstance(args_raw, str):
                    try:
                        args = json.loads(args_raw)
                    except json.JSONDecodeError:
                        args = {}
                else:
                    args = args_raw if isinstance(args_raw, dict) else {}

                tool_def = TOOL_REGISTRY.get(name)
                t0 = time.time()
                emit_event("tool.started", req_id=req_id, source="user",
                           tool=name, iter=iteration, args_keys=list(args.keys()))
                if not tool_def:
                    tool_result = {"error": f"unknown tool: {name}"}
                else:
                    try:
                        tool_result = tool_def["executor"](args, ctx)
                    except Exception as e:
                        tool_result = {"error": f"executor exception: {type(e).__name__}: {str(e)[:160]}"}
                t_ms = int((time.time() - t0) * 1000)
                ok = "error" not in tool_result
                emit_event("tool.completed", req_id=req_id, source="user",
                           tool=name, ok=ok, latency_ms=t_ms)

                ctx["tool_log"].append({
                    "name": name, "args": args, "ok": ok,
                    "summary": (tool_result.get("summary")
                                or tool_result.get("error")
                                or ("ok" if ok else "failed")),
                    "latency_ms": t_ms,
                })

                # LLM-facing result: strip large binary fields (b64, matrix can stay if small)
                llm_result = {k: v for k, v in tool_result.items() if k != "b64"}
                # Cap matrix size if huge
                if isinstance(llm_result.get("matrix"), list) and len(llm_result["matrix"]) > 16:
                    llm_result["matrix"] = "(omitted; too large; see ranked_pairs)"
                payload["messages"].append({
                    "role": "tool",
                    "tool_call_id": tc.get("id"),
                    "content": json.dumps(llm_result, ensure_ascii=False)[:8000],
                })

        # Loop exhausted
        ledger.add(recipe="tool_loop_exceeded", attempt=iteration, state="escalated",
                   summary=f"reached max tool iterations {MAX_TOOL_ITERS}")
        emit_event("turn.failed", req_id=req_id, source="user",
                   reason="tool_loop_exceeded", iters=MAX_TOOL_ITERS)
        self._send_json(502, {
            "error": {
                "message": f"[{req_id}] tool-call loop limit ({MAX_TOOL_ITERS}) reached",
                "status": 502, "req_id": req_id,
                "ledger": ledger.entries,
                "_nim_tool_log": ctx.get("tool_log") or [],
            }
        }, extra_headers={"X-Req-Id": req_id,
                          "X-Recovery-Steps": str(ledger.step_count())})


# ============================ TOOL REGISTRY ============================
#
# Exposes specialized NVIDIA NIM models — image generation, image editing, embeddings — as
# OpenAI-format function tools. Chat models (Llama 4, DeepSeek V4, Qwen3) call these via
# `tool_calls` in their response. The server executes the tool, accumulates artifacts in a
# turn-scoped context dict, and feeds a text-only summary back to the LLM.
#
# Adding new tools (simulation, biology, etc.):
#   1. Implement an executor function with signature (args: dict, ctx: dict) -> dict.
#      Return {"ok": True, "summary": "...", ...} on success, {"error": "..."} on failure.
#      For binary outputs, append to ctx["artifacts"] and return only a text summary.
#   2. Add a JSON-schema entry to TOOL_REGISTRY below.
#   3. Restart the server. The chat model will see the tool automatically.

NVIDIA_GENAI_BASE  = "https://ai.api.nvidia.com/v1/genai"
NVIDIA_EMBED_URL   = "https://integrate.api.nvidia.com/v1/embeddings"

# Image generation/edit models on the NVIDIA API catalog. Each entry is keyed by a
# short alias and resolves to (model_path, default_steps). The model_path is the
# vendor/model segment of the URL: https://ai.api.nvidia.com/v1/genai/{model_path}
IMG_MODEL_MAP = {
    "flux.1-schnell":  ("black-forest-labs/flux.1-schnell",  4),
    "flux.1-dev":      ("black-forest-labs/flux.1-dev",     50),
    "flux.1-kontext":  ("black-forest-labs/flux.1-kontext-dev", 30),
    "sdxl-turbo":      ("stabilityai/sdxl-turbo",            2),
    "sdxl":            ("stabilityai/stable-diffusion-xl",  50),
    "sd3-medium":      ("stabilityai/stable-diffusion-3-medium", 28),
}

# Embedding model aliases.
EMBED_MODEL_DEFAULT = "nvidia/nv-embedqa-e5-v5"
EMBED_MODEL_ALIASES = {
    "e5":         "nvidia/nv-embedqa-e5-v5",
    "llama-1b":   "nvidia/llama-3.2-nv-embedqa-1b-v2",
    "mistral-7b": "nvidia/nv-embedqa-mistral-7b-v2",
}


def _nv_post(url: str, payload: dict, api_key: str, timeout: int = 180):
    """Helper for tool executors making NIM REST calls. Returns parsed JSON
    or raises RuntimeError with a short, user-facing message. Tool executors
    receive the user's API key via ctx["api_key"] and pass it here."""
    if not api_key:
        raise RuntimeError("missing API key (no user context?)")
    try:
        r = requests.post(url, headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }, json=payload, timeout=timeout)
    except requests.RequestException as e:
        raise RuntimeError(f"network: {type(e).__name__}: {str(e)[:120]}")
    if not r.ok:
        body = (r.text or "")[:240]
        raise RuntimeError(f"HTTP {r.status_code}: {body}")
    try:
        return r.json()
    except ValueError as e:
        raise RuntimeError(f"non-JSON response: {e}")


def _decode_genai_response(data: dict) -> tuple:
    """Catalog responses come back in two slightly different shapes:
    Flux:  {"artifacts":[{"base64":"...","mimeType":"image/..."}]}
    SDXL:  {"artifacts":[{"base64":"...","finishReason":"..."}]}
    Some endpoints also return {"image":"data:image/...;base64,..."}.
    Returns (b64, mime). Raises RuntimeError on bad shape.
    """
    if isinstance(data.get("artifacts"), list) and data["artifacts"]:
        a0 = data["artifacts"][0]
        if isinstance(a0, dict) and a0.get("base64"):
            return a0["base64"], a0.get("mimeType") or "image/png"
    if isinstance(data.get("image"), str) and data["image"].startswith("data:"):
        m = re.match(r"^data:([^;]+);base64,(.+)$", data["image"])
        if m:
            return m.group(2), m.group(1)
    if isinstance(data.get("data"), list) and data["data"]:
        d0 = data["data"][0]
        if isinstance(d0, dict) and d0.get("b64_json"):
            return d0["b64_json"], "image/png"
    raise RuntimeError(f"unrecognized response shape: keys={list(data.keys())[:6]}")


def tool_image_generate(args: dict, ctx: dict) -> dict:
    """Generate an image via NVIDIA catalog Visual GenAI endpoints.
    URL: https://ai.api.nvidia.com/v1/genai/{vendor}/{model}
    Flux uses {prompt, cfg_scale, mode:'base', seed, steps}.
    SDXL uses {text_prompts:[{text}], cfg_scale, sampler, seed, steps}.
    """
    prompt = (args.get("prompt") or "").strip()
    if not prompt:
        return {"error": "empty prompt"}
    alias = args.get("model") or "flux.1-schnell"
    if alias not in IMG_MODEL_MAP:
        alias = "flux.1-schnell"
    model_path, default_steps = IMG_MODEL_MAP[alias]
    invoke_url = f"{NVIDIA_GENAI_BASE}/{model_path}"

    # Per-family payload shape
    is_flux = model_path.startswith("black-forest-labs/")
    is_sdxl = model_path.startswith("stabilityai/")
    seed = args.get("seed", 0)
    if not isinstance(seed, int):
        seed = 0
    steps = args.get("steps") or default_steps
    if not isinstance(steps, int) or steps < 1:
        steps = default_steps
    steps = min(steps, 100)

    # Resolution — Flux schnell supports 1024x1024 (default), 768x1344, 1344x768,
    # 896x1152, 1152x896, 832x1216, 1216x832 (per NVIDIA model card).
    # Other Flux variants accept the same set. SDXL family accepts 1024x1024.
    width = args.get("width") or 1024
    height = args.get("height") or 1024
    if not isinstance(width, int) or width < 64:
        width = 1024
    if not isinstance(height, int) or height < 64:
        height = 1024

    if is_flux:
        # Per NVIDIA's official `Get API Key` snippet for flux.1-schnell:
        #   payload = {prompt, width, height, seed, steps}
        # NO `mode` field. NO `cfg_scale` field. Sending unknown fields makes
        # NVIDIA reject the request with a confusing "Function id not found"
        # error — it's actually a payload-schema validation failure surfaced
        # as a routing failure.
        payload = {
            "prompt": prompt[:2000],
            "width": width,
            "height": height,
            "seed": seed,
            "steps": steps,
        }
    elif is_sdxl:
        payload = {
            "text_prompts": [{"text": prompt[:2000], "weight": 1.0}],
            "cfg_scale": 5,
            "sampler": "K_EULER_ANCESTRAL",
            "seed": seed,
            "steps": steps,
        }
    else:
        payload = {"prompt": prompt[:2000], "seed": seed, "steps": steps}

    try:
        data = _nv_post(invoke_url, payload, ctx.get("api_key", ""), timeout=240)
    except RuntimeError as e:
        return {"error": str(e)}
    try:
        b64, mime = _decode_genai_response(data)
    except RuntimeError as e:
        return {"error": str(e)}

    art = save_artifact(ctx, "image", mime, base64.b64decode(b64),
                        prompt=prompt, model=model_path)
    ctx["artifacts"].append(art)
    return {
        "ok": True, "artifact_id": art["id"], "model": model_path,
        "summary": (f"Image generated and rendered inline for the user "
                    f"(id={art['id']}, model={alias}, {steps} steps). "
                    "The user already sees the image. Briefly acknowledge in the "
                    "user's language and ask if adjustments are needed. Do not "
                    "describe the image content in detail."),
    }


def tool_image_edit(args: dict, ctx: dict) -> dict:
    """Edit user-uploaded images via FLUX.1 Kontext.
    Catalog endpoint: https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-kontext-dev
    Payload includes the input image as a data:image/...;base64,... string.
    """
    prompt = (args.get("prompt") or "").strip()
    if not prompt:
        return {"error": "empty prompt"}
    indices = args.get("image_indices") or [0]
    if isinstance(indices, int):
        indices = [indices]
    if not isinstance(indices, list):
        return {"error": "image_indices must be a list of integers"}

    user_images = ctx.get("user_images") or []
    if not user_images:
        return {"error": "no user-uploaded images available in conversation context"}
    selected = []
    for i in indices:
        if isinstance(i, int) and 0 <= i < len(user_images):
            selected.append(user_images[i])
    if not selected:
        return {"error": f"no valid image indices in {indices} (have {len(user_images)})"}

    # Catalog Kontext takes a single image
    model_path = "black-forest-labs/flux.1-kontext-dev"
    invoke_url = f"{NVIDIA_GENAI_BASE}/{model_path}"
    seed = args.get("seed", 0) if isinstance(args.get("seed"), int) else 0
    steps = args.get("steps") or 30
    steps = min(int(steps), 100) if isinstance(steps, int) else 30

    payload = {
        "prompt": prompt[:2000],
        "image": selected[0],
        "seed": seed,
        "steps": steps,
    }
    try:
        data = _nv_post(invoke_url, payload, ctx.get("api_key", ""), timeout=300)
    except RuntimeError as e:
        return {"error": str(e)}
    try:
        b64, mime = _decode_genai_response(data)
    except RuntimeError as e:
        return {"error": str(e)}

    art = save_artifact(ctx, "image", mime, base64.b64decode(b64),
                        prompt=prompt, model=model_path, edited_from=indices)
    ctx["artifacts"].append(art)
    return {
        "ok": True, "artifact_id": art["id"], "model": model_path,
        "summary": f"Edited image rendered inline (id={art['id']}, source images {indices}). The user already sees it.",
    }


def tool_embed_similarity(args: dict, ctx: dict) -> dict:
    """Compute pairwise cosine similarity for a list of texts."""
    import math
    texts = args.get("texts")
    if not isinstance(texts, list) or len(texts) < 2:
        return {"error": "need a list of >= 2 text strings"}
    if any(not isinstance(t, str) or not t.strip() for t in texts):
        return {"error": "all texts must be non-empty strings"}
    alias = args.get("model") or "e5"
    model = EMBED_MODEL_ALIASES.get(alias, alias if "/" in alias else EMBED_MODEL_DEFAULT)

    payload = {
        "model": model,
        "input": [t[:8000] for t in texts],
        "input_type": args.get("input_type") or "passage",
        "encoding_format": "float", "truncate": "END",
    }
    try:
        data = _nv_post(NVIDIA_EMBED_URL, payload, ctx.get("api_key", ""), timeout=60)
    except RuntimeError as e:
        return {"error": str(e)}

    try:
        embeddings = [item["embedding"] for item in data["data"]]
    except (KeyError, TypeError):
        return {"error": "bad response shape"}

    def cos(a, b):
        dot = sum(x*y for x, y in zip(a, b))
        na = math.sqrt(sum(x*x for x in a))
        nb = math.sqrt(sum(x*x for x in b))
        return dot / (na * nb) if na * nb > 0 else 0.0

    n = len(embeddings)
    matrix = [[round(cos(embeddings[i], embeddings[j]), 4) for j in range(n)] for i in range(n)]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append({"i": i, "j": j, "cos": matrix[i][j],
                          "preview_i": texts[i][:60], "preview_j": texts[j][:60]})
    pairs.sort(key=lambda p: -p["cos"])
    summary = (f"{n} texts compared with {model.split('/')[-1]}. Top pair: "
               f"{pairs[0]['cos']:.3f} between [{pairs[0]['i']}] and [{pairs[0]['j']}]") if pairs else "n/a"
    return {
        "ok": True, "model": model, "n": n,
        "dim": len(embeddings[0]) if embeddings else 0,
        "matrix": matrix, "ranked_pairs": pairs[:10],
        "summary": summary,
    }


def tool_math_eval(args: dict, ctx: dict) -> dict:
    """Symbolic math via SymPy (local, exact)."""
    try:
        import sympy
    except ImportError:
        return {"error": "sympy not installed (pip install sympy)"}
    expr_str = (args.get("expression") or "").strip()
    if not expr_str:
        return {"error": "empty expression"}
    op = (args.get("operation") or "simplify").lower()
    var_str = args.get("variable") or "x"
    try:
        expr = sympy.sympify(expr_str)
        x = sympy.Symbol(var_str)
        if   op == "simplify":     result = sympy.simplify(expr)
        elif op == "expand":       result = sympy.expand(expr)
        elif op == "factor":       result = sympy.factor(expr)
        elif op == "differentiate":result = sympy.diff(expr, x)
        elif op == "integrate":    result = sympy.integrate(expr, x)
        elif op == "solve":        result = sympy.solve(expr, x)
        elif op == "evaluate":     result = expr.evalf()
        else: return {"error": f"unknown operation: {op}"}
    except Exception as e:
        return {"error": f"sympy: {type(e).__name__}: {str(e)[:160]}"}
    res_str = str(result)
    try:
        latex = sympy.latex(result)
    except Exception:
        latex = ""
    return {
        "ok": True, "operation": op, "expression": expr_str,
        "variable": var_str, "result": res_str, "latex": latex,
        "summary": f"{op}({expr_str}) = {res_str[:140]}",
    }


# ---- new NVIDIA catalog endpoints ----
NVIDIA_RERANK_URL = "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking"
NVIDIA_OCR_URL    = "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-ocr"
NVIDIA_MOLMIM_URL = "https://health.api.nvidia.com/v1/biology/nvidia/molmim/generate"

# Local VibeVoice worker. Configurable via VIBEVOICE_WORKER env var.
VIBEVOICE_WORKER = os.environ.get("VIBEVOICE_WORKER", "http://127.0.0.1:8766")


def tool_rerank(args: dict, ctx: dict) -> dict:
    """Re-rank passages by relevance to a query using NVIDIA reranker NIM."""
    query = (args.get("query") or "").strip()
    passages = args.get("passages") or []
    if not query:
        return {"error": "empty query"}
    if not isinstance(passages, list) or len(passages) < 2:
        return {"error": "need a list of >= 2 passages"}
    if any(not isinstance(p, str) or not p.strip() for p in passages):
        return {"error": "all passages must be non-empty strings"}
    top_n = args.get("top_n")
    if not isinstance(top_n, int) or top_n < 1:
        top_n = len(passages)

    payload = {
        "model": "nvidia/llama-3.2-nv-rerankqa-1b-v2",
        "query": {"text": query[:8000]},
        "passages": [{"text": p[:8000]} for p in passages],
        "truncate": "END",
    }
    try:
        data = _nv_post(NVIDIA_RERANK_URL, payload, ctx.get("api_key", ""), timeout=30)
    except RuntimeError as e:
        return {"error": str(e)}

    rankings = data.get("rankings") or []
    if not rankings:
        return {"error": "no rankings returned"}
    rankings.sort(key=lambda r: -r.get("logit", 0))
    ranked = []
    for r in rankings[:top_n]:
        idx = r.get("index")
        if isinstance(idx, int) and 0 <= idx < len(passages):
            ranked.append({
                "rank": len(ranked) + 1,
                "index": idx,
                "score": round(float(r.get("logit", 0)), 4),
                "preview": passages[idx][:120],
            })
    summary = (f"Reranked {len(passages)} passages. Top: idx={ranked[0]['index']} "
               f"score={ranked[0]['score']}") if ranked else "no usable rankings"
    return {"ok": True, "ranked": ranked, "n_input": len(passages), "summary": summary}


def tool_ocr(args: dict, ctx: dict) -> dict:
    """Extract text from a user-uploaded image via NemoRetriever OCR."""
    indices = args.get("image_indices") or [0]
    if isinstance(indices, int):
        indices = [indices]
    if not isinstance(indices, list):
        return {"error": "image_indices must be a list of integers"}
    user_images = ctx.get("user_images") or []
    if not user_images:
        return {"error": "no user-uploaded images available"}
    selected = []
    for i in indices:
        if isinstance(i, int) and 0 <= i < len(user_images):
            selected.append(user_images[i])
    if not selected:
        return {"error": f"no valid image indices in {indices} (have {len(user_images)})"}

    merge_level = args.get("merge_level") or "word"
    if merge_level not in ("word", "line", "paragraph"):
        merge_level = "word"

    payload = {
        "input": [{"type": "image_url", "url": url} for url in selected],
        "merge_levels": [merge_level],
    }
    try:
        data = _nv_post(NVIDIA_OCR_URL, payload, ctx.get("api_key", ""), timeout=60)
    except RuntimeError as e:
        return {"error": str(e)}

    # Response shape: { "data": [ { "text_predictions": [...], "bounding_boxes": [[...]] } ] }
    out_per_image = []
    for img_idx, img_data in enumerate(data.get("data") or []):
        texts = img_data.get("text_predictions") or []
        joined = " ".join([t for t in texts if isinstance(t, str)])
        out_per_image.append({
            "image_index": indices[img_idx] if img_idx < len(indices) else img_idx,
            "text": joined.strip(),
            "n_tokens": len(texts),
        })
    if not out_per_image:
        return {"error": "OCR returned no data"}
    total_chars = sum(len(o["text"]) for o in out_per_image)
    return {
        "ok": True, "results": out_per_image,
        "summary": f"OCR extracted {total_chars} chars from {len(out_per_image)} image(s).",
    }


def tool_molmim_generate(args: dict, ctx: dict) -> dict:
    """Generate novel small molecules around a seed SMILES via MolMIM."""
    smi = (args.get("smiles") or "").strip()
    if not smi:
        return {"error": "empty SMILES"}
    n = args.get("num_molecules", 5)
    if not isinstance(n, int) or n < 1 or n > 30:
        n = 5
    algorithm = args.get("algorithm") or "CMA-ES"
    if algorithm not in ("CMA-ES", "none"):
        algorithm = "CMA-ES"
    prop = args.get("property") or "QED"
    if prop not in ("QED", "plogP"):
        prop = "QED"

    payload = {
        "smi": smi[:200], "num_molecules": n,
        "algorithm": algorithm, "property_name": prop,
        "minimize": False, "min_similarity": 0.3,
        "particles": 30, "iterations": 10,
        "scaled_radius": 1.0,
    }
    try:
        data = _nv_post(NVIDIA_MOLMIM_URL, payload, ctx.get("api_key", ""), timeout=120)
    except RuntimeError as e:
        return {"error": str(e)}
    generated = data.get("generated") or data.get("molecules") or []
    if not generated:
        return {"error": f"no molecules returned. raw keys={list(data.keys())[:6]}"}
    # Normalize: items may be strings or {smiles, score} dicts
    out = []
    for g in generated[:n]:
        if isinstance(g, str):
            out.append({"smiles": g, "score": None})
        elif isinstance(g, dict):
            out.append({
                "smiles": g.get("smiles") or g.get("smi") or "",
                "score": g.get("score") or g.get(prop.lower()),
            })
    return {
        "ok": True, "seed": smi, "property": prop,
        "n_generated": len(out), "molecules": out,
        "summary": f"Generated {len(out)} novel SMILES around seed (optimizing {prop}).",
    }


# GitHub fetch tools — let the model browse/inspect public repos without
# resorting to `git clone` from inside the sandbox (which has no network and
# no git binary). Two strict allowlists for SSRF safety: only `repo` strings
# of the form "owner/name" and only the two GitHub hostnames are reachable.

_GITHUB_RAW = "https://raw.githubusercontent.com"
_GITHUB_API = "https://api.github.com"
_GITHUB_REPO_RE = re.compile(r"^[A-Za-z0-9._\-]+/[A-Za-z0-9._\-]+$")
_GITHUB_PATH_RE = re.compile(r"^[A-Za-z0-9._\-/]*$")
# Git refs: branch / tag / SHA. Allowed chars per git's ref-format rules
# (intersected with what's safe in a URL path or query). Defense in depth —
# even though host is hard-coded, this stops a `&token=...` style payload
# from corrupting the query string in github_list_files.
_GITHUB_REF_RE = re.compile(r"^[A-Za-z0-9._/\-]{1,200}$")
_GITHUB_FETCH_MAX_BYTES = 256 * 1024


def tool_github_fetch_file(args: dict, ctx: dict) -> dict:
    """Fetch a single file's text content from a public GitHub repo via
    raw.githubusercontent.com. 256 KB cap; UTF-8 only."""
    repo = (args.get("repo") or "").strip()
    path = (args.get("path") or "").strip().lstrip("/")
    ref = (args.get("ref") or "HEAD").strip() or "HEAD"
    if not _GITHUB_REPO_RE.match(repo):
        return {"error": "repo must be 'owner/name' (no slashes in either)"}
    if not path or not _GITHUB_PATH_RE.match(path) or ".." in path:
        return {"error": "invalid path"}
    if not _GITHUB_REF_RE.match(ref):
        return {"error": "invalid ref (alphanumerics, '.', '_', '/', '-' only)"}
    url = f"{_GITHUB_RAW}/{repo}/{ref}/{path}"
    try:
        r = requests.get(url, timeout=30, stream=True,
                         headers={"User-Agent": "NIMINI/1.0"})
    except requests.RequestException as e:
        return {"error": f"network: {type(e).__name__}: {str(e)[:80]}"}
    try:
        if r.status_code == 404:
            return {"error": f"not found: {repo}@{ref}:{path}"}
        if not r.ok:
            return {"error": f"HTTP {r.status_code}"}
        # Read up to the cap; abort if more
        buf = b""
        for chunk in r.iter_content(chunk_size=8192):
            buf += chunk
            if len(buf) > _GITHUB_FETCH_MAX_BYTES:
                return {"error": f"file too large (>{_GITHUB_FETCH_MAX_BYTES // 1024} KiB cap)"}
    finally:
        r.close()
    try:
        text = buf.decode("utf-8")
    except UnicodeDecodeError:
        return {"error": "binary or non-UTF-8 file (this tool returns text only)"}
    return {
        "ok": True, "repo": repo, "path": path, "ref": ref,
        "size": len(buf), "content": text,
        "summary": f"Fetched {path} from {repo}@{ref} ({len(buf)} bytes).",
    }


def tool_github_list_files(args: dict, ctx: dict) -> dict:
    """List files/directories at a path in a public GitHub repo, via the
    GitHub Contents API. Returns name/path/type/size for each entry.
    Unauthenticated — subject to GitHub's 60-requests-per-hour-per-IP cap."""
    repo = (args.get("repo") or "").strip()
    path = (args.get("path") or "").strip().lstrip("/")
    ref = (args.get("ref") or "").strip()
    if not _GITHUB_REPO_RE.match(repo):
        return {"error": "repo must be 'owner/name'"}
    if path and (not _GITHUB_PATH_RE.match(path) or ".." in path):
        return {"error": "invalid path"}
    if ref and not _GITHUB_REF_RE.match(ref):
        return {"error": "invalid ref (alphanumerics, '.', '_', '/', '-' only)"}
    url = f"{_GITHUB_API}/repos/{repo}/contents/{path}"
    if ref:
        url += f"?ref={ref}"
    try:
        r = requests.get(url, timeout=20, headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "NIMINI/1.0",
        })
    except requests.RequestException as e:
        return {"error": f"network: {type(e).__name__}: {str(e)[:80]}"}
    if r.status_code == 404:
        return {"error": f"not found: {repo}{('@' + ref) if ref else ''}:{path or '.'}"}
    if r.status_code == 403:
        return {"error": "GitHub API rate-limit (60/hour unauth) — try again later"}
    if not r.ok:
        return {"error": f"HTTP {r.status_code}"}
    try:
        data = r.json()
    except ValueError:
        return {"error": "non-JSON response"}
    if isinstance(data, dict):  # single file
        return {
            "ok": True, "type": "file",
            "name": data.get("name"), "path": data.get("path"),
            "size": data.get("size"),
            "summary": f"{data.get('path')!r} is a single file ({data.get('size')} bytes); use github_fetch_file to read it.",
        }
    entries = []
    for item in data if isinstance(data, list) else []:
        if isinstance(item, dict):
            entries.append({
                "name": item.get("name"),
                "path": item.get("path"),
                "type": item.get("type"),
                "size": item.get("size"),
            })
    return {
        "ok": True, "type": "dir", "path": path or ".",
        "entries": entries[:200],
        "summary": f"{len(entries)} entries under {repo}{('@' + ref) if ref else ''}:{path or '.'}",
    }


def tool_web_search(args: dict, ctx: dict) -> dict:
    """Web search via DuckDuckGo Instant Answer + HTML scrape fallback.
    No API key needed. Returns top-N short snippets."""
    query = (args.get("query") or "").strip()
    if not query:
        return {"error": "empty query"}
    n = args.get("max_results", 5)
    if not isinstance(n, int) or n < 1 or n > 10:
        n = 5

    # 1) DDG Instant Answer JSON
    try:
        r = requests.get("https://api.duckduckgo.com/", params={
            "q": query, "format": "json", "no_redirect": "1", "no_html": "1",
        }, timeout=15, headers={"User-Agent": "NIMChat/0.3"})
    except requests.RequestException as e:
        return {"error": f"network: {type(e).__name__}: {str(e)[:80]}"}
    results = []
    if r.ok:
        try:
            data = r.json()
        except ValueError:
            data = {}
        if data.get("AbstractText"):
            results.append({
                "title": data.get("Heading") or query,
                "url": data.get("AbstractURL") or "",
                "snippet": data["AbstractText"][:400],
                "source": data.get("AbstractSource") or "DuckDuckGo",
            })
        for topic in (data.get("RelatedTopics") or [])[:n * 2]:
            if not isinstance(topic, dict):
                continue
            text = topic.get("Text") or ""
            url = topic.get("FirstURL") or ""
            if not text or not url:
                continue
            results.append({
                "title": text.split(" - ")[0][:120],
                "url": url, "snippet": text[:300], "source": "DDG",
            })
            if len(results) >= n:
                break

    # 2) Fallback: DDG HTML if Instant Answer was empty
    if not results:
        try:
            r2 = requests.get("https://html.duckduckgo.com/html/", params={"q": query},
                              timeout=15, headers={"User-Agent": "Mozilla/5.0 NIMChat/0.3"})
            if r2.ok:
                # crude regex parse — DDG HTML format is stable enough
                text = r2.text
                pattern = re.compile(
                    r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>'
                    r'.*?<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
                    re.DOTALL,
                )
                for m in pattern.finditer(text):
                    url, title, snippet = m.group(1), m.group(2), m.group(3)
                    title = re.sub(r"<[^>]+>", "", title).strip()
                    snippet = re.sub(r"<[^>]+>", "", snippet).strip()
                    # DDG sometimes wraps URLs in /l/?uddg=
                    if "uddg=" in url:
                        from urllib.parse import unquote, urlparse, parse_qs
                        try:
                            qs = parse_qs(urlparse(url).query)
                            if qs.get("uddg"):
                                url = unquote(qs["uddg"][0])
                        except Exception:
                            pass
                    results.append({
                        "title": title[:120], "url": url,
                        "snippet": snippet[:300], "source": "DDG-HTML",
                    })
                    if len(results) >= n:
                        break
        except requests.RequestException:
            pass

    if not results:
        return {"error": "no results"}
    summary = f"Found {len(results)} results for {query!r}. Top: {results[0]['title'][:80]}"
    return {"ok": True, "query": query, "n_results": len(results),
            "results": results[:n], "summary": summary}


# ============================ CODE_EXECUTE SANDBOX ============================
#
# Two backends, picked by capability at startup:
#
#   "runsc"   — gVisor user-space kernel. Strongest containment: own syscall
#               implementation, AF_ALG simply not implemented (EAFNOSUPPORT).
#               +~500ms cold-start, +~20MB RAM per call. Preferred when present.
#   "unshare" — Linux user/network/PID/IPC namespaces + seccomp BPF filter
#               that blocks AF_ALG socket creation. Always available on modern
#               Linux. ~100ms start, ~zero RAM overhead. Same kernel as host
#               so future kernel CVEs may bypass it.
#
# The choice is logged once at first use; both backends pass the same
# verification probes (network blocked, AF_ALG blocked, PID isolated).

_SANDBOX_BACKEND: str | None = None  # "runsc" | "unshare" | "none"


def _sandbox_backend() -> str:
    """Probe once and cache which sandbox backend is usable on this host.
    Returns one of: "runsc", "unshare", "none"."""
    global _SANDBOX_BACKEND
    if _SANDBOX_BACKEND is not None:
        return _SANDBOX_BACKEND
    import subprocess as _sp
    # Prefer gVisor when installed
    try:
        r = _sp.run(
            ["runsc", "--rootless", "-network=none", "-platform=systrap",
             "do", "true"],
            capture_output=True, timeout=10,
        )
        if r.returncode == 0:
            _SANDBOX_BACKEND = "runsc"
            print("[init] sandbox backend: runsc (gVisor)", file=sys.stderr)
            return _SANDBOX_BACKEND
    except (FileNotFoundError, OSError, _sp.TimeoutExpired):
        pass
    # Fallback: namespaces
    try:
        r = _sp.run(
            ["unshare", "-U", "-r", "-n", "-p", "-i", "--fork", "--mount-proc",
             "--", "true"],
            capture_output=True, timeout=5,
        )
        if r.returncode == 0:
            _SANDBOX_BACKEND = "unshare"
            print("[init] sandbox backend: unshare + seccomp "
                  "(install gVisor for stronger containment)", file=sys.stderr)
            return _SANDBOX_BACKEND
    except (FileNotFoundError, OSError, _sp.TimeoutExpired):
        pass
    _SANDBOX_BACKEND = "none"
    print("[init] sandbox backend: NONE — code_execute disabled", file=sys.stderr)
    return _SANDBOX_BACKEND


# In-script preamble: setrlimit + (unshare-only) seccomp BPF blocking AF_ALG.
# Under gVisor we skip the seccomp install — gVisor's user-space kernel doesn't
# implement AF_ALG at all (returns EAFNOSUPPORT), so the path is closed by the
# platform itself. Setting these limits in-script (vs prlimit on the parent)
# means the user's snippet inherits them transparently.
_SANDBOX_RLIMIT_PREAMBLE = (
    "import resource\n"
    "for _r, _lim in ("
    "    (resource.RLIMIT_AS,    (1<<30, 1<<30)),"
    "    (resource.RLIMIT_CPU,   (60, 60)),"
    "    (resource.RLIMIT_FSIZE, (32<<20, 32<<20)),"
    "    (resource.RLIMIT_NPROC, (32, 32)),"
    "    (resource.RLIMIT_CORE,  (0, 0)),"
    "):\n"
    "    try: resource.setrlimit(_r, _lim)\n"
    "    except Exception: pass\n"
)

# AF_ALG socket() block via seccomp BPF. Only used under the unshare backend;
# gVisor doesn't need it. Kept in a string so the child interpreter installs
# it itself (no host-side ctypes plumbing).
_SANDBOX_SECCOMP_PREAMBLE = (
    "try:\n"
    "    import ctypes as _c, struct as _s, errno as _e\n"
    "    _libc = _c.CDLL('libc.so.6', use_errno=True)\n"
    "    _libc.prctl.argtypes = [_c.c_int]+[_c.c_ulong]*4\n"
    "    _libc.prctl.restype  = _c.c_int\n"
    "    def _f(co,jt,jf,k): return _s.pack('HBBI',co,jt,jf,k&0xFFFFFFFF)\n"
    "    _prog = b''.join([\n"
    "        _f(0x20,0,0,4), _f(0x15,0,4,0xC000003E),\n"
    "        _f(0x20,0,0,0), _f(0x15,0,2,41),\n"
    "        _f(0x20,0,0,16),_f(0x15,1,0,38),\n"
    "        _f(0x06,0,0,0x7FFF0000),\n"
    "        _f(0x06,0,0,0x00050000|_e.EACCES),\n"
    "    ])\n"
    "    _buf = _c.create_string_buffer(_prog,len(_prog))\n"
    "    _fp  = _s.pack('H6xQ', len(_prog)//8, _c.addressof(_buf))\n"
    "    _fpb = _c.create_string_buffer(_fp,len(_fp))\n"
    "    assert _libc.prctl(38,1,0,0,0)==0\n"               # PR_SET_NO_NEW_PRIVS
    "    assert _libc.prctl(22,2,_c.addressof(_fpb),0,0)==0\n"  # PR_SET_SECCOMP filter
    "except Exception: pass\n"
)


def tool_code_execute(args: dict, ctx: dict) -> dict:
    """Execute a short Python snippet inside a sandbox.

    Backend chosen by `_sandbox_backend()`:
      - runsc   (gVisor user-space kernel; strongest)
      - unshare (Linux namespaces + seccomp AF_ALG block; fallback)

    Both backends:
      - Block all network (no exfil channel).
      - Block / not-implement AF_ALG (CVE-2026-31431 entry point).
      - Cap memory/cpu/file/proc via setrlimit in the snippet preamble.
      - Use a fresh tempdir as cwd; only image files written there are pulled
        out as artifacts.

    The forbidden-string filter that previously lived here was trivially
    bypassable (`__import__("sub"+"process")`); the sandbox is the actual
    containment layer."""
    import subprocess
    import tempfile
    import shutil

    code = args.get("code") or ""
    if not code.strip():
        return {"error": "empty code"}
    if len(code) > 10000:
        return {"error": "code too long (>10000 chars)"}
    timeout_s = args.get("timeout", 10)
    if not isinstance(timeout_s, (int, float)) or timeout_s < 1 or timeout_s > 60:
        timeout_s = 10

    backend = _sandbox_backend()
    if backend == "none":
        return {"error": ("code execution disabled: no sandbox backend available "
                          "(install gVisor 'runsc' or enable unprivileged user "
                          "namespaces).")}

    tmp = tempfile.mkdtemp(prefix="nimchat_exec_")
    try:
        script_path = os.path.join(tmp, "snippet.py")
        if backend == "runsc":
            preamble = _SANDBOX_RLIMIT_PREAMBLE
            # gVisor's syscall layer rejects AF_ALG natively; no seccomp needed.
            sandbox_cmd = [
                "runsc", "--rootless", "-network=none", "-platform=systrap",
                "do", "--cwd", tmp,
                sys.executable, script_path,
            ]
        else:  # "unshare"
            preamble = _SANDBOX_RLIMIT_PREAMBLE + _SANDBOX_SECCOMP_PREAMBLE
            sandbox_cmd = [
                "unshare", "-U", "-r", "-n", "-p", "-i",
                "--fork", "--mount-proc",
                "--", sys.executable, script_path,
            ]
        guarded = preamble + "# --- user code below ---\n" + code
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(guarded)
        try:
            proc = subprocess.run(
                sandbox_cmd, cwd=tmp, capture_output=True, text=True,
                timeout=timeout_s, check=False,
            )
        except subprocess.TimeoutExpired:
            return {"error": f"timeout after {timeout_s}s"}
        except FileNotFoundError:
            return {"error": f"sandbox unavailable: backend '{backend}' binary missing"}
        stdout = (proc.stdout or "")[-4000:]
        stderr = (proc.stderr or "")[-2000:]
        ok = proc.returncode == 0
        # If the snippet wrote any image files, attach the first as artifact
        artifact_summary = ""
        for fname in sorted(os.listdir(tmp)):
            if fname == "snippet.py":
                continue
            ext = fname.rsplit(".", 1)[-1].lower()
            if ext in ("png", "jpg", "jpeg", "webp"):
                fpath = os.path.join(tmp, fname)
                if os.path.getsize(fpath) > 8 * 1024 * 1024:
                    continue
                with open(fpath, "rb") as fp:
                    img_bytes = fp.read()
                mime = f"image/{'jpeg' if ext == 'jpg' else ext}"
                art = save_artifact(ctx, "image", mime, img_bytes,
                                    source="code_execute", filename=fname)
                ctx["artifacts"].append(art)
                artifact_summary = f" Image saved: {fname} → {art['id']}"
                break
        return {
            "ok": ok, "stdout": stdout, "stderr": stderr,
            "exit_code": proc.returncode,
            "summary": (f"exit={proc.returncode}; stdout {len(stdout)} chars" + artifact_summary),
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def tool_tts(args: dict, ctx: dict) -> dict:
    """Forward TTS request to the local VibeVoice worker.
    Returns an audio artifact ID, just like image tools.
    """
    text = (args.get("text") or "").strip()
    if not text:
        return {"error": "empty text"}
    voice = args.get("voice") or "Emma"
    try:
        r = requests.post(f"{VIBEVOICE_WORKER}/tts",
                          json={"text": text[:4000], "voice": voice},
                          timeout=180)
    except requests.RequestException as e:
        return {"error": (f"VibeVoice worker unreachable at {VIBEVOICE_WORKER}: "
                          f"{type(e).__name__}: {str(e)[:80]}. "
                          f"Start it with: python vibevoice_worker.py")}
    if not r.ok:
        return {"error": f"worker HTTP {r.status_code}: {r.text[:200]}"}
    audio_bytes = r.content
    if not audio_bytes:
        return {"error": "worker returned empty audio"}
    art = save_artifact(ctx, "audio", "audio/wav", audio_bytes,
                        voice=voice, text_preview=text[:80])
    ctx["artifacts"].append(art)
    return {
        "ok": True, "artifact_id": art["id"], "voice": voice,
        "summary": (f"Audio synthesized inline (id={art['id']}, voice={voice}, "
                    f"{len(audio_bytes)} bytes). The user can play it directly."),
    }


TOOL_REGISTRY = {
    "image_generate": {
        "schema": {
            "type": "function",
            "function": {
                "name": "image_generate",
                "description": (
                    "Generate an image from a text prompt using NVIDIA's hosted Visual GenAI "
                    "endpoints (FLUX.1, Stable Diffusion XL, SDXL Turbo). "
                    "Use ONLY when the user explicitly asks to generate, draw, create, paint, "
                    "render, or illustrate something visual. Do not call proactively. "
                    "Greetings, smalltalk, or generic conversational input (e.g. 'hi', "
                    "'thanks', 'how are you', or any short greeting in any language) "
                    "are NEVER image requests — respond in plain text instead. "
                    "When you do call this tool, ALWAYS pass a vivid, specific English prompt "
                    "— describe subject, style, composition, lighting, palette concretely. "
                    "GROUNDING RULE: If the user's request is abstract, emotional, or relies on "
                    "context (e.g. 'draw something for me', 'make a picture matching this poem', "
                    "'a matching image please' in any language), you MUST extract concrete imagery from the surrounding "
                    "conversation — the prose you just wrote, the topic the user is discussing, "
                    "or their stated mood — and translate that into specific visual elements "
                    "(setting, objects, color, lighting, mood). Never fall back to a generic "
                    "stock-photo subject (smiling person, mountain, coffee cup) unrelated to the "
                    "actual context. "
                    "The user will see the image immediately; do not describe its visual content "
                    "back at them — just briefly acknowledge."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "Detailed image description in English."},
                        "model":  {"type": "string", "enum": list(IMG_MODEL_MAP.keys()),
                                   "description": "flux.1-schnell (default, fastest, 4 steps) | flux.1-dev (higher quality, 50 steps) | sdxl-turbo (very fast SDXL, 2 steps) | sdxl (SDXL base) | sd3-medium (Stable Diffusion 3 Medium)."},
                        "steps":  {"type": "integer", "description": "Diffusion steps. Defaults are model-specific. More steps = slower + usually better."},
                        "seed":   {"type": "integer", "description": "Random seed; same seed + prompt → same image. Default 0."},
                        "width":  {"type": "integer", "enum": [1024, 768, 1344, 896, 1152, 832, 1216],
                                   "description": "Image width in pixels. Default 1024. Flux supports 1024x1024 (square), 1344x768 / 1216x832 / 1152x896 (landscape), 768x1344 / 832x1216 / 896x1152 (portrait)."},
                        "height": {"type": "integer", "enum": [1024, 768, 1344, 896, 1152, 832, 1216],
                                   "description": "Image height in pixels. Default 1024. See width for supported aspect ratio pairs."},
                    },
                    "required": ["prompt"],
                },
            },
        },
        "executor": tool_image_generate,
    },
    "image_edit": {
        "schema": {
            "type": "function",
            "function": {
                "name": "image_edit",
                "description": (
                    "Edit images that the user has uploaded, using FLUX.1 Kontext on the NVIDIA "
                    "catalog. Use when the user asks to modify, restyle, or transform an image "
                    "they attached (e.g., 'replace the sky', 'make this watercolor', 'add a red "
                    "hat'). Selects which uploaded image to edit by zero-based index."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt":        {"type": "string", "description": "Edit instruction in English."},
                        "image_indices": {"type": "array", "items": {"type": "integer"},
                                          "description": "Zero-based indices into user-uploaded images. [0] = first image."},
                        "steps":         {"type": "integer", "description": "Diffusion steps (default 30, max 100)."},
                        "seed":          {"type": "integer", "description": "Random seed (default 0)."},
                    },
                    "required": ["prompt"],
                },
            },
        },
        "executor": tool_image_edit,
    },
    "embed_similarity": {
        "schema": {
            "type": "function",
            "function": {
                "name": "embed_similarity",
                "description": (
                    "Compute pairwise cosine similarity between texts using a NVIDIA embedding NIM. "
                    "Use when the user asks which texts are most semantically similar, for "
                    "deduplication, paraphrase quality evaluation, clustering, or comparing concepts. "
                    "Returns a similarity matrix and the top-ranked pairs."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "texts":      {"type": "array", "items": {"type": "string"}, "minItems": 2,
                                       "description": "Two or more texts to compare."},
                        "model":      {"type": "string", "enum": list(EMBED_MODEL_ALIASES.keys()),
                                       "description": "e5 (default, balanced) | llama-1b | mistral-7b (largest)."},
                        "input_type": {"type": "string", "enum": ["query", "passage"],
                                       "description": "passage (default) for documents; query for search-query strings."},
                    },
                    "required": ["texts"],
                },
            },
        },
        "executor": tool_embed_similarity,
    },
    "math_eval": {
        "schema": {
            "type": "function",
            "function": {
                "name": "math_eval",
                "description": (
                    "Symbolic math via SymPy. Use whenever the user asks for an exact derivative, "
                    "integral, simplification, factorization, equation solution, or numerical "
                    "evaluation of an expression. ALWAYS prefer this over hand-deriving — it is "
                    "exact and avoids hallucination."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string",
                                       "description": "SymPy-parsable expression. Use ** for powers, * for multiplication. Ex: 'x**2 + 3*x - 4', 'sin(x)/x', 'exp(I*pi) + 1'."},
                        "operation":  {"type": "string",
                                       "enum": ["simplify", "expand", "factor", "differentiate", "integrate", "solve", "evaluate"],
                                       "description": "What to do. simplify (default) reduces to canonical; differentiate/integrate/solve use 'variable'; evaluate computes numeric."},
                        "variable":   {"type": "string", "description": "Variable for differentiate/integrate/solve. Default 'x'."},
                    },
                    "required": ["expression"],
                },
            },
        },
        "executor": tool_math_eval,
    },
    "rerank": {
        "schema": {
            "type": "function",
            "function": {
                "name": "rerank",
                "description": (
                    "Re-rank a list of text passages by relevance to a query, using NVIDIA's "
                    "Llama-3.2 reranker. Use when the user has multiple candidate passages, "
                    "search results, or document chunks and asks which is most relevant to a "
                    "specific question. Returns passages sorted by relevance score (logit)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query":    {"type": "string", "description": "The question or query passages should be ranked against."},
                        "passages": {"type": "array", "items": {"type": "string"}, "minItems": 2,
                                     "description": "Passages to rerank. Each up to 8000 chars."},
                        "top_n":    {"type": "integer", "description": "How many top passages to return. Default = all."},
                    },
                    "required": ["query", "passages"],
                },
            },
        },
        "executor": tool_rerank,
    },
    "ocr": {
        "schema": {
            "type": "function",
            "function": {
                "name": "ocr",
                "description": (
                    "Extract printed or handwritten text from user-uploaded images using "
                    "NVIDIA NemoRetriever OCR. Use when the user uploads a screenshot, photo "
                    "of a document, scanned page, or any image where text needs to be read. "
                    "Returns the extracted text per image. NOT for general image description — "
                    "use the vision-capable chat model itself for that."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_indices": {"type": "array", "items": {"type": "integer"},
                                          "description": "Zero-based indices of user-uploaded images to OCR."},
                        "merge_level":   {"type": "string", "enum": ["word", "line", "paragraph"],
                                          "description": "Granularity of text grouping (default: word)."},
                    },
                    "required": ["image_indices"],
                },
            },
        },
        "executor": tool_ocr,
    },
    "molmim_generate": {
        "schema": {
            "type": "function",
            "function": {
                "name": "molmim_generate",
                "description": (
                    "Generate novel small molecules around a seed SMILES string using NVIDIA "
                    "MolMIM (drug discovery). Use when the user asks for molecule generation, "
                    "drug-like analogs, or property-optimized SMILES variants. Optionally "
                    "optimizes for QED (drug-likeness) or plogP (lipophilicity)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "smiles":        {"type": "string", "description": "Seed molecule in canonical SMILES (e.g., 'CC(=O)Oc1ccccc1C(=O)O' for aspirin)."},
                        "num_molecules": {"type": "integer", "description": "How many novel molecules to generate (1-30, default 5)."},
                        "algorithm":     {"type": "string", "enum": ["CMA-ES", "none"],
                                          "description": "CMA-ES for property-guided optimization, 'none' for unguided sampling."},
                        "property":      {"type": "string", "enum": ["QED", "plogP"],
                                          "description": "Property to maximize. QED = drug-likeness, plogP = lipophilicity."},
                    },
                    "required": ["smiles"],
                },
            },
        },
        "executor": tool_molmim_generate,
    },
    "github_fetch_file": {
        "schema": {
            "type": "function",
            "function": {
                "name": "github_fetch_file",
                "description": (
                    "Fetch the text content of a single file from a public GitHub "
                    "repository. Use when the user asks you to inspect, review, "
                    "summarize, or quote from a specific file in a public repo "
                    "(README, source file, config). Do NOT try to clone repos via "
                    "code_execute — the sandbox has no network. 256 KiB cap per "
                    "file; UTF-8 only (binary files return an error). "
                    "NOTE: a single file at the cap can be ~65k tokens — that "
                    "may dominate or exceed the active model's context. For repos "
                    "you don't already know, call github_list_files first to see "
                    "structure and sizes, then fetch only the files that matter."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repo": {"type": "string",
                                 "description": "Repo as 'owner/name', e.g. 'been-seo/NIMINI'."},
                        "path": {"type": "string",
                                 "description": "File path within the repo, e.g. 'README.md' or 'src/main.py'."},
                        "ref":  {"type": "string",
                                 "description": "Branch, tag, or commit SHA. Default 'HEAD' (default branch)."},
                    },
                    "required": ["repo", "path"],
                },
            },
        },
        "executor": tool_github_fetch_file,
    },
    "github_list_files": {
        "schema": {
            "type": "function",
            "function": {
                "name": "github_list_files",
                "description": (
                    "List the files and directories at a path in a public GitHub "
                    "repository. Pair with github_fetch_file to navigate a repo "
                    "the user asks you to inspect. Empty path lists the repo root. "
                    "Subject to GitHub's 60-request-per-hour rate limit on "
                    "unauthenticated traffic from this server's IP."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repo": {"type": "string",
                                 "description": "Repo as 'owner/name', e.g. 'been-seo/NIMINI'."},
                        "path": {"type": "string",
                                 "description": "Directory path. Empty for repo root."},
                        "ref":  {"type": "string",
                                 "description": "Branch, tag, or commit SHA. Default branch if omitted."},
                    },
                    "required": ["repo"],
                },
            },
        },
        "executor": tool_github_list_files,
    },
    "web_search": {
        "schema": {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": (
                    "Search the web (DuckDuckGo) for current information. Use whenever the user "
                    "asks about news, current events, recent products, ongoing situations, or "
                    "anything where freshness matters. Returns titles, URLs, and snippets. "
                    "Cite sources when answering based on results."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query":       {"type": "string", "description": "Search query. Be specific."},
                        "max_results": {"type": "integer", "description": "Max results (1-10, default 5)."},
                    },
                    "required": ["query"],
                },
            },
        },
        "executor": tool_web_search,
    },
    "code_execute": {
        "schema": {
            "type": "function",
            "function": {
                "name": "code_execute",
                "description": (
                    "Execute a Python snippet in a sandboxed subprocess (1GiB RAM, configurable "
                    "timeout). Use for: numerical calculation, data manipulation, plotting "
                    "(matplotlib, save .png in cwd → returned as artifact), CSV/JSON parsing, "
                    "any deterministic computation. AVAILABLE: numpy, scipy, pandas, matplotlib, "
                    "sympy, math, statistics, datetime, json, re. NO network, NO subprocess. "
                    "Print results with print()."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code":    {"type": "string", "description": "Self-contained Python code. Up to 10000 chars."},
                        "timeout": {"type": "integer", "description": "Max runtime in seconds (1-60, default 10)."},
                    },
                    "required": ["code"],
                },
            },
        },
        "executor": tool_code_execute,
    },
}


def list_tools_meta() -> list:
    """Return tool metadata (without executors) for the /v1/tools endpoint."""
    return [{
        "name": name,
        "description": t["schema"]["function"]["description"],
        "parameters": t["schema"]["function"]["parameters"],
    } for name, t in TOOL_REGISTRY.items()]


# ============================ MAIN ============================

def discover_lan_ips() -> list:
    """Best-effort: enumerate non-loopback IPv4 addresses on this host.
    Used only to print friendly URLs at startup."""
    import socket
    ips = []
    try:
        # Trick: connect to a public IP without sending; OS picks our outbound IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            ips.append(s.getsockname()[0])
        finally:
            s.close()
    except OSError:
        pass
    try:
        host = socket.gethostname()
        for info in socket.getaddrinfo(host, None, socket.AF_INET):
            ip = info[4][0]
            if ip and ip not in ips and not ip.startswith("127."):
                ips.append(ip)
    except OSError:
        pass
    return ips


def main():
    global REGISTRATION_TOKEN, ALLOWED_ORIGINS
    p = argparse.ArgumentParser(
        description="Local NIM chat proxy & web UI (multi-user).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python serve.py                                       # localhost only, open registration\n"
            "  python serve.py --host 0.0.0.0                        # LAN; auto-generated registration token\n"
            "  python serve.py --host 0.0.0.0 --registration-token X # LAN; explicit registration token\n"
            "  python serve.py --host 0.0.0.0 --open-registration    # LAN, no token gate (CAUTION)\n"
            "\n"
            "Auth model:\n"
            "  All endpoints except /health, /v1/auth/register, /v1/auth/login require a\n"
            "  valid session cookie (obtained via login). Registration may be gated by a\n"
            "  shared registration token to prevent random external sign-ups.\n"
        ),
    )
    p.add_argument("port", nargs="?", type=int, default=8765,
                   help="Port to listen on (default: 8765)")
    p.add_argument("--host", default="127.0.0.1",
                   help="Bind address. '127.0.0.1' = localhost only (default), "
                        "'0.0.0.0' = all interfaces.")
    p.add_argument("--registration-token", default=None,
                   help="Token required as X-Registration-Token header (or "
                        "registration_token in JSON body) on /v1/auth/register. "
                        "Auto-generated if --host is non-loopback and not specified. "
                        "Use --open-registration to disable the gate entirely.")
    p.add_argument("--open-registration", action="store_true",
                   help="Disable the registration token gate. Anyone reaching the "
                        "server can create an account. Use only on trusted networks.")
    p.add_argument("--allowed-origins", default=None,
                   help="Comma-separated list of CORS Origins to allow. "
                        "Defaults to https://localhost + capacitor/ionic local "
                        "schemes. Use '*' to allow any origin (insecure).")
    # Deprecated alias for backward compatibility
    p.add_argument("--token", default=None,
                   help=argparse.SUPPRESS)  # legacy alias of --registration-token
    p.add_argument("--no-auth", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.no_auth:
        print("[serve] WARNING: --no-auth is deprecated; auth is now per-user "
              "and cannot be disabled globally. Use --open-registration if you "
              "want anyone to be able to register.", file=sys.stderr)

    is_loopback = args.host in ("127.0.0.1", "localhost", "::1", "")

    # CORS allowlist: CLI arg > env var > built-in defaults.
    explicit_origins = args.allowed_origins or os.environ.get(
        "NIMINI_ALLOWED_ORIGINS", "")
    if explicit_origins:
        ALLOWED_ORIGINS = [o.strip() for o in explicit_origins.split(",") if o.strip()]
    print(f"[serve] CORS allowed origins: {ALLOWED_ORIGINS}", file=sys.stderr)

    # Resolve REGISTRATION_TOKEN per policy
    explicit = args.registration_token or args.token
    if args.open_registration:
        REGISTRATION_TOKEN = ""
    elif explicit:
        REGISTRATION_TOKEN = explicit
    elif is_loopback:
        # Loopback bind — open registration is fine by default
        REGISTRATION_TOKEN = ""
    else:
        # External bind — auto-generate to prevent random sign-ups
        REGISTRATION_TOKEN = secrets.token_urlsafe(24)

    addr = (args.host, args.port)
    httpd = ThreadingHTTPServer(addr, Handler)

    # Pretty banner
    print(f"[serve] http://{args.host}:{args.port}/   (Ctrl+C to stop)")
    n_users = _user_count()
    print(f"[serve] {n_users} registered user(s)")
    if REGISTRATION_TOKEN:
        # Print to stderr (not stdout) so the token doesn't sit in tee'd log
        # files. systemd-journald STILL captures stderr unless the unit is
        # configured otherwise — print a scrub warning so the operator knows.
        print("", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print(f"  REGISTRATION TOKEN: {REGISTRATION_TOKEN}", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print(f"  Required for POST /v1/auth/register", file=sys.stderr)
        print(f"  Header: X-Registration-Token: <token>", file=sys.stderr)
        print(f"  WARNING: this token will appear in journald / docker logs / any", file=sys.stderr)
        print(f"           captured stderr. After all expected users have signed up,", file=sys.stderr)
        print(f"           rotate or unset it and scrub the line from logs.", file=sys.stderr)
        if not is_loopback:
            for ip in discover_lan_ips() or [args.host]:
                print(f"  Web UI: http://{ip}:{args.port}/", file=sys.stderr)
        print("", file=sys.stderr)
    else:
        print("[serve] Registration is OPEN (no token required).")
        if not is_loopback:
            print("[serve] WARNING: open registration on a non-loopback bind. "
                  "Anyone reaching this server can create an account.")

    # Background: purge expired sessions on startup
    try:
        n = _session_purge_expired()
        if n:
            print(f"[serve] purged {n} expired session(s)")
    except Exception:
        pass

    # claw US-007: kick off health probe + periodic re-probe
    probe_all_async()
    threading.Thread(target=health_loop, daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] bye")


if __name__ == "__main__":
    main()
