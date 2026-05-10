#!/usr/bin/env python3
"""VibeVoice-Realtime-0.5B local worker.

Standalone HTTP server that loads Microsoft's VibeVoice-Realtime-0.5B once into
memory and exposes:

    GET  /health                → {"ok": true, "ready": ..., "device": ...}
    GET  /voices                → {"voices": [{"name", "language", "description"}, ...]}
    POST /tts {text, voice}     → audio/wav bytes

This worker is referenced by serve.py via VIBEVOICE_WORKER env var (default
http://127.0.0.1:8766). Model loading takes ~30s on first run; the server reports
ready=false during that period.

Usage
-----
    pip install transformers torch accelerate vibevoice
    python vibevoice_worker.py            # listens on 0.0.0.0:8766
    python vibevoice_worker.py --port 8888 --device cuda

Korean voice
------------
The Realtime-0.5B model includes experimental multilingual voices in
DE / FR / IT / JP / KR / NL / PL / PT / ES alongside the English presets.
The KR voice file is downloaded automatically from the community fork on
first launch (see fetch_multilingual_voices()).

Hardware
--------
NVIDIA T4 / Apple M4 Pro = realtime (~300ms first-audio latency).
Weaker GPUs / CPU-only = slower than realtime but still usable.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import threading
import traceback
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Force PyTorch-only mode in transformers BEFORE it gets imported anywhere.
# transformers tries to auto-detect tensorflow / flax / jax and import them;
# if any of those packages is broken (e.g., NumPy ABI mismatch on Windows), the
# whole import chain explodes. We don't need them — VibeVoice is PyTorch-only.
os.environ.setdefault("USE_TF",   "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("USE_JAX",  "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# Lazy imports — done in load_model() so --help works without torch installed.
torch = None
np = None
_model = None
_processor = None
_tokenizer = None  # Required by VibeVoice generate() for CFG negative prompt
_printed_input_keys = False  # one-time diagnostic flag
_voice_cache: dict[str, "torch.Tensor"] = {}
_load_lock = threading.Lock()
_load_state = {"ready": False, "error": None, "device": "cpu", "model_path": ""}

# Voice presets are NOT bundled in the model repo — they live in .tar.gz archives
# linked from demo/download_experimental_voices.sh. The voice catalog is built
# at startup by downloading those archives once and scanning the resulting files.
# Filename pattern: {lang}-{Name}_{gender}.pt   (e.g., kr-Spk2_woman.pt)
ENGLISH_VOICES: list = []
ALL_VOICES: list = []
VALID_VOICE_NAMES: set = set()
DEFAULT_MODEL  = "microsoft/VibeVoice-Realtime-0.5B"
DEFAULT_PORT   = 8766
SAMPLE_RATE_HZ = 24000   # VibeVoice acoustic decoder native rate

# URL of the upstream shell script. We parse it once at startup to find the
# exact .tar.gz archive URLs Microsoft ships for multilingual voices.
VOICE_DISCOVERY_URLS = [
    "https://raw.githubusercontent.com/microsoft/VibeVoice/main/demo/download_experimental_voices.sh",
    "https://raw.githubusercontent.com/vibevoice-community/VibeVoice/main/demo/download_experimental_voices.sh",
]


def _fetch_url_bytes(url: str, timeout: int = 30) -> bytes:
    """Fetch a URL and return its bytes. Raises on failure."""
    from urllib.request import Request, urlopen
    req = Request(url, headers={"User-Agent": "vibevoice-worker/0.2"})
    with urlopen(req, timeout=timeout) as r:
        if r.status != 200:
            raise IOError(f"HTTP {r.status}")
        return r.read()


def discover_archive_urls() -> list:
    """Fetch download_experimental_voices.sh and parse out the archive URLs.
    Returns a list of (filename, url) tuples. Empty on network failure."""
    text = ""
    for url in VOICE_DISCOVERY_URLS:
        try:
            text = _fetch_url_bytes(url, timeout=15).decode("utf-8", errors="replace")
            break
        except (OSError, ValueError):
            continue
    if not text:
        return []
    # Each line in the FILES=( ... ) block looks like:
    #   "experimental_voices_kr.tar.gz|https://github.com/.../...tar.gz"
    pat = re.compile(r'"([\w.\-]+\.tar\.gz)\|(https?://[^"]+)"')
    return [(m.group(1), m.group(2)) for m in pat.finditer(text)]


def fetch_and_extract_voices() -> int:
    """Download every multilingual archive listed in the upstream script and
    extract them under voices/streaming_model/experimental_voices/. Returns
    the number of new .pt voice files discovered (0 on any failure)."""
    import tarfile

    archives = discover_archive_urls()
    if not archives:
        print("[vibevoice] could not reach upstream voice manifest; "
              "multilingual voices unavailable until a voices/ folder is "
              "placed manually.", flush=True)
        return 0

    target = _voices_dir() / "experimental_voices"
    target.mkdir(parents=True, exist_ok=True)
    marker = target / ".downloaded"
    if marker.is_file():
        # Already extracted previously — just count
        return sum(1 for _ in target.glob("*.pt"))

    print(f"[vibevoice] downloading {len(archives)} multilingual voice "
          "archives (~22MB total)...", flush=True)
    new_count = 0
    for fname, url in archives:
        try:
            data = _fetch_url_bytes(url, timeout=60)
        except (OSError, ValueError) as e:
            print(f"[vibevoice]   {fname} → {type(e).__name__}: {e}", flush=True)
            continue
        try:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
                for member in tf.getmembers():
                    if not member.isfile():
                        continue
                    if not member.name.endswith(".pt"):
                        continue
                    # Strip any leading directory components — flatten into target/
                    base = Path(member.name).name
                    out = target / base
                    extracted = tf.extractfile(member)
                    if extracted is None:
                        continue
                    out.write_bytes(extracted.read())
                    new_count += 1
            print(f"[vibevoice]   {fname} OK", flush=True)
        except (tarfile.TarError, OSError) as e:
            print(f"[vibevoice]   {fname} extract failed: {e}", flush=True)
    marker.write_text("ok")
    print(f"[vibevoice] extracted {new_count} multilingual voice files to "
          f"{target}", flush=True)
    return new_count


def scan_local_voices() -> list:
    """Build the voice catalog from files actually present on disk.
    Looks in voices/streaming_model/ and voices/streaming_model/experimental_voices/."""
    catalog = []
    seen = set()

    def lang_from_name(name: str) -> tuple:
        """Heuristic: extract language and gender hints from a voice filename."""
        # examples: ko-Soyeon_woman → ko, female
        #           Carter             → en, unknown
        #           kr_woman_v01       → ko, female
        m = re.match(r"^([a-z]{2})[-_]", name)
        lang = "en"
        if m:
            head = m.group(1).lower()
            mapping = {"kr": "ko", "sp": "es", "jp": "ja", "ch": "zh"}
            lang = mapping.get(head, head)
        gender = ""
        low = name.lower()
        if "_woman" in low or "_female" in low:
            gender = "female"
        elif "_man" in low or "_male" in low:
            gender = "male"
        return lang, gender

    for d in [_voices_dir(), _voices_dir() / "experimental_voices"]:
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.pt")):
            name = p.stem
            if name in seen:
                continue
            seen.add(name)
            # Pre-bundled English voices (Carter etc.) keep their static metadata
            eng = next((v for v in ENGLISH_VOICES if v["name"] == name), None)
            if eng:
                catalog.append(dict(eng))
                continue
            lang, gender = lang_from_name(name)
            desc = f"Experimental {lang.upper()}"
            if gender:
                desc += f" {gender}"
            catalog.append({"name": name, "language": lang, "description": desc})
    return catalog


# ----------------------------- model loading -----------------------------

def detect_device(prefer: str | None = None) -> str:
    """Return best available torch device string."""
    global torch
    if prefer in ("cpu", "cuda", "mps"):
        return prefer
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_path: str, device: str | None = None) -> None:
    """Load VibeVoice into _model. Heavy — call in background thread."""
    global torch, np, _model, _processor, _tokenizer
    with _load_lock:
        if _load_state["ready"]:
            return
        try:
            print(f"[vibevoice] loading deps...", flush=True)
            import torch as _torch
            import numpy as _np
            torch = _torch
            np = _np

            dev = detect_device(device)
            _load_state["device"] = dev
            _load_state["model_path"] = model_path
            print(f"[vibevoice] device={dev}, model={model_path}", flush=True)

            # Try the community/microsoft library layout first, then HF transformers fallback.
            # IMPORTANT: streaming inference REQUIRES VibeVoiceStreamingProcessor.
            # The plain VibeVoiceProcessor is for long-form/multi-speaker models and
            # does NOT produce tts_text_ids — passing its output to the streaming
            # generate() leads to AttributeError: 'NoneType' object has no attribute 'to'
            # at modeling_vibevoice_streaming_inference.py:473.
            try:
                from vibevoice.modular.modeling_vibevoice_streaming_inference import (
                    VibeVoiceStreamingForConditionalGenerationInference as VVModel,
                )
                from vibevoice.processor.vibevoice_streaming_processor import (
                    VibeVoiceStreamingProcessor as VVProcessor,
                )
                _vv_lib = "vibevoice"
            except ImportError as e:
                # Fallback: the model card lists library_name: transformers, so try AutoModel.
                # This works after `pip install transformers>=5.3` (per VibeVoice 2026-03 release).
                from transformers import AutoModel, AutoProcessor
                VVModel = AutoModel
                VVProcessor = AutoProcessor
                _vv_lib = "transformers"
                print(f"[vibevoice] vibevoice lib not found ({e}); using transformers AutoModel", flush=True)

            print(f"[vibevoice] loading processor ({_vv_lib})...", flush=True)
            _processor = VVProcessor.from_pretrained(model_path, trust_remote_code=True)
            print(f"[vibevoice] loading model ({_vv_lib})...", flush=True)
            # Always load in fp32. The voice prompts in the .pt files are saved
            # at fp32, and the official VibeVoice examples don't cast either side.
            # Loading the model at fp16 to "save VRAM" creates an attention dtype
            # mismatch with the cached prompt that we'd have to paper over with
            # fragile recursive casting through dict subclasses. 0.5B model in
            # fp32 is ~2GB VRAM — fine for any GPU running this in the first place.
            dtype = torch.float32
            _model = VVModel.from_pretrained(
                model_path,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            _model.to(dev)
            _model.eval()

            # VibeVoice's generate() takes a `tokenizer=` keyword for CFG negative
            # prompt construction (line ~467 of modeling_vibevoice_streaming_inference.py
            # calls tokenizer.convert_tokens_to_ids("<|image_pad|>")). Resolve and
            # cache it now so synthesize_to_wav can pass it through every call.
            _tokenizer = getattr(_processor, "tokenizer", None)
            if _tokenizer is None:
                # Some processors don't expose .tokenizer — load it directly.
                try:
                    from transformers import AutoTokenizer
                    _tokenizer = AutoTokenizer.from_pretrained(
                        model_path, trust_remote_code=True,
                    )
                    print(f"[vibevoice] tokenizer loaded separately via AutoTokenizer",
                          flush=True)
                except Exception as e:
                    print(f"[vibevoice] WARNING: tokenizer load failed: {e}. "
                          "Synthesis will likely fail with AttributeError on "
                          "convert_tokens_to_ids.", flush=True)

            # Also attach to model for libraries that read self.tokenizer instead of
            # taking a kwarg (older builds, AutoModel path).
            try:
                if _tokenizer is not None and getattr(_model, "tokenizer", None) is None:
                    _model.tokenizer = _tokenizer
                    print(f"[vibevoice] tokenizer also attached to model.tokenizer",
                          flush=True)
            except Exception as e:
                print(f"[vibevoice] tokenizer attach warning: {e}", flush=True)

            _load_state["ready"] = True
            print(f"[vibevoice] ready (lib={_vv_lib}, dtype={dtype})", flush=True)

            # Auto-fetch & extract every voice archive listed upstream
            global ALL_VOICES, VALID_VOICE_NAMES
            try:
                fetch_and_extract_voices()
                ALL_VOICES = scan_local_voices()
                VALID_VOICE_NAMES = {v["name"] for v in ALL_VOICES}
                if ALL_VOICES:
                    by_lang = {}
                    for v in ALL_VOICES:
                        by_lang.setdefault(v["language"], []).append(v["name"])
                    summary = ", ".join(f"{k}:{len(v)}" for k, v in sorted(by_lang.items()))
                    print(f"[vibevoice] {len(ALL_VOICES)} voices loaded ({summary})", flush=True)
                else:
                    print("[vibevoice] WARNING: no voices found. /tts will fail until "
                          "voice .pt files are placed in voices/streaming_model/.",
                          flush=True)
            except Exception as e:
                print(f"[vibevoice] voice setup warning: {e}", flush=True)
        except Exception as e:
            _load_state["error"] = f"{type(e).__name__}: {e}"
            traceback.print_exc()
            print(f"[vibevoice] LOAD FAILED: {_load_state['error']}", flush=True)


# ----------------------------- voice presets -----------------------------

def _voices_dir() -> Path:
    """Local cache dir for voice .pt files (next to this script)."""
    d = Path(__file__).resolve().parent / "voices" / "streaming_model"
    d.mkdir(parents=True, exist_ok=True)
    return d


def voice_preset_path(model_path: str, voice_name: str) -> Path | None:
    """Locate the .pt file for a voice preset on disk.
    Voices are populated by fetch_and_extract_voices() at startup.
    """
    # Files are flattened into voices/streaming_model/ (archive structure stripped)
    cache = _voices_dir() / f"{voice_name}.pt"
    if cache.is_file() and cache.stat().st_size > 1024:
        return cache
    # Also accept hand-placed files
    here = Path(__file__).resolve().parent
    for cand in [
        here / "voices" / f"{voice_name}.pt",
        _voices_dir() / "experimental_voices" / f"{voice_name}.pt",
    ]:
        if cand.is_file() and cand.stat().st_size > 1024:
            return cand
    return None


def load_voice(voice_name: str) -> "torch.Tensor | None":
    """Load (and cache) the voice preset KV tensor for `voice_name`."""
    if voice_name in _voice_cache:
        return _voice_cache[voice_name]
    p = voice_preset_path(_load_state["model_path"], voice_name)
    if p is None:
        return None
    try:
        # weights_only=False is required because voice presets contain
        # transformers.modeling_outputs.BaseModelOutputWithPast objects which
        # PyTorch 2.6+ refuses to unpickle by default. The presets ship with
        # the model checkpoint we control — not arbitrary external input —
        # so the arbitrary-code-execution risk does not apply here.
        kv = torch.load(p, map_location=_load_state["device"], weights_only=False)
    except Exception as e:
        print(f"[vibevoice] failed to load voice {voice_name}: {e}", flush=True)
        return None
    _voice_cache[voice_name] = kv
    return kv


# ----------------------------- voice resolution -----------------------------

def _detect_text_language(text: str) -> str:
    """Cheap heuristic: detect the dominant script of `text` and return a 2-letter
    language code. Used for picking a sensible voice when the caller asks for
    one we don't have."""
    if not text:
        return "en"
    counts = {"ko": 0, "ja": 0, "zh": 0, "en": 0}
    for ch in text:
        cp = ord(ch)
        # Hangul (Jamo + Syllables + Compatibility Jamo)
        if 0xAC00 <= cp <= 0xD7A3 or 0x1100 <= cp <= 0x11FF or 0x3130 <= cp <= 0x318F:
            counts["ko"] += 1
        # Hiragana + Katakana (Japanese-only scripts)
        elif 0x3040 <= cp <= 0x309F or 0x30A0 <= cp <= 0x30FF:
            counts["ja"] += 1
        # CJK Unified Ideographs (Chinese / shared with Japanese kanji)
        elif 0x4E00 <= cp <= 0x9FFF:
            counts["zh"] += 1
        elif ch.isalpha():
            counts["en"] += 1
    if counts["ko"] > 0:
        return "ko"  # Hangul is unambiguous
    if counts["ja"] > 0:
        return "ja"  # Kana is unambiguous
    if counts["zh"] > counts["en"]:
        return "zh"
    return "en"


def resolve_voice(requested: str | None, text: str) -> tuple[str, str]:
    """Map a requested voice name to one we actually have on disk.

    Returns (resolved_name, reason). Reason is one of:
      - "exact"     : caller's name matched a known voice
      - "language"  : caller's name unknown, picked a voice in the text's language
      - "default"   : no language match either, fell back to first available
    Raises RuntimeError if no voices are loaded at all.
    """
    if not ALL_VOICES:
        raise RuntimeError("no voices loaded; check worker startup logs")

    if requested and requested in VALID_VOICE_NAMES:
        return requested, "exact"

    target_lang = _detect_text_language(text)
    # Prefer female voices for default, but accept any matching language
    in_lang = [v for v in ALL_VOICES if v["language"] == target_lang]
    if in_lang:
        # Stable preference: female-marked first, then alphabetical
        in_lang.sort(key=lambda v: (
            "_woman" not in v["name"].lower() and "_female" not in v["name"].lower(),
            v["name"],
        ))
        return in_lang[0]["name"], "language"

    # Last resort: first English voice, or just first voice
    eng = [v for v in ALL_VOICES if v["language"] == "en"]
    pick = (eng or ALL_VOICES)[0]
    return pick["name"], "default"


def synthesize_to_wav(text: str, voice: str,
                      cfg_scale: float = 1.5,
                      ddpm_steps: int = 5) -> bytes:
    """Run VibeVoice inference and return a .wav blob (24kHz mono int16)."""
    if not _load_state["ready"]:
        raise RuntimeError(f"model not ready: {_load_state.get('error') or 'still loading'}")

    resolved, reason = resolve_voice(voice, text)
    if reason != "exact":
        print(f"[vibevoice] voice {voice!r} → {resolved} ({reason})", flush=True)
    voice = resolved

    voice_kv = load_voice(voice)
    if voice_kv is None:
        raise RuntimeError(
            f"voice preset {voice!r} listed in catalog but .pt file missing on disk. "
            "Try deleting the voices/ folder and restarting the worker to re-download."
        )

    # Microsoft VibeVoice-Realtime documented invocation pattern:
    #   inputs = processor(text=...)
    #   model.generate(**inputs, all_prefilled_outputs=deepcopy(voice_prompt), ...)
    # Source: https://www.mintlify.com/egarciaf2/VibeVoice/guides/custom-voices
    #
    # Crucial details we got wrong before:
    #  - `text=text` direct kwarg leaves tts_text_ids None → AttributeError at
    #    `tts_text_ids.to(self.device)`. The processor MUST tokenize first and
    #    we spread its output via `**inputs`.
    #  - voice prompt goes via `all_prefilled_outputs=` (NOT `voice_kv=`).
    #  - copy.deepcopy is required: model.generate mutates the prefilled
    #    outputs in-place, corrupting the cache for subsequent calls.
    audio_chunks = []
    try:
        # VibeVoiceStreamingProcessor.process_input_with_cached_prompt() is the
        # documented streaming entry point. Unlike the long-form VibeVoiceProcessor,
        # it accepts raw text without speaker labels — single-speaker context is
        # implied by the cached_prompt (one voice = one speaker). Adding a
        # "Speaker 1:" prefix to streaming input causes the label tokens to be
        # synthesized as audio ("speaker one") in the output.
        if not hasattr(_processor, "process_input_with_cached_prompt"):
            raise RuntimeError(
                "Loaded processor lacks process_input_with_cached_prompt. "
                f"Expected VibeVoiceStreamingProcessor, got {type(_processor).__name__}. "
                "This usually means the wrong processor class was imported in load_model()."
            )
        inputs = _processor.process_input_with_cached_prompt(
            text=text,
            cached_prompt=voice_kv,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if hasattr(inputs, "to"):
            inputs = inputs.to(_load_state["device"])

        # One-time diagnostic: log the keys produced by the processor so any
        # future shape mismatch is debuggable. Cleared after first successful call.
        global _printed_input_keys
        if not _printed_input_keys:
            try:
                keys = sorted(list(inputs.keys())) if hasattr(inputs, "keys") else "(no keys)"
                print(f"[vibevoice] processor inputs keys: {keys}", flush=True)
                _printed_input_keys = True
            except Exception:
                pass

        import copy as _copy

        # Just deepcopy — model is fp32, voice prompt is fp32, no casting needed.
        # The deepcopy is required because generate() mutates all_prefilled_outputs
        # in place, which would corrupt subsequent calls for the same voice.
        prefilled = _copy.deepcopy(voice_kv)

        gen_kwargs = dict(
            max_new_tokens=None,
            cfg_scale=cfg_scale,
            all_prefilled_outputs=prefilled,
            generation_config={"do_sample": False},
        )
        if _tokenizer is not None:
            gen_kwargs["tokenizer"] = _tokenizer

        # processor result is dict-like (BatchFeature) — spread it
        if hasattr(inputs, "keys"):
            result = _model.generate(**inputs, **gen_kwargs)
        else:
            result = _model.generate(inputs, **gen_kwargs)

        # Result shape varies: structured output (.speech_outputs / .audio /
        # .sequences), generator of chunks, or bare tensor. Handle in order.
        if hasattr(result, "speech_outputs") and result.speech_outputs is not None:
            audio_chunks.append(_to_numpy_audio(result.speech_outputs))
        elif hasattr(result, "audio") and result.audio is not None:
            audio_chunks.append(_to_numpy_audio(result.audio))
        elif hasattr(result, "__iter__") and not hasattr(result, "shape"):
            for chunk in result:
                audio_chunks.append(_to_numpy_audio(chunk))
        else:
            audio_chunks.append(_to_numpy_audio(result))
    except Exception:
        # No fallback path — let the real error propagate. The previous
        # "except TypeError -> retry with processor(text=...,voice=...)"
        # block was dead code that masked real errors with a misleading
        # second TypeError about missing kwargs.
        raise

    if not audio_chunks:
        raise RuntimeError("no audio generated")
    full = np.concatenate([a for a in audio_chunks if a.size > 0])
    return numpy_to_wav(full, SAMPLE_RATE_HZ)


def _to_numpy_audio(x) -> "np.ndarray":
    """Coerce a tensor/list/array of float audio samples in [-1, 1] to a 1-D
    float32 ndarray. Handles GPU tensors (moves to CPU) and lists of chunks
    (concatenates). VibeVoice generate() returns speech_outputs as a list
    per batch element — for batch=1 that's a single list of tensors."""
    if hasattr(x, "audio"):       # named tuple
        x = x.audio
    if hasattr(x, "waveform"):
        x = x.waveform
    # If it's a list/tuple of tensors (the typical VibeVoice case), concatenate
    # after moving each piece to CPU. Empty lists raise — generate() produced
    # no audio, which is a real error worth surfacing.
    if isinstance(x, (list, tuple)):
        if not x:
            raise RuntimeError("generate() returned empty speech_outputs list")
        pieces = []
        for piece in x:
            if hasattr(piece, "detach"):
                piece = piece.detach().to("cpu").numpy()
            pieces.append(np.asarray(piece, dtype=np.float32).reshape(-1))
        arr = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
    elif hasattr(x, "detach"):
        # Single tensor — move to CPU before numpy()
        arr = x.detach().to("cpu").numpy().astype(np.float32, copy=False)
    else:
        arr = np.asarray(x, dtype=np.float32)
    arr = arr.squeeze()
    if arr.ndim == 0:
        arr = arr.reshape((1,))
    if arr.ndim > 1:
        arr = arr.mean(axis=tuple(range(arr.ndim - 1)))  # mix down to mono
    return arr


def numpy_to_wav(audio_f32: "np.ndarray", sample_rate: int) -> bytes:
    """Encode mono float32 audio in [-1,1] to a 16-bit PCM WAV blob."""
    audio_f32 = np.clip(audio_f32, -1.0, 1.0)
    pcm16 = (audio_f32 * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16.tobytes())
    return buf.getvalue()


# ----------------------------- HTTP server -----------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "VibeVoiceWorker/0.1"

    def log_message(self, fmt, *a):
        sys.stderr.write("[vibevoice] " + (fmt % a) + "\n")

    def _send_json(self, code: int, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, code: int, body: bytes, mime: str):
        self.send_response(code)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):  # CORS preflight
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/health":
            self._send_json(200, {
                "ok": True,
                "ready": _load_state["ready"],
                "device": _load_state["device"],
                "model": _load_state["model_path"],
                "error": _load_state["error"],
            }); return
        if path == "/voices":
            # Annotate which voices are actually loadable
            available = []
            for v in ALL_VOICES:
                p = voice_preset_path(_load_state["model_path"], v["name"]) if _load_state["ready"] else None
                vv = dict(v)
                vv["loadable"] = p is not None
                available.append(vv)
            self._send_json(200, {"voices": available, "sample_rate_hz": SAMPLE_RATE_HZ})
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path != "/tts":
            self._send_json(404, {"error": "not found"}); return
        try:
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(length) if length > 0 else b""
            body = json.loads(raw.decode("utf-8")) if raw else {}
        except (ValueError, json.JSONDecodeError) as e:
            self._send_json(400, {"error": f"bad json: {e}"}); return
        if not isinstance(body, dict):
            self._send_json(400, {"error": "expected object"}); return

        text = (body.get("text") or "").strip()
        voice = body.get("voice") or "Emma"
        cfg = body.get("cfg_scale", 1.5)
        steps = body.get("ddpm_steps", 5)
        if not text:
            self._send_json(400, {"error": "empty text"}); return
        if len(text) > 4000:
            text = text[:4000]

        if not _load_state["ready"]:
            self._send_json(503, {
                "error": "model_not_ready",
                "detail": _load_state.get("error") or "still loading; try again in a moment",
            }); return
        try:
            wav = synthesize_to_wav(text, voice, cfg_scale=float(cfg), ddpm_steps=int(steps))
        except (ValueError, RuntimeError) as e:
            # Always log the traceback so silent 500s are debuggable. The progress
            # bar from generate() can swallow stderr lines if it doesn't end with
            # a newline — flush=True helps but isn't always enough on Windows.
            print(f"\n[vibevoice] synthesis failed ({type(e).__name__}): {e}",
                  file=sys.stderr, flush=True)
            traceback.print_exc()
            self._send_json(400 if isinstance(e, ValueError) else 500, {
                "error": type(e).__name__,
                "detail": str(e)[:500],
            }); return
        except Exception as e:
            print(f"\n[vibevoice] synthesis failed (unexpected): {e}",
                  file=sys.stderr, flush=True)
            traceback.print_exc()
            self._send_json(500, {"error": "synthesis_failed", "detail": str(e)[:500]}); return
        self._send_bytes(200, wav, "audio/wav")


# ----------------------------- main -----------------------------

def main():
    p = argparse.ArgumentParser(description="VibeVoice-Realtime worker.")
    p.add_argument("--model_path", default=DEFAULT_MODEL,
                   help=f"HF model id or local path (default: {DEFAULT_MODEL})")
    p.add_argument("--port", type=int, default=DEFAULT_PORT,
                   help=f"Port to listen on (default: {DEFAULT_PORT})")
    p.add_argument("--host", default="127.0.0.1",
                   help="Bind host (default: 127.0.0.1; use 0.0.0.0 to expose)")
    p.add_argument("--device", default=None, choices=[None, "cpu", "cuda", "mps"],
                   help="Force device (default: auto)")
    args = p.parse_args()

    # Kick off model load in background so HTTP can answer /health immediately
    t = threading.Thread(target=load_model, args=(args.model_path, args.device), daemon=True)
    t.start()

    addr = (args.host, args.port)
    print(f"[vibevoice] http://{args.host}:{args.port}/  (Ctrl+C to stop)", flush=True)
    print(f"[vibevoice] loading {args.model_path} in background...", flush=True)
    server = ThreadingHTTPServer(addr, Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[vibevoice] shutting down", flush=True)
        server.server_close()


if __name__ == "__main__":
    main()
