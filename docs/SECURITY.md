# Security model

This document describes what the project tries to protect against, what it
doesn't, and what to do if you find an issue.

## Threat model

This is a personal/small-team chatbot host. The realistic attackers we worry
about:

| Attacker | Capability | Concrete examples |
|---|---|---|
| Malicious chat user | Can submit any prompt, attach files, trigger any tool | Tries to read host secrets via `code_execute`, exfil chat history of other users, escalate to root |
| Network attacker | Can intercept un-TLS'd traffic | Reads session cookies, NVIDIA keys |
| Local user on host | Has shell on the box (e.g. SSH foothold) | Reads other users' DB rows, kernel LPE |

Out of scope: nation-state, supply-chain compromise of `requests`, physical
access, side-channel timing on scrypt.

## What's protected

### Per-user API keys (BYOK + envelope encryption)
- Each user supplies their own NVIDIA NIM key at registration.
- Server validates the key against NVIDIA before storing (catches typos and
  revoked keys before the user is created).
- **Keys are envelope-encrypted with a KEK derived from the user's password.**
  One scrypt call (N=2¹⁷, r=8, p=1, 64-byte output) yields:
  - first 32 bytes — stored as `password_hash`
  - last 32 bytes — used as the Fernet key, **never persisted**
  The encrypted blob lives in `users.nvidia_api_key_enc`. The KEK is held
  in an in-memory session cache for the lifetime of a login. A server
  restart wipes the cache; users get an explicit "session decryption key
  unavailable — please log in again" error rather than a confusing failure.
- Existing accounts that pre-date envelope encryption have their plaintext
  key migrated transparently on the next login (encrypt with the freshly
  derived KEK, NULL out `nvidia_api_key`, shorten `password_hash` to the
  new 32-byte form).
- DB-leak threat: an attacker with `chat.db` cannot decrypt the keys without
  brute-forcing each user's password through scrypt N=2¹⁷ (~2.5s per
  candidate on commodity hardware → days per character of weak password).
- Keys are never returned to the client. `/v1/auth/me` returns only a 4-char
  masked tail (`••••2cPY`) and a `needs_login` flag when the cache is empty.

### Per-user data isolation
- Conversations are scoped to `user_id` in every query. There is no admin path
  that exposes another user's messages.
- Tool-generated artifacts (images, audio) live under `artifacts/{id}.{ext}`
  with an `artifacts(user_id, …)` index. `GET /v1/artifact/{id}` checks
  the cookie's user matches the artifact's owner before streaming bytes.
  Cross-user requests get a 404, not a 403, to avoid id enumeration.
- Session cookies are `HttpOnly + SameSite=None + Secure`. Cross-origin
  deployments work; cookie theft via XSS is prevented (no JS access).
- `auth_check()` re-loads the user record on every request — deleting a user
  invalidates their sessions on next request without an explicit revoke step.

### Browser-side hardening
- **CSP** (HTTP header, not the often-stripped `<meta>` form):
  `default-src 'self'; script-src 'self' https://cdn.jsdelivr.net 'sha256-…';
  style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net …;
  font-src 'self' https://fonts.gstatic.com https://cdn.jsdelivr.net data:;
  img-src 'self' data: blob:; connect-src 'self'; frame-ancestors 'none';
  base-uri 'self'; form-action 'self'`. The hash for the inline `<script>`
  block is computed from the actual served bytes — JS edits don't need
  anyone to update a hash by hand.
- **SRI** on every CDN tag (`marked@13.0.3`, `dompurify@3.2.4`,
  `katex@0.16.11`). A jsDelivr supply-chain compromise becomes a load
  failure rather than silent JS injection.
- **CORS allowlist** (`ALLOWED_ORIGINS` env / `--allowed-origins` CLI; the
  default covers Capacitor / Ionic webview origins). Origin reflection
  with `Allow-Credentials: true` was the original behavior and let any
  page steal session cookies cross-origin — fixed.
- `escapeHtml()` covers attribute contexts: `&<>` plus `"`, `'`, `` ` ``,
  `=`, `/`. Filenames or LLM output that ends up in `<img alt>` /
  `<a title>` can't break out.
- `Server` / `sys_version` headers blanked so the implementation isn't
  fingerprinted to an attacker scanning for Python `BaseHTTPRequestHandler`
  CVEs.
- `X-Recovery-Steps` (a count) replaces `X-Recovery-Ledger` (a base64-JSON
  blob that could blow past nginx's default 8 KiB header buffer on long
  fallback chains). The full ledger remains in error response bodies.

### `code_execute` sandbox
The most exposed surface. Users (and tool-calling LLMs) can submit arbitrary
Python. We assume the code is hostile.

#### Layered defenses

1. **Backend selection** — at startup the server probes both:
   - `runsc` (gVisor): user-space syscall implementation, ~20MB RAM/call,
     ~500ms cold start. Native AF_ALG returns `EAFNOSUPPORT`.
   - `unshare`: user/network/PID/IPC namespaces + seccomp BPF. ~zero RAM
     overhead, ~100ms start. Same kernel as host.

   gVisor is preferred when present. Both pass the same verification probes.

2. **Network** — both backends have no network device / route. Outbound TCP /
   DNS / anything fails immediately.

3. **AF_ALG** — entry point for CVE-2026-31431 (Copy Fail, page-cache write
   via `authencesn`). gVisor doesn't implement AF_ALG; the unshare backend
   installs a seccomp BPF filter that returns `EACCES` for
   `socket(AF_ALG, ...)`.

4. **Resource limits** (in-script preamble):
   - RLIMIT_AS: 1 GiB virtual memory
   - RLIMIT_CPU: 60 seconds
   - RLIMIT_FSIZE: 32 MiB per file
   - RLIMIT_NPROC: 32 processes
   - RLIMIT_CORE: 0 (no coredumps)

5. **Filesystem** — fresh tempdir as cwd, deleted on return. Other host paths
   are read-only inside the sandbox; with no network there's no exfil channel.

6. **Per-call timeout** — 60 second hard cap, killed via subprocess.

#### What's NOT protected

- Host-level kernel CVEs (e.g. someone with shell on the host using
  `algif_aead` directly). Mitigate via the modprobe blacklist
  ([`docs/modprobe.d/cve-2026-dirtyfrag-copyfail.conf`](modprobe.d/cve-2026-dirtyfrag-copyfail.conf))
  and a patched kernel.
- Side-channel attacks (Spectre/Meltdown/Downfall variants). gVisor mitigates
  some; bare namespaces don't.
- Sandbox-internal correctness (the user's code can crash the sandbox or
  produce garbage results — that's their problem, not a security issue).

## Known mitigations applied

The repo includes a modprobe blacklist for three CVEs disclosed in 2026:

```bash
sudo cp docs/modprobe.d/cve-2026-dirtyfrag-copyfail.conf /etc/modprobe.d/
sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
```

| CVE | Component | Module | Status |
|---|---|---|---|
| [CVE-2026-31431][copy-fail] | Copy Fail (`authencesn` + AF_ALG + splice) | `algif_aead` | Often built-in to vendor kernels (e.g. el10uek). Patch kernel. Sandbox closes the path inside `code_execute`. |
| [CVE-2026-43284][dirtyfrag] | xfrm-ESP page-cache write | `esp4`, `esp6` | Blacklisted by the conf file → load fails. |
| [CVE-2026-43500][dirtyfrag] | RxRPC page-cache write | `rxrpc` | Blacklisted by the conf file → load fails. No upstream patch as of disclosure. |

[copy-fail]: https://github.com/theori-io/copy-fail-CVE-2026-31431 "Theori Xint disclosure + writeup at xint.io/blog/copy-fail-linux-distributions"
[dirtyfrag]: https://github.com/V4bel/dirtyfrag "Hyunwoo Kim's Dirty Frag disclosure (chains xfrm-ESP + RxRPC page-cache writes)"

<!-- Backup citations via the Internet Archive's Wayback Machine — if the
     upstream disclosure repos go private/archived/deleted, replace the
     primary links with these. Generate fresh snapshots before that
     happens via https://web.archive.org/save/<url>.
     - https://web.archive.org/web/2026*/https://github.com/theori-io/copy-fail-CVE-2026-31431
     - https://web.archive.org/web/2026*/https://github.com/V4bel/dirtyfrag
     - https://web.archive.org/web/2026*/https://xint.io/blog/copy-fail-linux-distributions  -->


Verify the blacklist is active:
```bash
sudo modprobe -n -v esp4    # → install /bin/false
sudo modprobe -n -v rxrpc   # → install /bin/false
lsmod | grep -E '^(esp4|esp6|rxrpc) '   # → empty
```

## Reporting

Found a vulnerability? Email the repo owner (see GitHub profile) rather than
opening a public issue. We'll respond within a week.

## Things to harden if you deploy publicly

- Run behind TLS (Caddy / nginx). Cookie is `Secure` so plain HTTP breaks
  sessions on browsers anyway.
- Use `--registration-token` to gate signup, or run with
  `--open-registration` only on a private network.
- Consider rate-limiting at the reverse proxy (NIMINI itself does no rate
  limiting beyond the `code_execute` per-call timeout).
- Back up `chat.db` (it has all the user keys + conversations) — but treat the
  backup as sensitive as the original.
