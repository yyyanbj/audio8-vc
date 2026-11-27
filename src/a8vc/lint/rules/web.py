"""Frontend WebUI and browser WebSocket inference lint rules (WEBxxx)."""

from __future__ import annotations

import re

from a8vc.lint.core import LintContext, LintFinding, register_rule


@register_rule(
    "WEB001",
    title="WebUI must parse API responses with parseApiResponse (no raw response.json() calls)",
    category="web",
    description=(
        "Calling `await res.json()` directly throws `SyntaxError: Unexpected token 'I'` when a proxy or "
        "server returns plain-text 500 `Internal Server Error`. Always route fetch responses through `parseApiResponse()`."
    ),
    history="v0.2.0 index.html crashed on plain-text HTTP 500 responses, masking error messages in the browser UI.",
)
def check_no_raw_res_json(ctx: LintContext) -> list[LintFinding]:
    rel = "src/a8vc/server/static/index.html"
    text = ctx.read_text(rel)
    if text is None:
        return [LintFinding(rule_id="WEB001", path=rel, message="Missing WebUI index.html.")]

    findings: list[LintFinding] = []
    if "async function parseApiResponse(" not in text:
        findings.append(
            LintFinding(
                rule_id="WEB001",
                path=rel,
                message="index.html is missing `async function parseApiResponse(res)` safe JSON/text parser.",
                hint="Define `parseApiResponse(res)` reading `await res.text()` with safe `JSON.parse` fallback.",
            )
        )

    for m in re.finditer(r"\b\w+\.json\s*\(", text):
        line = text.count("\n", 0, m.start()) + 1
        findings.append(
            LintFinding(
                rule_id="WEB001",
                path=rel,
                line=line,
                message=f"Unsafe direct `.json()` call (`{m.group(0).strip()}`) found in index.html.",
                hint="Use `await parseApiResponse(res)` so non-JSON 500/proxy errors display clearly instead of throwing SyntaxError.",
            )
        )

    return findings


@register_rule(
    "WEB002",
    title="Browser WebSocket VC must resume AudioContext, route ScriptProcessor via 0-gain sink, and handle Blob/error frames",
    category="web",
    description=(
        "Chromium/Edge browsers suspend `AudioContext` until explicitly resumed and stop firing "
        "`ScriptProcessorNode.onaudioprocess` unless connected to `audioCtx.destination` (through a 0-gain "
        "node to avoid mic feedback). WebSocket handlers must also accept both ArrayBuffer and Blob audio frames "
        "and display server `error` frames."
    ),
    history="Browser WebSocket voice changing failed silently in Chromium/WebView2 when AudioContext or ScriptProcessor stalled.",
)
def check_browser_vc_audio_pipeline(ctx: LintContext) -> list[LintFinding]:
    rel = "src/a8vc/server/static/index.html"
    text = ctx.read_text(rel)
    if text is None:
        return []

    findings: list[LintFinding] = []
    checks = [
        (
            r"async\s+function\s+ensureAudioContext\b",
            "Missing `ensureAudioContext()` helper to create/resume AudioContext.",
            "Create and `await audioCtx.resume()` inside user-gesture handlers before capture/playback.",
        ),
        (
            r"audioCtx\.resume\s*\(",
            "index.html does not call `audioCtx.resume()` to unlock browser autoplay/capture.",
            "Call `await audioCtx.resume()` when `audioCtx.state === 'suspended'`.",
        ),
        (
            r"createGain\s*\(",
            "Browser mic capture must route ScriptProcessorNode through a 0-gain GainNode to audioCtx.destination.",
            "Connect `micSource -> processor -> zeroGain (gain=0) -> audioCtx.destination` so onaudioprocess stays active without echo.",
        ),
        (
            r"gain\.value\s*=\s*0\b",
            "Zero-gain sink (`zeroGain.gain.value = 0`) missing from browser mic capture pipeline.",
            "Set `zeroGain.gain.value = 0` before connecting to `audioCtx.destination` to prevent mic loopback feedback.",
        ),
        (
            r"instanceof\s+Blob",
            "WebSocket onmessage should handle `Blob` binary frames in addition to `ArrayBuffer`.",
            "Handle both `ev.data instanceof ArrayBuffer` and `ev.data instanceof Blob`.",
        ),
        (
            r'msg\.type\s*===\s*"error"',
            'WebSocket onmessage must surface server `msg.type === "error"` messages to the user.',
            'Handle `msg.type === "error"` and display `msg.message` in logs and status metrics.',
        ),
    ]
    for pattern, msg, hint in checks:
        if not re.search(pattern, text):
            findings.append(
                LintFinding(
                    rule_id="WEB002",
                    path=rel,
                    message=msg,
                    hint=hint,
                )
            )
    return findings
