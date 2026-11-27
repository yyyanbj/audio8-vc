"""FastAPI HTTP and WebSocket API contract lint rules (APIxxx)."""

from __future__ import annotations

import ast
import re

from a8vc.lint.core import LintContext, LintFinding, register_rule


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


@register_rule(
    "API001",
    title="FastAPI app must register a global JSON exception handler (no plain-text Internal Server Error)",
    category="server",
    description=(
        "Unhandled server exceptions must return structured JSON (`{\"detail\": ..., \"error_type\": ...}`) "
        "rather than Starlette's plain-text `Internal Server Error`."
    ),
    history="v0.2.0 returned plain-text 500 Internal Server Error, hiding root causes and crashing frontend JSON parsing.",
)
def check_global_json_exception_handler(ctx: LintContext) -> list[LintFinding]:
    rel = "src/a8vc/server/app.py"
    tree = ctx.parse_python(rel)
    if tree is None:
        return [LintFinding(rule_id="API001", path=rel, message="Missing src/a8vc/server/app.py.")]

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            if (
                isinstance(dec, ast.Call)
                and isinstance(dec.func, ast.Attribute)
                and dec.func.attr == "exception_handler"
                and dec.args
                and isinstance(dec.args[0], ast.Name)
                and dec.args[0].id == "Exception"
            ):
                return []

    return [
        LintFinding(
            rule_id="API001",
            path=rel,
            message="create_app() is missing an `@app.exception_handler(Exception)` returning JSONResponse.",
            hint="Register `@app.exception_handler(Exception)` to return structured JSON error payloads.",
        )
    ]


@register_rule(
    "API002",
    title="/api/tts/stream must prime the first audio chunk before returning StreamingResponse",
    category="server",
    description=(
        "Returning StreamingResponse before invoking synthesis commits HTTP 200 headers too early, "
        "turning runtime/prefill errors into broken mid-stream disconnects instead of JSON errors."
    ),
    history="v0.2.0 /api/tts/stream raised inside async generator after HTTP 200 was already committed.",
)
def check_tts_stream_primes_first_chunk(ctx: LintContext) -> list[LintFinding]:
    rel = "src/a8vc/server/app.py"
    tree = ctx.parse_python(rel)
    text = ctx.read_text(rel) or ""
    if tree is None:
        return []

    fn = _find_function(tree, "tts_stream")
    if fn is None:
        return [
            LintFinding(
                rule_id="API002",
                path=rel,
                message="Could not locate `tts_stream` endpoint in src/a8vc/server/app.py.",
            )
        ]

    segment = ast.get_source_segment(text, fn) or ""
    stream_idx = segment.find("return StreamingResponse(")
    prime_idx = segment.find("first_pcm")
    if stream_idx == -1 or prime_idx == -1 or prime_idx > stream_idx:
        return [
            LintFinding(
                rule_id="API002",
                path=rel,
                line=fn.lineno,
                message="tts_stream must prime the first PCM chunk (`first_pcm`) inside try/except before returning StreamingResponse.",
                hint="Await the first synthesized chunk before returning StreamingResponse so failures raise HTTPException.",
            )
        ]
    return []


@register_rule(
    "API003",
    title="WebSocket /ws/vc must guard engine init, audio submission, and worker errors with JSON error frames",
    category="server",
    description=(
        "Errors during `/ws/vc` engine initialization, background synthesis worker execution, or audio "
        "chunk submission must send `{\"type\": \"error\", \"message\": ...}` to the browser instead of failing silently."
    ),
    history="v0.2.0 /ws/vc crashed before sending any error frame when VAD/ASR initialization failed.",
)
def check_ws_vc_error_handling(ctx: LintContext) -> list[LintFinding]:
    rel = "src/a8vc/server/app.py"
    tree = ctx.parse_python(rel)
    text = ctx.read_text(rel) or ""
    if tree is None:
        return []

    fn = _find_function(tree, "ws_voice_changer")
    if fn is None:
        return [
            LintFinding(
                rule_id="API003",
                path=rel,
                message="Could not locate `ws_voice_changer` endpoint in src/a8vc/server/app.py.",
            )
        ]

    segment = ast.get_source_segment(text, fn) or ""
    findings: list[LintFinding] = []

    if not re.search(r"try:\s*\n\s*engine\s*=\s*await\s+asyncio\.to_thread\(srv_state\.create_vc_engine\)", segment):
        findings.append(
            LintFinding(
                rule_id="API003",
                path=rel,
                line=fn.lineno,
                message="ws_voice_changer must wrap `create_vc_engine` in try/except and send a JSON error frame on failure.",
                hint="Catch initialization exceptions, send `{\"type\": \"error\", \"message\": ...}`, and close with code 1011.",
            )
        )

    if "on_error=" not in segment:
        findings.append(
            LintFinding(
                rule_id="API003",
                path=rel,
                line=fn.lineno,
                message="ws_voice_changer must pass `on_error=` to RealtimeVCWorker so worker errors reach the browser.",
                hint="Forward worker exceptions into `out_queue` as `{\"type\": \"error\", \"message\": ...}`.",
            )
        )

    if '"type": "error"' not in segment and "'type': 'error'" not in segment:
        findings.append(
            LintFinding(
                rule_id="API003",
                path=rel,
                line=fn.lineno,
                message="ws_voice_changer does not emit structured `{\"type\": \"error\"}` WebSocket messages.",
            )
        )

    return findings
