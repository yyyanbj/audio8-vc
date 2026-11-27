"""ONNX Runtime, Silero VAD, ASR, and GPU provider safety lint rules (RTxxx)."""

from __future__ import annotations

import ast

from a8vc.lint.core import LintContext, LintFinding, register_rule


@register_rule(
    "RT001",
    title="SileroVADOnnx must support sherpa ('x','h','c'), v5 ('input','state'), and v4 ('input','sr','h','c') exports",
    category="runtime",
    description=(
        "Different Silero VAD ONNX exports expose different input/output tensor names. "
        "`SileroVADOnnx` must inspect model inputs dynamically and `create_vad` must accept `str | Path` "
        "and safely fall back to EnergyVAD in `auto` mode."
    ),
    history=(
        "v0.2.0 hardcoded 'input'/'sr' inputs and crashed with `Missing required input 'x'` on the "
        "first frame of downloaded silero_vad.onnx; create_vad also crashed when passed a str path."
    ),
)
def check_vad_multi_signature(ctx: LintContext) -> list[LintFinding]:
    rel = "src/a8vc/audio/vad.py"
    text = ctx.read_text(rel)
    tree = ctx.parse_python(rel)
    if text is None or tree is None:
        return [LintFinding(rule_id="RT001", path=rel, message="Missing src/a8vc/audio/vad.py.")]

    findings: list[LintFinding] = []
    silero_cls: ast.ClassDef | None = None
    create_vad_fn: ast.FunctionDef | None = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "SileroVADOnnx":
            silero_cls = node
        elif isinstance(node, ast.FunctionDef) and node.name == "create_vad":
            create_vad_fn = node

    if silero_cls is None:
        return [
            LintFinding(
                rule_id="RT001",
                path=rel,
                message="SileroVADOnnx class not found in src/a8vc/audio/vad.py.",
            )
        ]

    # Gather all string literals inside SileroVADOnnx
    literals = {
        n.value
        for n in ast.walk(silero_cls)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }
    for required_tensor in ("x", "input", "state", "h", "c"):
        if required_tensor not in literals:
            findings.append(
                LintFinding(
                    rule_id="RT001",
                    path=rel,
                    line=silero_cls.lineno,
                    message=f"SileroVADOnnx does not handle tensor name `{required_tensor}`.",
                    hint="Support sherpa ('x','h','c'), v5 ('input','state'), and v4 ('input','sr','h','c').",
                )
            )

    if create_vad_fn is not None:
        has_path_coerce = any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "Path"
            for n in ast.walk(create_vad_fn)
        )
        has_fallback_try = any(isinstance(n, ast.Try) for n in ast.walk(create_vad_fn))
        if not has_path_coerce:
            findings.append(
                LintFinding(
                    rule_id="RT001",
                    path=rel,
                    line=create_vad_fn.lineno,
                    message="create_vad should coerce `vad_model_path` with `Path(...)` to accept both str and Path.",
                    hint="Convert `vad_model_path` via `Path(vad_model_path)` before calling `.is_file()`.",
                )
            )
        if not has_fallback_try:
            findings.append(
                LintFinding(
                    rule_id="RT001",
                    path=rel,
                    line=create_vad_fn.lineno,
                    message="create_vad(backend='auto') must wrap SileroVADOnnx init in try/except with EnergyVAD fallback.",
                    hint="Catch initialization exceptions in auto mode and fall back to EnergyVAD with a warning.",
                )
            )

    return findings


@register_rule(
    "RT002",
    title="DirectML INT8 AR models must route to CPU and runtime sessions must support CPU fallback",
    category="runtime",
    description=(
        "Audio8 `slow_ar_int8.onnx` and `fast_ar_int8.onnx` contain DynamicQuantizeLinear / MatMulInteger / "
        "ScatterND ops that fail with 80070057 on DirectML GPU. `select_model_provider` must keep INT8 AR "
        "models on CPU while accelerating FP16 codec models on DirectML, and runtime inference must fall back "
        "to CPU on GPU execution errors."
    ),
    history="DirectML builds crashed during AR synthesis on Windows GPUs before per-role provider selection and CPU fallback.",
)
def check_dml_int8_safety(ctx: LintContext) -> list[LintFinding]:
    findings: list[LintFinding] = []

    prov_rel = "src/a8vc/runtime/providers.py"
    prov_tree = ctx.parse_python(prov_rel)
    prov_text = ctx.read_text(prov_rel) or ""
    if prov_tree is None:
        findings.append(LintFinding(rule_id="RT002", path=prov_rel, message="Missing providers.py."))
    else:
        fn_names = {n.name for n in prov_tree.body if isinstance(n, ast.FunctionDef)}
        if "select_model_provider" not in fn_names:
            findings.append(
                LintFinding(
                    rule_id="RT002",
                    path=prov_rel,
                    message="Missing `select_model_provider` guard in providers.py.",
                    hint="Route DirectML INT8 `slow_ar`/`fast_ar` sessions to CPU while keeping FP16 codec on DML.",
                )
            )
        elif "DmlExecutionProvider" not in prov_text or "slow_ar" not in prov_text:
            findings.append(
                LintFinding(
                    rule_id="RT002",
                    path=prov_rel,
                    message="providers.py `select_model_provider` must guard `DmlExecutionProvider` for `slow_ar`/`fast_ar`.",
                )
            )

    rt_rel = "src/a8vc/runtime/audio8_onnx.py"
    rt_text = ctx.read_text(rt_rel) or ""
    if "select_model_provider" not in rt_text:
        findings.append(
            LintFinding(
                rule_id="RT002",
                path=rt_rel,
                message="Audio8OnnxRuntime must call `select_model_provider` when creating ONNX sessions.",
                hint="Use `select_model_provider(provider, model_role=..., precision=...)` for slow_ar/fast_ar/codec_decoder.",
            )
        )
    if "_fallback_session_to_cpu" not in rt_text:
        findings.append(
            LintFinding(
                rule_id="RT002",
                path=rt_rel,
                message="Audio8OnnxRuntime must implement `_fallback_session_to_cpu` for GPU runtime errors.",
                hint="Catch non-CPU session.run() errors and transparently recreate the session on CPUExecutionProvider.",
            )
        )

    return findings


@register_rule(
    "RT003",
    title="onnxruntime must be imported before sherpa_onnx to prevent Windows DLL shadowing",
    category="runtime",
    description=(
        "On Windows, DLLs are cached process-wide by basename (`onnxruntime.dll`). "
        "Importing `onnxruntime` before `sherpa_onnx` guarantees the flavor-matched ORT DLL "
        "(CPU/DirectML/CUDA) is loaded first."
    ),
    history="Importing sherpa_onnx first on Windows risked loading sherpa_onnx's CPU-only onnxruntime.dll before GPU ORT.",
)
def check_ort_imported_before_sherpa(ctx: LintContext) -> list[LintFinding]:
    rel = "src/a8vc/asr/sherpa.py"
    tree = ctx.parse_python(rel)
    if tree is None:
        return [LintFinding(rule_id="RT003", path=rel, message="Missing src/a8vc/asr/sherpa.py.")]

    findings: list[LintFinding] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        ort_line: int | None = None
        sherpa_line: int | None = None
        for sub in ast.walk(node):
            if isinstance(sub, ast.Import):
                for alias in sub.names:
                    if alias.name == "onnxruntime" and ort_line is None:
                        ort_line = sub.lineno
                    if alias.name == "sherpa_onnx" and sherpa_line is None:
                        sherpa_line = sub.lineno
            elif isinstance(sub, ast.ImportFrom):
                if sub.module == "onnxruntime" and ort_line is None:
                    ort_line = sub.lineno
                if sub.module == "sherpa_onnx" and sherpa_line is None:
                    sherpa_line = sub.lineno
        if sherpa_line is not None and (ort_line is None or ort_line > sherpa_line):
            findings.append(
                LintFinding(
                    rule_id="RT003",
                    path=rel,
                    line=sherpa_line,
                    message=(
                        f"`{node.name}` imports `sherpa_onnx` without importing `onnxruntime` first."
                    ),
                    hint="Import `onnxruntime` inside a try/except block before `import sherpa_onnx`.",
                )
            )
    return findings
