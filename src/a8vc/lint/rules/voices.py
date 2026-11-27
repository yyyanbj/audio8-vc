"""Built-in voice pack integrity and repository hygiene lint rules (VCxxx)."""

from __future__ import annotations

import json

from a8vc.lint.core import LintContext, LintFinding, register_rule
from a8vc.voices.format import A8VOICE_EXT, verify_voice_pack


@register_rule(
    "VC001",
    title="Bundled built-in .a8voice packs must match catalog.json and pass SHA-256 verification",
    category="voices",
    description=(
        "Every voice listed in `src/a8vc/voices/builtins/catalog.json` must exist on disk "
        "and pass `verify_voice_pack()`, with >= 15 voices and no unlisted orphan `.a8voice` files."
    ),
    history="Protects the 15 curated built-in voice packs bundled since v0.2.0 (71e6636).",
)
def check_builtin_voices_integrity(ctx: LintContext) -> list[LintFinding]:
    findings: list[LintFinding] = []
    builtins_rel = "src/a8vc/voices/builtins"
    builtins_dir = ctx.resolve(builtins_rel)
    if not builtins_dir.is_dir():
        return [
            LintFinding(
                rule_id="VC001",
                path=builtins_rel,
                message="Bundled built-in voices directory is missing.",
            )
        ]

    catalog_rel = f"{builtins_rel}/catalog.json"
    catalog_text = ctx.read_text(catalog_rel)
    if catalog_text is None:
        return [
            LintFinding(
                rule_id="VC001",
                path=catalog_rel,
                message="Bundled built-in voice catalog `catalog.json` is missing.",
                hint="Regenerate built-in voices with `scripts/build_builtin_voices.py`.",
            )
        ]

    try:
        catalog_data = json.loads(catalog_text)
    except json.JSONDecodeError as exc:
        return [
            LintFinding(
                rule_id="VC001",
                path=catalog_rel,
                message=f"Invalid JSON in catalog.json: {exc}",
            )
        ]

    voices = catalog_data.get("voices", [])
    if not isinstance(voices, list) or len(voices) < 15:
        findings.append(
            LintFinding(
                rule_id="VC001",
                path=catalog_rel,
                message=f"Expected at least 15 built-in voices in catalog.json, found {len(voices) if isinstance(voices, list) else 0}.",
                hint="Keep all curated built-in voice packs intact in `src/a8vc/voices/builtins/`.",
            )
        )
        if not isinstance(voices, list):
            return findings

    expected_files: set[str] = set()
    for item in voices:
        vid = str(item.get("voice_id", "")).strip()
        fname = str(item.get("filename") or f"{vid}{A8VOICE_EXT}")
        expected_files.add(fname)
        rel = f"{builtins_rel}/{fname}"
        fpath = ctx.resolve(rel)
        if not fpath.is_file():
            findings.append(
                LintFinding(
                    rule_id="VC001",
                    path=rel,
                    message=f"Built-in voice `{vid}` is listed in catalog.json but missing on disk.",
                    hint="Regenerate built-in voice packs with `scripts/build_builtin_voices.py`.",
                )
            )
            continue
        try:
            verify_voice_pack(fpath)
        except Exception as exc:
            findings.append(
                LintFinding(
                    rule_id="VC001",
                    path=rel,
                    message=f"Built-in voice pack failed verification: {type(exc).__name__}: {exc}",
                    hint="Rebuild or repair the corrupted .a8voice archive.",
                )
            )

    for actual in sorted(builtins_dir.glob(f"*{A8VOICE_EXT}")):
        if actual.name not in expected_files:
            findings.append(
                LintFinding(
                    rule_id="VC001",
                    path=f"{builtins_rel}/{actual.name}",
                    message=f"Orphan bundled voice file `{actual.name}` is not registered in catalog.json.",
                    hint="Register it in `catalog.json` or remove the unused file.",
                )
            )

    return findings


@register_rule(
    "VC002",
    title=".gitignore must ignore runtime artifacts (voices/.builtins_seeded, models, dist, build)",
    category="voices",
    description=(
        "Prevents generated runtime markers and build directories from polluting git commits."
    ),
    history="Local server runs created untracked `voices/.builtins_seeded` marker files.",
)
def check_gitignore_runtime_artifacts(ctx: LintContext) -> list[LintFinding]:
    rel = ".gitignore"
    text = ctx.read_text(rel)
    if text is None:
        return [LintFinding(rule_id="VC002", path=rel, message="Missing .gitignore.")]

    lines = {
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    required = [
        "voices/.builtins_seeded",
        "voices/*.a8voice",
        "models/",
        "dist/",
        "build/",
    ]
    findings: list[LintFinding] = []
    for entry in required:
        if entry not in lines:
            findings.append(
                LintFinding(
                    rule_id="VC002",
                    path=rel,
                    message=f".gitignore is missing required ignore entry `{entry}`.",
                    hint=f"Add `{entry}` to .gitignore.",
                )
            )
    return findings
