"""Extensible project lint framework for Audio8-VC.

Every historical GitHub Actions failure, Windows packaging bug, ABI conflict,
and runtime regression is codified here as an independent, discoverable rule.
To add a new lint rule later, decorate a check function with `@register_rule`
in any module under `src/a8vc/lint/rules/`.
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
import re
import tomllib
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(slots=True)
class LintFinding:
    """A single violation emitted by a lint rule."""

    rule_id: str
    message: str
    severity: Severity = Severity.ERROR
    path: str | None = None
    line: int | None = None
    hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity.value,
            "path": self.path,
            "line": self.line,
            "message": self.message,
            "hint": self.hint,
        }


@dataclass(slots=True)
class LintContext:
    """Shared read-only repository context with per-run file/AST/TOML caching."""

    repo_root: Path
    _text_cache: dict[str, str] = field(default_factory=dict, repr=False)
    _ast_cache: dict[str, ast.Module] = field(default_factory=dict, repr=False)
    _toml_cache: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)

    def resolve(self, rel_path: str) -> Path:
        return self.repo_root / rel_path

    def exists(self, rel_path: str) -> bool:
        return self.resolve(rel_path).exists()

    def read_text(self, rel_path: str) -> str | None:
        if rel_path in self._text_cache:
            return self._text_cache[rel_path]
        target = self.resolve(rel_path)
        if not target.is_file():
            return None
        text = target.read_text(encoding="utf-8")
        self._text_cache[rel_path] = text
        return text

    def parse_toml(self, rel_path: str) -> dict[str, Any] | None:
        if rel_path in self._toml_cache:
            return self._toml_cache[rel_path]
        text = self.read_text(rel_path)
        if text is None:
            return None
        data = tomllib.loads(text)
        self._toml_cache[rel_path] = data
        return data

    def parse_python(self, rel_path: str) -> ast.Module | None:
        if rel_path in self._ast_cache:
            return self._ast_cache[rel_path]
        text = self.read_text(rel_path)
        if text is None:
            return None
        tree = ast.parse(text, filename=rel_path)
        self._ast_cache[rel_path] = tree
        return tree

    def glob(self, pattern: str) -> list[str]:
        return sorted(
            p.relative_to(self.repo_root).as_posix()
            for p in self.repo_root.glob(pattern)
            if p.is_file()
        )

    @staticmethod
    def find_line(text: str, pattern: str | re.Pattern[str]) -> int | None:
        """Return 1-based line number of the first regex match in `text`."""
        rx = re.compile(pattern) if isinstance(pattern, str) else pattern
        m = rx.search(text)
        if not m:
            return None
        return text.count("\n", 0, m.start()) + 1


RuleCheckFn = Callable[[LintContext], list[LintFinding]]


@dataclass(slots=True)
class LintRule:
    """Metadata and check implementation for a single lint rule."""

    rule_id: str
    title: str
    category: str
    description: str
    history: str
    check: RuleCheckFn
    default_severity: Severity = Severity.ERROR

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "title": self.title,
            "category": self.category,
            "severity": self.default_severity.value,
            "description": self.description,
            "history": self.history,
        }


@dataclass(slots=True)
class LintReport:
    """Aggregated outcome of running a set of lint rules."""

    repo_root: str
    rules_run: list[str]
    findings: list[LintFinding]

    @property
    def error_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.WARNING)

    def ok(self, *, strict: bool = False) -> bool:
        if strict:
            return len(self.findings) == 0
        return self.error_count == 0

    def to_dict(self, *, strict: bool = False) -> dict[str, Any]:
        return {
            "ok": self.ok(strict=strict),
            "repo_root": self.repo_root,
            "rules_run": self.rules_run,
            "rules_count": len(self.rules_run),
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "findings": [f.to_dict() for f in self.findings],
        }


class LintRegistry:
    """Registry supporting auto-discovery and dynamic extension of lint rules."""

    def __init__(self) -> None:
        self._rules: dict[str, LintRule] = {}
        self._discovered = False

    def register(self, rule: LintRule) -> LintRule:
        if rule.rule_id in self._rules:
            raise ValueError(f"Duplicate lint rule_id registered: {rule.rule_id!r}")
        self._rules[rule.rule_id] = rule
        return rule

    def rule(
        self,
        rule_id: str,
        *,
        title: str,
        category: str,
        description: str,
        history: str,
        severity: Severity = Severity.ERROR,
    ) -> Callable[[RuleCheckFn], RuleCheckFn]:
        """Decorator that registers a rule function on this registry."""

        def _decorator(fn: RuleCheckFn) -> RuleCheckFn:
            self.register(
                LintRule(
                    rule_id=rule_id,
                    title=title,
                    category=category,
                    description=description,
                    history=history,
                    check=fn,
                    default_severity=severity,
                )
            )
            return fn

        return _decorator

    def discover_builtin_rules(self) -> None:
        """Auto-import all modules in `a8vc.lint.rules` so rules self-register."""
        if self._discovered:
            return
        import a8vc.lint.rules as rules_pkg

        for mod_info in sorted(
            pkgutil.iter_modules(rules_pkg.__path__), key=lambda m: m.name
        ):
            if mod_info.name.startswith("_"):
                continue
            importlib.import_module(f"{rules_pkg.__name__}.{mod_info.name}")
        self._discovered = True

    def list_rules(
        self,
        *,
        select: Iterable[str] | None = None,
        ignore: Iterable[str] | None = None,
        categories: Iterable[str] | None = None,
    ) -> list[LintRule]:
        self.discover_builtin_rules()
        select_set = {s.strip().upper() for s in (select or ()) if s.strip()}
        ignore_set = {s.strip().upper() for s in (ignore or ()) if s.strip()}
        cat_set = {c.strip().lower() for c in (categories or ()) if c.strip()}

        selected: list[LintRule] = []
        for rule_id in sorted(self._rules):
            r = self._rules[rule_id]
            if select_set and not any(
                rule_id == pat or rule_id.startswith(pat) for pat in select_set
            ):
                continue
            if ignore_set and any(
                rule_id == pat or rule_id.startswith(pat) for pat in ignore_set
            ):
                continue
            if cat_set and r.category.lower() not in cat_set:
                continue
            selected.append(r)
        return selected

    def run(
        self,
        repo_root: str | Path,
        *,
        select: Iterable[str] | None = None,
        ignore: Iterable[str] | None = None,
        categories: Iterable[str] | None = None,
    ) -> LintReport:
        root = Path(repo_root).resolve()
        ctx = LintContext(repo_root=root)
        rules = self.list_rules(select=select, ignore=ignore, categories=categories)

        findings: list[LintFinding] = []
        for r in rules:
            try:
                rule_findings = r.check(ctx)
            except Exception as exc:
                rule_findings = [
                    LintFinding(
                        rule_id=r.rule_id,
                        severity=Severity.ERROR,
                        message=f"Lint rule crashed with {type(exc).__name__}: {exc}",
                        hint="Fix the lint rule implementation or malformed input file.",
                    )
                ]
            findings.extend(rule_findings)

        return LintReport(
            repo_root=str(root),
            rules_run=[r.rule_id for r in rules],
            findings=findings,
        )


default_registry = LintRegistry()


def register_rule(
    rule_id: str,
    *,
    title: str,
    category: str,
    description: str,
    history: str,
    severity: Severity = Severity.ERROR,
) -> Callable[[RuleCheckFn], RuleCheckFn]:
    """Register a rule on the default Audio8-VC lint registry."""
    return default_registry.rule(
        rule_id,
        title=title,
        category=category,
        description=description,
        history=history,
        severity=severity,
    )


def discover_repo_root(start: Path | None = None) -> Path:
    """Locate the Audio8-VC repository root containing `pyproject.toml`."""
    candidates: list[Path] = []
    if start is not None:
        candidates.append(start.resolve())
    candidates.append(Path.cwd().resolve())
    candidates.append(Path(__file__).resolve().parents[3])

    for base in candidates:
        for cur in (base, *base.parents):
            if (cur / "pyproject.toml").is_file() and (cur / "src" / "a8vc").is_dir():
                return cur
    raise FileNotFoundError(
        "Could not locate Audio8-VC repository root (expected pyproject.toml + src/a8vc). "
        "Pass --root /path/to/repo explicitly."
    )


def format_report_text(report: LintReport, *, strict: bool = False) -> str:
    """Render a human-readable summary of lint findings."""
    lines: list[str] = []
    if not report.findings:
        lines.append(
            f"[PASS] Audio8-VC lint: {len(report.rules_run)} rules passed with 0 issues."
        )
        return "\n".join(lines)

    for finding in report.findings:
        loc = ""
        if finding.path:
            loc = f" {finding.path}"
            if finding.line is not None:
                loc += f":{finding.line}"
        sev = finding.severity.value.upper()
        lines.append(f"[{sev}] {finding.rule_id}{loc}: {finding.message}")
        if finding.hint:
            lines.append(f"        hint: {finding.hint}")

    status = "PASS" if report.ok(strict=strict) else "FAIL"
    lines.append(
        f"[{status}] Audio8-VC lint: {len(report.rules_run)} rules checked, "
        f"{report.error_count} error(s), {report.warning_count} warning(s)."
    )
    return "\n".join(lines)
