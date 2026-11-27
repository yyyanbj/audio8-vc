"""Auto-discovered lint rule modules for Audio8-VC.

How to add a new lint rule:
1. Pick an existing domain module in this package (`ci.py`, `packaging.py`,
   `deps.py`, `runtime.py`, `server.py`, `web.py`, `cli.py`, `voices.py`),
   or create a new `.py` file inside `src/a8vc/lint/rules/`.
2. Define a check function taking `ctx: LintContext` and returning
   `list[LintFinding]`, decorated with `@register_rule(...)`.
3. Add a unit test (both passing and failing fixture cases) in `tests/test_lint.py`.

All non-underscore modules in this directory are imported automatically by
`LintRegistry.discover_builtin_rules()`.
"""
