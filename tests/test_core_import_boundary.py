"""``akriti.core`` imports with neither numpy nor scipy. RFC-0001 §10.1 req. 2.

``akriti[core]`` declares scipy and :func:`fit_configuration` needs it, but the
promise the packaging makes is narrower than that: *importing* ``akriti.core``,
and holding a :class:`~akriti.core.Configuration`, must work on an install that
fetched nothing. Embedding does not -- that needs ``akriti[numpy]`` -- and an
earlier revision of this branch claimed otherwise in three places.

Until now the narrow promise was prose. `REVIEWING.md`: a trap a reader cannot
see becomes a standing regression test, not prose.

Two checks, because neither alone is enough. The syntax pass names the
offending line, which is what a contributor can act on; the subprocess pass
proves the whole import graph, including any module ``core`` reaches that
imports numpy on its own behalf.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "src" / "akriti" / "core"
LAZY_ONLY = ("numpy", "scipy")


def _module_scope_imports(source: str) -> set[str]:
    """Top-level distribution names imported by ``source``.

    Only ``tree.body`` -- an import inside a function is the lazy boundary
    §10.1 requires, not a violation of it.
    """
    names: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Import):
            names.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.partition(".")[0])
    return names


def test_core_imports_numpy_and_scipy_lazily_only() -> None:
    offenders = [
        f"{path.relative_to(ROOT).as_posix()}: {name}"
        for path in sorted(CORE.rglob("*.py"))
        for name in sorted(_module_scope_imports(path.read_text(encoding="utf-8")))
        if name in LAZY_ONLY
    ]
    assert offenders == []


_WITHOUT_NUMPY_OR_SCIPY = """
import sys


class Blocked:
    def find_spec(self, name, path=None, target=None):
        if name.partition(".")[0] in ("numpy", "scipy"):
            raise ImportError(f"{name} is blocked for this check")
        return None


sys.meta_path.insert(0, Blocked())
for loaded in [n for n in sys.modules if n.partition(".")[0] in ("numpy", "scipy")]:
    del sys.modules[loaded]

import akriti.core

assert akriti.core.Configuration is not None
assert akriti.core.fit_configuration is not None
print("imported")
"""


def test_core_imports_in_a_process_where_both_are_unimportable() -> None:
    """The claim end to end, with both packages made unimportable.

    A meta-path finder that raises rather than a venv without them: the test
    environment has both installed, so this is the only way the assertion runs
    where it matters. ``fit_configuration`` is reached as an attribute and not
    called -- calling it is what needs scipy.
    """
    result = subprocess.run(
        [sys.executable, "-c", _WITHOUT_NUMPY_OR_SCIPY],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("imported")
