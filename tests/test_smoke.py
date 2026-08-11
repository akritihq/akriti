"""Smoke tests for the packaging scaffold.

These check the install-quality commitments, not behaviour: the package must
import on a bare default install with no persistence backend present, and the
default closure must not quietly acquire one.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys

import pytest


def test_imports() -> None:
    import akriti

    assert akriti.__version__


def test_version_is_pep440() -> None:
    import re

    import akriti

    assert re.fullmatch(r"\d+\.\d+\.\d+(\.\w+\d*)?", akriti.__version__)


def test_package_is_typed() -> None:
    """py.typed must ship, or type hints are invisible to downstream users."""

    from pathlib import Path

    spec = importlib.util.find_spec("akriti")
    assert spec is not None
    assert spec.origin is not None
    assert (Path(spec.origin).parent / "py.typed").is_file()


@pytest.mark.parametrize("backend", ["gudhi", "ripser", "persim", "gtda"])
def test_no_backend_is_a_hard_dependency(backend: str) -> None:
    """The default install must not require any persistence backend.

    This does not assert the backend is absent -- a dev environment has them
    all. It asserts akriti imports without having imported one, which is what
    a hard dependency would force. See DEPENDENCIES.md.
    """
    import sys

    for mod in [m for m in sys.modules if m.split(".")[0] == "akriti"]:
        del sys.modules[mod]
    before = backend in sys.modules

    import akriti  # noqa: F401

    if not before:
        assert backend not in sys.modules, (
            f"importing akriti pulled in {backend}; it must stay optional"
        )


@pytest.mark.parametrize("module", ["akriti", "akriti.diagrams"])
def test_numpy_is_not_a_hard_dependency(module: str) -> None:
    """numpy is not in the default closure either. RFC-0001 §3.3, §10.1 (2).

    The declared-dependency half of this is checked by
    `tools/check_license_closure.py` against a clean venv, which is the only
    place `pip install akriti` can actually be observed. What that cannot
    check is the half that makes the declaration honest: `diagrams/core.py`
    and `diagrams/adapters.py` must import nothing beyond the standard library
    at import time and resolve the caller's namespace (native
    `__array_namespace__` or a documented lazy fallback), so a
    single convenience `import numpy` would make the empty closure a lie
    without failing any dependency check. The row-sequence adapter fallback,
    torch's array-api-compat resolver, and Parquet's PyArrow exporter are all
    lazy optional boundaries and are not import-time requirements.

    Run in a subprocess rather than by clearing `sys.modules`, which is how
    the backend test above manages it. numpy is imported by almost every other
    module in this suite, so an in-process check would be skipped or vacuously
    true depending on collection order -- the failure mode being that the test
    passes for the wrong reason exactly when it matters.
    """
    code = (
        "import importlib, sys; "
        f"importlib.import_module({module!r}); "
        "print('numpy' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "False", (
        f"importing {module} pulled in numpy; the default install declares no "
        "third-party dependency at all, and numpy belongs only inside "
        "row-sequence adapters and io.py's save/load as lazy, function-scoped "
        "imports"
    )
