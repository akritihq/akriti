"""Smoke tests for the packaging scaffold.

These check the install-quality commitments, not behaviour: the package must
import on a bare default install with no persistence backend present, and the
default closure must not quietly acquire one.
"""

from __future__ import annotations

import importlib.util

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
