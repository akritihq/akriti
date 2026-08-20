"""Smoke tests for the packaging scaffold.

These check the install-quality commitments, not behaviour: the package must
import on a bare default install with no persistence backend present, and the
default closure must not quietly acquire one.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
from packaging.version import InvalidVersion, Version


def test_imports() -> None:
    import akriti

    assert akriti.__version__


def test_version_is_pep440() -> None:
    import akriti

    Version(akriti.__version__)


@pytest.mark.parametrize("version", ["0.1.0rc1", "1.0.0a1"])
def test_pep440_prereleases_are_accepted(version: str) -> None:
    assert Version(version).public == version


def test_invalid_pep440_version_is_rejected() -> None:
    with pytest.raises(InvalidVersion):
        Version("1.2.3.foo")


def test_runtime_and_distribution_versions_agree() -> None:
    import akriti

    assert importlib.metadata.version("akriti") == akriti.__version__


def test_version_source_configuration() -> None:
    """The package and Hatch must have one version source without TOML parsing."""

    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    lines = pyproject.read_text(encoding="utf-8").splitlines()

    def section(name: str) -> list[str]:
        start = next(i for i, line in enumerate(lines) if line == f"[{name}]")
        end = next(
            (i for i in range(start + 1, len(lines)) if lines[i].startswith("[")),
            len(lines),
        )
        return lines[start + 1 : end]

    project = section("project")
    hatch_version = section("tool.hatch.version")
    optional_dependencies = section("project.optional-dependencies")
    build_system = section("build-system")

    dynamic = next(line for line in project if line.strip().startswith("dynamic ="))
    assert dynamic.strip() == 'dynamic = ["version"]'
    assert not any(line.strip().startswith("version =") for line in project)
    assert 'path = "src/akriti/__init__.py"' in {line.strip() for line in hatch_version}

    def assignments(section_lines: list[str]) -> dict[str, list[str]]:
        parsed: dict[str, list[str]] = {}
        active_key: str | None = None
        for raw_line in section_lines:
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            if active_key is None:
                key, separator, value = line.partition("=")
                if not separator:
                    continue
                active_key = key.strip()
                parsed[active_key] = [value.strip()]
                if not value.strip().startswith("[") or value.strip().endswith("]"):
                    active_key = None
            else:
                parsed[active_key].append(line)
                if line.endswith("]"):
                    active_key = None
        return parsed

    optional_requirements = assignments(optional_dependencies)
    packaging_keys = {
        key
        for key, values in optional_requirements.items()
        if any('"packaging>=22"' in value for value in values)
    }
    assert packaging_keys == {"test"}
    assert not any(
        '"packaging>=22"' in value
        for values in assignments(project).values()
        for value in values
    )
    assert not any(
        '"packaging>=22"' in value
        for values in assignments(build_system).values()
        for value in values
    )


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
