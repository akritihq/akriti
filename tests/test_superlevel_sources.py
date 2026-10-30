"""Focused tests for how the A.12 superlevel-sources script reports itself.

`rfcs/evidence/superlevel_sources.py` measures libraries CI does not install,
so its default run usually measures nothing. A run like that must not read as
a reproduction: it names every source it could not import, prints no
conclusion for one, and exits 1 when it measured none. Every library is forced
absent here, so the outcome does not depend on what happens to be installed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _ROOT / "rfcs/evidence/superlevel_sources.py"
_SPEC = importlib.util.spec_from_file_location("superlevel_sources", _SCRIPT_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
superlevel_sources = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(superlevel_sources)

# `None` in `sys.modules` makes the import raise `ImportError`.
_ABSENT = ("dionysus", "oineus", "torch", "torch_topological", "homcloud")


@pytest.fixture
def nothing_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (*_ABSENT, "homcloud.interface"):
        monkeypatch.setitem(sys.modules, name, None)


def test_a_run_that_measures_nothing_fails_and_says_so(
    nothing_installed: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert superlevel_sources.main([]) == 1
    out, err = capsys.readouterr()
    assert "=>" not in out
    assert "measured: nothing" in out
    assert f"unmeasured: {', '.join(superlevel_sources.SOURCES)}" in out
    assert "NOTHING MEASURED" in err


def test_require_all_fails_on_the_first_unmeasured_source(
    nothing_installed: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert superlevel_sources.main(["--require-all"]) == 1
    assert "DRIFT" in capsys.readouterr().err


def test_a_conclusion_is_printed_only_for_a_source_measured(
    nothing_installed: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """HomCloud stubbed to its gated values: its conclusion appears, the run
    exits 0, and the summary still names the four sources it did not reach."""
    by_grid = {
        str(superlevel_sources.GRIDS[name]): expected
        for name, expected in superlevel_sources.EXPECTED_HOMCLOUD.items()
    }

    def homcloud(grid: list[list[float]]) -> Any:
        return by_grid[str(grid)]

    monkeypatch.setattr(superlevel_sources, "homcloud_superlevel", homcloud)
    assert superlevel_sources.main([]) == 0
    out = capsys.readouterr().out
    assert "=> the caller supplies the death of a class that never dies." in out
    assert "measured: homcloud" in out
    assert "unmeasured: dionysus, oineus, torch_topological, oineus_sublevel" in out
