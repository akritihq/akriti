"""Focused tests for the A.12 evidence script's direction-switch gate.

RFC-0001 A.12's claim that no adapted Python backend offers a superlevel
switch rests on `_mentions_direction` firing on any name that reads as one
and staying quiet on the names the backends actually use. The gate's
word-boundary rule and its one exclusion are the parts a reader cannot check
from the appendix, so they are pinned here the way `test_probe_backends.py`
pins the sibling script's gates.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _ROOT / "rfcs/evidence/cubical_infinities.py"
_SPEC = importlib.util.spec_from_file_location("cubical_infinities", _SCRIPT_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
cubical_infinities = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cubical_infinities)

_mentions_direction = cubical_infinities._mentions_direction


@pytest.mark.parametrize(
    "name",
    [
        "sublevel",
        "superlevel",
        "filtration_direction",
        "orientation",
        "reverse",
        "reversed_filtration",
        "decreasing",
        "increasing",
        "negate",
        "invert",
        "flip",
        "descending",
        "ascending",
        # The short words fire only as a whole `_`-separated token.
        "sign",
        "sub",
        "super_level",
        "up",
        "top_down",
    ],
)
def test_direction_like_names_fire(name: str) -> None:
    assert _mentions_direction(name)


@pytest.mark.parametrize(
    "name",
    [
        # Every parameter A.12 tabulates, on the entry points it inspects.
        "top_dimensional_cells",
        "vertices",
        "dimensions",
        "perseus_file",
        "periodic_dimensions",
        "homology_dimensions",
        "input_type",
        "homology_coeff_field",
        "min_persistence",
        "n_jobs",
        "threshold",
        "num_collapses",
        "maxdim",
        "thresh",
        "coeff",
        "distance_matrix",
        "do_cocycles",
        "metric",
        "n_perm",
        "verbose",
        "img",
        "infinity_values",
        "reduced_homology",
        "max_edge_length",
        "max_edge_weight",
        "max_entries",
        "collapse_edges",
        "metric_params",
        "directed",
        "filtration",
        # The two substrings the whole-word rule exists for.
        "subsample",
        "design",
        "supersample",
        "upper_bound",
        "download",
    ],
)
def test_backend_parameter_names_stay_quiet(name: str) -> None:
    assert not _mentions_direction(name)


def test_the_one_exclusion_is_by_exact_name() -> None:
    """`make_filtration_non_decreasing` matches `decreas` and is a
    monotonicity repair, not a switch; the exclusion must not widen to a
    prefix, or a real switch spelled near it would go unseen.
    """
    assert not _mentions_direction("make_filtration_non_decreasing")
    assert _mentions_direction("make_filtration_non_decreasing_v2")
    assert _mentions_direction("make_filtration_decreasing")


def test_the_gate_is_case_insensitive() -> None:
    assert _mentions_direction("Sublevel")
    assert _mentions_direction("FILTRATION_DIRECTION")
