"""Shared fixtures: the frozen backend output the adapter suite runs against.

RFC-0001 §11.2 requires adapter tests to run against real backend output and
accepts a frozen fixture as real, provided it was captured from an actual call
and committed verbatim. `tools/capture_backend_fixtures.py` and
`tools/capture_giotto_fixture.py` are what produced these files; nothing here
constructs a bar by hand.

The reconstruction below restores each array's captured dtype rather than
letting `np.asarray` infer one. Dtype is under test -- §6.1 requires the
adapter to convert it and §8 requires it recorded as `source_dtype` -- so a
loader that quietly produced `float64` everywhere would erase the fact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def _array(spec: dict[str, Any]) -> np.ndarray:
    """Rebuild a captured array exactly: same values, same dtype, same shape."""
    return np.asarray(spec["data"], dtype=spec["dtype"]).reshape(spec["shape"])


@pytest.fixture(scope="session")
def backend_output() -> dict[str, Any]:
    """GUDHI and Ripser output, as captured. See the module docstring."""
    return json.loads((FIXTURES / "backend_output.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def giotto_output() -> dict[str, Any]:
    """giotto-tda output, captured in the pinned environment §9.2 forces."""
    return json.loads((FIXTURES / "giotto_output.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def rebuild_array() -> Any:
    """The array rebuilder, for tests that need to reach into a fixture."""
    return _array


@pytest.fixture
def gudhi_pairs(backend_output: dict[str, Any]) -> Any:
    """`SimplexTree.persistence()` output as GUDHI returns it.

    `list[(dim, (birth, death))]` -- tuples, not lists, because that is what
    the backend returns and §11 fixes the accepted input formats against what
    the backends actually emit.
    """

    def get(cloud: str, *, full: bool = False) -> list[tuple[int, tuple[float, float]]]:
        key = "persistence_full" if full else "persistence"
        pairs = backend_output["gudhi"][cloud][key]
        if pairs is None:
            pytest.skip(f"no {key} captured for {cloud}")
        return [(int(dim), (float(b), float(d))) for dim, (b, d) in pairs]

    return get


@pytest.fixture
def gudhi_intervals(backend_output: dict[str, Any]) -> Any:
    """`persistence_intervals_in_dimension(k)` output: an `(n, 2)` array."""

    def get(cloud: str, k: int) -> np.ndarray:
        return _array(backend_output["gudhi"][cloud]["intervals"][str(k)])

    return get


@pytest.fixture
def ripser_dgms(backend_output: dict[str, Any]) -> Any:
    """`ripser(X)["dgms"]` -- a list of `(n, 2)` arrays, degree by index."""

    def get(cloud: str, *, key: str = "dgms") -> list[np.ndarray]:
        return [_array(spec) for spec in backend_output["ripser"][cloud][key]]

    return get


@pytest.fixture
def giotto_array(giotto_output: dict[str, Any]) -> Any:
    """`VietorisRipsPersistence.fit_transform` output: `(n_samples, n_bars, 3)`.

    Captured with `infinity_values=inf`, the one setting `from_giotto`
    accepts, so essential classes arrive as `inf` rather than as a death at
    `max_edge_length`. This is the capture `essential_bars="faithful"` may be
    asserted over; the sentinel-carrying one is `giotto_default_array`.
    """

    def get(*, reduced: bool, sample: str = "single") -> np.ndarray:
        key = f"reduced_{str(reduced).lower()}"
        return _array(giotto_output["samples"][key][sample])

    return get


@pytest.fixture
def giotto_default_array(giotto_output: dict[str, Any]) -> Any:
    """The same calls with giotto's own `infinity_values=None`. §11.2.

    This array is the one the finite sentinel is actually in: every class
    still alive at `max_edge_length` comes back with a death of exactly that
    cutoff, indistinguishable from a bar that genuinely died there (§5).

    §11.2 requires `from_giotto`'s `infinity_values` refusals to run against
    it rather than against `giotto_array`, "since that array is the one the
    sentinel is actually in; a suite that exercises them only on a
    hand-written array proves the check fires but not that it fires on the
    input it exists for". Nothing may assert `essential_bars="faithful"` over
    this capture -- that label is false of it, which is the whole reason the
    adapter refuses the setting that produced it.
    """

    def get(*, reduced: bool, sample: str = "single") -> np.ndarray:
        key = f"reduced_{str(reduced).lower()}"
        return _array(giotto_output["samples_default_infinity"][key][sample])

    return get
