"""Capture real backend output and freeze it for the adapter test suite.

RFC-0001 §11.2 requires adapter round-trip tests to run against **real backend
output**, and accepts a frozen fixture as real provided it was captured from an
actual call to the backend and committed verbatim. It also requires the default
test environment to carry no backend at all (`pyproject.toml`, `[tool.pytest]`
`backend` marker), so a suite that only ran live would not run at all in the
environment CI treats as canonical.

This script is the bridge. It runs the backends, records exactly what they
returned, and writes `tests/fixtures/backend_output.json`. Nothing here is
edited by hand afterwards -- a hand-written array that matches what a backend
is believed to return does not satisfy §11.2 either way.

Usage::

    pip install -e ".[test,rips,alpha,distances]"
    python tools/capture_backend_fixtures.py

The output records the version of every backend involved, so a fixture whose
provenance is doubted can be regenerated and diffed rather than argued about.

**Infinity in JSON.** `json` writes `Infinity` and reads it back as
`float("inf")`. That is a Python extension to the format rather than strict
JSON, and it is deliberate here: RFC-0001 §5 stores essential bars as `inf`,
never a sentinel, and a fixture format that could not hold one would quietly
drop the single most important thing these fixtures exist to carry.
"""

from __future__ import annotations

import json
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

OUT = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "fixtures"
    / "backend_output.json"
)

# The cross-backend comparison in the suite pins the coefficient field on both
# sides (RFC-0001 §9.3): GUDHI defaults to Z/11 and Ripser to Z/2, so an
# unpinned comparison matches bars across two different homology theories.
COEFF_FIELD = 2
MAX_EDGE = 4.0
MAX_DIM = 2


def circle(n: int = 40, noise: float = 0.05, seed: int = 0) -> np.ndarray:
    """The 40-point noisy circle every fixture and the live suite use.

    **Not Appendix A.1's cloud, nor the backend-claims suite's**, which this
    docstring used to claim it was: both of those draw their angles with
    `rng.uniform` (`rfcs/evidence/probe_backends.py`,
    `tests/test_rfc0001_backend_claims.py`) where this uses `linspace`, so the
    same `n`, `noise` and `seed` give a different point set. See
    `tools/capture_giotto_fixture.py` for what the difference is worth.
    """
    rng = np.random.default_rng(seed)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.column_stack([np.cos(theta), np.sin(theta)])
    return pts + rng.normal(0, noise, pts.shape)


def clouds() -> dict[str, np.ndarray]:
    """The point clouds, each chosen for one case in §11.2's minimum list."""
    return {
        # Essential bars in H0 and a long-lived H1 class.
        "circle": circle(),
        # One point: H0 is a single essential bar, H1 is empty. §11.2's
        # "empty in one degree but not another".
        "point": np.zeros((1, 2)),
        # Two identical points plus a third: the duplicate pair gives a bar
        # with birth == death, a *genuine* zero-persistence bar rather than
        # batch padding (§4, §11.1).
        "duplicate": np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]]),
        # Two congruent well-separated pairs: the two H0 bars are identical,
        # so multiplicity has something to survive (§11.2).
        "twin_pairs": np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]]),
    }


def capture_gudhi(points: np.ndarray) -> dict[str, Any]:
    """Both input forms §11 requires `from_gudhi` to accept."""
    import gudhi

    st = gudhi.RipsComplex(points=points, max_edge_length=MAX_EDGE).create_simplex_tree(
        max_dimension=MAX_DIM
    )
    pairs = st.persistence(homology_coeff_field=COEFF_FIELD)
    # GUDHI's own defaults drop two things the suite needs. `min_persistence=0`
    # is documented as "strictly greater than", so a genuine zero-persistence
    # bar -- §11.2 requires one -- never appears; and `persistence_dim_max` is
    # False, which on a 0-dimensional complex skips H0 and returns nothing at
    # all. The second capture asks for both. Neither is less real than the
    # first: both are calls to the backend, recorded verbatim.
    # Only for the small clouds: on the 40-point circle at this edge length
    # the same call returns tens of thousands of zero-persistence pairs, which
    # is a fixture nobody can read and a repository nobody wants.
    full = (
        st.persistence(
            homology_coeff_field=COEFF_FIELD,
            min_persistence=-1.0,
            persistence_dim_max=True,
        )
        if points.shape[0] <= 4
        else None
    )
    return {
        "call": (
            f"RipsComplex(points, max_edge_length={MAX_EDGE})"
            f".create_simplex_tree(max_dimension={MAX_DIM})"
            f".persistence(homology_coeff_field={COEFF_FIELD})"
        ),
        "call_full": (
            f".persistence(homology_coeff_field={COEFF_FIELD}, "
            "min_persistence=-1.0, persistence_dim_max=True)"
        ),
        # list[(dim, (birth, death))], verbatim.
        "persistence": [[int(dim), [float(b), float(d)]] for dim, (b, d) in pairs],
        "persistence_full": (
            None
            if full is None
            else [[int(dim), [float(b), float(d)]] for dim, (b, d) in full]
        ),
        # persistence_intervals_in_dimension(k) -> (n, 2). Degree 2 is
        # requested deliberately: on these clouds it comes back empty, which
        # is §11.2's empty-diagram case in real backend output.
        "intervals": {
            str(k): _array(st.persistence_intervals_in_dimension(k))
            for k in range(MAX_DIM + 1)
        },
    }


def capture_ripser(points: np.ndarray) -> dict[str, Any]:
    """Both input forms §11 requires `from_ripser` to accept."""
    import ripser

    result = ripser.ripser(points, maxdim=1, coeff=COEFF_FIELD)
    rips = ripser.Rips(maxdim=1, coeff=COEFF_FIELD, verbose=False)
    return {
        "call": f"ripser(X, maxdim=1, coeff={COEFF_FIELD})",
        "dgms": [_array(d) for d in result["dgms"]],
        "fit_transform": [_array(d) for d in rips.fit_transform(points)],
    }


def _array(arr: np.ndarray) -> dict[str, Any]:
    """An array recorded with its dtype, so the fixture reconstructs exactly.

    Dtype is part of what the adapters are tested against -- §8's
    `source_dtype` records it and §6.1 requires the adapter to convert it --
    so a fixture that dropped it would lose the fact under test.
    """
    return {
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "data": arr.tolist(),
    }


def main() -> int:
    versions = {"python": platform.python_version(), "numpy": np.__version__}
    for dist in ("gudhi", "ripser", "persim"):
        try:
            versions[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            print(f"{dist} is not installed; cannot capture", file=sys.stderr)
            return 1

    data: dict[str, Any] = {
        "_comment": (
            "Real backend output, captured verbatim by "
            "tools/capture_backend_fixtures.py. RFC-0001 §11.2. Do not edit by "
            "hand: a hand-written array is not real backend output no matter "
            "how closely it matches one."
        ),
        "versions": versions,
        "coeff_field": COEFF_FIELD,
        "clouds": {},
        "gudhi": {},
        "ripser": {},
    }

    for name, points in clouds().items():
        data["clouds"][name] = _array(points)
        data["gudhi"][name] = capture_gudhi(points)
        data["ripser"][name] = capture_ripser(points)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
