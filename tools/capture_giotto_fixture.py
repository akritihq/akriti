"""Capture real giotto-tda output and freeze it for `from_giotto`'s tests.

Separate from `capture_backend_fixtures.py` because it cannot run in the same
environment. RFC-0001 §9.2 records that giotto-tda 0.6.2 does not run on
current scikit-learn, and §11.2 answers what to do about it: a frozen fixture
counts as real backend output, "real" being about provenance rather than about
whether the call happens live in this run.

So this script runs in a **pinned** environment and is not expected to run in
CI or on a developer machine that has the project's own test extras installed:

    uv venv --python 3.11 /tmp/gtdaenv
    uv pip install --python /tmp/gtdaenv/bin/python \
        "giotto-tda==0.6.2" "scikit-learn<1.8" "numpy<2"
    /tmp/gtdaenv/bin/python tools/capture_giotto_fixture.py

It writes `tests/fixtures/giotto_output.json`, recording the versions it ran
against. The environment is part of the fixture: giotto's padding behaviour
(§4, Appendix A.2) and its `reduced_homology` H0 loss (§5.1) are properties of
that release, and a fixture that did not say which release produced it could
not be checked later.
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
    Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "giotto_output.json"
)

HOMOLOGY_DIMENSIONS = (0, 1)
MAX_EDGE = 4.0

# RFC-0001 §5, and **both settings are captured, under separate keys.**
#
# giotto's default is `None`, which gives every class still alive at
# `MAX_EDGE` a death of `MAX_EDGE` -- a finite sentinel, indistinguishable
# from a bar that genuinely died there, which §5 refuses outright.
# `from_giotto` refuses such output rather than labelling it
# `essential_bars="faithful"`.
#
# A capture of the accepted setting alone is not enough, and §11.2 says why:
# the `infinity_values` refusal cases "MUST run against real giotto output
# captured with that default, since that array is the one the sentinel is
# actually in". Capturing only `inf` leaves those refusals exercised over an
# array with no sentinel in it, which proves the check fires but not that it
# fires on the input it exists for. So both run, and the fixture keeps them
# apart:
#
#   samples                    infinity_values=inf  -- the accepted input.
#                              `essential_bars="faithful"` may be asserted
#                              over this one and only this one.
#   samples_default_infinity   infinity_values=None -- giotto's own default,
#                              carrying the finite sentinel. The refusal
#                              cases run against this. Nothing may assert
#                              `"faithful"` over it (§11.2).
#
# Anyone re-running this needs the pinned environment (giotto-tda 0.6.2 on
# scikit-learn 1.3.2, named below); §11.2 forbids editing a fixture by hand,
# and that includes substituting `inf` for the sentinel.
INFINITY_VALUES = np.inf
DEFAULT_INFINITY_VALUES = None

# RFC-0001 §11's impossibility check is three-termed, and this is the capture
# that makes the third term testable.
#
# The check refuses a diagram declared `reduced_homology=False` and
# `infinity_values=inf` that carries at least one degree-0 row all of whose
# deaths are finite. "All H0 deaths are finite" is a reduction over an empty
# selection, so it is **vacuously true of a diagram with no H0 rows at all** --
# and `homology_dimensions` excluding 0 is an ordinary giotto request rather
# than a perverse one. A two-termed check scoped only to non-empty diagrams
# refuses such a call, which is the false positive Appendix A.10 measured and
# the reason the clause gained its middle term.
#
# `HOMOLOGY_DIMENSIONS` above cannot supply this control: it is `(0, 1)`, so
# every sample it captures has H0 rows and the vacuous case is untestable
# against it. Hence a third section, captured in the same run and the same
# environment, over the same clouds and the same cutoff, differing in the one
# argument -- exactly the controlled-pair discipline the two `infinity_values`
# sections already follow.
#
# `(1, 2)` rather than `(1,)` so the array carries more than one degree and a
# test cannot pass by accident on a single-column shape. A.10 measures it at
# `(1, 5, 3)`: non-empty, correct, and holding no degree-0 row.
NO_H0_HOMOLOGY_DIMENSIONS = (1, 2)


def circle(n: int = 40, noise: float = 0.05, seed: int = 0) -> np.ndarray:
    """40 points on a noisy unit circle, at **evenly spaced** angles.

    **Not Appendix A.1's cloud, which this docstring used to claim.**
    `rfcs/evidence/probe_backends.py` draws its angles with
    `rng.uniform(0, 2*pi, n)`; the same `n`, `noise` and `seed` therefore give
    a different point set, and a different one in a way that shows: evenly
    spaced points leave one clean H1 class, random angles leave a clump that
    adds a second, short one. Measured with GUDHI on both clouds, at
    `max_edge_length` 4.0 and `inf` alike -- 1 H1 here, 2 there. H0 is
    identical (40 bars, one essential), which is the column §5.1's derivation
    rests on.

    Kept as it is rather than switched to A.1's draw: the fixtures, the live
    suite and `tools/capture_backend_fixtures.py` all use *this* circle, so
    the cross-checks between them are sound and changing it would invalidate
    every committed fixture. What it must not do is claim to be the other one.
    """
    rng = np.random.default_rng(seed)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.column_stack([np.cos(theta), np.sin(theta)])
    return pts + rng.normal(0, noise, pts.shape)


def _array(arr: np.ndarray) -> dict[str, Any]:
    return {"dtype": str(arr.dtype), "shape": list(arr.shape), "data": arr.tolist()}


def main() -> int:
    try:
        from gtda.homology import VietorisRipsPersistence
    except ImportError as exc:  # pragma: no cover - environment-dependent
        print(f"giotto-tda is not importable here: {exc}", file=sys.stderr)
        print(__doc__, file=sys.stderr)
        return 1

    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "giotto-tda": metadata.version("giotto-tda"),
        "scikit-learn": metadata.version("scikit-learn"),
    }

    ring = circle()
    # A second cloud with the *same* point count and a different number of
    # bars: giotto takes a rectangular (n_samples, n_points, n_features)
    # input, so unequal bar counts have to come from topology rather than
    # from cloud size. A gaussian blob has no persistent H1 class, so the
    # batch has to be padded to a common row count -- Appendix A.2's
    # measurement, and the reason §11.1 exists.
    blob = np.random.default_rng(1).normal(0, 1, (40, 2))
    # Deliberately (n_samples, n_points, n_features): giotto's own input shape.
    batch_in = np.stack([ring, blob])

    data: dict[str, Any] = {
        "_comment": (
            "Real giotto-tda output, captured verbatim by "
            "tools/capture_giotto_fixture.py in the pinned environment named "
            "in that file. RFC-0001 §9.2, §11.2. Do not edit by hand. "
            "'samples' was captured with infinity_values=inf, the one setting "
            "from_giotto accepts; 'samples_default_infinity' with giotto's "
            "own default of None, whose finite sentinel the adapter refuses. "
            "essential_bars='faithful' may be asserted over the first only. "
            "'samples_no_h0' was captured with homology_dimensions=(1, 2), "
            "which returns a non-empty array carrying no degree-0 row: the "
            "negative control for §11's three-termed impossibility check "
            "(A.10)."
        ),
        "versions": versions,
        "homology_dimensions": list(HOMOLOGY_DIMENSIONS),
        "clouds": {"circle40": _array(ring), "blob40": _array(blob)},
        "samples": {},
        "samples_default_infinity": {},
        "samples_no_h0": {},
    }

    # Both `infinity_values` settings, into the two top-level keys the module
    # comment describes. The loop is shared rather than written twice so the
    # two captures differ in exactly the one argument and nothing else: same
    # clouds, same cutoff, same homology dimensions, same call ordering.
    for section, infinity_values, homology_dimensions in (
        ("samples", INFINITY_VALUES, HOMOLOGY_DIMENSIONS),
        ("samples_default_infinity", DEFAULT_INFINITY_VALUES, HOMOLOGY_DIMENSIONS),
        ("samples_no_h0", INFINITY_VALUES, NO_H0_HOMOLOGY_DIMENSIONS),
    ):
        for reduced in (True, False):
            vr = VietorisRipsPersistence(
                homology_dimensions=homology_dimensions,
                max_edge_length=MAX_EDGE,
                reduced_homology=reduced,
                infinity_values=infinity_values,
            )
            key = f"reduced_{str(reduced).lower()}"
            data[section][key] = {
                "call": (
                    "VietorisRipsPersistence(homology_dimensions="
                    f"{homology_dimensions}, max_edge_length={MAX_EDGE}, "
                    f"reduced_homology={reduced}, "
                    f"infinity_values={infinity_values}).fit_transform(X)"
                ),
                # n_samples == 1: §11 requires `from_giotto` to return a
                # DiagramBatch of length one here rather than a bare diagram.
                "single": _array(vr.fit_transform(ring[None, :, :])),
                # n_samples == 2 with unequal bar counts: the padded case.
                "batch": _array(vr.fit_transform(batch_in)),
            }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
