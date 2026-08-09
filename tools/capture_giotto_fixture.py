"""Capture real giotto-tda output and freeze it for `from_giotto`'s tests.

Separate from `capture_backend_fixtures.py` because it cannot run in the same
environment. RFC-0001 §9.2 records that giotto-tda 0.6.2 does not run on
current scikit-learn, and §11.2 answers what to do about it: a frozen fixture
counts as real backend output, "real" being about provenance rather than about
whether the call happens live in this run.

So this script runs in a **pinned** environment and is not expected to run in
CI or on a developer machine that has the project's own test extras installed:

    uv venv --python 3.11 /tmp/gtdaenv
    /tmp/gtdaenv/bin/pip install "giotto-tda==0.6.2" "scikit-learn<1.8" numpy
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


def circle(n: int = 40, noise: float = 0.05, seed: int = 0) -> np.ndarray:
    """The same 40-point noisy circle Appendix A.1 measured the H0 loss on."""
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
            "in that file. RFC-0001 §9.2, §11.2. Do not edit by hand."
        ),
        "versions": versions,
        "homology_dimensions": list(HOMOLOGY_DIMENSIONS),
        "clouds": {"circle40": _array(ring), "blob40": _array(blob)},
        "samples": {},
    }

    for reduced in (True, False):
        vr = VietorisRipsPersistence(
            homology_dimensions=HOMOLOGY_DIMENSIONS,
            max_edge_length=MAX_EDGE,
            reduced_homology=reduced,
        )
        key = f"reduced_{str(reduced).lower()}"
        data["samples"][key] = {
            "call": (
                "VietorisRipsPersistence(homology_dimensions="
                f"{HOMOLOGY_DIMENSIONS}, max_edge_length={MAX_EDGE}, "
                f"reduced_homology={reduced}).fit_transform(X)"
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
