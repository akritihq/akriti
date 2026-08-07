#!/usr/bin/env python3
"""Untruncated bar counts per persistence diagram — evidence for RFC-0001 D12.

Run from the `classify` repository root, which supplies `lib.datasets`:

    python /path/to/akriti/rfcs/evidence/bar_counts.py

It does not run from this repository: the point clouds live in `classify`, and
the cached diagrams there are truncated to `top_n=50` and diagonal-padded, so
they saturate and measure nothing. These counts are recomputed from the clouds.

Authored by the project lead and reported on PR #10; the figures it produced
are RFC-0001 Appendix A.6. Committed here verbatim in behaviour — only
formatting changed, to satisfy this repository's `ruff` configuration.

Appendix A.6's second table, the .npz/csv/sqlite3 format comparison, was
reported without its script and has none here yet.
"""

from __future__ import annotations

import sys

import gudhi
import numpy as np

sys.path.insert(0, ".")  # run from the classify repo root

from lib.datasets import (
    load_orbit5k_mini,
    load_synthetic_singlecell,
)


def raw_counts(clouds, label: str, n: int = 60) -> np.ndarray:
    per_dim: dict[int, list[int]] = {0: [], 1: [], 2: []}
    total: list[int] = []
    for X in clouds[:n]:
        st = gudhi.AlphaComplex(points=X).create_simplex_tree()
        st.compute_persistence()
        c = {d: len(st.persistence_intervals_in_dimension(d)) for d in (0, 1, 2)}
        for d in (0, 1, 2):
            per_dim[d].append(c[d])
        total.append(sum(c.values()))

    t = np.array(total)
    print(f"\n{label}  ({len(t)} diagrams, {clouds[0].shape[0]} pts each, alpha)")
    print(
        f"  ALL DIMS  min={t.min():5d}  med={int(np.median(t)):5d}  "
        f"p90={int(np.percentile(t, 90)):5d}  max={t.max():5d}"
    )
    for d in (0, 1, 2):
        a = np.array(per_dim[d])
        if a.max() == 0:
            continue
        print(
            f"  H{d}        min={a.min():5d}  med={int(np.median(a)):5d}  "
            f"p90={int(np.percentile(a, 90)):5d}  max={a.max():5d}"
        )
    return t


def main() -> None:
    orbit = load_orbit5k_mini(n_per_class=12, n_pts=500, seed=0)["point_clouds"]
    single = load_synthetic_singlecell()["point_clouds"]

    to = raw_counts(orbit, "orbit5k (alpha)")
    ts = raw_counts(single, "synthetic single-cell (alpha)")

    combined = int(np.median(np.concatenate([to, ts])))
    print(f"\nCombined median: {combined} bars/diagram")
    print(
        "\n=> H0 equals the point count exactly, so bar count scales linearly in\n"
        "   cloud size. RFC-0001 D12, Appendix A.6."
    )


if __name__ == "__main__":
    main()
