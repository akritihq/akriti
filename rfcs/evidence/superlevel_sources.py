#!/usr/bin/env python3
"""How do sources outside the adapted backends write superlevel persistence?

Run:  python rfcs/evidence/superlevel_sources.py [--require-all]

RFC-0001 A.12 measures that none of the four adapted backends offers a
superlevel switch, so superlevel diagrams reach akriti through `from_persim`
and `from_array`. Several Python libraries outside those four do offer one, and
§11's normalisation is only as good as its reading of what they write. Two
things vary between them, and neither is visible in the bars:

  A. How the class that never dies is written: as `-inf` (the superlevel
     filtration's own end), as `+inf` (a convention-free "never"), as a finite
     value substituted upstream, or not at all. §11 negates every birth and
     every finite death, and carries both infinite markers onto §5's `+inf`.
  B. Whether an infinite *input* value survives into the bars, or is replaced
     by a finite number such as the largest double. Oineus does this in
     sublevel mode too, which is a finite value substituted for a `-inf`
     birth -- the case §5 and D28 say no adapted backend shows.

Every figure is gated the way `probe_backends.py` gates Appendix A: a release
that changes one fails this script rather than reaching a reviewer.

Measured 2026-10-02 with dionysus 2.2.3, oineus 0.9.38, torch-topological 0.1.9
(on gudhi 3.11.0 and torch 2.14.1+cpu; it raises `TypeError` on gudhi 3.13.0),
homcloud 5.4.0, numpy 2.5.3, Python 3.12.13. None of these libraries is
installed by CI's `rfc-evidence` job, so A.12's rows from this script are
measured rather than re-run. A run names every source it could not import and
exits 1 if it measured none of them, since a run that checked no figure
reproduces none; `--require-all` makes any unmeasured source a failure.

H0 only, on 1 x n grids, so every library's grid conventions agree.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
import tempfile
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

# One drift gate for the whole appendix, not a second copy of it.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_backends import ProbeDriftError, _require

# Not suppressed, on the rule probe_backends.py states.
warnings.simplefilter("always")

INF = float("inf")
DBL_MAX = float(np.finfo(np.float64).max)

Bars = list[tuple[float, float]]

GRIDS: dict[str, list[list[float]]] = {
    # Two peaks; the lower (3) merges into the higher (5) at the valley 0.
    "two_peaks": [[0.0, 5.0, 0.0, 3.0, 0.0]],
    # A +inf cell, which a superlevel filtration takes first.
    "pos_inf_peak": [[0.0, INF, 0.0, 3.0, 0.0]],
    # -inf cells, which a superlevel filtration takes last.
    "neg_inf_valleys": [[-INF, 5.0, -INF, 3.0, -INF]],
    # Every cell at the superlevel filtration's end.
    "all_neg_inf": [[-INF, -INF, -INF]],
    # A -inf cell under the *sublevel* filtration, for B.
    "neg_inf_trench": [[0.0, -INF, 0.0, 3.0, 0.0]],
}

# Sorted, so the gate does not depend on the order a library emits bars in.
EXPECTED_SUPERLEVEL: dict[str, dict[str, Bars]] = {
    "dionysus": {
        "two_peaks": [(3.0, 0.0), (5.0, INF)],
        "pos_inf_peak": [(3.0, 0.0), (INF, INF)],
        "neg_inf_valleys": [(3.0, -INF), (5.0, INF)],
    },
    "oineus": {
        "two_peaks": [(3.0, 0.0), (5.0, -INF)],
        "pos_inf_peak": [(3.0, 0.0), (DBL_MAX, -INF)],
        "neg_inf_valleys": [(3.0, -INF), (5.0, -INF)],
        "all_neg_inf": [(-INF, -INF)],
    },
    # Already negated: the sublevel diagram of -f, essential deaths set to the
    # image's maximum (the library's own "fake destroyers").
    "torch_topological": {
        "two_peaks": [(-5.0, -0.0), (-3.0, -0.0)],
        "pos_inf_peak": [(-INF, -0.0), (-3.0, -0.0)],
        "neg_inf_valleys": [(-5.0, INF), (-3.0, INF)],
    },
}

# HomCloud returns no death for a class that never dies: finite pairs, and
# essential births separately.
EXPECTED_HOMCLOUD: dict[str, tuple[Bars, list[float]]] = {
    "two_peaks": ([(3.0, 0.0)], [5.0]),
    "pos_inf_peak": ([(3.0, 0.0)], [INF]),
    "neg_inf_valleys": ([], [3.0, 5.0]),
}

# B: Oineus in sublevel mode replaces a -inf birth with -DBL_MAX.
EXPECTED_OINEUS_SUBLEVEL: dict[str, Bars] = {
    "neg_inf_trench": [(-DBL_MAX, INF), (0.0, 3.0)],
    "all_neg_inf": [(-DBL_MAX, INF)],
}


def dionysus_superlevel(grid: list[list[float]]) -> Bars:
    import dionysus as d

    f = d.fill_freudenthal(np.asarray(grid, dtype=np.float32), reverse=True)
    dgms = d.init_diagrams(d.homology_persistence(f), f)
    return sorted((float(p.birth), float(p.death)) for p in dgms[0])


def oineus_diagram(grid: list[list[float]], *, negate: bool) -> Bars:
    import oineus as oin

    dgms = oin.compute_diagrams_ls(
        np.asarray(grid, dtype=np.float64), negate=negate, max_dim=1
    )
    return sorted((float(b), float(e)) for b, e in np.asarray(dgms.in_dimension(0)))


def oineus_superlevel(grid: list[list[float]]) -> Bars:
    return oineus_diagram(grid, negate=True)


def torch_topological_superlevel(grid: list[list[float]]) -> Bars:
    import torch
    from torch_topological.nn import CubicalComplex

    out = CubicalComplex(superlevel=True, dim=2)(
        torch.tensor(grid, dtype=torch.float64)
    )
    h0 = next(info for info in out if info.dimension == 0)
    return sorted((float(b), float(e)) for b, e in h0.diagram.tolist())


def homcloud_superlevel(grid: list[list[float]]) -> tuple[Bars, list[float]]:
    import homcloud.interface as hc

    with tempfile.TemporaryDirectory() as tmp:
        pdlist = hc.PDList.from_bitmap_levelset(
            np.asarray(grid, dtype=np.float64),
            mode="superlevel",
            save_to=str(Path(tmp) / "pd.pdgm"),
        )
        pd = pdlist.dth_diagram(0)
        finite = sorted(zip(pd.births.tolist(), pd.deaths.tolist(), strict=True))
        return finite, sorted(pd.essential_births.tolist())


def normalised(bars: Bars) -> Bars:
    """RFC-0001 §11 under `"superlevel"`: negate every birth and every finite
    death, and carry an infinite death of either sign onto `+inf`."""
    return [(-b, INF if d in (INF, -INF) else -d) for b, d in bars]


def negated(bars: Bars) -> Bars:
    """Plain componentwise negation, for contrast."""
    return [(-b, -d) for b, d in bars]


def verdict(bars: Bars) -> str:
    """§3.1's I6 and I10 on stored bars."""
    if any(b == -INF and d == -INF for b, d in bars):
        return "refused by I10"
    if any(d < b for b, d in bars):
        return "refused by I6"
    return "admitted"


def _version(dist: str) -> str:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


SUPERLEVEL: dict[str, tuple[str, Callable[[list[list[float]]], Bars]]] = {
    "dionysus": ("dionysus fill_freudenthal(reverse=True)", dionysus_superlevel),
    "oineus": ("oineus compute_diagrams_ls(negate=True)", oineus_superlevel),
    "torch_topological": (
        "torch_topological CubicalComplex(superlevel=True)",
        torch_topological_superlevel,
    ),
}

#: Every source a run can measure, in report order. Oineus's sublevel run
#: (section B) is counted apart from its superlevel one.
SOURCES = (*SUPERLEVEL, "homcloud", "oineus_sublevel")


def section_a(*, require_all: bool) -> set[str]:
    """Print section A and return the sources it measured."""
    measured: set[str] = set()
    print("A. Superlevel output, and what §11's normalisation makes of it\n")
    for key, (label, fn) in SUPERLEVEL.items():
        print(f"   {label}")
        for name, expected in EXPECTED_SUPERLEVEL[key].items():
            try:
                bars = fn(GRIDS[name])
            except ImportError as error:
                _require(not require_all, "A.12", f"{label}: {error}")
                print(f"      not installed ({error.name}); unmeasured\n")
                break
            _require(
                bars == expected,
                "A.12",
                f"{label} on {name} changed: {bars!r}, expected {expected!r}",
            )
            measured.add(key)
            # torch_topological has already negated: its output is declared
            # "sublevel", so §11 stores it as it is.
            stored = bars if key == "torch_topological" else normalised(bars)
            print(f"      {name:<16} {GRIDS[name][0]}")
            print(f"         returned  {bars}")
            print(f"         stored    {stored}  -> {verdict(stored)}")
            if key != "torch_topological":
                print(f"         negated   {verdict(negated(bars))}")
        print()

    print("   homcloud from_bitmap_levelset(mode='superlevel')")
    for name, expected in EXPECTED_HOMCLOUD.items():
        try:
            finite, essential = homcloud_superlevel(GRIDS[name])
        except ImportError as error:
            _require(not require_all, "A.12", f"homcloud: {error}")
            print(f"      not installed ({error.name}); unmeasured\n")
            return measured
        _require(
            (finite, essential) == expected,
            "A.12",
            f"homcloud on {name} changed: {(finite, essential)!r}, "
            f"expected {expected!r}",
        )
        print(f"      {name:<16} finite {finite}  essential births {essential}")
    print("   => the caller supplies the death of a class that never dies.\n")
    measured.add("homcloud")
    return measured


def section_b(*, require_all: bool) -> set[str]:
    """Print section B and return the sources it measured."""
    print("B. Oineus in sublevel mode: a -inf birth written as -DBL_MAX\n")
    for name, expected in EXPECTED_OINEUS_SUBLEVEL.items():
        try:
            bars = oineus_diagram(GRIDS[name], negate=False)
        except ImportError as error:
            _require(not require_all, "A.12", f"oineus: {error}")
            print(f"   not installed ({error.name}); unmeasured\n")
            return set()
        _require(
            bars == expected,
            "A.12",
            f"oineus sublevel on {name} changed: {bars!r}, expected {expected!r}",
        )
        print(f"   {name:<16} {GRIDS[name][0]} -> {bars}")
    print("   => a finite birth no adapter can tell was substituted.\n")
    return {"oineus_sublevel"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="fail if any measured library cannot be imported",
    )
    args = parser.parse_args(argv)
    for dist in ("dionysus", "oineus", "torch-topological", "homcloud", "gudhi"):
        print(f"{dist} {_version(dist)}")
    print(f"numpy {np.__version__}\n")
    try:
        measured = section_a(require_all=args.require_all)
        measured |= section_b(require_all=args.require_all)
    except ProbeDriftError as error:
        print(f"\nDRIFT: {error}", file=sys.stderr)
        return 1
    except Exception as error:  # a library failing outright is drift too
        print(f"\nDRIFT: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    unmeasured = [name for name in SOURCES if name not in measured]
    print(f"measured: {', '.join(sorted(measured)) or 'nothing'}")
    print(f"unmeasured: {', '.join(unmeasured) or 'nothing'}")
    if not measured:
        print(
            "\nNOTHING MEASURED: no figure from this script reproduced.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
