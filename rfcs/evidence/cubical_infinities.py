#!/usr/bin/env python3
"""Can a filtration take an infinite value, and does RFC-0001 admit the result?

Run:  python rfcs/evidence/cubical_infinities.py [--require-giotto]

RFC-0001 §3.1's I4 read "`births` are all finite and non-`NaN`", justified with
"a class that is never born is not a class". That reads as a definition. It is
not: it is an assumption about the **range of the filtration function**, and it
fails the moment the function takes -inf.

Raised by @corybrunson (tdaverse/phutil) in the comment window
(tdaverse/phutil#61, akritihq/akriti#44) with a reprex in R, using
`TDA::gridDiag(..., sublevel = TRUE)` over grids holding infinities. This
script reproduces the finding Python-side, over GUDHI's own cubical complex,
so the appendix carries a runnable measurement rather than a citation.

Every grid below is filtered **sublevel**, GUDHI offering no other option.
Orientation is not involved in this half of the finding: the trigger is the
value, not the direction.

Three things are measured, and every figure is gated the way
`probe_backends.py` gates Appendix A: a backend release that changes one of
them fails this script rather than reaching a reviewer.

  A. What GUDHI returns for grids holding +inf, -inf, and both, and which of
     those bars I4 and I5 admitted as of 1.2.x. Two grids hold
     *only* +inf, which is the object the first 1.3.0 draft's evidence base
     did not contain: their essential class is born at +inf, and that draft's
     I10 refused it (D27). The same bar is measured by the route a batch of
     images takes -- GUDHI's sklearn `CubicalPersistence` on a batch with one
     all-+inf member -- and from ripser's `lower_star_img`. Two grids hold
     only -inf, the mirror image, and neither GUDHI nor ripser reports a
     `(-inf, -inf)` bar for them. GUDHI reports the `(inf, inf)` bar as an
     unpaired cell, which every `min_persistence` keeps, while the
     zero-persistence pairs that enter at an infinity are discarded at every
     threshold, `-inf` included, where the same pair at a finite value is
     kept. So neither `(-inf, -inf)` nor a paired `(inf, inf)` reaches an
     adapter, and the `(inf, inf)` that does is essential -- the reading D27
     rests on.
     The `peak` grid, `[0, +inf, 0]`, is what §2's definition of `essential`
     is written against: its two `(0.0, inf)` bars are one class that never
     dies and one that died when the +inf cell entered, which
     `cofaces_of_persistence_pairs()` still tells apart and the bars do not.
     Its pair structure is gated against the finite wall `[0, 5, 0]`'s.
     A `SimplexTree` with one vertex entering at +inf beside a finite
     component returns `(inf, inf)` next to a finite death, and the rank of
     H_0(K_t) at a finite t is measured on it: the bar is in no finite K_t,
     which is what §5's statement of the essential count has to account for.
  B. Whether any Python backend RFC-0001 adapts offers a sublevel/superlevel
     switch, which is what decides how narrowly §11's `filtration_direction`
     argument has to be scoped. Every entry point §11 names is inspected, not
     only the `CubicalComplex` section A uses: `from_gudhi` also takes the
     sklearn-compatible form (D20) and a `SimplexTree`, and a switch would
     most likely arrive as a constructor argument on one of those, or as a
     parameter of the `persistence()` call itself, rather than as a
     module-level name -- so the parameters of `persistence()` and
     `compute_persistence()` are read on every GUDHI complex class too, not
     only their names. giotto-tda's homology estimators are inspected
     the same way -- every public class of `gtda.homology`, by signature --
     and `--require-giotto` fails the run if that row is merely unimportable
     and therefore silently unmeasured.
  C. Whether negation -- §11's normalising transform -- is exactly involutive
     in float64, which is what lets §11 normalise rather than condition.

Measured 2026-09-09 with gudhi 3.13.0, numpy 2.5.1, Python 3.14; (B) widened
to the full entry-point list and re-measured 2026-09-10, unchanged in its
conclusion. The all-+inf and all--inf grids, ripser's `lower_star_img` rows and
the giotto row were added and measured 2026-09-13 with gudhi 3.13.0, ripser
0.6.15, numpy 2.5.1, Python 3.14.6, and the sklearn `CubicalPersistence` batch
row, `peak`'s pair structure and `lower_star_img` row, and the `persistence()`
parameter sweep on 2026-09-16 in that environment, the `SimplexTree` row
on 2026-10-01 and the zero-persistence-pair row on 2026-10-02, likewise; the
giotto row in the environment CI's `rfc-evidence` job builds -- giotto-tda
0.6.2, scikit-learn 1.9.1, numpy 2.5.3, Python 3.12.13.

Clean-room note: giotto-tda is AGPLv3. This script inspects the signatures of
giotto's public estimators and calls nothing on them. No giotto source is
read, and giotto source MUST NOT be read while implementing
akriti.compat.giotto (RFC-0001 §9.2).
"""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import struct
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

# One drift gate for the whole appendix, not a second copy of it.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_backends import ProbeDriftError, _require

# Not suppressed, on the rule probe_backends.py states: a script whose purpose
# is to observe third-party behaviour must not filter what that behaviour says.
warnings.simplefilter("always")

INF = float("inf")

# 1-D top-dimensional cells unless noted. `mixed` is the one to test against:
# it carries all three admissible non-finite shapes plus a finite bar at once.
# `all_pos_inf*` and `all_neg_inf*` are D27's: a grid whose every cell is at
# one infinity, so the whole complex is one component born there.
GRIDS: dict[str, Any] = {
    "baseline": [0.0, 1.0, 0.0],
    "peak": [0.0, INF, 0.0],
    "trench": [0.0, -INF, 0.0],
    "both": [-INF, 0.0, INF],
    "two_trenches": [0.0, -INF, 0.0, -INF, 0.0],
    "mixed": [-INF, 0.0, -INF, 1.0, 0.0],
    "all_pos_inf": [INF, INF],
    "all_pos_inf_2d": [[INF, INF], [INF, INF]],
    "all_neg_inf": [-INF, -INF],
    "all_neg_inf_2d": [[-INF, -INF], [-INF, -INF]],
}

# The bars each grid returns, as `(dim, birth, death)` in GUDHI's own order.
# Pinned exactly: every coordinate here is 0, 1 or an infinity.
Bars = list[tuple[int, float, float]]
EXPECTED_BARS: dict[str, Bars] = {
    "baseline": [(0, 0.0, INF), (0, 0.0, 1.0)],
    "peak": [(0, 0.0, INF), (0, 0.0, INF)],
    "trench": [(0, -INF, INF)],
    "both": [(0, -INF, INF)],
    "two_trenches": [(0, -INF, 0.0), (0, -INF, INF)],
    "mixed": [(0, -INF, 0.0), (0, -INF, INF), (0, 0.0, 1.0)],
    "all_pos_inf": [(0, INF, INF)],
    "all_pos_inf_2d": [(0, INF, INF)],
    "all_neg_inf": [(0, -INF, INF)],
    "all_neg_inf_2d": [(0, -INF, INF)],
}

# ripser's lower-star image filtration on the same two D27 shapes, and on
# `peak`: the same bars from a second backend, so neither D27 nor §2's
# definition of `essential` is reopened against one library's choice.
LOWER_STAR_IMAGES: dict[str, list[list[float]]] = {
    "all_pos_inf": [[INF, INF, INF], [INF, INF, INF]],
    "all_neg_inf": [[-INF, -INF, -INF], [-INF, -INF, -INF]],
    "peak": [[0.0, INF, 0.0], [0.0, INF, 0.0]],
}
EXPECTED_LOWER_STAR: dict[str, list[tuple[float, float]]] = {
    "all_pos_inf": [(INF, INF)],
    "all_neg_inf": [(-INF, INF)],
    "peak": [(0.0, INF), (0.0, INF)],
}

# What §2's `essential` means once +inf is a value the filtration takes. On
# `peak` the +inf cell enters at t = +inf and merges the two components, so one
# of the two `(0.0, inf)` bars is a class that *died* there -- and GUDHI writes
# it exactly as it writes the never-dying one. `cofaces_of_persistence_pairs()`
# is where the two are still distinguishable: it returns the paired cells and
# the unpaired (essential) cells separately, as top-cell indices. `wall` is the
# same grid with a finite wall, whose pair structure `peak` must match.
PAIRING_GRIDS: dict[str, tuple[list[float], list[list[int]], list[int]]] = {
    #                 cells              paired (birth, death)   essential
    "peak": ([0.0, INF, 0.0], [[0, 1]], [2]),
    "wall": ([0.0, 5.0, 0.0], [[0, 1]], [2]),
}

# GUDHI's sklearn-compatible form (D20) on a batch with one fully-masked
# member: the `(inf, inf)` bar by the route a batch of images actually takes
# into `from_gudhi`, not only from a `CubicalComplex` built by hand. One
# `(n, 2)` block per member, degree 0 only.
SKLEARN_BATCH: list[list[list[float]]] = [
    [[0.0, 1.0], [0.0, 1.0]],
    [[INF, INF], [INF, INF]],
]
EXPECTED_SKLEARN_BATCH: list[list[tuple[float, float]]] = [
    [(0.0, INF)],
    [(INF, INF)],
]

# GUDHI's reading of the D27 bar, through `min_persistence`. The default `0.0`
# drops every pair of zero persistence and keeps every essential class; `-1.0`
# keeps zero-persistence pairs at finite values -- the second grid's
# `(0.0, 0.0)` -- but not at an infinity (ZERO_PAIR_VALUES below).
READING_GRIDS: dict[str, tuple[list[float], dict[float, Bars]]] = {
    "all_pos_inf": (
        [INF, INF],
        {0.0: [(0, INF, INF)], -1.0: [(0, INF, INF)]},
    ),
    "neg_inf_pair_with_zero": (
        [-INF, -INF, 0.0],
        {0.0: [(0, -INF, INF)], -1.0: [(0, -INF, INF), (0, 0.0, 0.0)]},
    ),
}

# Two vertices joined by an edge, all three at one value `v`: one
# zero-persistence pair at `v`. At `min_persistence` `-1.0` and `-inf` GUDHI
# reports it for finite `v` and drops it -- from `persistence()` and
# `persistence_pairs()` alike -- for `v = +-inf`. `-inf` is the most
# permissive threshold there is, so a pair it drops is dropped at every one;
# `inf - inf` is `nan`, which clears none. So no `(-inf, -inf)` and no
# *paired* `(inf, inf)` reaches an adapter from GUDHI at any threshold, and
# the `(inf, inf)` of an all-+inf grid is an unpaired cell (D27).
ZERO_PAIR_VALUES = (0.0, -INF, INF)
ZERO_PAIR_THRESHOLDS = (-1.0, -INF)
ALL_POS_INF_CELLS = [INF, INF]

# A complex that takes +inf without being *only* +inf: two vertices at 0.0
# joined by an edge at 1.0, and a third vertex entering at +inf, built through
# `SimplexTree`, one of `from_gudhi`'s input forms. Its `(inf, inf)` bar sits
# beside a finite death, which is what §11.2 needs to reach §5's bound in
# `at="max_finite_death"` mode; and its two essential bars exceed the rank of
# H_0(K_t) at a finite t past the last finite value, which is one -- the case
# §5's statement of what the essential count is the rank of is written against.
SIMPLEX_TREE_TOP: list[tuple[list[int], float]] = [
    ([0], 0.0),
    ([1], 0.0),
    ([0, 1], 1.0),
    ([2], INF),
]
EXPECTED_SIMPLEX_TREE_TOP: Bars = [(0, 0.0, INF), (0, 0.0, 1.0), (0, INF, INF)]
# Past the last finite value, so K_t is everything but the +inf vertex.
SIMPLEX_TREE_FINITE_T = 1.5
EXPECTED_RANK_AT_FINITE_T = 1

# What a level-set orientation switch would be called. Fragments are matched
# as substrings; the short words only as a whole `_`-separated token, because
# `sub` and `sign` as substrings match `subsample` and `design` and nothing
# this document is looking for. `upper_star` / `upperstar` is the lower-star
# vocabulary's name for the superlevel case -- the twin `ripser.lower_star_img`
# would gain -- and `upper` alone cannot stand in for it, `upper_bound` being a
# threshold. A switch spelled outside both lists is what this gate cannot see,
# and A.12 says so.
DIRECTION_FRAGMENTS = (
    "level",
    "upper_star",
    "upperstar",
    "direction",
    "orientation",
    "revers",
    "decreas",
    "increas",
    "negat",
    "invert",
    "flip",
    "descend",
    "ascend",
)
DIRECTION_WORDS = ("sign", "sub", "super", "up", "down")
# Names the lists match that are not switches, excluded by name so the gate
# still fires on a new one. `make_filtration_non_decreasing` raises each
# simplex to at least its faces' value -- a monotonicity repair on a filtration
# already chosen, in one direction only.
NOT_A_SWITCH = frozenset({"make_filtration_non_decreasing"})


def _mentions_direction(name: str) -> bool:
    """Whether a parameter, method or module name reads as a direction switch."""
    if name in NOT_A_SWITCH:
        return False
    lowered = name.lower()
    return any(fragment in lowered for fragment in DIRECTION_FRAGMENTS) or any(
        word in lowered.split("_") for word in DIRECTION_WORDS
    )


def verdict_as_of_1_2_x(birth: float, death: float) -> str:
    """I4, I5 and I6's verdict on one bar as of 1.2.x, by rule.

    I4: births finite and non-NaN.  I5: deaths non-NaN, +inf ok, -inf not.
    I6: death >= birth.  Reported per rule so the appendix can say which one
    rejected a bar rather than only that something did.
    """
    if birth != birth or death != death:
        return "rejected (NaN)"
    if birth == -INF or birth == INF:
        return "rejected by I4 (birth not finite)"
    if death == -INF:
        return "rejected by I5 (-inf death)"
    if death < birth:
        return "rejected by I6"
    return "admitted"


def shape(birth: float, death: float) -> str:
    """The bar's class by which of its two coordinates are infinite (§3.1,
    §9.1), spelled so that a `+inf` birth -- the shape D27 is about, and one
    I6+I10's four-shape surface does not contain -- is visible as such."""
    lo = "-inf" if birth == -INF else "+inf" if birth == INF else "finite"
    hi = "+inf" if death == INF else "-inf" if death == -INF else "finite"
    return f"({lo}, {hi})"


def section_a() -> None:
    import gudhi

    print(f"A. GUDHI {gudhi.__version__} cubical persistence, sublevel, Z/2")
    print("   grid -> bars, and the verdict as of 1.2.x on each\n")
    for name, cells in GRIDS.items():
        cc = gudhi.CubicalComplex(
            top_dimensional_cells=np.asarray(cells, dtype=np.float64)
        )
        bars = [
            (int(dim), float(birth), float(death))
            for dim, (birth, death) in cc.persistence(homology_coeff_field=2)
        ]
        _require(
            bars == EXPECTED_BARS[name],
            "A.12",
            f"GUDHI's bars for {name} changed: {bars!r}, "
            f"expected {EXPECTED_BARS[name]!r}",
        )
        print(f"   {name}: {cells}")
        for dim, birth, death in bars:
            print(
                f"      dim {dim}  birth {birth!r:>7}  death {death!r:>7}  "
                f"{shape(birth, death):<18} {verdict_as_of_1_2_x(birth, death)}"
            )
        print()

    from gudhi.sklearn.cubical_persistence import CubicalPersistence

    print("   sklearn CubicalPersistence, a batch with one all-+inf member\n")
    transformed = CubicalPersistence(homology_dimensions=[0]).fit_transform(
        [np.asarray(image, dtype=np.float64) for image in SKLEARN_BATCH]
    )
    for image, per_degree, expected in zip(
        SKLEARN_BATCH, transformed, EXPECTED_SKLEARN_BATCH, strict=True
    ):
        bars = [(float(b), float(d)) for b, d in np.asarray(per_degree[0])]
        _require(
            bars == expected,
            "A.12",
            f"CubicalPersistence's bars for {image} changed: {bars!r}, "
            f"expected {expected!r}",
        )
        print(f"   {image}")
        for birth, death in bars:
            print(
                f"      dim 0  birth {birth!r:>7}  death {death!r:>7}  "
                f"{shape(birth, death):<18} {verdict_as_of_1_2_x(birth, death)}"
            )
    print("   => the same (inf, inf) bar, by the route a batch of images takes.\n")

    print("   GUDHI's reading of (inf, inf), by min_persistence\n")
    for name, (cells, expected_by_threshold) in READING_GRIDS.items():
        cc = gudhi.CubicalComplex(
            top_dimensional_cells=np.asarray(cells, dtype=np.float64)
        )
        for threshold, expected in expected_by_threshold.items():
            bars = [
                (int(dim), float(birth), float(death))
                for dim, (birth, death) in cc.persistence(
                    homology_coeff_field=2, min_persistence=threshold
                )
            ]
            _require(
                bars == expected,
                "A.12",
                f"GUDHI's bars for {name} at min_persistence={threshold} changed: "
                f"{bars!r}, expected {expected!r}",
            )
            print(f"   {name} {cells} min_persistence={threshold:>4}: {bars}")
    print("   => (inf, inf) is kept where (0.0, 0.0) is dropped.\n")

    print("   A zero-persistence pair at v, at min_persistence -1.0 and -inf (D27)\n")
    for threshold in ZERO_PAIR_THRESHOLDS:
        for value in ZERO_PAIR_VALUES:
            tree = gudhi.SimplexTree()
            for simplex in ([0], [1], [0, 1]):
                tree.insert(simplex, filtration=value)
            bars = [
                (int(dim), float(birth), float(death))
                for dim, (birth, death) in tree.persistence(
                    homology_coeff_field=2, min_persistence=threshold
                )
            ]
            paired = [pair for pair in tree.persistence_pairs() if len(pair[1]) > 0]
            finite = value not in (INF, -INF)
            # The essential bar is always there, written `(v, inf)`; the zero
            # pair `(v, v)` joins it only at a finite `v`.
            expected = [(0, value, INF)] + ([(0, value, value)] if finite else [])
            _require(
                bars == expected and bool(paired) == finite,
                "A.12",
                f"GUDHI's zero-persistence pair at {value!r}, min_persistence="
                f"{threshold!r}, changed: bars={bars!r}, pairs={paired!r}; "
                f"expected {expected!r}, paired iff finite",
            )
            print(
                f"   min_persistence={threshold!r:>5} v = {value!r:>5}: {bars}  "
                f"paired simplices: {len(paired)}"
            )
    cc = gudhi.CubicalComplex(
        top_dimensional_cells=np.asarray(ALL_POS_INF_CELLS, dtype=np.float64)
    )
    cc.compute_persistence(homology_coeff_field=2, min_persistence=-1.0)
    regular, essential = cc.cofaces_of_persistence_pairs()
    pairs = [[int(b), int(d)] for b, d in regular[0]] if regular else []
    unpaired = [int(c) for c in essential[0]] if essential else []
    _require(
        pairs == [] and unpaired == [1],
        "A.12",
        f"GUDHI's pair structure for {ALL_POS_INF_CELLS} changed: "
        f"paired={pairs!r}, essential={unpaired!r}",
    )
    print(f"   {ALL_POS_INF_CELLS}: paired {pairs}  essential cells {unpaired}")
    print("   => pairs at an infinity are discarded at every threshold; the")
    print("      (inf, inf) that remains is an unpaired, essential cell.\n")

    print("   GUDHI's pair structure on peak, against a finite wall (§2)\n")
    for name, (cells, expected_pairs, expected_essential) in PAIRING_GRIDS.items():
        cc = gudhi.CubicalComplex(
            top_dimensional_cells=np.asarray(cells, dtype=np.float64)
        )
        cc.compute_persistence(homology_coeff_field=2)
        regular, essential = cc.cofaces_of_persistence_pairs()
        # Degree 0 only: each is a list indexed by degree, empty where a
        # degree has no pairs at all.
        pairs = [[int(b), int(d)] for b, d in regular[0]] if regular else []
        unpaired = [int(c) for c in essential[0]] if essential else []
        bars = [
            (float(b), float(d)) for b, d in cc.persistence_intervals_in_dimension(0)
        ]
        _require(
            pairs == expected_pairs and unpaired == expected_essential,
            "A.12",
            f"GUDHI's pair structure for {name} changed: paired={pairs!r}, "
            f"essential={unpaired!r}, expected {expected_pairs!r} / "
            f"{expected_essential!r}",
        )
        print(f"   {name}: {cells}")
        print(f"      bars {bars}  paired {pairs}  essential cells {unpaired}")
    print("   => the same one paired class and one essential cell either way: on")
    print("      peak, one of the two (0.0, inf) bars died when the +inf cell")
    print("      entered, and is written like the class that never dies.\n")

    print("   GUDHI SimplexTree, a +inf vertex beside a finite component (§5)\n")
    tree = gudhi.SimplexTree()
    for simplex, value in SIMPLEX_TREE_TOP:
        tree.insert(simplex, filtration=value)
    bars = [
        (int(dim), float(birth), float(death))
        for dim, (birth, death) in tree.persistence(homology_coeff_field=2)
    ]
    _require(
        bars == EXPECTED_SIMPLEX_TREE_TOP,
        "A.12",
        f"GUDHI's SimplexTree bars changed: {bars!r}, "
        f"expected {EXPECTED_SIMPLEX_TREE_TOP!r}",
    )
    # K_t itself, built rather than read off the bars, so the rank is measured.
    k_t = gudhi.SimplexTree()
    for simplex, value in SIMPLEX_TREE_TOP:
        if value <= SIMPLEX_TREE_FINITE_T:
            k_t.insert(simplex, filtration=value)
    k_t.compute_persistence(homology_coeff_field=2)
    rank = k_t.betti_numbers()[0]
    essential_count = sum(1 for _, _, death in bars if death == INF)
    _require(
        rank == EXPECTED_RANK_AT_FINITE_T,
        "A.12",
        f"rank of H_0(K_t) at t = {SIMPLEX_TREE_FINITE_T} changed: {rank}, "
        f"expected {EXPECTED_RANK_AT_FINITE_T}",
    )
    print(f"   {SIMPLEX_TREE_TOP}")
    for dim, birth, death in bars:
        print(
            f"      dim {dim}  birth {birth!r:>7}  death {death!r:>7}  "
            f"{shape(birth, death):<18} {verdict_as_of_1_2_x(birth, death)}"
        )
    print(
        f"   => {essential_count} bars with death == +inf; rank of H_0(K_t) at "
        f"t = {SIMPLEX_TREE_FINITE_T} is {rank}. The (inf, inf) bar is in no"
    )
    print("      K_t with finite t, so the essential count is that rank plus one.\n")

    try:
        import ripser
    except ImportError:  # pragma: no cover - reported, not skipped
        print("   ripser: not installed; lower_star_img rows unmeasured\n")
        return
    print(f"   ripser {ripser.__version__} lower_star_img, the same shapes\n")
    for name, image in LOWER_STAR_IMAGES.items():
        dgm = ripser.lower_star_img(np.asarray(image, dtype=np.float64))
        bars = [(float(b), float(d)) for b, d in np.asarray(dgm)]
        _require(
            bars == EXPECTED_LOWER_STAR[name],
            "A.12",
            f"ripser's lower_star_img bars for {name} changed: {bars!r}, "
            f"expected {EXPECTED_LOWER_STAR[name]!r}",
        )
        print(f"   {name}: {image}")
        for birth, death in bars:
            print(
                f"      dim 0  birth {birth!r:>7}  death {death!r:>7}  "
                f"{shape(birth, death):<18} {verdict_as_of_1_2_x(birth, death)}"
            )
        print()
    print("   => the essential class of an all-+inf grid is born at +inf, from")
    print("      both backends, under default arguments. No backend reports a")
    print("      (-inf, -inf) bar for the all--inf grid. RFC-0001 D27.\n")


def section_b(*, require_giotto: bool) -> None:
    print("B. Does any adapted Python backend offer a superlevel switch?\n")
    entry_points: list[tuple[str, list[str]]] = []
    try:
        import gudhi
        import gudhi.sklearn.cubical_persistence
        import gudhi.sklearn.rips_persistence

        # Every GUDHI entry point §11 adapts, not only the one section A uses.
        # A scan of top-level names would miss a constructor argument on a
        # class, and `from_gudhi` takes the sklearn form (D20) as well as a
        # `SimplexTree`, so a probe of `CubicalComplex` alone measures less
        # than §11's claim needs. On the three complex classes the parameters
        # of `persistence()` and `compute_persistence()` are read as well: a
        # `superlevel=` keyword there is spelled inside the vocabulary and
        # sits on the call that computes the diagram, and a scan of method
        # *names* would never see it.
        for label, obj in (
            ("gudhi.CubicalComplex.__init__", gudhi.CubicalComplex.__init__),
            (
                "gudhi.PeriodicCubicalComplex.__init__",
                gudhi.PeriodicCubicalComplex.__init__,
            ),
            (
                "gudhi.sklearn.cubical_persistence.CubicalPersistence.__init__",
                gudhi.sklearn.cubical_persistence.CubicalPersistence.__init__,
            ),
            (
                "gudhi.sklearn.rips_persistence.RipsPersistence.__init__",
                gudhi.sklearn.rips_persistence.RipsPersistence.__init__,
            ),
        ):
            entry_points.append((label, _parameters(obj)))
        for cls in (
            gudhi.CubicalComplex,
            gudhi.PeriodicCubicalComplex,
            gudhi.SimplexTree,
        ):
            for method in ("persistence", "compute_persistence"):
                _require(
                    hasattr(cls, method),
                    "A.12",
                    f"gudhi.{cls.__name__}.{method} is gone; A.12 sweeps it",
                )
                entry_points.append(
                    (
                        f"gudhi.{cls.__name__}.{method}",
                        _parameters(getattr(cls, method)),
                    )
                )
        entry_points.append(
            ("gudhi.SimplexTree methods", _direction_names(gudhi.SimplexTree))
        )
        entry_points.append(("gudhi module names", _direction_names(gudhi)))
    except ImportError:  # pragma: no cover - reported, not skipped
        print("   gudhi: not installed, unmeasured")
    try:
        import ripser

        entry_points.append(("ripser.ripser", _parameters(ripser.ripser)))
        entry_points.append(("ripser.Rips.__init__", _parameters(ripser.Rips.__init__)))
        entry_points.append(("ripser module names", _direction_names(ripser)))
        _require(
            hasattr(ripser, "lower_star_img"),
            "A.12",
            "ripser.lower_star_img is gone; A.12's lower-star row names it",
        )
        entry_points.append(
            ("ripser.lower_star_img", _parameters(ripser.lower_star_img))
        )
    except ImportError:  # pragma: no cover
        print("   ripser: not installed, unmeasured")
    try:
        import persim

        entry_points.append(("persim module names", _direction_names(persim)))
    except ImportError:  # pragma: no cover
        print("   persim: not installed, unmeasured")

    giotto_measured = False
    try:
        import gtda.homology

        # Signatures only. Nothing is fitted, so §9.2's scikit-learn shim is
        # not needed here; `probe_backends.py` carries it for the rows that
        # compute.
        for name in sorted(dir(gtda.homology)):
            cls = getattr(gtda.homology, name)
            if name.startswith("_") or not inspect.isclass(cls):
                continue
            entry_points.append(
                (f"gtda.homology.{name}.__init__", _parameters(cls.__init__))
            )
        entry_points.append(
            ("gtda.homology module names", _direction_names(gtda.homology))
        )
        giotto_measured = True
        # The distribution's own record rather than a module attribute the
        # package is not known to expose.
        print(f"   giotto-tda {importlib.metadata.version('giotto-tda')}: measured")
    except ImportError:
        print("   giotto-tda: not installable here (RFC-0001 §9.2), unmeasured")
    _require(
        giotto_measured or not require_giotto,
        "A.12",
        "--require-giotto was passed and giotto-tda could not be imported",
    )

    for label, names in entry_points:
        print(f"   {label}: {names}")
        hits = [n for n in names if _mentions_direction(n)]
        _require(
            not hits,
            "A.12",
            f"{label} now carries a direction-like parameter {hits!r}; §11's "
            "reopen condition has fired",
        )
    print()


def _direction_names(namespace: object) -> list[str]:
    """Names on a module or class that read as a direction switch."""
    return [name for name in dir(namespace) if _mentions_direction(name)]


def _parameters(callable_: Any) -> list[str]:
    """A callable's parameter names, or a note where they are not inspectable.

    GUDHI's classes are Cython-backed, so a signature is not guaranteed to be
    recoverable; an unreadable one is reported rather than allowed to look
    like an empty one.
    """
    try:
        return [
            name
            for name in inspect.signature(callable_).parameters
            if name not in ("self", "args", "kwargs")
        ]
    except (TypeError, ValueError) as error:  # pragma: no cover - reported
        return [f"<signature not inspectable: {error}>"]


def negate(x: Any) -> Any:
    """RFC-0001 §11's transform, applied once. Spelled as a function so the
    involution below reads as two applications rather than as `-(-x)`, which
    is the same thing and looks like an operator Python does not have.
    """
    return -x


def section_c() -> None:
    print("C. Is negation exactly involutive in float64? (§11's transform)\n")
    probe = [
        0.0,
        -0.0,
        INF,
        -INF,
        1.0,
        -1.0,
        2.0**53,  # 2**53 + 1 is not a float64; it rounds to this
        5e-324,
        1.7976931348623157e308,
        0.1,
        -1e-300,
    ]
    failures = [
        v for v in probe if struct.pack("<d", negate(negate(v))) != struct.pack("<d", v)
    ]
    _require(not failures, "A.12", f"negation is no longer involutive: {failures!r}")
    print(f"   probe values: {len(probe)}")
    print(f"   bit-identical under double negation: {not failures}")
    arr = np.asarray(probe, dtype=np.float64)
    twice = negate(negate(arr))
    _require(
        twice.tobytes() == arr.tobytes(),
        "A.12",
        "array-form negation is no longer bit-identical under two applications",
    )
    print(f"   array form bit-identical: {twice.tobytes() == arr.tobytes()}")
    print()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-giotto",
        action="store_true",
        help="fail if giotto-tda cannot be imported and inspected",
    )
    args = parser.parse_args(argv)
    print(f"numpy {np.__version__}\n")
    try:
        section_a()
        section_b(require_giotto=args.require_giotto)
        section_c()
    except ProbeDriftError as error:
        print(f"\nDRIFT: {error}", file=sys.stderr)
        return 1
    except ImportError as error:
        # Section A is the measurement; without its backend nothing here is
        # evidence, so this is a failed run and not a skipped row. Section B
        # reports its own missing backends as unmeasured, giotto's under
        # `--require-giotto`.
        print(f"\nUNMEASURED: {error.name} is not installed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
