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
     those bars the 1.1.1 invariant table admits. Two grids hold *only* +inf,
     which is the object the first 1.3.0 draft's evidence base did not
     contain: their essential class is born at +inf, and that draft's I10
     refused it (D27). Two hold only -inf, the mirror image, and neither
     GUDHI nor ripser reports a `(-inf, -inf)` bar for them. GUDHI's own
     reading of the `(inf, inf)` bar is measured through `min_persistence`:
     the default drops every zero-persistence pair and keeps this one, so the
     backend calls it essential rather than trivial -- the fact D27's
     resolution rests on.
  B. Whether any Python backend RFC-0001 adapts offers a sublevel/superlevel
     switch, which is what decides how narrowly §11's `filtration_direction`
     argument has to be scoped. Every entry point §11 names is inspected, not
     only the `CubicalComplex` section A uses: `from_gudhi` also takes the
     sklearn-compatible form (D20) and a `SimplexTree`, and a switch would
     most likely arrive as a constructor argument on one of those rather than
     as a module-level name. giotto-tda's homology estimators are inspected
     the same way -- every public class of `gtda.homology`, by signature --
     and `--require-giotto` fails the run if that row is merely unimportable
     and therefore silently unmeasured.
  C. Whether negation -- §11's normalising transform -- is exactly involutive
     in float64, which is what lets §11 normalise rather than condition.

Measured 2026-09-09 with gudhi 3.13.0, numpy 2.5.1, Python 3.14; (B) widened
to the full entry-point list and re-measured 2026-09-10, unchanged in its
conclusion. The all-+inf and all--inf grids, ripser's `lower_star_img` rows and
the giotto row were added and measured 2026-09-13 with gudhi 3.13.0, ripser
0.6.15, numpy 2.5.1, Python 3.14.6; the giotto row in the environment CI's
`rfc-evidence` job builds -- giotto-tda 0.6.2, scikit-learn 1.9.1, numpy 2.5.3,
Python 3.12.13.

Clean-room note: giotto-tda is AGPLv3. This script inspects the signatures of
giotto's public estimators and calls nothing on them. No giotto source is
read, and giotto source MUST NOT be read while implementing
akriti.compat.giotto (RFC-0001 §9.2).
"""

from __future__ import annotations

import argparse
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

# ripser's lower-star image filtration on the same two D27 shapes: the same
# bar from a second backend, so D27 is not reopened against one library's
# choice.
LOWER_STAR_IMAGES: dict[str, list[list[float]]] = {
    "all_pos_inf": [[INF, INF, INF], [INF, INF, INF]],
    "all_neg_inf": [[-INF, -INF, -INF], [-INF, -INF, -INF]],
}
EXPECTED_LOWER_STAR: dict[str, list[tuple[float, float]]] = {
    "all_pos_inf": [(INF, INF)],
    "all_neg_inf": [(-INF, INF)],
}

# GUDHI's reading of the D27 bar, through the one knob that separates trivial
# from essential. `min_persistence=0.0` (the default) drops every pair of zero
# persistence and keeps every essential class; `-1.0` keeps the zero-persistence
# pairs too. `(inf, inf)` survives the default, where the second grid's
# `(0.0, 0.0)` does not -- and no `(-inf, -inf)` appears under either.
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

# Parameter-name fragments a level-set orientation switch would carry. `sub`
# and `super` are deliberately not on the list: they match `subsample` and
# nothing this document is looking for.
DIRECTION_FRAGMENTS = ("level", "direction", "orientation")


def admitted_at_1_1_1(birth: float, death: float) -> str:
    """The 1.1.1 invariant table's verdict on one bar, by rule.

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
    print("   grid -> bars, and the 1.1.1 verdict on each\n")
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
                f"{shape(birth, death):<18} {admitted_at_1_1_1(birth, death)}"
            )
        print()

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
    print("   => (inf, inf) is kept where (0.0, 0.0) is dropped: essential, not")
    print("      trivial, by GUDHI's own filter.\n")

    try:
        import ripser
    except ImportError:  # pragma: no cover - reported, not skipped
        print("   ripser: not installed; lower_star_img rows unmeasured\n")
        return
    print(f"   ripser {ripser.__version__} lower_star_img, the same two shapes\n")
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
                f"{shape(birth, death):<18} {admitted_at_1_1_1(birth, death)}"
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
        # than §11's claim needs.
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
        entry_points.append(
            ("gudhi.SimplexTree methods", _level_names(gudhi.SimplexTree))
        )
        entry_points.append(("gudhi module names", _level_names(gudhi)))
    except ImportError:  # pragma: no cover - reported, not skipped
        print("   gudhi: not installed, unmeasured")
    try:
        import ripser

        entry_points.append(("ripser.ripser", _parameters(ripser.ripser)))
        entry_points.append(("ripser.Rips.__init__", _parameters(ripser.Rips.__init__)))
        entry_points.append(("ripser module names", _level_names(ripser)))
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

        entry_points.append(("persim module names", _level_names(persim)))
    except ImportError:  # pragma: no cover
        print("   persim: not installed, unmeasured")

    giotto_measured = False
    try:
        import gtda
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
        entry_points.append(("gtda.homology module names", _level_names(gtda.homology)))
        giotto_measured = True
        print(f"   giotto-tda {gtda.__version__}: measured")
    except ImportError:
        print("   giotto-tda: not installable here (RFC-0001 §9.2), unmeasured")
    _require(
        giotto_measured or not require_giotto,
        "A.12",
        "--require-giotto was passed and giotto-tda could not be imported",
    )

    for label, names in entry_points:
        print(f"   {label}: {names}")
        hits = [
            n
            for n in names
            if any(fragment in n.lower() for fragment in DIRECTION_FRAGMENTS)
        ]
        _require(
            not hits,
            "A.12",
            f"{label} now carries a direction-like parameter {hits!r}; §11's "
            "reopen condition has fired",
        )
    print()


def _level_names(namespace: object) -> list[str]:
    """Names on a module or class mentioning a level, of either direction."""
    return [name for name in dir(namespace) if "level" in name.lower()]


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
        2.0**53 + 1,
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
