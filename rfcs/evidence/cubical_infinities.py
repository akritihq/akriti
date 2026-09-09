#!/usr/bin/env python3
"""Can a filtration take an infinite value, and does RFC-0001 admit the result?

Run:  python rfcs/evidence/cubical_infinities.py

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

Three things are measured:

  A. What GUDHI returns for grids holding +inf, -inf, and both, and which of
     those bars the 1.1.1 invariant table admits.
  B. Whether any Python backend RFC-0001 adapts offers a sublevel/superlevel
     switch, which is what decides how narrowly §11's `filtration_direction`
     argument has to be scoped.
  C. Whether negation -- §11's normalising transform -- is exactly involutive
     in float64, which is what lets §11 normalise rather than condition.

Measured 2026-09-09 with gudhi 3.13.0, numpy 2.5.1, Python 3.14. ripser and
persim are inspected for (B) where importable; giotto-tda is not installable
alongside a current scikit-learn (§9.2) and its row is reported as unmeasured
rather than skipped silently.

Clean-room note: giotto-tda is AGPLv3. This script imports no giotto source
and reads none.
"""

from __future__ import annotations

import inspect
import struct
import warnings
from typing import Any

import numpy as np

# Not suppressed, on the rule probe_backends.py states: a script whose purpose
# is to observe third-party behaviour must not filter what that behaviour says.
warnings.simplefilter("always")

INF = float("inf")

# 1-D top-dimensional cells. `both_separated` is the interesting one: it
# carries all three admissible non-finite shapes plus a finite bar at once.
GRIDS: dict[str, list[float]] = {
    "baseline": [0.0, 1.0, 0.0],
    "peak": [0.0, INF, 0.0],
    "trench": [0.0, -INF, 0.0],
    "both": [-INF, 0.0, INF],
    "two_trenches": [0.0, -INF, 0.0, -INF, 0.0],
    "mixed": [-INF, 0.0, -INF, 1.0, 0.0],
}


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
    """The bar's class under 1.2.0's I6+I10 surface (§3.1, §9.1)."""
    lo = "-inf" if birth == -INF else "finite"
    hi = "+inf" if death == INF else "finite"
    return f"({lo}, {hi})"


def section_a() -> None:
    import gudhi

    print(f"A. GUDHI {gudhi.__version__} cubical persistence, sublevel, Z/2")
    print("   grid -> bars, and the 1.1.1 verdict on each\n")
    for name, cells in GRIDS.items():
        cc = gudhi.CubicalComplex(
            top_dimensional_cells=np.asarray(cells, dtype=np.float64)
        )
        bars = cc.persistence(homology_coeff_field=2)
        print(f"   {name}: {cells}")
        for dim, (birth, death) in bars:
            print(
                f"      dim {dim}  birth {birth!r:>7}  death {death!r:>7}  "
                f"{shape(birth, death):<18} {admitted_at_1_1_1(birth, death)}"
            )
        print()


def section_b() -> None:
    print("B. Does any adapted Python backend offer a superlevel switch?\n")
    try:
        import gudhi

        params = list(inspect.signature(gudhi.CubicalComplex.__init__).parameters)
        print(f"   gudhi.CubicalComplex.__init__: {params}")
        print(f"   gudhi names containing 'level': {_level_names(gudhi)}")
    except ImportError:  # pragma: no cover - reported, not skipped
        print("   gudhi: not installed, unmeasured")
    try:
        import ripser

        print(f"   ripser.ripser: {list(inspect.signature(ripser.ripser).parameters)}")
        print(f"   ripser names containing 'level': {_level_names(ripser)}")
        print(
            "   ripser.lower_star_img present: "
            f"{hasattr(ripser, 'lower_star_img')} (lower-star, no direction "
            "argument)"
        )
    except ImportError:  # pragma: no cover
        print("   ripser: not installed, unmeasured")
    try:
        import persim

        print(f"   persim names containing 'level': {_level_names(persim)}")
    except ImportError:  # pragma: no cover
        print("   persim: not installed, unmeasured")
    try:
        import gtda  # noqa: F401
    except ImportError:
        print("   giotto-tda: not installable here (RFC-0001 §9.2), unmeasured")
    print()


def _level_names(module: object) -> list[str]:
    return [name for name in dir(module) if "level" in name.lower()]


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
    print(f"   probe values: {len(probe)}")
    print(f"   bit-identical under double negation: {not failures}")
    if failures:
        print(f"   failures: {failures}")
    arr = np.asarray(probe, dtype=np.float64)
    twice = negate(negate(arr))
    print(f"   array form bit-identical: {twice.tobytes() == arr.tobytes()}")
    print()


def main() -> None:
    print(f"numpy {np.__version__}\n")
    section_a()
    section_b()
    section_c()


if __name__ == "__main__":
    main()
