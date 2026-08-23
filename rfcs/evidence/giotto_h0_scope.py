#!/usr/bin/env python3
"""Does RFC-0001 §11's `reduced_homology=False` impossibility check false-positive?

Run:  python rfcs/evidence/giotto_h0_scope.py

§11 requires:

    A non-empty diagram declared `reduced_homology=False` and
    `infinity_values=inf` whose H0 deaths are all finite is therefore not
    merely unlikely but impossible: one of the two declarations is false.
    `from_giotto` MUST raise `ValueError` naming both arguments together.

The predicate as written is "all H0 deaths are finite". That is **vacuously
true of a diagram with no H0 rows**, and a giotto transformer constructed with
`homology_dimensions=(1, 2)` -- or `(1,)`, or `(2,)` -- returns exactly such an
array: non-empty, correct, and carrying no degree-0 row for the clause to look
at. Scoping the clause to non-empty diagrams does not exclude it, because the
diagram *is* non-empty; what is empty is its H0 sub-diagram.

This script measures that. It also runs the two controls that decide whether
the clause is salvageable rather than merely wrong:

  * the H0-**present** control (`homology_dimensions=(0, 1)`), where the
    essential bar is there and the check must not fire; and
  * the genuinely-impossible **positive control** -- real giotto output with 40
    H0 bars, every death finite, produced under a finite `max_edge_length` and
    giotto's own `infinity_values=None` -- which is the array the clause exists
    to reject.

Input is Appendix A.1's cloud, constructed exactly as
`rfcs/evidence/probe_backends.py` constructs it: 40 points sampled uniformly on
the unit circle with Gaussian noise sigma = 0.05, numpy `default_rng` seed 0.

Environments. giotto-tda 0.6.2 pins `scikit-learn==1.3.2` and calls
`check_array(force_all_finite=...)`, which scikit-learn removed in 1.8 (§9.2).
akriti requires numpy >= 2.0 (D6), which scikit-learn 1.3.2 does not support.
No single default environment holds both, so this script runs in either half
and says which half it got:

    # pinned -- giotto measured live, adapter half unavailable
    uv venv --python 3.11 ./envs/giotto
    uv pip install --python ./envs/giotto/bin/python \
        "giotto-tda==0.6.2" "scikit-learn==1.3.2" "numpy==1.26.4"

    # modern -- both halves live, giotto reached through the same
    # public-API shim rfcs/evidence/probe_backends.py already installs
    uv venv --python 3.11 ./envs/giotto-modern
    uv pip install --python ./envs/giotto-modern/bin/python --no-deps \
        "giotto-tda==0.6.2"
    uv pip install --python ./envs/giotto-modern/bin/python \
        "scikit-learn==1.8.0" giotto-ph pyflagser igraph plotly joblib

`--dump-arrays DIR` / `--adapter-only DIR` split the two halves across two
interpreters when that is preferred to the shim.

Clean-room note: giotto-tda is AGPLv3. This script calls its public API and
inspects returned arrays. No giotto source is read, and giotto source MUST NOT
be read while implementing akriti.compat.giotto.
"""

from __future__ import annotations

import argparse
import inspect
import platform
import sys
import warnings
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path

import numpy as np

# Not suppressed, for the reason probe_backends.py gives at length: a script
# whose purpose is to observe third-party behaviour must not hide the
# third-party behaviour. `from_giotto` warns about trivial rows (§11.1) and
# that warning is part of what is being measured.
warnings.simplefilter("always")

SEED = 0
N = 40
NOISE = 0.05

# A.1's giotto calls pass no `max_edge_length`, so giotto's own default of
# `inf` applies and `infinity_values=inf` is the faithful setting (§11).
# The positive control needs the opposite: a finite cutoff, so that the
# essential class is alive at the end of the filtration and giotto's
# `infinity_values=None` default writes a finite sentinel over it (§5).
POSITIVE_CONTROL_MAX_EDGE = 4.0

ENV_PACKAGES = (
    "giotto-tda",
    "scikit-learn",
    "numpy",
    "scipy",
    "joblib",
    "giotto-ph",
    "pyflagser",
    "igraph",
    "plotly",
    "threadpoolctl",
    "akriti",
)


class ProbeDriftError(RuntimeError):
    """Raised when a measured RFC-0001 claim in this script has changed."""


def _fail(section: str, message: str) -> None:
    raise ProbeDriftError(f"{section} drift: {message}")


def _require(condition: bool, section: str, message: str) -> None:
    if not condition:
        _fail(section, message)


def rule(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def sample_circle(n: int, noise: float, rng: np.random.Generator) -> np.ndarray:
    """Appendix A.1's cloud. Copied verbatim from probe_backends.py."""
    theta = rng.uniform(0, 2 * np.pi, n)
    pts = np.c_[np.cos(theta), np.sin(theta)] + rng.normal(0, noise, (n, 2))
    return np.ascontiguousarray(pts, dtype=np.float64)


def _make_check_array_shim(original):
    """Adapt giotto's old keyword only when sklearn's public API requires it.

    Identical in behaviour to probe_backends.py's shim of the same name, and
    kept in step with it deliberately: if the two diverged, this script and
    the appendix's own reproduction script would be measuring different
    giotto.
    """
    parameters = inspect.signature(original).parameters
    supports_new = "ensure_all_finite" in parameters
    supports_old = "force_all_finite" in parameters

    def shim(*args, **kwargs):
        if supports_new and not supports_old and "force_all_finite" in kwargs:
            old_value = kwargs.pop("force_all_finite")
            if (
                "ensure_all_finite" in kwargs
                and kwargs["ensure_all_finite"] != old_value
            ):
                raise TypeError(
                    "force_all_finite and ensure_all_finite specify conflicting values"
                )
            kwargs["ensure_all_finite"] = old_value
        return original(*args, **kwargs)

    return shim


def patch_giotto() -> bool:
    """Translate the removed keyword so the rest of the probe can run (§9.2).

    A local workaround at the public-API boundary, not a fix and not a patch
    this project ships.
    """
    import sklearn
    import sklearn.utils

    try:
        import gtda.utils.validation as gval
    except ImportError:
        return False

    original = sklearn.utils.check_array
    shim = _make_check_array_shim(original)
    sklearn.utils.check_array = shim
    gval.check_array = shim
    parameters = inspect.signature(original).parameters
    translation = (
        "translated force_all_finite -> ensure_all_finite"
        if "ensure_all_finite" in parameters and "force_all_finite" not in parameters
        else "preserved sklearn check_array keyword compatibility"
    )
    print(f"  [shim] scikit-learn {sklearn.__version__}: {translation} for giotto-tda")
    return True


def print_environment() -> None:
    rule("ENVIRONMENT")
    print(f"  CPython {platform.python_version()}  ({sys.executable})")
    for package in ENV_PACKAGES:
        try:
            print(f"  {package} {metadata.version(package)}")
        except metadata.PackageNotFoundError:
            print(f"  {package} NOT INSTALLED")


def describe(label: str, arr: np.ndarray) -> dict[str, object]:
    """Report one giotto transform result. Returns the facts F2 turns on."""
    _require(
        arr.ndim == 3 and arr.shape[2] == 3,
        "F2",
        f"{label}: giotto no longer returns (n_samples, n_bars, 3); got {arr.shape}",
    )
    sample = arr[0]
    degrees = sorted({float(value) for value in sample[:, 2].tolist()})
    h0 = sample[sample[:, 2] == 0]
    facts = {
        "shape": tuple(int(size) for size in arr.shape),
        "degrees": degrees,
        "n_rows": int(sample.shape[0]),
        "nonempty": bool(sample.shape[0] > 0),
        "h0_rows": int(h0.shape[0]),
        "h0_infinite_deaths": int(np.isinf(h0[:, 1]).sum()) if h0.shape[0] else 0,
    }
    # "All H0 deaths are finite" -- §11's predicate, spelled the way the
    # clause spells it. `np.all` of an empty selection is True, which is the
    # whole finding.
    facts["all_h0_deaths_finite"] = bool(np.all(np.isfinite(h0[:, 1])))
    facts["clause_fires"] = bool(facts["nonempty"] and facts["all_h0_deaths_finite"])
    print(f"  {label}")
    print(
        f"    shape={facts['shape']}  rows={facts['n_rows']}  "
        f"non-empty={facts['nonempty']}  degrees={facts['degrees']}"
    )
    print(
        f"    H0 rows={facts['h0_rows']}  "
        f"H0 deaths at inf={facts['h0_infinite_deaths']}  "
        f"all H0 deaths finite={facts['all_h0_deaths_finite']}"
    )
    print(f"    §11 predicate fires (MUST raise ValueError) = {facts['clause_fires']}")
    for row in sample[:3].tolist():
        print(f"      row {row}")
    if facts["n_rows"] > 3:
        print(f"      ... {facts['n_rows'] - 3} more rows")
    return facts


def measure_giotto(cloud: np.ndarray) -> dict[str, np.ndarray]:
    """Every giotto call F2 needs, keyed by the name the report uses."""
    from gtda.homology import VietorisRipsPersistence

    arrays: dict[str, np.ndarray] = {}
    rule("F2  DOES A CORRECT GIOTTO ARRAY TRIP §11's IMPOSSIBILITY CLAUSE?")
    print("  All calls: reduced_homology=False, infinity_values=numpy.inf,")
    print("  no max_edge_length (giotto's own default of inf applies).\n")
    for name, dims in (
        ("hd_1_2", (1, 2)),
        ("hd_0_1", (0, 1)),
        ("hd_1", (1,)),
        ("hd_2", (2,)),
    ):
        vr = VietorisRipsPersistence(
            homology_dimensions=dims,
            reduced_homology=False,
            infinity_values=np.inf,
        )
        arr = vr.fit_transform(cloud[None])
        arrays[name] = arr
        label = f"homology_dimensions={dims}"
        if dims == (0, 1):
            label += "   [control -- giotto's own default]"
        facts = describe(label, arr)
        print(f"    estimator infinity_values_ = {vr.infinity_values_!r}\n")

        if dims == (0, 1):
            _require(
                facts["h0_rows"] == 40 and facts["h0_infinite_deaths"] == 1,
                "F2",
                "the H0-present control changed: expected 40 H0 rows and one "
                f"death at inf, got {facts['h0_rows']} and "
                f"{facts['h0_infinite_deaths']} (A.1)",
            )
            _require(
                not facts["clause_fires"],
                "F2",
                "the H0-present control now trips §11's clause, which would "
                "make the clause wrong about A.1's own row",
            )
        else:
            _require(
                facts["h0_rows"] == 0,
                "F2",
                f"{label} unexpectedly carries {facts['h0_rows']} degree-0 rows",
            )
            _require(
                facts["nonempty"],
                "F2",
                f"{label} returned an empty diagram; §11 already scopes the "
                "clause away from those, so the finding would not bite",
            )
            _require(
                facts["clause_fires"],
                "F2",
                f"{label} no longer trips §11's clause -- re-read the finding",
            )

    # The positive control. Real giotto output that the clause is *right*
    # about: 40 H0 bars, none of them essential, because a finite cutoff plus
    # giotto's `infinity_values=None` default wrote a finite sentinel over the
    # class that never dies (§5, §11). Declaring this array
    # `infinity_values=inf` is exactly the false declaration §11 describes.
    rule("F2  POSITIVE CONTROL — the array §11's clause exists to reject")
    vr = VietorisRipsPersistence(
        homology_dimensions=(0, 1),
        max_edge_length=POSITIVE_CONTROL_MAX_EDGE,
        reduced_homology=False,
        infinity_values=None,
    )
    arr = vr.fit_transform(cloud[None])
    arrays["impossible"] = arr
    facts = describe(
        f"homology_dimensions=(0, 1), max_edge_length={POSITIVE_CONTROL_MAX_EDGE}, "
        "infinity_values=None",
        arr,
    )
    print(f"    estimator infinity_values_ = {vr.infinity_values_!r}")
    print(
        "    -> declaring this array infinity_values=inf is false, and §11 is "
        "right to refuse it."
    )
    _require(
        facts["h0_rows"] == 40 and facts["h0_infinite_deaths"] == 0,
        "F2",
        "the positive control changed: expected 40 H0 rows and no death at "
        f"inf, got {facts['h0_rows']} and {facts['h0_infinite_deaths']}",
    )
    _require(
        facts["clause_fires"],
        "F2",
        "the positive control no longer trips §11's clause",
    )
    return arrays


def run_adapter(arrays: dict[str, np.ndarray]) -> bool:
    """Put every measured array through `from_giotto`. Returns False if it cannot."""
    rule("F2  WHAT `from_giotto` DOES WITH THESE ARRAYS TODAY")
    try:
        from akriti.diagrams import from_giotto
    except ImportError as error:
        print(f"  akriti NOT IMPORTABLE: {error}")
        return False

    # akriti reaches an array through `__array_namespace__`, which numpy grew
    # in the main namespace only at 2.0 (D6), and the pinned environment §9.2
    # forces on giotto is numpy 1.26.4. Probe that boundary once, on an array
    # nothing else depends on, rather than letting the resulting ImportError
    # arrive five times inside the loop below dressed as a measurement: an
    # environment that cannot run the adapter is not an adapter that raised.
    try:
        from_giotto(
            np.zeros((1, 1, 3), dtype=np.float64),
            reduced_homology=True,
            infinity_values=np.inf,
        )
    except ImportError as error:
        print(f"  akriti CANNOT ACCEPT THIS ENVIRONMENT'S ARRAYS: {error}")
        print(
            "  The giotto half above stands; the adapter half needs numpy >= 2.0. "
            "Re-run with --dump-arrays DIR here and --adapter-only DIR there, "
            "or use the modern environment this module's docstring describes."
        )
        return False
    except Exception:
        # Only ImportError is an environment fact. Anything else this probe
        # array provokes is the adapter working, and is measured below.
        pass

    cases = (
        ("hd_1_2", "homology_dimensions=(1, 2)", "no H0 rows -- MUST NOT raise"),
        ("hd_1", "homology_dimensions=(1,)", "no H0 rows -- MUST NOT raise"),
        ("hd_2", "homology_dimensions=(2,)", "no H0 rows -- MUST NOT raise"),
        (
            "hd_0_1",
            "homology_dimensions=(0, 1)",
            "H0 essential present -- MUST NOT raise",
        ),
        ("impossible", "positive control", "§11: MUST raise ValueError"),
    )
    raised: dict[str, str | None] = {}
    for key, label, expectation in cases:
        arr = arrays.get(key)
        if arr is None:
            print(f"  {label:32s} array not available")
            continue
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                batch = from_giotto(arr, reduced_homology=False, infinity_values=np.inf)
            diagram = batch[0]
            degrees = sorted({int(value) for value in diagram.dims})
            raised[key] = None
            print(f"  {label:32s} {expectation}")
            print(
                f"      from_giotto -> NO EXCEPTION; batch len={len(batch)}  "
                f"bars={int(diagram.births.shape[0])}  degrees={degrees}  "
                f"essential_bars={diagram.meta.provenance.get('essential_bars')!r}"
            )
            for warning in caught:
                print(f"      warned: {warning.category.__name__}: {warning.message}")
        # The exception IS the measurement here, so every type is caught and
        # reported rather than any one being singled out.
        except Exception as error:
            raised[key] = f"{type(error).__name__}: {error}"
            print(f"  {label:32s} {expectation}")
            print(f"      from_giotto RAISED {raised[key]}")

    print()
    if raised.get("impossible") is None and "impossible" in raised:
        print(
            "  => §11's impossibility check is NOT IMPLEMENTED. The positive "
            "control -- an array the clause is unambiguously right about -- "
            "passes through untouched."
        )
        print(
            "     So F2 is a specification finding, not yet a code defect: the "
            "clause must be re-scoped before it is implemented, or the "
            "implementation will reject correct input."
        )
    elif "impossible" in raised:
        print("  => the impossibility check IS implemented.")
        for key, _label, _expectation in cases[:4]:
            if raised.get(key) is not None:
                print(
                    f"     FALSE POSITIVE on {key}: it raised on correct input "
                    f"-- {raised[key]}"
                )
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dump-arrays",
        metavar="DIR",
        help="write every measured giotto array to DIR as .npy and stop",
    )
    parser.add_argument(
        "--adapter-only",
        metavar="DIR",
        help="skip giotto; replay arrays written by --dump-arrays through from_giotto",
    )
    parser.add_argument(
        "--require-adapter",
        action="store_true",
        help="fail if akriti cannot be imported and measured",
    )
    args = parser.parse_args(argv)

    print_environment()

    if args.adapter_only:
        directory = Path(args.adapter_only)
        arrays = {path.stem: np.load(path) for path in sorted(directory.glob("*.npy"))}
        print(f"\n  replaying {len(arrays)} arrays from {directory}")
        if not run_adapter(arrays) and args.require_adapter:
            _fail("F2", "akriti stopped being importable")
        return 0

    rng = np.random.default_rng(SEED)
    A = sample_circle(N, NOISE, rng)

    if not patch_giotto():
        print("\n  giotto-tda NOT IMPORTABLE — nothing to measure (§9.2).")
        return 1
    arrays = measure_giotto(A)

    if args.dump_arrays:
        directory = Path(args.dump_arrays)
        directory.mkdir(parents=True, exist_ok=True)
        for name, arr in arrays.items():
            np.save(directory / f"{name}.npy", arr)
        print(f"\n  wrote {len(arrays)} arrays to {directory}")
        print(f"  replay with:  python giotto_h0_scope.py --adapter-only {directory}")
        return 0

    if not run_adapter(arrays) and args.require_adapter:
        _fail("F2", "akriti stopped being importable")

    rule("VERDICT")
    print("  §11's predicate, as written, is 'all H0 deaths are finite'.")
    print("  numpy.all over an empty selection is True, so a non-empty giotto")
    print("  array with no degree-0 rows satisfies it vacuously. Three ordinary")
    print("  transformer configurations produce exactly that. The clause must")
    print("  be scoped to diagrams that HAVE an H0 sub-diagram:")
    print()
    print("      raise ValueError only when the diagram is non-empty AND")
    print("      carries at least one degree-0 row AND every degree-0 death")
    print("      is finite.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
