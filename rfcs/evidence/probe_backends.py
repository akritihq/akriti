#!/usr/bin/env python3
"""Reproduce every measured claim in RFC-0001.

Run:  python rfcs/evidence/probe_backends.py

Sections A.1-A.4 measured 2026-07-29 with gudhi 3.11.0, ripser 0.6.14,
persim 0.3.8, giotto-tda 0.6.2, numpy 2.4.4, scikit-learn 1.8.0,
Python 3.12.11.

A.4's primordial and both-infinite rows and its ground-metric comparison were
added and measured 2026-09-10 with persim 0.3.8, numpy 2.5.1, Python 3.14.6.
The four rows that predate them are unchanged from the 2026-07-29 run.

Section A.5 (RFC-0001 D17) was added and measured 2026-08-06 with gudhi 3.13.0,
ripser 0.6.15, persim 0.3.8, numpy 2.5.1, scikit-learn 1.9.0. giotto-tda is not
installed in that environment, so its A.5 row is unmeasured (RFC-0001 §9.2) and
the script reports it as such rather than skipping it silently.

Clean-room note: giotto-tda is AGPLv3. This script calls its
public API and inspects returned arrays. No giotto source is read, and giotto
source MUST NOT be read while implementing akriti.compat.giotto.
"""

from __future__ import annotations

import argparse
import inspect
import warnings
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

# Deliberately NOT suppressed. An earlier version of this script began with
# warnings.filterwarnings("ignore"), which hid the UserWarning persim raises in
# A.4 and put a false claim into RFC-0001 §9.1 -- that persim failed silently.
# It does not. Blanket-suppressing warnings in a script whose entire purpose is
# to observe third-party behaviour is self-defeating. Warnings are captured and
# reported explicitly below.
warnings.simplefilter("always")

SEED = 0
N = 40
NOISE = 0.05
MAX_EDGE = 4.0
EMPTY_FINITE_WASSERSTEIN = 0.9899494936611666
A4_RTOL = 1e-12
A4_ATOL = 1e-12
DGM1_WARNING = "dgm1 has points with non-finite death times;ignoring those points"
DGM2_WARNING = "dgm2 has points with non-finite death times;ignoring those points"
#: numpy's, not persim's. On a pair of primordial bars persim subtracts one
#: -inf birth from another; this is the only diagnostic emitted on that path
#: (RFC-0001 A.4). Its wording is numpy's and may move; the point measured is
#: that no persim UserWarning accompanies it.
SUBTRACT_WARNING = "invalid value encountered in subtract"
#: A returned float that is not a number, as distinct from a raised exception.
NAN = "nan"


class ProbeDriftError(RuntimeError):
    """Raised when a measured RFC-0001 backend claim has changed."""


def _fail(section: str, message: str) -> None:
    raise ProbeDriftError(f"{section} drift: {message}")


def _require(condition: bool, section: str, message: str) -> None:
    """Fail loudly instead of allowing optimized Python to disable a check."""
    if not condition:
        _fail(section, message)


def _require_same_shape(a: np.ndarray, b: np.ndarray, *, section: str) -> None:
    """Require equal shapes before any comparison can broadcast them."""
    _require(
        a.shape == b.shape,
        section,
        f"shape changed: left={a.shape!r}, right={b.shape!r}",
    )


def _require_array_shape(
    value, *, columns: int, section: str, label: str
) -> np.ndarray:
    """Require a backend diagram array before any row or column access."""
    _require(
        isinstance(value, np.ndarray),
        section,
        f"{label} type changed: {type(value).__name__}, expected ndarray",
    )
    _require(
        value.ndim == 2 and value.shape[1] == columns,
        section,
        f"{label} shape changed: {value.shape!r}, expected (n, {columns})",
    )
    return value


def _require_float32_values(value, *, section: str, label: str) -> None:
    """Require a float64 result whose values are exactly float32-representable."""
    _require(
        isinstance(value, np.ndarray) and value.dtype == np.dtype(np.float64),
        section,
        f"{label} must be a float64 ndarray before its precision can be measured",
    )
    round_trip = value.astype(np.float32).astype(np.float64)
    _require(
        bool(np.array_equal(value, round_trip)),
        section,
        f"{label} values no longer round-trip exactly through float32",
    )


def _require_batch_shape(
    value, *, samples: int, section: str, label: str
) -> np.ndarray:
    """Require a giotto batch before indexing its sample or coordinate axes."""
    _require(
        isinstance(value, np.ndarray),
        section,
        f"{label} type changed: {type(value).__name__}, expected ndarray",
    )
    _require(
        value.ndim == 3 and value.shape[0] == samples and value.shape[2] == 3,
        section,
        f"{label} shape changed: {value.shape!r}, expected ({samples}, n, 3)",
    )
    return value


def _trivial_mask(diagram: np.ndarray) -> np.ndarray:
    """Identify giotto's measured padding representation exactly."""
    return diagram[:, 0] == diagram[:, 1]


def _require_close(
    actual: float,
    expected: float,
    *,
    section: str,
    label: str,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> None:
    _require(
        bool(np.isclose(actual, expected, rtol=rtol, atol=atol)),
        section,
        f"{label} changed: observed={actual!r}, expected={expected!r}",
    )


def _measure_with_warnings(operation, *args):
    """Run a backend operation and return its value with every warning emitted.

    A.4 measures inputs on which persim raises rather than returns, so an
    exception is a measured outcome here and is handed back in the value's
    place instead of propagating.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            value = operation(*args)
        except Exception as error:  # the raise is the datum, not a failure
            value = error
    return value, caught


def _require_warnings(caught, expected, *, section: str, operation: str) -> None:
    """Require the exact (category, message) pairs an operation emitted.

    Category is part of the measurement rather than an implementation detail.
    persim's own UserWarning names a class it dropped; numpy's RuntimeWarning
    on ``-inf - -inf`` is the *only* diagnostic on the rows persim's guard
    cannot see, and telling them apart is what RFC-0001 §9.1 turns on. A row
    expecting no warning must emit none: silence is the measurement there.
    """
    if expected and not caught:
        _fail(section, f"{operation} stopped warning entirely")
    observed = sorted(
        (warning.category.__name__, str(warning.message)) for warning in caught
    )
    wanted = sorted((category.__name__, message) for category, message in expected)
    _require(
        observed == wanted,
        section,
        f"{operation} warnings changed: observed={observed!r}, expected={wanted!r}",
    )


def _require_measurement(observed, expected, *, section: str, label: str) -> None:
    """Compare a measurement against a float, the NAN sentinel, or a raise."""
    if isinstance(expected, type) and issubclass(expected, BaseException):
        _require(
            isinstance(observed, expected),
            section,
            f"{label} no longer raises {expected.__name__}: {observed!r}",
        )
        return
    _require(
        np.isscalar(observed),
        section,
        f"{label} is no longer a scalar result: {observed!r}",
    )
    value = float(observed)
    if expected is NAN:
        _require(np.isnan(value), section, f"{label} is no longer nan: {value!r}")
        return
    if np.isinf(expected):
        _require(value == expected, section, f"{label} changed: {value!r}")
        return
    _require_close(
        value, expected, section=section, label=label, rtol=A4_RTOL, atol=A4_ATOL
    )


def _format_measurement(value) -> str:
    """Render a float or a raised exception for the A.4 table."""
    if isinstance(value, BaseException):
        return type(value).__name__
    return f"{float(value):.4f}"


def _coefficient_carriers(value) -> list[str]:
    """Return top-level or nested names that could carry a coefficient field."""
    carriers: set[str] = set()
    seen: set[int] = set()

    def is_carrier(name: object) -> bool:
        lowered = str(name).lower()
        return (
            "coeff" in lowered
            or lowered == "field"
            or lowered.endswith("_field")
            or lowered == "characteristic"
            or lowered == "prime"
        )

    def visit(current, path: str, depth: int = 0) -> None:
        if depth > 5 or id(current) in seen:
            return
        seen.add(id(current))
        if isinstance(current, dict):
            for name, child in current.items():
                child_path = f"{path}.{name}" if path else str(name)
                if is_carrier(name):
                    carriers.add(child_path)
                visit(child, child_path, depth + 1)
            return
        if isinstance(current, (list, tuple)):
            fields = getattr(type(current), "_fields", ())
            if fields:
                for name, child in zip(fields, current, strict=True):
                    child_path = f"{path}.{name}" if path else str(name)
                    if is_carrier(name):
                        carriers.add(child_path)
                    visit(child, child_path, depth + 1)
            else:
                for index, child in enumerate(current):
                    visit(child, f"{path}[{index}]", depth + 1)
            return

        try:
            attributes = vars(current)
        except TypeError:
            attributes = {}
        for name, child in attributes.items():
            if name.startswith("_"):
                continue
            child_path = f"{path}.{name}" if path else name
            if is_carrier(name):
                carriers.add(child_path)
            visit(child, child_path, depth + 1)

        for name in dir(current):
            if not name.startswith("_") and is_carrier(name):
                carriers.add(f"{path}.{name}" if path else name)

    visit(value, "")
    return sorted(carriers)


def _require_no_coefficient_carriers(value, *, section: str, label: str) -> list[str]:
    """Fail with measured-section context if a returned object carries a field."""
    carriers = _coefficient_carriers(value)
    _require(
        not carriers,
        section,
        f"{label} exposes coefficient fields: {carriers}",
    )
    return carriers


def _parameter_default(callable_obj, name: str, *, section: str, label: str):
    """Read a load-bearing backend default with a section-specific failure."""
    try:
        parameters = inspect.signature(callable_obj).parameters
    except (TypeError, ValueError) as exc:
        _fail(section, f"{label} signature is no longer inspectable: {exc}")
    _require(
        name in parameters,
        section,
        f"{label} parameter {name!r} is missing",
    )
    return parameters[name].default


def _required_attribute(value, name: str, *, section: str, label: str):
    """Read a load-bearing public attribute with a diagnostic on removal."""
    try:
        inspect.getattr_static(value, name)
        return getattr(value, name)
    except AttributeError:
        _fail(section, f"{label} attribute {name!r} is missing")


def sample_circle(n: int, noise: float, rng: np.random.Generator) -> np.ndarray:
    theta = rng.uniform(0, 2 * np.pi, n)
    pts = np.c_[np.cos(theta), np.sin(theta)] + rng.normal(0, noise, (n, 2))
    return np.ascontiguousarray(pts, dtype=np.float64)


def rule(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def _make_check_array_shim(original):
    """Adapt giotto's old keyword only when sklearn's public API requires it."""
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
    """giotto-tda 0.6.2 calls check_array(force_all_finite=...), which
    scikit-learn renamed in 1.6 and removed in 1.8 (RFC-0001 §9.2).

    Translate the kwarg so the rest of the probe can run. This is a local
    workaround at the public-API boundary, not a fix and not a patch we ship.
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-giotto",
        action="store_true",
        help="fail if giotto-tda cannot be imported and measured",
    )
    args = parser.parse_args(argv)
    rng = np.random.default_rng(SEED)
    A = sample_circle(N, NOISE, rng)  # noisy circle: one clear H1 class
    B = rng.normal(0, 1, (N, 2))  # gaussian blob: many short H1 bars

    # ---------------------------------------------------------------- A.1
    rule("A.1  ESSENTIAL BARS — what each backend does with the infinite bar")

    import gudhi

    st = gudhi.RipsComplex(points=A, max_edge_length=MAX_EDGE).create_simplex_tree(
        max_dimension=2
    )
    persistence_rows = st.persistence()
    _require(
        isinstance(persistence_rows, list) and bool(persistence_rows),
        "A.1",
        "GUDHI persistence() no longer returns a nonempty list",
    )
    g0 = _require_array_shape(
        st.persistence_intervals_in_dimension(0),
        columns=2,
        section="A.1",
        label="GUDHI H0",
    )
    g1 = _require_array_shape(
        st.persistence_intervals_in_dimension(1),
        columns=2,
        section="A.1",
        label="GUDHI H1",
    )
    print(
        f"  gudhi   H0={len(g0):3d}  essential={int(np.isinf(g0[:, 1]).sum())}"
        f"  H1={len(g1)}"
    )
    _require(len(g0) == 40, "A.1", f"GUDHI H0 count is {len(g0)}, expected 40")
    _require(
        int(np.isinf(g0[:, 1]).sum()) == 1,
        "A.1",
        "GUDHI essential H0 count changed from 1",
    )
    _require(len(g1) == 2, "A.1", f"GUDHI H1 count is {len(g1)}, expected 2")
    print(f"          persistence() entry form: {persistence_rows[:1]}")

    from ripser import ripser

    ripser_output = ripser(A, maxdim=1, thresh=MAX_EDGE)
    _require(
        isinstance(ripser_output, dict) and "dgms" in ripser_output,
        "A.1",
        "Ripser output is no longer a dict containing 'dgms'",
    )
    dgms = ripser_output["dgms"]
    _require(
        isinstance(dgms, Sequence)
        and not isinstance(dgms, (str, bytes))
        and len(dgms) >= 2,
        "A.1",
        "Ripser 'dgms' no longer contains H0 and H1 diagrams",
    )
    r0, r1 = dgms[0], dgms[1]
    r0 = _require_array_shape(r0, columns=2, section="A.1", label="Ripser H0")
    r1 = _require_array_shape(r1, columns=2, section="A.1", label="Ripser H1")
    print(
        f"  ripser  H0={len(r0):3d}  essential={int(np.isinf(r0[:, 1]).sum())}"
        f"  H1={len(r1)}"
    )
    _require(len(r0) == 40, "A.1", f"Ripser H0 count is {len(r0)}, expected 40")
    _require(
        int(np.isinf(r0[:, 1]).sum()) == 1,
        "A.1",
        "Ripser essential H0 count changed from 1",
    )
    _require(len(r1) == 2, "A.1", f"Ripser H1 count is {len(r1)}, expected 2")

    have_giotto = patch_giotto()
    if args.require_giotto and not have_giotto:
        _fail("A.1", "giotto-tda stopped being importable")
    if have_giotto:
        from gtda.homology import VietorisRipsPersistence

        for iv in (None, np.inf, 99.0):
            vr = VietorisRipsPersistence(homology_dimensions=(0, 1), infinity_values=iv)
            transformed = _require_batch_shape(
                vr.fit_transform(A[None]),
                samples=1,
                section="A.1",
                label="giotto result",
            )
            g = transformed[0]
            h0 = g[g[:, 2] == 0]
            h0_count = len(h0)
            essential_count = int((~np.isfinite(g)).sum())
            h1_count = int((g[:, 2] == 1).sum())
            infinity_value = _required_attribute(
                vr,
                "infinity_values_",
                section="A.1",
                label="giotto estimator",
            )
            print(
                f"  giotto  H0={h0_count:3d}  essential={essential_count}"
                f"  H1={h1_count}"
                f"   infinity_values={iv!r} -> {infinity_value}"
            )
            _require(
                h0_count == 39, "A.1", f"giotto H0 count is {h0_count}, expected 39"
            )
            _require(
                essential_count == 0,
                "A.1",
                f"giotto essential count is {essential_count}, expected 0",
            )
            _require(h1_count == 2, "A.1", f"giotto H1 count is {h1_count}, expected 2")
        print("  => giotto drops the essential class under every setting.")

        # A.1's second half. The loop above holds `reduced_homology` at
        # giotto's default of True and varies `infinity_values`, so it
        # establishes only that `infinity_values` is *not* the cause. Flipping
        # `reduced_homology` shows the cause directly, which is what §5.1's
        # adapter requirement rests on. Counts rather than coordinates, so
        # this row survives the float-level environment sensitivity the
        # fixture capture has.
        for iv, expected_essential in ((None, 1), (np.inf, 1), (99.0, 0)):
            vr = VietorisRipsPersistence(
                homology_dimensions=(0, 1),
                infinity_values=iv,
                reduced_homology=False,
            )
            transformed = _require_batch_shape(
                vr.fit_transform(A[None]),
                samples=1,
                section="A.1",
                label="giotto result (reduced_homology=False)",
            )
            g = transformed[0]
            h0_count = len(g[g[:, 2] == 0])
            essential_count = int((~np.isfinite(g)).sum())
            h1_count = int((g[:, 2] == 1).sum())
            print(
                f"  giotto  H0={h0_count:3d}  essential={essential_count}"
                f"  H1={h1_count}"
                f"   infinity_values={iv!r}  reduced_homology=False"
            )
            _require(
                h0_count == 40,
                "A.1",
                f"giotto H0 count is {h0_count} with reduced_homology=False, "
                "expected 40 -- the essential class should be present",
            )
            _require(
                essential_count == expected_essential,
                "A.1",
                f"giotto essential count is {essential_count} for "
                f"infinity_values={iv!r}, expected {expected_essential}",
            )
            _require(h1_count == 2, "A.1", f"giotto H1 count is {h1_count}, expected 2")
        print("  => reduced_homology decides whether the class exists;")
        print("     infinity_values decides how its death is represented.")

    # ---------------------------------------------------------------- A.2
    rule("A.2  GIOTTO BATCH PADDING — the diagram depends on the batch")

    if have_giotto:
        vr = VietorisRipsPersistence(homology_dimensions=(0, 1))
        solo = {}
        for name, X in (("A", A), ("B", B)):
            transformed = _require_batch_shape(
                vr.fit_transform(X[None]),
                samples=1,
                section="A.2",
                label=f"giotto {name}-alone result",
            )
            solo[name] = transformed[0]
        batched = _require_batch_shape(
            vr.fit_transform(np.stack([A, B])),
            samples=2,
            section="A.2",
            label="giotto batched result",
        )

        expected_solo = {
            "A": (41, 39, 2, 0),
            "B": (50, 39, 11, 0),
        }
        for name, g in solo.items():
            h0_count = int((g[:, 2] == 0).sum())
            h1_count = int((g[:, 2] == 1).sum())
            trivial_count = int(_trivial_mask(g).sum())
            print(
                f"  {name} alone   rows={len(g):3d}  "
                f"H0={h0_count:3d}  H1={h1_count:3d}  "
                f"trivial={trivial_count}"
            )
            _require(
                (len(g), h0_count, h1_count, trivial_count) == expected_solo[name],
                "A.2",
                f"{name} alone counts changed: "
                f"observed={(len(g), h0_count, h1_count, trivial_count)}, "
                f"expected={expected_solo[name]}",
            )

        expected_batched = {
            "A": (50, 39, 11, 9),
            "B": (50, 39, 11, 0),
        }
        for i, name in enumerate("AB"):
            g = batched[i]
            triv = _trivial_mask(g)
            h0_count = int((g[:, 2] == 0).sum())
            h1_count = int((g[:, 2] == 1).sum())
            trivial_count = int(triv.sum())
            print(
                f"  {name} batched rows={len(g):3d}  "
                f"H0={h0_count:3d}  H1={h1_count:3d}  "
                f"trivial={trivial_count}"
            )
            _require(
                (len(g), h0_count, h1_count, trivial_count) == expected_batched[name],
                "A.2",
                f"{name} batched counts changed: "
                f"observed={(len(g), h0_count, h1_count, trivial_count)}, "
                f"expected={expected_batched[name]}",
            )
            if trivial_count:
                print(f"     padding rows look like: {g[triv][0]}")
        print("  => A yields 2 H1 bars alone and 11 batched. Padding is written")
        print("     with a real birth value, so it is indistinguishable from a")
        print("     genuine zero-persistence bar.")

    # ---------------------------------------------------------------- A.3
    rule("A.3  PRECISION AND ORDERING — gudhi vs ripser on identical input")

    print(f"  ripser raw order:\n{r1}")
    print(f"  gudhi  raw order:\n{g1}")
    _require_array_shape(r1, columns=2, section="A.3", label="Ripser H1")
    _require_array_shape(g1, columns=2, section="A.3", label="GUDHI H1")
    _require_same_shape(r1, g1, section="A.3")
    _require(
        r1.dtype == np.dtype(np.float64), "A.3", f"Ripser dtype changed: {r1.dtype}"
    )
    _require(
        g1.dtype == np.dtype(np.float64), "A.3", f"GUDHI dtype changed: {g1.dtype}"
    )
    _require_float32_values(r1, section="A.3", label="Ripser H1")
    raw_agrees = np.allclose(r1, g1, rtol=1e-6, atol=0)
    _require(not bool(raw_agrees), "A.3", "raw row order unexpectedly agrees")
    print(f"  same row order: {raw_agrees}")

    rs = r1[np.lexsort((r1[:, 1], r1[:, 0]))]
    gs = g1[np.lexsort((g1[:, 1], g1[:, 0]))]
    _require_same_shape(rs, gs, section="A.3")
    sorted_agrees = np.allclose(rs, gs, rtol=1e-6, atol=0)
    _require(bool(sorted_agrees), "A.3", "sorted diagrams no longer agree at rtol=1e-6")
    tight_agrees = np.allclose(rs, gs, rtol=1e-12, atol=0)
    _require(
        not bool(tight_agrees),
        "A.3",
        "sorted diagrams unexpectedly agree at rtol=1e-12",
    )
    diff = np.abs(rs - gs).max()
    scale = np.abs(gs).max()
    print(f"  dtypes: ripser={r1.dtype} gudhi={g1.dtype}")
    print(f"  max |diff| after sorting : {diff:.3e}")
    print(f"  float32 eps at this scale: {np.finfo(np.float32).eps * scale:.3e}")
    print(f"  float64 eps at this scale: {np.finfo(np.float64).eps * scale:.3e}")
    print("  => ripser returns float64 arrays holding float32-precision values.")

    # ---------------------------------------------------------------- A.4
    rule("A.4  PERSIM — what it does with each non-finite class")

    import persim

    # One diagram per class of RFC-0001 §9.1's partition, each carrying the
    # same finite bar so the classes are what differ between rows.
    ess_d = np.array([[0.0, np.inf], [0.1, 0.5]])  # (finite, +inf)
    pri_d = np.array([[-np.inf, 0.5], [0.1, 0.5]])  # (-inf, finite)
    pri_e = np.array([[-np.inf, 2.0], [0.1, 0.5]])  # (-inf, finite), moved
    both_d = np.array([[-np.inf, np.inf], [0.1, 0.5]])  # (-inf, +inf)
    fin_d = np.array([[0.0, 1.0], [0.1, 0.5]])  # (finite, finite)
    empty = np.zeros((0, 2))

    dgm1_warned = ((UserWarning, DGM1_WARNING),)
    both_warned = ((UserWarning, DGM1_WARNING), (UserWarning, DGM2_WARNING))
    subtracted = ((RuntimeWarning, SUBTRACT_WARNING),)
    silent = ()

    class A4Case(NamedTuple):
        """One row of RFC-0001 A.4.

        `truth` is the correct distance, stated independently of persim; the
        measured columns are what persim gives. Where they differ the row is
        one of the defects §9.1 is written against.
        """

        label: str
        dgm1: np.ndarray
        dgm2: np.ndarray
        bottleneck: object
        bottleneck_warnings: tuple
        wasserstein: object
        wasserstein_warnings: tuple
        truth: str

    half = float(np.sqrt(0.5))
    cases = [
        A4Case(
            "ess vs itself",
            ess_d,
            ess_d,
            0.0,
            both_warned,
            0.0,
            both_warned,
            "0.0",
        ),
        A4Case(
            "ess vs finite",
            ess_d,
            fin_d,
            0.5,
            dgm1_warned,
            half,
            dgm1_warned,
            "inf",
        ),
        A4Case(
            "both vs itself",
            both_d,
            both_d,
            0.0,
            both_warned,
            0.0,
            both_warned,
            "0.0",
        ),
        A4Case(
            "both vs finite",
            both_d,
            fin_d,
            0.5,
            dgm1_warned,
            half,
            dgm1_warned,
            "inf",
        ),
        A4Case(
            "pri vs itself",
            pri_d,
            pri_d,
            NAN,
            subtracted,
            ValueError,
            silent,
            "0.0",
        ),
        A4Case(
            "pri vs pri'",
            pri_d,
            pri_e,
            NAN,
            subtracted,
            ValueError,
            silent,
            "1.5",
        ),
        A4Case(
            "pri vs finite",
            pri_d,
            fin_d,
            np.inf,
            silent,
            ValueError,
            silent,
            "inf",
        ),
        A4Case(
            "empty vs empty",
            empty,
            empty,
            0.0,
            silent,
            0.0,
            silent,
            "0.0",
        ),
        A4Case(
            "empty vs finite",
            empty,
            fin_d,
            0.5,
            silent,
            EMPTY_FINITE_WASSERSTEIN,
            silent,
            "0.5",
        ),
    ]

    print(
        f"  {'case':<16}{'bottleneck':>12}{'wasserstein':>14}"
        f"{'warns b/w':>12}   correct"
    )
    for case in cases:
        bn, bn_caught = _measure_with_warnings(persim.bottleneck, case.dgm1, case.dgm2)
        wn, wn_caught = _measure_with_warnings(
            persim.wasserstein, case.dgm1, case.dgm2
        )
        _require_measurement(
            bn, case.bottleneck, section="A.4", label=f"{case.label} bottleneck"
        )
        _require_measurement(
            wn, case.wasserstein, section="A.4", label=f"{case.label} wasserstein"
        )
        _require_warnings(
            bn_caught,
            case.bottleneck_warnings,
            section="A.4",
            operation=f"{case.label} bottleneck",
        )
        _require_warnings(
            wn_caught,
            case.wasserstein_warnings,
            section="A.4",
            operation=f"{case.label} wasserstein",
        )
        counts = f"{len(bn_caught)}/{len(wn_caught)}"
        print(
            f"  {case.label:<16}{_format_measurement(bn):>12}"
            f"{_format_measurement(wn):>14}{counts:>12}   {case.truth}"
        )

    print(f"\n  warning text: UserWarning: {DGM1_WARNING}")
    print("  => the guard reads DEATHS. It drops (finite, +inf) and (-inf, +inf)")
    print("     alike -- a plausible finite number, warned about -- and never")
    print("     inspects a birth, so (-inf, finite) reaches the cost matrix and")
    print("     comes back as nan from bottleneck and a ValueError from")
    print("     wasserstein, with no persim warning on either path. The warning")
    print("     also fires twice where persim is right and once where it is")
    print("     wrong, so its presence cannot certify a result and its absence")
    print("     cannot condemn one. core/distances.py must partition on all four")
    print("     classes, not on `essential` alone (RFC-0001 §9.1).")

    # ---------------------------------------------------------------- A.5
    rule("A.5  COEFFICIENT FIELD — is it recoverable from what a backend returns?")

    # RFC-0001 D17. The question is not whether a backend accepts a coefficient
    # field, but whether the object it hands back carries the value it was
    # computed with. If it does not, the adapter cannot record it without
    # being told, and D17 is the reduced_homology question again (§5.1).

    default_g = _parameter_default(
        gudhi.SimplexTree.persistence,
        "homology_coeff_field",
        section="A.5",
        label="GUDHI",
    )
    _require(
        default_g == 11, "A.5", f"GUDHI default coefficient field changed: {default_g}"
    )
    st3 = gudhi.RipsComplex(points=A, max_edge_length=MAX_EDGE).create_simplex_tree(
        max_dimension=2
    )
    res3 = st3.persistence(homology_coeff_field=3)
    carriers_g = _require_no_coefficient_carriers(
        st3, section="A.5", label="GUDHI object"
    )
    _require_no_coefficient_carriers(res3, section="A.5", label="GUDHI returned value")
    _require(
        isinstance(res3, list) and bool(res3),
        "A.5",
        "GUDHI persistence() no longer returns a nonempty list",
    )
    first_gudhi_record = res3[0]
    _require(
        isinstance(first_gudhi_record, tuple)
        and len(first_gudhi_record) == 2
        and isinstance(first_gudhi_record[1], tuple)
        and len(first_gudhi_record[1]) == 2,
        "A.5",
        f"GUDHI persistence record shape changed: {first_gudhi_record!r}",
    )
    print(
        "  gudhi   parameter: persistence(homology_coeff_field=...) "
        f"default={default_g}"
    )
    print(
        f"          returns   : {type(res3).__name__} of "
        f"{type(first_gudhi_record).__name__}, e.g. {first_gudhi_record}"
    )
    print(f"          SimplexTree attrs naming a coeff field: {carriers_g or 'NONE'}")

    default_r = _parameter_default(ripser, "coeff", section="A.5", label="Ripser")
    _require(
        default_r == 2, "A.5", f"Ripser default coefficient field changed: {default_r}"
    )
    out3 = ripser(A, maxdim=1, thresh=MAX_EDGE, coeff=3)
    _require(
        isinstance(out3, dict),
        "A.5",
        f"Ripser returned {type(out3).__name__}, expected dict",
    )
    carriers_r = _require_no_coefficient_carriers(
        out3, section="A.5", label="Ripser returned value"
    )
    print(f"  ripser  parameter: ripser(..., coeff=...) default={default_r}")
    print(f"          returns   : dict keys {sorted(out3.keys())}")
    print(f"          keys naming a coeff field: {carriers_r or 'NONE'}")

    print("  persim  consumes diagrams; computes no homology. No coefficient field.")
    print("  array   no backend. No coefficient field.")
    if have_giotto:
        vr3 = VietorisRipsPersistence(homology_dimensions=(0, 1), coeff=3)
        arr3 = _require_batch_shape(
            vr3.fit_transform(A[None]),
            samples=1,
            section="A.5",
            label="giotto coefficient-field result",
        )
        _require_no_coefficient_carriers(
            arr3, section="A.5", label="giotto returned value"
        )
        estimator_coeff = _required_attribute(
            vr3,
            "coeff",
            section="A.5",
            label="giotto estimator",
        )
        _require(
            estimator_coeff == 3,
            "A.5",
            f"giotto estimator coeff changed: {estimator_coeff}",
        )
        print(
            "  giotto  parameter: VietorisRipsPersistence(coeff=...) "
            f"-> {estimator_coeff}"
        )
        print(
            f"          returns   : {type(arr3).__name__} shape "
            f"{arr3.shape}, dtype {arr3.dtype}"
        )
        print("          the value lives on the estimator, not on the returned array;")
        print("          from_giotto (§11) receives the array.")
    else:
        print("  giotto  NOT MEASURED — not importable in this environment (§9.2).")
        print("          from_giotto (§11) receives the (n_samples, n_bars, 3) array,")
        print("          which has no slot for a coefficient field regardless.")

    print("  => No backend returns the coefficient field it computed with. It is a")
    print("     call parameter on every one of them and is absent from every")
    print("     returned object, so an adapter cannot recover it from its input.")
    print("     The defaults also disagree: gudhi Z/11, ripser Z/2. An unrecorded")
    print("     coeff_field is therefore not conventionally Z/2 -- it is unknown.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
