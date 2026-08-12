#!/usr/bin/env python3
"""Reproduce every measured claim in RFC-0001.

Run:  python rfcs/evidence/probe_backends.py

Sections A.1-A.4 measured 2026-07-29 with gudhi 3.11.0, ripser 0.6.14,
persim 0.3.8, giotto-tda 0.6.2, numpy 2.4.4, scikit-learn 1.8.0,
Python 3.12.11.

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
    """Run a backend operation and return its value with every warning emitted."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = operation(*args)
    return value, caught


def _require_warnings(
    caught, expected_messages, *, section: str, operation: str
) -> None:
    """Require persim's exact warning category and per-argument messages."""
    if expected_messages:
        _require(
            bool(caught),
            section,
            f"{operation} stopped warning about non-finite death times",
        )
    for index, warning in enumerate(caught, start=1):
        _require(
            warning.category is UserWarning,
            section,
            f"{operation} warning {index} category changed: {warning.category!r}",
        )
    observed_messages = sorted(str(warning.message) for warning in caught)
    expected = sorted(expected_messages)
    _require(
        observed_messages == expected,
        section,
        f"{operation} warning messages changed: "
        f"observed={observed_messages!r}, expected={expected!r}",
    )


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
    rule("A.4  PERSIM — finite distance between infinitely distant diagrams")

    import persim

    inf_d = np.array([[0.0, np.inf], [0.1, 0.5]])
    fin_d = np.array([[0.0, 1.0], [0.1, 0.5]])
    empty = np.zeros((0, 2))

    cases = [
        ("inf vs itself", inf_d, inf_d, "0.0"),
        ("inf vs finite", inf_d, fin_d, "inf"),
        ("empty vs empty", empty, empty, "0.0"),
        ("empty vs finite", empty, fin_d, "0.5"),
    ]
    expected_bottleneck = (0.0, 0.5, 0.0, 0.5)
    expected_wasserstein = (0.0, np.sqrt(0.5), 0.0, EMPTY_FINITE_WASSERSTEIN)
    expected_warning_messages = (
        (DGM1_WARNING, DGM2_WARNING),
        (DGM1_WARNING,),
        (),
        (),
    )
    header = f"  {'case':<18}{'bottleneck':>12}{'wasserstein':>14}"
    print(f"{header}{'warns b/w':>11}   correct bottleneck")
    first_warning_text: str | None = None
    for index, (name, a, b, expected) in enumerate(cases):
        bn, bn_caught = _measure_with_warnings(persim.bottleneck, a, b)
        wn, wn_caught = _measure_with_warnings(persim.wasserstein, a, b)
        _require(
            np.isscalar(bn),
            "A.4",
            f"{name} bottleneck result is no longer scalar: {type(bn).__name__}",
        )
        _require(
            np.isscalar(wn),
            "A.4",
            f"{name} wasserstein result is no longer scalar: {type(wn).__name__}",
        )
        bn_value = float(bn)
        wn_value = float(wn)
        _require_close(
            bn_value,
            expected_bottleneck[index],
            section="A.4",
            label=f"{name} bottleneck",
            rtol=0,
            atol=0,
        )
        _require_close(
            wn_value,
            expected_wasserstein[index],
            section="A.4",
            label=f"{name} wasserstein",
            rtol=A4_RTOL,
            atol=A4_ATOL,
        )
        expected_messages = expected_warning_messages[index]
        _require_warnings(
            bn_caught,
            expected_messages,
            section="A.4",
            operation=f"{name} bottleneck",
        )
        _require_warnings(
            wn_caught,
            expected_messages,
            section="A.4",
            operation=f"{name} wasserstein",
        )
        if expected_messages and first_warning_text is None:
            first_warning_text = str(bn_caught[0].message)
        print(
            f"  {name:<18}{bn_value:>12.4f}{wn_value:>14.4f}"
            f"{len(bn_caught)}/{len(wn_caught):>9}   {expected}"
        )

    _require(
        first_warning_text is not None,
        "A.4",
        "no measured warning text was available for reporting",
    )
    print(f"\n  warning text: UserWarning: {first_warning_text}")
    print("  => persim drops the essential bar and returns a plausible finite")
    print("     number. It DOES warn -- but the warning names the mechanism, not")
    print("     the consequence, and it fires twice on the case persim gets right")
    print("     and once on the case it gets wrong. Presence or absence of the")
    print("     warning cannot be used to detect the failure.")
    print("     core/distances.py must partition on `essential` before")
    print("     delegating (RFC-0001 §9.1).")

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
    carriers_g = _coefficient_carriers(st3)
    returned_carriers_g = _coefficient_carriers(res3)
    _require(
        not carriers_g, "A.5", f"GUDHI object exposes coefficient fields: {carriers_g}"
    )
    _require(
        not returned_carriers_g,
        "A.5",
        f"GUDHI returned value exposes coefficient fields: {returned_carriers_g}",
    )
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
    carriers_r = _coefficient_carriers(out3)
    _require(not carriers_r, "A.5", f"Ripser returned coefficient fields: {carriers_r}")
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
