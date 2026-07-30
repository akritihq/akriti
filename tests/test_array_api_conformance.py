"""Pin the array-API facts RFC-0001 §3.4 depends on.

The onboarding document requires `core/` to be written against the Python array
API rather than hard-coding NumPy, and PersistenceDiagram is the input to every
function in `core/`. The first draft of RFC-0001 specified `np.ndarray` anyway
and nobody noticed until a reviewer read both documents together.

A written requirement of that kind decays. These tests make it enforceable:
they encode what the standard does and does not provide, so an implementation
that quietly assumes NumPy fails here rather than two years from now.

`array_api_strict` is the conformance reference. It matters because NumPy's own
`__array_namespace__` returns the `numpy` module itself, so every `hasattr(xp,
...)` check passes and NumPy-only functions look portable.
"""

from __future__ import annotations

import numpy as np
import pytest

xps = pytest.importorskip("array_api_strict")


def test_numpy_namespace_is_numpy_itself() -> None:
    """Why we cannot self-check conformance using NumPy.

    This is the trap: `hasattr(xp, "lexsort")` is True under NumPy purely
    because the namespace *is* NumPy. Conformance must be checked against a
    strict implementation.
    """
    xp = np.arange(3.0).__array_namespace__()
    assert xp is np
    assert hasattr(xp, "lexsort")  # passes, and proves nothing
    assert not hasattr(xps, "lexsort")  # the actual standard


def test_operations_the_diagram_type_needs_are_in_the_standard() -> None:
    required = [
        "inf",
        "isinf",
        "isnan",
        "sort",
        "argsort",
        "unique_values",
        "nonzero",
        "concat",
        "take",
        "astype",
        "asarray",
        "isdtype",
        "float64",
        "int32",
        "all",
        "any",
        "equal",
        "where",
        "abs",
    ]
    missing = [name for name in required if not hasattr(xps, name)]
    assert not missing, f"RFC-0001 assumes these are portable, but: {missing}"


def test_canonical_order_composes_from_stable_argsort() -> None:
    """RFC-0001 §7: canonical order without `lexsort`.

    Successive stable sorts, least-significant key first, must reproduce
    lexsort's ordering exactly -- including ties, which are the only case where
    stability is observable.
    """
    rng = np.random.default_rng(0)
    n = 200
    dims_n = rng.integers(0, 3, n)
    births_n = np.round(rng.uniform(0, 1, n), 2)  # deliberate ties
    deaths_n = np.round(births_n + rng.uniform(0, 1, n), 2)

    expected = np.lexsort((deaths_n, births_n, dims_n))

    births, deaths, dims = (xps.asarray(a) for a in (births_n, deaths_n, dims_n))
    order = xps.argsort(deaths, stable=True)
    order = xps.take(order, xps.argsort(xps.take(births, order), stable=True))
    order = xps.take(order, xps.argsort(xps.take(dims, order), stable=True))
    got = np.asarray(order)

    key = lambda idx: list(  # noqa: E731
        zip(dims_n[idx], births_n[idx], deaths_n[idx], strict=True)
    )
    assert key(got) == key(expected)
    assert key(got) == sorted(key(got))
    assert sorted(got.tolist()) == list(range(n))


def test_infinity_is_representable_and_detectable() -> None:
    """RFC-0001 §5: essential bars are `inf`, in whatever namespace."""
    deaths = xps.asarray([1.0, xps.inf, 0.5])
    essential = xps.isinf(deaths)
    assert np.asarray(essential).tolist() == [False, True, False]


def test_dtype_checks_must_not_use_numpy_dtype_objects() -> None:
    """RFC-0001 §6.1: compare against the namespace's dtypes, not NumPy's."""
    a = xps.asarray([1.0, 2.0], dtype=xps.float64)
    assert a.dtype == xps.float64
    assert xps.isdtype(a.dtype, "real floating")

    # The failure mode, and note that the standard's own reference
    # implementation warns about it rather than merely returning False --
    # cross-library dtype comparison is unsupported, not just inadvisable.
    with pytest.warns(UserWarning, match="NumPy native dtype"):
        assert a.dtype != np.float64


def test_filtering_is_data_dependent_and_therefore_eager_only() -> None:
    """RFC-0001 §3.4: `.finite` and `.dim(k)` cannot run under jit.

    Boolean-mask selection works on eager backends and is explicitly not
    guaranteed on lazy or compiled ones, because the output shape depends on
    the values. Documented as a limit rather than discovered later.
    """
    deaths = xps.asarray([1.0, xps.inf, 0.5])
    finite = deaths[~xps.isinf(deaths)]
    assert np.asarray(finite).tolist() == [1.0, 0.5]
    assert finite.shape != deaths.shape  # shape came from the data
