"""Pin the array-API facts RFC-0001 §3.3 depends on.

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

from akriti.diagrams import DiagramBatch, PersistenceDiagram  # noqa: E402


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
    """RFC-0001 §3.3: `.finite` and `.dim(k)` cannot run under jit.

    Boolean-mask selection works on eager backends and is explicitly not
    guaranteed on lazy or compiled ones, because the output shape depends on
    the values. Documented as a limit rather than discovered later.
    """
    deaths = xps.asarray([1.0, xps.inf, 0.5])
    finite = deaths[~xps.isinf(deaths)]
    assert np.asarray(finite).tolist() == [1.0, 0.5]
    assert finite.shape != deaths.shape  # shape came from the data


# -- the diagram types themselves, under the strict namespace --------------
#
# Everything above pins facts about the *standard*. These build a
# PersistenceDiagram and a DiagramBatch out of strict arrays and drive them
# through construct / validate / slice / canonicalise / compare / hash, which
# is what RFC-0001 §3.3 means by running the diagram suite against
# array_api_strict. Without them the array-API requirement is enforced by a
# module that never constructs the type it exists to protect.


def strict(
    dims: list[int], births: list[float], deaths: list[float]
) -> PersistenceDiagram:
    return PersistenceDiagram(
        dims=xps.asarray(dims, dtype=xps.int32),
        births=xps.asarray(births, dtype=xps.float64),
        deaths=xps.asarray(deaths, dtype=xps.float64),
    )


def test_diagram_constructs_and_validates_under_strict() -> None:
    """§3.1's six reductions must run on a namespace that is not NumPy."""
    d = strict([1, 0], [0.5, 0.0], [1.5, xps.inf])
    assert d.n_bars == 2
    assert d.xp is xps

    with pytest.raises(ValueError, match="I6"):
        strict([0], [1.0], [0.5])
    with pytest.raises(ValueError, match="I2"):
        PersistenceDiagram(
            dims=xps.asarray([0], dtype=xps.int32),
            births=xps.asarray([0.0], dtype=xps.float32),
            deaths=xps.asarray([1.0], dtype=xps.float64),
        )


def test_accessors_and_canonical_order_under_strict() -> None:
    d = strict([1, 0, 0], [0.5, 0.25, 0.0], [1.5, 0.75, xps.inf])

    assert np.asarray(d.essential).tolist() == [False, False, True]
    assert np.asarray(d.dimensions).tolist() == [0, 1]
    assert d.dim(0).n_bars == 2
    assert d.finite.n_bars == 2

    c = d.canonical()
    assert np.asarray(c.dims).tolist() == [0, 0, 1]
    assert np.asarray(c.births).tolist() == [0.0, 0.25, 0.5]


def test_equality_and_finitize_under_strict() -> None:
    """`==` reaches `xp.equal`; `finitize` reaches `where` and `asarray`."""
    a = strict([0, 0], [0.0, 0.25], [xps.inf, 0.75])
    b = strict([0, 0], [0.25, 0.0], [0.75, xps.inf])  # same multiset, reordered
    assert a == b
    assert a.allclose(b)

    dropped = a.finitize(at="drop")
    assert dropped.n_bars == 1
    assert dropped.meta.provenance["essential_bars_dropped"] == 1

    filled = a.finitize(at=2.0)
    assert np.asarray(filled.deaths).tolist() == [2.0, 0.75]


def test_batch_construction_slicing_and_empty_path_under_strict() -> None:
    """Covers `from_diagrams`' empty branch, which builds arrays itself.

    `xp.asarray([], dtype=xp.int32)` and the `int64` offsets are constructed
    by `core.py` rather than handed in by a caller, so they are the one place
    the module's own array-creation calls are exercised.
    """
    a = strict([0], [0.0], [1.0])
    b = strict([1, 1], [0.0, 0.5], [1.0, 2.0])
    batch = DiagramBatch.from_diagrams([a, b])

    assert len(batch) == 2
    assert np.asarray(batch.bar_counts).tolist() == [1, 2]
    assert batch[0] == a
    assert batch[1] == b
    assert batch.canonical() == batch

    empty = DiagramBatch.from_diagrams([], xp=xps)
    assert len(empty) == 0
    assert empty.dims.dtype == xps.int32
    assert empty.offsets.dtype == xps.int64
    assert empty.content_hash != batch.content_hash


def test_cross_namespace_comparison_raises_under_strict() -> None:
    """§6.3: a NumPy diagram and a strict diagram are not comparable."""
    numpy_backed = PersistenceDiagram(
        dims=np.asarray([0], dtype=np.int32),
        births=np.asarray([0.0], dtype=np.float64),
        deaths=np.asarray([1.0], dtype=np.float64),
    )
    with pytest.raises(ValueError, match="across array namespaces"):
        _ = strict([0], [0.0], [1.0]) == numpy_backed
