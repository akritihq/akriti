"""`akriti.diagrams.adapters` against RFC-0001 §11, §11.1, §5.1, §8.

Written from the RFC rather than from the implementation, and before it: every
assertion here cites the clause it enforces, so a failure says which
requirement broke rather than which line changed.

**These tests need no backend installed.** They run against the frozen output
in `tests/fixtures/`, which §11.2 admits as real backend output because it was
captured from an actual call and committed verbatim. That matters more than
convenience: the default test environment carries no backend by design
(`pyproject.toml`), so a suite that only ran live would not run at all in the
environment CI treats as canonical. Live-backend tests are in
`test_rfc0001_adapters_live.py`, marked `backend`.

The clauses under test, in one place:

- §11 signatures, and the two deviations on `from_giotto`.
- §11: every adapter validates against §3.1, populates `backend`,
  `backend_version` and `provenance`, preserves backend row order, and never
  finitizes, sorts or deduplicates.
- §11 (D17): `from_gudhi` and `from_ripser` record `coeff_field` and
  `provenance["coeff_field_source"]` in both directions.
- §11.1: the three `strip_padding` modes.
- §5.1: `from_giotto` derives `essential_bars` from `reduced_homology`, records
  the flag in `params`, and fabricates nothing.
- §8: reserved provenance keys, and JSON-representable values.
- §3.1: I6 violations are the adapter's to clamp, and it must warn.
"""

from __future__ import annotations

import inspect
import json
import math
import warnings
from importlib import metadata
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

import akriti.diagrams.adapters as adapters_module
from akriti.diagrams import DiagramBatch, PersistenceDiagram
from akriti.diagrams.adapters import (
    from_array,
    from_giotto,
    from_gudhi,
    from_persim,
    from_ripser,
)

# `tools/capture_giotto_fixture.py`'s `MAX_EDGE`, which is the value giotto's
# default `infinity_values=None` writes into the death column of every class
# still alive at the cutoff. Asserted against the fixture's own recorded call
# string rather than trusted, so a recapture at a different cutoff fails here
# instead of silently weakening C1's regression tests.
_GIOTTO_MAX_EDGE = 4.0


def test_adapters_are_reachable_from_the_package_surface() -> None:
    """§1: `akriti.diagrams` is the interchange layer, and "diagrams in" is
    the primary user path. An adapter reachable only by its module path makes
    the entry point to this library an implementation detail."""
    import akriti.diagrams as pkg

    for name in (
        "from_array",
        "from_giotto",
        "from_gudhi",
        "from_persim",
        "from_ripser",
    ):
        assert name in pkg.__all__
        assert getattr(pkg, name) is globals()[name]


def installed_version(dist: str) -> str | None:
    """What an adapter is expected to record for `backend_version`.

    An adapter cannot import the backend -- §3.3 keeps `adapters.py` to the
    standard library -- so the version comes from installed distribution
    metadata, and is `None` where the distribution is absent. The fixtures
    make the absent case ordinary rather than exotic: these tests run in an
    environment with no backend installed at all.
    """
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


class _FakeNumpy:
    """Small import target for the lazy NumPy fallback tests."""

    def __init__(self, version: str, namespace: Any) -> None:
        self.__version__ = version
        self._namespace = namespace

    def empty(self, size: int) -> Any:
        assert size == 0
        namespace = self._namespace

        class _Probe:
            def __array_namespace__(self) -> Any:
                return namespace

        return _Probe()


def _patch_numpy_import(
    monkeypatch: pytest.MonkeyPatch,
    *,
    version: str = "2.0.0",
    namespace: Any | None = None,
) -> Any:
    """Install a deterministic fake NumPy module and distribution record."""
    if namespace is None:
        namespace = object()
    fake = _FakeNumpy(version, namespace)
    monkeypatch.setattr(adapters_module, "import_module", lambda name: fake)
    monkeypatch.setattr(adapters_module.metadata, "version", lambda name: version)
    return namespace


def test_numpy_rows_missing_top_level_import_names_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§3.3: a missing lazy dependency gives an actionable install hint."""
    missing = ModuleNotFoundError("No module named 'numpy'", name="numpy")
    monkeypatch.setattr(
        adapters_module, "import_module", lambda name: (_ for _ in ()).throw(missing)
    )

    with pytest.raises(ImportError, match=r"akriti\[numpy\]") as exc_info:
        adapters_module._namespace_for_rows()

    assert exc_info.value.__cause__ is missing


@pytest.mark.parametrize("version", ["1.26.4", "1.0", "2.0rc1"])
def test_numpy_rows_rejects_below_floor_and_same_floor_prerelease(
    monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    """§3.3/D6: NumPy must be >=2.0, including release ordering."""
    _patch_numpy_import(monkeypatch, version=version)

    with pytest.raises(ImportError, match=r"numpy.*2\.0.*akriti\[numpy\]"):
        adapters_module._namespace_for_rows()


@pytest.mark.parametrize("metadata_error", [metadata.PackageNotFoundError, ValueError])
def test_numpy_rows_rejects_missing_or_unparseable_distribution_metadata(
    monkeypatch: pytest.MonkeyPatch, metadata_error: type[Exception]
) -> None:
    """§3.3: broken distribution metadata is an actionable dependency error."""
    fake = _FakeNumpy("2.0.0", object())
    monkeypatch.setattr(adapters_module, "import_module", lambda name: fake)

    def broken_metadata(name: str) -> str:
        raise metadata_error("numpy metadata is unavailable")

    monkeypatch.setattr(adapters_module.metadata, "version", broken_metadata)

    with pytest.raises(ImportError, match=r"numpy.*metadata.*akriti\[numpy\]"):
        adapters_module._namespace_for_rows()


def test_numpy_rows_rejects_an_unparseable_version_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§3.3: a non-version distribution record cannot satisfy the floor."""
    fake = _FakeNumpy("development", object())
    monkeypatch.setattr(adapters_module, "import_module", lambda name: fake)
    monkeypatch.setattr(adapters_module.metadata, "version", lambda name: "development")

    with pytest.raises(ImportError, match=r"parse.*numpy.*akriti\[numpy\]"):
        adapters_module._namespace_for_rows()


def test_numpy_rows_keeps_the_namespace_feature_probe_after_version_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§3.3/D6: valid metadata cannot stand in for the required array API."""
    fake = type(
        "FakeNumpyWithoutArrayAPI", (), {"empty": lambda self, size: object()}
    )()
    monkeypatch.setattr(adapters_module, "import_module", lambda name: fake)
    monkeypatch.setattr(adapters_module.metadata, "version", lambda name: "2.0.0")

    with pytest.raises(ImportError, match=r"array API.*akriti\[numpy\]"):
        adapters_module._namespace_for_rows()


def test_numpy_rows_propagates_transitive_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§3.3: a NumPy subdependency failure must not be laundered."""
    missing = ModuleNotFoundError(
        "No module named 'numpy.core._multiarray_umath'",
        name="numpy.core._multiarray_umath",
    )
    monkeypatch.setattr(
        adapters_module, "import_module", lambda name: (_ for _ in ()).throw(missing)
    )

    with pytest.raises(ModuleNotFoundError) as exc_info:
        adapters_module._namespace_for_rows()

    assert exc_info.value is missing


@pytest.mark.parametrize(
    "version",
    ["2.0", "2.0.0", "2.0.0.post1.dev1", "2.1rc1", "1!2.0", "2.0+local"],
)
def test_numpy_rows_accepts_supported_pep440_versions(
    monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    """§3.3/D6: valid releases at or above the floor reach the namespace."""
    namespace = _patch_numpy_import(monkeypatch, version=version)

    assert adapters_module._namespace_for_rows() is namespace


def assert_json_representable(mapping: Any) -> None:
    """§8: every `params` and `provenance` value must survive `meta.json`.

    A NumPy scalar is the specific hazard -- it formats identically to a
    Python one in a repr and raises only at `save()`, arbitrarily far from the
    adapter that recorded it (§8, §10.2).
    """
    json.dumps(dict(mapping))
    for key, value in mapping.items():
        assert not isinstance(value, np.generic), (
            f"provenance[{key!r}] is a NumPy scalar; §8 requires a Python "
            "type, since §10.2 stores this mapping as JSON"
        )


# ---------------------------------------------------------------------------
# from_gudhi -- §11's two accepted input forms
# ---------------------------------------------------------------------------


def test_from_gudhi_accepts_the_persistence_list_form(gudhi_pairs: Any) -> None:
    """§11: `persistence()` -> `list[(dim, (b, d))]`, carrying every degree."""
    pairs = gudhi_pairs("circle")

    d = from_gudhi(pairs)

    assert isinstance(d, PersistenceDiagram)
    assert d.n_bars == len(pairs)
    assert [int(x) for x in d.dims] == [dim for dim, _ in pairs]
    assert [float(x) for x in d.births] == [b for _, (b, _) in pairs]
    assert [float(x) for x in d.deaths] == [d_ for _, (_, d_) in pairs]


def test_from_gudhi_accepts_the_intervals_array_form(gudhi_intervals: Any) -> None:
    """§11: `persistence_intervals_in_dimension(k)` -> `(n, 2)`, degree stated."""
    intervals = gudhi_intervals("circle", 1)

    d = from_gudhi(intervals, dim=1)

    assert d.n_bars == intervals.shape[0]
    assert all(int(x) == 1 for x in d.dims)
    assert [float(x) for x in d.births] == list(intervals[:, 0])


def test_from_gudhi_rejects_a_two_column_array_without_a_degree(
    gudhi_intervals: Any,
) -> None:
    """An `(n, 2)` array carries no degree; guessing one would invent data."""
    intervals = gudhi_intervals("circle", 1)

    with pytest.raises(TypeError, match="dim"):
        from_gudhi(intervals)


def test_from_gudhi_rejects_a_three_column_array() -> None:
    """§11: GUDHI's array form is only persistence intervals, `(n, 2)`."""
    with pytest.raises(ValueError, match=r"only.*\(n, 2\)"):
        from_gudhi(np.zeros((1, 3)), dim=0)


def test_from_gudhi_rejects_degree_with_the_degree_carrying_list(
    gudhi_pairs: Any,
) -> None:
    with pytest.raises(TypeError, match="already carries a degree"):
        from_gudhi(gudhi_pairs("circle"), dim=0)


def test_from_gudhi_preserves_infinite_deaths(gudhi_pairs: Any) -> None:
    """§5, §5.1: GUDHI is faithful and the adapter keeps `inf` as `inf`."""
    pairs = gudhi_pairs("circle")
    expected = sum(1 for _, (_, death) in pairs if math.isinf(death))

    d = from_gudhi(pairs)

    assert expected == 1, "the fixture should carry exactly one essential bar"
    assert int(np.sum(np.asarray(d.essential))) == expected


def test_from_gudhi_records_essential_bars_as_faithful(gudhi_pairs: Any) -> None:
    """§5.1, §8, §11: GUDHI loses nothing, and both keys say so together."""
    d = from_gudhi(gudhi_pairs("circle"))

    assert d.meta.provenance["essential_bars"] == "faithful"
    assert d.meta.provenance["essential_bars_source"] == "faithful"


def test_from_gudhi_populates_backend_identity(gudhi_pairs: Any) -> None:
    """§8, §11: `backend` and `backend_version` are the adapter's to record."""
    d = from_gudhi(gudhi_pairs("circle"))

    assert d.meta.backend == "gudhi"
    assert d.meta.backend_version == installed_version("gudhi")


def test_from_gudhi_preserves_backend_row_order(gudhi_pairs: Any) -> None:
    """§7, §11: adapters preserve backend row order and never sort.

    GUDHI's own output on this fixture is not in canonical order -- it leads
    with an H1 bar and follows with three H0 bars -- so a sorting adapter
    changes the answer visibly rather than coincidentally.

    The comparison is against `canonical()`'s *rows*, not against `==`: §6.3
    makes equality order-insensitive, so `d == d.canonical()` holds for every
    diagram and would assert nothing here.
    """
    pairs = gudhi_pairs("duplicate", full=True)

    d = from_gudhi(pairs)

    assert [int(x) for x in d.dims] == [1, 0, 0, 0], "fixture order changed"
    assert [int(x) for x in d.canonical().dims] != [int(x) for x in d.dims], (
        "backend row order was silently canonicalised"
    )


def test_from_gudhi_keeps_a_genuine_zero_persistence_bar(gudhi_pairs: Any) -> None:
    """§11.2: a bar with `birth == death` is data, not padding, and stays."""
    d = from_gudhi(gudhi_pairs("duplicate", full=True))

    zero = [
        (float(b), float(x))
        for b, x in zip(d.births, d.deaths, strict=True)
        if float(b) == float(x)
    ]
    assert (0.0, 0.0) in zero


def test_from_gudhi_keeps_repeated_bars(gudhi_pairs: Any) -> None:
    """§11.2: multiplicity survives -- adapters never deduplicate."""
    pairs = gudhi_pairs("twin_pairs")

    d = from_gudhi(pairs)

    rows = [
        (int(k), float(b), float(x))
        for k, b, x in zip(d.dims, d.births, d.deaths, strict=True)
    ]
    assert rows.count((0, 0.0, 1.0)) == 2, "identical finite bars were collapsed"
    assert rows.count((0, 0.0, math.inf)) == 2, (
        "identical essential bars were collapsed"
    )


def test_from_gudhi_accepts_an_empty_persistence_list(gudhi_pairs: Any) -> None:
    """§11.2: an empty diagram is a diagram. GUDHI really returns `[]` here."""
    pairs = gudhi_pairs("point")

    d = from_gudhi(pairs)

    assert pairs == [], "the fixture should be empty"
    assert d.n_bars == 0
    assert d.meta.backend == "gudhi"


def test_from_gudhi_accepts_an_empty_intervals_array(gudhi_intervals: Any) -> None:
    """The array form of the same case: a real `(0, 2)` array from GUDHI."""
    intervals = gudhi_intervals("circle", 2)

    d = from_gudhi(intervals, dim=2)

    assert intervals.shape == (0, 2), "the fixture should be empty"
    assert d.n_bars == 0


def test_from_gudhi_is_empty_in_one_degree_and_not_another(
    gudhi_intervals: Any,
) -> None:
    """§11.2's mixed case, from one GUDHI call: H1 has a bar, H2 has none."""
    d1 = from_gudhi(gudhi_intervals("circle", 1), dim=1)
    d2 = from_gudhi(gudhi_intervals("circle", 2), dim=2)

    assert d1.n_bars > 0
    assert d2.n_bars == 0


# ---------------------------------------------------------------------------
# D17 -- the coefficient field, both directions, both backends (§9.3, §11)
# ---------------------------------------------------------------------------


def test_from_gudhi_records_the_callers_coefficient_field(gudhi_pairs: Any) -> None:
    """§11: a stated field is recorded as the caller's."""
    d = from_gudhi(gudhi_pairs("circle"), coeff_field=2)

    assert d.meta.coeff_field == 2
    assert d.meta.provenance["coeff_field_source"] == "caller"


def test_from_gudhi_records_gudhis_default_coefficient_field(
    gudhi_pairs: Any,
) -> None:
    """§9.3, §11: GUDHI computes over Z/11 unless told otherwise, and the
    adapter records that as an assumption rather than leaving it silent."""
    d = from_gudhi(gudhi_pairs("circle"))

    assert d.meta.coeff_field == 11
    assert d.meta.provenance["coeff_field_source"] == "backend_default"


def test_from_ripser_records_the_callers_coefficient_field(ripser_dgms: Any) -> None:
    """§11, the other direction: a suite testing only the default case passes
    on an adapter that ignores the argument outright."""
    d = from_ripser(ripser_dgms("circle"), coeff_field=3)

    assert d.meta.coeff_field == 3
    assert d.meta.provenance["coeff_field_source"] == "caller"


def test_from_ripser_records_ripsers_default_coefficient_field(
    ripser_dgms: Any,
) -> None:
    """§9.3, §11: Ripser computes over Z/2 by default -- not GUDHI's Z/11."""
    d = from_ripser(ripser_dgms("circle"))

    assert d.meta.coeff_field == 2
    assert d.meta.provenance["coeff_field_source"] == "backend_default"


@pytest.mark.parametrize("adapter_name", ["from_persim", "from_array"])
def test_adapters_without_a_backend_record_no_coefficient_field(
    adapter_name: str,
) -> None:
    """§11: `from_persim` and `from_array` are out of scope for D17 -- one
    computes no homology, the other has no backend, so neither may assert a
    field it cannot know."""
    arr = np.array([[0.0, 1.0]])
    d = from_persim([arr]) if adapter_name == "from_persim" else from_array(arr, dim=0)

    assert d.meta.coeff_field is None
    assert "coeff_field_source" not in d.meta.provenance


def test_from_giotto_records_no_coefficient_field(giotto_array: Any) -> None:
    """§11: giotto is excluded on evidence -- A.5 could not measure its
    default, and this project does not assert a backend default it has not
    measured."""
    b = from_giotto(
        giotto_array(reduced=True), reduced_homology=True, infinity_values=math.inf
    )

    assert b[0].meta.coeff_field is None
    assert "coeff_field_source" not in b[0].meta.provenance


# ---------------------------------------------------------------------------
# from_ripser -- §11's two accepted input forms
# ---------------------------------------------------------------------------


def test_from_ripser_accepts_the_result_dict(ripser_dgms: Any) -> None:
    """§11: `ripser(X)` returns a dict; `"dgms"` is the diagram list."""
    dgms = ripser_dgms("circle")

    d = from_ripser({"dgms": dgms})

    assert d.n_bars == sum(int(x.shape[0]) for x in dgms)
    assert d.meta.backend == "ripser"


def test_from_ripser_records_the_rips_filtration() -> None:
    """§8, §11: Ripser's input form determines the filtration."""
    d = from_ripser([np.array([[0.0, 1.0]])])

    assert d.meta.filtration == "rips"


def test_from_ripser_accepts_matching_filtration_and_rejects_conflict() -> None:
    """A caller may restate the fact, but may not make the diagram lie."""
    assert (
        from_ripser([np.array([[0.0, 1.0]])], filtration="rips").meta.filtration
        == "rips"
    )
    assert (
        from_ripser([np.array([[0.0, 1.0]])], filtration=None).meta.filtration == "rips"
    )
    with pytest.raises(TypeError, match=r"filtration.*rips"):
        from_ripser([np.array([[0.0, 1.0]])], filtration="alpha")


def test_from_ripser_accepts_the_fit_transform_list(
    backend_output: dict[str, Any], ripser_dgms: Any
) -> None:
    """§11: `Rips().fit_transform(X)` returns the same list without the dict."""
    d = from_ripser(ripser_dgms("circle", key="fit_transform"))

    assert d.n_bars > 0
    assert d.meta.backend == "ripser"


def test_from_ripser_reads_degree_from_list_position(ripser_dgms: Any) -> None:
    """§11: "Index in the list *is* the degree." Nothing else states it."""
    dgms = ripser_dgms("circle")

    d = from_ripser(dgms)

    expected = [0] * int(dgms[0].shape[0]) + [1] * int(dgms[1].shape[0])
    assert [int(x) for x in d.dims] == expected


def test_from_ripser_preserves_row_order_within_a_degree(ripser_dgms: Any) -> None:
    """§7, §11: no sort, so degree 0's rows arrive in Ripser's own order."""
    dgms = ripser_dgms("circle")

    d = from_ripser(dgms)

    n0 = int(dgms[0].shape[0])
    assert [float(x) for x in d.deaths][:n0] == list(dgms[0][:, 1])


def test_from_ripser_preserves_infinite_deaths(ripser_dgms: Any) -> None:
    """§5.1: Ripser is faithful; the essential bar is `inf`, never a sentinel."""
    dgms = ripser_dgms("circle")

    d = from_ripser(dgms)

    assert int(np.sum(np.asarray(d.essential))) == int(np.isinf(dgms[0]).sum())
    assert d.meta.provenance["essential_bars"] == "faithful"
    assert d.meta.provenance["essential_bars_source"] == "faithful"


def test_from_ripser_accepts_a_degree_that_is_empty(ripser_dgms: Any) -> None:
    """§11.2: empty in one degree, not another. Ripser's own H1 on one point."""
    dgms = ripser_dgms("point")

    d = from_ripser(dgms)

    assert [int(x.shape[0]) for x in dgms] == [1, 0], "fixture changed"
    assert d.n_bars == 1
    assert int(d.dims[0]) == 0


@pytest.mark.parametrize("adapter", [from_ripser, from_persim])
def test_the_degree_list_adapters_keep_repeated_bars(
    adapter: Any, ripser_dgms: Any
) -> None:
    """§11.2: multiplicity survives -- adapters never deduplicate.

    Asserted here as well as on `from_gudhi` because the two reach storage by
    different code. GUDHI's `persistence()` list is unpacked row by row into
    Python lists; Ripser's and persim's blocks are stacked and concatenated as
    arrays, and a `unique` slipped into either path would be invisible to a
    test of the other. §11's table has persim consuming exactly what Ripser
    emits, which is why one fixture serves both.

    The twin-pairs cloud is two coincident pairs, so the repetition is a
    property of the data rather than of the capture."""
    dgms = ripser_dgms("twin_pairs")
    assert dgms[0].tolist().count([0.0, 1.0]) == 2, "fixture changed"

    d = adapter(dgms)

    rows = [
        (int(k), float(b), float(x))
        for k, b, x in zip(d.dims, d.births, d.deaths, strict=True)
    ]
    assert rows.count((0, 0.0, 1.0)) == 2, "identical finite bars were collapsed"
    assert d.n_bars == sum(int(block.shape[0]) for block in dgms)


def test_from_ripser_rejects_a_dict_without_dgms(ripser_dgms: Any) -> None:
    """A dict from somewhere else is not a Ripser result; say so at the call."""
    with pytest.raises((TypeError, ValueError), match="dgms"):
        from_ripser({"cocycles": []})


# ---------------------------------------------------------------------------
# from_persim -- §11's shape, and its silence about essential bars
# ---------------------------------------------------------------------------


def test_from_persim_reads_degree_from_list_position(ripser_dgms: Any) -> None:
    """§11: persim consumes `list[(n, 2)]`, "same shape as Ripser's dgms"."""
    dgms = ripser_dgms("circle")

    d = from_persim(dgms)

    assert d.meta.backend == "persim"
    assert [int(x) for x in d.dims] == [0] * int(dgms[0].shape[0]) + [1] * int(
        dgms[1].shape[0]
    )


def test_from_persim_makes_no_claim_about_essential_bars(ripser_dgms: Any) -> None:
    """§5.1: persim "consumes (n,2) arrays; no opinion". An adapter that
    computed no homology cannot certify that none was lost, and §8's key
    means the verdict at computation time."""
    d = from_persim(ripser_dgms("circle"))

    assert "essential_bars" not in d.meta.provenance
    assert "essential_bars_source" not in d.meta.provenance


def test_from_persim_accepts_an_empty_list() -> None:
    """A degenerate but legal input: no degrees at all, so no bars."""
    d = from_persim([])

    assert d.n_bars == 0
    assert d.meta.backend == "persim"


# ---------------------------------------------------------------------------
# from_array -- §11's two shapes
# ---------------------------------------------------------------------------


def test_from_array_accepts_two_columns_with_an_explicit_degree() -> None:
    """§11: `(n, 2)` with explicit `dim=`."""
    arr = np.array([[0.0, 1.0], [0.5, math.inf]])

    d = from_array(arr, dim=1)

    assert [int(x) for x in d.dims] == [1, 1]
    assert [float(x) for x in d.deaths] == [1.0, math.inf]
    assert d.meta.backend == "array"


def test_from_array_accepts_three_columns_in_giottos_order() -> None:
    """§11: `(n, 3)` is `(birth, death, dim)` -- matching giotto deliberately."""
    arr = np.array([[0.0, 1.0, 0.0], [0.5, 2.0, 1.0]])

    d = from_array(arr)

    assert [int(x) for x in d.dims] == [0, 1]
    assert [float(x) for x in d.births] == [0.0, 0.5]
    assert [float(x) for x in d.deaths] == [1.0, 2.0]


def test_from_array_signature_has_columns_before_dim() -> None:
    """§10.3, §11: the public signature is part of the interchange contract."""
    parameters = inspect.signature(from_array).parameters
    assert list(parameters)[:3] == ["arr", "columns", "dim"]
    assert parameters["columns"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["columns"].default is None
    assert parameters["dim"].kind is inspect.Parameter.KEYWORD_ONLY


def test_from_gudhi_dim_is_keyword_only() -> None:
    """§11: "`dim` is keyword-only on the two adapters whose input may carry no
    degree." Asserted rather than assumed, because every other test states it
    by name and would go on passing against a positional parameter -- at which
    point `from_gudhi(intervals, 1)` is legal and the argument is silently
    part of the positional contract this document does not give it."""
    parameters = inspect.signature(from_gudhi).parameters

    assert parameters["dim"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["dim"].default is None


def test_from_giotto_bar_data_controls_are_keyword_only() -> None:
    """§11, §5.1: `reduced_homology` is keyword-only so that omitting it is a
    `TypeError` at the call site, and `strip_padding` so that §11.1's three
    modes cannot be selected by position. Both would still raise on omission
    if they were positional-or-keyword; neither would raise on a caller who
    passed them in the wrong order."""
    parameters = inspect.signature(from_giotto).parameters

    assert parameters["reduced_homology"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["reduced_homology"].default is inspect.Parameter.empty
    assert parameters["strip_padding"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["strip_padding"].default is None


def test_from_array_columns_are_case_insensitive_and_override_position() -> None:
    """§10.3: names, not position, carry a supplied header's meaning."""
    arr = np.array([[0.0, 2.0, 0.0], [1.0, 3.0, 1.0]])

    d = from_array(arr, columns=["DIM", "Death", "BIRTH"])

    assert [float(x) for x in d.births] == [0.0, 1.0]
    assert [float(x) for x in d.deaths] == [2.0, 3.0]
    assert [int(x) for x in d.dims] == [0, 1]


def test_from_array_two_column_columns_can_reorder_with_external_dim() -> None:
    arr = np.array([[9.0, 2.0], [10.0, 3.0]])

    d = from_array(arr, columns=("DEATH", "birth"), dim=4)

    assert [float(x) for x in d.births] == [2.0, 3.0]
    assert [float(x) for x in d.deaths] == [9.0, 10.0]
    assert [int(x) for x in d.dims] == [4, 4]


def test_from_array_columns_support_empty_arrays() -> None:
    d2 = from_array(np.empty((0, 2)), columns=["birth", "death"], dim=0)
    d3 = from_array(np.empty((0, 3)), columns=["birth", "death", "dim"])

    assert d2.n_bars == d3.n_bars == 0


@pytest.mark.parametrize(
    ("columns", "error", "match"),
    [
        ("birth,death", TypeError, "sequence"),
        (1, TypeError, "sequence"),
        # Not "length": §10.3 decides `["birth"]` on the argument, where it is
        # a header naming no death, before any array width is consulted. The
        # length rule needs an otherwise-valid header and has its own test.
        (["birth"], ValueError, "missing.*death"),
        (["birth", 1], TypeError, "string"),
        (["birth", "birth"], ValueError, "duplicate"),
        (["birth", "dim"], ValueError, "missing.*death"),
        (["birth", "death", "diagram_id"], TypeError, r"diagram_id.*\.akd"),
        (["birth", "death", "other"], ValueError, "unknown.*other"),
    ],
)
def test_from_array_rejects_invalid_columns(
    columns: Any, error: type[Exception], match: str
) -> None:
    n_columns = 2 if isinstance(columns, list) and len(columns) == 2 else 3
    arr = np.zeros((1, n_columns))
    with pytest.raises(error, match=match):
        from_array(arr, columns=columns, dim=0)


def test_from_array_rejects_columns_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length"):
        from_array(np.zeros((1, 3)), columns=["birth", "death"])


@pytest.mark.parametrize(
    ("n_columns", "columns", "error", "match"),
    [
        (3, "birth,death,dim", TypeError, "sequence"),
        (3, ["birth", "birth", "dim"], ValueError, "duplicate"),
        (2, ["birth", "dim"], ValueError, "missing.*death"),
        (3, ["birth", "death", "zzz"], ValueError, "unknown.*zzz"),
        (3, ["birth", "death", 1], TypeError, "string"),
        (3, ["birth", "death", "diagram_id"], TypeError, r"diagram_id.*\.akd"),
    ],
)
def test_from_array_validates_columns_before_it_reads_the_data(
    n_columns: int, columns: Any, error: type[Exception], match: str
) -> None:
    """§10.3: "Both MUST raise on the argument, before `arr` is inspected, so
    the failure does not depend on the data".

    `test_from_array_rejects_invalid_columns` runs each rule against an `arr`
    that would construct cleanly, which is §11.2's requirement and proves the
    check is not §3.1 catching the data instead. It does not prove the
    ordering, because an implementation that inspected the values first would
    find nothing to complain about and reach the same error by the same route.

    These arrays fail I4, I5 and I3 at once -- `nan` birth, `-inf` death,
    non-integral degree -- so an implementation that read them first raises
    about death times or degrees, naming the data. §10.3 puts `finitize`'s
    `at` and §6.3's cross-namespace check under the same ordering rule for the
    same reason, and both of those have this test; `columns` did not.

    The width is parametrised alongside so that `columns` always matches the
    array's column count. §10.3 scopes the length rule to `arr`'s shape rather
    than to the argument alone, so it is the one check that must consult the
    array -- and it must never be the thing that fires here, or the case would
    prove nothing about the rule it names. The missing-`death` case needs two
    columns for a second reason: with three entries, no duplicate and no
    unknown name, the three recognised names cannot omit `death`."""
    row = [math.nan, -math.inf, 1.5]
    arr = np.array([row[:n_columns]])

    with pytest.raises(error, match=match):
        from_array(arr, columns=columns)


def test_from_array_named_two_columns_still_require_a_degree() -> None:
    """§11: "MUST raise `TypeError` when handed a degreeless input without
    `dim=`" -- on the named path as much as the positional one.

    Naming the two columns says which is `birth` and which is `death`. It
    says nothing about the degree, which is the fact a two-column array does
    not hold however its columns are labelled, so `columns=` must not become
    a second way to reach the guess §11 forbids."""
    with pytest.raises(TypeError, match="degree"):
        from_array(np.zeros((1, 2)), columns=["birth", "death"])

    with pytest.raises(TypeError, match="degree"):
        from_array(np.zeros((1, 2)), columns=["death", "birth"])


def test_from_array_named_columns_still_reject_an_unreadable_shape() -> None:
    """§11's two shapes bind both paths: a `columns=` a caller happens to have
    does not make a rank-1 array readable."""
    with pytest.raises(ValueError, match="shape"):
        from_array(np.zeros(3), columns=["birth", "death"], dim=0)


def test_from_array_rejects_external_dim_for_named_three_column_data() -> None:
    with pytest.raises(TypeError, match="dim"):
        from_array(
            np.array([[0.0, 1.0, 2.0]]),
            columns=["birth", "death", "dim"],
            dim=0,
        )


def test_from_array_names_diagram_id_before_rejecting_four_column_shape() -> None:
    """§10.3: the batch-column refusal remains actionable on a wider table."""
    with pytest.raises(TypeError, match=r"diagram_id.*\.akd"):
        from_array(
            np.zeros((1, 4)),
            columns=["diagram_id", "dim", "birth", "death"],
        )


def test_from_array_rejects_two_columns_without_a_degree() -> None:
    """Silence is not degree 0. Guessing would fabricate the one fact absent."""
    with pytest.raises(TypeError, match="dim"):
        from_array(np.array([[0.0, 1.0]]))


def test_from_array_rejects_three_columns_with_a_degree() -> None:
    """Two sources for one fact; the adapter must not pick a winner."""
    with pytest.raises((TypeError, ValueError), match="dim"):
        from_array(np.array([[0.0, 1.0, 0.0]]), dim=1)


@pytest.mark.parametrize("shape", [(3,), (2, 4), (2, 2, 3)])
def test_from_array_rejects_shapes_it_cannot_read(shape: tuple[int, ...]) -> None:
    """§11's table admits `(n, 2)` and `(n, 3)`. Anything else is a mistake."""
    with pytest.raises(ValueError, match="shape"):
        from_array(np.zeros(shape), dim=0)


def test_from_array_rejects_a_non_integral_degree_column() -> None:
    """A degree of 1.5 is not a homological degree; truncating hides the bug."""
    with pytest.raises(ValueError, match="integ"):
        from_array(np.array([[0.0, 1.0, 1.5]]))


def test_from_array_records_no_backend_version() -> None:
    """§8: `backend` is `"array"`, and there is no version to report. Honest
    absence beats a version number invented for a backend that does not
    exist."""
    d = from_array(np.array([[0.0, 1.0]]), dim=0)

    assert d.meta.backend == "array"
    assert d.meta.backend_version is None


def test_from_array_makes_no_claim_about_essential_bars() -> None:
    """§8: nothing about a caller's array says whether anything was lost."""
    d = from_array(np.array([[0.0, 1.0]]), dim=0)

    assert "essential_bars" not in d.meta.provenance
    assert "essential_bars_source" not in d.meta.provenance


# ---------------------------------------------------------------------------
# §3.1 -- what the adapters must refuse, and what they must repair
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("row", "invariant"),
    [
        ([0.0, math.nan], "I5"),
        ([math.nan, 1.0], "I4"),
        ([0.0, -math.inf], "I5"),
        ([math.inf, math.inf], "I4"),
    ],
)
def test_from_array_refuses_invalid_coordinates(
    row: list[float], invariant: str
) -> None:
    """§11: every adapter validates against §3.1. An invalid diagram MUST NOT
    be constructible, and the adapter is not allowed to launder one."""
    with pytest.raises(ValueError, match=invariant):
        from_array(np.array([row]), dim=0)


def test_from_array_refuses_a_negative_degree() -> None:
    """I3: a homological degree is non-negative."""
    with pytest.raises(ValueError, match="I3"):
        from_array(np.array([[0.0, 1.0, -1.0]]))


def test_adapter_clamps_a_floating_point_i6_violation_and_warns() -> None:
    """§3.1: "Observed floating-point violations are a real occurrence at the
    1e-16 level in some filtration code; the adapter (not the core type) is
    the correct place to clamp, and it MUST warn when it does." """
    birth = 1.0
    death = math.nextafter(birth, 0.0)  # one ulp below: death < birth
    assert death < birth

    with pytest.warns(UserWarning, match="I6"):
        d = from_array(np.array([[birth, death]]), dim=0)

    assert float(d.deaths[0]) == birth, "the row should be repaired to death == birth"
    assert d.meta.provenance["clamped_rows"] == 1


def test_clamping_scales_with_magnitude() -> None:
    """Floating-point noise is relative. A 1e-10 violation at 1e6 is the same
    defect as a 1e-16 violation at 1.0, and an absolute-only threshold would
    treat one as noise and the other as a bug."""
    birth = 1e6
    death = math.nextafter(birth, 0.0)

    with pytest.warns(UserWarning, match="I6"):
        d = from_array(np.array([[birth, death]]), dim=0)

    assert float(d.deaths[0]) == birth


def test_adapter_does_not_clamp_a_real_i6_violation() -> None:
    """§3.1: "A backend that returns `death < birth` has a bug ... and we
    surface it rather than absorb it." Clamping is for noise, never for a
    violation large enough to be a defect."""
    with pytest.raises(ValueError, match="I6"):
        from_array(np.array([[1.0, 0.5]]), dim=0)


@pytest.mark.parametrize("birth", [1e12, 1.0, 1e-12])
def test_large_ulp_real_i6_violation_is_not_clamped(birth: float) -> None:
    """§3.1: a 4096-local-ULP gap is a real violation, not noise."""
    spacing = birth - math.nextafter(birth, -math.inf)
    death = birth - 4096 * spacing
    assert (birth - death) / spacing == 4096

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="I6"):
            from_array(np.array([[birth, death]]), dim=0)

    assert not any("I6" in str(warning.message) for warning in record)


@pytest.mark.parametrize("birth", [1.0, 1e-12, 1e12, -1e6])
def test_the_clamp_threshold_is_exactly_eight_local_ulps(birth: float) -> None:
    """§3.1 requires the adapter to clamp representational noise and to
    surface a real violation, and fixes no number between them.
    `adapters.py` fixes one and states it: eight local downward float64 ULPs.

    That constant is the whole of the boundary and nothing pinned it. The
    suite's other clamp tests sit at one and four ULPs on the repaired side
    and 4096 on the refused side, so every value from two to some hundreds
    would pass them all -- a threshold widened to 512 by accident absorbs
    filtration errors §3.1 wants surfaced, and one narrowed to two starts
    raising on the 1e-16 noise §3.1 names as a real occurrence.

    Asserted at four magnitudes because the threshold is *local*: the spacing
    is recomputed per row from `birth`, so a constant that had drifted into an
    absolute tolerance would still pass at 1.0 alone. The negative magnitude
    is there because the local spacing below a negative value is the one a
    `nextafter` toward `-inf` gets right and an `abs()` does not."""
    spacing = birth - math.nextafter(birth, -math.inf)

    with pytest.warns(UserWarning, match="I6"):
        d = from_array(np.array([[birth, birth - 8 * spacing]]), dim=0)
    assert float(d.deaths[0]) == birth, "eight local ULPs is inside the threshold"
    assert d.meta.provenance["clamped_rows"] == 1

    with pytest.raises(ValueError, match="I6"):
        from_array(np.array([[birth, birth - 9 * spacing]]), dim=0)


def test_a_repair_at_zero_uses_the_minimum_subnormal_spacing() -> None:
    """The clamp's own special case, exercised where it actually repairs.

    `nextafter(0.0, -inf)` is a subnormal, so the local spacing at zero cannot
    be found the way it is found everywhere else without raising under strict
    floating-point errors. `_clamp_i6` probes a benign normal in those lanes
    and substitutes the exact minimum-subnormal spacing, and the two existing
    subnormal tests take that branch with valid rows -- `candidate` false
    throughout -- so the repair arithmetic on it has never run.

    Eight minimum subnormals is therefore the threshold at zero, and zero
    births are ubiquitous in H0 (§8.1), which makes this the lane a real
    diagram is most likely to reach. Run under `np.errstate(all="raise")`
    because a branch that computed the spacing the ordinary way would produce
    the right answer and an underflow diagnostic on the way to it."""
    smallest = float.fromhex("0x0.0000000000001p-1022")

    with np.errstate(all="raise"), pytest.warns(UserWarning, match="I6"):
        d = from_array(np.array([[0.0, -8 * smallest]]), dim=0)
    assert float(d.deaths[0]) == 0.0

    with np.errstate(all="raise"), pytest.raises(ValueError, match="I6"):
        from_array(np.array([[0.0, -9 * smallest]]), dim=0)


def test_a_repair_at_zero_preserves_the_sign_of_the_birth() -> None:
    """A repair writes the birth into the death, so it inherits its sign.

    §8.1 normalises `-0.0` to `+0.0` before hashing *because* storage does
    not, and §6.3 compares them equal -- so a clamp that had normalised on the
    way in would be invisible to both `==` and `content_hash` while making
    §8.1's clause unreachable, exactly as `test_from_array_preserves_the_sign
    _of_zero` argues for the unrepaired path."""
    smallest = float.fromhex("0x0.0000000000001p-1022")

    with pytest.warns(UserWarning, match="I6"):
        d = from_array(np.array([[-0.0, -smallest]]), dim=0)

    assert math.copysign(1.0, float(d.deaths[0])) == -1.0
    assert float(d.deaths[0]) == 0.0


def test_a_repair_just_below_the_smallest_normal_survives_strict_errors() -> None:
    """The other side of the same branch: a birth that is itself subnormal.

    The mask is `abs(birth) <= smallest_normal`, so it covers subnormal births
    as well as zero, and a diagram whose filtration values are that small is
    the one where an ordinary `nextafter` spacing underflows."""
    smallest = float.fromhex("0x0.0000000000001p-1022")

    with np.errstate(all="raise"), pytest.warns(UserWarning, match="I6"):
        d = from_array(np.array([[smallest, 0.0]]), dim=0)

    assert float(d.births[0]) == smallest
    assert float(d.deaths[0]) == smallest


def test_a_repair_at_the_bottom_of_the_float_range_does_not_overflow() -> None:
    """`_clamp_i6` clips its threshold at `-finfo.max` so that subtracting
    eight ULPs from a birth near that endpoint cannot overflow.

    The clip changes no answer -- an overflowed threshold of `-inf` admits the
    same rows -- so only the arithmetic distinguishes an implementation with
    it from one without, and only under strict floating-point errors. This is
    the narrowest reachable candidate at that endpoint: one ULP above
    `-finfo.max`, whose only finite death below it is `-finfo.max` itself."""
    minimum = float(-np.finfo(np.float64).max)
    birth = math.nextafter(minimum, 0.0)

    with np.errstate(all="raise"), pytest.warns(UserWarning, match="I6"):
        d = from_array(np.array([[birth, minimum]]), dim=0)

    assert float(d.deaths[0]) == birth
    assert d.meta.provenance["clamped_rows"] == 1


@pytest.mark.parametrize(
    ("birth", "death"),
    [
        (-np.finfo(np.float64).max, np.finfo(np.float64).max),
        (0.0, 1.0),
    ],
)
def test_valid_rows_survive_strict_numpy_float_errors(
    birth: float, death: float
) -> None:
    """Valid I6 rows must not perform invalid clamp arithmetic."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with np.errstate(all="raise"):
            d = from_array(np.array([[birth, death]], dtype=np.float64), dim=0)

    assert float(d.births[0]) == birth
    assert float(d.deaths[0]) == death
    assert not record


def test_subnormal_valid_row_survives_strict_numpy_float_errors() -> None:
    """A precomputed subnormal row must not trigger strict-error diagnostics."""
    birth = np.nextafter(0.0, np.inf)
    death = np.nextafter(birth, np.inf)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with np.errstate(all="raise"):
            d = from_array(np.array([[birth, death]], dtype=np.float64), dim=0)

    assert float(d.births[0]) == birth
    assert float(d.deaths[0]) == death
    assert not record


def test_clamped_rows_is_recorded_as_zero_when_nothing_was_repaired() -> None:
    """§8, on §11.1's precedent for `padding_removed`: the key records what
    was actually repaired, so its meaning does not change with the outcome."""
    d = from_array(np.array([[0.0, 1.0]]), dim=0)

    assert d.meta.provenance["clamped_rows"] == 0


def test_clamping_reports_every_repaired_row() -> None:
    """The count is a count, not a flag: a caller auditing provenance needs to
    know how much was absorbed."""
    births = np.array([1.0, 2.0, 3.0])
    deaths = np.array([math.nextafter(b, 0.0) for b in births])

    with pytest.warns(UserWarning, match="3 of 3"):
        d = from_array(np.stack([births, deaths], axis=1), dim=0)

    assert d.meta.provenance["clamped_rows"] == 3


# ---------------------------------------------------------------------------
# §6.1 / §3.3 -- dtype is converted, namespace is not
# ---------------------------------------------------------------------------


def test_adapter_converts_dtype_to_the_storage_types(ripser_dgms: Any) -> None:
    """§6.1, I2: storage is the namespace's own `float64` and `int32`,
    whatever the backend handed over."""
    dgms = [d.astype(np.float32) for d in ripser_dgms("circle")]

    d = from_ripser(dgms)

    assert d.births.dtype == np.float64
    assert d.deaths.dtype == np.float64
    assert d.dims.dtype == np.int32


def test_adapter_records_the_source_dtype(ripser_dgms: Any) -> None:
    """§8: `source_dtype` is "dtype of the input array", recorded as a string
    because §8 requires every provenance value to be JSON-representable."""
    dgms = [d.astype(np.float32) for d in ripser_dgms("circle")]

    d = from_ripser(dgms)

    assert d.meta.provenance["source_dtype"] == "float32"
    assert_json_representable(d.meta.provenance)


def test_every_array_carrying_adapter_records_the_source_dtype(
    gudhi_intervals: Any, giotto_array: Any
) -> None:
    """§8: `source_dtype` is "dtype of the input array", so every adapter that
    receives one records it -- not only the one whose test happens to exist.
    Each is asserted against the dtype the input actually carried, which is
    what a `str(arr.dtype)` written against the *converted* column would get
    wrong (§6.1 upcasts every coordinate to float64)."""
    intervals = gudhi_intervals("circle", 1).astype(np.float32)
    table = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    giotto = giotto_array(reduced=True).astype(np.float32)

    from_gudhi_dtype = from_gudhi(intervals, dim=1).meta.provenance["source_dtype"]
    from_array_dtype = from_array(table).meta.provenance["source_dtype"]
    giotto_batch = from_giotto(
        giotto, reduced_homology=True, infinity_values=math.inf, strip_padding=False
    )

    assert from_gudhi_dtype == "float32"
    assert from_array_dtype == "float32"
    assert giotto_batch[0].meta.provenance["source_dtype"] == "float32"


def test_the_gudhi_list_form_records_no_source_dtype(gudhi_pairs: Any) -> None:
    """§8's key is "dtype of the input array", and this form has no array.
    Recording `"float64"` here would state a fact about our own conversion
    rather than about the input."""
    d = from_gudhi(gudhi_pairs("circle"))

    assert "source_dtype" not in d.meta.provenance


@pytest.mark.parametrize("adapter", [from_ripser, from_persim])
def test_a_degree_list_records_the_dtype_every_block_shares(adapter: Any) -> None:
    """§8: `source_dtype` describes the input, and a uniform list has one
    answer for the whole diagram whichever block is read first."""
    dgms = [
        np.array([[0.0, 1.0]], dtype=np.float32),
        np.array([[0.5, 2.0]], dtype=np.float32),
    ]

    d = adapter(dgms)

    assert d.meta.provenance["source_dtype"] == "float32"


@pytest.mark.parametrize("adapter", [from_ripser, from_persim])
def test_a_degree_list_records_no_source_dtype_when_the_blocks_disagree(
    adapter: Any,
) -> None:
    """C3b. `source_dtype` used to be read off the first block alone, so a
    `[float32, float64]` list recorded `"float32"` -- a statement about degree
    0 presented as one about the diagram. §8 has one slot and no spelling for
    a disagreement, so the key is omitted rather than half-answered.

    The diagram itself is still built: the bars are valid whatever their
    incoming dtypes, and it is only the record about them that cannot be
    written."""
    dgms = [
        np.array([[0.0, 1.0]], dtype=np.float32),
        np.array([[0.5, 2.0]], dtype=np.float64),
    ]

    d = adapter(dgms)

    assert "source_dtype" not in d.meta.provenance
    assert d.n_bars == 2
    assert [int(x) for x in d.dims] == [0, 1]


def test_from_persim_preserves_row_order_within_a_degree() -> None:
    """§7, §11: no sort, so a degree's rows arrive in the caller's own order.
    `from_ripser` has covered this since the adapters landed and `from_persim`
    did not, the two sharing `_columns_from_degree_list`."""
    dgms = [np.array([[0.9, 3.0], [0.1, 2.0], [0.5, 1.5]])]

    d = from_persim(dgms)

    assert [float(x) for x in d.births] == [0.9, 0.1, 0.5]
    assert [float(x) for x in d.deaths] == [3.0, 2.0, 1.5]


def test_adapter_preserves_the_input_namespace() -> None:
    """§3.3: "Adapters preserve the input namespace. `from_*` MUST NOT
    force-convert to NumPy ... What adapters convert is *dtype*, not
    namespace." """
    xps = pytest.importorskip("array_api_strict")
    arr = xps.asarray([[0.0, 1.0, 0.0], [0.5, 2.0, 1.0]], dtype=xps.float64)

    d = from_array(arr)

    assert d.xp is xps
    assert d.births.dtype == xps.float64
    assert d.dims.dtype == xps.int32


def test_the_gudhi_array_form_preserves_the_input_namespace(
    gudhi_intervals: Any,
) -> None:
    """§3.3, on GUDHI's other input form. `from_gudhi` has two arms and only
    one of them touches an array; the list arm is numpy by construction, so a
    suite that tested the namespace rule on the list arm alone would be
    testing the fallback."""
    xps = pytest.importorskip("array_api_strict")
    intervals = xps.asarray(gudhi_intervals("circle", 1), dtype=xps.float64)

    d = from_gudhi(intervals, dim=1)

    assert d.xp is xps
    assert d.dims.dtype == xps.int32


@pytest.mark.parametrize("strip_padding", [None, True, False])
def test_from_giotto_preserves_the_input_namespace(
    giotto_array: Any, strip_padding: bool | None
) -> None:
    """§3.3 for the one adapter that indexes a rank-3 array, under all three
    of §11.1's modes.

    The array API standard requires an index per axis or an ellipsis, and
    every namespace a developer is likely to have installed -- NumPy, torch,
    JAX -- accepts the short `arr[i]` form regardless. `array_api_strict`
    raises `IndexError`, which is what makes it the backend this rule has to
    be tested against rather than the one it is convenient to skip.

    `strip_padding=True` is parametrised in for a second reason: it is the
    only mode that boolean-masks the columns before construction, so it is the
    only one where an index expression the standard does not define could hide
    behind a mode the default never reaches."""
    xps = pytest.importorskip("array_api_strict")
    arr = xps.asarray(giotto_array(reduced=True, sample="batch"), dtype=xps.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        b = from_giotto(
            arr,
            reduced_homology=True,
            infinity_values=math.inf,
            strip_padding=strip_padding,
        )

    assert len(b) == arr.shape[0]
    assert b.xp is xps
    assert b[0].xp is xps
    assert b[0].dims.dtype == xps.int32
    assert b[0].births.dtype == xps.float64


def test_a_list_input_falls_back_to_numpy() -> None:
    """§11 fixes `from_gudhi(obj, **meta)` with no namespace argument, and
    GUDHI's primary form is a Python list carrying no array at all. The
    namespace has to come from somewhere; numpy is imported lazily on this
    path alone, which no caller reaches without having installed a backend
    that already depends on it."""
    d = from_gudhi([(0, (0.0, 1.0))])

    assert d.xp is np


def test_the_row_fallback_resolves_through_the_one_namespace_function(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§3.3: "Namespace resolution goes through exactly one function."

    The row fallback builds its own NumPy probe, so it is the one place in
    either module that could answer the namespace question without asking
    `namespace_of` -- and calling `probe.__array_namespace__()` directly is a
    second spelling that agrees with the resolver today and is exactly what
    the rule forbids. §3.3 names the hazard: `array_api_compat.array_namespace`
    on a NumPy array returns `array_api_compat.numpy` rather than `numpy`
    (A.7.5), so two spellings become two namespace objects for one backend and
    I7's `is` raises on arrays that legitimately share one.

    Asserted by substitution rather than by reading the source: the resolver
    is replaced, and the fallback must return what it returned."""
    sentinel = object()
    monkeypatch.setattr(adapters_module, "namespace_of", lambda x: sentinel)

    assert adapters_module._namespace_for_rows() is sentinel


def test_a_list_sourced_and_an_array_sourced_diagram_share_one_namespace(
    gudhi_pairs: Any, gudhi_intervals: Any
) -> None:
    """What the rule above buys, stated as the failure it prevents.

    GUDHI's two accepted forms take the two namespace paths -- the
    `persistence()` list reaches `_namespace_for_rows`, the interval array
    reaches `namespace_of` -- and I7 compares by `is`. Two spellings that
    returned `numpy` and `array_api_compat.numpy` would give a caller two
    diagrams from one backend, one call, that cannot be composed."""
    from_list = from_gudhi(gudhi_pairs("circle"))
    from_arr = from_gudhi(gudhi_intervals("circle", 1), dim=1)

    assert from_list.xp is from_arr.xp
    assert DiagramBatch.from_diagrams([from_list, from_arr]).xp is from_list.xp


class _NoNativeArray:
    """Array-shaped fake with no `__array_namespace__`, like torch today."""

    ndim = 2
    shape = (1, 2)
    dtype = np.dtype("float64")


@pytest.mark.parametrize(
    "call",
    [
        lambda arr: from_array(arr, dim=0),
        lambda arr: from_gudhi(arr, dim=0),
        lambda arr: from_giotto(arr, reduced_homology=False, infinity_values=math.inf),
        lambda arr: from_ripser([arr]),
        lambda arr: from_persim([arr]),
    ],
)
def test_no_native_array_reaches_shared_namespace_resolver(
    call: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§3.3: expected-array entrypoints must not reject on the attribute gate."""
    supplied = _NoNativeArray()
    seen: list[object] = []

    def reached_shared_resolver(value: object) -> object:
        seen.append(value)
        raise RuntimeError("reached shared namespace resolver")

    monkeypatch.setattr(adapters_module, "namespace_of", reached_shared_resolver)
    with pytest.raises(RuntimeError, match="reached shared namespace resolver"):
        call(supplied)
    assert seen == [supplied]


# ---------------------------------------------------------------------------
# §8 -- what the caller may say, and what the adapter insists on
# ---------------------------------------------------------------------------


def test_caller_metadata_is_carried_through(gudhi_pairs: Any) -> None:
    """§8's fields are the caller's to state; the adapter adds to them."""
    d = from_gudhi(
        gudhi_pairs("circle"),
        filtration="rips",
        description="40-point noisy circle",
        params={"max_edge_length": 4.0},
    )

    assert d.meta.filtration == "rips"
    assert d.meta.description == "40-point noisy circle"
    assert d.meta.params["max_edge_length"] == 4.0


def test_only_ripser_knows_its_filtration(
    gudhi_pairs: Any, gudhi_intervals: Any, giotto_array: Any
) -> None:
    """§8: an adapter "MUST also populate `filtration` where its own input
    form determines it, and MUST NOT guess otherwise". One adapter is in that
    position and the other four are not: a GUDHI `SimplexTree` carries no
    record of what built it -- Rips, alpha, cubical and lower-star all arrive
    as the same object -- `from_giotto` receives a bare array, and
    `from_array` and `from_persim` have no backend to ask.

    The positive half is asserted elsewhere. This is the half that fails
    silently: a suite holding only `from_ripser(...).meta.filtration ==
    "rips"` passes unchanged against an adapter that wrote `"rips"` into every
    diagram it built, and a GUDHI alpha-complex diagram would then carry a
    provenance entry naming the wrong filtration with nothing to contradict
    it."""
    unstated = [
        from_gudhi(gudhi_pairs("circle")),
        from_gudhi(gudhi_intervals("circle", 1), dim=1),
        from_persim([np.array([[0.0, 1.0]])]),
        from_array(np.array([[0.0, 1.0]]), dim=0),
        from_giotto(
            giotto_array(reduced=True),
            reduced_homology=True,
            infinity_values=math.inf,
            strip_padding=False,
        )[0],
    ]

    assert [d.meta.filtration for d in unstated] == [None] * 5


def test_the_four_adapters_that_cannot_know_still_carry_a_stated_filtration(
    gudhi_pairs: Any, giotto_array: Any
) -> None:
    """§8's other half of the same sentence: those adapters "MUST leave
    `filtration` at whatever the caller passed through `**meta`". Not knowing
    is not the same as overriding, and a caller who computed an alpha complex
    is the one party who does know."""
    d = from_gudhi(gudhi_pairs("circle"), filtration="alpha")
    b = from_giotto(
        giotto_array(reduced=True),
        reduced_homology=True,
        infinity_values=math.inf,
        strip_padding=False,
        filtration="rips",
    )

    assert d.meta.filtration == "alpha"
    assert b[0].meta.filtration == "rips"


def test_caller_provenance_is_merged_and_a_measured_fact_is_refused(
    gudhi_pairs: Any,
) -> None:
    """`provenance` is the honest-accounting channel (§8). A caller's own keys
    are kept; a key the adapter measured is refused.

    This asserted a silent overwrite until the refusal replaced it. Overwriting
    is the weaker rule and was weaker in a way that mattered: it protects a key
    only where the adapter writes one of its own, so the same
    `provenance={"essential_bars": ...}` that lost here survived intact on
    `from_persim` and `from_array`, which record no essential-bar claim (§11,
    D2). See `test_no_adapter_lets_a_caller_write_a_reserved_provenance_key`."""
    with pytest.raises(TypeError, match="essential_bars"):
        from_gudhi(
            gudhi_pairs("circle"),
            provenance={"analyst": "eb", "essential_bars": "lost_upstream"},
        )

    d = from_gudhi(gudhi_pairs("circle"), provenance={"analyst": "eb"})

    assert d.meta.provenance["analyst"] == "eb"
    assert d.meta.provenance["essential_bars"] == "faithful"


def test_adapter_refuses_to_have_its_backend_identity_dictated(
    gudhi_pairs: Any,
) -> None:
    """§8, §11: `backend` and `backend_version` are adapter-recorded facts. A
    caller who could set them could produce a diagram that lies about where it
    came from, which is the one thing provenance exists to prevent."""
    with pytest.raises(TypeError, match="backend"):
        from_gudhi(gudhi_pairs("circle"), backend="gudhi")


def test_adapter_rejects_an_unknown_metadata_field(gudhi_pairs: Any) -> None:
    """`**meta` is §8's field set, not a free-form bag: a misspelled
    `filtraton=` must not vanish into a diagram that reports nothing."""
    with pytest.raises(TypeError, match="filtraton"):
        from_gudhi(gudhi_pairs("circle"), filtraton="rips")


def test_adapter_output_composes_into_a_batch(
    gudhi_pairs: Any, ripser_dgms: Any
) -> None:
    """§4.2: "The common path: N separate `from_gudhi`/`from_ripser` calls
    that have to become one batch." """
    diagrams = [from_gudhi(gudhi_pairs("circle")), from_ripser(ripser_dgms("circle"))]

    batch = DiagramBatch.from_diagrams(diagrams)

    assert len(batch) == 2
    assert batch[0].n_bars == diagrams[0].n_bars
    assert batch[1].meta.backend == "ripser"


# ---------------------------------------------------------------------------
# from_giotto -- §11's deviations, §11.1's three modes, §5.1's derivation
# ---------------------------------------------------------------------------


def test_from_giotto_requires_reduced_homology(giotto_array: Any) -> None:
    """§5.1, §11: "Omitting it MUST raise, not fall back to giotto's own
    default", and §11 fixes that as a `TypeError` at the call site."""
    with pytest.raises(TypeError, match="reduced_homology"):
        from_giotto(  # type: ignore[call-arg]
            giotto_array(reduced=True), infinity_values=math.inf
        )


def test_from_giotto_requires_infinity_values(giotto_array: Any) -> None:
    """§5, §11, §11.2. Required for `reduced_homology`'s reason -- no property of the
    returned array says which setting produced it -- and a `TypeError` at the
    call site rather than a default, because the wrong assumption here is the
    one that writes `essential_bars="faithful"` over a finitized diagram."""
    with pytest.raises(TypeError, match="infinity_values"):
        from_giotto(  # type: ignore[call-arg]
            giotto_array(reduced=True), reduced_homology=True
        )


@pytest.mark.parametrize("value", [4.0, 0.0, 99.0, -1.5, -math.inf, math.nan])
def test_from_giotto_refuses_any_infinity_values_but_inf(
    giotto_array: Any, value: float
) -> None:
    """§5: only `inf` records an essential bar. A finite value finitizes it,
    and `-inf` and `nan` are not deaths §5 recognises for a class that never
    dies -- all four are refused on the same ground rather than by a branch
    each."""
    with pytest.raises(ValueError, match="RFC-0001 §5"):
        from_giotto(
            giotto_array(reduced=True), reduced_homology=True, infinity_values=value
        )


@pytest.mark.parametrize("value", ["inf", object()])
def test_from_giotto_refuses_a_non_real_infinity_values(
    giotto_array: Any, value: Any
) -> None:
    """`_as_coordinate`'s argument, one argument over: `float("inf")` is `inf`,
    so a string would pass the equality test below and record a diagram whose
    caller never stated a filtration value at all."""
    with pytest.raises(TypeError, match="infinity_values"):
        from_giotto(
            giotto_array(reduced=True), reduced_homology=True, infinity_values=value
        )


def test_from_giotto_refuses_a_boolean_infinity_values(giotto_array: Any) -> None:
    """`bool` registers as `numbers.Real`, so it needs excluding by name.
    `True` would otherwise read as a finite sentinel of 1.0."""
    with pytest.raises(TypeError, match="infinity_values"):
        from_giotto(
            giotto_array(reduced=True), reduced_homology=True, infinity_values=True
        )


def test_from_giotto_always_returns_a_batch(giotto_array: Any) -> None:
    """§11: a fixed return type. "Nothing about the adapter's own return type
    is allowed to depend on how many samples the particular call happened to
    carry." """
    single = giotto_array(reduced=True, sample="single")

    b = from_giotto(single, reduced_homology=True, infinity_values=math.inf)

    assert isinstance(b, DiagramBatch)
    assert single.shape[0] == 1, "fixture changed"
    assert len(b) == 1
    assert isinstance(b[0], PersistenceDiagram)


def test_from_giotto_returns_one_diagram_per_sample(giotto_array: Any) -> None:
    """§4: the batch is ragged, one entry per input sample, order preserved."""
    batch = giotto_array(reduced=True, sample="batch")

    b = from_giotto(
        batch, reduced_homology=True, infinity_values=math.inf, strip_padding=False
    )

    assert len(b) == batch.shape[0]
    assert [b[i].n_bars for i in range(len(b))] == [batch.shape[1]] * batch.shape[0]


def test_from_giotto_maps_each_sample_to_its_own_slot() -> None:
    """§4, §11: sample `i` of the input is diagram `i` of the batch.

    The test above asserts the cardinality and this one the correspondence,
    which are different claims and only the first is checked by a count. A
    giotto array is dense (§4, A.2), so every sample carries the same row
    count under `strip_padding=False`, and a `from_diagrams` that assembled
    the members in any order at all would satisfy `[n_bars] * n_samples`
    exactly as the right one does.

    §11.2 makes the same argument for the batch round trip -- "one whose
    diagrams are in an order that no sort would produce, since ... a `load`
    that recovered every diagram into the wrong slot passes a test built from
    identical members". The samples here are one bar each and distinguishable
    by birth, in an order neither ascending nor descending, so a permutation
    anywhere between `arr[i, :, :]` and `DiagramBatch.from_diagrams` fails
    rather than passes on a coincidence of the fixture's shape."""
    arr = np.array([[[7.0, 8.0, 0.0]], [[1.0, 2.0, 0.0]], [[4.0, 5.0, 0.0]]])

    b = from_giotto(arr, reduced_homology=False, infinity_values=math.inf)

    assert [float(b[i].births[0]) for i in range(len(b))] == [7.0, 1.0, 4.0]
    assert [float(b[i].deaths[0]) for i in range(len(b))] == [8.0, 2.0, 5.0]


def test_from_giotto_reads_columns_as_birth_death_dim(giotto_array: Any) -> None:
    """§11: giotto's columns are `(birth, death, dim)`, in that order."""
    arr = giotto_array(reduced=True)

    b = from_giotto(
        arr, reduced_homology=True, infinity_values=math.inf, strip_padding=False
    )
    d = b[0]

    assert [float(x) for x in d.births] == list(arr[0][:, 0])
    assert [float(x) for x in d.deaths] == list(arr[0][:, 1])
    assert [int(x) for x in d.dims] == [int(v) for v in arr[0][:, 2]]


def test_from_giotto_records_reduced_homology_in_params(giotto_array: Any) -> None:
    """§5.1: it is "a raw fact of the original call, the same category as
    `max_edge_length`", so it belongs in `params`, not in `provenance`."""
    b = from_giotto(
        giotto_array(reduced=True), reduced_homology=True, infinity_values=math.inf
    )

    assert b[0].meta.params["reduced_homology"] is True


def test_from_giotto_derives_lost_upstream_from_reduced_homology(
    giotto_array: Any,
) -> None:
    """§5.1: `"lost_upstream"` when `reduced_homology` is `True` -- derived
    from the flag, never authored independently, and `essential_bars_source`
    set to the same value in the same construction (§8, §11)."""
    b = from_giotto(
        giotto_array(reduced=True), reduced_homology=True, infinity_values=math.inf
    )

    assert b[0].meta.provenance["essential_bars"] == "lost_upstream"
    assert b[0].meta.provenance["essential_bars_source"] == "lost_upstream"


def test_the_giotto_fixture_still_carries_a_finite_sentinel(
    giotto_array: Any, giotto_output: dict[str, Any]
) -> None:
    """C1's evidence, asserted so that it cannot be recaptured away silently.

    `tools/capture_giotto_fixture.py` passed no `infinity_values`, so giotto's
    default of `None` applied and the essential H0 class came back with a death
    of `max_edge_length` rather than `inf`. This is the Appendix A.1 row the
    RFC said was missing, and it falsifies §5.1's `"faithful"` derivation.

    A recapture with `infinity_values=numpy.inf` -- which the capture tool now
    performs -- will fail this test. That is the intended signal, not a
    regression: at that point the sentinel is gone and the two tests below
    change with it."""
    unreduced = giotto_array(reduced=False)
    call = giotto_output["samples"]["reduced_false"]["call"]
    assert f"max_edge_length={_GIOTTO_MAX_EDGE}" in call, "the cutoff moved"

    h0 = unreduced[0][unreduced[0][:, 2] == 0]

    assert not np.any(np.isinf(h0[:, 1])), "the capture already carries inf"
    assert int(np.sum(h0[:, 1] == _GIOTTO_MAX_EDGE)) == 1, (
        "exactly one H0 class should survive to the cutoff and be finitized"
    )


def test_from_giotto_refuses_the_fixtures_own_default_infinity_values(
    giotto_array: Any,
) -> None:
    """§5: a finite sentinel is "unrecoverable", so the adapter refuses rather
    than labels. Run against real backend output (§11.2) -- this is the exact
    array that `essential_bars="faithful"` used to be written for."""
    unreduced = giotto_array(reduced=False)

    with pytest.raises(ValueError, match="giotto's default"):
        from_giotto(unreduced, reduced_homology=False, infinity_values=None)  # type: ignore[arg-type]


def test_from_giotto_derives_faithful_when_homology_is_not_reduced(
    giotto_array: Any,
) -> None:
    """§5.1: `"faithful"` when `reduced_homology` is `False`.

    **Not run against the committed fixture, and the reason is C1.** That
    capture predates the `infinity_values` requirement, so its unreduced sample
    holds the essential bar as `4.0` rather than `inf` -- asserting
    `"faithful"` over it would bless the defect. The label is exercised here
    over a giotto-shaped array carrying a genuine `inf`, which is what a
    recapture will produce; the fixture is still used for A.1's bar counts
    below, which `infinity_values` does not affect."""
    faithful = np.array([[[0.0, math.inf, 0.0], [0.0, 1.0, 1.0]]])

    b = from_giotto(faithful, reduced_homology=False, infinity_values=math.inf)

    assert b[0].meta.provenance["essential_bars"] == "faithful"
    assert b[0].meta.provenance["essential_bars_source"] == "faithful"
    assert bool(np.any(np.asarray(b[0].essential))), "the inf did not survive"


def test_the_giotto_fixture_measures_a1s_missing_reduced_homology_row(
    giotto_array: Any,
) -> None:
    """A.1's own gap: the table varies `infinity_values` and holds
    `reduced_homology` at `True`, so the 39-against-40 claim was an inference.
    The capture measures it directly, and `infinity_values` does not enter --
    reduced homology drops the class upstream of it (§5.1)."""
    unreduced = giotto_array(reduced=False)
    reduced = giotto_array(reduced=True)

    h0 = int((unreduced[0][:, 2] == 0).sum())

    assert h0 == 40
    assert h0 - int((reduced[0][:, 2] == 0).sum()) == 1, "A.1's H0 loss changed"


def test_from_giotto_does_not_fabricate_the_missing_essential_bar(
    giotto_array: Any,
) -> None:
    """§5.1: "MUST NOT fabricate an essential bar to compensate". Reconstructing
    its birth as 0 is a coincidence of the unweighted example, not a property
    of the elder rule."""
    arr = giotto_array(reduced=True)

    b = from_giotto(
        arr, reduced_homology=True, infinity_values=math.inf, strip_padding=False
    )

    assert b[0].n_bars == arr.shape[1], "a row was invented"
    assert not bool(np.any(np.asarray(b.essential))), "giotto emits no inf (A.1)"


def test_from_giotto_default_keeps_padding_and_warns_once(giotto_array: Any) -> None:
    """§11.1: default `strip_padding=None` keeps every row, warns once if any
    trivial rows are present, and records `padding_removed = 0`."""
    arr = giotto_array(reduced=True, sample="batch")
    trivial = int((arr[:, :, 0] == arr[:, :, 1]).sum())
    assert trivial > 0, "the fixture should carry padding"

    with pytest.warns(UserWarning, match="trivial") as record:
        b = from_giotto(arr, reduced_homology=True, infinity_values=math.inf)

    assert len(record) == 1, "§11.1 says warn once, not once per sample"
    assert [b[i].n_bars for i in range(len(b))] == [arr.shape[1]] * arr.shape[0]
    assert all(b[i].meta.provenance["padding_removed"] == 0 for i in range(len(b)))


def test_from_giotto_strips_padding_when_told_to(giotto_array: Any) -> None:
    """§11.1: `strip_padding=True` drops trivial rows and records the count."""
    arr = giotto_array(reduced=True, sample="batch")
    per_sample = [int((s[:, 0] == s[:, 1]).sum()) for s in arr]
    assert per_sample[0] > 0, "the fixture should pad the first sample"

    b = from_giotto(
        arr, reduced_homology=True, infinity_values=math.inf, strip_padding=True
    )

    for i, dropped in enumerate(per_sample):
        assert b[i].n_bars == arr.shape[1] - dropped
        assert b[i].meta.provenance["padding_removed"] == dropped


def test_from_giotto_preserves_row_order_among_the_rows_stripping_leaves() -> None:
    """§7, §11: "preserve backend row order" binds the mode that removes rows
    as much as the two that do not.

    `from_gudhi`, `from_ripser` and `from_persim` each have this test and
    `from_giotto` had it only through `strip_padding=False`, where the columns
    reach construction untouched. `strip_padding=True` is the one adapter path
    that boolean-masks them first, so it is the one place an implementation
    could reorder -- a mask spelled through `nonzero` and a `take` on sorted
    indices, or a partition that groups the survivors by degree -- and the
    existing coverage would not notice: `test_from_giotto_strips_padding_when
    _told_to` asserts counts, and the multiplicity test asserts membership.

    The rows below are in an order no sort produces. Degree descends then
    ascends, births descend then ascend, and the trivial row sits in the
    middle rather than at either end, so a reordering cannot coincide with the
    input by landing on a sort key that happens to agree."""
    arr = np.array(
        [
            [
                [5.0, 9.0, 1.0],
                [1.0, 1.0, 0.0],  # trivial; the only row stripping removes
                [3.0, 8.0, 0.0],
                [0.0, 2.0, 1.0],
            ]
        ]
    )

    d = from_giotto(
        arr, reduced_homology=False, infinity_values=math.inf, strip_padding=True
    )[0]

    rows = [
        (float(b), float(x), int(k))
        for b, x, k in zip(d.births, d.deaths, d.dims, strict=True)
    ]
    assert rows == [(5.0, 9.0, 1), (3.0, 8.0, 0), (0.0, 2.0, 1)]


def test_from_giotto_keeps_padding_silently_when_told_to(giotto_array: Any) -> None:
    """§11.1: `strip_padding=False` keeps silently, "and
    `provenance['padding_removed'] = 0` regardless of how many trivial rows
    are present -- the key records what was actually removed, never what was
    merely observed"."""
    arr = giotto_array(reduced=True, sample="batch")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        b = from_giotto(
            arr, reduced_homology=True, infinity_values=math.inf, strip_padding=False
        )

    assert all(b[i].meta.provenance["padding_removed"] == 0 for i in range(len(b)))
    assert b[0].n_bars == arr.shape[1]


def test_from_giotto_keeps_every_repeated_zero_persistence_row(
    giotto_array: Any,
) -> None:
    """§11.2's multiplicity and zero-persistence minimums, on giotto's own
    output and through giotto's own code path.

    The two arrive together here because giotto's padding *is* both: the
    fixture's first sample ends in five byte-identical `(b, b, 1)` rows, which
    is exactly why §11.1 says the adapter cannot tell padding from genuine
    trivial bars and must not guess. `strip_padding=False` is the mode that
    says keep them, so all five must be present -- a `unique` anywhere on this
    path would silently collapse them to one and still satisfy the existing
    tests, which assert `n_bars` against the input's row count and would then
    fail for a reason naming padding rather than deduplication.

    §11.1's counterpart is asserted in the same test: the five removed under
    `strip_padding=True` are these five and nothing else."""
    arr = giotto_array(reduced=True, sample="batch")
    sample = arr[0]
    trivial = [tuple(row) for row in sample if row[0] == row[1]]
    assert len(trivial) == 5, "fixture changed"
    assert len(set(trivial)) == 1, "the five trivial rows should be identical"

    kept = from_giotto(
        arr, reduced_homology=True, infinity_values=math.inf, strip_padding=False
    )[0]
    stripped = from_giotto(
        arr, reduced_homology=True, infinity_values=math.inf, strip_padding=True
    )[0]

    rows = [
        (float(b), float(x), int(k))
        for b, x, k in zip(kept.births, kept.deaths, kept.dims, strict=True)
    ]
    assert rows.count(trivial[0]) == 5, "identical trivial rows were collapsed"

    survivors = [
        (float(b), float(x), int(k))
        for b, x, k in zip(stripped.births, stripped.deaths, stripped.dims, strict=True)
    ]
    assert trivial[0] not in survivors
    assert stripped.meta.provenance["padding_removed"] == 5


def test_from_giotto_does_not_warn_when_there_is_no_padding() -> None:
    """§11.1 warns "if any trivial rows are present" -- not unconditionally."""
    arr = np.array([[[0.0, 1.0, 0.0], [0.0, 2.0, 1.0]]])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        b = from_giotto(arr, reduced_homology=False, infinity_values=math.inf)

    assert len(b) == 1


def test_from_giotto_rejects_an_array_that_is_not_a_batch(giotto_array: Any) -> None:
    """§11's table: giotto output is `(n_samples, n_bars, 3)`. A 2-D array is
    a single sample the caller forgot to wrap, and guessing which is which is
    exactly the shape-depends-on-the-data hazard §4 rules out."""
    with pytest.raises(ValueError, match="shape"):
        from_giotto(
            giotto_array(reduced=True)[0],
            reduced_homology=True,
            infinity_values=math.inf,
        )


# ---------------------------------------------------------------------------
# Property-based -- §11's "never sorts, deduplicates or finitizes" is
# universally quantified over every input the adapters admit, which is what a
# property test states directly and an example test only samples. The
# strategies below deliberately generate what hand-written examples miss:
# signed zeros, subnormals, values spanning the float64 exponent range, and
# `inf` deaths alongside finite ones in the same diagram.
# ---------------------------------------------------------------------------

_AWKWARD = [0.0, -0.0, 5e-324, 2.2250738585072014e-308, 1e-300, 1e300, 1.5e15]

_births = st.one_of(
    st.sampled_from(_AWKWARD),
    st.floats(allow_nan=False, allow_infinity=False, width=64),
)
_gaps = st.one_of(
    st.just(0.0),
    st.just(math.inf),
    st.sampled_from(_AWKWARD),
    st.floats(min_value=0.0, allow_nan=False, allow_infinity=False, width=64),
)


@st.composite
def _bars(draw: Any, max_size: int = 12) -> list[tuple[int, float, float]]:
    """`(dim, birth, death)` triples satisfying §3.1, and nothing more."""
    n = draw(st.integers(min_value=0, max_value=max_size))
    rows = []
    for _ in range(n):
        birth = draw(_births)
        gap = draw(_gaps)
        death = math.inf if math.isinf(gap) else birth + abs(gap)
        # `birth + abs(gap)` can round below `birth` for no gap at all, and a
        # generator that produced I6 violations would test the clamp rather
        # than the property under test.
        rows.append(
            (draw(st.integers(min_value=0, max_value=9)), birth, max(birth, death))
        )
    return rows


@given(bars=_bars())
def test_from_array_changes_nothing_but_dtype(
    bars: list[tuple[int, float, float]],
) -> None:
    """§11: adapters never sort, deduplicate or finitize, and §3.3 says what
    they do convert is dtype. So the rows out are the rows in, in order."""
    arr = np.array([[b, d, k] for k, b, d in bars], dtype=np.float64).reshape(
        len(bars), 3
    )

    d = from_array(arr)

    assert [int(x) for x in d.dims] == [k for k, _, _ in bars]
    assert [float(x) for x in d.births] == [b for _, b, _ in bars]
    assert [float(x) for x in d.deaths] == [x for _, _, x in bars]


@given(bars=_bars())
def test_from_array_preserves_the_sign_of_zero(
    bars: list[tuple[int, float, float]],
) -> None:
    """§6.3, §8.1: `-0.0 == 0.0`, so a diagram that silently normalised the
    sign would compare equal and hash differently. §8.1 normalises inside the
    hash precisely because storage does not; an adapter that normalised on the
    way in would make that clause unreachable."""
    arr = np.array([[b, d, k] for k, b, d in bars], dtype=np.float64).reshape(
        len(bars), 3
    )

    d = from_array(arr)

    assert [math.copysign(1.0, float(x)) for x in d.births] == [
        math.copysign(1.0, b) for _, b, _ in bars
    ]


@given(blocks=st.lists(_bars(max_size=6), max_size=4))
def test_from_ripser_maps_list_position_to_degree(
    blocks: list[list[tuple[int, float, float]]],
) -> None:
    """§11: "Index in the list *is* the degree" -- for every list, including
    lists with empty degrees between populated ones."""
    dgms = [
        np.array([[b, d] for _, b, d in rows], dtype=np.float64).reshape(len(rows), 2)
        for rows in blocks
    ]

    d = from_ripser(dgms)

    expected = [k for k, rows in enumerate(blocks) for _ in rows]
    assert [int(x) for x in d.dims] == expected
    assert d.n_bars == sum(len(rows) for rows in blocks)


def test_from_giotto_populates_backend_identity(giotto_array: Any) -> None:
    """§8, §11: every adapter records these, and every sample carries them."""
    b = from_giotto(
        giotto_array(reduced=True, sample="batch"),
        reduced_homology=True,
        infinity_values=math.inf,
        strip_padding=False,
    )

    for i in range(len(b)):
        assert b[i].meta.backend == "giotto"
        assert b[i].meta.backend_version == installed_version("giotto-tda")
        assert_json_representable(b[i].meta.provenance)


def test_no_adapter_aliases_an_array_the_caller_keeps(
    gudhi_intervals: Any, ripser_dgms: Any, giotto_array: Any
) -> None:
    """I8's third obligation, stated over the adapters by name: "Every
    **public** construction path -- the `PersistenceDiagram` constructor,
    every `from_*` adapter, and `DiagramBatch.from_diagrams` -- MUST therefore
    copy the arrays it is given rather than store them."

    The hole I8 was added for is the one no method of ours can close: a caller
    who passes an array, keeps their reference, and writes through it
    afterwards has mutated a constructed diagram without any of our code
    having run. §3.1 spends three bullets on it because `frozen=True` stops
    `d.births = other` and stops nothing about `d.births[0] = 5.0`.

    Asserted at the adapter boundary rather than trusted to the constructor
    tests, which is the difference between a property and a coincidence.
    `core.py` has internal paths that deliberately do not copy -- `__getitem__`
    aliases a batch's buffer on purpose, and `from_diagrams` reuses its own
    concat output -- so "the adapters are safe" is a statement about which
    path each one happens to take today, and the four here take three
    different ones.

    `from_giotto` is checked through both the batch's buffers and a member
    diagram's arrays: §4.2 makes the second a view onto the first, so a copy
    that had been skipped would show in either."""
    intervals = np.array(gudhi_intervals("circle", 1), copy=True)
    dgms = [np.array(block, copy=True) for block in ripser_dgms("circle")]
    batch_input = np.array(giotto_array(reduced=True, sample="batch"), copy=True)
    table = np.array([[0.0, 1.0, 0.0], [0.5, 2.0, 1.0]])

    from_gudhi_out = from_gudhi(intervals, dim=1)
    from_ripser_out = from_ripser(dgms)
    from_persim_out = from_persim(dgms)
    from_array_out = from_array(table)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from_giotto_out = from_giotto(
            batch_input, reduced_homology=True, infinity_values=math.inf
        )

    expected = {
        "gudhi": [float(x) for x in from_gudhi_out.births],
        "ripser": [float(x) for x in from_ripser_out.births],
        "persim": [float(x) for x in from_persim_out.births],
        "array": [float(x) for x in from_array_out.births],
        "giotto_batch": [float(x) for x in from_giotto_out.births],
        "giotto_member": [float(x) for x in from_giotto_out[0].births],
    }

    intervals[:, 0] = -99.0
    dgms[0][:, 0] = -99.0
    table[:, 0] = -99.0
    batch_input[:, :, 0] = -99.0

    assert [float(x) for x in from_gudhi_out.births] == expected["gudhi"]
    assert [float(x) for x in from_ripser_out.births] == expected["ripser"]
    assert [float(x) for x in from_persim_out.births] == expected["persim"]
    assert [float(x) for x in from_array_out.births] == expected["array"]
    assert [float(x) for x in from_giotto_out.births] == expected["giotto_batch"]
    assert [float(x) for x in from_giotto_out[0].births] == expected["giotto_member"]
    assert -99.0 not in expected["array"], "the mutation must be observable at all"


# ---------------------------------------------------------------------------
# Malformed input -- §11 fixes what each adapter accepts, so everything else
# has to fail as a refusal that names the expected form, not as whatever
# exception the first arithmetic on bad data happens to raise. An opaque
# `ValueError: too many values to unpack` is a bug report we cannot action and
# a user cannot act on.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("adapter", [from_gudhi, from_ripser, from_persim])
def test_list_adapters_reject_a_string(adapter: Any) -> None:
    """`str` and `bytes` are `Sequence`s, so a type gate spelled as
    `isinstance(obj, Sequence)` admits them and then fails somewhere inside the
    row loop. §11 accepts a sequence *of rows*; a string is never that."""
    for text in ("hello", b"hello", bytearray(b"hello")):
        with pytest.raises(TypeError, match=r"got (str|bytes|bytearray)"):
            adapter(text)


@pytest.mark.parametrize(
    ("rows", "bad_row"),
    [
        ([(0, 1.0, 2.0)], 0),  # the nested pair flattened
        ([(0, [1.0, 2.0, 3.0])], 0),  # three coordinates in the pair
        ([(0, (0.0, 1.0)), (1, 2.0)], 1),  # one good row, one not
        ([0.5], 0),  # not a row at all
        ([(0, (0.0, 1.0))] * 4 + [(1, (2.0,))], 4),  # the last of five
    ],
)
def test_from_gudhi_rejects_a_malformed_persistence_row(
    rows: Any, bad_row: int
) -> None:
    """§11: `persistence()` returns `list[(dim, (birth, death))]`. Anything
    else must be refused by name, and the refusal must say which row, since a
    caller who mis-shaped one row of ten cannot find it otherwise.

    The expected index is asserted, not merely the presence of one. A `row
    \\d+` pattern passes against an implementation that reports `row 0` for
    every input, which is the one thing the index is there to rule out -- and
    an off-by-one in the enumeration would read as correct on any suite whose
    bad row is always the first."""
    with pytest.raises(
        ValueError, match=rf"(?s)row {bad_row}\b.*\(dim, \(birth, death\)\)"
    ):
        from_gudhi(rows)


def test_from_gudhi_rejects_the_extended_persistence_list() -> None:
    """§11: `extended_persistence()` is a third input form and is out of scope.

    It returns a four-element **list of lists** of
    `list[(dim, (birth, death))]` -- ordinary,
    relative, extended+ and extended- -- structurally distinct from
    `persistence()`'s flat list, so this one it can actually detect. The
    refusal must name the scope exclusion rather than the shape: told that
    row 0 is mis-shaped, a caller goes hunting for a typo in data that is
    exactly what GUDHI handed them.

    `TypeError` rather than `ValueError` because this is an input *form* the
    adapter does not accept, which is the category `from_gudhi`'s existing
    fallthrough already raises `TypeError` for. §11 fixes that the outer list is
    refused and what the message names; it does not fix the type."""
    extended = [
        [(0, (0.0, 1.0))],  # ordinary
        [(1, (3.0, 2.0))],  # relative -- death < birth by construction
        [(1, (0.5, 2.5))],  # extended+
        [(0, (2.0, 0.5))],  # extended- -- death < birth by construction
    ]

    with pytest.raises(TypeError, match="extended persistence"):
        from_gudhi(extended)


def test_from_gudhi_still_accepts_a_four_row_persistence_list() -> None:
    """The guard on the test above. `extended_persistence()` is detected as a
    four-element outer list, and a `persistence()` result with four bars is
    also four things long -- so a rejection keyed on length alone would refuse
    ordinary GUDHI output. What separates them is that the extended members
    are *lists of rows* and a `persistence()` row is `(dim, (birth, death))`."""
    four_bars = [
        (0, (0.0, 1.0)),
        (0, (0.0, 2.0)),
        (1, (0.5, 1.5)),
        (0, (0.0, math.inf)),
    ]

    d = from_gudhi(four_bars)

    assert [int(x) for x in d.dims] == [0, 0, 1, 0]
    assert math.isinf(float(d.deaths[3]))


@pytest.mark.parametrize("n_bars", [3, 4, 5])
def test_a_persistence_list_of_list_rows_is_accepted_at_every_length(
    n_bars: int,
) -> None:
    """The detector discriminates on member *shape*, not on cardinality.

    `_columns_from_pairs` unpacks a row by sequence, so a `persistence()`
    result whose rows are lists rather than tuples is accepted -- which is
    what a result that has been through a serializer looks like, and §11.2's
    frozen fixtures are exactly that. A rejection keyed on "four members, all
    lists" made that form legal at three bars and at five and illegal at
    four, so whether an input was accepted depended on how many bars it
    happened to carry. §4 rules that dependence out inside an array; it has no
    more business in the adapter's gate.

    GUDHI itself returns tuple rows, so nothing here changes what live backend
    output does -- `test_live_gudhi_extended_persistence_is_rejected` pins
    that end."""
    rows = [[k % 2, [0.0, float(k) + 1.0]] for k in range(n_bars)]

    d = from_gudhi(rows)

    assert d.n_bars == n_bars
    assert [int(x) for x in d.dims] == [k % 2 for k in range(n_bars)]


@pytest.mark.parametrize("n_bars", [3, 4, 5])
def test_a_persistence_list_whose_intervals_are_arrays_is_accepted_at_every_length(
    n_bars: int,
) -> None:
    """The same rule as the test above, for the spelling it did not cover.

    `_is_persistence_row` decided a row by asking whether its interval was a
    two-element `Sequence`, and `numpy.ndarray` is not a registered
    `Sequence`. So `[dim, array([b, d])]` was not a row, four of them were
    four members that are lists and not rows, and the detector called that
    extended persistence -- while three and five of the identical thing
    constructed cleanly. That is the cardinality-dependent acceptance §4
    rules out and `_is_extended_persistence` is written to avoid, surviving in
    the one input spelling the suite did not state.

    Array intervals are the same category of input as the list rows above:
    not what GUDHI returns, and what a `persistence()` result looks like once
    something has rebuilt it. `test_live_gudhi_extended_persistence_is_rejected`
    pins the live end."""
    rows = [[k % 2, np.array([0.0, float(k) + 1.0])] for k in range(n_bars)]

    d = from_gudhi(rows)

    assert d.n_bars == n_bars
    assert [int(x) for x in d.dims] == [k % 2 for k in range(n_bars)]
    assert [float(x) for x in d.deaths] == [float(k) + 1.0 for k in range(n_bars)]


def test_a_four_row_persistence_list_reports_its_own_bad_coordinate() -> None:
    """The detector must not answer a question about values. §11.

    `_is_persistence_row` is documented as structural -- it separates a row
    from a list of rows and leaves admissibility to `_columns_from_pairs` --
    but it decided an interval by asking whether both entries were
    `numbers.Real`, which is a question about values. A row whose coordinates
    are strings therefore stopped being a row, and four such rows became
    "extended persistence": a message about scope for an input whose actual
    defect is a string where a filtration value belongs, and only ever at
    four."""
    rows = [[0, ["a", "b"]] for _ in range(4)]

    with pytest.raises(TypeError, match="real filtration value"):
        from_gudhi(rows)


def test_extended_persistence_is_rejected_however_its_rows_are_spelled() -> None:
    """The guard on the test above: widening what counts as a row must not
    narrow what counts as extended persistence. A four-element list whose
    members are lists *of* rows is still the form §11 excludes, whether those
    rows are tuples or lists."""
    extended = [
        [[0, [0.0, 1.0]]],  # ordinary
        [[1, [3.0, 2.0]]],  # relative -- death < birth by construction
        [[1, [0.5, 2.5]]],  # extended+
        [[0, [2.0, 0.5]]],  # extended- -- death < birth by construction
    ]

    with pytest.raises(TypeError, match="extended persistence"):
        from_gudhi(extended)


def test_extended_persistence_is_rejected_when_its_intervals_are_arrays() -> None:
    """The guard on widening what counts as an interval.

    Admitting a rank-1 two-element array as a `(birth, death)` pair must not
    make an extended member look like a row. It does not, and for the reason
    that keeps the whole detector coherent: a member is a *list of* rows, so
    a member of two rows has a row where an interval would have to be, and a
    row is not a pair of coordinates however either is spelled."""
    extended = [
        [[0, np.array([0.0, 1.0])], [0, np.array([0.5, 2.0])]],  # ordinary
        [[1, np.array([3.0, 2.0])], [1, np.array([4.0, 1.0])]],  # relative
        [[1, np.array([0.5, 2.5])], [1, np.array([0.6, 2.6])]],  # extended+
        [[0, np.array([2.0, 0.5])], [0, np.array([3.0, 0.5])]],  # extended-
    ]

    with pytest.raises(TypeError, match="extended persistence"):
        from_gudhi(extended)


def test_an_empty_extended_persistence_result_is_still_rejected() -> None:
    """A complex with nothing in it returns four empty sub-diagrams. The form
    is what §11 excludes, not the contents, so an empty one is refused for the
    same reason -- and an empty member is not a row, which is what keeps the
    two clauses of the detector agreeing."""
    with pytest.raises(TypeError, match="extended persistence"):
        from_gudhi([[], [], [], []])


def test_from_gudhi_rejects_a_flat_tuple_of_rows() -> None:
    with pytest.raises(TypeError, match="tuple"):
        from_gudhi(((0, (0.0, 1.0)),))


@pytest.mark.parametrize("bad_dim", [1.5, "2", True, 2.0])
def test_dim_must_be_an_integral_degree(bad_dim: Any) -> None:
    """I3: a homological degree is an integer, and `dim=` is the caller's only
    way to state one for an `(n, 2)` array. `int(1.5)` is 1 -- a diagram that
    is clean, plausible and wrong, which is the failure mode `_as_dims` refuses
    for the degree *column* and which must not survive on this path either.

    `True` is an `int` in Python and would silently mean degree 1; `2.0` is
    integral but is not how a degree is spelled at a call site."""
    with pytest.raises(TypeError, match="dim"):
        from_array(np.array([[0.0, 1.0]]), dim=bad_dim)


def test_from_ripser_rejects_a_dgms_that_is_not_a_list() -> None:
    """§11: `ripser(X)["dgms"]` is a list of `(n, 2)` arrays, degree by index.
    The key's presence is not the same fact as the key's shape."""
    with pytest.raises(TypeError, match="dgms"):
        from_ripser({"dgms": 5})


@pytest.mark.parametrize("outer", [(), (np.array([[0.0, 1.0]]),)])
def test_from_ripser_rejects_tuple_outer_forms(outer: tuple[Any, ...]) -> None:
    """§11: both direct and mapping forms require an outer list."""
    with pytest.raises(TypeError, match="list"):
        from_ripser(outer)
    with pytest.raises(TypeError, match="list"):
        from_ripser({"dgms": outer})


@pytest.mark.parametrize("outer", [(), (np.array([[0.0, 1.0]]),)])
def test_from_persim_rejects_tuple_outer_forms(outer: tuple[Any, ...]) -> None:
    """§11: persim's degree-list outer container is also exactly a list."""
    with pytest.raises(TypeError, match="list"):
        from_persim(outer)


def test_degree_lists_scan_past_python_blocks_for_first_array_namespace() -> None:
    """§3.3: Python/empty blocks convert into the first real array namespace."""
    xps = pytest.importorskip("array_api_strict")
    strict = xps.asarray([[0.0, 1.0]], dtype=xps.float64)
    blocks = [[], [[0.0, 0.5]], strict]

    ripser_diagram = from_ripser(blocks)
    persim_diagram = from_persim(blocks)

    for diagram in (ripser_diagram, persim_diagram):
        assert diagram.xp is xps
        assert [int(x) for x in diagram.dims] == [1, 2]
        assert [float(x) for x in diagram.births] == [0.0, 0.0]
        assert [float(x) for x in diagram.deaths] == [0.5, 1.0]


def test_degree_lists_reject_a_later_real_namespace_mismatch() -> None:
    """§3.3/I7: scanning all blocks cannot silently coerce a later array."""
    xps = pytest.importorskip("array_api_strict")
    strict = xps.asarray([[0.0, 1.0]], dtype=xps.float64)
    blocks = [[], strict, np.array([[0.0, 2.0]])]

    with pytest.raises(ValueError, match="namespace"):
        from_ripser(blocks)
    with pytest.raises(ValueError, match="namespace"):
        from_persim(blocks)


def test_a_degree_list_must_share_one_namespace() -> None:
    """I7: one diagram has one array namespace. `core.py` checks this across
    the diagrams of a batch, for a reason that applies just as much within one
    diagram -- concatenating across namespaces "would either raise something
    opaque from the backend or silently coerce a foreign array"."""
    xps = pytest.importorskip("array_api_strict")
    mixed = [xps.asarray([[0.0, 1.0]], dtype=xps.float64), np.array([[0.0, 2.0]])]

    with pytest.raises(ValueError, match="namespace"):
        from_ripser(mixed)


# ---------------------------------------------------------------------------
# §3.1's clamp -- what the warning says, and how often it says it
# ---------------------------------------------------------------------------


def test_clamp_warning_reports_the_largest_gap_it_absorbed() -> None:
    """The magnitude in the warning is what the sentence around it calls
    "floating-point noise", so it has to be a gap that was actually absorbed
    and it has to be within the threshold that made absorbing it legitimate."""
    spacing = 1.0 - math.nextafter(1.0, -math.inf)
    rows = np.array([[1.0, math.nextafter(1.0, 0.0)], [1.0, 1.0 - 4 * spacing]])

    with pytest.warns(UserWarning, match="I6") as record:
        d = from_array(rows, dim=0)

    message = str(record[0].message)
    reported = float(
        message.split("the largest by ", 1)[1].split(" These", 1)[0].rstrip(".")
    )
    assert math.isclose(reported, 4 * spacing, rel_tol=2e-3), (
        "the larger of the two absorbed local-ULP gaps"
    )
    assert d.meta.provenance["clamped_rows"] == 2


def test_a_refused_diagram_warns_about_nothing() -> None:
    """§3.1's warning says what was absorbed into a diagram. When the diagram
    is refused -- a real I6 violation alongside a clamped one -- nothing was
    absorbed into anything, and a warning would be describing repairs to an
    object that does not exist, quoting a magnitude next to the word "noise"
    that the very same call is raising over."""
    rows = np.array([[1.0, math.nextafter(1.0, 0.0)], [0.0, -5.0]])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="I6"):
            from_array(rows, dim=0)


def test_clamp_warning_points_at_the_callers_line() -> None:
    """A warning attributed to a line inside `adapters.py` tells the user
    nothing: they cannot act on our line numbers, and `stacklevel` exists so
    that they do not have to."""
    birth = 1.0
    death = math.nextafter(birth, 0.0)

    with pytest.warns(UserWarning, match="I6") as record:
        from_array(np.array([[birth, death]]), dim=0)

    assert record[0].filename == __file__, (
        f"attributed to {record[0].filename}:{record[0].lineno}, which is not "
        "the caller's line"
    )


def test_from_giotto_warns_once_per_call_about_clamping() -> None:
    """The same argument §11.1 settles for padding: warning per sample makes
    the count a property of the batch shape rather than of the data, and a
    500-sample batch with systematic filtration rounding would emit 500
    identical warnings."""
    arr = np.zeros((4, 2, 3))
    arr[:, :, 0] = 1.0
    arr[:, :, 1] = math.nextafter(1.0, 0.0)

    with pytest.warns(UserWarning, match="I6") as record:
        b = from_giotto(
            arr, reduced_homology=True, infinity_values=math.inf, strip_padding=False
        )

    assert len([w for w in record if "I6" in str(w.message)]) == 1
    assert all(b[i].meta.provenance["clamped_rows"] == 2 for i in range(len(b)))


def test_from_giotto_aggregates_its_one_clamp_warning_over_the_whole_batch() -> None:
    """The single warning §3.1 owes the caller must describe the whole call.

    A homogeneous batch cannot show that: every candidate arithmetic -- the
    sum, the last sample's count, the first's -- agrees when the samples are
    identical, so a suite built on one passes against an adapter that reports
    whichever sample it saw last. The samples here disagree deliberately, in
    all three quantities the warning states: how many rows were repaired
    (2 + 1 + 0 = 3), out of how many (2 + 2 + 2 = 6), and the largest gap
    absorbed, which belongs to the middle sample alone.

    §3.1's `worst` is a maximum over *repaired* rows, so the untouched I6-
    clean rows in the third sample must not enter it, and neither may any gap
    the adapter left for the core type to raise on."""
    small = math.nextafter(1.0, 0.0)
    big = math.nextafter(1000.0, 0.0)  # a wider absolute gap, still noise at 1e3
    arr = np.array(
        [
            [[1.0, small, 0.0], [1.0, small, 1.0]],  # two repaired, narrow gaps
            [[1000.0, big, 0.0], [0.0, 1.0, 0.0]],  # one repaired, the worst gap
            [[0.0, 1.0, 0.0], [2.0, 3.0, 1.0]],  # nothing to repair
        ]
    )

    with pytest.warns(UserWarning, match="I6") as record:
        b = from_giotto(
            arr, reduced_homology=False, infinity_values=math.inf, strip_padding=False
        )

    messages = [str(w.message) for w in record if "I6" in str(w.message)]
    assert len(messages) == 1
    assert "clamped 3 of 6 rows" in messages[0], messages[0]

    # The largest gap absorbed is the 1000.0 one, ~1.1e-13, not the ~1.1e-16
    # gaps in the first sample. Read back off the message rather than
    # recomputed, so a per-sample maximum that reported the first sample's
    # figure would fail here.
    reported = messages[0].split("the largest by ")[1].split(" ")[0].rstrip(".")
    assert math.isclose(float(reported), 1000.0 - big, rel_tol=1e-2), reported

    assert [b[i].meta.provenance["clamped_rows"] for i in range(len(b))] == [2, 1, 0]


def test_from_giotto_does_not_strip_a_row_that_violates_an_invariant() -> None:
    """§11.1's `strip_padding` drops giotto's padding, which is `(b, b, dim)`
    for a *finite* b. An `(inf, inf, dim)` row is an I4 violation, and §3.1
    says a violation is surfaced rather than absorbed -- so it must raise
    whatever `strip_padding` says, not vanish into `padding_removed`."""
    arr = np.array([[[math.inf, math.inf, 0.0], [0.0, 1.0, 0.0]]])

    for strip_padding in (None, True, False):
        with pytest.raises(ValueError, match="I4"):
            from_giotto(
                arr,
                reduced_homology=True,
                infinity_values=math.inf,
                strip_padding=strip_padding,
            )


@pytest.mark.parametrize("strip_padding", [None, True, False])
def test_from_giotto_does_not_strip_a_row_the_clamp_made_trivial(
    strip_padding: bool | None,
) -> None:
    """§11.1's padding is decided on the rows giotto emitted, so §3.1's clamp
    cannot create one.

    giotto pads with `(b, b, dim)` (§4, A.2). A row that arrives as
    `(b, b - 1ulp, dim)` is not that row: it is an I6 violation at the noise
    level, which §3.1 makes the adapter's to repair, and repairing it lands
    the death exactly on the birth. Whether the repaired row is then treated
    as padding is the question, and the answer has to be no -- `padding_removed`
    would otherwise count a row giotto never padded with, and §11.1's key
    "records what was actually removed" of the *input*.

    Every existing clamp test on this adapter passes `strip_padding=False`, so
    nothing pinned which side of the mask the clamp falls on. Both orders run
    clean and they disagree: clamping first makes the repaired row trivial and
    strippable, and the row vanishes into a count naming padding.

    All three modes are asserted together for the reason the degree-validation
    test above gives -- the hazard is that they disagree about the same array.
    The counts are the whole assertion: one genuine trivial row, one repaired
    row that is not one, and a clamp the provenance still records."""
    noise = math.nextafter(1.0, 0.0)
    arr = np.array(
        [
            [
                [1.0, noise, 0.0],  # I6 noise; trivial only after the repair
                [2.0, 2.0, 0.0],  # giotto's own padding
                [0.0, 5.0, 1.0],  # an ordinary bar
            ]
        ]
    )
    stripping = strip_padding is True

    # Recorded rather than `pytest.warns`, because the default mode owes a
    # second warning about the padding it kept and only this mode does. The
    # clamp warning is asserted for all three; §11.1's is the test below.
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        d = from_giotto(
            arr,
            reduced_homology=False,
            infinity_values=math.inf,
            strip_padding=strip_padding,
        )[0]

    assert [w for w in record if "I6" in str(w.message)], "the repair must warn"
    assert d.n_bars == (2 if stripping else 3)
    assert d.meta.provenance["padding_removed"] == (1 if stripping else 0)
    assert d.meta.provenance["clamped_rows"] == 1

    # The repaired row is present in every mode, and it is now trivial -- which
    # is exactly why it would have been stripped had the clamp run first.
    rows = [(float(b), float(x)) for b, x in zip(d.births, d.deaths, strict=True)]
    assert (1.0, 1.0) in rows


def test_the_giotto_padding_warning_counts_the_rows_before_the_clamp() -> None:
    """The other half of the test above, on §11.1's default mode.

    `strip_padding=None` "warns once if any trivial rows are present", and
    what makes a row present is the same question the mask asks. A repaired
    row is trivial in the constructed diagram and was not trivial in giotto's
    output, so the count the warning reports is the input's -- otherwise the
    sentence telling the caller to "pass strip_padding=True to drop them"
    names a row that mode does not drop.

    Read off the message rather than recomputed, so a count taken after the
    repair reports 2 and fails here."""
    noise = math.nextafter(1.0, 0.0)
    arr = np.array([[[1.0, noise, 0.0], [2.0, 2.0, 0.0], [0.0, 5.0, 1.0]]])

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        from_giotto(arr, reduced_homology=False, infinity_values=math.inf)

    trivial = [str(w.message) for w in record if "trivial" in str(w.message)]
    assert len(trivial) == 1
    assert "carries 1 trivial rows" in trivial[0], trivial[0]


@pytest.mark.parametrize(
    ("degree", "expected"),
    [
        (-1.0, "non-negative"),  # I3
        (0.5, "non-integral"),  # I3
        (math.nan, "non-finite"),  # I3
        (float(2**31), "int32"),  # I2
    ],
)
def test_from_giotto_validates_degrees_before_it_strips_padding(
    degree: float, expected: str
) -> None:
    """The same argument as the test above, applied to the third column.

    A row is giotto's padding when it is a row giotto could have emitted:
    `(b, b, dim)` for a finite `b` **and a real homological degree**. `(0, 0,
    -1)` and `(0, 0, nan)` satisfy the birth/death half and nothing else, so
    they are a corrupt array rather than padding, and §3.1's answer to a
    violation is to surface it.

    All three `strip_padding` modes are asserted together, because the hazard
    is precisely that they disagree: masking the rows out before validating
    lets `strip_padding=True` delete the evidence and count the deletion as
    padding, while the other two modes raise on the same input. §11.1 lets the
    caller decide whether giotto's padding is kept, not whether their array is
    checked."""
    arr = np.array([[[0.0, 0.0, degree], [0.0, 1.0, 0.0]]])

    for strip_padding in (None, True, False):
        with pytest.raises(ValueError, match=expected):
            from_giotto(
                arr,
                reduced_homology=False,
                infinity_values=math.inf,
                strip_padding=strip_padding,
            )


@pytest.mark.parametrize("flag", ["False", "", 0, 1, None, object()])
def test_from_giotto_requires_a_real_boolean_for_reduced_homology(flag: Any) -> None:
    """§5.1, §11: `reduced_homology` is required because no property of the
    array can confirm or contradict what it claims -- which is exactly why a
    truthy stand-in for it cannot be allowed to pass.

    `reduced_homology="False"` is truthy, so a duck-typed read records
    `params={"reduced_homology": True}` and `essential_bars="lost_upstream"`
    on a diagram whose caller meant the opposite. Nothing downstream can catch
    that: §8's whole premise is that a consumer trusts these keys because the
    adapter is the one party that saw the backend's output."""
    with pytest.raises(TypeError, match="reduced_homology"):
        from_giotto(
            np.array([[[0.0, 1.0, 0.0]]]),
            reduced_homology=flag,
            infinity_values=math.inf,
        )


@pytest.mark.parametrize("flag", ["False", "True", "", 0, 1, object()])
def test_from_giotto_requires_a_real_boolean_for_strip_padding(flag: Any) -> None:
    """§11.1 fixes three modes -- `None`, `True`, `False` -- and a truthy
    stand-in changes the data, not just the record: `strip_padding="False"` is
    both not-`None` and truthy, so it strips every trivial row from a call
    that asked for none to be stripped, and records the count as though the
    caller had asked."""
    arr = np.array([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])

    with pytest.raises(TypeError, match="strip_padding"):
        from_giotto(
            arr, reduced_homology=False, infinity_values=math.inf, strip_padding=flag
        )


def test_from_giotto_accepts_a_batch_with_no_samples() -> None:
    """§4.2: "An empty batch is perfectly valid". A filter that selects no
    samples is an ordinary outcome, not an error to be raised at the adapter."""
    b = from_giotto(
        np.zeros((0, 3, 3)), reduced_homology=True, infinity_values=math.inf
    )

    assert len(b) == 0
    assert b.xp is np


# ---------------------------------------------------------------------------
# I2: what the storage dtypes will and will not hold
#
# Every clause here guards the same failure: a conversion that succeeds on a
# value it cannot represent. `astype` and the builtins agree on reporting
# nothing, so each of these inputs would otherwise become a diagram that is
# clean, plausible and wrong (§9) rather than an input that was refused.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("degree", [1.5, "1", True, 2.0, None, 1 + 0j])
def test_from_gudhi_refuses_a_coerced_degree_in_the_persistence_list(
    degree: Any,
) -> None:
    """I3, §11: the `persistence()` list is the one input form whose degrees
    reach storage without passing through an array, and a bare `int()` there
    reopens on this path what `_as_dims` closes on every other. `int(1.5)` is
    1 and `int("1")` is 1: a degree the caller never wrote, in a diagram that
    validates cleanly.

    `True` is an `int` in Python and would mean degree 1, which is a
    coincidence of the language rather than anything GUDHI emitted."""
    with pytest.raises(TypeError, match=r"degree in row 0"):
        from_gudhi([(degree, (0.0, 1.0))])


@pytest.mark.parametrize("value", ["0.5", True, None, 1 + 0j])
def test_from_gudhi_refuses_a_coerced_coordinate_in_the_persistence_list(
    value: Any,
) -> None:
    """§6.1, and the argument above applied to the other two columns:
    `float("0.5")` is 0.5 and `float(True)` is 1.0, so a row holding a string
    or a flag where a filtration value belongs would become a birth or a death
    the caller never wrote."""
    with pytest.raises(TypeError, match=r"(birth|death) in row 0"):
        from_gudhi([(0, (value, 2.0))])
    with pytest.raises(TypeError, match=r"(birth|death) in row 0"):
        from_gudhi([(0, (0.0, value))])


def test_from_gudhi_accepts_a_degree_that_is_integral_but_not_an_int() -> None:
    """The refusal above must not catch the ordinary case. GUDHI's own output
    is Python `int`s, but a caller assembling rows from an array hands over
    `numpy.int64` degrees and `numpy.float32` coordinates, which are an
    `Integral` and a `Real` respectively and are exactly representable in the
    storage dtypes I2 fixes."""
    d = from_gudhi([(np.int64(1), (np.float32(0.5), np.float32(2.0)))])

    assert [int(x) for x in d.dims] == [1]
    assert [float(x) for x in d.births] == [0.5]


@pytest.mark.parametrize("degree", [2**31, 2**32, -(2**31) - 1])
def test_a_degree_outside_int32_is_refused_by_every_path(degree: int) -> None:
    """I2 fixes `int32` as the degree dtype, so a degree outside its range is
    not one this type can hold, and every route to storage must say so.

    The routes disagree about *how* they go wrong, which is why all three are
    asserted rather than one: numpy wraps an out-of-range `int64` (2**32
    arrives as 0, a perfectly valid H0 label) and saturates an out-of-range
    `float64` (2**32 arrives as 2147483647), so an unchecked cast produces two
    different wrong answers depending on the input's dtype, and only one of
    them is implausible enough for a reader to question."""
    with pytest.raises(ValueError, match="int32"):
        from_array(np.array([[0.0, 1.0]]), dim=degree)
    with pytest.raises(ValueError, match="int32"):
        from_array(np.array([[0.0, 1.0, float(degree)]]))
    with pytest.raises(ValueError, match="int32"):
        from_array(np.array([[0, 1, degree]], dtype=np.int64))
    with pytest.raises(TypeError, match=r"degree in row 0"):
        # `_as_degree`'s bound, reached through the other spelling of a
        # caller-stated degree. Refused as an int32 overflow, not as a
        # mis-shaped row.
        from_gudhi([(float(degree), (0.0, 1.0))])


def test_an_integer_coordinate_outside_float64s_exact_range_is_refused() -> None:
    """C2. §6.1 stores coordinates as `float64`, which holds every integer up
    to `2**53` exactly and only some of them above.

    The cast runs before §3.1's invariants, so above that bound the rounding
    can erase the violation the check exists to catch: `2**53 + 1` and `2**53`
    are an I6 violation as integers and the same float afterwards, so the row
    would be stored as a zero-persistence bar rather than raising. This is
    `_require_int32_range`'s argument one column over -- the cast does not
    report -- and both routes to storage must refuse it."""
    beyond = 2**53 + 1

    with pytest.raises(ValueError, match=r"2\*\*53"):
        from_array(np.array([[beyond, 2**53]], dtype=np.int64), dim=0)
    with pytest.raises(ValueError, match=r"2\*\*53"):
        from_gudhi([(0, (beyond, 2**53))])
    with pytest.raises(ValueError, match=r"2\*\*53"):
        from_persim([np.array([[0, beyond]], dtype=np.int64)])
    with pytest.raises(ValueError, match=r"2\*\*53"):
        from_array(np.array([[-beyond, 0]], dtype=np.int64), dim=0)


def test_an_integer_coordinate_at_the_float64_boundary_is_accepted() -> None:
    """The bound is inclusive: `2**53` is exactly representable, so refusing it
    would reject an input nothing goes wrong with. Fixing the boundary in a
    test is the point -- an off-by-one here silently changes which diagrams are
    constructible."""
    d = from_array(np.array([[2**53 - 1, 2**53]], dtype=np.int64), dim=0)

    assert [float(x) for x in d.births] == [float(2**53 - 1)]
    assert [float(x) for x in d.deaths] == [float(2**53)]


def test_a_float_coordinate_beyond_the_exact_range_is_still_accepted() -> None:
    """The guard is about *integral* input, which is exactly representable and
    which we would be the ones to damage. A float that large was already
    rounded onto the grid by whoever built it, and there is nothing left to
    detect -- refusing it would reject ordinary float64 data for the sake of a
    check that cannot help it."""
    d = from_array(np.array([[0.0, 1e300]]), dim=0)

    assert [float(x) for x in d.deaths] == [1e300]


@pytest.mark.parametrize("degree", [0, 1, 2**31 - 1])
def test_a_degree_inside_int32_survives_every_path(degree: int) -> None:
    """The bound is `int32`'s, so its largest value is admissible. A check
    written with `<` where `<=` belongs would pass every test above and lose
    the boundary, and a caller cannot tell a refused degree from an impossible
    one."""
    assert int(from_array(np.array([[0.0, 1.0]]), dim=degree).dims[0]) == degree
    assert int(from_array(np.array([[0.0, 1.0, float(degree)]])).dims[0]) == degree
    assert int(from_gudhi([(degree, (0.0, 1.0))]).dims[0]) == degree


@pytest.mark.parametrize(
    "arr",
    [
        np.array([[0.0 + 1.0j, 1.0 + 2.0j]]),  # complex coordinates
        np.array([[True, True]]),  # a boolean array of "coordinates"
    ],
)
def test_from_array_refuses_a_dtype_that_cannot_convert_losslessly(
    arr: np.ndarray,
) -> None:
    """§6.1 stores `float64`, and `astype` reaches it from a complex dtype by
    discarding the imaginary part -- reported as a `ComplexWarning`, which a
    caller who filtered warnings never sees, and which leaves a diagram whose
    births are the real parts of numbers that were never real. From `bool` it
    reaches it by writing 0.0 and 1.0.

    Both are refused at the boundary rather than converted, for §9's reason:
    the resulting diagram is well-formed, passes every invariant, and is not
    the caller's data."""
    with pytest.raises(TypeError, match="dtype"):
        from_array(arr, dim=0)


def test_a_complex_degree_column_is_refused_before_it_is_rounded() -> None:
    """The same clause on the third column. `isfinite` and `round` are both
    defined on complex values, so the degree column's existing integrality
    check passes a `1+2j` degree straight through to a cast that keeps the
    real part."""
    with pytest.raises(TypeError, match="dtype"):
        from_array(np.array([[0.0, 1.0, 1.0 + 2.0j]]))


# ---------------------------------------------------------------------------
# §8: what may be recorded
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        np.int64(3),  # the scalar §8 names
        np.float64(1.5),  # a `float` subclass: `isinstance` would admit it
        np.float32(1.5),
        (1, 2),  # JSON-dumpable, but reloads as a list
        {1: "a"},  # not str-keyed
        [1, [2, np.int64(3)]],  # nested
        object(),
    ],
)
def test_an_adapter_refuses_metadata_that_cannot_be_saved(value: Any) -> None:
    """§8: "Every value in `params` and `provenance` MUST be
    JSON-representable", because §10.2 stores both as UTF-8 JSON.

    §8 gives the reason this belongs at the adapter and not at `save()`: a
    mapping holding a `Path` or a NumPy scalar "is a diagram that satisfies
    §3.1 and §8 completely and cannot be saved", and the failure then surfaces
    "arbitrarily far from the adapter that wrote the offending value".

    `numpy.float64` is the case an `isinstance` gate misses -- it subclasses
    `float` -- and the tuple is the case `json.dumps` misses, since it dumps
    happily and reloads as a list, failing §10.1 requirement 1's
    `load(dump(d)) == d` instead of failing to write."""
    with pytest.raises(TypeError, match=r"JSON-representable|str-keyed"):
        from_array(np.array([[0.0, 1.0]]), dim=0, provenance={"k": value})
    with pytest.raises(TypeError, match=r"JSON-representable|str-keyed"):
        from_array(np.array([[0.0, 1.0]]), dim=0, params={"k": value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("filtration", 3.5),
        ("filtration", object()),
        ("backend_version", 2),
        ("description", 42),
        ("description", b"a torus"),
        ("coeff_field", 2.0),
        ("coeff_field", True),
        ("coeff_field", "two"),
    ],
)
def test_an_adapter_refuses_a_scalar_metadata_field_of_the_wrong_type(
    field: str, value: Any
) -> None:
    """§8 types five `DiagramMeta` fields as scalars, and `**meta` is the door
    every one of them arrives through.

    The check lives in `DiagramMeta.__post_init__` rather than in five
    adapters -- see the core suite -- and this is the assertion that it is
    actually reachable from the adapter surface, which is where a caller
    types the value. `from_array` stands for all five: `_build_meta` hands
    `**meta` to the same constructor on every path.

    `backend` is absent from the cases because §11 refuses it earlier and for
    a different reason: it is "the fact that says where this diagram came
    from", so an adapter rejects any value of it, well-typed or not."""
    with pytest.raises(TypeError, match=field):
        from_array(np.array([[0.0, 1.0]]), dim=0, **{field: value})


def test_an_adapter_still_accepts_the_scalar_fields_a_caller_may_state() -> None:
    """The refusal above must not reach past its target. §8's opening
    concession is that every field is optional and caller-supplied text is
    ordinary, and §8 requires the four adapters that cannot know their
    filtration to leave it "at whatever the caller passed through `**meta`"."""
    d = from_array(
        np.array([[0.0, 1.0]]),
        dim=0,
        filtration="alpha",
        description="a noisy circle, 40 points",
        coeff_field=11,
    )

    assert d.meta.filtration == "alpha"
    assert d.meta.description == "a noisy circle, 40 points"
    assert d.meta.coeff_field == 11


def test_an_adapter_narrows_an_array_scalar_coefficient_field_to_a_builtin() -> None:
    """The one place the adapter boundary is deliberately wider than the type.

    `DiagramMeta` requires an exact builtin `int`, on
    `_require_json_representable`'s house rule. `_require_coeff_field` admits
    any `numbers.Integral` and converts, so that a field read out of an array
    -- the ordinary way a caller loops over degrees -- is not refused for
    being an `int64`. The widening is only sound because the conversion
    happens: storing the `int64` unconverted would put a value in
    `coeff_field` that §8's `int | None` does not describe, and that §10.2
    cannot serialise. `type` rather than `==`, since `np.int64(11) == 11`."""
    d = from_array(np.array([[0.0, 1.0]]), dim=0, coeff_field=np.int64(11))

    assert type(d.meta.coeff_field) is int
    assert d.meta.coeff_field == 11


@pytest.mark.parametrize("field", ["params", "provenance"])
@pytest.mark.parametrize(
    "value",
    [
        0,  # falsy: silently became {}
        False,  # falsy
        "",  # falsy
        [],  # falsy
        set(),  # falsy
        1.5,  # truthy, and not iterable at all
        {"a"},  # truthy: `dict()`'s own words, naming nothing
        [("a", 1)],  # truthy, and quietly *accepted* as a mapping
    ],
)
def test_an_adapter_refuses_a_metadata_mapping_that_is_not_a_mapping(
    field: str, value: Any
) -> None:
    """§8 types `params` and `provenance` as `Mapping[str, Any]`, and the
    adapter is where a caller's argument arrives.

    `dict(value or {})` answered three different ways for one mistake. A falsy
    non-mapping -- `0`, `False`, `""`, `[]` -- was silently discarded, so
    `provenance=0` produced a diagram recording nothing and reporting nothing,
    which is the outcome §11 refuses `backend=` and unknown fields to prevent.
    A truthy one raised `dictionary update sequence element #0 has length 1`,
    which names neither the adapter nor the argument. And a sequence of pairs
    was *accepted*, storing a mapping §8's type does not describe from an
    argument that is not one.

    `DiagramMeta(provenance=0)` raises, so the adapter was looser than the
    type it wraps on the one path §11 makes it the boundary for."""
    with pytest.raises(TypeError, match=rf"{field}=.*mapping"):
        from_array(np.array([[0.0, 1.0]]), dim=0, **{field: value})


def test_an_adapter_reads_an_omitted_metadata_mapping_as_unstated() -> None:
    """The other side of the refusal above: `None` stays "stated nothing".

    §8 makes every field optional and spells an absent value `None`, so
    `provenance=None` is the caller saying they have none rather than passing
    a bad mapping -- the same reading `_coeff_field` gives `coeff_field=None`.
    The adapter's own keys are still recorded."""
    d = from_array(np.array([[0.0, 1.0]]), dim=0, params=None, provenance=None)

    assert d.meta.params == {}
    assert d.meta.provenance["clamped_rows"] == 0


@pytest.mark.parametrize("mapping", [{1: "x"}, {None: "x"}, {(1, 2): "x"}])
def test_an_adapter_refuses_metadata_that_is_not_str_keyed(mapping: Any) -> None:
    """§8 admits `str`-keyed mappings, and the top level is a mapping too.

    `json.dumps` does not refuse a non-string key -- it *rewrites* it, so
    `params={1: "x"}` is written as `{"1": "x"}` and reloads under a key the
    caller never used. That is a §10.1 requirement 1 failure rather than a
    save failure, and it is invisible until someone reads the file back."""
    with pytest.raises(TypeError, match="str-keyed"):
        from_array(np.array([[0.0, 1.0]]), dim=0, params=mapping)
    with pytest.raises(TypeError, match="str-keyed"):
        from_array(np.array([[0.0, 1.0]]), dim=0, provenance={"nested": mapping})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_an_adapter_refuses_non_finite_metadata(value: float) -> None:
    """§10.2: "`inf` lives in `bars.npz`, where NumPy represents it correctly,
    and never in the JSON. This is the reason for the split."

    `json.dumps` writes the bare tokens `NaN` and `Infinity`, which no
    conforming reader accepts -- §10.3 makes the same point from the other
    side, noting that Parquet's `double` is IEEE 754 "unlike JSON's". `nan`
    fails §10.1 requirement 1 on its own terms besides, since `nan != nan`
    leaves a round-tripped diagram comparing unequal to itself.

    This constrains metadata only. An essential bar's `inf` death is stored in
    `deaths` and is unaffected, which is precisely what §10.2's split is
    for."""
    with pytest.raises(TypeError, match="non-finite"):
        from_array(np.array([[0.0, 1.0]]), dim=0, params={"scale": value})
    with pytest.raises(TypeError, match="non-finite"):
        from_array(np.array([[0.0, 1.0]]), dim=0, provenance={"a": [{"b": value}]})


def test_non_finite_metadata_does_not_constrain_an_essential_bar() -> None:
    """The other half of the clause above, asserted so the refusal cannot
    creep into the bars: §5 stores an essential death as `inf`, never a
    sentinel, and §10.2 keeps it in `bars.npz` where it survives exactly."""
    d = from_array(np.array([[0.0, math.inf]]), dim=0, params={"scale": 1.0})

    assert math.isinf(float(d.deaths[0]))
    assert bool(d.essential[0])


def test_an_adapter_keeps_metadata_that_json_admits() -> None:
    """The refusal above must not narrow §8's actual list, which is `str`,
    `int`, `float`, `bool`, `None`, and lists or `str`-keyed mappings of
    those -- nesting included."""
    recorded = {"a": 1, "b": 1.5, "c": "x", "d": True, "e": None, "f": [1, {"g": 2}]}

    d = from_array(np.array([[0.0, 1.0]]), dim=0, params=recorded)

    assert dict(d.meta.params) == recorded
    assert_json_representable(d.meta.params)


@pytest.mark.parametrize("field", ["two", 2.0, 2.5, True])
def test_a_stated_coefficient_field_must_be_an_integer(field: Any) -> None:
    """§8 types `coeff_field` as `int | None`, and D17's whole argument is
    that a reader checks `coeff_field_source` before trusting the value. A
    non-integer recorded with `"caller"` is a provenance entry that reads as
    authoritative and describes no field at all -- the one outcome the source
    key exists to prevent."""
    with pytest.raises(TypeError, match="coeff_field"):
        from_gudhi(np.array([[0.0, 1.0]]), dim=0, coeff_field=field)
    with pytest.raises(TypeError, match="coeff_field"):
        from_ripser([np.array([[0.0, 1.0]])], coeff_field=field)


@pytest.mark.parametrize("field", ["two", 2.0, True])
def test_the_adapters_d17_excludes_still_type_check_a_stated_field(
    field: Any,
) -> None:
    """§11 excuses `from_array`, `from_persim` and `from_giotto` from
    *recording* a coefficient field -- `from_giotto` on A.5's evidence, the
    other two for having no backend and computing no homology. None of that
    makes `coeff_field="two"` admissible on them: §8 types the field
    `int | None` for every diagram however it was built, and a caller who
    states one is stating it about the same field the other two adapters
    record."""
    with pytest.raises(TypeError, match="coeff_field"):
        from_array(np.array([[0.0, 1.0]]), dim=0, coeff_field=field)
    with pytest.raises(TypeError, match="coeff_field"):
        from_persim([np.array([[0.0, 1.0]])], coeff_field=field)
    with pytest.raises(TypeError, match="coeff_field"):
        from_giotto(
            np.array([[[0.0, 1.0, 0.0]]]),
            reduced_homology=False,
            infinity_values=math.inf,
            coeff_field=field,
        )


@pytest.mark.parametrize(
    ("adapter", "call"),
    [
        (
            "from_gudhi",
            lambda f: from_gudhi(np.array([[0.0, 1.0]]), dim=0, coeff_field=f),
        ),
        ("from_ripser", lambda f: from_ripser([np.array([[0.0, 1.0]])], coeff_field=f)),
        (
            "from_array",
            lambda f: from_array(np.array([[0.0, 1.0]]), dim=0, coeff_field=f),
        ),
        ("from_persim", lambda f: from_persim([np.array([[0.0, 1.0]])], coeff_field=f)),
        (
            "from_giotto",
            lambda f: from_giotto(
                np.array([[[0.0, 1.0, 0.0]]]),
                reduced_homology=False,
                infinity_values=math.inf,
                coeff_field=f,
            )[0],
        ),
    ],
)
def test_an_integral_coefficient_field_is_stored_as_a_builtin_int(
    adapter: str, call: Any
) -> None:
    """An accepted `coeff_field` must be *normalised*, not merely validated.

    `numbers.Integral` is the admitting test rather than `int` on purpose: a
    caller looping over degrees reads `np.int64` out of an array, and refusing
    that would be refusing the ordinary spelling. The consequence is that the
    accepted value can be a NumPy scalar, and storing it unconverted puts in
    `coeff_field` a value §8's `int | None` does not describe and `json.dumps`
    refuses -- §10.2's failure arriving from the one §8 field
    `_require_json_representable` cannot reach, since it walks `params` and
    `provenance` and `coeff_field` is neither.

    Parametrised across all five adapters because the earlier defect was a
    disagreement *between* them: `from_gudhi` and `from_ripser` normalised via
    `_coeff_field` on their way to recording a default, and the three D17
    excludes never call it, so the stored type depended on which adapter the
    caller reached for."""
    d = call(np.int64(3))

    assert d.meta.coeff_field == 3
    assert type(d.meta.coeff_field) is int, (
        f"{adapter} stored {type(d.meta.coeff_field).__name__}; §8 types the "
        "field int | None and §10.2 stores it as JSON"
    )
    json.dumps({"coeff_field": d.meta.coeff_field})


def test_an_omitted_coefficient_field_and_an_explicit_none_agree(
    gudhi_intervals: Any,
) -> None:
    """`coeff_field=None` is "stated nothing", not "stated no field".

    §8 spells the absence of a value as `None` on the field itself, so the two
    are one statement arriving by two routes, and §11's requirement is about
    what the backend would have done rather than about what the caller typed.
    Asserted rather than left implicit because the alternative reading --
    treating an explicit `None` as a caller-stated value -- would record
    `coeff_field=None` beside `coeff_field_source="caller"`, which §8 requires
    `DiagramMeta` to reject outright."""
    intervals = gudhi_intervals("circle", 1)

    stated = from_gudhi(intervals, dim=1, coeff_field=None)
    omitted = from_gudhi(intervals, dim=1)

    assert stated.meta.coeff_field == omitted.meta.coeff_field == 11
    assert (
        stated.meta.provenance["coeff_field_source"]
        == omitted.meta.provenance["coeff_field_source"]
        == "backend_default"
    )


# ---------------------------------------------------------------------------
# Input forms the branches above reach only indirectly
#
# Each of these is a documented refusal or a documented acceptance that the
# rest of the suite exercises only as a side effect of testing something else.
# A branch reached but never asserted on is a branch whose message can rot.
# ---------------------------------------------------------------------------


def test_from_ripser_accepts_a_degree_block_of_plain_python_rows() -> None:
    """§11 fixes the shapes, not the container. `Rips().fit_transform(X)`
    returns arrays, but a caller round-tripping through JSON or a fixture hands
    over lists of lists, and the degree-by-index reading is the same."""
    d = from_ripser([[[0.0, 1.0], [0.5, 2.0]], [[1.0, math.inf]]])

    assert [int(x) for x in d.dims] == [0, 0, 1]
    assert [float(x) for x in d.births] == [0.0, 0.5, 1.0]
    assert math.isinf(float(d.deaths[2]))


def test_from_persim_accepts_a_degree_block_of_plain_python_rows() -> None:
    """The same input form through the other adapter that reads degree by
    index, since persim consumes exactly what Ripser emits."""
    d = from_persim([[[0.0, 1.0]], [[1.0, 2.0]]])

    assert [int(x) for x in d.dims] == [0, 1]
    assert d.meta.backend == "persim"


@pytest.mark.parametrize("adapter", [from_ripser, from_persim])
@pytest.mark.parametrize(
    "block",
    [
        [[False, True]],  # flags, which float() reads as 0.0 and 1.0
        [["0.0", "1.0"]],  # strings, which float() parses
    ],
)
def test_a_plain_python_block_is_held_to_the_same_dtypes_as_an_array(
    adapter: Any, block: Any
) -> None:
    """§6.1's dtype rule reaches a block the caller did not wrap in an array.

    The two adapters that read degree by index accept blocks that are not
    arrays, and converting one with an explicit `dtype=float64` performs
    exactly the coercion `_require_real` exists to refuse: `[[False, True]]`
    would become the bar (0.0, 1.0), and `[["0.0", "1.0"]]` the same, on this
    path alone -- the identical rows inside a NumPy array are refused. An
    adapter whose strictness depends on whether the caller wrapped their rows
    in `np.asarray` first is an adapter with two contracts.

    **The rule is the inferred dtype, not the type of each element**, and the
    two part company on a mixed block: `[[False, 1.0]]` infers `float64` and
    is accepted, here and inside an array both. That is the intended reading
    rather than a hole in this one -- the contract being defended is that the
    two containers agree, and they do, because the conversion above is left to
    infer exactly what `np.asarray` would infer. Validating elements ahead of
    inference would refuse the plain list while the identical rows inside an
    array still passed, which is the second contract this test exists to
    prevent. See `test_a_mixed_block_is_read_by_its_inferred_dtype`."""
    with pytest.raises(TypeError, match="dtype"):
        adapter([block])


@pytest.mark.parametrize("adapter", [from_ripser, from_persim])
def test_a_plain_python_block_of_integers_is_still_accepted(adapter: Any) -> None:
    """The refusal above must not reach past its target. Integers are exact in
    `float64` and are how a hand-written or JSON-round-tripped block spells a
    whole-numbered filtration value."""
    d = adapter([[[0, 1], [1, 2]]])

    assert [float(x) for x in d.births] == [0.0, 1.0]


@pytest.mark.parametrize("adapter", [from_ripser, from_persim])
@pytest.mark.parametrize(
    "block",
    [
        [[False, 1.0]],  # bool with a float: infers float64
        [[0, True]],  # bool with an int: infers int64
    ],
)
def test_a_mixed_block_is_read_by_its_inferred_dtype(adapter: Any, block: Any) -> None:
    """A block mixing `bool` with a number is accepted, and must be accepted
    identically whether or not the caller wrapped it in an array first.

    This is the boundary of the refusal two tests up, and it is drawn where it
    is on purpose. `[[False, True]]` infers `bool` and is refused; `[[False,
    1.0]]` infers `float64` -- NumPy promoted it before the adapter saw a
    dtype at all -- and is accepted. Reading elements ahead of the conversion
    would let the adapter refuse a plain list whose array form it admits,
    which is the divergence the refusal above exists to close, so the looser
    answer on a mixed block is the price of the two paths agreeing at all.

    Asserting *both* containers is the whole test. Either result is defensible
    alone; what is not defensible is the two disagreeing. `==` compares bars
    and not `meta` (§8), which is what makes the comparison meaningful here:
    `provenance["source_dtype"]` differs by construction, a plain list having
    no dtype to record."""
    from_list = adapter([block])
    from_array_form = adapter([np.array(block)])

    assert [float(x) for x in from_list.births] == [0.0]
    assert [float(x) for x in from_list.deaths] == [1.0]
    assert from_list == from_array_form


@pytest.mark.parametrize(
    "dgms",
    [
        [np.array([[0.0, 1.0, 2.0]])],  # three columns
        [np.array([0.0, 1.0])],  # rank 1
        [np.array([[0.0, 1.0]]), np.zeros((2, 3))],  # the second block
    ],
)
def test_a_degree_block_that_is_not_n_by_2_is_refused_by_index(dgms: Any) -> None:
    """§11: each block is `(n, 2)`, and the refusal names which one. A caller
    whose degree-3 block is mis-shaped cannot find it in a message that only
    says a shape is wrong."""
    with pytest.raises(ValueError, match=r"diagram at index \d+ must have shape"):
        from_ripser(dgms)


@pytest.mark.parametrize("adapter", [from_ripser, from_persim])
@pytest.mark.parametrize(
    ("dgms", "index"),
    [
        ([[[0.0, 1.0], [0.0, 1.0, 2.0]]], 0),  # rows of two widths
        ([[[0.0, 1.0], [2.0]]], 0),  # a row of one
        ([np.array([[0.0, 1.0]]), [[0.0, 1.0], [2.0]]], 1),  # the second block
    ],
)
def test_a_python_block_that_is_not_rectangular_is_refused_by_index(
    adapter: Any, dgms: Any, index: int
) -> None:
    """§11 fixes what these adapters accept, so a block that is not `(n, 2)`
    must be refused in our words wherever the failure surfaces.

    The shape check one test up runs on a block the namespace already read.
    A plain Python block is converted first, and a non-rectangular one fails
    *inside* that conversion -- so without a guard the caller gets NumPy's
    "setting an array element with a sequence. The requested array has an
    inhomogeneous shape after 1 dimensions", which names neither the adapter,
    the argument, nor which block was wrong. Both paths owe the same refusal,
    and the index is the part of it the caller cannot work out for
    themselves."""
    with pytest.raises(ValueError, match=rf"diagram at index {index} must have shape"):
        adapter(dgms)


@pytest.mark.parametrize("adapter", [from_ripser, from_persim])
@pytest.mark.parametrize(
    ("dgms", "index"),
    [
        ([None], 0),
        ([3], 0),
        ([{}], 0),
        ([np.array([[0.0, 1.0]]), object()], 1),
        ([np.array([[0.0, 1.0]]), "ab"], 1),
    ],
)
def test_a_degree_block_that_is_neither_array_nor_rows_is_refused_by_index(
    adapter: Any, dgms: Any, index: int
) -> None:
    """The third path into a mis-shaped block, and the one that used to escape.

    §11 accepts a list of `(n, 2)` arrays or of blocks of rows. A block that
    is neither -- `None`, a bare number, a mapping -- reached
    `namespace_of` untouched and failed there, with
    `array_namespace requires at least one non-scalar array input`: the
    namespace's words, naming neither the adapter, the argument, nor which
    block was wrong. That is exactly what the two tests above exist to stop,
    on the two paths that happened to be guarded, so this one owes the same
    refusal for the same reason.

    The first array block is also what these adapters resolve the namespace
    *from*, so the refusal has to happen before that resolution and not
    inside the row loop -- `[3]` failed one line earlier than `[None]` did,
    for no reason a caller could see."""
    with pytest.raises(ValueError, match=rf"diagram at index {index} must have shape"):
        adapter(dgms)


@pytest.mark.parametrize("adapter", [from_ripser, from_persim])
def test_a_gudhi_persistence_list_at_the_wrong_adapter_names_from_gudhi(
    adapter: Any, gudhi_pairs: Any
) -> None:
    """The likeliest mix-up on this surface, and the one direction that used
    to fail opaquely.

    GUDHI's `persistence()` list is `list[(dim, (birth, death))]` and Ripser's
    `dgms` is `list[(n, 2)]`; §11 accepts each at one adapter. Handed the
    first, `from_ripser` reads `(0, (0.0, 1.0))` as a degree block and cannot
    convert it, because the row is not rectangular. The reverse mistake --
    Ripser's `dgms` at `from_gudhi` -- has always been refused by name, so
    this direction is the asymmetry rather than a new requirement.

    The refusal names `from_gudhi` because the input form is recognisable:
    a `(dim, (birth, death))` row is what `_is_persistence_row` already
    identifies for the extended-persistence gate."""
    with pytest.raises(ValueError, match="from_gudhi") as excinfo:
        adapter(gudhi_pairs("circle"))

    assert "diagram at index 0 must have shape" in str(excinfo.value)


def test_from_gudhi_refuses_a_persistence_list_with_a_stated_degree(
    gudhi_pairs: Any,
) -> None:
    """§11: the `persistence()` list carries a degree per bar, so `dim=`
    alongside it is a second source for one fact. Refusing beats picking a
    winner: silently preferring either one turns a caller's mistake into a
    diagram whose degrees are not the ones GUDHI computed."""
    with pytest.raises(TypeError, match="second source"):
        from_gudhi(gudhi_pairs("circle"), dim=1)


@pytest.mark.parametrize("adapter", [from_array, from_giotto])
def test_the_array_adapters_refuse_a_non_array(adapter: Any) -> None:
    """§3.3: these two read shapes and dtypes off the object, so the namespace
    is the contract. A list would otherwise fail later and deeper, with an
    `AttributeError` naming `ndim` rather than a sentence naming the argument.

    `from_giotto` is passed both of its required keywords so the refusal under
    test is the one about the array, not §5.1's or §5's about a missing one."""
    kwargs = (
        {"reduced_homology": True, "infinity_values": math.inf}
        if adapter is from_giotto
        else {"dim": 0}
    )

    with pytest.raises(TypeError, match="__array_namespace__"):
        adapter([[0.0, 1.0]], **kwargs)


@pytest.mark.parametrize("degree", [-1, -(2**31)])
def test_a_stated_degree_is_checked_even_when_the_column_is_empty(
    degree: int,
) -> None:
    """I3 on a value the assembled column cannot speak for.

    The core type enforces `dims >= 0` over the column, which is the right
    owner for a degree that arrived *in* the data. A degree the caller states
    by hand fills a column whose length is the array's, and over a column of
    length zero "every degree is non-negative" is vacuously true -- so
    `dim=-1` on an empty array would build a diagram, having been told
    nothing, while the same argument on a one-row array raises.

    An empty diagram is legitimate (§4.2) and an empty *input* is ordinary, so
    the refusal has to come from the value rather than from the data."""
    with pytest.raises(ValueError, match="non-negative"):
        from_array(np.empty((0, 2)), dim=degree)
    with pytest.raises(ValueError, match="non-negative"):
        from_gudhi(np.empty((0, 2)), dim=degree)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"backend": "fake"}, "recorded by the adapter"),
        ({"backend_version": "9.9"}, "recorded by the adapter"),
        ({"filtraton": "rips"}, "unexpected keyword"),
        ({"provenance": {"p": object()}}, "JSON-representable"),
        ({"coeff_field": "two"}, "coeff_field"),
    ],
)
def test_from_giotto_validates_metadata_on_a_batch_with_no_samples(
    kwargs: Any, match: str
) -> None:
    """Whether a caller's metadata is checked must not depend on how many
    samples their batch carried.

    `from_giotto` is the one adapter that builds its metadata inside a loop
    over the data, so a zero-sample batch used to skip every check in it: a
    misspelled `filtraton=`, a forged `backend=`, and a `provenance` value
    that cannot be saved were all accepted and silently dropped, while the
    identical call one sample later raised. That is §4's
    shape-depends-on-what-else-was-there hazard reappearing as a property of
    the adapter, which is the thing §11 fixes the return type to avoid.

    An empty batch is valid and stays valid (§4.2) -- what is refused here is
    the metadata, not the shape."""
    with pytest.raises(TypeError, match=match):
        from_giotto(
            np.zeros((0, 2, 3)),
            reduced_homology=False,
            infinity_values=math.inf,
            **kwargs,
        )

    # The same refusal, from the same call with one sample in it. Asserting
    # both is the point: the test is about the two agreeing, not about either
    # message on its own.
    with pytest.raises(TypeError, match=match):
        from_giotto(
            np.zeros((1, 2, 3)),
            reduced_homology=False,
            infinity_values=math.inf,
            **kwargs,
        )


@pytest.mark.parametrize("dtype", [bool, complex, object])
def test_from_giotto_validates_dtype_on_a_batch_with_no_samples(dtype: Any) -> None:
    """The same argument as the test above, applied to the data rather than to
    the metadata.

    Dtype is the one check in the sample loop that is not a statement about
    rows -- integrality, int32 range, non-negativity, the `births == deaths`
    padding mask and §3.1's invariants are all vacuously true of a sample with
    no rows, while `bool`, `complex` and `object` are properties the array has
    with zero samples in it. Left inside the loop, they were refused at one
    sample and accepted at none, which is the shape-dependent acceptance §4
    warns about and §11 keeps out of the adapters."""
    with pytest.raises(TypeError, match="dtype"):
        from_giotto(
            np.zeros((0, 2, 3), dtype=dtype),
            reduced_homology=False,
            infinity_values=math.inf,
        )

    with pytest.raises(TypeError, match="dtype"):
        from_giotto(
            np.zeros((1, 2, 3), dtype=dtype),
            reduced_homology=False,
            infinity_values=math.inf,
        )


def test_from_giotto_refuses_an_adapter_owned_provenance_key_like_the_rest() -> None:
    """The zero-sample preflight must not diverge from the construction it
    stands in for, in either direction.

    This is the property, and it survives the rule changing under it. It used
    to read "every adapter silently overwrites a caller's `clamped_rows`, so
    `from_giotto` must too"; the key is now refused everywhere, so what has to
    agree is the refusal. What must never happen is that whether a caller's
    key is caught depends on how many samples their batch carried, which is
    §4's shape-depends-on-what-else-was-there hazard in the adapter's own
    behaviour -- and a zero-sample batch never enters the loop where real
    construction happens.

    Asserted against `from_array` in the same test so that "like the rest" is
    checked rather than assumed."""
    junk = {"clamped_rows": object()}

    with pytest.raises(TypeError, match="clamped_rows"):
        from_array(np.array([[0.0, 1.0]]), dim=0, provenance=junk)

    # A non-trivial bar, so that the `strip_padding=None` padding warning --
    # which `(0, 0, 0)` would trip -- stays out of a test about provenance.
    for arr in (np.zeros((0, 1, 3)), np.array([[[0.0, 1.0, 0.0]]])):
        with pytest.raises(TypeError, match="clamped_rows"):
            from_giotto(
                arr, reduced_homology=False, infinity_values=math.inf, provenance=junk
            )


def test_from_giotto_keeps_valid_metadata_on_a_batch_with_no_samples() -> None:
    """The validation above must not turn an empty batch into an error, and
    must not consume the caller's metadata on the way through -- the check is
    run against a copy and its result discarded."""
    b = from_giotto(
        np.zeros((0, 2, 3)),
        reduced_homology=False,
        infinity_values=math.inf,
        filtration="rips",
        description="S^1",
    )

    assert len(b) == 0


def test_from_giotto_accepts_a_sample_with_no_bars() -> None:
    """§4.2: an empty diagram is valid, and a batch may hold one. giotto emits
    exactly this when one sample's filtration produces nothing and the batch is
    padded to a width of zero -- and the degree validation added ahead of the
    padding mask must not trip over a column with nothing in it."""
    b = from_giotto(
        np.zeros((2, 0, 3)),
        reduced_homology=False,
        infinity_values=math.inf,
        strip_padding=True,
    )

    assert [b[i].n_bars for i in range(len(b))] == [0, 0]
    assert all(b[i].meta.provenance["padding_removed"] == 0 for i in range(len(b)))


def test_from_giotto_strips_one_sample_empty_and_leaves_another_with_bars() -> None:
    """§4.2's `offsets`, exercised rather than degenerate.

    The test above empties every member, so `offsets` is all zeros and any
    arithmetic at all -- a cumulative sum, a constant, a length -- produces
    it. §11.2 asks for the mixed batch for that reason: "The suite MUST cover
    a batch whose diagrams have different bar counts, so `offsets` is
    exercised rather than degenerate; one containing an empty diagram, so a
    zero-length segment is". `strip_padding=True` is the only adapter path
    that produces both from one input, and it puts the zero-length segment
    *first*, where an off-by-one in the segment boundaries shows and a
    `from_diagrams` that quietly dropped empty members would leave a batch of
    length one.

    The surviving member's bars are asserted by value, not by count: a member
    that had absorbed its empty neighbour's segment would still be length two
    here and hold the wrong rows."""
    arr = np.array(
        [
            [[1.0, 1.0, 0.0], [2.0, 2.0, 0.0]],  # both trivial; emptied
            [[0.0, 5.0, 0.0], [1.0, 3.0, 1.0]],  # neither; kept whole
        ]
    )

    b = from_giotto(
        arr, reduced_homology=False, infinity_values=math.inf, strip_padding=True
    )

    assert len(b) == 2
    assert [b[i].n_bars for i in range(len(b))] == [0, 2]
    assert [int(x) for x in b.offsets] == [0, 0, 2]
    assert [b[i].meta.provenance["padding_removed"] for i in range(len(b))] == [2, 0]
    assert [float(x) for x in b[1].births] == [0.0, 1.0]
    assert [float(x) for x in b[1].deaths] == [5.0, 3.0]


# ---------------------------------------------------------------------------
# §8's reserved provenance keys are the adapter's to write, not the caller's
# ---------------------------------------------------------------------------

# §8's reserved-key table, in full. Spelled out here rather than imported from
# `adapters.py` so that the test states the requirement and the module states
# the implementation: importing the set would make this test pass against any
# set the module happened to hold, including an empty one.
_RESERVED_PROVENANCE_KEYS = (
    "essential_bars",
    "essential_bars_dropped",
    "essential_bars_finitized_at",
    "essential_bars_source",
    "coeff_field_source",
    "source_dtype",
    "clamped_rows",
    "padding_removed",
)


def _call_every_adapter(**meta: Any) -> dict[str, Any]:
    """Every adapter, on its smallest valid input, with `**meta` passed on."""
    return {
        "from_gudhi": lambda: from_gudhi([(0, (0.0, 1.0))], **meta),
        "from_ripser": lambda: from_ripser([np.array([[0.0, 1.0]])], **meta),
        "from_persim": lambda: from_persim([np.array([[0.0, 1.0]])], **meta),
        "from_array": lambda: from_array(np.array([[0.0, 1.0]]), dim=0, **meta),
        "from_giotto": lambda: from_giotto(
            np.array([[[0.0, 1.0, 0.0]]]),
            reduced_homology=False,
            infinity_values=math.inf,
            **meta,
        ),
    }


@pytest.mark.parametrize("key", _RESERVED_PROVENANCE_KEYS)
@pytest.mark.parametrize("adapter", list(_call_every_adapter()))
def test_no_adapter_lets_a_caller_write_a_reserved_provenance_key(
    key: str, adapter: str
) -> None:
    """§8: every reserved key names the writer that measured it, and none of
    them is the caller.

    `backend` and `backend_version` are already refused on exactly this
    ground -- "a caller who could set them could produce a diagram that lies
    about where it came from". §8's `provenance` table is seven more facts of
    the same kind: `essential_bars` has two named writers and neither is a
    caller ("Both writers, `from_giotto` at construction and `finitize()`
    later, MUST be the only places that set this key"); `essential_bars_source`
    is "Written only by `from_*`"; the rest are counts and dtypes the adapter
    measured while reading the backend's output.

    Parametrised over every adapter and every key, because the defect this
    replaces was that protection depended on which keys an adapter happened to
    write: a caller's key lost the merge where the adapter set one of its own
    and survived where it did not. Whether a fact can be forged must not depend
    on which adapter is asked."""
    value = 1 if key.endswith(("_rows", "_removed", "_dropped")) else "faithful"

    with pytest.raises(TypeError, match=key):
        _call_every_adapter(provenance={key: value})[adapter]()


@pytest.mark.parametrize("adapter", ["from_persim", "from_array"])
def test_the_adapters_that_make_no_essential_bar_claim_cannot_be_given_one(
    adapter: str,
) -> None:
    """§11, §5.1: the case that found the defect.

    `from_persim` and `from_array` record neither essential-bar key -- persim
    computes no homology ("no opinion"), an array has no backend -- so neither
    key was in the adapter's own mapping and a caller's survived the merge
    untouched. The diagram that came out carried `essential_bars` with **no**
    `essential_bars_source`, which is precisely what §11 forbids: "An adapter
    that records `provenance['essential_bars']` MUST record
    `provenance['essential_bars_source']` with the same value in the same
    construction."

    The pairing cannot be enforced in `DiagramMeta` instead. `finitize` (§5)
    legitimately writes `essential_bars` onto a diagram that never had a
    source -- a `from_array` diagram has none to inherit -- so a constructor
    rule would refuse the one writer §8 requires. The refusal belongs at the
    adapter boundary, which is where the caller is."""
    with pytest.raises(TypeError, match="essential_bars"):
        _call_every_adapter(provenance={"essential_bars": "faithful"})[adapter]()

    with pytest.raises(TypeError, match="essential_bars_source"):
        _call_every_adapter(provenance={"essential_bars_source": "faithful"})[adapter]()


def test_a_caller_keeps_every_provenance_key_that_is_not_reserved(
    gudhi_pairs: Any,
) -> None:
    """§8: `provenance` stays the honest-accounting channel. Only the seven
    reserved names are refused; an ordinary key is kept as it always was."""
    d = from_gudhi(
        gudhi_pairs("circle"),
        provenance={"analyst": "eb", "capture_host": "ci", "run": 3},
    )

    assert d.meta.provenance["analyst"] == "eb"
    assert d.meta.provenance["capture_host"] == "ci"
    assert d.meta.provenance["run"] == 3
    assert d.meta.provenance["essential_bars"] == "faithful"


def test_the_reserved_refusal_names_the_key_and_not_just_the_argument() -> None:
    """A caller who passed one of seven keys needs to know which one."""
    with pytest.raises(TypeError, match=r"padding_removed.*§8"):
        from_array(
            np.array([[0.0, 1.0]]), dim=0, provenance={"padding_removed": 99, "ok": 1}
        )


# ---------------------------------------------------------------------------
# `columns=`' own rules outrank §11's shape refusal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_columns", "columns", "match"),
    [
        (1, ["birth"], "missing.*death"),
        (4, ["birth", "death", "dim", "dim"], "duplicate"),
        (5, ["birth", "death", "dim", "x", "y"], "unknown.*x"),
    ],
)
def test_from_array_judges_columns_before_it_looks_at_the_width(
    n_columns: int, columns: list[str], match: str
) -> None:
    """§10.3: `columns=` MUST raise on the argument, before `arr` is inspected.

    **This inverts what this suite previously asserted**, which was that §11's
    shape refusal outranked the vocabulary rules. §10.3 settled it the other
    way, and the reason is that `columns=` is now what answers §11's degree
    question rather than the column count: a header naming two births and no
    death is wrong on its own terms whatever it is passed beside, so the
    failure must not depend on the array's width -- or, as the sibling test
    below shows, on the array being inspectable at all."""
    with pytest.raises(ValueError, match=match):
        from_array(np.zeros((1, n_columns)), columns=columns)


class _ExplodingArray:
    """An `arr` that raises on every attribute a reader could reach for.

    Not a mock of an array -- the point is that it is *not* one, so any
    implementation that consults it at all fails loudly rather than by
    reporting the wrong defect.
    """

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"arr was inspected: reached .{name}")


@pytest.mark.parametrize(
    ("columns", "error", "match"),
    [
        ("birth,death", TypeError, "sequence"),
        (["birth", 1], TypeError, "string"),
        (["birth", "birth"], ValueError, "duplicate"),
        (["birth", "dim"], ValueError, "missing.*death"),
        (["birth", "death", "other"], ValueError, "unknown.*other"),
        (["birth", "death", "diagram_id"], TypeError, r"diagram_id.*\.akd"),
    ],
)
def test_invalid_columns_raise_without_touching_arr_at_all(
    columns: Any, error: type[Exception], match: str
) -> None:
    """§10.3's ordering rule, proved rather than asserted.

    Passing an object that raises on any attribute access establishes that the
    refusal came from `columns=` alone. A test using a valid array cannot
    distinguish "judged the argument first" from "judged the argument second
    and the array happened to be fine", which is the whole content of the
    rule."""
    with pytest.raises(error, match=match):
        from_array(_ExplodingArray(), columns=columns, dim=0)


def test_the_batch_column_refusal_still_outranks_the_shape_error() -> None:
    """The one deliberate exception, unchanged (§10.3).

    A four-column table headed `diagram_id,dim,birth,death` is a batch CSV, and
    that caller needs to be sent to the `.akd` format rather than told that
    arrays are `(n, 2)` or `(n, 3)` -- which is true, and answers a question
    they did not ask."""
    with pytest.raises(TypeError, match=r"diagram_id.*\.akd"):
        from_array(np.zeros((1, 4)), columns=["diagram_id", "dim", "birth", "death"])


def test_a_columns_argument_is_still_refused_on_its_own_terms_first() -> None:
    """§10.3: the checks that need no array still run before the array is
    inspected, so the failure does not depend on the data or its width."""
    with pytest.raises(TypeError, match="sequence"):
        from_array(np.zeros((1, 7)), columns="birth,death,dim")

    with pytest.raises(TypeError, match="string"):
        from_array(np.zeros((1, 7)), columns=["birth", "death", "dim", 4, 5, 6, 7])
