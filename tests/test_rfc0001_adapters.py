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

import json
import math
import warnings
from importlib import metadata
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from akriti.diagrams import DiagramBatch, PersistenceDiagram
from akriti.diagrams.adapters import (
    from_array,
    from_giotto,
    from_gudhi,
    from_persim,
    from_ripser,
)


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
    b = from_giotto(giotto_array(reduced=True), reduced_homology=True)

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


def test_a_list_input_falls_back_to_numpy() -> None:
    """§11 fixes `from_gudhi(obj, **meta)` with no namespace argument, and
    GUDHI's primary form is a Python list carrying no array at all. The
    namespace has to come from somewhere; numpy is imported lazily on this
    path alone, which no caller reaches without having installed a backend
    that already depends on it."""
    d = from_gudhi([(0, (0.0, 1.0))])

    assert d.xp is np


# ---------------------------------------------------------------------------
# §8 -- what the caller may say, and what the adapter insists on
# ---------------------------------------------------------------------------


def test_caller_metadata_is_carried_through(gudhi_pairs: Any) -> None:
    """§8's fields are the caller's to state; the adapter adds to them."""
    d = from_gudhi(
        gudhi_pairs("circle"),
        filtration="rips",
        space="40-point noisy circle",
        params={"max_edge_length": 4.0},
    )

    assert d.meta.filtration == "rips"
    assert d.meta.space == "40-point noisy circle"
    assert d.meta.params["max_edge_length"] == 4.0


def test_caller_provenance_is_merged_and_never_overwrites_a_measured_fact(
    gudhi_pairs: Any,
) -> None:
    """`provenance` is the honest-accounting channel (§8). A caller's own keys
    are kept, but a key the adapter measured is the adapter's: it is the one
    party that saw the backend's output."""
    d = from_gudhi(
        gudhi_pairs("circle"),
        provenance={"analyst": "eb", "essential_bars": "lost_upstream"},
    )

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
# from_giotto -- §11's two deviations, §11.1's three modes, §5.1's derivation
# ---------------------------------------------------------------------------


def test_from_giotto_requires_reduced_homology(giotto_array: Any) -> None:
    """§5.1, §11: "Omitting it MUST raise, not fall back to giotto's own
    default", and §11 fixes that as a `TypeError` at the call site."""
    with pytest.raises(TypeError, match="reduced_homology"):
        from_giotto(giotto_array(reduced=True))  # type: ignore[call-arg]


def test_from_giotto_always_returns_a_batch(giotto_array: Any) -> None:
    """§11: a fixed return type. "Nothing about the adapter's own return type
    is allowed to depend on how many samples the particular call happened to
    carry." """
    single = giotto_array(reduced=True, sample="single")

    b = from_giotto(single, reduced_homology=True)

    assert isinstance(b, DiagramBatch)
    assert single.shape[0] == 1, "fixture changed"
    assert len(b) == 1
    assert isinstance(b[0], PersistenceDiagram)


def test_from_giotto_returns_one_diagram_per_sample(giotto_array: Any) -> None:
    """§4: the batch is ragged, one entry per input sample, order preserved."""
    batch = giotto_array(reduced=True, sample="batch")

    b = from_giotto(batch, reduced_homology=True, strip_padding=False)

    assert len(b) == batch.shape[0]
    assert [b[i].n_bars for i in range(len(b))] == [batch.shape[1]] * batch.shape[0]


def test_from_giotto_reads_columns_as_birth_death_dim(giotto_array: Any) -> None:
    """§11: giotto's columns are `(birth, death, dim)`, in that order."""
    arr = giotto_array(reduced=True)

    b = from_giotto(arr, reduced_homology=True, strip_padding=False)
    d = b[0]

    assert [float(x) for x in d.births] == list(arr[0][:, 0])
    assert [float(x) for x in d.deaths] == list(arr[0][:, 1])
    assert [int(x) for x in d.dims] == [int(v) for v in arr[0][:, 2]]


def test_from_giotto_records_reduced_homology_in_params(giotto_array: Any) -> None:
    """§5.1: it is "a raw fact of the original call, the same category as
    `max_edge_length`", so it belongs in `params`, not in `provenance`."""
    b = from_giotto(giotto_array(reduced=True), reduced_homology=True)

    assert b[0].meta.params["reduced_homology"] is True


def test_from_giotto_derives_lost_upstream_from_reduced_homology(
    giotto_array: Any,
) -> None:
    """§5.1: `"lost_upstream"` when `reduced_homology` is `True` -- derived
    from the flag, never authored independently, and `essential_bars_source`
    set to the same value in the same construction (§8, §11)."""
    b = from_giotto(giotto_array(reduced=True), reduced_homology=True)

    assert b[0].meta.provenance["essential_bars"] == "lost_upstream"
    assert b[0].meta.provenance["essential_bars_source"] == "lost_upstream"


def test_from_giotto_derives_faithful_when_homology_is_not_reduced(
    giotto_array: Any,
) -> None:
    """§5.1: `"faithful"` when `reduced_homology` is `False`. The fixture
    carries Appendix A.1's own measurement -- 40 H0 bars unreduced against 39
    reduced -- so the two branches differ in the data as well as the label."""
    unreduced = giotto_array(reduced=False)
    reduced = giotto_array(reduced=True)

    b = from_giotto(unreduced, reduced_homology=False)

    assert b[0].meta.provenance["essential_bars"] == "faithful"
    assert b[0].meta.provenance["essential_bars_source"] == "faithful"
    h0 = int((unreduced[0][:, 2] == 0).sum())
    assert h0 - int((reduced[0][:, 2] == 0).sum()) == 1, "A.1's H0 loss changed"


def test_from_giotto_does_not_fabricate_the_missing_essential_bar(
    giotto_array: Any,
) -> None:
    """§5.1: "MUST NOT fabricate an essential bar to compensate". Reconstructing
    its birth as 0 is a coincidence of the unweighted example, not a property
    of the elder rule."""
    arr = giotto_array(reduced=True)

    b = from_giotto(arr, reduced_homology=True, strip_padding=False)

    assert b[0].n_bars == arr.shape[1], "a row was invented"
    assert not bool(np.any(np.asarray(b.essential))), "giotto emits no inf (A.1)"


def test_from_giotto_default_keeps_padding_and_warns_once(giotto_array: Any) -> None:
    """§11.1: default `strip_padding=None` keeps every row, warns once if any
    trivial rows are present, and records `padding_removed = 0`."""
    arr = giotto_array(reduced=True, sample="batch")
    trivial = int((arr[:, :, 0] == arr[:, :, 1]).sum())
    assert trivial > 0, "the fixture should carry padding"

    with pytest.warns(UserWarning, match="trivial") as record:
        b = from_giotto(arr, reduced_homology=True)

    assert len(record) == 1, "§11.1 says warn once, not once per sample"
    assert [b[i].n_bars for i in range(len(b))] == [arr.shape[1]] * arr.shape[0]
    assert all(b[i].meta.provenance["padding_removed"] == 0 for i in range(len(b)))


def test_from_giotto_strips_padding_when_told_to(giotto_array: Any) -> None:
    """§11.1: `strip_padding=True` drops trivial rows and records the count."""
    arr = giotto_array(reduced=True, sample="batch")
    per_sample = [int((s[:, 0] == s[:, 1]).sum()) for s in arr]
    assert per_sample[0] > 0, "the fixture should pad the first sample"

    b = from_giotto(arr, reduced_homology=True, strip_padding=True)

    for i, dropped in enumerate(per_sample):
        assert b[i].n_bars == arr.shape[1] - dropped
        assert b[i].meta.provenance["padding_removed"] == dropped


def test_from_giotto_keeps_padding_silently_when_told_to(giotto_array: Any) -> None:
    """§11.1: `strip_padding=False` keeps silently, "and
    `provenance['padding_removed'] = 0` regardless of how many trivial rows
    are present -- the key records what was actually removed, never what was
    merely observed"."""
    arr = giotto_array(reduced=True, sample="batch")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        b = from_giotto(arr, reduced_homology=True, strip_padding=False)

    assert all(b[i].meta.provenance["padding_removed"] == 0 for i in range(len(b)))
    assert b[0].n_bars == arr.shape[1]


def test_from_giotto_does_not_warn_when_there_is_no_padding() -> None:
    """§11.1 warns "if any trivial rows are present" -- not unconditionally."""
    arr = np.array([[[0.0, 1.0, 0.0], [0.0, 2.0, 1.0]]])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        b = from_giotto(arr, reduced_homology=False)

    assert len(b) == 1


def test_from_giotto_rejects_an_array_that_is_not_a_batch(giotto_array: Any) -> None:
    """§11's table: giotto output is `(n_samples, n_bars, 3)`. A 2-D array is
    a single sample the caller forgot to wrap, and guessing which is which is
    exactly the shape-depends-on-the-data hazard §4 rules out."""
    with pytest.raises(ValueError, match="shape"):
        from_giotto(giotto_array(reduced=True)[0], reduced_homology=True)


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
        strip_padding=False,
    )

    for i in range(len(b)):
        assert b[i].meta.backend == "giotto"
        assert b[i].meta.backend_version == installed_version("giotto-tda")
        assert_json_representable(b[i].meta.provenance)


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


def test_from_gudhi_rejects_the_extended_persistence_tuple() -> None:
    """§11: `extended_persistence()` is a third input form and is out of scope.

    It returns a **4-tuple** of `list[(dim, (birth, death))]` -- ordinary,
    relative, extended+ and extended- -- structurally distinct from
    `persistence()`'s flat list, so this one it can actually detect. The
    refusal must name the scope exclusion rather than the shape: told that
    row 0 is mis-shaped, a caller goes hunting for a typo in data that is
    exactly what GUDHI handed them.

    `TypeError` rather than `ValueError` because this is an input *form* the
    adapter does not accept, which is the category `from_gudhi`'s existing
    fallthrough already raises `TypeError` for. §11 fixes that the tuple is
    refused and what the message names; it does not fix the type."""
    extended = (
        [(0, (0.0, 1.0))],  # ordinary
        [(1, (3.0, 2.0))],  # relative -- death < birth by construction
        [(1, (0.5, 2.5))],  # extended+
        [(0, (2.0, 0.5))],  # extended- -- death < birth by construction
    )

    with pytest.raises(TypeError, match="extended persistence"):
        from_gudhi(extended)


def test_from_gudhi_still_accepts_a_four_row_persistence_list() -> None:
    """The guard on the test above. `extended_persistence()` is detected as a
    4-tuple, and a `persistence()` result with four bars is also four things
    long -- so a rejection keyed on length alone would refuse ordinary GUDHI
    output. What separates them is that the tuple's members are *lists of
    rows* and a `persistence()` row is `(dim, (birth, death))`."""
    four_bars = [
        (0, (0.0, 1.0)),
        (0, (0.0, 2.0)),
        (1, (0.5, 1.5)),
        (0, (0.0, math.inf)),
    ]

    d = from_gudhi(tuple(four_bars))

    assert [int(x) for x in d.dims] == [0, 0, 1, 0]
    assert math.isinf(float(d.deaths[3]))


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
    rows = np.array([[1.0, math.nextafter(1.0, 0.0)], [1.0, 1.0 - 5e-13]])

    with pytest.warns(UserWarning, match="I6") as record:
        d = from_array(rows, dim=0)

    reported = float(str(record[0].message).split("the largest by ")[1].split(".")[0])
    assert 4e-13 < reported < 6e-13, "the larger of the two absorbed gaps"
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
        b = from_giotto(arr, reduced_homology=True, strip_padding=False)

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
        b = from_giotto(arr, reduced_homology=False, strip_padding=False)

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
            from_giotto(arr, reduced_homology=True, strip_padding=strip_padding)


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
            from_giotto(arr, reduced_homology=False, strip_padding=strip_padding)


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
        from_giotto(np.array([[[0.0, 1.0, 0.0]]]), reduced_homology=flag)


@pytest.mark.parametrize("flag", ["False", "True", "", 0, 1, object()])
def test_from_giotto_requires_a_real_boolean_for_strip_padding(flag: Any) -> None:
    """§11.1 fixes three modes -- `None`, `True`, `False` -- and a truthy
    stand-in changes the data, not just the record: `strip_padding="False"` is
    both not-`None` and truthy, so it strips every trivial row from a call
    that asked for none to be stripped, and records the count as though the
    caller had asked."""
    arr = np.array([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])

    with pytest.raises(TypeError, match="strip_padding"):
        from_giotto(arr, reduced_homology=False, strip_padding=flag)


def test_from_giotto_accepts_a_batch_with_no_samples() -> None:
    """§4.2: "An empty batch is perfectly valid". A filter that selects no
    samples is an ordinary outcome, not an error to be raised at the adapter."""
    b = from_giotto(np.zeros((0, 3, 3)), reduced_homology=True)

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
            np.array([[[0.0, 1.0, 0.0]]]), reduced_homology=False, coeff_field=field
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
                np.array([[[0.0, 1.0, 0.0]]]), reduced_homology=False, coeff_field=f
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


def test_from_gudhi_refuses_a_persistence_list_with_a_stated_degree(
    gudhi_pairs: Any,
) -> None:
    """§11: the `persistence()` list carries a degree per bar, so `dim=`
    alongside it is a second source for one fact. Refusing beats picking a
    winner: silently preferring either one turns a caller's mistake into a
    diagram whose degrees are not the ones GUDHI computed."""
    with pytest.raises(ValueError, match="second source"):
        from_gudhi(gudhi_pairs("circle"), dim=1)


@pytest.mark.parametrize("adapter", [from_array, from_giotto])
def test_the_array_adapters_refuse_a_non_array(adapter: Any) -> None:
    """§3.3: these two read shapes and dtypes off the object, so the namespace
    is the contract. A list would otherwise fail later and deeper, with an
    `AttributeError` naming `ndim` rather than a sentence naming the argument.

    `from_giotto` is passed its required flag so the refusal under test is the
    one about the array, not §5.1's about the missing keyword."""
    kwargs = {"reduced_homology": True} if adapter is from_giotto else {"dim": 0}

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
        from_giotto(np.zeros((0, 2, 3)), reduced_homology=False, **kwargs)

    # The same refusal, from the same call with one sample in it. Asserting
    # both is the point: the test is about the two agreeing, not about either
    # message on its own.
    with pytest.raises(TypeError, match=match):
        from_giotto(np.zeros((1, 2, 3)), reduced_homology=False, **kwargs)


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
        from_giotto(np.zeros((0, 2, 3), dtype=dtype), reduced_homology=False)

    with pytest.raises(TypeError, match="dtype"):
        from_giotto(np.zeros((1, 2, 3), dtype=dtype), reduced_homology=False)


def test_from_giotto_overwrites_an_adapter_owned_provenance_key_like_the_rest() -> None:
    """The zero-sample preflight must not be *stricter* than the construction
    it stands in for, in either direction.

    `_build_meta` documents that a key the adapter measured wins over a
    caller's key of the same name -- the adapter is the party that saw the
    backend's output, and `provenance` is auditable rather than assertable.
    `clamped_rows` is such a key, added by `_diagram_from_columns`, so every
    adapter overwrites a caller's value for it, junk included. The preflight
    listed the other adapter-owned keys and not this one, so `from_giotto`
    alone refused what its four siblings silently corrected -- a divergence in
    the direction the preflight's own comment promises cannot happen."""
    junk = {"clamped_rows": object()}

    assert (
        from_array(np.array([[0.0, 1.0]]), dim=0, provenance=junk).meta.provenance[
            "clamped_rows"
        ]
        == 0
    )

    # A non-trivial bar, so that the `strip_padding=None` padding warning --
    # which `(0, 0, 0)` would trip -- stays out of a test about provenance.
    for arr in (np.zeros((0, 1, 3)), np.array([[[0.0, 1.0, 0.0]]])):
        b = from_giotto(arr, reduced_homology=False, provenance=junk)
        assert all(b[i].meta.provenance["clamped_rows"] == 0 for i in range(len(b)))


def test_from_giotto_keeps_valid_metadata_on_a_batch_with_no_samples() -> None:
    """The validation above must not turn an empty batch into an error, and
    must not consume the caller's metadata on the way through -- the check is
    run against a copy and its result discarded."""
    b = from_giotto(
        np.zeros((0, 2, 3)), reduced_homology=False, filtration="rips", space="S^1"
    )

    assert len(b) == 0


def test_from_giotto_accepts_a_sample_with_no_bars() -> None:
    """§4.2: an empty diagram is valid, and a batch may hold one. giotto emits
    exactly this when one sample's filtration produces nothing and the batch is
    padded to a width of zero -- and the degree validation added ahead of the
    padding mask must not trip over a column with nothing in it."""
    b = from_giotto(np.zeros((2, 0, 3)), reduced_homology=False, strip_padding=True)

    assert [b[i].n_bars for i in range(len(b))] == [0, 0]
    assert all(b[i].meta.provenance["padding_removed"] == 0 for i in range(len(b)))
