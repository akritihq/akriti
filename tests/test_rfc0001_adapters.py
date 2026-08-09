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

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        b = from_giotto(arr, reduced_homology=True, strip_padding=False)

    assert all(b[i].meta.provenance["padding_removed"] == 0 for i in range(len(b)))
    assert b[0].n_bars == arr.shape[1]


def test_from_giotto_does_not_warn_when_there_is_no_padding() -> None:
    """§11.1 warns "if any trivial rows are present" -- not unconditionally."""
    arr = np.array([[[0.0, 1.0, 0.0], [0.0, 2.0, 1.0]]])

    import warnings

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
