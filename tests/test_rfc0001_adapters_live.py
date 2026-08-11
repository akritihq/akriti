"""Adapter tests that call a live backend. RFC-0001 §9.3, §11.2.

Two jobs the frozen fixtures cannot do.

**Keep the fixtures honest.** `test_rfc0001_adapters.py` runs against captured
output, which §11.2 admits as real. What it cannot notice is a backend release
that changes what it emits, at which point the suite goes on testing an
adapter against a past that no longer exists. The round-trip tests here run
the same clouds live and compare, so drift fails the build in the environment
that installs backends rather than surviving unmentioned.

**Assert the two coefficient-field defaults against the installed backend.**
§9.3 makes GUDHI's Z/11 and Ripser's Z/2 load-bearing: §11 has the adapters
write those numbers into the provenance of diagrams whose caller stated no
field, so a change upstream stops being documentation drift and becomes
silently wrong provenance on every diagram recorded afterwards. §11.2 requires
one test per backend, marked `backend`.

Everything here is marked `backend` and skips where the backend is absent,
which is the default environment by design (`pyproject.toml`).
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pytest

from akriti.diagrams.adapters import from_gudhi, from_ripser

pytestmark = pytest.mark.backend

# Ripser computes in single precision, so cross-backend agreement is bounded by
# float32 epsilon rather than float64. RFC-0001 §6.2.
CROSS_BACKEND_RTOL = 1e-6

# Both backends are pinned to the same field. RFC-0001 §9.3: GUDHI defaults to
# Z/11 and Ripser to Z/2, neither returns the field it used (A.5), so a
# comparison taking the defaults matches bars across two different homology
# theories and returns True anyway on torsion-free test data.
PINNED_COEFF_FIELD = 2

MAX_EDGE = 4.0


@pytest.fixture(scope="module")
def circle() -> np.ndarray:
    """The 40-point noisy circle Appendix A.1 and the fixtures both use."""
    rng = np.random.default_rng(0)
    theta = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    pts = np.column_stack([np.cos(theta), np.sin(theta)])
    return pts + rng.normal(0, 0.05, pts.shape)


@pytest.mark.alpha
def test_gudhi_default_coefficient_field_is_eleven() -> None:
    """§9.3: GUDHI computes over Z/11 unless told otherwise.

    §11 writes this number into every diagram whose caller stated no field, so
    an upstream change MUST break the build rather than reach a user's
    provenance. If this fails, upstream changed: §9.3's table, §11's fallback
    and `adapters.py`'s constant all move together, and Appendix A.5 needs
    re-measuring.
    """
    gudhi = pytest.importorskip("gudhi")

    default = (
        inspect.signature(gudhi.SimplexTree.persistence)
        .parameters["homology_coeff_field"]
        .default
    )

    assert default == 11


@pytest.mark.rips
def test_ripser_default_coefficient_field_is_two() -> None:
    """§9.3: Ripser computes over Z/2 -- not GUDHI's Z/11. Same consequence."""
    ripser = pytest.importorskip("ripser")

    default = inspect.signature(ripser.ripser).parameters["coeff"].default

    assert default == 2


@pytest.mark.alpha
def test_from_gudhi_records_the_field_the_installed_backend_would_use(
    circle: np.ndarray,
) -> None:
    """§11: the recorded default is a claim about this backend, checked here
    against the backend rather than against our own constant."""
    gudhi = pytest.importorskip("gudhi")

    st = gudhi.RipsComplex(points=circle, max_edge_length=MAX_EDGE).create_simplex_tree(
        max_dimension=2
    )
    d = from_gudhi(st.persistence())

    documented = (
        inspect.signature(gudhi.SimplexTree.persistence)
        .parameters["homology_coeff_field"]
        .default
    )
    assert d.meta.coeff_field == documented
    assert d.meta.provenance["coeff_field_source"] == "backend_default"


@pytest.mark.rips
def test_from_ripser_records_the_field_the_installed_backend_would_use(
    circle: np.ndarray,
) -> None:
    """The other half of the same check."""
    ripser = pytest.importorskip("ripser")

    d = from_ripser(ripser.ripser(circle, maxdim=1))

    documented = inspect.signature(ripser.ripser).parameters["coeff"].default
    assert d.meta.coeff_field == documented
    assert d.meta.provenance["coeff_field_source"] == "backend_default"


@pytest.mark.alpha
def test_live_gudhi_output_matches_the_frozen_fixture(
    circle: np.ndarray, gudhi_pairs: Any
) -> None:
    """The fixture is real backend output; this is what keeps it current."""
    gudhi = pytest.importorskip("gudhi")

    st = gudhi.RipsComplex(points=circle, max_edge_length=MAX_EDGE).create_simplex_tree(
        max_dimension=2
    )
    live = from_gudhi(st.persistence(homology_coeff_field=PINNED_COEFF_FIELD))
    frozen = from_gudhi(gudhi_pairs("circle"))

    assert live == frozen, (
        "live GUDHI output no longer matches tests/fixtures/backend_output.json; "
        "re-run tools/capture_backend_fixtures.py and read the diff before "
        "committing it"
    )


@pytest.mark.rips
def test_live_ripser_output_matches_the_frozen_fixture(
    circle: np.ndarray, ripser_dgms: Any
) -> None:
    """Same, for Ripser's dict form."""
    ripser = pytest.importorskip("ripser")

    live = from_ripser(ripser.ripser(circle, maxdim=1, coeff=PINNED_COEFF_FIELD))
    frozen = from_ripser(ripser_dgms("circle"))

    assert live == frozen, (
        "live Ripser output no longer matches tests/fixtures/backend_output.json; "
        "re-run tools/capture_backend_fixtures.py and read the diff before "
        "committing it"
    )


@pytest.mark.cross_backend
def test_gudhi_and_ripser_diagrams_agree_within_single_precision(
    circle: np.ndarray,
) -> None:
    """§11.2: cross-backend agreement at an explicit `rtol=1e-6` (§6.2), with
    the coefficient field pinned on both sides (§9.3).

    Unpinned, this sets GUDHI's Z/11 against Ripser's Z/2 -- two homology
    theories rather than one computation done twice -- and passes anyway on
    torsion-free data, establishing nothing.
    """
    gudhi = pytest.importorskip("gudhi")
    ripser = pytest.importorskip("ripser")

    st = gudhi.RipsComplex(points=circle, max_edge_length=MAX_EDGE).create_simplex_tree(
        max_dimension=2
    )
    g = from_gudhi(
        st.persistence(homology_coeff_field=PINNED_COEFF_FIELD),
        coeff_field=PINNED_COEFF_FIELD,
    )
    r = from_ripser(
        ripser.ripser(circle, maxdim=1, coeff=PINNED_COEFF_FIELD),
        coeff_field=PINNED_COEFF_FIELD,
    )

    assert g.meta.coeff_field == r.meta.coeff_field
    # GUDHI computes H2 here and Ripser stops at H1; compare where both spoke.
    for k in (0, 1):
        assert g.dim(k).allclose(r.dim(k), rtol=CROSS_BACKEND_RTOL, atol=0.0), (
            f"H{k} disagrees beyond single precision"
        )


@pytest.mark.cross_backend
def test_live_essential_bars_survive_the_adapter(circle: np.ndarray) -> None:
    """§5.1, live: GUDHI and Ripser are faithful, and so is the adapter."""
    gudhi = pytest.importorskip("gudhi")
    ripser = pytest.importorskip("ripser")

    st = gudhi.RipsComplex(points=circle, max_edge_length=MAX_EDGE).create_simplex_tree(
        max_dimension=2
    )
    g = from_gudhi(st.persistence(homology_coeff_field=PINNED_COEFF_FIELD))
    r = from_ripser(ripser.ripser(circle, maxdim=1, coeff=PINNED_COEFF_FIELD))

    assert int(np.sum(np.asarray(g.essential))) == 1
    assert int(np.sum(np.asarray(r.essential))) == 1


@pytest.mark.alpha
def test_live_gudhi_extended_persistence_is_rejected(circle: np.ndarray) -> None:
    """§1, §11: the actual GUDHI 3.13 shape is a four-element outer LIST."""
    gudhi = pytest.importorskip("gudhi")

    st = gudhi.RipsComplex(points=circle, max_edge_length=MAX_EDGE).create_simplex_tree(
        max_dimension=2
    )
    extended = st.extended_persistence(homology_coeff_field=PINNED_COEFF_FIELD)

    assert isinstance(extended, list)
    assert len(extended) == 4
    assert all(isinstance(member, list) for member in extended)
    with pytest.raises(TypeError, match="extended persistence"):
        from_gudhi(extended)
