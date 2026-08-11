"""Verify the backend behaviour RFC-0001 is written against.

RFC-0001 makes specific, measured claims about what GUDHI, Ripser and persim
do. Those claims are load-bearing: the adapter contract, the equality
semantics and the delegation guardrail in core/distances.py all depend on
them. If a backend release changes one, we want to hear it from CI rather
than from a reviewer.

These tests therefore assert the *current* behaviour of third-party code,
including behaviour we consider wrong. A failure here does not necessarily
mean something broke -- it may mean upstream improved, in which case the RFC
and the code that depends on it must be updated. Each assertion says which.

Reproduces the appendix of RFC-0001; see rfcs/evidence/probe_backends.py.
"""

from __future__ import annotations

import numpy as np
import pytest

SEED = 0
N_POINTS = 40
NOISE = 0.05
MAX_EDGE = 4.0

# Ripser computes in single precision, so cross-backend agreement is bounded
# by float32 epsilon, not float64. RFC-0001 §6.2.
CROSS_BACKEND_RTOL = 1e-6


@pytest.fixture
def circle() -> np.ndarray:
    """40 points on a noisy unit circle: one essential H0 bar, one clear H1."""
    rng = np.random.default_rng(SEED)
    theta = rng.uniform(0, 2 * np.pi, N_POINTS)
    pts = np.c_[np.cos(theta), np.sin(theta)] + rng.normal(0, NOISE, (N_POINTS, 2))
    return np.ascontiguousarray(pts, dtype=np.float64)


def _sorted_bars(dgm: np.ndarray) -> np.ndarray:
    """Bars in canonical order. Backends do not agree on row order (§7)."""
    return dgm[np.lexsort((dgm[:, 1], dgm[:, 0]))]


@pytest.mark.backend
@pytest.mark.alpha
def test_gudhi_encodes_essential_bars_as_inf(circle: np.ndarray) -> None:
    """RFC-0001 §5.1: GUDHI is faithful -- the essential bar is inf."""
    gudhi = pytest.importorskip("gudhi")

    st = gudhi.RipsComplex(points=circle, max_edge_length=MAX_EDGE).create_simplex_tree(
        max_dimension=2
    )
    st.persistence()
    h0 = st.persistence_intervals_in_dimension(0)

    assert len(h0) == N_POINTS, "one H0 bar per point"
    assert np.isinf(h0[:, 1]).sum() == 1, (
        "a connected sample has exactly one essential H0 class; "
        "if this changed, RFC-0001 §5.1 needs revisiting"
    )


@pytest.mark.backend
@pytest.mark.rips
def test_ripser_encodes_essential_bars_as_inf(circle: np.ndarray) -> None:
    """RFC-0001 §5.1: Ripser is faithful -- the essential bar is inf."""
    ripser_mod = pytest.importorskip("ripser")

    dgms = ripser_mod.ripser(circle, maxdim=1)["dgms"]

    assert len(dgms[0]) == N_POINTS
    assert np.isinf(dgms[0][:, 1]).sum() == 1
    # Index in the returned list is the homological degree (§11).
    assert len(dgms) == 2


@pytest.mark.backend
@pytest.mark.cross_backend
def test_gudhi_and_ripser_agree_within_float32_precision(circle: np.ndarray) -> None:
    """RFC-0001 §6.2: cross-backend agreement needs rtol=1e-6, not exactness.

    Ripser returns float64 arrays holding float32-precision values. This is the
    reason exact and approximate equality are separate methods on
    PersistenceDiagram, and the reason cross-backend tests must state their
    tolerance explicitly.
    """
    gudhi = pytest.importorskip("gudhi")
    ripser_mod = pytest.importorskip("ripser")

    st = gudhi.RipsComplex(points=circle, max_edge_length=MAX_EDGE).create_simplex_tree(
        max_dimension=2
    )
    st.persistence()
    g1 = _sorted_bars(st.persistence_intervals_in_dimension(1))
    r1 = _sorted_bars(ripser_mod.ripser(circle, maxdim=1)["dgms"][1])

    assert g1.shape == r1.shape, "backends disagree on the number of H1 bars"

    # They agree as multisets, to single precision.
    assert np.allclose(g1, r1, rtol=CROSS_BACKEND_RTOL, atol=0.0)

    # But not to double precision. If this assertion starts failing, Ripser has
    # moved to float64 -- update RFC-0001 §6.2 and relax the tolerance rather
    # than deleting this test.
    assert not np.allclose(g1, r1, rtol=1e-12, atol=0.0), (
        "Ripser now agrees with GUDHI to double precision. RFC-0001 §6.2 "
        "documents a float32 divergence that no longer holds -- update the spec."
    )


@pytest.mark.backend
@pytest.mark.cross_backend
def test_backends_are_declared_float64(circle: np.ndarray) -> None:
    """RFC-0001 §6.1: dtype is float64 even where the precision is not."""
    gudhi = pytest.importorskip("gudhi")
    ripser_mod = pytest.importorskip("ripser")

    st = gudhi.RipsComplex(points=circle, max_edge_length=MAX_EDGE).create_simplex_tree(
        max_dimension=2
    )
    st.persistence()

    assert st.persistence_intervals_in_dimension(1).dtype == np.float64
    assert ripser_mod.ripser(circle, maxdim=1)["dgms"][1].dtype == np.float64


@pytest.mark.backend
@pytest.mark.distances
def test_persim_returns_finite_distance_between_infinitely_distant_diagrams() -> None:
    """RFC-0001 §9.1: the delegation hazard core/distances.py must guard against.

    The bottleneck distance between a diagram with an essential class and one
    without is infinite -- the essential bar cannot be matched to the diagonal
    at finite cost. persim drops the essential bar and returns a small finite
    number instead. That is why core/distances.py must partition on `essential`
    and return inf itself rather than delegating.

    persim does warn. The warning is asserted here because the first draft of
    RFC-0001 wrongly claimed it did not, having been measured with warnings
    suppressed. Pinning it means the record cannot drift again.
    """
    persim = pytest.importorskip("persim")

    with_essential = np.array([[0.0, np.inf], [0.1, 0.5]])
    all_finite = np.array([[0.0, 1.0], [0.1, 0.5]])

    with pytest.warns(UserWarning, match="non-finite death times"):
        distance = persim.bottleneck(with_essential, all_finite)

    assert np.isfinite(distance), (
        "persim now returns a non-finite distance for diagrams differing in "
        "essential classes. If it returns inf, upstream has fixed the hazard in "
        "RFC-0001 §9.1 -- update the spec and simplify core/distances.py."
    )
    # It is not merely finite, it is small enough to look like agreement.
    assert distance < 1.0


@pytest.mark.backend
@pytest.mark.distances
def test_persim_warning_does_not_distinguish_right_from_wrong() -> None:
    """RFC-0001 §9.1 / A.4: the warning cannot be used to detect the failure.

    Comparing a diagram against itself raises two warnings and gives the right
    answer. Comparing it against a diagram without the essential class raises
    one and gives the wrong answer. The warning tracks whether an argument held
    an essential bar, not whether the result means anything.
    """
    persim = pytest.importorskip("persim")

    with_essential = np.array([[0.0, np.inf], [0.1, 0.5]])
    all_finite = np.array([[0.0, 1.0], [0.1, 0.5]])

    with pytest.warns(UserWarning, match="non-finite death times") as correct_case:
        assert persim.bottleneck(with_essential, with_essential) == 0.0

    with pytest.warns(UserWarning, match="non-finite death times") as wrong_case:
        persim.bottleneck(with_essential, all_finite)

    assert len(correct_case) == 2
    assert len(wrong_case) == 1
    assert len(correct_case) > len(wrong_case), (
        "the correct case warns more loudly than the incorrect one; if this "
        "inverts, persim's warnings have become diagnostic -- update RFC-0001 A.4"
    )


@pytest.mark.backend
@pytest.mark.distances
def test_persim_handles_empty_diagrams() -> None:
    """RFC-0001 §11.2: an empty diagram is a legitimate input, not an error."""
    persim = pytest.importorskip("persim")

    empty = np.zeros((0, 2))
    assert persim.bottleneck(empty, empty) == 0.0
