"""`allclose` at both levels. RFC-0001 §6.2, §6.3.

Scope: the approximate half of §6.3's split -- `PersistenceDiagram.allclose`
and `DiagramBatch.allclose`. `==` is exact and is tested where the invariants
it depends on are; this file is about the method that has to hold up when the
two sides came from *different backends*, which §6.2 guarantees will never be
exactly equal.

**The bijection is checked against a brute-force oracle written from the RFC,
not against the implementation.** §6.3 defines the relation as: there exists a
bijection between the two diagrams' bars under which every matched pair shares
a `dim` exactly and agrees on both coordinates within tolerance.
`_matching_exists` below is that sentence transcribed into
`itertools.permutations` and a four-line tolerance, deliberately exponential
and deliberately naive. The augmenting-path search in `core.py` is an
*optimisation* of that definition, and a test that reimplemented the
optimisation would agree with it for the same wrong reasons.

Three clauses of §6.3 get worked examples rather than being left to the
property tests at the bottom, because each is a case the property tests would
reach only by luck and each is a reason the method has the shape it has:

- the sorted-pairwise false negative (D14), where canonicalising both sides
  and comparing row by row reports `False` for two diagrams that do have a
  bar-for-bar partner within `rtol`;
- equal bar counts with every bar on both sides holding a candidate partner
  and still no bijection -- §6.3's "necessary and not sufficient", and the
  only case that reaches the matching's failure path;
- the symmetric tolerance, on a pair where `numpy.allclose` answers
  differently depending on which argument came first.

Everything here runs on NumPy in the default test environment. The one test
that needs a second namespace asks for it inline; §6.3's cross-namespace rule
is a fact about two backends, and no other test in this file can state it.
"""

from __future__ import annotations

import math
from itertools import permutations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from akriti.diagrams import DiagramBatch, PersistenceDiagram

# Appendix A.3: the measured maximum coordinate difference between GUDHI and
# Ripser on the same H1 bars of the same point cloud, `float32` epsilon at that
# scale being 2.02e-7. Every "two backends disagree by this much" number in
# this file is this one, so a revision of the measurement lands in one place.
CROSS_BACKEND_DELTA = 2.69e-8

# §6.3: the tolerance a cross-backend test MUST state explicitly, rather than
# receive silently from a default.
CROSS_BACKEND_RTOL = 1e-6

Bar = tuple[int, float, float]


def bars(
    dims: list[int], births: list[float], deaths: list[float]
) -> PersistenceDiagram:
    return PersistenceDiagram(
        dims=np.asarray(dims, dtype=np.int32),
        births=np.asarray(births, dtype=np.float64),
        deaths=np.asarray(deaths, dtype=np.float64),
    )


def as_bars(diagram: PersistenceDiagram) -> list[Bar]:
    """A diagram as Python triples, for the oracle to work over."""
    return [
        (int(d), float(b), float(x))
        for d, b, x in zip(
            diagram.dims.tolist(),
            diagram.births.tolist(),
            diagram.deaths.tolist(),
            strict=True,
        )
    ]


# -- the oracle: §6.3's definition, transcribed ---------------------------


def _close(x: float, y: float, rtol: float, atol: float) -> bool:
    """§6.3's symmetric tolerance: `|a - b| <= atol + rtol * max(|a|, |b|)`."""
    return abs(x - y) <= atol + rtol * max(abs(x), abs(y))


def _bar_within(left: Bar, right: Bar, rtol: float, atol: float) -> bool:
    """May these two bars be matched to each other? §6.3, §5.

    `dim` exactly, both coordinates within tolerance, and `inf` matched only
    against `inf` -- a tolerance is undefined there, and §5 stores essential
    bars as `inf` precisely so that they cannot be silently absorbed.
    """
    dim_left, birth_left, death_left = left
    dim_right, birth_right, death_right = right
    if dim_left != dim_right:
        return False
    if not _close(birth_left, birth_right, rtol, atol):
        return False
    if math.isinf(death_left) or math.isinf(death_right):
        return math.isinf(death_left) and math.isinf(death_right)
    return _close(death_left, death_right, rtol, atol)


def _matching_exists(
    left: list[Bar], right: list[Bar], rtol: float, atol: float
) -> bool:
    """Is there a bijection under which every matched pair is within tolerance?

    Every permutation, tried. O(n!) and correct by inspection, which is the
    whole of its value: it is the definition, not an implementation of it.
    Usable only on the four-bar diagrams generated below.
    """
    if len(left) != len(right):
        return False
    return any(
        all(
            _bar_within(bar, right[j], rtol, atol)
            for bar, j in zip(left, order, strict=True)
        )
        for order in permutations(range(len(right)))
    )


# -- §6.3: allclose is a matching -----------------------------------------


def test_a_diagram_is_allclose_to_itself_and_to_its_own_permutations() -> None:
    """Reflexive, and order-insensitive within a diagram (§7).

    Bar order is not part of a diagram's identity, so a backend that emits the
    same bars in a different order -- which §7 measures GUDHI and Ripser doing
    on identical input -- must not change the answer.
    """
    diagram = bars([0, 0, 1], [0.0, 0.25, 0.5], [np.inf, 0.75, 1.5])
    reordered = bars([1, 0, 0], [0.5, 0.25, 0.0], [1.5, 0.75, np.inf])

    assert diagram.allclose(diagram)
    assert diagram.allclose(reordered)
    assert reordered.allclose(diagram)


def test_the_empty_bijection_is_a_bijection() -> None:
    """Two empty diagrams are allclose; empty against non-empty is not."""
    empty = bars([], [], [])
    assert empty.allclose(empty)
    assert not empty.allclose(bars([0], [0.0], [1.0]))
    assert not bars([0], [0.0], [1.0]).allclose(empty)


def test_unequal_bar_counts_are_never_allclose() -> None:
    """A bijection needs equal cardinality, whatever the tolerance.

    A diagram whose bars are all *contained* in another's is the tempting
    case: every bar on the left has a partner, and there is still no
    bijection.
    """
    two = bars([0, 0], [0.0, 1.0], [1.0, 2.0])
    one = bars([0], [0.0], [1.0])
    assert not two.allclose(one, rtol=1.0, atol=100.0)
    assert not one.allclose(two, rtol=1.0, atol=100.0)


def test_equal_bar_counts_and_a_partner_for_every_bar_are_not_sufficient() -> None:
    """§6.3: "necessary and not sufficient", and the case that proves it.

    Every bar on the left has a candidate on the right and every bar on the
    right has one on the left -- so both of the cheap array reductions in
    `allclose` pass -- and no bijection exists, because the left's two bars at
    0.0 compete for the right's single bar at 0.05. This is the only shape of
    input that reaches the augmenting-path search's failing exit, and an
    implementation that stopped at "is every bar matchable?" would answer
    `True` here.
    """
    left = bars([0, 0, 0], [0.0, 0.0, 1.0], [3.0, 3.0, 3.0])
    right = bars([0, 0, 0], [0.05, 1.0, 1.05], [3.0, 3.0, 3.0])
    rtol, atol = 0.0, 0.1

    left_bars, right_bars = as_bars(left), as_bars(right)
    assert all(
        any(_bar_within(a, b, rtol, atol) for b in right_bars) for a in left_bars
    ), "every left bar must have a candidate, or this tests the wrong thing"
    assert all(
        any(_bar_within(a, b, rtol, atol) for a in left_bars) for b in right_bars
    ), "every right bar must have a candidate, or this tests the wrong thing"

    assert not _matching_exists(left_bars, right_bars, rtol, atol)
    assert not left.allclose(right, rtol=rtol, atol=atol)
    assert not right.allclose(left, rtol=rtol, atol=atol)


def test_a_near_tie_in_birth_must_not_produce_a_false_negative() -> None:
    """§6.3 / D14: why this is a matching and not a sorted pairwise comparison.

    Two backends return the same two bars and disagree by `2.69e-8` on a birth
    time that is very nearly tied, so they canonicalise into opposite orders
    (§7). Sorting both sides and comparing row by row then pairs each bar
    against the other's partner and reports `False` for two diagrams that do
    have a bar-for-bar partner within `rtol` -- the false negative §6.3
    rejects, at exactly the magnitude Appendix A.3 measures.

    The middle assertion pins that the pairwise comparison really does fail
    here, so this stays a regression test for D14 rather than a test that
    passes because the fast path answered first.
    """
    left = bars([1, 1], [1.0, 1.0 + CROSS_BACKEND_DELTA], [2.0, 5.0])
    right = bars([1, 1], [1.0, 1.0 + CROSS_BACKEND_DELTA], [5.0, 2.0])

    a, b = left.canonical(), right.canonical()
    assert not np.all(
        np.abs(a.deaths - b.deaths)
        <= CROSS_BACKEND_RTOL * np.maximum(np.abs(a.deaths), np.abs(b.deaths))
    ), "canonical order no longer separates these; rebuild the near tie"

    assert _matching_exists(as_bars(left), as_bars(right), CROSS_BACKEND_RTOL, 0.0)
    assert left.allclose(right, rtol=CROSS_BACKEND_RTOL)
    assert right.allclose(left, rtol=CROSS_BACKEND_RTOL)


def test_the_same_near_tie_with_essential_bars() -> None:
    """D14 where the swapped partner is `inf`. §5, §6.3.

    The pair matrix has to reach the right answer with `inf` in cells the
    tolerance formula must never be applied to: `inf - inf` is `nan` and
    `inf - 2.0` is `inf`, and neither may decide anything. The matching is
    forced off the diagonal, so both the `inf`-to-`inf` cells and the mixed
    cells are consulted.
    """
    left = bars([0, 0], [1.0, 1.0 + CROSS_BACKEND_DELTA], [np.inf, 2.0])
    right = bars([0, 0], [1.0, 1.0 + CROSS_BACKEND_DELTA], [2.0, np.inf])

    assert _matching_exists(as_bars(left), as_bars(right), CROSS_BACKEND_RTOL, 0.0)
    assert left.allclose(right, rtol=CROSS_BACKEND_RTOL)
    assert right.allclose(left, rtol=CROSS_BACKEND_RTOL)


# -- §6.2, §6.3: the tolerance itself -------------------------------------


def test_the_default_rtol_is_tighter_than_cross_backend_reality() -> None:
    """§6.2, §6.3: a cross-backend comparison MUST state its tolerance.

    The two bars are Appendix A.3's measurements verbatim -- the same H1
    bar as GUDHI and as Ripser returned it. Under the default `rtol=1e-9`
    they are not allclose, which is the point: a default that silently
    absorbed a `float32` divergence would absorb genuine disagreement
    everywhere else.
    """
    gudhi_bar = bars([1], [0.52018979], [1.69521069])
    ripser_bar = bars([1], [0.52018976], [1.6952107])

    assert not gudhi_bar.allclose(ripser_bar)
    assert gudhi_bar.allclose(ripser_bar, rtol=CROSS_BACKEND_RTOL)


def test_the_default_atol_is_zero() -> None:
    """A relative tolerance alone decides nothing near zero.

    Zero births are ubiquitous in H0, so this is the ordinary case rather
    than a constructed one: with `atol=0.0` two births straddling zero must
    agree almost exactly, and a caller who wants otherwise passes `atol`.
    """
    at_zero = bars([0], [0.0], [1.0])
    just_above = bars([0], [1e-300], [1.0])

    assert not at_zero.allclose(just_above)
    assert at_zero.allclose(just_above, atol=1e-12)


def test_the_tolerance_is_symmetric_where_numpy_allclose_is_not() -> None:
    """§6.3: `atol + rtol * max(|a|, |b|)`, and the divergence is deliberate.

    `numpy.allclose` scales `rtol` by its second argument alone, so swapping
    the arguments changes its answer on this pair. Which diagram was written
    first is not part of the question this method answers.
    """
    rtol = 0.5
    left = bars([0], [1.0], [3.0])
    right = bars([0], [1.6], [3.0])

    assert np.allclose(1.0, 1.6, rtol=rtol, atol=0.0)
    assert not np.allclose(1.6, 1.0, rtol=rtol, atol=0.0), (
        "numpy.allclose is symmetric on this pair now; the divergence "
        "RFC-0001 §6.3 documents needs re-examining, not this test"
    )

    assert left.allclose(right, rtol=rtol)
    assert right.allclose(left, rtol=rtol)


def test_allclose_is_not_transitive() -> None:
    """§6.3: reflexive and symmetric, and not an equivalence relation.

    Two tolerance-width steps span twice the tolerance. Pinned rather than
    merely documented, because the documentation exists to stop callers using
    `allclose` to deduplicate or group diagrams -- where the answer would
    depend on visit order -- and a behaviour nobody tests is one somebody
    later "fixes".
    """
    a = bars([0], [0.0], [1.0])
    b = bars([0], [0.06], [1.0])
    c = bars([0], [0.12], [1.0])
    atol = 0.1

    assert a.allclose(b, rtol=0.0, atol=atol)
    assert b.allclose(c, rtol=0.0, atol=atol)
    assert not a.allclose(c, rtol=0.0, atol=atol)


# -- §5, §6.3: essential bars are matched exactly -------------------------


def test_an_essential_bar_matches_only_an_essential_bar() -> None:
    """§5: `inf` is the storage, and no tolerance reaches it.

    A finite death of 1e300 is not nearly infinite, however wide the tolerance
    is opened. The alternative -- a large `atol` quietly making a class that
    never dies equal to one that does -- is the substitution §5 requires to be
    explicit, arrived at by accident.
    """
    essential = bars([0], [0.0], [np.inf])
    enormous = bars([0], [0.0], [1e300])

    assert not essential.allclose(enormous, rtol=1.0, atol=1e300)
    assert not enormous.allclose(essential, rtol=1.0, atol=1e300)
    assert essential.allclose(essential)


def test_the_zero_substituted_for_an_inf_decides_nothing() -> None:
    """The masking, tested at the one value that would expose its absence.

    `core.py` substitutes `0.0` for `inf` before applying the tolerance, to
    keep `inf - inf` -> `nan` and `inf - x` -> `inf` out of the arithmetic in
    cells the masks discard anyway. A bar that genuinely dies at `0.0` is
    therefore the input that tells the two apart: if the substituted value
    ever reached the comparison, this pair would be allclose, and a class
    that never dies would have been matched to one that dies immediately.
    """
    essential = bars([0], [0.0], [np.inf])
    dies_at_zero = bars([0], [0.0], [0.0])

    assert not essential.allclose(dies_at_zero, rtol=1.0, atol=1.0)
    assert not dies_at_zero.allclose(essential, rtol=1.0, atol=1.0)


def test_essential_bars_still_have_to_agree_on_birth() -> None:
    """Matching the deaths exactly does not exempt the births from tolerance."""
    born_early = bars([0], [0.0], [np.inf])
    born_late = bars([0], [5.0], [np.inf])

    assert not born_early.allclose(born_late)
    assert born_early.allclose(born_late, rtol=0.0, atol=10.0)


def test_a_diagram_of_only_essential_bars_is_allclose_to_itself() -> None:
    """`inf - inf` is `nan`, and no `nan` may reach the comparison.

    A `nan` compares false against everything, so an implementation that let
    one through would report `False` here -- for a diagram against itself.
    """
    all_essential = bars([0, 0, 1], [0.0, 1.0, 2.0], [np.inf, np.inf, np.inf])
    assert all_essential.allclose(all_essential)
    assert all_essential.allclose(bars([1, 0, 0], [2.0, 1.0, 0.0], [np.inf] * 3))


# -- §6.3: dims are compared exactly --------------------------------------


def test_dims_are_compared_exactly_at_any_tolerance() -> None:
    """§6.3: the tolerance is over coordinates, never over homological degree."""
    h0 = bars([0], [1.0], [2.0])
    h1 = bars([1], [1.0], [2.0])

    assert not h0.allclose(h1, rtol=1e3, atol=1e3)
    assert not h1.allclose(h0, rtol=1e3, atol=1e3)


def test_a_bijection_may_not_pair_bars_across_degrees() -> None:
    """The coordinates match perfectly; the degrees are swapped.

    A matching that ignored `dim` would find a bijection here, and the two
    diagrams say different things about the space.
    """
    left = bars([0, 1], [0.0, 5.0], [1.0, 6.0])
    right = bars([1, 0], [0.0, 5.0], [1.0, 6.0])

    assert not _matching_exists(as_bars(left), as_bars(right), 1e-9, 0.0)
    assert not left.allclose(right)


# -- §6.3: what allclose refuses to answer --------------------------------


def test_a_non_diagram_raises_typeerror_rather_than_returning_false() -> None:
    """§6.3: `==` may be asked to compare against anything; this may not.

    A `DiagramBatch` is in the list because it is the plausible mistake: it
    has an `allclose` of its own with the same signature, and a `False` here
    would look like an answer about its contents.
    """
    diagram = bars([0], [0.0], [1.0])
    batch = DiagramBatch.from_diagrams([diagram])

    for other in (object(), None, [diagram], batch):
        with pytest.raises(TypeError, match="PersistenceDiagram"):
            diagram.allclose(other)  # type: ignore[arg-type]


def test_cross_namespace_raises_before_the_length_comparison() -> None:
    """§6.3: the failure MUST NOT depend on the data.

    The two diagrams have different bar counts, so an implementation that
    checked lengths first would return a clean, plausible `False` -- §9's
    whole category of bug -- and would do it only for the pairs that happen to
    differ in length. Both directions, since either side may be the receiver.
    """
    xps = pytest.importorskip("array_api_strict")

    numpy_backed = bars([0], [0.0], [1.0])
    strict_backed = PersistenceDiagram(
        dims=xps.asarray([0, 0], dtype=xps.int32),
        births=xps.asarray([0.0, 1.0], dtype=xps.float64),
        deaths=xps.asarray([1.0, 2.0], dtype=xps.float64),
    )

    with pytest.raises(ValueError, match="allclose cannot compare across"):
        numpy_backed.allclose(strict_backed)
    with pytest.raises(ValueError, match="allclose cannot compare across"):
        strict_backed.allclose(numpy_backed)


# -- §6.3: DiagramBatch -- order-sensitive over an order-insensitive thing --


def test_batch_compares_position_by_position() -> None:
    """§6.3: `len` equal and `b1[i].allclose(b2[i])` for every `i`."""
    first = bars([0], [1.0], [2.0])
    second = bars([1], [0.5], [np.inf])
    shifted_first = bars([0], [1.0 + CROSS_BACKEND_DELTA], [2.0])

    left = DiagramBatch.from_diagrams([first, second])
    right = DiagramBatch.from_diagrams([shifted_first, second])

    assert left.allclose(right, rtol=CROSS_BACKEND_RTOL)
    assert right.allclose(left, rtol=CROSS_BACKEND_RTOL)

    # One position disagreeing is enough, and it is the *only* position that
    # differs -- a batch comparison that reduced over the concatenated buffer
    # instead of over positions could lose this.
    disagreeing = DiagramBatch.from_diagrams([first, bars([1], [0.5], [9.0])])
    assert not left.allclose(disagreeing, rtol=CROSS_BACKEND_RTOL)


def test_batch_forwards_rtol_and_atol_to_every_member() -> None:
    """A tolerance the caller states MUST reach the per-diagram comparison."""
    left = DiagramBatch.from_diagrams([bars([0], [1.0], [2.0])])
    right = DiagramBatch.from_diagrams([bars([0], [1.0 + CROSS_BACKEND_DELTA], [2.0])])

    assert not left.allclose(right)
    assert left.allclose(right, rtol=CROSS_BACKEND_RTOL)
    assert left.allclose(right, rtol=0.0, atol=1e-6)


def test_batch_order_is_meaningful_and_bar_order_is_not() -> None:
    """§6.3: a batch is an order-sensitive container of order-insensitive things.

    Both halves in one test, because it is the pair that is easy to get
    wrong: permuting the bars *inside* each diagram must change nothing, and
    permuting the diagrams themselves must change everything.
    """
    a = bars([0, 0], [0.0, 0.25], [np.inf, 0.75])
    b = bars([1, 1], [0.5, 0.6], [1.5, 1.6])
    a_reordered = bars([0, 0], [0.25, 0.0], [0.75, np.inf])
    b_reordered = bars([1, 1], [0.6, 0.5], [1.6, 1.5])

    batch = DiagramBatch.from_diagrams([a, b])
    assert batch.allclose(DiagramBatch.from_diagrams([a_reordered, b_reordered]))
    assert not batch.allclose(DiagramBatch.from_diagrams([b, a]))


def test_batches_of_different_lengths_are_not_allclose() -> None:
    diagram = bars([0], [0.0], [1.0])
    one = DiagramBatch.from_diagrams([diagram])
    two = DiagramBatch.from_diagrams([diagram, diagram])

    assert not one.allclose(two, rtol=1.0, atol=100.0)
    assert not two.allclose(one, rtol=1.0, atol=100.0)


def test_empty_batches_are_allclose() -> None:
    """A batch a permutation test filtered down to nothing is still a batch."""
    empty = DiagramBatch.from_diagrams([], xp=np)
    populated = DiagramBatch.from_diagrams([bars([0], [0.0], [1.0])])

    assert empty.allclose(empty)
    assert not empty.allclose(populated)
    assert not populated.allclose(empty)


def test_batch_rejects_a_non_batch_including_a_bare_diagram() -> None:
    """A one-diagram batch and a diagram are not interchangeable (§4.1)."""
    diagram = bars([0], [0.0], [1.0])
    batch = DiagramBatch.from_diagrams([diagram])

    for other in (object(), None, [diagram], diagram):
        with pytest.raises(TypeError, match="DiagramBatch"):
            batch.allclose(other)  # type: ignore[arg-type]


def test_batch_cross_namespace_raises_before_the_length_comparison() -> None:
    """§6.3, hoisted to the batch: the same data-independence, one level up.

    The lengths differ here too, so a batch that left the namespace check to
    its first `self[i].allclose(other[i])` would return `False` for this pair
    and raise for the equal-length one -- the failure depending on the data
    after all.
    """
    xps = pytest.importorskip("array_api_strict")

    numpy_backed = DiagramBatch.from_diagrams([bars([0], [0.0], [1.0])])
    strict_backed = DiagramBatch.from_diagrams(
        [
            PersistenceDiagram(
                dims=xps.asarray([0], dtype=xps.int32),
                births=xps.asarray([0.0], dtype=xps.float64),
                deaths=xps.asarray([1.0], dtype=xps.float64),
            )
        ]
        * 2
    )

    with pytest.raises(ValueError, match="allclose cannot compare across"):
        numpy_backed.allclose(strict_backed)
    with pytest.raises(ValueError, match="allclose cannot compare across"):
        strict_backed.allclose(numpy_backed)


# -- property-based: the implementation against the definition ------------
#
# The generator draws a *comparison* -- two diagrams and the tolerance they
# will be compared at -- rather than drawing the three independently, and the
# second diagram is usually the first one permuted and nudged by a fraction of
# that tolerance's own width. Both choices are load-bearing, and each was made
# after watching a deliberately broken `core.py` survive the generator that
# lacked it:
#
# - Independently drawn coordinates are never within tolerance of one another,
#   so a generator over `st.floats` spends every example confirming that two
#   unrelated diagrams differ. A `core.py` whose matching always answers
#   `False` -- a sorted-pairwise implementation in all but name -- passes it.
# - A perturbation of fixed size is either far inside the tolerance or far
#   outside it, and both are cases the two cheap array reductions in
#   `allclose` settle before the matching runs. Scaling the nudge by
#   `atol + rtol * |x|` puts pairs *at* the boundary, which is the only place
#   the bijection is in doubt. Measured: a fixed-size nudge reached the
#   augmenting-path search in 5 examples out of 1000.
# - The permutation is what forces the bijection off the diagonal. Ties in
#   `BIRTHS` are frequent by design, and two bars sharing a birth with
#   different deaths are what canonical order cannot separate stably (D14).
# - The nudge flips a bar between essential and finite now and then. Without
#   it `inf` is only ever compared against `inf`.

# Few birth values and several lifetimes, so that two bars sharing a birth
# and differing in death -- the pair canonical order cannot separate stably,
# and the whole of D14 -- is a common draw rather than a rare coincidence.
BIRTHS = [-1.0, 0.0, 1.0, 2.0]
LIFETIMES = [0.0, 0.25, 1.0, 3.0]

# The last entry is what makes the symmetric tolerance observable: with a
# nudge of `0.4 * 1.0` off a birth at 1.0, `numpy.allclose`'s asymmetric form
# answers one way from the left and the other from the right.
TOLERANCES = [(1e-9, 0.0), (CROSS_BACKEND_RTOL, 0.0), (0.0, 0.1), (0.4, 0.0)]

# Fractions of the tolerance window. `INSIDE` keeps every nudged coordinate
# within tolerance of the one it came from, so a bijection survives however
# far the bars have moved; `STRADDLING` adds fractions that leave the window.
# Which of the two a comparison draws matters more than any other knob here:
# with `STRADDLING` alone, a single out-of-tolerance bar leaves `allclose`'s
# two array reductions with nothing to match, and the augmenting-path search
# is never reached at all. Measured: 2 examples in 1000.
INSIDE_FRACTIONS = [0.0, 0.5, 0.9]
STRADDLING_FRACTIONS = [0.0, 0.9, 1.1, 3.0]

_BAR = st.tuples(
    # Weighted towards H0, as a real diagram is, and for a second reason:
    # canonical order sorts on `dim` first, so two bars can only be a near
    # tie in birth if they already share a degree.
    st.sampled_from([0, 0, 0, 1, 1, 2]),
    st.sampled_from(BIRTHS),
    # `None` is an essential bar; anything else is a lifetime, added to the
    # birth to give a death, so I6 holds by construction.
    st.one_of(st.sampled_from(LIFETIMES), st.none()),
)


def _from_bar_list(raw: list[Bar]) -> PersistenceDiagram:
    return bars(
        [dim for dim, _, _ in raw],
        [birth for _, birth, _ in raw],
        [death for _, _, death in raw],
    )


@st.composite
def diagrams(draw: st.DrawFn) -> PersistenceDiagram:
    raw = draw(st.lists(_BAR, max_size=4))
    return _from_bar_list(
        [
            (dim, birth, math.inf if lifetime is None else birth + lifetime)
            for dim, birth, lifetime in raw
        ]
    )


@st.composite
def _nudged(
    draw: st.DrawFn, value: float, rtol: float, atol: float, fractions: list[float]
) -> float:
    """`value` moved by a drawn fraction of the tolerance window around it."""
    fraction = draw(st.sampled_from(fractions))
    sign = draw(st.sampled_from([1.0, -1.0]))
    return value + sign * fraction * (atol + rtol * abs(value))


@st.composite
def perturbed(
    draw: st.DrawFn,
    source: PersistenceDiagram,
    rtol: float,
    atol: float,
    fractions: list[float],
) -> PersistenceDiagram:
    """`source` permuted and nudged: what a second backend would have sent.

    Deaths are clamped back up to the birth where a nudge would break I6. An
    invalid diagram is not a case this comparison has to answer for, and the
    clamped bar is an ordinary one that the oracle sees exactly as it is --
    the clamp lands between the birth and the death it came from, so it does
    not leave a window `fractions` kept it inside.
    """
    n = source.n_bars
    order = draw(st.permutations(range(n)))
    source_bars = as_bars(source)

    # At most one bar per diagram gets its degree changed or gets flipped
    # between essential and finite, and usually none does. Both are
    # unmatchable-bar edits -- `dims` are compared exactly, and `inf` matches
    # only `inf` -- and one unmatchable bar is all it takes for `allclose`'s
    # two array reductions to answer before the matching runs. Applying them
    # per bar independently left only a tenth of four-bar pairs intact, and
    # the augmenting-path search unreached in all but 8 examples of 1000.
    corrupt = None
    if n and draw(st.sampled_from([True, False, False, False])):
        corrupt = draw(st.integers(min_value=0, max_value=n - 1))
    corruption = draw(st.sampled_from(["degree", "essential"]))

    out: list[Bar] = []
    for position, i in enumerate(order):
        dim, birth, death = source_bars[i]
        birth = draw(_nudged(birth, rtol, atol, fractions))
        if not math.isinf(death):
            death = max(birth, draw(_nudged(death, rtol, atol, fractions)))
        if position == corrupt:
            if corruption == "degree":
                dim = (dim + 1) % 3
            else:
                death = birth if math.isinf(death) else math.inf
        out.append((dim, birth, death))

    # D14, constructed rather than waited for. Two bars that shared a degree
    # and a birth in `source`, and differ in death, exchange the births they
    # were just nudged to. Both nudges came off the same source value, so each
    # bar stays within tolerance of the bar it came from and a bijection still
    # exists -- and the two are now ordered by birth in the way the nudges
    # fell, which is exactly what makes canonical order pair each of them
    # against the other's partner. Waiting for the independent nudges to land
    # this way reached the matching in 11 examples of 1000.
    swappable = [
        (p, q)
        for p in range(n)
        for q in range(p + 1, n)
        if source_bars[order[p]][:2] == source_bars[order[q]][:2]
        and source_bars[order[p]][2] != source_bars[order[q]][2]
    ]
    if swappable and draw(st.booleans()):
        p, q = draw(st.sampled_from(swappable))
        out[p], out[q] = (
            _rebirth(out[p], out[q][1]),
            _rebirth(out[q], out[p][1]),
        )

    return _from_bar_list(out)


def _rebirth(bar: Bar, birth: float) -> Bar:
    """`bar` given a new birth, with its death clamped back up to satisfy I6.

    The clamp matters for a bar whose lifetime is zero, where a birth moved
    up by even one nudge would otherwise leave a death below it and make the
    diagram unconstructible.
    """
    dim, _, death = bar
    return (dim, birth, death if math.isinf(death) else max(birth, death))


@st.composite
def comparisons(
    draw: st.DrawFn,
) -> tuple[PersistenceDiagram, PersistenceDiagram, float, float]:
    """A pair and the tolerance to compare it at, drawn together."""
    rtol, atol = draw(st.sampled_from(TOLERANCES))
    left = draw(diagrams())
    kind = draw(st.sampled_from(["independent", "inside", "inside", "straddling"]))
    if kind == "independent":
        return left, draw(diagrams()), rtol, atol
    fractions = INSIDE_FRACTIONS if kind == "inside" else STRADDLING_FRACTIONS
    return left, draw(perturbed(left, rtol, atol, fractions)), rtol, atol


@given(comparison=comparisons())
@settings(max_examples=500, deadline=None)
def test_allclose_is_exactly_the_bijection_the_rfc_defines(
    comparison: tuple[PersistenceDiagram, PersistenceDiagram, float, float],
) -> None:
    """§6.3's definition, against `core.py`'s augmenting-path optimisation.

    Equality of the two answers, not implication: a spurious `True` would
    certify agreement that does not exist, and a spurious `False` sends a
    caller to widen `rtol` in their own source until the comparison passes --
    the silent loosening §6.3 exists to keep out of user code.
    """
    left, right, rtol, atol = comparison
    assert left.allclose(right, rtol=rtol, atol=atol) == _matching_exists(
        as_bars(left), as_bars(right), rtol, atol
    )


@given(comparison=comparisons())
@settings(max_examples=300, deadline=None)
def test_allclose_is_reflexive_and_symmetric(
    comparison: tuple[PersistenceDiagram, PersistenceDiagram, float, float],
) -> None:
    """§6.3: the two properties it does have. Transitivity is tested absent."""
    left, right, rtol, atol = comparison
    assert left.allclose(left, rtol=rtol, atol=atol)
    assert left.allclose(right, rtol=rtol, atol=atol) == right.allclose(
        left, rtol=rtol, atol=atol
    )


@given(comparison=comparisons(), data=st.data())
@settings(max_examples=300, deadline=None)
def test_permuting_bars_never_changes_the_answer(
    comparison: tuple[PersistenceDiagram, PersistenceDiagram, float, float],
    data: st.DataObject,
) -> None:
    """§7: row order is arbitrary, so no comparison may depend on it."""
    left, right, rtol, atol = comparison
    order = data.draw(st.permutations(range(right.n_bars)))
    shuffled = _from_bar_list([as_bars(right)[i] for i in order])

    assert left.allclose(right, rtol=rtol, atol=atol) == left.allclose(
        shuffled, rtol=rtol, atol=atol
    )


@given(
    members=st.lists(comparisons(), max_size=3),
    truncate=st.sampled_from([None, "left", "right"]),
    tol=st.sampled_from(TOLERANCES),
)
@settings(max_examples=200, deadline=None)
def test_batch_allclose_is_the_positionwise_conjunction(
    members: list[tuple[PersistenceDiagram, PersistenceDiagram, float, float]],
    truncate: str | None,
    tol: tuple[float, float],
) -> None:
    """§6.3: equal lengths and every position allclose, and nothing else.

    `truncate` drops the last diagram from one side, so unequal lengths are
    generated as often as equal ones rather than only when the two drawn
    lists happen to differ in length.
    """
    rtol, atol = tol
    left = [a for a, _, _, _ in members]
    right = [b for _, b, _, _ in members]
    if truncate == "left":
        left = left[:-1]
    elif truncate == "right":
        right = right[:-1]

    expected = len(left) == len(right) and all(
        a.allclose(b, rtol=rtol, atol=atol) for a, b in zip(left, right, strict=False)
    )
    left_batch = DiagramBatch.from_diagrams(left, xp=np)
    right_batch = DiagramBatch.from_diagrams(right, xp=np)

    assert left_batch.allclose(right_batch, rtol=rtol, atol=atol) == expected
