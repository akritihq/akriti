"""Enforcement points in `diagrams/core.py`. RFC-0001 §3.1, §4.2, §5, §8.

Scope: the places where `core.py` *rejects* something, plus the two accessors
whose failure modes are easy to get subtly wrong. This is not §11.2's
round-trip suite -- that one requires real backend output and belongs with the
adapters, which do not exist yet.

The common thread is that every check here fires on construction or on a bad
argument, never on the data path. §3.1 requires an invalid instance to be
unconstructible, and a requirement stated only as an obligation on writers is
one every future writer has to remember independently.
"""

from __future__ import annotations

import collections.abc as cabc

import numpy as np
import pytest

from akriti.diagrams import DiagramBatch, DiagramMeta, PersistenceDiagram


def bars(
    dims: list[int], births: list[float], deaths: list[float], **meta: object
) -> PersistenceDiagram:
    return PersistenceDiagram(
        dims=np.asarray(dims, dtype=np.int32),
        births=np.asarray(births, dtype=np.float64),
        deaths=np.asarray(deaths, dtype=np.float64),
        **meta,  # type: ignore[arg-type]
    )


@pytest.fixture
def essential() -> PersistenceDiagram:
    return bars([0, 0], [0.0, 0.25], [np.inf, 0.75])


# -- §8 provenance consistency, enforced at construction ------------------


def test_dropped_count_without_finitized_dropped_is_rejected() -> None:
    """§8: `essential_bars_dropped` present *iff* `"finitized_dropped"`.

    The forward direction. A count outliving the substitution that replaced
    the drop is §8's named failure -- a diagram claiming both a cardinality
    change and a value substitution, with a count belonging to neither.
    """
    with pytest.raises(ValueError, match="essential_bars_dropped"):
        DiagramMeta(
            provenance={"essential_bars": "faithful", "essential_bars_dropped": 5}
        )


def test_finitized_dropped_without_a_count_is_rejected() -> None:
    """§8's iff, in the other direction: the record half-made."""
    with pytest.raises(ValueError, match="missing"):
        DiagramMeta(provenance={"essential_bars": "finitized_dropped"})


def test_essential_bars_source_rejects_a_finitized_value() -> None:
    """§8: the key records the adapter-time verdict, which is never finitized.

    This is the copy-forward mistake §5 rejects, caught at the point it would
    have to be written rather than left to reading discipline.
    """
    with pytest.raises(ValueError, match="essential_bars_source"):
        DiagramMeta(provenance={"essential_bars_source": "finitized_at:1.0"})


def test_adapter_shaped_provenance_is_accepted() -> None:
    """The check must not reject what §5.1 requires `from_giotto` to write."""
    meta = DiagramMeta(
        backend="giotto",
        params={"reduced_homology": True},
        provenance={
            "essential_bars": "lost_upstream",
            "essential_bars_source": "lost_upstream",
        },
    )
    assert meta.provenance["essential_bars_source"] == "lost_upstream"


def test_unreserved_provenance_keys_are_left_alone() -> None:
    """§8 reserves names inside `provenance`; it does not close the mapping."""
    meta = DiagramMeta(provenance={"anything": object(), "source_dtype": "float32"})
    assert "anything" in meta.provenance


# -- §5 finitize keeps provenance consistent across calls -----------------


def test_drop_then_substitute_clears_the_stale_count(
    essential: PersistenceDiagram,
) -> None:
    """§8's iff across two calls -- the case a naive dict merge breaks.

    `finitize(at="drop")` leaves no essential bar, so §5's return-unchanged
    rule makes the second call a no-op and the count correctly survives. The
    order that does reach a substitution is the one built below.
    """
    dropped = essential.finitize(at="drop")
    assert dropped.meta.provenance["essential_bars"] == "finitized_dropped"
    assert dropped.meta.provenance["essential_bars_dropped"] == 1

    # A diagram carrying a completed drop record that still has an essential
    # bar: what `load` would hand back if a caller finitized, saved, and later
    # concatenated. The substitution must clear the count, not merge past it.
    reloaded = bars(
        [0, 0],
        [0.0, 0.25],
        [np.inf, 0.75],
        meta=DiagramMeta(provenance=dict(dropped.meta.provenance)),
    )
    substituted = reloaded.finitize(at=2.0)
    assert substituted.meta.provenance["essential_bars"] == "finitized_at:2.0"
    assert "essential_bars_dropped" not in substituted.meta.provenance


def test_finitize_never_writes_or_disturbs_essential_bars_source() -> None:
    """§5: the second key has one writer and it is the adapter."""
    diagram = bars(
        [0],
        [0.0],
        [np.inf],
        meta=DiagramMeta(
            provenance={
                "essential_bars": "lost_upstream",
                "essential_bars_source": "lost_upstream",
            }
        ),
    )
    finitized = diagram.finitize(at=1.0)
    assert finitized.meta.provenance["essential_bars"] == "finitized_at:1.0"
    assert finitized.meta.provenance["essential_bars_source"] == "lost_upstream"


def test_no_essential_bars_returns_the_diagram_untouched() -> None:
    """§5: no substitution and no drop means nothing to record."""
    finite = bars([0], [0.0], [1.0], meta=DiagramMeta(provenance={"order": "backend"}))
    assert finite.finitize(at="drop") is finite
    assert finite.finitize(at=99.0) is finite
    assert "essential_bars" not in finite.finitize(at="drop").meta.provenance


# -- §5 `at` is validated in full, before the data is consulted -----------


@pytest.mark.parametrize("at", [None, [1.0], b"drop", b"2.0", True, False])
def test_non_numeric_at_raises_typeerror_regardless_of_the_data(
    at: object, essential: PersistenceDiagram
) -> None:
    """§5: `TypeError` for a call no diagram could make meaningful.

    `True` is in this list because `bool` is an `int`: without an explicit
    rejection it would substitute a death time of `1.0` and record
    `"finitized_at:1.0"`, a provenance entry describing a substitution nobody
    asked for. `b"2.0"` is here because `float(b"2.0")` succeeds.
    """
    finite = bars([0], [0.0], [1.0])
    for diagram in (essential, finite):
        with pytest.raises(TypeError):
            diagram.finitize(at=at)  # type: ignore[arg-type]


@pytest.mark.parametrize("at", [np.inf, -np.inf, float("nan")])
def test_non_finite_at_raises_on_the_argument_not_the_death_times(
    at: float, essential: PersistenceDiagram
) -> None:
    """§5: `at=inf` would leave every essential bar essential and say otherwise."""
    with pytest.raises(ValueError, match="finite substitution value"):
        essential.finitize(at=at)


def test_unknown_mode_name_raises_valueerror_on_a_diagram_without_essentials() -> None:
    """§5: the typo must not depend on the data to be caught."""
    with pytest.raises(ValueError, match="unknown finitize mode"):
        bars([0], [0.0], [1.0]).finitize(at="drpo")


def test_max_finite_death_below_an_essential_birth_is_attributed_to_the_data() -> None:
    """§5: I6 is the right judge here, and its message names death times."""
    diagram = bars([0, 0], [0.0, 2.0], [0.5, np.inf])
    with pytest.raises(ValueError, match=r"deaths must be >= births"):
        diagram.finitize(at="max_finite_death")


# -- §4.2 indexing --------------------------------------------------------


def test_out_of_range_index_reports_what_the_caller_passed() -> None:
    """A normalised index in the error names a value never asked for."""
    batch = DiagramBatch.from_diagrams([bars([0], [0.0], [1.0])] * 2)
    with pytest.raises(IndexError, match=r"index -5 .*length 2"):
        batch[-5]
    with pytest.raises(IndexError, match=r"index 7 .*length 2"):
        batch[7]


def test_negative_indices_still_count_from_the_end() -> None:
    a, b = bars([0], [0.0], [1.0]), bars([1], [0.5], [2.0])
    batch = DiagramBatch.from_diagrams([a, b])
    assert batch[-1] == b
    assert batch[-2] == a


def test_offsets_are_read_once_and_reused() -> None:
    """The cache `__getitem__` and the B2-B4 checks share.

    Not a performance assertion -- it pins that the cached bounds agree with
    the array they came from, since a stale or wrong cache would slice the
    shared buffer at the wrong boundaries and every diagram in the batch
    would be quietly wrong.
    """
    diagrams = [bars([0], [0.0], [1.0]), bars([1, 1], [0.0, 0.5], [1.0, 2.0])]
    batch = DiagramBatch.from_diagrams(diagrams)
    assert batch._bounds() == (0, 1, 3)
    assert batch._bounds() is batch._bounds()
    assert [d.n_bars for d in batch] == [1, 2]
    assert batch[1] == diagrams[1]


# -- §8 metadata is not hashable, and says so before it is asked ----------


def test_diagram_meta_is_unhashable_by_protocol_not_by_accident() -> None:
    """A generated `__hash__` would pass `Hashable` and fail at call time.

    `@dataclass(frozen=True, eq=True)` adds one by default. Left in place it
    raises `TypeError: unhashable type: 'dict'` from inside the generated
    tuple hash -- naming neither this class nor the field responsible -- while
    `isinstance(m, Hashable)` answers `True` and any guard branching on it
    takes the wrong path.
    """
    meta = DiagramMeta(backend="gudhi")
    assert DiagramMeta.__hash__ is None
    assert not isinstance(meta, cabc.Hashable)
    with pytest.raises(TypeError, match="DiagramMeta"):
        hash(meta)


def test_metadata_still_compares_equal_through_the_mapping_proxy() -> None:
    """Freezing the mappings must not disturb `same_provenance` (§8)."""
    left = bars([0], [0.0], [1.0], meta=DiagramMeta(params={"max_edge_length": 4.0}))
    right = bars([0], [0.0], [1.0], meta=DiagramMeta(params={"max_edge_length": 4.0}))
    assert left.same_provenance(right)
    assert left.meta.params == {"max_edge_length": 4.0}


def test_caller_dict_cannot_be_written_through_after_construction() -> None:
    supplied: dict[str, object] = {"order": "backend"}
    meta = DiagramMeta(provenance=supplied)
    supplied["order"] = "canonical"
    assert meta.provenance["order"] == "backend"
    with pytest.raises(TypeError):
        meta.provenance["order"] = "canonical"  # type: ignore[index]


# -- §3.1 validator messages carry the offending value --------------------


def test_invariant_errors_name_the_value_that_broke_them() -> None:
    """§3.1 anticipates 1e-16 I6 violations, where the magnitude is the point."""
    with pytest.raises(ValueError, match=r"I6.*1 of 2 rows.*worst by -"):
        bars([0, 0], [0.0, 1.0], [1.0, 0.5])
    with pytest.raises(ValueError, match=r"I1.*got 1, 2, 2"):
        PersistenceDiagram(
            dims=np.asarray([0], dtype=np.int32),
            births=np.asarray([0.0, 1.0], dtype=np.float64),
            deaths=np.asarray([1.0, 2.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match=r"I2.*got float32"):
        PersistenceDiagram(
            dims=np.asarray([0], dtype=np.int32),
            births=np.asarray([0.0], dtype=np.float32),
            deaths=np.asarray([1.0], dtype=np.float64),
        )
