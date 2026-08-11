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
import json
import math
import sys
import types
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from akriti.diagrams import DiagramBatch, DiagramMeta, PersistenceDiagram, core
from akriti.diagrams.core import namespace_of


@pytest.fixture(autouse=True)
def clear_namespace_cache() -> None:
    core._clear_namespace_cache()
    yield
    core._clear_namespace_cache()


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


@pytest.mark.parametrize(
    "value",
    [
        "banana",
        "finitized_at:1.0",
        "Faithful",
        "finitized",
        "",
        0,
        None,
    ],
)
def test_essential_bars_rejects_anything_outside_the_closed_vocabulary(
    value: object,
) -> None:
    """§8 closes `essential_bars` to four values, validated at construction.

    `"finitized_at:1.0"` is in the list deliberately: it is the spelling this
    key used before §8 moved the substituted death into its own numeric slot,
    so a diagram carrying it is one written by superseded code rather than a
    typo, and it must fail as loudly as `"banana"`. `"Faithful"` covers the
    case that the vocabulary is exact rather than case-insensitive, and `None`
    the case that an explicit null is not the same as an absent key.
    """
    with pytest.raises(ValueError, match="essential_bars"):
        DiagramMeta(provenance={"essential_bars": value})


def test_finitized_at_without_the_substituted_death_is_rejected() -> None:
    """§8's second iff, forward: the state without the value it qualifies."""
    with pytest.raises(ValueError, match="missing"):
        DiagramMeta(provenance={"essential_bars": "finitized_at"})


def test_the_substituted_death_without_finitized_at_is_rejected() -> None:
    """§8's second iff, backward -- and the failure that motivates it.

    A qualifier outliving the state it described is how a diagram comes to
    assert a substituted death of 1.0 while `essential_bars` says the essential
    bars were dropped. Both keys are written by `finitize`, so the only way
    this pairing arrives is a writer that updated one and forgot the other.
    """
    with pytest.raises(ValueError, match="essential_bars_finitized_at"):
        DiagramMeta(
            provenance={
                "essential_bars": "finitized_dropped",
                "essential_bars_dropped": 1,
                "essential_bars_finitized_at": 1.0,
            }
        )


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_a_non_finite_substituted_death_is_rejected(value: float) -> None:
    """§8: the key holds the *finite* death that replaced `inf`.

    `inf` is the exact thing finitizing removes, so recording it as the
    substitute asserts that the operation did nothing while claiming it did.
    `NaN` is not a death time under any reading (I4, I5).

    **Asserted as an outcome rather than against a particular check**, and
    `TypeError` is what arrives: §10.2 keeps non-finite floats out of
    `provenance` altogether and runs first, so §8's finiteness requirement is
    already carried by a stronger rule one layer up. The test is here because
    §8 states the requirement and it should be seen to hold, not because
    `_validate_provenance` is the layer that has to hold it.
    """
    with pytest.raises(TypeError, match="essential_bars_finitized_at"):
        DiagramMeta(
            provenance={
                "essential_bars": "finitized_at",
                "essential_bars_finitized_at": value,
            }
        )


@pytest.mark.parametrize("value", ["1.0", True, None, [1.0]])
def test_a_non_numeric_substituted_death_is_rejected(value: object) -> None:
    """§8 types it as the numeric death, not as a rendering of one.

    `"1.0"` is the shape the superseded string spelling would leave behind if
    a writer split it rather than replacing it, and `True` is refused on the
    ground `coeff_field` refuses it: it is an `int` by an accident of the
    language, and nobody means a death time of one by writing it.
    """
    with pytest.raises(TypeError, match="essential_bars_finitized_at"):
        DiagramMeta(
            provenance={
                "essential_bars": "finitized_at",
                "essential_bars_finitized_at": value,
            }
        )


def test_a_coherent_finitized_at_record_is_accepted() -> None:
    """The check must admit exactly what `finitize` writes (§5, §8)."""
    meta = DiagramMeta(
        provenance={
            "essential_bars": "finitized_at",
            "essential_bars_finitized_at": 2.5,
            "essential_bars_source": "faithful",
        }
    )

    assert meta.provenance["essential_bars_finitized_at"] == 2.5


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
    meta = DiagramMeta(
        provenance={"anything": {"nested": [True, None, 3]}, "source_dtype": "float32"}
    )
    assert "anything" in meta.provenance


@pytest.mark.parametrize("field", ["params", "provenance"])
def test_metadata_accepts_json_safe_nested_values_and_freezes_the_mapping(
    field: str,
) -> None:
    """§8: metadata values are JSON-safe, while the two fields are read-only."""
    value = {
        "boolean": True,
        "null": None,
        "string": "ok",
        "integer": 7,
        "finite_float": 1.25,
        "list": [False, "item", 2],
        "mapping": {"child": "value"},
    }
    meta = DiagramMeta(**{field: {"nested": value}})  # type: ignore[arg-type]

    assert meta.__dict__[field]["nested"] == value
    with pytest.raises(TypeError):
        meta.__dict__[field]["new"] = "rejected"


@pytest.mark.parametrize("field", ["params", "provenance"])
def test_metadata_owns_nested_json_values_and_keeps_them_json_safe(field: str) -> None:
    """§8/§10.1: construction owns the complete metadata value tree.

    A shallow copy of the outer mapping is not enough: callers can retain a
    nested list or mapping and otherwise change the provenance of an already
    constructed diagram after validation.
    """
    supplied = {"nested": {"items": ["before"], "child": {"answer": 42}}}
    meta = DiagramMeta(**{field: supplied})  # type: ignore[arg-type]

    supplied["nested"]["items"].append("after")  # type: ignore[index]
    supplied["nested"]["child"]["answer"] = 99  # type: ignore[index]

    expected = {"nested": {"items": ["before"], "child": {"answer": 42}}}
    stored = getattr(meta, field)
    assert stored == expected
    assert json.loads(json.dumps(stored)) == expected


@pytest.mark.parametrize("field", ["params", "provenance"])
def test_metadata_nested_json_values_are_read_only(field: str) -> None:
    """§8: read-only metadata includes lists and mappings below the root."""
    meta = DiagramMeta(
        **{
            field: {
                "nested": {
                    "items": ["value"],
                    "child": {"answer": 42},
                }
            }
        }
    )  # type: ignore[arg-type]

    stored = getattr(meta, field)
    with pytest.raises(TypeError):
        stored["nested"]["items"].append("rejected")
    with pytest.raises(TypeError):
        stored["nested"]["child"]["answer"] = 0


def test_metadata_nested_values_preserve_json_list_and_mapping_semantics() -> None:
    """§10.1: freezing does not change the normalized JSON value types."""
    expected = {
        "nested": {
            "items": [False, None, "text", 3, 1.5],
            "child": {"answer": 42},
        }
    }
    meta = DiagramMeta(params=expected)

    normalized = json.loads(json.dumps(meta.params))
    assert normalized == expected
    assert type(normalized["nested"]["items"]) is list
    assert type(normalized["nested"]["child"]) is dict


def test_metadata_internal_frozen_values_can_be_reconstructed() -> None:
    """§5/§8: metadata transformations accept their own frozen containers."""
    meta = DiagramMeta(
        params={"nested": {"items": [1, 2]}},
        provenance={"nested": {"child": {"answer": 42}}},
    )

    copied = replace(meta, description="copied")
    assert copied.params == meta.params
    assert copied.provenance == meta.provenance


def test_same_provenance_compares_equivalent_nested_json_metadata() -> None:
    """§8: provenance equality is structural, including nested JSON values."""
    left = bars(
        [0],
        [0.0],
        [1.0],
        meta=DiagramMeta(
            params={"nested": {"items": [1, {"answer": True}]}},
            provenance={"source": {"steps": ["import", "validate"]}},
        ),
    )
    right = bars(
        [0],
        [0.0],
        [1.0],
        meta=DiagramMeta(
            params={"nested": {"items": [1, {"answer": True}]}},
            provenance={"source": {"steps": ["import", "validate"]}},
        ),
    )

    assert left.same_provenance(right)


@pytest.mark.parametrize("field", ["params", "provenance"])
@pytest.mark.parametrize("nested", [False, True])
def test_metadata_rejects_non_string_keys_with_a_field_path(
    field: str, nested: bool
) -> None:
    """§8: every metadata mapping key, including nested keys, is a string."""
    payload: object = {1: "bad"}
    if nested:
        payload = {"outer": payload}

    with pytest.raises(TypeError, match=rf"{field}.*(?:outer.*)?1"):
        DiagramMeta(**{field: payload})  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["params", "provenance"])
@pytest.mark.parametrize(
    ("bad_value", "path_fragment"),
    [
        (object(), "bad"),
        (Path("not-json"), "bad"),
        (np.int64(1), "bad"),
        (np.float64(1.25), "bad"),
        (np.bool_(True), "bad"),
        (float("nan"), "bad"),
        (float("inf"), "bad"),
        (float("-inf"), "bad"),
        ((1, 2), "bad"),
        ({1, 2}, "bad"),
    ],
)
@pytest.mark.parametrize("nested", [False, True])
def test_metadata_rejects_non_json_values_with_a_field_path(
    field: str, bad_value: object, path_fragment: str, nested: bool
) -> None:
    """§8: unsupported values fail at construction with their metadata path."""
    payload: object = {path_fragment: bad_value}
    if nested:
        payload = {"outer": payload}

    expected_path = rf"{field}.*(?:outer.*)?{path_fragment}"
    with pytest.raises(TypeError, match=expected_path):
        DiagramMeta(**{field: payload})  # type: ignore[arg-type]


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
    assert substituted.meta.provenance["essential_bars"] == "finitized_at"
    assert substituted.meta.provenance["essential_bars_finitized_at"] == 2.0
    assert "essential_bars_dropped" not in substituted.meta.provenance


def test_substitute_then_drop_clears_the_stale_substituted_death() -> None:
    """§8, the mirror of the drop-then-substitute case above.

    A diagram finitized by substitution, then reloaded alongside a fresh
    essential bar and dropped. `essential_bars_finitized_at` described a death
    that is no longer in `deaths`, and `finitize` MUST clear it rather than
    leave a diagram asserting both a cardinality change and a substitution.

    This direction is the one that was missing: the substitution branch
    already cleared the drop count, so only the branch added alongside the new
    key had never been exercised.
    """
    substituted = bars([0, 0], [0.0, 0.25], [np.inf, 0.75]).finitize(at=2.0)
    assert substituted.meta.provenance["essential_bars_finitized_at"] == 2.0

    reloaded = bars(
        [0, 0],
        [0.0, 0.25],
        [np.inf, 0.75],
        meta=DiagramMeta(provenance=dict(substituted.meta.provenance)),
    )
    dropped = reloaded.finitize(at="drop")

    assert dropped.meta.provenance["essential_bars"] == "finitized_dropped"
    assert dropped.meta.provenance["essential_bars_dropped"] == 1
    assert "essential_bars_finitized_at" not in dropped.meta.provenance


def test_finitizing_twice_overwrites_the_substituted_death() -> None:
    """§8: the recorded death is the one in `deaths`, not the first one.

    Repeated finitization is not an error and does not accumulate. The second
    call sees a diagram whose essential bar is fresh (this rebuilds one, since
    the first call left none) and MUST record its own substitution, because a
    reader takes the key as describing the diagram in front of them.
    """
    once = bars([0], [0.0], [np.inf]).finitize(at=1.0)
    assert once.meta.provenance["essential_bars_finitized_at"] == 1.0

    again = bars(
        [0],
        [0.0],
        [np.inf],
        meta=DiagramMeta(provenance=dict(once.meta.provenance)),
    ).finitize(at=3.0)

    assert again.meta.provenance["essential_bars"] == "finitized_at"
    assert again.meta.provenance["essential_bars_finitized_at"] == 3.0


def test_finitize_records_the_computed_death_in_max_finite_death_mode() -> None:
    """§8: "whichever mode computed it" -- the key is not substitute-only.

    `at="max_finite_death"` derives the value rather than taking it, and a
    writer that recorded only the explicitly-passed case would leave this mode
    writing `"finitized_at"` with no qualifier, which §8 rejects outright.
    """
    finitized = bars([0, 0], [0.0, 0.25], [np.inf, 0.75]).finitize(
        at="max_finite_death"
    )

    assert finitized.meta.provenance["essential_bars"] == "finitized_at"
    assert finitized.meta.provenance["essential_bars_finitized_at"] == 0.75


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
    assert finitized.meta.provenance["essential_bars"] == "finitized_at"
    assert finitized.meta.provenance["essential_bars_finitized_at"] == 1.0
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


def test_finitize_low_explicit_value_names_replacement_and_birth() -> None:
    """§5: a caller-supplied replacement below an essential birth is invalid.

    The error must identify the replacement the caller supplied and the
    offending (maximum) essential birth, rather than leaking the generic I6
    constructor message about deaths and births.
    """
    diagram = bars([0, 0], [0.0, 2.0], [0.5, np.inf])
    with pytest.raises(ValueError, match=r"at=0\.5.*essential birth.*2\.0"):
        diagram.finitize(at=0.5)


def test_finitize_low_computed_value_names_max_and_birth() -> None:
    """§5: the computed maximum and offending essential birth are reported."""
    diagram = bars([0, 0], [0.0, 2.0], [0.5, np.inf])
    with pytest.raises(
        ValueError, match=r"max_finite_death.*0\.5.*essential birth.*2\.0"
    ):
        diagram.finitize(at="max_finite_death")


def test_max_finite_death_on_all_essential_bars_names_missing_finite_death() -> None:
    """§5: the mode has no finite death from which to compute a maximum."""
    diagram = bars([0, 1], [0.0, 2.0], [np.inf, np.inf])
    with pytest.raises(ValueError, match=r"max_finite_death.*finite death"):
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


# -- §3.3 namespace resolution and public-construction ownership ----------


def test_namespace_of_prefers_native_method_when_compat_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = object()

    class NativeArray:
        def __array_namespace__(self) -> object:
            return native

    compat = types.SimpleNamespace(array_namespace=lambda _: object())
    monkeypatch.setitem(sys.modules, "array_api_compat", compat)

    assert namespace_of(NativeArray()) is native


def test_namespace_of_dispatches_fallback_to_compat_with_original_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supplied = object()
    resolved = object()
    seen: list[object] = []

    def array_namespace(value: object) -> object:
        seen.append(value)
        return resolved

    compat = types.SimpleNamespace(array_namespace=array_namespace)
    monkeypatch.setitem(sys.modules, "array_api_compat", compat)
    monkeypatch.setattr(core.metadata, "version", lambda _: "1.15.0")

    assert namespace_of(supplied) is resolved
    assert seen == [supplied]


@pytest.mark.parametrize(
    "version",
    [
        "1.15",
        "1.15.0.post1",
        "1.15.0.post1.dev1",
        "1.15.0+local",
        "1.15.1rc1",
        "1!1.0",
    ],
)
def test_namespace_of_accepts_valid_versions_at_or_above_floor(
    monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    resolved = object()
    compat = types.SimpleNamespace(array_namespace=lambda _: resolved)
    monkeypatch.setitem(sys.modules, "array_api_compat", compat)
    monkeypatch.setattr(core.metadata, "version", lambda _: version)

    assert namespace_of(object()) is resolved


@pytest.mark.parametrize(
    "version", ["1.14.99", "1.15.0rc1", "1.15.0.0rc1", "1.15.0.dev1"]
)
def test_namespace_of_rejects_versions_below_or_prerelease_at_floor(
    monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    compat = types.SimpleNamespace(array_namespace=lambda _: object())
    monkeypatch.setitem(sys.modules, "array_api_compat", compat)
    monkeypatch.setattr(core.metadata, "version", lambda _: version)

    with pytest.raises(
        ImportError, match=r"array-api-compat.*1\.15\.0.*akriti\[torch\]"
    ):
        namespace_of(object())


def test_namespace_of_rejects_unparseable_fallback_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compat = types.SimpleNamespace(array_namespace=lambda _: object())
    monkeypatch.setitem(sys.modules, "array_api_compat", compat)
    monkeypatch.setattr(core.metadata, "version", lambda _: "development")

    with pytest.raises(ImportError, match=r"could not parse.*array-api-compat"):
        namespace_of(object())


def test_namespace_of_fallback_missing_dependency_names_torch_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def blocked_import(name: str) -> object:
        raise ModuleNotFoundError("blocked for contract test", name=name)

    monkeypatch.setattr(core, "import_module", blocked_import)
    with pytest.raises(ImportError, match=r"akriti\[torch\]"):
        namespace_of(object())


def test_namespace_of_propagates_transitive_module_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def blocked_import(name: str) -> object:
        raise ModuleNotFoundError("transitive dependency missing", name="other")

    monkeypatch.setattr(core, "import_module", blocked_import)
    with pytest.raises(ModuleNotFoundError, match="transitive dependency"):
        namespace_of(object())


def test_namespace_of_validates_fallback_version_only_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    resolved = object()

    def version(_: str) -> str:
        nonlocal calls
        calls += 1
        return "1.15.0"

    compat = types.SimpleNamespace(array_namespace=lambda _: resolved)
    monkeypatch.setitem(sys.modules, "array_api_compat", compat)
    monkeypatch.setattr(core.metadata, "version", version)

    assert namespace_of(object()) is resolved
    assert namespace_of(object()) is resolved
    assert calls == 1


def test_from_diagrams_uses_owned_concat_buffers_without_second_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagram = bars([0], [0.0], [1.0])

    def fail_copy(*args: object, **kwargs: object) -> object:
        raise AssertionError("owned concat buffers must not be copied again")

    monkeypatch.setattr(core, "_copy_array", fail_copy)
    batch = DiagramBatch.from_diagrams([diagram])
    assert batch[0] == diagram


def test_persistence_diagram_copies_public_coordinate_arrays() -> None:
    dims = np.asarray([0], dtype=np.int32)
    births = np.asarray([0.0], dtype=np.float64)
    deaths = np.asarray([1.0], dtype=np.float64)
    diagram = PersistenceDiagram(dims=dims, births=births, deaths=deaths)

    dims[0] = 7
    births[0] = 4.0
    deaths[0] = 9.0

    assert np.asarray(diagram.dims).tolist() == [0]
    assert np.asarray(diagram.births).tolist() == [0.0]
    assert np.asarray(diagram.deaths).tolist() == [1.0]


def test_diagram_batch_copies_public_buffers_and_metas_sequence() -> None:
    dims = np.asarray([0], dtype=np.int32)
    births = np.asarray([0.0], dtype=np.float64)
    deaths = np.asarray([1.0], dtype=np.float64)
    offsets = np.asarray([0, 1], dtype=np.int64)
    metas = [DiagramMeta(description="before")]
    batch = DiagramBatch(
        dims=dims, births=births, deaths=deaths, offsets=offsets, metas=metas
    )

    dims[0] = 7
    births[0] = 4.0
    deaths[0] = 9.0
    offsets[1] = 0
    metas.append(DiagramMeta(description="after"))

    assert np.asarray(batch.dims).tolist() == [0]
    assert np.asarray(batch.births).tolist() == [0.0]
    assert np.asarray(batch.deaths).tolist() == [1.0]
    assert np.asarray(batch.offsets).tolist() == [0, 1]
    assert len(batch.metas) == 1
    assert batch.metas[0].description == "before"


def test_diagram_batch_getitem_retains_deliberate_zero_copy_views() -> None:
    batch = DiagramBatch.from_diagrams([bars([0, 1], [0.0, 0.5], [1.0, 2.0])])
    view = batch[0]

    assert np.shares_memory(np.asarray(batch.dims), np.asarray(view.dims))
    assert np.shares_memory(np.asarray(batch.births), np.asarray(view.births))
    assert np.shares_memory(np.asarray(batch.deaths), np.asarray(view.deaths))


def test_diagram_meta_uses_description_and_same_provenance_ignores_it() -> None:
    left = DiagramMeta(description="left", backend="array")
    right = DiagramMeta(description="right", backend="array")
    assert left.description == "left"
    assert left != right

    one = bars([0], [0.0], [1.0], meta=left)
    two = bars([0], [0.0], [1.0], meta=right)
    assert one.same_provenance(two)


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("filtration", "alpha"),
        ("backend", "ripser"),
        ("backend_version", "0.7.0"),
        ("coeff_field", 11),
        ("params", {"max_dimension": 3}),
        ("provenance", {"coeff_field_source": "caller", "source_dtype": "float32"}),
    ],
)
def test_same_provenance_compares_every_non_description_field(
    field: str, changed: object
) -> None:
    common = {
        "filtration": "rips",
        "backend": "array",
        "backend_version": "1.0.0",
        "coeff_field": 2,
        "params": {"max_dimension": 2},
        "provenance": {"coeff_field_source": "caller", "source_dtype": "float64"},
    }
    left_meta = DiagramMeta(**common)
    right_values = {**common, field: changed}
    right_meta = DiagramMeta(**right_values)
    left = bars([0], [0.0], [1.0], meta=left_meta)
    right = bars([0], [0.0], [1.0], meta=right_meta)

    assert not left.same_provenance(right)


def test_batch_same_provenance_is_type_checked_and_order_sensitive() -> None:
    first = bars([0], [0.0], [1.0], meta=DiagramMeta(backend="first"))
    second = bars([0], [0.0], [1.0], meta=DiagramMeta(backend="second"))
    left = DiagramBatch.from_diagrams([first, second])
    reversed_batch = DiagramBatch.from_diagrams([second, first])

    assert not left.same_provenance(reversed_batch)
    with pytest.raises(TypeError, match="DiagramBatch"):
        left.same_provenance(object())  # type: ignore[arg-type]


@pytest.mark.parametrize("source", ["caller", "backend_default"])
def test_coeff_field_source_requires_a_coefficient_field(source: str) -> None:
    with pytest.raises(ValueError, match="coeff_field_source"):
        DiagramMeta(provenance={"coeff_field_source": source})


def test_coeff_field_source_none_is_not_treated_as_absent() -> None:
    with pytest.raises(ValueError, match="coeff_field_source"):
        DiagramMeta(provenance={"coeff_field_source": None})


def test_coeff_field_source_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="coeff_field_source"):
        DiagramMeta(coeff_field=2, provenance={"coeff_field_source": "guess"})


@pytest.mark.parametrize(
    "field", ["filtration", "backend", "backend_version", "description"]
)
@pytest.mark.parametrize("value", [3.5, object(), b"rips", ["rips"], 0])
def test_metadata_string_fields_reject_a_non_string(field: str, value: Any) -> None:
    """§8 types four fields `str | None`, and `__post_init__` must enforce it.

    `params` and `provenance` have been checked since D17; these four never
    were, so `DiagramMeta(filtration=3.5)` constructed and every `from_*`
    adapter passed `**meta` straight into it. Two consequences, and the second
    is the one that bites: the type does not describe its own contents, and
    §10.2 stores `meta.json` as UTF-8 JSON, so `filtration=object()` is a
    diagram that satisfies §3.1 and §8 completely and cannot be saved -- the
    failure `_require_json_representable` already prevents one field over,
    surfacing arbitrarily far from the adapter that wrote it.

    `bytes` is listed among the refusals deliberately: it is the near miss,
    since `str(b"rips")` is `"b'rips'"` and `json.dumps` refuses it outright.
    """
    with pytest.raises(TypeError, match=field):
        DiagramMeta(**{field: value})


@pytest.mark.parametrize(
    "field", ["filtration", "backend", "backend_version", "description"]
)
def test_metadata_string_fields_accept_a_string_or_none(field: str) -> None:
    """The refusal above must not reach past its target: §8's own concession
    is that every field is optional, so `None` stays legal on all four."""
    assert getattr(DiagramMeta(**{field: "rips"}), field) == "rips"
    assert getattr(DiagramMeta(**{field: None}), field) is None


@pytest.mark.parametrize("value", [2.0, "2", True, False, np.int64(2), object()])
def test_metadata_coeff_field_rejects_a_non_int(value: Any) -> None:
    """§8 types `coeff_field: int | None` -- the characteristic of the field
    homology was computed over (§9.3).

    `bool` is refused for the reason `adapters._as_degree` gives one field
    over: `coeff_field=True` would record the field of one element, which is
    a coincidence of Python rather than anything a caller means. A NumPy
    scalar is refused on `_require_json_representable`'s existing house rule
    -- exact builtin types, converted at the call site -- which is what keeps
    §10.2's JSON round trip total. Adapters convert before they reach here
    (`_require_coeff_field` admits any `numbers.Integral`), so the widening
    lives at the adapter boundary and the stored value is always a builtin.
    """
    with pytest.raises(TypeError, match="coeff_field"):
        DiagramMeta(coeff_field=value)


def test_metadata_coeff_field_accepts_an_int_or_none() -> None:
    assert DiagramMeta(coeff_field=11).coeff_field == 11
    assert DiagramMeta(coeff_field=None).coeff_field is None


def test_same_provenance_rejects_non_diagram() -> None:
    diagram = bars([0], [0.0], [1.0])
    with pytest.raises(TypeError, match="PersistenceDiagram"):
        diagram.same_provenance(object())  # type: ignore[arg-type]


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
