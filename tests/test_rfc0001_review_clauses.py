"""Tests for the normative clauses RFC-0001 gained in the review pass.

Scope: the changes branch ``rfc/0001/review`` made to
``rfcs/0001-persistence-diagram-interchange.md`` -- Appendix D entries 68-76,
which carried the document from 1.0.0 to 1.1.0.

These tests were written from the specification alone. No file under ``src/``
was read while writing them: every assertion below is derived from a quoted
clause, and each test names the Appendix C clause id and section it binds. A
failure here therefore means one of two things -- the implementation does not
satisfy the clause, or the clause and the implementation disagree about a name.
Both are findings; neither is a test that needs relaxing to match the code.

Clause ids (``N3.1-14`` and so on) are Appendix C's, which is generated from the
body by ``tools/normative_index.py``. Where a clause carries no id -- Appendix C
indexes only sections 1 through 11 -- the section is cited instead.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import struct
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from akriti.diagrams import (
    DiagramBatch,
    DiagramMeta,
    PersistenceDiagram,
    load,
    save,
)

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

RFC_PATH = (
    Path(__file__).resolve().parents[1]
    / "rfcs"
    / ("0001-persistence-diagram-interchange.md")
)

#: The document version the review pass landed (header row, and the value
#: §10.2 requires ``save`` to write as ``spec_version``).
SPEC_VERSION = "1.2.0"


def diagram(
    dims: list[int],
    births: list[float],
    deaths: list[float],
    meta: DiagramMeta | None = None,
) -> PersistenceDiagram:
    """A ``PersistenceDiagram`` over NumPy, at the dtypes §3 requires."""
    kwargs: dict[str, Any] = {
        "dims": np.asarray(dims, dtype=np.int32),
        "births": np.asarray(births, dtype=np.float64),
        "deaths": np.asarray(deaths, dtype=np.float64),
    }
    if meta is not None:
        kwargs["meta"] = meta
    return PersistenceDiagram(**kwargs)


def sample() -> PersistenceDiagram:
    """Two essential bars and three finite ones, over two degrees."""
    return diagram(
        dims=[0, 0, 0, 1, 1],
        births=[0.0, 0.0, 0.0, 1.0, 1.5],
        deaths=[1.0, math.inf, 2.0, math.inf, 2.5],
    )


def batch_of(*diagrams: PersistenceDiagram) -> DiagramBatch:
    return DiagramBatch.from_diagrams(list(diagrams))


def spec_content_hash(dims: list[int], births: list[float], deaths: list[float]) -> str:
    """§8.1's hashed message, transcribed from the document.

    The message is the tag, the bar count as eight big-endian bytes, then the
    three columns in canonical order (§7: ``(dim, birth, death)`` ascending,
    stable), ``>i`` for ``dims`` and ``>d`` for the coordinates. Negative zero
    is normalised to ``+0.0`` before either byte path, which adding ``0.0``
    does exactly for every value I4 and I5 admit.
    """
    rows = sorted(zip(dims, births, deaths, strict=True))
    message = b"akriti.PersistenceDiagram.v1\x00" + len(rows).to_bytes(8, "big")
    message += b"".join(struct.pack(">i", int(row[0])) for row in rows)
    message += b"".join(struct.pack(">d", float(row[1]) + 0.0) for row in rows)
    message += b"".join(struct.pack(">d", float(row[2]) + 0.0) for row in rows)
    return hashlib.sha256(message).hexdigest()


def read_meta_json(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read("meta.json").decode("utf-8"))


def rewrite_meta_json(src: Path, dst: Path, meta: dict[str, Any]) -> None:
    """Rewrite an ``.akd`` with a patched ``meta.json``, member order kept.

    §10.2: an archive "MUST contain exactly these two named entries, under
    exactly these names, written in this order".
    """
    with zipfile.ZipFile(src) as archive:
        bars = archive.read("bars.npz")
    with zipfile.ZipFile(dst, "w") as archive:
        archive.writestr("meta.json", json.dumps(meta).encode("utf-8"))
        archive.writestr("bars.npz", bars)


# --------------------------------------------------------------------------
# §2 -- the diagonal clause, rewritten so that trivial bars are ordinary bars
#
# Entry 68: "§2's diagonal clause forbade the trivial bars §4, §11.1 and §11.2
# all depend on; it now names the diagonal-as-a-multiset."
#
#   "A bar with `birth == death` is **trivial**: it has zero persistence."
#   "That is a rule about the diagonal, not about the bars that lie on it. **A
#    trivial bar is an ordinary bar** -- stored, counted, hashed and
#    round-tripped like any other."
# --------------------------------------------------------------------------


def test_s2_trivial_bar_constructs() -> None:
    """A bar with ``birth == death`` satisfies I6 and is admitted."""
    d = diagram(dims=[0], births=[1.0], deaths=[1.0])
    assert d.n_bars == 1


def test_s2_trivial_bar_is_counted() -> None:
    """ "stored, counted" -- it is not filtered out on the way in."""
    d = diagram(dims=[0, 0], births=[0.0, 1.0], deaths=[1.0, 1.0])
    assert d.n_bars == 2


def test_s2_trivial_bar_has_zero_persistence() -> None:
    """ "it has zero persistence"."""
    d = diagram(dims=[0], births=[1.5], deaths=[1.5])
    assert float(np.asarray(d.persistence)[0]) == 0.0


def test_s2_trivial_bar_is_not_essential() -> None:
    """§2 defines essential as ``death == +inf``; trivial is a separate case."""
    d = diagram(dims=[0], births=[1.0], deaths=[1.0])
    assert not bool(np.asarray(d.essential)[0])
    assert d.finite.n_bars == 1


def test_s2_trivial_bars_are_a_multiset_not_a_set() -> None:
    """ "two bars with identical coordinates are two bars".

    The deduplicating implementation §2 calls wrong is most tempting exactly
    here, where both rows sit on the diagonal.
    """
    d = diagram(dims=[0, 0], births=[1.0, 1.0], deaths=[1.0, 1.0])
    assert d.n_bars == 2
    assert d.canonical().n_bars == 2


def test_s2_trivial_bar_is_hashed(tmp_path: Path) -> None:
    """ "hashed" -- it contributes to ``content_hash`` like any other bar."""
    one = diagram(dims=[0], births=[1.0], deaths=[1.0])
    two = diagram(dims=[0, 0], births=[1.0, 1.0], deaths=[1.0, 1.0])
    assert one.content_hash != two.content_hash
    assert two.content_hash == spec_content_hash([0, 0], [1.0, 1.0], [1.0, 1.0])


def test_s2_trivial_bar_round_trips(tmp_path: Path) -> None:
    """ "round-tripped like any other"."""
    d = diagram(dims=[0, 1], births=[0.0, 2.0], deaths=[0.0, 2.0])
    path = tmp_path / "trivial.akd"
    save(d, path)
    back = load(path)
    assert isinstance(back, PersistenceDiagram)
    assert back.n_bars == 2
    assert back.content_hash == d.content_hash


# --------------------------------------------------------------------------
# §3.1 -- N3.1-6: ``frozen=True, eq=False`` on both, both unhashable
#
#   "It MUST be `frozen=True, eq=False` on both, and both MUST be unhashable.
#    The two generated methods `frozen=True` brings with it are each wrong
#    here."
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [PersistenceDiagram, DiagramBatch])
def test_n3_1_6_dataclass_params(cls: type) -> None:
    params = cls.__dataclass_params__  # type: ignore[attr-defined]
    assert params.frozen is True, f"{cls.__name__} must be frozen=True"
    assert params.eq is False, f"{cls.__name__} must be eq=False"


@pytest.mark.parametrize("cls", [PersistenceDiagram, DiagramBatch])
def test_n3_1_6_unhashable(cls: type) -> None:
    """ "both MUST be unhashable"."""
    assert cls.__hash__ is None
    obj = sample() if cls is PersistenceDiagram else batch_of(sample())
    with pytest.raises(TypeError):
        hash(obj)


def test_n3_1_6_frozen_refuses_field_assignment() -> None:
    d = sample()
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.births = np.zeros(5)  # type: ignore[misc]


def test_s8_diagram_meta_is_frozen() -> None:
    """§8: "`DiagramMeta` already is" frozen, which N3.1-5 leans on."""
    assert DiagramMeta.__dataclass_params__.frozen is True  # type: ignore[attr-defined]
    with pytest.raises(dataclasses.FrozenInstanceError):
        DiagramMeta().filtration = "rips"  # type: ignore[misc]


# --------------------------------------------------------------------------
# §3.1 -- N3.1-8, N3.1-10: the copy rule, and what it now binds
#
#   N3.1-8: "Every public construction path -- the `PersistenceDiagram`
#    constructor, every `from_*` adapter, and `DiagramBatch.from_diagrams` --
#    MUST therefore copy any array it is given that the caller could still
#    write to, rather than store it."
#   N3.1-10: "`DiagramBatch` MUST copy the `metas` sequence on the same rule".
# --------------------------------------------------------------------------


def test_n3_1_8_constructor_copies_caller_arrays() -> None:
    births = np.asarray([0.0, 0.0], dtype=np.float64)
    d = PersistenceDiagram(
        dims=np.asarray([0, 0], dtype=np.int32),
        births=births,
        deaths=np.asarray([1.0, 2.0], dtype=np.float64),
    )
    births[0] = 99.0
    assert float(np.asarray(d.births)[0]) == 0.0


def test_n3_1_8_from_diagrams_copies_caller_arrays() -> None:
    dims = np.asarray([0], dtype=np.int32)
    births = np.asarray([0.0], dtype=np.float64)
    deaths = np.asarray([1.0], dtype=np.float64)
    d = PersistenceDiagram(dims=dims, births=births, deaths=deaths)
    b = DiagramBatch.from_diagrams([d])
    births[0] = 99.0
    assert float(np.asarray(b.births)[0]) == 0.0


def test_n3_1_10_batch_copies_the_metas_sequence() -> None:
    """A caller who appends to the list they passed must not change ``len(b)``.

    B1 is stated over the length of ``metas``, so the aliasing failure this
    rules out breaks an invariant *after* the construction that enforced it.
    """
    metas = [DiagramMeta(description="first"), DiagramMeta(description="second")]
    b = DiagramBatch(
        dims=np.asarray([0, 0, 1], dtype=np.int32),
        births=np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
        deaths=np.asarray([1.0, 1.5, 2.0], dtype=np.float64),
        offsets=np.asarray([0, 2, 3], dtype=np.int64),
        metas=metas,
    )
    assert len(b) == 2
    metas.append(DiagramMeta(description="appended"))
    assert len(b) == 2
    assert b[1].meta.description == "second"


# --------------------------------------------------------------------------
# §3.1 -- N3.1-11, N3.1-12: I8's copy rule reaches ``params`` and ``provenance``
#
# Entry 72: "I8's copy rule binds `params` and `provenance`, not only the
# arrays and `metas`."
#
#   N3.1-12: "Unlike the array case this is enforceable -- the mappings hold
#    JSON-representable values (§8), which are copyable -- so `DiagramMeta`
#    MUST copy them, recursively over nested lists and mappings, and MUST
#    expose them read-only."
# --------------------------------------------------------------------------


def test_n3_1_12_params_copied_at_the_top_level() -> None:
    params: dict[str, Any] = {"maxdim": 1}
    m = DiagramMeta(params=params)
    params["maxdim"] = 99
    assert m.params["maxdim"] == 1


def test_n3_1_12_provenance_copied_at_the_top_level() -> None:
    provenance: dict[str, Any] = {"essential_bars": "faithful"}
    m = DiagramMeta(provenance=provenance)
    provenance["essential_bars"] = "lost_upstream"
    assert m.provenance["essential_bars"] == "faithful"


def test_n3_1_12_copy_recurses_into_nested_mappings() -> None:
    """ "recursively over nested lists and mappings"."""
    params: dict[str, Any] = {"nested": {"k": 1}}
    m = DiagramMeta(params=params)
    params["nested"]["k"] = 99
    assert m.params["nested"]["k"] == 1


def test_n3_1_12_copy_recurses_into_nested_lists() -> None:
    params: dict[str, Any] = {"dims": [0, 1]}
    m = DiagramMeta(params=params)
    params["dims"].append(2)
    assert list(m.params["dims"]) == [0, 1]


def test_n3_1_12_copy_recurses_into_a_list_of_mappings() -> None:
    params: dict[str, Any] = {"steps": [{"name": "rips"}]}
    m = DiagramMeta(params=params)
    params["steps"][0]["name"] = "alpha"
    assert m.params["steps"][0]["name"] == "rips"


def test_n3_1_12_params_exposed_read_only() -> None:
    """ "and MUST expose them read-only"."""
    m = DiagramMeta(params={"maxdim": 1})
    with pytest.raises(TypeError):
        m.params["maxdim"] = 2  # type: ignore[index]


def test_n3_1_12_provenance_exposed_read_only() -> None:
    m = DiagramMeta(provenance={"essential_bars": "faithful"})
    with pytest.raises(TypeError):
        m.provenance["essential_bars"] = "lost_upstream"  # type: ignore[index]


def test_n3_1_12_nested_mapping_exposed_read_only() -> None:
    """Read-only that stops at the top level leaves the same hole open."""
    m = DiagramMeta(params={"nested": {"k": 1}})
    with pytest.raises(TypeError):
        m.params["nested"]["k"] = 2  # type: ignore[index]


# --------------------------------------------------------------------------
# §3.2 -- the general ``meta`` propagation rule for derived diagrams
#
# Entry 68: "§3.2 gains a general `meta` propagation rule for derived diagrams
# and with it the requirement that `d.finite` record its drop."
#
#   "**Every derived diagram carries `meta` through unchanged, except where the
#    derivation invalidates what `provenance` says about the bars.**"
#
# The table names five derivations. ``canonical()``, ``dim(k)`` and ``b[i]``
# carry ``meta`` unchanged; ``finite`` and ``finitize`` record.
# --------------------------------------------------------------------------


def tagged() -> PersistenceDiagram:
    """A diagram whose provenance carries every key the rule reasons about."""
    return diagram(
        dims=[0, 0, 1, 1],
        births=[0.0, 0.0, 1.0, 1.5],
        deaths=[1.0, math.inf, math.inf, 2.5],
        meta=DiagramMeta(
            filtration="rips",
            backend="ripser",
            provenance={
                "essential_bars": "faithful",
                "essential_bars_source": "faithful",
            },
        ),
    )


def test_s3_2_canonical_carries_meta_through_unchanged() -> None:
    """ "a permutation; no bar added, none removed"."""
    d = tagged()
    assert d.canonical().meta == d.meta


def test_s3_2_dim_carries_meta_through_unchanged() -> None:
    """ "a restriction of degree, not a deletion within one"."""
    d = tagged()
    assert d.dim(0).meta == d.meta
    assert d.dim(1).meta == d.meta


def test_s3_2_dim_of_an_absent_degree_carries_meta_through() -> None:
    """The empty answer is still a degree restriction, not a drop."""
    d = tagged()
    assert d.dim(7).meta == d.meta


def test_s3_2_batch_item_carries_the_members_own_meta() -> None:
    """ "the member's own metadata, not a derivation"."""
    first = diagram([0], [0.0], [1.0], DiagramMeta(description="first"))
    second = diagram([1], [0.5], [2.0], DiagramMeta(description="second"))
    b = batch_of(first, second)
    assert b[0].meta.description == "first"
    assert b[1].meta.description == "second"


# --------------------------------------------------------------------------
# §3.2 -- N3.2-2, N3.2-3: ``d.finite`` MUST record the drop it performs
#
#   N3.2-2: "`d.finite` MUST record the drop it performs, on exactly the terms
#    §5 sets for `finitize(at="drop")`: `provenance["essential_bars"] =
#    "finitized_dropped"` and `essential_bars_dropped`, the count removed, with
#    `essential_bars_finitized_at` cleared and `essential_bars_source`
#    untouched."
#   N3.2-3: "The two produce the same diagram bar for bar, so they MUST produce
#    the same provenance."
# --------------------------------------------------------------------------


def test_n3_2_2_finite_records_finitized_dropped() -> None:
    d = tagged()
    assert d.finite.meta.provenance["essential_bars"] == "finitized_dropped"


def test_n3_2_2_finite_records_the_count_removed() -> None:
    """ "`essential_bars_dropped`, the count removed"."""
    d = tagged()
    dropped = int(np.asarray(d.essential).sum())
    assert dropped == 2
    assert d.finite.meta.provenance["essential_bars_dropped"] == dropped


def test_n3_2_2_finite_clears_essential_bars_finitized_at() -> None:
    """ "with `essential_bars_finitized_at` cleared".

    A drop has no substituted value to name, so the key must not survive from
    an earlier substitution. Cleared is satisfied by removal or by ``None``.
    """
    d = diagram(
        dims=[0, 0],
        births=[0.0, 0.0],
        deaths=[1.0, math.inf],
        meta=DiagramMeta(
            provenance={
                "essential_bars": "finitized_at",
                "essential_bars_finitized_at": 9.0,
                "essential_bars_source": "faithful",
            }
        ),
    )
    provenance = d.finite.meta.provenance
    assert provenance.get("essential_bars_finitized_at") is None


def test_n3_2_2_finite_leaves_essential_bars_source_untouched() -> None:
    """ "and `essential_bars_source` untouched" -- it is the adapter's key."""
    d = diagram(
        dims=[0, 0],
        births=[0.0, 0.0],
        deaths=[1.0, math.inf],
        meta=DiagramMeta(
            provenance={
                "essential_bars": "faithful",
                "essential_bars_source": "lost_upstream",
            }
        ),
    )
    assert d.finite.meta.provenance["essential_bars_source"] == "lost_upstream"


def test_n3_2_3_finite_and_finitize_drop_agree_on_provenance() -> None:
    """ "they MUST produce the same provenance"."""
    d = tagged()
    assert dict(d.finite.meta.provenance) == dict(d.finitize(at="drop").meta.provenance)


def test_n3_2_3_finite_and_finitize_drop_agree_bar_for_bar() -> None:
    d = tagged()
    assert d.finite == d.finitize(at="drop")
    assert d.finite.content_hash == d.finitize(at="drop").content_hash


def test_n3_2_2_finite_carries_the_non_provenance_fields_through() -> None:
    """Only ``essential_bars`` and its qualifiers are in that position."""
    d = tagged()
    finite = d.finite
    assert finite.meta.filtration == "rips"
    assert finite.meta.backend == "ripser"


def test_s5_finite_on_a_diagram_with_no_essential_bar_is_unchanged() -> None:
    """§5, bound to ``finite`` by §3.2: "A diagram with **no** essential bar
    is returned with `meta` untouched, on §5's terms -- nothing was dropped,
    and a recorded drop of zero bars asserts a change that did not happen."
    """
    d = diagram(
        dims=[0, 1],
        births=[0.0, 1.0],
        deaths=[1.0, 2.0],
        meta=DiagramMeta(provenance={"essential_bars": "lost_upstream"}),
    )
    assert int(np.asarray(d.essential).sum()) == 0
    assert d.finite.meta == d.meta
    assert d.finite.meta.provenance["essential_bars"] == "lost_upstream"


def test_s3_2_finite_is_idempotent_without_a_false_second_claim() -> None:
    """ "`d.finite.finitize(at="drop")` then meets §5's return-unchanged rule".

    The second call finds no essential bar, so it must change nothing -- the
    count recorded by the first call must survive intact rather than being
    overwritten with a drop of zero.
    """
    d = tagged()
    once = d.finite
    twice = once.finitize(at="drop")
    assert twice.meta == once.meta
    assert twice.meta.provenance["essential_bars_dropped"] == 2


# --------------------------------------------------------------------------
# §3.2 -- ``d.essential`` and ``d.finite`` are not complements
#
# Entry 74: "§3.2 states that `d.essential` and `d.finite` are not
# complements."
#
#   "`d.essential` is a **mask** over bars, shape `(n_bars,)`; `d.finite` is a
#    **diagram**. The complement of `d.essential` is `~d.essential`, another
#    mask, and that is the mask `d.finite` selects on".
#   "There is no `d.finite_mask` and none is needed".
# --------------------------------------------------------------------------


def test_s3_2_essential_is_a_mask_of_shape_n_bars() -> None:
    d = sample()
    mask = np.asarray(d.essential)
    assert mask.dtype == np.bool_
    assert mask.shape == (d.n_bars,)


def test_s3_2_finite_is_a_diagram_not_a_mask() -> None:
    d = sample()
    assert isinstance(d.finite, PersistenceDiagram)


def test_s3_2_finite_selects_on_the_complement_of_essential() -> None:
    d = sample()
    mask = np.asarray(d.essential)
    assert d.finite.n_bars == int((~mask).sum())
    np.testing.assert_array_equal(
        np.asarray(d.finite.births), np.asarray(d.births)[~mask]
    )


def test_s3_2_there_is_no_finite_mask_accessor() -> None:
    """ "There is no `d.finite_mask` and none is needed"."""
    assert not hasattr(PersistenceDiagram, "finite_mask")


# --------------------------------------------------------------------------
# §3.2 -- N3.2-1: ``d.dim(k)`` for an absent ``k``
#
#   "`d.dim(k)` for a `k` not present MUST return an empty diagram, not raise.
#    Empty is a legitimate answer to "what are the 7-dimensional cycles"."
# --------------------------------------------------------------------------


def test_n3_2_1_dim_of_an_absent_degree_returns_empty() -> None:
    d = sample()
    assert 7 not in [int(k) for k in np.asarray(d.dimensions)]
    assert d.dim(7).n_bars == 0


def test_n3_2_1_dim_of_a_gap_in_the_dimensions_returns_empty() -> None:
    """§3's H0-and-H2-but-no-H1 diagram, whose gap is representable."""
    d = diagram(dims=[0, 2], births=[0.0, 1.0], deaths=[1.0, 2.0])
    assert d.dim(1).n_bars == 0


def test_n3_2_4_essential_bars_is_a_claim_about_its_own_diagram() -> None:
    """N3.2-4, and the stated imprecision it makes safe: a ``lost_upstream``
    inherited by a degree whose essential classes all survived is
    conservative, so ``dim`` carries it rather than weakening it.
    """
    d = diagram(
        dims=[0, 1],
        births=[0.0, 1.0],
        deaths=[math.inf, 2.0],
        meta=DiagramMeta(provenance={"essential_bars": "lost_upstream"}),
    )
    assert d.dim(1).meta.provenance["essential_bars"] == "lost_upstream"


# --------------------------------------------------------------------------
# §4.2 -- the four rules now governing ``DiagramBatch.__getitem__``
#
# Entry 68: "§4.2's `__getitem__` sketch read `offsets[i]` without
# normalising, so `i = -1` returned an **empty** diagram carrying the last
# diagram's metadata; four rules now govern indexing, and §4 gains `__iter__`."
#
#   "**`__getitem__` MUST normalise its index before it reads `offsets`**, and
#    the three lines the sketch above spends on that are normative rather than
#    illustrative."
# --------------------------------------------------------------------------


def three_diagrams() -> DiagramBatch:
    return batch_of(
        diagram([0], [0.0], [1.0], DiagramMeta(description="d0")),
        diagram([0, 1], [0.0, 1.0], [2.0, 3.0], DiagramMeta(description="d1")),
        diagram(
            [1, 1, 1], [0.0, 0.5, 1.0], [4.0, 5.0, 6.0], DiagramMeta(description="d2")
        ),
    )


def test_s4_2_negative_index_normalised_before_reading_offsets() -> None:
    """The regression the rule was written for: without normalisation ``b[-1]``
    reads ``offsets[-1]`` and ``offsets[0]``, slices ``[total_bars:0]``, and
    returns an **empty** diagram carrying the last diagram's metadata -- §9's
    clean-plausible-wrong category.
    """
    b = three_diagrams()
    last = b[-1]
    assert last.n_bars == 3, "b[-1] must not be the empty slice [total_bars:0]"
    assert last.meta.description == "d2"
    assert last == b[len(b) - 1]


@pytest.mark.parametrize("index", [-1, -2, -3])
def test_s4_2_every_negative_index_matches_its_normalised_form(index: int) -> None:
    """ "A negative index MUST be normalised to `i + len(batch)`"."""
    b = three_diagrams()
    assert b[index] == b[index + len(b)]
    assert b[index].meta == b[index + len(b)].meta


@pytest.mark.parametrize("index", ["0", 1.0, None, 1.5, b"0"])
def test_s4_2_non_integer_index_raises_type_error(index: Any) -> None:
    """ "A non-integer index MUST raise `TypeError`, through `operator.index`
    rather than an `isinstance` check".
    """
    b = three_diagrams()
    with pytest.raises(TypeError):
        b[index]


def test_s4_2_slice_is_refused() -> None:
    """ "Slicing a batch is **not** supported" -- and ``operator.index`` "is
    also what refuses a `slice`".
    """
    b = three_diagrams()
    with pytest.raises(TypeError):
        b[0:2]


def test_s4_2_object_implementing_index_is_accepted() -> None:
    """ "so that any object implementing `__index__` works"."""

    class Index:
        def __index__(self) -> int:
            return 1

    b = three_diagrams()
    assert b[Index()] == b[1]


def test_s4_2_numpy_integer_is_accepted() -> None:
    """The ordinary case of an ``__index__`` object a caller will actually hold."""
    b = three_diagrams()
    assert b[np.int64(2)] == b[2]


def test_s4_2_out_of_range_positive_raises_index_error() -> None:
    b = three_diagrams()
    with pytest.raises(IndexError):
        b[len(b)]


def test_s4_2_out_of_range_negative_raises_index_error() -> None:
    b = three_diagrams()
    with pytest.raises(IndexError):
        b[-len(b) - 1]


def test_s4_2_index_error_names_the_index_the_caller_passed() -> None:
    """ "naming the index **the caller passed** rather than the normalised
    form: an `IndexError(-3)` raised for a `batch[-5]` names an index nobody
    asked for and sends the reader after the wrong bug."

    Three diagrams, so ``b[-5]`` normalises to ``-2``... which is in range for
    a naive reader but not for the rule; use the document's own arithmetic on a
    two-diagram batch, where ``-5`` normalises to ``-3``.
    """
    b = batch_of(diagram([0], [0.0], [1.0]), diagram([0], [0.0], [2.0]))
    with pytest.raises(IndexError) as excinfo:
        b[-5]
    message = str(excinfo.value)
    assert "-5" in message, "the IndexError must name the index the caller passed"
    assert "-3" not in message, "it must not name the normalised form"


def test_s4_2_index_error_names_a_positive_out_of_range_index() -> None:
    b = three_diagrams()
    with pytest.raises(IndexError) as excinfo:
        b[9]
    assert "9" in str(excinfo.value)


def test_s4_2_getitem_does_not_deep_copy_the_buffers() -> None:
    """ "`__getitem__` MUST NOT deep-copy the buffers: it MUST construct its
    `PersistenceDiagram` from ordinary slices of the batch's own `dims`,
    `births` and `deaths`, and MUST NOT materialise an independent copy of the
    bar data."

    Whether the slice is a view is the backend's business; on NumPy it is, so
    on NumPy the absence of a copy is observable as shared memory.
    """
    b = three_diagrams()
    d = b[1]
    assert np.shares_memory(np.asarray(d.births), np.asarray(b.births))
    assert np.shares_memory(np.asarray(d.deaths), np.asarray(b.deaths))
    assert np.shares_memory(np.asarray(d.dims), np.asarray(b.dims))


def test_s4_2_getitem_returns_the_right_window_of_the_buffer() -> None:
    """The slice arithmetic itself, which B1-B7 exist to keep honest."""
    b = three_diagrams()
    np.testing.assert_array_equal(np.asarray(b[0].deaths), [1.0])
    np.testing.assert_array_equal(np.asarray(b[1].deaths), [2.0, 3.0])
    np.testing.assert_array_equal(np.asarray(b[2].deaths), [4.0, 5.0, 6.0])


# --------------------------------------------------------------------------
# §4.2 -- ``DiagramBatch`` MUST be iterable
#
#   "**`DiagramBatch` MUST be iterable, yielding its diagrams in batch
#    order.** ... so `__iter__` is stated rather than inherited from that
#    accident. ... re-iterating a batch MUST yield the same diagrams again."
# --------------------------------------------------------------------------


def test_s4_2_iter_is_stated_not_inherited() -> None:
    assert hasattr(DiagramBatch, "__iter__")


def test_s4_2_iteration_yields_diagrams_in_batch_order() -> None:
    b = three_diagrams()
    assert [d.meta.description for d in b] == ["d0", "d1", "d2"]


def test_s4_2_iteration_yields_persistence_diagrams() -> None:
    b = three_diagrams()
    assert all(isinstance(d, PersistenceDiagram) for d in b)


def test_s4_2_iteration_terminates_and_matches_len() -> None:
    """The ``IndexError`` rule is what keeps this from running forever."""
    b = three_diagrams()
    assert len(list(b)) == len(b) == 3


def test_s4_2_re_iterating_yields_the_same_diagrams_again() -> None:
    """ "re-iterating a batch MUST yield the same diagrams again" -- so
    ``__iter__`` MUST NOT return a one-shot iterator that is the batch itself.
    """
    b = three_diagrams()
    first = [d.content_hash for d in b]
    second = [d.content_hash for d in b]
    assert first == second
    assert len(second) == 3


def test_s4_3_dimensions_comprehension_from_the_document_runs() -> None:
    """ "§4.3's `[d.dimensions for d in b]` already depends on it"."""
    b = three_diagrams()
    assert [[int(k) for k in np.asarray(d.dimensions)] for d in b] == [
        [0],
        [0, 1],
        [1],
    ]


# --------------------------------------------------------------------------
# §8.1 -- per-element packing, and signed-zero normalisation before either
#         byte path
#
# Entry 73: "§8.1's per-element packing, with signed-zero normalisation placed
# before either byte path".
#
#   "**Every other namespace MUST pack element by element as `>d` for `births`
#    and `deaths` and `>i` for `dims`**, in canonical order, reading each
#    element as a Python `float` or `int`."
#   "**Signed-zero normalisation happens before either path, on the array**,
#    and this placement is normative."
#   "**Negative zero MUST be normalised to `+0.0` before hashing.**"
# --------------------------------------------------------------------------


def test_s8_1_hash_matches_the_documented_message_for_one_bar() -> None:
    """One bar: canonical order is trivially itself, so the whole message is
    computable from §8.1 alone.
    """
    d = diagram(dims=[1], births=[0.25], deaths=[2.5])
    assert d.content_hash == spec_content_hash([1], [0.25], [2.5])


def test_s8_1_hash_matches_the_documented_message_for_many_bars() -> None:
    """Canonical order is §7's ``(dim, birth, death)`` ascending, stable."""
    dims = [1, 0, 1, 0]
    births = [1.5, 0.0, 1.0, 0.0]
    deaths = [2.5, 2.0, 3.0, 1.0]
    d = diagram(dims=dims, births=births, deaths=deaths)
    assert d.content_hash == spec_content_hash(dims, births, deaths)


def test_s8_1_hash_covers_essential_bars_as_big_endian_inf() -> None:
    d = diagram(dims=[0], births=[0.0], deaths=[math.inf])
    assert d.content_hash == spec_content_hash([0], [0.0], [math.inf])


def test_s8_1_empty_diagram_is_not_sha256_of_nothing() -> None:
    """ "the length keeps an empty diagram from hashing to `sha256(b"")` -- a
    published constant, indistinguishable from a bug that hashed nothing at
    all."
    """
    d = diagram(dims=[], births=[], deaths=[])
    assert d.content_hash != hashlib.sha256(b"").hexdigest()
    assert d.content_hash == spec_content_hash([], [], [])


def test_s8_1_negative_zero_birth_hashes_as_positive_zero() -> None:
    """ "Zero births are ubiquitous in H0 and `-0.0` is an ordinary product of
    filtration arithmetic, so this is a live case rather than a curiosity."
    """
    positive = diagram(dims=[0], births=[0.0], deaths=[1.0])
    negative = diagram(dims=[0], births=[-0.0], deaths=[1.0])
    assert math.copysign(1.0, float(np.asarray(negative.births)[0])) == -1.0, (
        "the fixture must actually carry -0.0 for this test to mean anything"
    )
    assert negative.content_hash == positive.content_hash


def test_s8_1_negative_zero_death_hashes_as_positive_zero() -> None:
    """A trivial bar on the diagonal at zero -- §2's case meeting §8.1's."""
    positive = diagram(dims=[0], births=[0.0], deaths=[0.0])
    negative = diagram(dims=[0], births=[-0.0], deaths=[-0.0])
    assert negative.content_hash == positive.content_hash


def test_s8_1_equal_diagrams_never_have_different_hashes() -> None:
    """The failure the normalisation rules out: "making `d1 == d2` with
    differing `content_hash`es reachable".
    """
    a = diagram(dims=[0, 0], births=[-0.0, 1.0], deaths=[1.0, 2.0])
    b = diagram(dims=[0, 0], births=[0.0, 1.0], deaths=[1.0, 2.0])
    assert a == b
    assert a.content_hash == b.content_hash


def test_s8_1_hash_is_independent_of_the_order_bars_arrived_in() -> None:
    """ "computed over the diagram's own canonical-ordered arrays"."""
    forward = diagram(dims=[0, 0, 1], births=[0.0, 1.0, 2.0], deaths=[1.0, 3.0, 4.0])
    shuffled = diagram(dims=[1, 0, 0], births=[2.0, 1.0, 0.0], deaths=[4.0, 3.0, 1.0])
    assert forward.content_hash == shuffled.content_hash


def test_s8_1_hash_is_stable_across_calls() -> None:
    d = sample()
    assert d.content_hash == d.content_hash


# --------------------------------------------------------------------------
# §10.1 -- ``save`` MUST refuse a non-host-resident array
#
# Entry 73: "`save()` MUST refuse a non-host-resident array by name."
#
#   "So `save` MUST raise `ValueError` naming the device and the remedy -- the
#    caller's own `.cpu()`, `jax.device_get`, or `xp.from_dlpack` -- rather
#    than let the backend's message reach them, and MUST make that check
#    before it opens the destination, so a failed save leaves no partial file."
# --------------------------------------------------------------------------


def _device_resident_diagram() -> PersistenceDiagram | None:
    """A diagram whose arrays live off-host, or ``None`` if none can be built."""
    try:
        import torch
    except ImportError:
        pass
    else:
        if torch.cuda.is_available():
            return PersistenceDiagram(
                dims=torch.tensor([0], dtype=torch.int32, device="cuda"),
                births=torch.tensor([0.0], dtype=torch.float64, device="cuda"),
                deaths=torch.tensor([1.0], dtype=torch.float64, device="cuda"),
            )
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        return None
    accelerators = [device for device in jax.devices() if device.platform != "cpu"]
    if not accelerators:
        return None
    device = accelerators[0]
    return PersistenceDiagram(
        dims=jax.device_put(jnp.asarray([0], dtype=jnp.int32), device),
        births=jax.device_put(jnp.asarray([0.0], dtype=jnp.float64), device),
        deaths=jax.device_put(jnp.asarray([1.0], dtype=jnp.float64), device),
    )


@pytest.mark.backend
def test_s10_1_save_refuses_a_non_host_resident_array(tmp_path: Path) -> None:
    d = _device_resident_diagram()
    if d is None:
        pytest.skip("no non-CPU device available to place a diagram on")
    path = tmp_path / "device.akd"
    with pytest.raises(ValueError) as excinfo:  # noqa: PT011 -- message asserted below
        save(d, path)
    message = str(excinfo.value)
    assert any(
        remedy in message for remedy in (".cpu()", "device_get", "from_dlpack")
    ), "the ValueError must name the remedy, not just the refusal"
    assert not path.exists(), (
        "the check MUST happen before the destination is opened, "
        "so a failed save leaves no partial file"
    )


# --------------------------------------------------------------------------
# §10.2 -- the document is 1.2.0, and ``save`` writes that
#
# Entry 76: "the document becomes 1.1.0 ... `io.py`'s `_SPEC_VERSION` and the
# four `spec_version` pins in the I/O tests follow." Quoted as written; the
# document has moved twice since, and entry 77 records the minor.
#
#   "`spec_version` | `str` | ... `"1.2.0"` at time of writing."
# --------------------------------------------------------------------------


def rfc_text() -> str:
    return RFC_PATH.read_text(encoding="utf-8")


def test_s10_2_header_version_row_is_the_review_version() -> None:
    for line in rfc_text().splitlines():
        if line.startswith("| **Version** |"):
            assert line.split("|")[2].strip().startswith(SPEC_VERSION)
            return
    pytest.fail("the RFC header has no Version row")


def test_s10_2_schema_example_carries_the_same_version() -> None:
    assert f'"spec_version": "{SPEC_VERSION}"' in rfc_text()


def test_s10_2_schema_table_carries_the_same_version() -> None:
    assert f'`"{SPEC_VERSION}"` at time of writing' in rfc_text()


def test_s10_2_no_stale_version_survives_the_bump() -> None:
    """The three places entry 76 had to move together, and the one that would
    silently disagree if only two were edited.
    """
    text = rfc_text()
    assert '"spec_version": "0.3.0"' not in text
    assert '`"0.3.0"` at time of writing' not in text


def test_s10_2_save_writes_the_document_version(tmp_path: Path) -> None:
    """The implementation half of the same bump."""
    path = tmp_path / "version.akd"
    save(sample(), path)
    assert read_meta_json(path)["spec_version"] == SPEC_VERSION


def test_s10_2_save_writes_the_version_for_a_batch(tmp_path: Path) -> None:
    path = tmp_path / "version-batch.akd"
    save(three_diagrams(), path)
    assert read_meta_json(path)["spec_version"] == SPEC_VERSION


def test_s10_2_load_does_not_branch_on_spec_version(tmp_path: Path) -> None:
    """ "Recorded for audit; `load` MUST NOT branch on it -- ... one that does
    not [change what `load` must do] is a revision older readers are entitled
    to ignore."
    """
    original = tmp_path / "original.akd"
    save(sample(), original)
    meta = read_meta_json(original)
    for version in ("0.1.0", "0.3.0", "9.9.9"):
        patched = tmp_path / f"patched-{version}.akd"
        rewrite_meta_json(original, patched, {**meta, "spec_version": version})
        back = load(patched)
        assert isinstance(back, PersistenceDiagram)
        assert back.content_hash == sample().content_hash


def test_s10_2_archive_holds_exactly_two_named_entries_in_order(
    tmp_path: Path,
) -> None:
    """ "An `.akd` archive MUST contain exactly these two named entries, under
    exactly these names, written in this order."
    """
    path = tmp_path / "layout.akd"
    save(sample(), path)
    with zipfile.ZipFile(path) as archive:
        assert archive.namelist() == ["meta.json", "bars.npz"]


# --------------------------------------------------------------------------
# §3.1 -- N3.1-3: the clamp has one target and one record
#
# Entry 73: "`clamped_rows` gets a target and a recording obligation".
#
#   "An adapter that repairs such a row MUST set `death := birth`, ... and it
#    MUST record the count in `provenance["clamped_rows"]` (§8) -- including
#    `0` where it looked and found none".
#
# The threshold is deliberately unfixed ("this document sets no number"), so
# nothing below asserts a specific epsilon: only that *if* a row is repaired,
# it is repaired to `death := birth` and counted.
# --------------------------------------------------------------------------


def test_n3_1_3_clean_input_records_a_zero_count() -> None:
    """ "including `0` where it looked and found none"."""
    from akriti.diagrams import from_array

    arr = np.asarray([[0.0, 1.0, 0.0], [0.5, 2.0, 1.0]], dtype=np.float64)
    d = from_array(arr, columns=("birth", "death", "dim"))
    assert d.meta.provenance["clamped_rows"] == 0


def test_n3_1_3_repaired_row_is_clamped_to_its_birth_and_counted() -> None:
    """ "MUST set `death := birth`" -- exactly, not approximately."""
    from akriti.diagrams import from_array

    birth = 1.0
    arr = np.asarray([[birth, np.nextafter(birth, -np.inf), 0.0]], dtype=np.float64)
    with pytest.warns(Warning):  # noqa: PT030 -- the RFC requires a warning but names no class
        d = from_array(arr, columns=("birth", "death", "dim"))
    assert float(np.asarray(d.deaths)[0]) == birth
    assert d.meta.provenance["clamped_rows"] == 1
    assert d.n_bars == 1, "a repaired row is an ordinary bar, not a dropped one"


# --------------------------------------------------------------------------
# §6.3 -- a diagram compared against a batch is False, not a ValueError
#
#   "**A `PersistenceDiagram` compared against a `DiagramBatch` is `False`, not
#    a `ValueError`.** ... The same holds for any other type, `None` included."
# --------------------------------------------------------------------------


def test_s6_3_diagram_against_batch_is_false() -> None:
    d = sample()
    b = batch_of(d)
    assert (d == b) is False
    assert (b == d) is False
    assert (d != b) is True
    assert (b != d) is True


@pytest.mark.parametrize("other", [None, 0, "x", [0.0, 1.0], object()])
def test_s6_3_diagram_against_any_other_type_is_false(other: Any) -> None:
    assert (sample() == other) is False
    assert (batch_of(sample()) == other) is False


def test_s6_3_partiality_stops_at_the_type_boundary() -> None:
    """ "each `__eq__` returns `NotImplemented` for an operand it does not
    recognise, Python tries the reflected call, and `==` is then `False`."
    """
    d = sample()
    b = batch_of(d)
    assert d.__eq__(b) is NotImplemented
    assert b.__eq__(d) is NotImplemented


def test_s6_3_container_membership_does_not_raise() -> None:
    assert (sample() in [batch_of(sample())]) is False


# --------------------------------------------------------------------------
# §8 -- N8-9's scope narrowed to the four `str` fields
#
# The Unicode clause itself predates this pass; what the review changed is
# which fields it ranges over: "the five scalar fields" became "the four `str`
# fields -- `filtration`, `backend`, `backend_version` and `description`,
# `coeff_field` being an `int` and outside a rule about text".
# --------------------------------------------------------------------------


def test_n8_9_coeff_field_is_outside_the_text_rule() -> None:
    """The regression the narrowing exists to prevent: an `int` field swept
    into a rule about text.
    """
    m = DiagramMeta(coeff_field=11, provenance={"coeff_field_source": "caller"})
    assert m.coeff_field == 11


@pytest.mark.parametrize(
    "field", ["filtration", "backend", "backend_version", "description"]
)
def test_n8_9_the_four_str_fields_refuse_an_unpaired_surrogate(
    field: str,
) -> None:
    # The RFC requires the refusal but names no exception type, so the
    # assertion is that construction fails -- not which class it fails with.
    with pytest.raises(Exception):  # noqa: B017, PT011 -- see above
        DiagramMeta(**{field: "\ud800"})


def test_n8_9_ordinary_non_ascii_text_is_not_the_target() -> None:
    """ "excludes what cannot be encoded, not what is not ASCII"."""
    m = DiagramMeta(description="Omega-persistence, cafe, 日本語")
    assert m.description is not None


# --------------------------------------------------------------------------
# §8.1 -- the two byte paths agree
#
#   "§11.2 requires the buffer path and the per-element path to agree byte for
#    byte on signed zero, `inf`, subnormals and the `int32` extremes, and a
#    requirement that two paths agree is only testable if both are defined."
#
# NumPy exposes a C-contiguous buffer, so it may take the fast path;
# `array_api_strict` does not, so it MUST pack element by element. The two
# digests must be equal, and equal to the documented message.
# --------------------------------------------------------------------------


CASES = [
    ("signed_zero", [0, 0], [-0.0, 0.0], [1.0, 2.0]),
    ("infinity", [0], [0.0], [math.inf]),
    ("subnormal", [0], [5e-324], [5e-324]),
    ("int32_max", [2147483647], [0.0], [1.0]),
]


@pytest.mark.parametrize(
    ("name", "dims", "births", "deaths"), CASES, ids=[c[0] for c in CASES]
)
def test_s8_1_both_byte_paths_agree(
    name: str, dims: list[int], births: list[float], deaths: list[float]
) -> None:
    strict = pytest.importorskip("array_api_strict")
    numpy_backed = diagram(dims, births, deaths)
    strict_backed = PersistenceDiagram(
        dims=strict.asarray(dims, dtype=strict.int32),
        births=strict.asarray(births, dtype=strict.float64),
        deaths=strict.asarray(deaths, dtype=strict.float64),
    )
    expected = spec_content_hash(dims, births, deaths)
    assert numpy_backed.content_hash == expected
    assert strict_backed.content_hash == expected


# --------------------------------------------------------------------------
# §9.1 -- the bottleneck convention `inf - inf = 0`
#
#   "by convention, here, $\infty-\infty = 0$. Note: an implementation MUST NOT
#    reach that value by subtracting the deaths; by an opposing convention,
#    Python returns `NaN`."
# --------------------------------------------------------------------------


@pytest.mark.distances
def test_s9_1_two_identical_essential_bars_are_zero_apart() -> None:
    distances = pytest.importorskip("akriti.core.distances")
    left = diagram([0], [0.0], [math.inf])
    right = diagram([0], [0.0], [math.inf])
    result = float(distances.bottleneck(left, right))
    assert not math.isnan(result), "reached by subtracting the deaths"
    assert result == 0.0


@pytest.mark.distances
def test_s9_1_essential_bars_cost_their_birth_difference() -> None:
    distances = pytest.importorskip("akriti.core.distances")
    left = diagram([0], [0.0], [math.inf])
    right = diagram([0], [0.5], [math.inf])
    assert float(distances.bottleneck(left, right)) == pytest.approx(0.5)


# --------------------------------------------------------------------------
# §10.3 -- `to_csv()` has no reader half
#
# Entry 73: "§10.3 states that `to_csv()` has no reader half."
#
#   "**It has no reader half, and this document does not add one.** ... The
#    caller supplies the parse ... **§11.2's round-trip case for this pair is
#    therefore `to_csv` then a caller-side parse then `from_array`**".
# --------------------------------------------------------------------------


def test_s10_3_there_is_no_from_csv() -> None:
    import akriti.diagrams as diagrams

    assert "from_csv" not in diagrams.__all__
    assert not hasattr(diagrams, "from_csv")


def test_s10_3_to_csv_has_a_header_row_naming_the_three_columns() -> None:
    d = diagram([0], [0.0], [1.0])
    with pytest.warns(Warning):  # noqa: PT030 -- the RFC requires a warning but names no class
        text = d.to_csv() if hasattr(d, "to_csv") else _to_csv(d)
    assert text.splitlines()[0].strip() == "dim,birth,death"


def _to_csv(obj: Any) -> str:
    from akriti.diagrams import to_csv

    return to_csv(obj)


def test_s10_3_to_csv_writes_inf_as_the_literal() -> None:
    d = diagram([0], [0.0], [math.inf])
    with pytest.warns(Warning):  # noqa: PT030 -- the RFC requires a warning but names no class
        text = _to_csv(d)
    assert "inf" in text.splitlines()[1]
    assert "Infinity" not in text


def test_s10_3_round_trip_is_to_csv_then_a_caller_side_parse() -> None:
    """The three-step round trip §11.2 now requires, with no `from_csv` in it.

    Needs `akriti[numpy]` to run, which is what the clause says the case must
    say of itself.
    """
    import io as _io

    from akriti.diagrams import from_array

    d = diagram([0, 0, 1], [0.0, 0.0, 0.2], [math.inf, 1.0, 0.9])
    with pytest.warns(Warning):  # noqa: PT030 -- the RFC requires a warning but names no class
        text = _to_csv(d)
    header = tuple(text.splitlines()[0].split(","))
    arr = np.genfromtxt(_io.StringIO(text), delimiter=",", skip_header=1)
    back = from_array(arr, columns=header)
    assert back.content_hash == d.content_hash


# --------------------------------------------------------------------------
# §11.2 -- N11.2-9: the determinism mechanism is asserted, not slept for
#
# Entry 74: "§11.2's determinism case asserting the pinned `ZipInfo` fields
# rather than sleeping 2.5 s per case."
#
#   "The mechanism MUST be asserted directly -- every member's
#    `ZipInfo.date_time` is the zip epoch, `(1980, 1, 1, 0, 0, 0)`, and its
#    `compress_type` is `ZIP_STORED` -- rather than only inferred from two
#    writes agreeing."
# --------------------------------------------------------------------------


ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)


def assert_pinned(archive: zipfile.ZipFile) -> None:
    infos = archive.infolist()
    assert infos, "an archive with no members proves nothing"
    for info in infos:
        assert info.date_time == ZIP_EPOCH, info.filename
        assert info.compress_type == zipfile.ZIP_STORED, info.filename


@pytest.mark.parametrize("kind", ["diagram", "batch"])
def test_n11_2_9_outer_archive_members_are_pinned(kind: str, tmp_path: Path) -> None:
    obj = sample() if kind == "diagram" else three_diagrams()
    path = tmp_path / f"{kind}.akd"
    save(obj, path)
    with zipfile.ZipFile(path) as archive:
        assert_pinned(archive)


@pytest.mark.parametrize("kind", ["diagram", "batch"])
def test_n11_2_9_inner_bars_npz_members_are_pinned(kind: str, tmp_path: Path) -> None:
    """ "both archive layers" -- `bars.npz` is itself a zip."""
    import io as _io

    obj = sample() if kind == "diagram" else three_diagrams()
    path = tmp_path / f"{kind}.akd"
    save(obj, path)
    with zipfile.ZipFile(path) as outer:
        payload = outer.read("bars.npz")
    with zipfile.ZipFile(_io.BytesIO(payload)) as inner:
        assert_pinned(inner)


def test_n11_2_10_two_writes_agree_byte_for_byte(tmp_path: Path) -> None:
    """ "A second write MAY still be compared, but it is a corroboration rather
    than the test." -- no sleep, because nothing depends on the clock.
    """
    first = tmp_path / "first.akd"
    second = tmp_path / "second.akd"
    save(sample(), first)
    save(sample(), second)
    assert first.read_bytes() == second.read_bytes()


# --------------------------------------------------------------------------
# §11 -- the `reduced_homology=False` check becomes three-termed
#
# Entry 69: "§11's `reduced_homology=False` check refused three ordinary calls.
# "All H0 deaths are finite" is a reduction over an empty selection, so it
# holds vacuously wherever `homology_dimensions` excludes 0".
#
#   "The predicate an implementation tests is therefore three-termed -- the
#    diagram is non-empty, it carries at least one degree-0 row, and every
#    degree-0 death is finite -- and all three terms MUST hold before it
#    raises."
#
# giotto's output is `(n_samples, n_rows, 3)` with columns `(birth, death,
# dim)`, which is what these arrays are.
# --------------------------------------------------------------------------


def giotto_array(rows: list[list[float]]) -> np.ndarray:
    return np.asarray([rows], dtype=np.float64)


def from_giotto_(arr: np.ndarray, **kwargs: Any) -> DiagramBatch:
    from akriti.diagrams import from_giotto

    return from_giotto(arr, **kwargs)


def test_s11_all_three_terms_hold_so_the_declaration_is_refused() -> None:
    """Non-empty, carries H0, every H0 death finite: impossible under
    `reduced_homology=False`, so it MUST be refused.
    """
    arr = giotto_array([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.5, 1.5, 1.0]])
    with pytest.raises(ValueError) as excinfo:  # noqa: PT011 -- message asserted below
        from_giotto_(arr, reduced_homology=False, infinity_values=math.inf)
    message = str(excinfo.value)
    assert "reduced_homology" in message
    assert "infinity_values" in message


def test_s11_term_two_fails_when_homology_dimensions_excludes_zero() -> None:
    """The regression entry 69 names: a non-empty diagram whose H0 sub-diagram
    is absent makes "all H0 deaths are finite" vacuously true.
    """
    arr = giotto_array([[0.5, 1.5, 1.0], [0.6, 1.6, 2.0]])
    b = from_giotto_(arr, reduced_homology=False, infinity_values=math.inf)
    assert isinstance(b, DiagramBatch)
    assert len(b) == 1


@pytest.mark.parametrize(
    "rows",
    [
        [[0.5, 1.5, 1.0]],
        [[0.5, 1.5, 2.0]],
        [[0.5, 1.5, 1.0], [0.6, 1.6, 2.0]],
    ],
    ids=["h1_only", "h2_only", "h1_and_h2"],
)
def test_s11_no_h0_sub_diagram_is_accepted(rows: list[list[float]]) -> None:
    """A.10's measured cases: `homology_dimensions` of `(1,)`, `(2,)`, `(1, 2)`."""
    b = from_giotto_(
        giotto_array(rows), reduced_homology=False, infinity_values=math.inf
    )
    assert len(b) == 1


def test_s11_term_one_fails_on_an_empty_diagram() -> None:
    """ "An empty diagram has no H0 bar to be non-finite, and refusing it here
    would reject what §3.2 and §8.2 both treat as valid."
    """
    arr = np.zeros((1, 0, 3), dtype=np.float64)
    b = from_giotto_(arr, reduced_homology=False, infinity_values=math.inf)
    assert len(b) == 1
    assert b[0].n_bars == 0


def test_s11_term_three_fails_when_an_h0_death_is_non_finite() -> None:
    """The essential H0 class is present, so the declaration is consistent."""
    arr = giotto_array([[0.0, math.inf, 0.0], [0.0, 1.0, 0.0]])
    b = from_giotto_(arr, reduced_homology=False, infinity_values=math.inf)
    assert len(b) == 1


def test_s11_the_check_does_not_extend_to_reduced_homology_true() -> None:
    """ "The check does not extend to `reduced_homology=True`, where the
    essential class is dropped by design ... the asymmetry is stated."
    """
    arr = giotto_array([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]])
    b = from_giotto_(arr, reduced_homology=True, infinity_values=math.inf)
    assert len(b) == 1


# --------------------------------------------------------------------------
# §11 -- the batch length is the leading axis, for every `n_samples`
#
#   "giotto returns a single `(n_samples, n_rows, 3)` array whose leading axis
#    is unambiguous, so `from_giotto` can read it without asking and MUST
#    return a batch of that length."
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_samples", [1, 2, 4])
def test_s11_batch_length_is_the_leading_axis(n_samples: int) -> None:
    arr = np.zeros((n_samples, 2, 3), dtype=np.float64)
    arr[:, :, 1] = 1.0
    arr[:, :, 2] = 1.0
    b = from_giotto_(arr, reduced_homology=True, infinity_values=math.inf)
    assert isinstance(b, DiagramBatch)
    assert len(b) == n_samples


# --------------------------------------------------------------------------
# §11 -- `from_gudhi` takes one sample where `fit_transform` returns many
#
# Entry 73: "§11 says `from_gudhi` takes one sample where `fit_transform`
# returns many."
#
# D20's measured mapping: index position in `homology_dimensions` is not the
# homological degree. "`homology_dimensions=[2, 0]` returns H2 first and H0
# second, and `[1]` a length-one list holding H1."
# --------------------------------------------------------------------------


def from_gudhi_(obj: Any, **kwargs: Any) -> PersistenceDiagram:
    from akriti.diagrams import from_gudhi

    return from_gudhi(obj, **kwargs)


def test_s11_sklearn_form_labels_by_homology_dimensions_not_position() -> None:
    """`[2, 0]` returns H2 first and H0 second."""
    first = np.asarray([[0.0, 1.0], [0.1, 1.1]], dtype=np.float64)
    second = np.asarray([[0.0, 2.0]], dtype=np.float64)
    d = from_gudhi_([first, second], homology_dimensions=[2, 0])
    assert sorted(int(k) for k in np.asarray(d.dimensions)) == [0, 2]
    assert d.dim(2).n_bars == 2
    assert d.dim(0).n_bars == 1
    np.testing.assert_allclose(np.asarray(d.dim(0).deaths), [2.0])


def test_s11_a_length_one_list_holds_the_degree_it_names() -> None:
    """`[1]` is a length-one list holding H1 -- not H0."""
    block = np.asarray([[0.0, 1.0]], dtype=np.float64)
    d = from_gudhi_([block], homology_dimensions=[1])
    assert sorted(int(k) for k in np.asarray(d.dimensions)) == [1]
    assert d.dim(0).n_bars == 0, "the absent degree is empty, not a raise"


def test_s11_from_gudhi_returns_one_sample_not_a_batch() -> None:
    """ "`from_gudhi` returns a scalar `PersistenceDiagram`, so the caller
    indexes ... and `DiagramBatch.from_diagrams` (§4.2) assembles the batch."
    """
    result = [
        [np.asarray([[0.0, 1.0]]), np.asarray([[0.5, 2.0]])],
        [np.asarray([[0.0, 3.0]]), np.asarray([[1.0, 4.0]])],
    ]
    per_sample = [
        from_gudhi_(sample_, homology_dimensions=[0, 1]) for sample_ in result
    ]
    assert all(isinstance(d, PersistenceDiagram) for d in per_sample)
    assert len(DiagramBatch.from_diagrams(per_sample)) == 2


def test_s11_dim_alongside_the_sklearn_form_is_refused() -> None:
    """ "Passing `dim=` alongside this form MUST raise `TypeError` on the same
    grounds the plain `list` form is refused it".
    """
    obj = [np.asarray([[0.0, 1.0]]), np.asarray([[0.5, 2.0]])]
    with pytest.raises(TypeError):
        from_gudhi_(obj, homology_dimensions=[0, 1], dim=0)


def test_s11_omitting_homology_dimensions_is_a_type_error() -> None:
    obj = [np.asarray([[0.0, 1.0]]), np.asarray([[0.5, 2.0]])]
    with pytest.raises(TypeError):
        from_gudhi_(obj)


@pytest.mark.parametrize("dimensions", [[0], [0, 1, 2]])
def test_s11_a_mismatched_length_is_a_value_error(dimensions: list[int]) -> None:
    """ "passing a sequence whose length does not match the outer list MUST
    raise `ValueError`" -- the type discipline is the point: missing is a
    `TypeError`, wrong length is a `ValueError`.
    """
    obj = [np.asarray([[0.0, 1.0]]), np.asarray([[0.5, 2.0]])]
    with pytest.raises(ValueError):  # noqa: PT011 -- the RFC pins the type, not the text
        from_gudhi_(obj, homology_dimensions=dimensions)


# --------------------------------------------------------------------------
# §4.2 -- N4.2-16: `from_diagrams`'s `xp` disagreement
#
#   "Passing `xp` alongside a non-empty `diagrams` is permitted and MUST be
#    rejected with `ValueError` if it disagrees with the diagrams' own
#    namespace, **compared by `is` and not by any weaker test**".
# --------------------------------------------------------------------------


def test_n4_2_16_a_disagreeing_namespace_is_a_value_error() -> None:
    strict = pytest.importorskip("array_api_strict")
    with pytest.raises(ValueError):  # noqa: PT011 -- the RFC pins the type, not the text
        DiagramBatch.from_diagrams([sample()], xp=strict)


def test_n4_2_16_the_matching_namespace_is_accepted() -> None:
    d = sample()
    assert len(DiagramBatch.from_diagrams([d], xp=d.xp)) == 1


# --------------------------------------------------------------------------
# §4.3 -- the gaps stay gaps
#
#   "A batch-level `finite` would additionally have to rewrite each `metas[i]`
#    per segment, on §3.2's propagation rule, since how many bars it dropped
#    differs per diagram."
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["finite", "dim", "dimensions"])
def test_s4_3_batch_level_gaps_are_not_half_filled(name: str) -> None:
    assert not hasattr(DiagramBatch, name)


def test_s4_3_batch_essential_is_a_mask_one_level_up() -> None:
    """ "§4.3 carries the same asymmetry one level up, where `b.essential` is a
    mask and a batch-level `finite` is a stated gap."
    """
    b = three_diagrams()
    mask = np.asarray(b.essential)
    assert mask.dtype == np.bool_
    assert mask.shape == (np.asarray(b.dims).shape[0],)


# --------------------------------------------------------------------------
# Appendix C -- the generated normative-requirements index
#
# Entry 75: "**New Appendix C, the normative-requirements index** ... It is
# **generated** ... It is placed **before** the changelog".
#
# These check the document's own internal consistency, without invoking the
# generator: a reader who trusts the index needs the count line, the coverage
# rule and the placement to hold on the file as committed.
# --------------------------------------------------------------------------


def appendix_c_rows() -> list[list[str]]:
    inside = False
    rows: list[list[str]] = []
    for line in rfc_text().splitlines():
        if line.startswith("## Appendix C"):
            inside = True
            continue
        if inside and line.startswith("## Appendix D"):
            break
        if inside and line.startswith("| `N"):
            rows.append([cell.strip() for cell in line.split("|")[1:-1]])
    return rows


def appendix_c_count_line() -> str:
    inside = False
    for line in rfc_text().splitlines():
        if line.startswith("## Appendix C"):
            inside = True
            continue
        if inside and line.endswith("clauses:") is False and " clauses: " in line:
            return line.strip()
    pytest.fail("Appendix C has no count line")


def test_appendix_c_is_placed_before_the_changelog() -> None:
    """ "so that removal leaves no gap in the lettering"."""
    text = rfc_text()
    index = text.index("## Appendix C — Normative requirements index")
    changelog = text.index("## Appendix D — Changelog")
    assert index < changelog
    assert "## Appendix C — Changelog" not in text


def test_appendix_c_letters_are_contiguous() -> None:
    letters = [
        line.split()[2]
        for line in rfc_text().splitlines()
        if line.startswith("## Appendix ")
    ]
    assert letters == ["A", "B", "C", "D"]


def test_appendix_c_count_line_matches_the_table() -> None:
    """The count line is the one summary a reader checks without counting."""
    rows = appendix_c_rows()
    line = appendix_c_count_line()
    total = int(line.split()[0])
    assert total == len(rows)
    tally: dict[str, int] = {}
    for row in rows:
        tally[row[2].replace("*", "")] = tally.get(row[2].replace("*", ""), 0) + 1
    body = line.split(": ", 1)[1].rstrip(".")
    for part in body.split(", "):
        keyword, count = part.rsplit(" ", 1)
        assert tally.get(keyword) == int(count), keyword
    assert sum(tally.values()) == total


def test_appendix_c_indexes_sections_one_through_eleven_only() -> None:
    """ "§1 through §11 are indexed: §12 records decisions ... and the
    appendices hold evidence and rationale".
    """
    allowed = {str(n) for n in range(1, 12)}
    for row in appendix_c_rows():
        section = row[1].lstrip("§")
        assert section.split(".")[0] in allowed, row[0]


def test_appendix_c_ids_agree_with_their_section_and_are_contiguous() -> None:
    seen: dict[str, int] = {}
    for row in appendix_c_rows():
        ident = row[0].strip("`")
        section, number = ident.lstrip("N").rsplit("-", 1)
        assert f"§{section}" == row[1], ident
        expected = seen.get(section, 0) + 1
        assert int(number) == expected, ident
        seen[section] = expected


def test_appendix_c_ids_are_unique() -> None:
    ids = [row[0] for row in appendix_c_rows()]
    assert len(ids) == len(set(ids))


# --------------------------------------------------------------------------
# §12 -- the header loses the clause contradicting the same sentence
#
# Entry 68: "§12's header loses a clause contradicting the same sentence."
# Entry 71 opened D24; entry 75 closed it, so §12.1 now carries one row.
# --------------------------------------------------------------------------


def test_s12_header_no_longer_says_the_open_section_is_empty() -> None:
    """The retired clause survives in Appendix D as a record of what an
    earlier revision said, so the assertion is scoped to §12's own header.
    """
    header = rfc_text().split("## 12. Decisions")[1].split("### 12.1 Open")[0]
    assert "§12.1 is empty" not in header
    assert "are open" in header or "is open" in header


def test_s12_open_decisions_are_d22() -> None:
    text = rfc_text()
    open_section = text.split("### 12.1 Open")[1].split("### 12.2")[0]
    open_ids = {
        line.split("**")[1]
        for line in open_section.splitlines()
        if line.startswith("| **D")
    }
    assert open_ids == {"D22"}
