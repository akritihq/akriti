"""Pin `content_hash`'s bytes. RFC-0001 §8.1, §8.2.

`content_hash` is what a paper pins, which makes it the one value in this
library that must not change quietly. §8.1 specifies the hashed message
exactly -- domain tag, bar count, three big-endian columns -- rather than
leaving it to the implementation, and these tests hold the implementation to
that spec rather than to whatever it currently emits.

**The reason this file exists separately from
`test_array_api_conformance.py`:** `_big_endian_block` has two paths that MUST
produce identical bytes, and the conformance suite structurally cannot compare
them. It runs under `array_api_strict`, whose arrays expose no buffer, so it
takes the per-element `struct.pack` fallback every time. The buffer-protocol
fast path is the one every NumPy-backed diagram takes -- which is to say
essentially every diagram anyone will ever hash -- and it was, until this
file, the path no test reached. A divergence between the two would silently
partition published digests by backend, and CI would have agreed with both.

So the agreement test runs on NumPy directly and does not `importorskip`
anything: it must run in the default test environment, not only where the
conformance extra is installed.
"""

from __future__ import annotations

import array
import hashlib
import struct
import sys
from typing import Any

import numpy as np
import pytest

from akriti.diagrams import DiagramBatch, DiagramMeta, PersistenceDiagram
from akriti.diagrams.core import _big_endian_block

# Values chosen to exercise every case §8.1 reasons about: both signs of zero
# (the normalisation), +inf (essential bars), a subnormal and a large finite
# value (where a byte-swap bug would not show up in the low bits), and the
# int32 extremes.
FLOAT_EDGE_CASES = [0.0, -0.0, np.inf, 5e-324, 1.7976931348623157e308, -1.5, 0.1]
INT_EDGE_CASES = [0, 1, 2, 2**31 - 1]


class _NoBuffer:
    """An array-like that refuses the buffer protocol, forcing the slow path.

    Stands in for `array_api_strict`, torch and JAX without requiring any of
    them to be installed: `_big_endian_block` selects its path on whether
    `memoryview()` succeeds, so anything that raises there takes the same
    branch those backends do.
    """

    def __init__(self, values: np.ndarray) -> None:
        self._values = values
        self.shape = values.shape

    def __getitem__(self, i: int) -> object:
        return self._values[i]


def _reference_block(values: list[float] | list[int], typecode: str) -> bytes:
    """§8.1's column layout, written out longhand.

    Deliberately not a refactor of either path in `core.py`: a test that
    shares an implementation with the thing it checks cannot catch a bug in
    the shared part.
    """
    return b"".join(struct.pack(">" + typecode, v) for v in values)


@pytest.fixture
def diagram() -> PersistenceDiagram:
    return PersistenceDiagram(
        dims=np.asarray([1, 0, 0], dtype=np.int32),
        births=np.asarray([0.5, 0.0, 0.25], dtype=np.float64),
        deaths=np.asarray([1.5, np.inf, 0.75], dtype=np.float64),
    )


# -- §8.1 the two paths must agree ---------------------------------------


@pytest.mark.parametrize(
    ("values", "typecode", "dtype"),
    [
        (FLOAT_EDGE_CASES, "d", np.float64),
        (INT_EDGE_CASES, "i", np.int32),
    ],
)
def test_buffer_fast_path_and_struct_fallback_produce_identical_bytes(
    values: list[float], typecode: str, dtype: type
) -> None:
    """The invariant `_big_endian_block`'s docstring states but nothing checked.

    A published `content_hash` must not depend on which backend recomputed
    it. NumPy takes the buffer path, every other backend takes the fallback,
    and the digest is only backend-independent if the two emit the same bytes.
    """
    array = np.asarray(values, dtype=dtype)

    fast = _big_endian_block(array, typecode)
    slow = _big_endian_block(_NoBuffer(array), typecode)
    reference = _reference_block(values, typecode)

    assert fast == slow, "buffer fast path and struct fallback disagree"
    assert fast == reference, "fast path does not match §8.1's byte layout"


def test_numpy_actually_takes_the_fast_path() -> None:
    """Guard the premise of the test above.

    If NumPy ever stopped exposing a conforming buffer, the agreement test
    would still pass -- by comparing the fallback against itself -- and the
    fast path would go back to being untested without anything failing.
    """
    view = memoryview(np.asarray([1.0], dtype=np.float64))
    assert view.c_contiguous
    assert view.format == "d"
    assert view.itemsize == struct.calcsize(">d")


def test_empty_column_hashes_to_empty_bytes() -> None:
    for typecode, dtype in (("d", np.float64), ("i", np.int32)):
        array = np.zeros(0, dtype=dtype)
        assert _big_endian_block(array, typecode) == b""
        assert _big_endian_block(_NoBuffer(array), typecode) == b""


# -- §8.1 the specified message ------------------------------------------


def test_hash_matches_the_message_rfc_specifies(diagram: PersistenceDiagram) -> None:
    """§8.1's message, rebuilt from the spec text rather than from `core.py`."""
    canonical = diagram.canonical()
    expected = hashlib.sha256(
        b"akriti.PersistenceDiagram.v1\x00"
        + (3).to_bytes(8, "big")
        + _reference_block([0, 0, 1], "i")
        + _reference_block([0.0, 0.25, 0.5], "d")
        + _reference_block([np.inf, 0.75, 1.5], "d")
    ).hexdigest()

    assert [int(v) for v in canonical.dims] == [0, 0, 1]
    assert diagram.content_hash == expected


def test_empty_diagram_does_not_hash_to_sha256_of_nothing() -> None:
    """§8.1: the length prefix is what stops this being `sha256(b"")`."""
    empty = PersistenceDiagram(
        dims=np.zeros(0, dtype=np.int32),
        births=np.zeros(0, dtype=np.float64),
        deaths=np.zeros(0, dtype=np.float64),
    )
    assert empty.content_hash != hashlib.sha256(b"").hexdigest()
    assert (
        empty.content_hash
        == hashlib.sha256(
            b"akriti.PersistenceDiagram.v1\x00" + (0).to_bytes(8, "big")
        ).hexdigest()
    )


def test_hash_ignores_row_order(diagram: PersistenceDiagram) -> None:
    """§7: the hash is over the multiset, via canonical order."""
    order = [2, 0, 1]
    shuffled = PersistenceDiagram(
        dims=diagram.dims[order],
        births=diagram.births[order],
        deaths=diagram.deaths[order],
    )
    assert shuffled.content_hash == diagram.content_hash


def test_hash_ignores_metadata(diagram: PersistenceDiagram) -> None:
    """§8.1: bars only, never metadata."""
    annotated = PersistenceDiagram(
        dims=diagram.dims,
        births=diagram.births,
        deaths=diagram.deaths,
        meta=DiagramMeta(backend="gudhi", backend_version="3.11.0"),
    )
    assert annotated.content_hash == diagram.content_hash


# -- §8.1 negative zero ---------------------------------------------------


def test_negative_zero_normalised_so_equal_diagrams_never_disagree() -> None:
    """§8.1, and the reachability §6.3 describes from the other end.

    `-0.0 == 0.0`, so `==` calls these equal and a stable canonical sort
    cannot separate them. Without the normalisation their digests would
    differ, making `d1 == d2` with differing hashes reachable -- which is the
    specific contradiction §8.1 requires the normalisation to prevent.
    """
    negative = PersistenceDiagram(
        dims=np.asarray([0], dtype=np.int32),
        births=np.asarray([-0.0], dtype=np.float64),
        deaths=np.asarray([1.0], dtype=np.float64),
    )
    positive = PersistenceDiagram(
        dims=np.asarray([0], dtype=np.int32),
        births=np.asarray([0.0], dtype=np.float64),
        deaths=np.asarray([1.0], dtype=np.float64),
    )

    assert np.signbit(negative.births[0])  # the input really is -0.0
    assert negative == positive
    assert negative.content_hash == positive.content_hash


def test_normalisation_does_not_disturb_inf_or_ordinary_values(
    diagram: PersistenceDiagram,
) -> None:
    """`+ 0.0` must be a no-op on everything except the sign of zero."""
    for value in FLOAT_EDGE_CASES:
        array = np.asarray([value], dtype=np.float64)
        normalised = array + 0.0
        if value == 0.0:
            assert not np.signbit(normalised[0])
        elif np.isinf(value):
            assert np.isinf(normalised[0])
        else:
            assert normalised[0] == value


# -- §8.2 batch hash ------------------------------------------------------


def test_batch_of_one_cannot_collide_with_the_diagram_it_wraps(
    diagram: PersistenceDiagram,
) -> None:
    """§8.2: domain separation must be structural, not incidental."""
    batch = DiagramBatch.from_diagrams([diagram])
    assert batch.content_hash != diagram.content_hash


def test_batch_hash_matches_the_composition_rfc_specifies(
    diagram: PersistenceDiagram,
) -> None:
    """§8.2: composed from member hashes, never re-serialised from the buffer."""
    batch = DiagramBatch.from_diagrams([diagram, diagram])
    expected = hashlib.sha256(
        b"akriti.DiagramBatch.v1\x00"
        + (2).to_bytes(8, "big")
        + bytes.fromhex(diagram.content_hash)
        + bytes.fromhex(diagram.content_hash)
    ).hexdigest()
    assert batch.content_hash == expected


def test_batch_hash_is_order_sensitive(diagram: PersistenceDiagram) -> None:
    """§8.2: `[A, B]` and `[B, A]` are different batches."""
    other = PersistenceDiagram(
        dims=np.asarray([0], dtype=np.int32),
        births=np.asarray([0.0], dtype=np.float64),
        deaths=np.asarray([2.0], dtype=np.float64),
    )
    forward = DiagramBatch.from_diagrams([diagram, other])
    backward = DiagramBatch.from_diagrams([other, diagram])
    assert forward.content_hash != backward.content_hash


def test_empty_batch_has_a_hash_and_it_is_not_an_empty_diagrams(
    diagram: PersistenceDiagram,
) -> None:
    """§8.2: the explicit `len(b)` is what separates empty from truncated."""
    empty = DiagramBatch.from_diagrams([], xp=np)
    assert (
        empty.content_hash
        == hashlib.sha256(
            b"akriti.DiagramBatch.v1\x00" + (0).to_bytes(8, "big")
        ).hexdigest()
    )
    assert empty.content_hash != DiagramBatch.from_diagrams([diagram]).content_hash


# -- §8.2 conformance-backend parity --------------------------------------


def test_hash_is_identical_across_array_namespaces() -> None:
    """The property the two paths in `_big_endian_block` exist to preserve.

    This is the end-to-end form of the agreement test above: the same bars,
    hashed through NumPy's buffer path and through `array_api_strict`'s
    fallback, must produce one digest. Skips when the conformance extra is
    absent, which is exactly why the byte-level test above does not.
    """
    xps = pytest.importorskip("array_api_strict")

    # Fixture covers every case §8.1 singles out, plus a repeated identical
    # bar: §2 makes a diagram a multiset, so multiplicity must reach the
    # digest, and a hash built from a deduplicating set would agree with
    # itself across namespaces while being wrong on both.
    dims = [1, 0, 0, 0]
    births = [0.5, -0.0, 0.25, 0.25]
    deaths = [1.5, np.inf, 0.75, 0.75]

    def build(xp: object) -> PersistenceDiagram:
        return PersistenceDiagram(
            dims=xp.asarray(dims, dtype=xp.int32),  # type: ignore[attr-defined]
            births=xp.asarray(births, dtype=xp.float64),  # type: ignore[attr-defined]
            deaths=xp.asarray(deaths, dtype=xp.float64),  # type: ignore[attr-defined]
        )

    assert build(np).content_hash == build(xps).content_hash

    # And the repeated bar is load-bearing: dropping one changes the digest.
    deduplicated = PersistenceDiagram(
        dims=np.asarray(dims[:3], dtype=np.int32),
        births=np.asarray(births[:3], dtype=np.float64),
        deaths=np.asarray(deaths[:3], dtype=np.float64),
    )
    assert deduplicated.content_hash != build(np).content_hash


# -- §8.1 on a big-endian host -------------------------------------------
#
# `_big_endian_block`'s fast path byte-swaps *iff* the host is little-endian,
# and every machine this suite has ever run on is little-endian, so the
# `sys.byteorder == "big"` arm has never executed. It is the one line standing
# between §8.1 and a digest that depends on the architecture that computed it.
#
# The host cannot be changed, but the branch can still be exercised honestly.
# `array.frombytes(...)` followed by `.tobytes()` with no swap in between is
# an identity on bytes, so on a genuine big-endian machine the fast path
# returns its input buffer unchanged -- and that buffer is big-endian, because
# native order is. Feeding a *pre-swapped* buffer while `sys.byteorder` reads
# `"big"` reproduces exactly that situation: same bytes in, same bytes out,
# and the result must equal the same endian-independent reference the rest of
# this file checks against.


@pytest.mark.parametrize(
    ("values", "typecode"),
    [
        ([0.5, -0.0, 1.25, float("inf")], "d"),
        ([0, 1, -1, 2**31 - 1], "i"),
    ],
)
def test_the_big_endian_host_branch_emits_unswapped_bytes(
    values: list[Any], typecode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§8.1: the digest is a property of the bars, not of the machine."""
    native = array.array(typecode, values)
    native.byteswap()  # now holds §8.1's big-endian bytes, as a big host would

    monkeypatch.setattr(sys, "byteorder", "big")
    emitted = _big_endian_block(native, typecode)

    assert emitted == _reference_block(values, typecode)


def test_the_big_endian_simulation_is_not_vacuous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The control the test above needs to mean anything.

    If `_big_endian_block` ignored `sys.byteorder` entirely, the assertion
    above would still pass on some inputs by coincidence. These two calls
    differ *only* in what `sys.byteorder` reports, so a differing result is
    proof the branch is live and that the test reaches it.
    """
    values = [0.5, -0.0, 1.25]
    pre_swapped = array.array("d", values)
    pre_swapped.byteswap()

    monkeypatch.setattr(sys, "byteorder", "big")
    as_big_endian_host = _big_endian_block(pre_swapped, "d")

    monkeypatch.setattr(sys, "byteorder", "little")
    as_little_endian_host = _big_endian_block(pre_swapped, "d")

    assert as_big_endian_host != as_little_endian_host
    assert as_big_endian_host == _reference_block(values, "d")
