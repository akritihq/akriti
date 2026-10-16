"""``Configuration``'s dataclass surface. RFC-0002 §2.4.

The generated ``__eq__`` and ``__hash__`` of a frozen dataclass compare and
hash every field as a tuple, and a tuple holding arrays raises on both. The
type opts out with ``eq=False``, and that is invisible from a call site: one
line of the decorator reverting would bring the raise back on the first
``==`` or ``set`` a caller writes. `REVIEWING.md`: a trap a reader cannot see
becomes a standing regression test, not prose.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from akriti.core import Configuration


def _configuration() -> Configuration:
    return Configuration(
        positions=np.zeros((3, 2)),
        radii=np.ones(3),
        weights=np.full(3, 1 / np.sqrt(3)),
        frame=1.0,
        truncation=3,
        fitted_on="sha256:pilot",
    )


def test_equality_and_hash_are_identity_rather_than_raising() -> None:
    a, b = _configuration(), _configuration()
    assert a == a
    assert a != b
    assert len({a, b}) == 2


def test_fields_cannot_be_rebound() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        _configuration().positions = np.ones((3, 2))  # type: ignore[misc]
