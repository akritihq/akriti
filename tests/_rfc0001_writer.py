"""The revision the writer implements, for tests that pin it. RFC-0001 §10.2.

`io._SPEC_VERSION` may trail the document's Version row, and
`io._SPEC_VERSION_GAP` says why. Tests that state the current revision's
surface are conditioned on `BELOW_1_3_0`, so they xfail strictly while the
writer is held and run as ordinary tests the moment it claims 1.3.0. Tests
that pin the 1.2.0 writer carry `FINITE_IS_A_DIAGRAM` where they read
`d.finite`, and fail and go when it moves.

Defined once here so the condition and the filter cannot drift between the
files that use them.
"""

from __future__ import annotations

import pytest

import akriti.diagrams.io as diagrams_io

#: `io._SPEC_VERSION` as a tuple, for comparison against a revision.
IMPLEMENTED_VERSION = tuple(int(part) for part in diagrams_io._SPEC_VERSION.split("."))

#: True while the writer is held below RFC-0001 1.3.0.
BELOW_1_3_0 = IMPLEMENTED_VERSION < (1, 3, 0)

#: The writer's diagram-valued `d.finite` warns, on every call, that RFC-0001
#: 1.3.0 makes it a mask (D28); `test_s3_2_finite_warns_that_it_becomes_a_mask`
#: in `tests/test_rfc0001_review_clauses.py` asserts the warning. Tests pinning
#: the diagram-valued accessor filter that one message and nothing else.
FINITE_IS_A_DIAGRAM = pytest.mark.filterwarnings(
    "ignore:d.finite will return a boolean mask:FutureWarning"
)
