"""CASTLE — applied-statistics tools over persistence diagrams.

Four tools and a reporting card, specified by RFC-0002 and resting on Paper III
(arXiv:2609.07691). CASTLE is code rather than a paper: its inference is Paper
III's, and what lives here are the decisions Paper III does not make — the
default protocol, the calibrations reported, what a result object carries, and
what a caller may claim about it.

Tool 1 (:func:`two_sample`) is scaffolded; Tools 2 to 4 and the card are not
written. Every numerical body raises ``NotImplementedError``.
"""

from __future__ import annotations

from akriti.castle.two_sample import (
    Calibration,
    InferenceSampleFitError,
    TwoSampleResult,
    two_sample,
)

__all__ = [
    "Calibration",
    "InferenceSampleFitError",
    "TwoSampleResult",
    "two_sample",
]
