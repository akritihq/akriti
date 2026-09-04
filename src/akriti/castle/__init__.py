"""`akriti.castle` -- the statistical inference layer. Papers I-IV.

The four tools this library exists for: two-sample testing, sample-size
calculation, per-region significance, and robustness certification, plus the
reporting card that presents them.

Only Tool 1 is scaffolded so far, and its numerical bodies are unported.
"""

from __future__ import annotations

from akriti.castle.two_sample import StructuralAxisWarning, TwoSampleResult, two_sample

__all__ = ["StructuralAxisWarning", "TwoSampleResult", "two_sample"]
