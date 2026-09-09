"""CASTLE Tool 1 — two-sample testing on persistence diagrams.

Answers the question this library exists for: *given two groups of samples, do
they differ topologically, and what may I say about the answer?*

The statistical content is Paper III's (arXiv:2609.07691). What this module adds
is the set of decisions Paper III deliberately does not make: which protocol is
the default, which calibrations run, what a result object carries, and — the
part that matters most — what a caller is entitled to claim about the number
they got back. Those decisions are specified in RFC-0002; every obligation below
cites the section that imposes it.

**Structure only. Every numerical body raises ``NotImplementedError``.** That is
deliberate and it is why the previous scaffold (#15) cost an afternoon rather
than a retraction when Paper III's final form withdrew the guarantee it was
built on. Bodies are ported once RFC-0002 §3 has been reviewed.

NumPy-backed by decision: see onboarding §9's dated deviation of 2026-08-09.
``diagrams/`` remains array-API-pure (RFC-0001 §3.3); ``core/`` and ``castle/``
land on NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from akriti.diagrams import DiagramBatch

__all__ = [
    "Calibration",
    "InferenceSampleFitError",
    "TwoSampleResult",
    "two_sample",
]

DEFAULT_PILOT_FRACTION = 1 / 3
DEFAULT_N_PERMUTATIONS = 1000


class InferenceSampleFitError(ValueError):
    """Raised when a configuration would be fitted on the inference sample.

    RFC-0002 §2.3. This is an error rather than a warning on a specific ground:
    the result is not degraded but *invalid*, and a caller who filters warnings
    would be left holding a number with no property at all. For any bounded
    feature map the plug-in carries bias ``tr(Sigma_+)/n_+ + tr(Sigma_-)/n_-``,
    so a configuration chosen to maximise separation on the inference sample is
    chosen partly for its ambient scale.
    """


@dataclass(frozen=True)
class Calibration:
    """One calibration's verdict, carrying the null it is valid under.

    The pairing is the point. RFC-0002 §3.3 makes it non-conforming to render a
    ``p_value`` without ``valid_under`` beside it, because the three
    calibrations do not answer the same question and the one most callers reach
    for is the one that does not cover this module's own estimand.

    Attributes
    ----------
    name:
        ``"spectral"``, ``"permutation"`` or ``"chebyshev"``.
    p_value:
        The p-value.
    valid_under:
        The null hypothesis under which ``p_value`` has its stated level.
        ``"H0_mean"`` for equality of mean embeddings; ``"P == Q"`` for the
        sharp null. **Permutation calibration is exact under the sharp null and
        is not, in general, valid under the mean null** — which is the estimand
        of this whole tool. Paper III states it; this field is how a caller
        cannot miss it.
    finite_sample:
        Whether the level guarantee is finite-sample or asymptotic. True only
        for ``"chebyshev"``.
    """

    name: Literal["spectral", "permutation", "chebyshev"]
    p_value: float
    valid_under: Literal["H0_mean", "P == Q"]
    finite_sample: bool


@dataclass(frozen=True)
class TwoSampleResult:
    """Outcome of a two-sample test, with everything needed to report it honestly.

    Attributes
    ----------
    delta_lower, delta_upper:
        The ``(1 - alpha)`` confidence interval for ``Delta^Psi_nu(P, Q)``
        (Paper III, ``cor:two-sample-confidence-ball``). **This is the primary
        inferential output**, ahead of any p-value, and it is asymptotically
        valid and potentially conservative.

        Note the superscript. This is the separation *in the truncated
        coordinate system actually computed*, not an unqualified distance
        between the two populations (RFC-0002 §2.1). ``truncation`` is carried
        alongside it so a reader can tell which estimand a number refers to.
    transport_lower_bound:
        Lower confidence bound on ``W_inf,0`` between the **padded mean
        measures**, equal to ``delta_lower / (n * lipschitz)``
        (``prop:mean-embedding-transfer``).

        Named for the quantity it bounds rather than for "bottleneck", on
        RFC-0002 §3.5's rule that a field name is the claim for every caller who
        does not read the docstring. It is *not* a bound on the bottleneck
        distance between the two distributions, *not* a statement about exemplar
        diagrams, and *not* ``T / L_Phi`` — that form rested on a uniform
        population converse which ``ex:pop-obstruction`` shows does not exist.

        It requires only the additive Lipschitz interface: no distortion-floor
        assumption, no structured population model, no restriction on P and Q.
    transport_bound_is_informative:
        Whether ``transport_lower_bound`` is large enough to act on. Paper III
        states the caveat and this module propagates it rather than suppressing
        the number: the bound's strength depends on ``n * lipschitz``, and **a
        zero or small value is inconclusive rather than evidence that the mean
        measures are close**.
    calibrations:
        One :class:`Calibration` per calibration run. Never a bare p-value.
    estimand:
        The sentence every rendering MUST display (RFC-0002 §3.8). A rejection
        means the mean embeddings differ under this configuration at this
        truncation. It does not mean the distributions differ, and the distance
        between those two readings is where a practitioner will over-claim.
    truncation:
        ``K``, the number of retained coordinates. Required by RFC-0002 §2.1.
    lipschitz:
        ``L_nu``, the point-level Lipschitz constant of the embedding. Reported
        because ``transport_lower_bound`` is uninterpretable without it.
    n_a, n_b, n_eff:
        Inference-split group sizes and ``n_eff = m * m' / (m + m')``.
    pilot_size:
        Diagrams spent fixing ``nu``. Not available for inference.

    Notes
    -----
    Two fields are deliberately absent while RFC-0002 D6 and D7 are open:
    the structured-exclusion verdict from the interval's *upper* endpoint, and
    the truncation error ``epsilon_K``. Both are specified in RFC-0002 §3.6 and
    §4.1 and neither is settled. They are named here rather than added
    provisionally, because a field that appears and then changes meaning is
    worse than one that arrives late.
    """

    delta_lower: float
    delta_upper: float
    transport_lower_bound: float
    transport_bound_is_informative: bool
    calibrations: tuple[Calibration, ...]
    estimand: str
    truncation: int
    lipschitz: float
    n_a: int
    n_b: int
    n_eff: float
    pilot_size: int


def two_sample(
    group_a: DiagramBatch,
    group_b: DiagramBatch,
    *,
    alpha: float = 0.05,
    nu: Any | None = None,
    truncation: int | None = None,
    pilot_fraction: float = DEFAULT_PILOT_FRACTION,
    calibration: Literal["spectral", "permutation", "chebyshev", "all"] = "all",
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    seed: int | None = None,
) -> TwoSampleResult:
    """Test whether two groups of persistence diagrams differ in mean embedding.

    Parameters
    ----------
    group_a, group_b:
        The two groups. Single-filtration only; see Notes.
    alpha:
        Confidence and significance level.
    nu:
        The embedding configuration. If ``None`` it is fitted on the pilot
        split, which is the default protocol (RFC-0002 §2.2). Passing a ``nu``
        that was fitted on the inference split raises
        :class:`InferenceSampleFitError`.
    truncation:
        ``K``. Selected on the pilot if ``None``. **Selecting it by minimising a
        p-value or maximising separation on the inference sample generally
        invalidates the fixed-K calibration** (Paper III,
        ``sec:truncation-practical-protocol``), which is why it is a pilot
        quantity and not a tuning knob.
    pilot_fraction:
        Fraction of each group spent fixing ``nu`` and ``K``.
    calibration:
        Which calibrations to run. ``"all"`` runs the three and returns all
        three, each labelled with the null it covers.
    n_permutations:
        Permutation null size. The smallest attainable p-value is
        ``1 / (n_permutations + 1)``.
    seed:
        Seeds the split, the pilot fit and the permutation null.

    Returns
    -------
    TwoSampleResult

    Raises
    ------
    InferenceSampleFitError
        If ``nu`` was fitted on data that is also in the inference split.

    Notes
    -----
    **Read the interval before the p-value.** Paper III calls the confidence
    interval for the mean-embedding separation the primary inferential output,
    and this API orders the result object accordingly.

    **The three calibrations are not interchangeable.** Spectral is asymptotic
    and covers the mean null. Chebyshev is conservative and is the only one with
    a finite-sample level guarantee. Permutation is exact under ``P == Q`` and
    **does not generally control level under the weaker null of equal mean
    embeddings**, which is this tool's own estimand.

    **Multi-filtration input is out of scope** (RFC-0002 §1.2). Routing it
    through the interchange layer means grouping one batch by
    ``DiagramMeta.params["filtration"]``, an API decision that should not be
    taken under deadline. Single-filtration results are valid on their own
    terms, simply less powerful.

    **Covariates and k-sample designs are out of scope.** A third group is not a
    loop over pairs.
    """
    raise NotImplementedError(
        "Bodies are ported once RFC-0002 §3 has been reviewed. Order: partition "
        "into pilot and inference splits; fit nu and K on the pilot; embed the "
        "inference split with the additive Phi; form delta-hat; run the "
        "requested calibrations; form the confidence interval; derive the "
        "transport bound and decide whether it is informative."
    )


def _confidence_interval(
    delta_hat: Any, radius: Any
) -> tuple[float, float]:  # pragma: no cover - scaffold
    """The interval ``[max(||delta|| - r, 0), ||delta|| + r]`` for ``Delta^Psi``.

    Paper III ``cor:two-sample-confidence-ball``. **Cite the corollary by number
    in this docstring once the arXiv numbering is transcribed.**

    The lower endpoint is clamped at zero by the corollary's own definition, not
    as a convenience: it is what makes the endpoint an effect size a caller can
    report rather than a signed quantity they have to interpret.
    """
    raise NotImplementedError


def _transport_lower_bound(
    delta_lower: float, cardinality_bound: int, lipschitz: float
) -> float:  # pragma: no cover - scaffold
    """``delta_lower / (n * L_nu)`` — the geometric effect-size certificate.

    Paper III ``prop:mean-embedding-transfer``, applied to the lower endpoint.
    **Cite the proposition by number once transcribed.**

    This is the function a referee checks first, because it is where an
    embedding-space statistic becomes a claim about the diagram metric. It is
    squarely in the human-derived tier of onboarding §10, and it is the exact
    place the previous scaffold went wrong: it computed a Lipschitz constant for
    a bound on the bottleneck distance between distributions, which does not
    exist.

    What this returns bounds ``W_inf,0`` between padded *mean measures*, at
    confidence ``1 - alpha``. Nothing about individual diagrams follows from it.
    """
    raise NotImplementedError


def _is_informative(bound: float, scale: float) -> bool:  # pragma: no cover
    """Whether the transport bound is large enough for a caller to act on.

    RFC-0002 §3.5. Paper III states that a large cardinality bound or a
    conservative Lipschitz constant makes the certificate small, and that a
    small certificate is **inconclusive** rather than evidence of closeness.

    The threshold is not yet specified and must not be invented here. Deciding
    it by picking a number that makes the chemistry results look good is exactly
    the failure this module exists to prevent, so it wants either a defensible
    scale from the paper or an explicit caller-supplied one.
    """
    raise NotImplementedError
