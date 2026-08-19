"""CASTLE Tool 1 — two-sample testing on persistence diagrams.

Answers the question this library exists for: *given two groups of samples, do
they differ topologically, and can I defend the answer?*

The test embeds each diagram with PALACE, compares the two groups by
embedding-space MMD, calibrates against a permutation null, and — the part no
other Python library offers — converts a rejection in embedding space into a
**certified lower bound on the bottleneck distance** between the underlying
diagram distributions, via the bidirectional bound of Papers I-II.

Ported from ``PESOSE-27/stat-papers/experiments/exp_orbit5k_two_sample_demo.py``.
That script remains the reference for the numerical content; this module is its
library form, not a reimplementation.

NumPy-backed by decision: see onboarding §9's dated deviation of 2026-08-09.
``diagrams/`` remains array-API-pure; ``core/`` and ``castle/`` land on NumPy
until ``torch.Tensor`` implements ``__array_namespace__`` (RFC-0001 D18).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from akriti.diagrams import DiagramBatch

__all__ = ["StructuralAxisWarning", "TwoSampleResult", "two_sample"]


# Defaults from the standard PALACE configuration used throughout Papers I-IV.
# These are NOT tuned-for-you values -- see the guardrail note in `two_sample`.
DEFAULT_K = 200  # landmarks per filtration
DEFAULT_SIGMA = 1e-3  # kernel bandwidth
DEFAULT_SHRINK = 1.75  # alpha, the non-uniform shrink exponent
DEFAULT_N_PERMUTATIONS = 1000


class StructuralAxisWarning(UserWarning):
    """Raised when a parameter is varied along an axis where closed-form
    selection is provably unreliable (Paper III; onboarding §7)."""


@dataclass(frozen=True)
class TwoSampleResult:
    """Outcome of a two-sample test, with everything needed to report it.

    Attributes
    ----------
    statistic:
        Embedding-space MMD, ``T = ||mu_A - mu_B||_H``. Not comparable across
        different ``(K, sigma)`` settings -- it lives in the embedding space
        those parameters define.
    p_value:
        Permutation p-value, computed as ``(#{null >= observed} + 1) / (n + 1)``.
        The ``+1`` correction is deliberate: it keeps the test valid at finite
        permutation counts rather than admitting p-values of exactly zero.
    reject:
        Whether ``p_value < alpha``. Recorded so a report does not have to
        restate the threshold.
    certified_bottleneck:
        Lower bound on the bottleneck distance between the two diagram
        distributions, ``T / L_phi``. **This is the claim the library exists to
        make**: a rejection in embedding space, converted into a guarantee about
        the diagram metric itself. Valid only when ``reject`` is True -- a bound
        derived from a non-significant statistic certifies nothing.
    lipschitz_constant:
        ``L_phi``, the embedding's Lipschitz constant at this ``(K, sigma)``.
        Reported because ``certified_bottleneck`` is meaningless without it.
    n_a, n_b:
        Group sizes. Small groups make the permutation null coarse; see
        `akriti.castle.sample_size` (Tool 2) for what is detectable at a given n.
    embedding_dim:
        Total embedding dimension across filtrations.
    """

    statistic: float
    p_value: float
    reject: bool
    certified_bottleneck: float
    lipschitz_constant: float
    n_a: int
    n_b: int
    embedding_dim: int


def two_sample(
    group_a: DiagramBatch,
    group_b: DiagramBatch,
    *,
    alpha: float = 0.05,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    landmarks: int = DEFAULT_K,
    sigma: float = DEFAULT_SIGMA,
    shrink: float = DEFAULT_SHRINK,
    domain_scale: float | None = None,
    seed: int | None = None,
) -> TwoSampleResult:
    """Test whether two groups of persistence diagrams differ.

    Parameters
    ----------
    group_a, group_b:
        The two groups. Both must be single-filtration for now; multi-filtration
        input (PALACE's usual configuration concatenates across filtrations) is
        not yet supported -- see the note below.
    alpha:
        Significance threshold. Only used to populate ``reject``; the p-value is
        returned regardless, and reporting it is preferable to reporting a bare
        accept/reject.
    n_permutations:
        Permutation null size. The smallest attainable p-value is
        ``1 / (n_permutations + 1)``, so testing at alpha = 0.05 needs at least
        19 and is uncomfortable below a few hundred.
    landmarks, sigma, shrink:
        PALACE configuration. **Structural axes -- read the guardrail below.**
    domain_scale:
        ``L``, the embedding domain bound. Defaults to ``1.05 x`` the largest
        finite death across both groups, matching the reference script. Passed
        explicitly when comparing results across datasets, since a data-derived
        scale makes two runs incommensurable.
    seed:
        Seeds both landmark placement and the permutation null.

    Returns
    -------
    TwoSampleResult

    Guardrail: ``landmarks`` and ``sigma`` are structural axes
    -----------------------------------------------------------
    Paper III proves that closed-form selection via ``delta_hat / sqrt(l)`` is
    reliable only on *within-structure* axes. ``landmarks`` (K) and ``sigma``
    are **not** among them: structural bias can dominate the signal and reverse
    the ranking.

    So the defaults here are a *starting configuration*, not a recommendation
    for your data, and the following is not a valid way to choose them::

        for k in (50, 100, 200, 400):          # DON'T
            r = two_sample(a, b, landmarks=k)  # and keep whichever rejects

    That procedure has no error control and Paper III shows it can invert the
    true ordering. If these need choosing for your data, cross-validate over a
    held-out split. `akriti.core.selectors` will carry the diagnostic; until it
    lands, treat a result whose significance depends on K or sigma as
    uninterpreted rather than as a finding.

    Warns
    -----
    StructuralAxisWarning
        If ``sigma`` or ``landmarks`` is given a non-default value, as a
        reminder that the axis is unguarded. Silence with
        ``warnings.filterwarnings`` once you have read the above.

    Notes
    -----
    Multi-filtration input is the reference configuration and is deliberately
    excluded from this first version. The reference script concatenates
    embeddings across two filtrations before forming the Gram matrix; doing that
    through the interchange layer means grouping one batch by
    ``DiagramMeta.params["filtration"]``, which is an API decision that should
    not be taken under deadline. Single-filtration results are valid on their
    own terms, simply less powerful.
    """
    if landmarks != DEFAULT_K or sigma != DEFAULT_SIGMA:
        warnings.warn(
            "landmarks and sigma are structural axes: closed-form selection is "
            "provably unreliable on them (Paper III). If you are choosing these "
            "by comparing outcomes, the result has no error control -- "
            "cross-validate instead. See this function's docstring.",
            StructuralAxisWarning,
            stacklevel=2,
        )

    raise NotImplementedError(
        "Port from exp_orbit5k_two_sample_demo.run_pair. Order: derive "
        "domain_scale if None, embed both groups via "
        "embedding.nonuniform.init_nonuniform_from_data, form the Gram matrix, "
        "compute observed MMD, run the permutation null, then the certificate."
    )


def _weighted_linear_gram(x: np.ndarray, sigma: float) -> np.ndarray:
    """Weighted linear kernel Gram matrix over embedded diagrams.

    Port of ``exp_orbit5k_two_sample_demo.wlk_gram``. Cite the kernel's
    definition by equation number from Paper II in this docstring when porting.

    Parameters
    ----------
    x:
        ``(n_samples, embedding_dim)`` embedded diagrams.
    sigma:
        Kernel bandwidth.

    Returns
    -------
    ``(n_samples, n_samples)`` Gram matrix.
    """
    raise NotImplementedError


def _mmd2_from_gram(gram: np.ndarray, idx_a: np.ndarray, idx_b: np.ndarray) -> float:
    """Biased MMD-squared between two index sets of a shared Gram matrix.

    Port of ``exp_orbit5k_two_sample_demo.mmd_stat_from_gram``. State in this
    docstring whether the estimator is biased or unbiased and cite the equation;
    the permutation null is valid either way, but a reader comparing against a
    published value needs to know which.

    Returns a squared quantity -- callers take the square root, clamping at zero,
    since the biased estimator can go slightly negative.
    """
    raise NotImplementedError


def _lipschitz_upper(embedding_dim: int, sigma: float, shrink: float) -> float:
    """Upper bound on the PALACE embedding's Lipschitz constant, ``L_phi``.

    Port of ``exp_orbit5k_two_sample_demo.lipschitz_upper``. **Cite the theorem
    by number from Paper I or II in this docstring.** This is the function that
    turns an embedding-space statistic into a claim about the bottleneck metric,
    so it is the one a referee will check first, and it is squarely in the
    human-derived tier of onboarding §10.

    Being an *upper* bound on L_phi makes ``T / L_phi`` a *lower* bound on the
    bottleneck distance -- the conservative direction. A tighter L_phi is a
    stronger claim; a wrong one invalidates every certificate the library emits.
    """
    raise NotImplementedError
