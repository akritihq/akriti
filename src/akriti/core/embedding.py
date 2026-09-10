"""Landmark embeddings — the interface between diagrams and Hilbert space.

A :class:`Configuration` is a fixed set of landmarks. Embedding a diagram
through one turns a multiset of intervals into a vector, and every statistical
claim `akriti.castle` makes is a claim about those vectors, transferred back to
diagram space through the additive interface (Paper III,
``prop:mean-embedding-transfer``).

**The split in this module is between holding a configuration and fitting one.**
:meth:`Configuration.embed` needs numpy alone. :func:`fit_configuration` needs
scipy, so it lives behind ``akriti[core]`` and imports it lazily, on RFC-0001
§10.1 requirement 2's terms. A caller handed a fitted configuration — by a
collaborator, or out of a file — can embed with it on a default install.

**The numerics are not written here yet.** This module defines the type and the
boundary; the bodies port from ``PESOSE-27/stat-papers/embedding/nonuniform.py``
and are human-derived tier under onboarding §10.

Papers: PLACE (arXiv:2605.02836), PALACE (arXiv:2605.04046), and Paper III
(arXiv:2609.07691) for what the embedding is required to satisfy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

__all__ = ["Configuration", "fit_configuration"]


@dataclass(frozen=True)
class Configuration:
    """A fixed landmark configuration, and the provenance of how it was fixed.

    Frozen because every guarantee downstream is stated for a configuration
    held fixed across the inference split. A configuration that can be mutated
    after a diagram has been embedded through it silently invalidates the
    interval computed from those embeddings, and nothing would report it —
    which is I8's argument in RFC-0001 §3.1, reaching this layer unchanged.

    Attributes
    ----------
    positions:
        ``(K, 2)`` landmark positions in the diagram frame.
    radii:
        ``(K,)`` landmark radii, positive.
    weights:
        ``(K,)`` configuration weights, with unit sum of squares.
    frame:
        ``L``, the frame size. Diagram coordinates are read relative to it, so
        two configurations with different frames are not comparable and neither
        are results computed under them.
    truncation:
        ``K``, the number of retained coordinates. Carried because RFC-0002
        §2.1 requires a result to say which estimand it refers to: the
        confidence interval is for the separation *in these coordinates*, not
        for an unqualified distance between populations.
    fitted_on:
        How this configuration was obtained. RFC-0002 §2.3 requires an
        implementation to **raise** when a configuration would be fitted on the
        inference sample, and a bare bundle of arrays cannot answer the
        question that rule asks. A content hash of the pilot diagrams is the
        intended spelling — the same instrument §8 uses — so that the check is
        mechanical rather than a matter of the caller remembering.
    """

    positions: Any
    radii: Any
    weights: Any
    frame: float
    truncation: int
    fitted_on: str

    def embed(self, diagram: Any) -> np.ndarray:
        """Embed one diagram, returning a ``(K,)`` vector. NumPy only.

        Ports from ``NonUniformEmbedding.embed``.
        """
        raise NotImplementedError(
            "Port from embedding/nonuniform.py: NonUniformEmbedding.embed, "
            "with hard_coords and soft_coords beneath it."
        )

    def embed_batch(self, batch: Any) -> np.ndarray:
        """Embed a ``DiagramBatch``, returning ``(n_diagrams, K)``.

        Ports from ``NonUniformEmbedding.embed_dataset``, taking a
        `DiagramBatch` (RFC-0001 §4) rather than a list of arrays: the batch is
        the interchange layer's own container and §4 requires numerical
        functions to take a leading batch dimension rather than loop.
        """
        raise NotImplementedError(
            "Port from embedding/nonuniform.py: NonUniformEmbedding.embed_dataset."
        )


def fit_configuration(
    pilot: Any,
    *,
    truncation: int,
    frame: float,
    tau: float | None = None,
    seed: int | None = None,
) -> Configuration:
    """Fit a configuration on a **pilot** split. Requires ``akriti[core]``.

    Ports from ``init_nonuniform_from_data``, whose class-aware farthest-point
    sampling is the one part of this module needing scipy.

    The argument is named ``pilot`` rather than ``diagrams`` deliberately.
    RFC-0002 §2.2 makes the pilot/inference split the default protocol, and
    §2.3 prohibits fitting on the inference sample outright — the parameter
    name is the first place a caller can be told which half this takes.
    """
    raise NotImplementedError(
        "Port from embedding/nonuniform.py: init_nonuniform_from_data, with "
        "farthest_point_sampling beneath it. scipy is imported lazily inside "
        "the body, never at module scope (RFC-0001 §10.1 requirement 2)."
    )
