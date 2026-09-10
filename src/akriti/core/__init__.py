"""Shared numerical primitives: embeddings, selectors, distances.

`core` holds what more than one product needs. The landmark embedding of PLACE
and PALACE lives here rather than in per-paper modules because `akriti.castle`
consumes it and does not define it, and `core/distances.py` — specified by
RFC-0001 §9.1 and not yet written — is the other resident.

Holding a :class:`Configuration` and embedding with it needs numpy alone.
Fitting one needs `akriti[core]`.
"""

from __future__ import annotations

from akriti.core.embedding import Configuration, fit_configuration

__all__ = ["Configuration", "fit_configuration"]
