"""Akriti — the statistically grounded entry point to topological data analysis.

Akriti delegates computation to the ecosystem's existing engines (GUDHI, Ripser,
persim) and supplies natively the layer Python lacks: statistical inference on
persistence diagrams.

This package is in early development. There is no stable API yet, and the
public surface below is a placeholder while `akriti.diagrams` is implemented
against RFC-0001.

`pip install akriti` installs no third-party package at all — not a
persistence backend, and not numpy. `akriti.diagrams` works against whatever
array library the caller already has, through the Python array API standard
(RFC-0001 §3.3), so it has nothing to install on its behalf. Install a
backend when you need one::

    pip install akriti[rips]        # Ripser
    pip install akriti[alpha]       # GUDHI  (GPLv3 -- see DEPENDENCIES.md)
    pip install akriti[distances]   # persim
    pip install akriti[torch]       # tensor backend, for the NN path
    pip install akriti[bio]         # anndata interop

See https://github.com/akritihq/akriti/tree/main/rfcs for the specifications
the implementation is written against.
"""

from __future__ import annotations

__version__ = "0.0.1.dev0"

__all__ = ["__version__"]
