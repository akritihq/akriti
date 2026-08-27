"""Akriti — the statistically grounded entry point to topological data analysis.

Akriti delegates computation to the ecosystem's existing engines (GUDHI, Ripser,
persim) and supplies natively the layer Python lacks: statistical inference on
persistence diagrams.

This package is in early development. There is no stable API yet, and the
public surface below is a placeholder while `akriti.diagrams` is implemented
against RFC-0001.

`pip install akriti` installs no third-party package at all — not a
persistence backend, and not numpy. Native array inputs retain the caller's
Python array API namespace; lazy extras cover inputs with no namespace and
backends that need a compatibility resolver (RFC-0001 §3.3). Install only the
boundary you need::

    pip install akriti[rips]        # Ripser
    pip install akriti[alpha]       # GUDHI  (GPLv3 -- see DEPENDENCIES.md)
    pip install akriti[distances]   # persim
    pip install akriti[numpy]       # NumPy array namespace and row fallback
    pip install akriti[io]          # .akd save/load (includes NumPy)
    pip install akriti[parquet]     # to_parquet() (PyArrow, Apache-2.0)
    pip install akriti[torch]       # tensor backend + array-api-compat
    pip install akriti[bio]         # anndata interop

Array inputs keep their own array namespace. The accepted Python-row adapter
forms have no namespace to preserve, so they lazily use `akriti[numpy]`;
torch tensors use the lazy `array-api-compat` fallback from `akriti[torch]`.
Parquet imports PyArrow only when `to_parquet()` is called. See
https://github.com/akritihq/akriti/tree/main/rfcs for the specifications the
implementation is written against.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
