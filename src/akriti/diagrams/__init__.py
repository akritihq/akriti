"""`akriti.diagrams` -- the persistence diagram interchange layer. RFC-0001."""

from __future__ import annotations

from akriti.diagrams.adapters import (
    from_array,
    from_giotto,
    from_gudhi,
    from_persim,
    from_ripser,
    to_arrays,
    to_csv,
    to_parquet,
)
from akriti.diagrams.core import Array, DiagramBatch, DiagramMeta, PersistenceDiagram
from akriti.diagrams.io import load, save

__all__ = [
    "Array",
    "DiagramBatch",
    "DiagramMeta",
    "PersistenceDiagram",
    "from_array",
    "from_giotto",
    "from_gudhi",
    "from_persim",
    "from_ripser",
    "load",
    "save",
    "to_arrays",
    "to_csv",
    "to_parquet",
]
