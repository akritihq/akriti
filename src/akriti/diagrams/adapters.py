"""`from_*` adapters: backend output in, `PersistenceDiagram` out. RFC-0001 §11.

Five adapters, one per source this project supports:

    from_gudhi(obj, **meta)   -> PersistenceDiagram
    from_ripser(obj, **meta)  -> PersistenceDiagram
    from_giotto(arr, *, reduced_homology, **meta)  -> DiagramBatch
    from_persim(obj, **meta)  -> PersistenceDiagram
    from_array(arr, **meta)   -> PersistenceDiagram

`from_giotto` deviates twice, both deliberately (§11): `reduced_homology` is a
required keyword-only argument, because omitting it MUST be a `TypeError` at
the call site rather than a value that slips past inside `**meta` (§5.1); and
its return type is fixed at `DiagramBatch`, length one for a single sample,
because nothing about an adapter's return type may depend on how many samples
the particular call happened to carry (§4).

**What every adapter does** (§11): validates against §3.1 -- by construction,
since `PersistenceDiagram` refuses to exist otherwise; populates `backend`,
`backend_version` and `provenance`; preserves backend row order; and never
finitizes, sorts, or deduplicates. What it converts is *dtype* (§6.1), never
namespace (§3.3): a diagram built from JAX arrays stays JAX-backed.

**Assumptions this module makes about its input.**

- The object handed in is what the named backend actually returned. An
  adapter cannot verify that, and §11.2 is why the test suite runs against
  real backend output rather than hand-written arrays.
- `**meta` is §8's field set. `backend` and `backend_version` are refused:
  they are the adapter's to record, and a caller who could set them could
  produce a diagram that lies about where it came from.
- Where a backend takes a coefficient field and the caller does not pass one
  on, the backend's documented default is recorded as an assumption and
  marked as one (§11, §9.3). A caller who set a non-default field on the
  backend and did not pass it here gets the default recorded. **Pass
  `coeff_field=` whenever you set it on the backend.**

**Imports.** Standard library only (§3.3, §10.1 requirement 2), with one
exception: an input carrying no array at all -- GUDHI's `persistence()` list,
an empty diagram list -- has no namespace to derive, and §11 fixes the
signatures with no namespace argument to derive one from instead. numpy is
imported lazily on that path alone, function-scoped, and is unreachable
without an input shaped that way. See `_namespace_for_rows`.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from importlib import import_module, metadata
from typing import Any

from akriti.diagrams.core import (
    Array,
    DiagramBatch,
    DiagramMeta,
    PersistenceDiagram,
)

__all__ = [
    "from_array",
    "from_giotto",
    "from_gudhi",
    "from_persim",
    "from_ripser",
]

# §9.3, Appendix A.5: the two backends compute over different fields by
# default and neither returns the field it used, so these numbers are what an
# adapter records for a caller who stated none. They are load-bearing and
# asserted against the installed backend in `test_rfc0001_adapters_live.py`:
# an upstream change must break the build rather than reach a user's
# provenance.
_GUDHI_DEFAULT_COEFF_FIELD = 11
_RIPSER_DEFAULT_COEFF_FIELD = 2

# §3.1: "Observed floating-point violations are a real occurrence at the 1e-16
# level in some filtration code; the adapter (not the core type) is the
# correct place to clamp, and it MUST warn when it does." The RFC fixes no
# threshold, so one is fixed here and stated: a `death < birth` gap is
# absorbed only when it is within floating-point noise of zero, scaled to the
# magnitude of the birth value, since noise is relative -- a 1e-10 gap at 1e6
# is the same defect as a 1e-16 gap at 1.0. Anything larger is a backend bug
# and reaches §3.1's I6 check unmodified, which reports its magnitude.
#
# The floor sits four orders of magnitude above float64 epsilon (~2.2e-16),
# which absorbs accumulated rounding through a filtration without coming near
# any persistence value a diagram would be read for.
_CLAMP_RTOL = 1e-12
_CLAMP_ATOL = 1e-12


# ---------------------------------------------------------------------------
# Namespace and dtype
# ---------------------------------------------------------------------------


def _namespace_of(x: Array) -> Any:
    """The array's namespace (§3.3).

    Spelled as a direct `__array_namespace__` call, matching `core.py`. §3.3's
    resolution rule adds an `array_api_compat` fallback for backends that
    implement the standard without declaring it (torch alone, D18); that
    resolver does not exist yet, and adding it here alone would produce
    diagrams whose namespace `core.py` cannot re-derive. Both move together;
    see TODO.md.
    """
    return x.__array_namespace__()


def _has_namespace(x: Any) -> bool:
    return hasattr(x, "__array_namespace__")


def _namespace_for_rows() -> Any:
    """numpy, for an input carrying no array to derive a namespace from.

    Reached only by GUDHI's `persistence()` list form and by empty diagram
    lists -- inputs with no array anywhere -- so no caller reaches this
    without either having installed a backend that already depends on numpy,
    or having passed a bare Python list.

    The version is checked, not merely presence, for D6's reason: numpy below
    2.0 has no main-namespace array API, so a presence-only import would
    proceed to an `AttributeError` on the first namespace call instead of
    saying what is wrong. Both failures name the extra rather than the bare
    package -- "install numpy" is not an instruction a user who already has
    numpy 1.24 can act on.

    Spelled `import_module("numpy")` rather than `import numpy` so that the
    type checker does not resolve a package this one does not depend on. A
    plain import makes numpy's stubs part of every `mypy` run, where they fail
    to parse against this project's 3.10 floor -- a third-party syntax error
    standing between us and type-checking our own code.
    """
    try:
        numpy = import_module("numpy")
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "this input carries no array to derive an array namespace from, "
            "so building one needs numpy, which is not installed. Install "
            "`akriti[numpy]`, or pass an array instead of a Python list."
        ) from exc

    probe = numpy.empty(0)
    if not hasattr(probe, "__array_namespace__"):  # pragma: no cover - old numpy
        raise ImportError(
            f"numpy {numpy.__version__} has no main-namespace array API "
            "(`__array_namespace__`), which arrived in numpy 2.0, so it "
            "cannot back a diagram. Install `akriti[numpy]`, which declares "
            "the >=2.0 floor."
        )
    return probe.__array_namespace__()


def _as_float64(column: Array, xp: Any) -> Array:
    """I2, §6.1: storage is the namespace's own `float64`, whatever arrived."""
    return xp.astype(column, xp.float64)


def _as_dims(column: Array, xp: Any) -> Array:
    """I2, I3: the degree column as `int32`, refusing anything non-integral.

    A degree of 1.5 is not a homological degree, and `astype` would silently
    truncate it to 1 -- a diagram that is clean, plausible and wrong, which is
    §9's category. Non-finite values are refused first, `astype` on `inf` or
    `nan` being undefined rather than merely wrong.
    """
    if xp.isdtype(column.dtype, "integral"):
        return xp.astype(column, xp.int32)

    if not bool(xp.all(xp.isfinite(column))):
        raise ValueError(
            "the degree column contains a non-finite value; a homological "
            "degree is a non-negative integer (I3)"
        )
    if not bool(xp.all(column == xp.round(column))):
        raise ValueError(
            "the degree column contains a non-integral value; a homological "
            "degree is a non-negative integer (I3)"
        )
    return xp.astype(column, xp.int32)


def _clamp_i6(births: Array, deaths: Array, xp: Any) -> tuple[Array, int]:
    """Repair `death < birth` rows that are floating-point noise. §3.1.

    Returns the (possibly repaired) deaths and the number of rows repaired,
    which the caller records as `provenance["clamped_rows"]` (§8). Warns when
    it repairs anything: §3.1 requires it, and a silent repair is a backend
    defect absorbed without trace.

    Violations larger than `_CLAMP_ATOL + _CLAMP_RTOL * |birth|` are left
    exactly as they arrived, so §3.1's I6 check raises on them and names the
    magnitude -- "a backend that returns `death < birth` has a bug ... and we
    surface it rather than absorb it".
    """
    # Only a row with two finite coordinates can carry an I6 violation worth
    # repairing: an `inf` death is an essential bar (§5) and violates nothing,
    # and any other non-finite value violates I4 or I5, which are core's to
    # refuse rather than the adapter's to absorb. Both operands are masked to
    # zero rather than subtracted and filtered afterwards, because `inf - inf`
    # is `nan` and raises an invalid-operation warning on the way.
    comparable = xp.isfinite(births) & xp.isfinite(deaths)
    zero = xp.zeros_like(births)
    gap = xp.where(comparable, births, zero) - xp.where(comparable, deaths, zero)
    tolerance = _CLAMP_ATOL + _CLAMP_RTOL * xp.abs(xp.where(comparable, births, zero))
    repair = (gap > 0.0) & (gap <= tolerance)
    n_repaired = int(xp.sum(xp.astype(repair, xp.int64)))
    if n_repaired == 0:
        return deaths, 0

    warnings.warn(
        f"clamped {n_repaired} of {int(births.shape[0])} rows where "
        f"death < birth (I6), the largest by {float(xp.max(gap)):.3g}. These "
        "are within floating-point noise of zero and are treated as "
        "filtration rounding (RFC-0001 §3.1); provenance['clamped_rows'] "
        "records the count.",
        UserWarning,
        stacklevel=3,
    )
    return xp.where(repair, births, deaths), n_repaired


# ---------------------------------------------------------------------------
# Metadata assembly (§8)
# ---------------------------------------------------------------------------


def _build_meta(
    *,
    backend: str,
    backend_version: str | None,
    provenance: dict[str, Any],
    params: dict[str, Any] | None = None,
    meta: dict[str, Any],
) -> DiagramMeta:
    """Merge the adapter's recorded facts with the caller's metadata. §8.

    The caller's `provenance` and `params` are kept, and a key the adapter
    measured wins over a caller's key of the same name: the adapter is the one
    party that saw the backend's output, and `provenance` exists to be
    auditable rather than assertable.

    `backend` and `backend_version` are refused outright rather than merged.
    An unknown field raises `TypeError` from `DiagramMeta` itself, naming the
    field, so a misspelled `filtraton=` cannot vanish into a diagram that
    reports nothing.
    """
    for reserved in ("backend", "backend_version"):
        if reserved in meta:
            raise TypeError(
                f"{reserved!r} is recorded by the adapter and cannot be "
                "passed in: it is the fact that says where this diagram came "
                "from (RFC-0001 §8, §11)"
            )

    caller_provenance = dict(meta.pop("provenance", {}) or {})
    caller_params = dict(meta.pop("params", {}) or {})
    caller_provenance.update(provenance)
    caller_params.update(params or {})

    return DiagramMeta(
        backend=backend,
        backend_version=backend_version,
        params=caller_params,
        provenance=caller_provenance,
        **meta,
    )


def _installed_version(distribution: str) -> str | None:
    """The backend's version, from installed distribution metadata. §8.

    Read rather than imported: §3.3 keeps this module to the standard library,
    and an adapter that imported its backend to ask its version would make
    every backend a dependency of every call. `None` where the distribution is
    absent -- which is ordinary, since a diagram can be adapted from output
    that was captured elsewhere (§11.2's frozen fixtures are exactly that).
    """
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _coeff_field(meta: dict[str, Any], default: int) -> tuple[int, str]:
    """D17, §11: the coefficient field and where it came from.

    The caller's value with `"caller"` if one arrived, the backend's
    documented default with `"backend_default"` otherwise. The second is an
    assumption and the source key is what marks it as one: no backend returns
    the field it computed with (A.5), so an adapter cannot verify that the
    caller left the default in place.
    """
    stated = meta.pop("coeff_field", None)
    if stated is None:
        return default, "backend_default"
    return stated, "caller"


# ---------------------------------------------------------------------------
# Shared construction
# ---------------------------------------------------------------------------


def _diagram_from_columns(
    *,
    dims: Array,
    births: Array,
    deaths: Array,
    xp: Any,
    backend: str,
    backend_version: str | None,
    provenance: dict[str, Any],
    params: dict[str, Any] | None = None,
    meta: dict[str, Any],
) -> PersistenceDiagram:
    """Convert dtypes, clamp §3.1's noise, and construct. Order is preserved."""
    births = _as_float64(births, xp)
    deaths = _as_float64(deaths, xp)
    deaths, clamped = _clamp_i6(births, deaths, xp)

    recorded = {"clamped_rows": clamped, **provenance}
    return PersistenceDiagram(
        dims=_as_dims(dims, xp),
        births=births,
        deaths=deaths,
        meta=_build_meta(
            backend=backend,
            backend_version=backend_version,
            provenance=recorded,
            params=params,
            meta=meta,
        ),
    )


def _columns_from_table(arr: Array, xp: Any, *, dim: int | None) -> tuple[Array, ...]:
    """Split an `(n, 2)` or `(n, 3)` array into `(dims, births, deaths)`. §11.

    `(n, 3)` columns are `(birth, death, dim)` -- giotto's order, matched
    deliberately (§11). `(n, 2)` carries no degree, so one must be stated:
    guessing 0 would fabricate the single fact the array does not hold.
    """
    if arr.ndim != 2 or arr.shape[1] not in (2, 3):
        raise ValueError(
            "expected an array of shape (n, 2) or (n, 3) (RFC-0001 §11); got "
            f"shape {tuple(arr.shape)}"
        )

    if arr.shape[1] == 3:
        if dim is not None:
            raise ValueError(
                "an (n, 3) array already carries a degree column, so dim= "
                "would be a second source for one fact; drop it, or pass an "
                "(n, 2) array"
            )
        return arr[:, 2], arr[:, 0], arr[:, 1]

    if dim is None:
        raise TypeError(
            "an (n, 2) array carries no homological degree; pass dim=<k> "
            "(RFC-0001 §11). Degree 0 is not a safe guess -- it is the one "
            "fact the array does not hold."
        )
    return xp.full((arr.shape[0],), int(dim), dtype=xp.int32), arr[:, 0], arr[:, 1]


def _columns_from_degree_list(
    dgms: Sequence[Any], xp: Any
) -> tuple[Array, Array, Array]:
    """Stack `list[(n, 2)]` where list position is the degree. §11.

    Ripser's `dgms` and persim's input share this shape. Row order within a
    degree is preserved exactly; degrees follow the list, which is the
    backend's own order for the merged diagram.
    """
    dim_blocks, birth_blocks, death_blocks = [], [], []
    for degree, block in enumerate(dgms):
        block = block if _has_namespace(block) else xp.asarray(block, dtype=xp.float64)
        if block.ndim != 2 or block.shape[1] != 2:
            raise ValueError(
                f"diagram at index {degree} must have shape (n, 2) "
                f"(RFC-0001 §11); got shape {tuple(block.shape)}"
            )
        dim_blocks.append(xp.full((block.shape[0],), degree, dtype=xp.int32))
        birth_blocks.append(_as_float64(block[:, 0], xp))
        death_blocks.append(_as_float64(block[:, 1], xp))

    if not dim_blocks:
        return (
            xp.asarray([], dtype=xp.int32),
            xp.asarray([], dtype=xp.float64),
            xp.asarray([], dtype=xp.float64),
        )
    return (
        xp.concat(dim_blocks),
        xp.concat(birth_blocks),
        xp.concat(death_blocks),
    )


def _source_dtype(arr: Any) -> dict[str, str]:
    """§8's `source_dtype`, as a string because §8 requires JSON values.

    Empty where the input carried no array -- a Python list of tuples has no
    dtype, and inventing `"float64"` for one would record a fact about our own
    conversion rather than about the input.
    """
    return {"source_dtype": str(arr.dtype)} if _has_namespace(arr) else {}


# ---------------------------------------------------------------------------
# §11's five adapters
# ---------------------------------------------------------------------------


def from_gudhi(obj: Any, *, dim: int | None = None, **meta: Any) -> PersistenceDiagram:
    """A GUDHI persistence result as a `PersistenceDiagram`. §11.

    Accepts both measured input forms (§11):

    - `SimplexTree.persistence()` -> `list[(dim, (birth, death))]`, carrying
      every degree at once. This form has no array, so the diagram is
      numpy-backed; see the module docstring.
    - `SimplexTree.persistence_intervals_in_dimension(k)` -> `(n, 2)`, which
      states no degree, so `dim=k` is required.

    GUDHI is faithful to essential bars (§5.1, A.1): `inf` arrives as `inf`
    and is stored as `inf`, never a sentinel, so `provenance["essential_bars"]`
    and `["essential_bars_source"]` are both `"faithful"`.

    **Coefficient field.** Recorded in every construction (D17, §11): the
    value passed as `coeff_field=` with `provenance["coeff_field_source"] ==
    "caller"`, or GUDHI's documented default of 11 with `"backend_default"`.
    The second is an assumption -- GUDHI does not report the field it used
    (A.5) -- so **pass `coeff_field=` whenever you set
    `homology_coeff_field=` on the backend.**

    Assumes `obj` is GUDHI output. Row order is preserved exactly; nothing is
    sorted, deduplicated or finitized (§7, §11).
    """
    field, source = _coeff_field(meta, _GUDHI_DEFAULT_COEFF_FIELD)
    provenance = {
        "essential_bars": "faithful",
        "essential_bars_source": "faithful",
        "coeff_field_source": source,
    }

    if _has_namespace(obj):
        xp = _namespace_of(obj)
        dims, births, deaths = _columns_from_table(obj, xp, dim=dim)
        provenance.update(_source_dtype(obj))
    elif isinstance(obj, Sequence):
        if dim is not None:
            raise ValueError(
                "a persistence() list already carries a degree per bar, so "
                "dim= would be a second source for one fact"
            )
        xp = _namespace_for_rows()
        rows = list(obj)
        dims = xp.asarray([int(k) for k, _ in rows], dtype=xp.int32)
        births = xp.asarray([float(b) for _, (b, _) in rows], dtype=xp.float64)
        deaths = xp.asarray([float(d) for _, (_, d) in rows], dtype=xp.float64)
    else:
        raise TypeError(
            "from_gudhi accepts SimplexTree.persistence() output "
            "(list[(dim, (birth, death))]) or "
            "persistence_intervals_in_dimension(k) output ((n, 2), with "
            f"dim=k); got {type(obj).__name__}"
        )

    return _diagram_from_columns(
        dims=dims,
        births=births,
        deaths=deaths,
        xp=xp,
        backend="gudhi",
        backend_version=_installed_version("gudhi"),
        provenance=provenance,
        meta={"coeff_field": field, **meta},
    )


def from_ripser(obj: Any, **meta: Any) -> PersistenceDiagram:
    """A Ripser result as a `PersistenceDiagram`. §11.

    Accepts both measured input forms (§11):

    - `ripser(X)` -> `dict` carrying `"dgms"`.
    - `Rips().fit_transform(X)` -> `list[(n, 2)]`.

    **Index in the list is the degree.** Nothing else in Ripser's output
    states it, and the adapter preserves row order within each degree.

    Ripser is faithful to essential bars (§5.1, A.1), so both essential-bar
    provenance keys are `"faithful"`. Its arrays hold float32-precision values
    (§6.2), which is why cross-backend comparison needs `rtol=1e-6` rather
    than exactness -- the adapter converts dtype (§6.1) and cannot restore
    precision that was never computed.

    **Coefficient field.** Recorded in every construction (D17, §11), with
    Ripser's documented default of 2 where the caller passed none. **Pass
    `coeff_field=` whenever you set `coeff=` on the backend.**
    """
    if isinstance(obj, Mapping):
        if "dgms" not in obj:
            raise ValueError(
                "a ripser() result carries its diagrams under 'dgms'; this "
                f"mapping has keys {sorted(map(str, obj.keys()))}"
            )
        dgms: Sequence[Any] = obj["dgms"]
    elif isinstance(obj, Sequence):
        dgms = obj
    else:
        raise TypeError(
            "from_ripser accepts ripser(X) output (a dict with 'dgms') or "
            f"Rips().fit_transform(X) output (list[(n, 2)]); got "
            f"{type(obj).__name__}"
        )

    field, source = _coeff_field(meta, _RIPSER_DEFAULT_COEFF_FIELD)
    xp = (
        _namespace_of(dgms[0])
        if dgms and _has_namespace(dgms[0])
        else _namespace_for_rows()
    )
    dims, births, deaths = _columns_from_degree_list(dgms, xp)

    provenance = {
        "essential_bars": "faithful",
        "essential_bars_source": "faithful",
        "coeff_field_source": source,
    }
    if dgms and _has_namespace(dgms[0]):
        provenance.update(_source_dtype(dgms[0]))

    return _diagram_from_columns(
        dims=dims,
        births=births,
        deaths=deaths,
        xp=xp,
        backend="ripser",
        backend_version=_installed_version("ripser"),
        provenance=provenance,
        meta={"coeff_field": field, **meta},
    )


def from_persim(obj: Sequence[Any], **meta: Any) -> PersistenceDiagram:
    """A persim-shaped diagram list as a `PersistenceDiagram`. §11.

    `list[(n, 2)]`, degree by index -- the same shape as Ripser's `dgms`.

    **No claim is made about essential bars, and none about the coefficient
    field.** persim consumes diagrams and computes no homology (§5.1: "no
    opinion"), so it cannot certify that nothing was lost upstream, and §8's
    `essential_bars_source` means the verdict at computation time. Both keys
    are therefore absent rather than guessed, and §11 puts `from_persim` out
    of scope for D17 for the same reason.
    """
    if not isinstance(obj, Sequence):
        raise TypeError(
            f"from_persim accepts list[(n, 2)], degree by index; got "
            f"{type(obj).__name__}"
        )

    xp = (
        _namespace_of(obj[0])
        if obj and _has_namespace(obj[0])
        else _namespace_for_rows()
    )
    dims, births, deaths = _columns_from_degree_list(obj, xp)

    provenance = dict(_source_dtype(obj[0])) if obj and _has_namespace(obj[0]) else {}
    return _diagram_from_columns(
        dims=dims,
        births=births,
        deaths=deaths,
        xp=xp,
        backend="persim",
        backend_version=_installed_version("persim"),
        provenance=provenance,
        meta=meta,
    )


def from_array(
    arr: Array, *, dim: int | None = None, **meta: Any
) -> PersistenceDiagram:
    """A raw array as a `PersistenceDiagram`. §11.

    Two accepted shapes:

    - `(n, 2)` with an explicit `dim=k`.
    - `(n, 3)` with columns `(birth, death, dim)` -- giotto's order, matched
      deliberately (§11).

    The array's namespace is preserved (§3.3): an array from JAX gives a
    JAX-backed diagram. What is converted is dtype (§6.1, I2).

    **No backend, so no version and no essential-bar claim.**
    `backend_version` is `None`, and `provenance` records neither
    `essential_bars` nor `coeff_field_source`: nothing about a caller's array
    says what produced it, and §8's keys record what an adapter observed
    rather than what it hopes.
    """
    if not _has_namespace(arr):
        raise TypeError(
            "from_array accepts an array carrying __array_namespace__ "
            f"(RFC-0001 §3.3); got {type(arr).__name__}"
        )

    xp = _namespace_of(arr)
    dims, births, deaths = _columns_from_table(arr, xp, dim=dim)
    return _diagram_from_columns(
        dims=dims,
        births=births,
        deaths=deaths,
        xp=xp,
        backend="array",
        backend_version=None,
        provenance=dict(_source_dtype(arr)),
        meta=meta,
    )


def from_giotto(
    arr: Array,
    *,
    reduced_homology: bool,
    strip_padding: bool | None = None,
    **meta: Any,
) -> DiagramBatch:
    """A giotto-tda transform result as a `DiagramBatch`. §11, §11.1, §5.1.

    Input is `(n_samples, n_bars, 3)` with columns `(birth, death, dim)`.

    **`reduced_homology` is required, and required for a measured reason.**
    With giotto's default of `True`, exactly one H0 class -- the essential one
    -- is missing from the output, and no property of the returned array says
    so (§5.1, A.1: 39 H0 bars where GUDHI and Ripser return 40). The adapter
    cannot recover this from the array, since a filtration truncated by
    `max_edge_length` shows the same absence, so it must come from the caller.
    Pass `reduced_homology=vr.reduced_homology` off the fitted transformer.
    `provenance["essential_bars"]` is *derived* from it -- `"lost_upstream"`
    when `True`, `"faithful"` when `False` -- and `essential_bars_source` is
    set to the same value in the same construction (§8). The flag itself is
    recorded in `params`, being a raw fact of the original call (§5.1).

    **Nothing is fabricated.** The missing essential bar is not reconstructed:
    its birth is the minimum vertex birth across the cloud, which is
    structurally absent from the remaining rows, and "reconstruct as 0" is a
    coincidence of unweighted Rips rather than a property of the elder rule
    (§5.1).

    **Padding is the caller's call** (§11.1). giotto pads a batch to a common
    row count with rows of the form `(b, b, dim)`, which are byte-identical to
    genuine zero-persistence bars (§4, A.2), so the adapter must not guess:

    - `strip_padding=None` (default): keep every row, warn once if any trivial
      rows are present, record `padding_removed = 0`.
    - `strip_padding=True`: drop trivial rows, record how many.
    - `strip_padding=False`: keep silently, record `padding_removed = 0`
      regardless of how many trivial rows are present -- the key records what
      was removed, never what was merely observed.

    **Always a `DiagramBatch`**, of length one when `n_samples == 1` (§11).
    Unwrap explicitly with `batch[0]`; an adapter whose return type depended
    on the data would move §4's shape-depends-on-what-else-was-there hazard
    into the type system.

    **No coefficient field is recorded**, unlike `from_gudhi` and
    `from_ripser`: A.5 could not measure giotto's default (§9.2), and this
    project does not assert a backend default it has not measured (§11).
    """
    if not _has_namespace(arr):
        raise TypeError(
            "from_giotto accepts an array carrying __array_namespace__ "
            f"(RFC-0001 §3.3); got {type(arr).__name__}"
        )
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            "giotto output has shape (n_samples, n_bars, 3) (RFC-0001 §11); "
            f"got shape {tuple(arr.shape)}. A single sample is "
            "arr[None, :, :] -- the adapter will not guess which axis is "
            "which."
        )

    xp = _namespace_of(arr)
    essential_bars = "lost_upstream" if reduced_homology else "faithful"
    version = _installed_version("giotto-tda")

    diagrams = []
    trivial_seen = 0
    for i in range(int(arr.shape[0])):
        sample = arr[i]
        births, deaths, dims = sample[:, 0], sample[:, 1], sample[:, 2]

        trivial = births == deaths
        n_trivial = int(xp.sum(xp.astype(trivial, xp.int64)))
        trivial_seen += n_trivial

        removed = 0
        if strip_padding:
            keep = ~trivial
            births, deaths, dims = births[keep], deaths[keep], dims[keep]
            removed = n_trivial

        diagrams.append(
            _diagram_from_columns(
                dims=dims,
                births=births,
                deaths=deaths,
                xp=xp,
                backend="giotto",
                backend_version=version,
                provenance={
                    "essential_bars": essential_bars,
                    "essential_bars_source": essential_bars,
                    "padding_removed": removed,
                    **_source_dtype(arr),
                },
                params={"reduced_homology": bool(reduced_homology)},
                meta=dict(meta),
            )
        )

    if strip_padding is None and trivial_seen:
        warnings.warn(
            f"this giotto batch carries {trivial_seen} trivial rows "
            "(birth == death) across its samples. giotto pads a batch to a "
            "common row count with rows of exactly that form, which are "
            "indistinguishable from genuine zero-persistence bars (RFC-0001 "
            "§4, §11.1, A.2). Every row was kept. Pass strip_padding=True to "
            "drop them, or strip_padding=False to keep them without this "
            "warning.",
            UserWarning,
            stacklevel=2,
        )

    return DiagramBatch.from_diagrams(diagrams, xp=xp)
