"""`from_*` adapters: backend output in, `PersistenceDiagram` out. RFC-0001 §11.

Five adapters, one per source this project supports:

    from_gudhi(obj, *, dim=None, **meta)   -> PersistenceDiagram
    from_ripser(obj, **meta)               -> PersistenceDiagram
    from_giotto(arr, *, reduced_homology, strip_padding=None, **meta)
                                           -> DiagramBatch
    from_persim(obj, **meta)               -> PersistenceDiagram
    from_array(arr, *, dim=None, **meta)   -> PersistenceDiagram

`from_giotto` deviates twice, both deliberately (§11): `reduced_homology` is a
required keyword-only argument, because omitting it MUST be a `TypeError` at
the call site rather than a value that slips past inside `**meta` (§5.1); and
its return type is fixed at `DiagramBatch`, length one for a single sample,
because nothing about an adapter's return type may depend on how many samples
the particular call happened to carry (§4).

`dim=` and `strip_padding=` are written out above where §11's own signature
block omits them, which is a defect in the RFC rather than in these
signatures: §11's input table requires `(n, 2)` input to state its degree
"with explicit `dim=`", and `dim` is not a `DiagramMeta` field, so it cannot
arrive inside `**meta`; `strip_padding` is the argument §11.1 spends a section
specifying. Both are keyword-only and both default to the behaviour the RFC
describes. See TODO.md, "§11's signature block omits two arguments it
requires".

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

import math
import numbers
import warnings
from collections.abc import Mapping, Sequence
from importlib import import_module, metadata
from typing import Any, NamedTuple

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

# I2 fixes `int32` as the storage dtype for degrees, so a degree outside this
# range is not one this type can hold. Named rather than inlined because two
# paths -- the degree column and a caller's `dim=` -- must refuse the same
# values, and a bound written twice is a bound that can drift.
_INT32_MIN = -(2**31)
_INT32_MAX = 2**31 - 1


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


def _is_row_sequence(obj: Any) -> bool:
    """Whether `obj` is a sequence *of rows*, which `str` and `bytes` never are.

    Spelled out rather than left to `isinstance(obj, Sequence)` because both
    are registered `Sequence`s: a bare `isinstance` gate admits `"hello"`, and
    the refusal that §11 owes the caller then arrives from inside the row loop
    as an exception about a single character.
    """
    return isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray))


def _is_extended_persistence(obj: Any) -> bool:
    """Whether `obj` is GUDHI's `extended_persistence()` 4-tuple. §11.

    §11's table lists two GUDHI forms and this is neither, so it is refused
    rather than attempted. It is also the one part of that exclusion an
    adapter can see: the call returns a *tuple* of exactly four
    `list[(dim, (birth, death))]` -- ordinary, relative, extended+ and
    extended- -- where `persistence()` returns one flat list of rows.

    Keyed on both facts, because either alone refuses real input. A
    `persistence()` result of four bars handed over as a tuple is also four
    things long; what separates them is that these members are *lists of
    rows* and a `persistence()` row is a `(dim, (birth, death))` pair.

    A single member passed alone is not detected, being indistinguishable
    from ordinary output; `from_gudhi`'s docstring states that residual case.
    """
    return (
        isinstance(obj, tuple)
        and len(obj) == 4
        and all(isinstance(member, list) for member in obj)
    )


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


def _require_real(column: Array, xp: Any, what: str) -> None:
    """Refuse a column that is not real-numeric, before any `astype`. I2.

    `astype` from a complex dtype to a real one discards the imaginary part
    and reports it only as a warning, which a caller who filtered warnings
    never sees; from `bool` it produces 0.0/1.0. Both are §9's category --
    clean, plausible and wrong -- and both are cheaper to refuse at the
    boundary than to explain in a diagram afterwards.
    """
    if not xp.isdtype(column.dtype, ("real floating", "integral")):
        raise TypeError(
            f"{what} has dtype {column.dtype}; §6.1 stores real coordinates "
            "as float64 and degrees as int32 (I2), and converting from this "
            "dtype would silently discard part of every value"
        )


def _require_int32_range(column: Array, xp: Any) -> None:
    """Refuse a degree column that `int32` cannot hold. I2.

    Checked before `astype` rather than trusted to it, because the cast does
    not report: numpy *wraps* an out-of-range `int64` (2**32 arrives as 0) and
    *saturates* an out-of-range `float64` (2**32 arrives as 2147483647), so
    the surviving diagram is clean, plausible and wrong, and which of the two
    wrong answers it holds depends on the input dtype. `dim=` refuses the same
    values through `_as_degree`, with the same message.
    """
    if int(column.shape[0]) == 0:
        return
    low, high = float(xp.min(column)), float(xp.max(column))
    if low < _INT32_MIN or high > _INT32_MAX:
        raise ValueError(
            f"the degree column holds a value outside int32 [{_INT32_MIN}, "
            f"{_INT32_MAX}], which I2 fixes as the storage dtype for degrees; "
            f"its range is [{low:.0f}, {high:.0f}]"
        )


def _as_float64(column: Array, xp: Any, what: str = "a coordinate column") -> Array:
    """I2, §6.1: storage is the namespace's own `float64`, whatever arrived."""
    _require_real(column, xp, what)
    return xp.astype(column, xp.float64)


def _as_dims(column: Array, xp: Any) -> Array:
    """I2, I3: the degree column as `int32`, refusing anything non-integral.

    A degree of 1.5 is not a homological degree, and `astype` would silently
    truncate it to 1 -- a diagram that is clean, plausible and wrong, which is
    §9's category. Non-finite values are refused first, `astype` on `inf` or
    `nan` being undefined rather than merely wrong, and out-of-range values
    last, for the reason `_require_int32_range` gives.
    """
    _require_real(column, xp, "the degree column")

    if xp.isdtype(column.dtype, "integral"):
        _require_int32_range(column, xp)
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
    _require_int32_range(column, xp)
    return xp.astype(column, xp.int32)


def _reject_negative_degrees(dims: Array, xp: Any) -> None:
    """I3 on a column that may be filtered before `core.py` ever sees it.

    `_as_degree` explains why this check normally belongs to the core type and
    not here: one owner, one message naming the offending value. `from_giotto`
    is the exception, because `strip_padding=True` removes rows *before*
    construction, so a row whose degree is -1 would leave through the padding
    mask and core would validate a column it never contained. The message is
    the core type's verbatim, so the same input reports the same thing
    whichever of the two refused it.
    """
    if int(dims.shape[0]) == 0:
        return
    minimum = int(xp.min(dims))
    if minimum < 0:
        raise ValueError(f"dims must be non-negative (I3); minimum is {minimum}")


def _as_degree(dim: Any, *, where: str = "dim=") -> int:
    """A caller-stated value as a homological degree. I3.

    The degree *column* is guarded by `_as_dims`, for a reason that applies
    just as much to the one degree a caller states by hand: `int(1.5)` is 1,
    and a diagram whose degrees are silently truncated is clean, plausible and
    wrong. So an integer is required rather than coerced.

    `bool` is rejected despite being an `int` in Python: `dim=True` would mean
    degree 1, which is a coincidence of the language rather than anything the
    caller meant. `numbers.Integral` rather than `int` so that a degree read
    out of an array -- the ordinary way a caller loops over degrees -- is not
    refused for being an `int64`.

    Both bounds are checked here, and deferring either to the core type is
    what an earlier version of this function got wrong. I3 is the core type's
    to enforce and it does enforce it -- on the *assembled column*, which is
    the thing that is not always there: `from_array(np.empty((0, 2)),
    dim=-1)` fills a column of length zero, over which "every degree is
    non-negative" is vacuously true, so the invalid degree the caller actually
    typed passes unremarked. A caller-stated scalar is known at the point it
    is stated and is checked there. The `int32` bound never reaches the core
    type either, for a different reason: `xp.full` raises first, with an
    `OverflowError` naming a dtype rather than the invariant.

    The non-negativity message is the core type's verbatim, so the same
    mistake reads the same way whichever of the two refused it.

    `where` names the caller's spelling in the message. GUDHI's
    `persistence()` list states a degree per row rather than through `dim=`,
    and a message about `dim=` sends that caller looking for an argument they
    did not pass.
    """
    if isinstance(dim, bool) or not isinstance(dim, numbers.Integral):
        raise TypeError(
            f"{where} must be an integer homological degree (I3); got "
            f"{dim!r} of type {type(dim).__name__}"
        )
    degree = int(dim)
    if not _INT32_MIN <= degree <= _INT32_MAX:
        raise ValueError(
            f"{where} is {degree}, outside int32 [{_INT32_MIN}, {_INT32_MAX}], "
            "which I2 fixes as the storage dtype for degrees"
        )
    if degree < 0:
        raise ValueError(f"dims must be non-negative (I3); minimum is {degree}")
    return degree


def _as_coordinate(value: Any, *, where: str) -> float:
    """A caller-stated birth or death as a `float`. I2, §6.1.

    The argument `_as_degree` makes for degrees applies unchanged to
    coordinates: `float("0.5")` is 0.5 and `float(True)` is 1.0, so a row that
    holds a string or a flag where a filtration value belongs would become a
    diagram that is clean, plausible and wrong rather than one that is
    refused. `numbers.Real` rather than `float`, so that a coordinate read out
    of an array -- a `float32` scalar from a backend -- is not refused for
    being the wrong flavour of real.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(
            f"{where} must be a real filtration value (§6.1); got "
            f"{value!r} of type {type(value).__name__}"
        )
    return float(value)


class _Clamped(NamedTuple):
    """What one call to `_clamp_i6` repaired, for the warning and for §8.

    `worst` is a maximum over the repaired rows alone, so that the figure in
    the warning is a gap that was actually absorbed rather than the largest in
    the array -- which could be an unrepaired backend bug, quoted inside a
    sentence calling it floating-point noise.

    That masking is defensive rather than load-bearing, and deliberately so:
    in any call that *returns*, every positive gap was either repaired or
    raised through I6, so a maximum over all rows would agree. The property
    holds only because the warning now follows construction, which is a fact
    about the caller. Stating it locally means this stays true if that
    ordering ever changes.
    """

    rows: int
    total: int
    worst: float


def _warn_clamped(clamped: _Clamped, *, stacklevel: int = 3) -> None:
    """§3.1's mandatory warning, once per adapter call.

    Separated from `_clamp_i6` so that `from_giotto` can sum across a batch and
    warn once. Warning per sample would make the number of warnings a property
    of the batch's shape rather than of its data -- the argument §11.1 already
    settles for padding (D8) -- and a 500-sample batch with systematic
    filtration rounding would emit 500 identical warnings.

    `stacklevel=3` is `warnings.warn` here, the adapter that called this, then
    the user's own line, which is the only one of the three they can act on.
    """
    if not clamped.rows:
        return
    warnings.warn(
        f"clamped {clamped.rows} of {clamped.total} rows where death < birth "
        f"(I6), the largest by {clamped.worst:.3g}. These are within "
        "floating-point noise of zero and are treated as filtration rounding "
        "(RFC-0001 §3.1); provenance['clamped_rows'] records the count.",
        UserWarning,
        stacklevel=stacklevel,
    )


def _clamp_i6(births: Array, deaths: Array, xp: Any) -> tuple[Array, _Clamped]:
    """Repair `death < birth` rows that are floating-point noise. §3.1.

    Returns the (possibly repaired) deaths and a report of what was repaired:
    the count, which the caller records as `provenance["clamped_rows"]` (§8),
    and the largest gap absorbed, which `_warn_clamped` reports. §3.1 requires
    the warning -- a silent repair is a backend defect absorbed without trace
    -- but this function does not issue it, so that a batch adapter can warn
    once for the whole call.

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
    total = int(births.shape[0])
    if n_repaired == 0:
        return deaths, _Clamped(0, total, 0.0)

    worst = float(xp.max(xp.where(repair, gap, zero)))
    return xp.where(repair, births, deaths), _Clamped(n_repaired, total, worst)


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

    `coeff_field` is checked here rather than only in `_coeff_field`, which
    the three adapters D17 excludes never call. §11 excuses `from_array`,
    `from_persim` and `from_giotto` from *recording* a field; it does not make
    `coeff_field="two"` admissible on them, and §8 types the field `int |
    None` for every diagram however it was built.

    The checked value is written *back*, which is the half an earlier version
    left out. `_require_coeff_field` admits any `numbers.Integral` so that a
    field read out of an array is not refused for being an `int64` -- and then
    storing that `int64` unconverted puts a value in `coeff_field` that §8's
    `int | None` does not describe and that `json.dumps` refuses, which is
    §10.2's failure arriving from the one §8 field `_require_json_representable`
    does not reach. `from_gudhi` and `from_ripser` already store a builtin
    `int`, having gone through `_coeff_field`; assigning here is what makes the
    other three agree rather than differ by which adapter was called.
    """
    for reserved in ("backend", "backend_version"):
        if reserved in meta:
            raise TypeError(
                f"{reserved!r} is recorded by the adapter and cannot be "
                "passed in: it is the fact that says where this diagram came "
                "from (RFC-0001 §8, §11)"
            )
    if meta.get("coeff_field") is not None:
        meta["coeff_field"] = _require_coeff_field(meta["coeff_field"])

    caller_provenance = dict(meta.pop("provenance", {}) or {})
    caller_params = dict(meta.pop("params", {}) or {})
    caller_provenance.update(provenance)
    caller_params.update(params or {})

    _require_json_representable(caller_provenance, "provenance")
    _require_json_representable(caller_params, "params")

    return DiagramMeta(
        backend=backend,
        backend_version=backend_version,
        params=caller_params,
        provenance=caller_provenance,
        **meta,
    )


def _require_json_representable(mapping: dict[str, Any], field: str) -> None:
    """§8: every `params` and `provenance` value must survive `meta.json`.

    §8 admits `str`, `int`, `float`, `bool`, `None`, and lists or `str`-keyed
    mappings of those, and requires adapters to convert at the point of
    recording -- `str(arr.dtype)`, a Python `int` for the counts, which the
    adapter-side keys already do. This check covers the other half of the
    merged mapping: what the caller passed in through `**meta`.

    Checked at the adapter rather than at `save()`, which is §8's own stated
    reason for the rule. A `Path` in `provenance` produces a diagram that
    satisfies §3.1 and §8's key rules completely and cannot be written, and
    the exception then names `meta.json` at a call arbitrarily far from the
    adapter that accepted the value.

    `tuple` is not admitted, deliberately, despite `json.dumps` accepting one:
    it round-trips back as a `list`, so a diagram carrying one would fail
    §10.1 requirement 1's `load(dump(d)) == d` rather than fail to save. A
    value that cannot survive the round trip is refused at the same boundary
    as one that cannot make it.

    Scalars are matched by exact type rather than `isinstance`, which is the
    one place this function is deliberately stricter than `json.dumps`. §8
    names the NumPy scalar as the hazard, and `numpy.float64` subclasses
    `float`: an `isinstance` gate admits it, and §3.3 keeps this module to the
    standard library, so there is no `numpy.generic` to test against instead.
    Exact types catch it, and catch `float32` -- which is not a `float`
    subclass and does not serialise -- through the same clause. §8's remedy is
    unchanged either way: convert at the call site.

    `nan` and `inf` are refused for §10.2's stated reason: "`inf` lives in
    `bars.npz`, where NumPy represents it correctly, and never in the JSON.
    This is the reason for the split." Neither is valid JSON -- `json.dumps`
    emits the bare tokens `NaN` and `Infinity` by default, which any
    conforming reader rejects, and §10.3 makes the same point about Parquet's
    IEEE 754 `double` being unlike JSON's number. `nan` fails §10.1
    requirement 1 outright besides, since `nan != nan` makes a round-tripped
    diagram compare unequal to itself. Essential bars are unaffected: they
    live in `deaths`, not in metadata.
    """
    scalars = (str, bool, int, float)

    def check_keys(m: Mapping[Any, Any], path: str) -> None:
        """§8 admits `str`-keyed mappings, at the top level and below it.

        Applied to the mapping itself as well as to nested ones, because
        `json.dumps` does not refuse an `int` key -- it rewrites it as a
        string, so `params={1: "x"}` reloads as `{"1": "x"}` and the diagram
        that comes back is not the one that went out.
        """
        for key in m:
            if not isinstance(key, str):
                raise TypeError(
                    f"{field}{path} has the key {key!r} of type "
                    f"{type(key).__name__}; §8 admits str-keyed mappings "
                    f"only, since §10.2 stores this as JSON -- json.dumps "
                    f"would silently rewrite it as {str(key)!r}"
                )

    def check(value: Any, path: str) -> None:
        if value is None or type(value) in scalars:
            if type(value) is float and not math.isfinite(value):
                raise TypeError(
                    f"{field}{path} is {value!r}; §10.2 keeps non-finite "
                    "values out of the JSON entirely -- json.dumps writes "
                    "them as the bare tokens NaN and Infinity, which are not "
                    "valid JSON, and nan does not even survive a round trip "
                    "as itself (§10.1 requirement 1)"
                )
            return
        if isinstance(value, list):
            for i, item in enumerate(value):
                check(item, f"{path}[{i}]")
            return
        if isinstance(value, Mapping):
            check_keys(value, path)
            for key, item in value.items():
                check(item, f"{path}[{key!r}]")
            return
        raise TypeError(
            f"{field}{path} is {value!r} of type {type(value).__name__}, "
            "which §8 does not admit: every value must be JSON-representable "
            "(str, int, float, bool, None, or a list or str-keyed mapping of "
            "those), because §10.2 stores this mapping as UTF-8 JSON. A NumPy "
            "scalar is the usual culprit -- convert it at the call site"
        )

    check_keys(mapping, "")
    for key, value in mapping.items():
        check(value, f"[{key!r}]")


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


def _require_coeff_field(stated: Any) -> int:
    """§8's `coeff_field: int | None`, checked wherever a caller states one.

    `DiagramMeta` validates the *source* key and not the value beside it, so
    without this a `coeff_field="two"` recorded with `coeff_field_source =
    "caller"` is a provenance entry that reads as authoritative and describes
    no field at all -- the one outcome D17's source key exists to prevent.

    `bool` is excluded for `_as_degree`'s reason: the field of one element is
    not what a caller means by `coeff_field=True`.
    """
    if isinstance(stated, bool) or not isinstance(stated, numbers.Integral):
        raise TypeError(
            "coeff_field= must be the integer characteristic of the field "
            f"homology was computed over (§8, §9.3); got {stated!r} of type "
            f"{type(stated).__name__}"
        )
    return int(stated)


def _coeff_field(meta: dict[str, Any], default: int) -> tuple[int, str]:
    """D17, §11: the coefficient field and where it came from.

    The caller's value with `"caller"` if one arrived, the backend's
    documented default with `"backend_default"` otherwise. The second is an
    assumption and the source key is what marks it as one: no backend returns
    the field it computed with (A.5), so an adapter cannot verify that the
    caller left the default in place.

    `coeff_field=None` is read as "stated nothing", not as "stated no field",
    and records the backend default. §8 spells the absence of a value `None`
    on the field itself, so the two are the same statement arriving by
    different routes, and §11's requirement is about what the backend would
    have done rather than about what the caller typed.
    """
    stated = meta.pop("coeff_field", None)
    if stated is None:
        return default, "backend_default"
    return _require_coeff_field(stated), "caller"


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
) -> tuple[PersistenceDiagram, _Clamped]:
    """Convert dtypes, clamp §3.1's noise, and construct. Order is preserved.

    Returns the clamp report alongside the diagram rather than warning here:
    the caller knows whether it is building one diagram or a batch of them, and
    §3.1's warning is owed once per call either way. See `_warn_clamped`.
    """
    births = _as_float64(births, xp, "the birth column")
    deaths = _as_float64(deaths, xp, "the death column")
    deaths, clamped = _clamp_i6(births, deaths, xp)

    recorded = {"clamped_rows": clamped.rows, **provenance}
    diagram = PersistenceDiagram(
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
    return diagram, clamped


def _columns_from_table(
    arr: Array, xp: Any, *, dim: int | None
) -> tuple[Array, Array, Array]:
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
    degree = _as_degree(dim)
    return xp.full((arr.shape[0],), degree, dtype=xp.int32), arr[:, 0], arr[:, 1]


def _columns_from_degree_list(
    dgms: Sequence[Any], xp: Any
) -> tuple[Array, Array, Array]:
    """Stack `list[(n, 2)]` where list position is the degree. §11.

    Ripser's `dgms` and persim's input share this shape. Row order within a
    degree is preserved exactly; degrees follow the list, which is the
    backend's own order for the merged diagram.

    Every block must share one namespace (I7). `core.py` checks the same thing
    across the diagrams of a batch and says why: without it, `xp.concat` would
    "either raise something opaque from the backend or silently coerce a
    foreign array", and a silently coerced diagram then validates cleanly with
    the mixed input already erased.
    """
    dim_blocks, birth_blocks, death_blocks = [], [], []
    for degree, block in enumerate(dgms):
        if _has_namespace(block):
            block_xp = _namespace_of(block)
            if block_xp is not xp:
                raise ValueError(
                    f"the diagram at index {degree} has array namespace "
                    f"{block_xp.__name__!r}, not {xp.__name__!r}: one diagram "
                    "has one namespace (I7)"
                )
        else:
            # Converted without forcing a dtype, so that the block arrives
            # with the dtype its contents imply and `_as_float64` can refuse
            # it. Asking for `float64` here instead performs the coercion the
            # refusal exists to prevent: a block of `[[False, True]]` or
            # `[["0.0", "1.0"]]` would become a clean, plausible and wrong
            # (0.0, 1.0) bar, and only on this path -- the same rows inside a
            # NumPy array are refused.
            block = xp.asarray(block)
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


def _columns_from_pairs(rows: Sequence[Any], xp: Any) -> tuple[Array, Array, Array]:
    """Split GUDHI's `list[(dim, (birth, death))]` into three columns. §11.

    Each row is unpacked under its own index so that a mis-shaped row is
    refused by position. Unpacked in bulk, one bad row among ten yields
    `ValueError: too many values to unpack (expected 2)` -- true, useless, and
    not obviously about this library at all.

    Values are converted through `_as_degree` and `_as_coordinate` rather than
    `int()` and `float()`. This is the one input form that reaches storage
    without passing an array through `_as_dims`, and a bare `int()` here would
    reopen on this path exactly what `_as_dims` closes on every other:
    `int(1.5)` is 1, and a diagram whose degrees were silently truncated is
    clean, plausible and wrong (§9).

    The conversions sit outside the `except`, which catches the *shape* of a
    row. Inside it, `_as_degree`'s "got 1.5 of type float" -- the message that
    says what is actually wrong -- would be swallowed and reported as a
    mis-shaped row.
    """
    degrees: list[int] = []
    births: list[float] = []
    deaths: list[float] = []
    for index, row in enumerate(rows):
        try:
            degree, (birth, death) = row
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"row {index} of a persistence() list must be "
                f"(dim, (birth, death)) (RFC-0001 §11); got {row!r}"
            ) from exc
        degrees.append(
            _as_degree(
                degree, where=f"the degree in row {index} of a persistence() list"
            )
        )
        births.append(
            _as_coordinate(
                birth, where=f"the birth in row {index} of a persistence() list"
            )
        )
        deaths.append(
            _as_coordinate(
                death, where=f"the death in row {index} of a persistence() list"
            )
        )
    return (
        xp.asarray(degrees, dtype=xp.int32),
        xp.asarray(births, dtype=xp.float64),
        xp.asarray(deaths, dtype=xp.float64),
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

    **Extended persistence is out of scope, and only partly detectable.**
    `extended_persistence()` returns a 4-tuple of `list[(dim, (birth, death))]`
    -- ordinary, relative, extended+ and extended- -- which is structurally
    distinct from `persistence()`'s flat list, so the tuple itself is
    rejected by name. **A single member of it, passed alone, is not.** The
    relative and extended- members raise at construction on I6, with an error
    about death times rather than about scope; the ordinary and extended+
    members satisfy every invariant and construct cleanly into a diagram
    whose bars nothing here can mark as extended. Closing that needs a fourth
    per-bar field this type does not have, so it stays open: **do not route
    extended persistence through this adapter one sub-diagram at a time.**

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
    if _is_extended_persistence(obj):
        raise TypeError(
            "extended persistence is out of scope, so extended_persistence() "
            "is not an accepted input form (RFC-0001 §11): its relative and "
            "extended- bars carry death < birth by construction, which I6 "
            "forbids exactly, and the bars that would survive cannot be "
            "marked as extended by this type. from_gudhi accepts "
            "SimplexTree.persistence() or "
            "persistence_intervals_in_dimension(k)."
        )

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
    elif _is_row_sequence(obj):
        if dim is not None:
            raise ValueError(
                "a persistence() list already carries a degree per bar, so "
                "dim= would be a second source for one fact"
            )
        xp = _namespace_for_rows()
        dims, births, deaths = _columns_from_pairs(obj, xp)
    else:
        raise TypeError(
            "from_gudhi accepts SimplexTree.persistence() output "
            "(list[(dim, (birth, death))]) or "
            "persistence_intervals_in_dimension(k) output ((n, 2), with "
            f"dim=k); got {type(obj).__name__}"
        )

    diagram, clamped = _diagram_from_columns(
        dims=dims,
        births=births,
        deaths=deaths,
        xp=xp,
        backend="gudhi",
        backend_version=_installed_version("gudhi"),
        provenance=provenance,
        meta={"coeff_field": field, **meta},
    )
    _warn_clamped(clamped)
    return diagram


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
        if not _is_row_sequence(dgms):
            raise TypeError(
                "a ripser() result carries its diagrams under 'dgms' as a "
                "list of (n, 2) arrays, degree by index (RFC-0001 §11); this "
                f"mapping's 'dgms' is {type(dgms).__name__}"
            )
    elif _is_row_sequence(obj):
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

    diagram, clamped = _diagram_from_columns(
        dims=dims,
        births=births,
        deaths=deaths,
        xp=xp,
        backend="ripser",
        backend_version=_installed_version("ripser"),
        provenance=provenance,
        meta={"coeff_field": field, **meta},
    )
    _warn_clamped(clamped)
    return diagram


def from_persim(obj: Any, **meta: Any) -> PersistenceDiagram:
    """A persim-shaped diagram list as a `PersistenceDiagram`. §11.

    `list[(n, 2)]`, degree by index -- the same shape as Ripser's `dgms`.

    **No claim is made about essential bars, and none about the coefficient
    field.** persim consumes diagrams and computes no homology (§5.1: "no
    opinion"), so it cannot certify that nothing was lost upstream, and §8's
    `essential_bars_source` means the verdict at computation time. Both keys
    are therefore absent rather than guessed, and §11 puts `from_persim` out
    of scope for D17 for the same reason.
    """
    if not _is_row_sequence(obj):
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
    diagram, clamped = _diagram_from_columns(
        dims=dims,
        births=births,
        deaths=deaths,
        xp=xp,
        backend="persim",
        backend_version=_installed_version("persim"),
        provenance=provenance,
        meta=meta,
    )
    _warn_clamped(clamped)
    return diagram


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
    diagram, clamped = _diagram_from_columns(
        dims=dims,
        births=births,
        deaths=deaths,
        xp=xp,
        backend="array",
        backend_version=None,
        provenance=dict(_source_dtype(arr)),
        meta=meta,
    )
    _warn_clamped(clamped)
    return diagram


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

    **What counts as padding does not depend on the mode.** The degree column
    is validated across the whole sample before any row is dropped, so
    `(b, b, -1)` and `(b, b, nan)` are refused under all three modes rather
    than deleted under one of them: they are not rows giotto emits, and §3.1's
    answer to a violation is to surface it. The caller decides whether to keep
    giotto's padding, not whether this adapter checks its input.

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

    # Both flags are read for their truth value further down, and Python's
    # truthiness would take any object for an answer. That is the wrong
    # default for these two specifically: `reduced_homology` exists only to
    # record a claim about the data that no property of the array can confirm
    # or contradict (§5.1), so `reduced_homology="False"` would write
    # `params={"reduced_homology": True}` and `essential_bars="lost_upstream"`
    # into a diagram whose caller meant the opposite, with nothing downstream
    # able to notice. `strip_padding` is three-valued, which makes the same
    # slip change the data as well as the record: `"False"` is not None and is
    # truthy, so it strips.
    if not isinstance(reduced_homology, bool):
        raise TypeError(
            "reduced_homology= must be True or False -- pass "
            "vr.reduced_homology off the fitted transformer (RFC-0001 §5.1, "
            f"§11); got {reduced_homology!r} of type "
            f"{type(reduced_homology).__name__}"
        )
    if strip_padding is not None and not isinstance(strip_padding, bool):
        raise TypeError(
            "strip_padding= must be True, False or None, which are §11.1's "
            f"three modes; got {strip_padding!r} of type "
            f"{type(strip_padding).__name__}"
        )

    xp = _namespace_of(arr)

    # Checked here rather than only inside the loop, for the reason the
    # metadata preflight below gives at length: a zero-sample batch never
    # enters the loop, so `from_giotto(zeros((0, 2, 3), dtype=bool), ...)`
    # used to succeed while the same call one sample later raised.
    #
    # Dtype is the whole of what needs hoisting, and that is a fact rather
    # than a judgement: every other check in the loop -- `_as_dims`'s
    # integrality and int32 range, `_reject_negative_degrees`, the `births ==
    # deaths` padding mask, `_clamp_i6`, and §3.1's invariants -- is a
    # statement about rows, and is vacuously true of a sample that has none.
    # Dtype is a property of the array itself, present with zero samples and
    # zero bars alike, so hoisting it leaves nothing behind to drift.
    #
    # One call covers all three columns because giotto's output is one array
    # and therefore one dtype: the check `_as_float64` runs on births and
    # deaths and the one `_as_dims` runs on degrees are the same check here.
    _require_real(arr, xp, "the giotto array")

    essential_bars = "lost_upstream" if reduced_homology else "faithful"
    version = _installed_version("giotto-tda")
    source_dtype = _source_dtype(arr)

    # `**meta` is validated once here, before the loop, and the result is
    # discarded. This is the only adapter whose metadata is checked inside a
    # loop over the data, and a batch with no samples never enters it: without
    # this line `from_giotto(zeros((0, 2, 3)), ..., backend="fake")` succeeds
    # and drops the argument, while the same call one sample later raises.
    # Whether a caller's typo is caught would then depend on how many samples
    # their batch happened to carry, which is §4's
    # shape-depends-on-what-else-was-there hazard reappearing in the one place
    # §11 was careful to keep it out of -- the adapter's own behaviour.
    #
    # Spelled as a real `_build_meta` call rather than a subset of its checks
    # so that the two cannot drift: whatever a one-sample batch refuses, a
    # zero-sample batch refuses identically, including `DiagramMeta`'s own
    # `TypeError` for an unknown field.
    #
    # `clamped_rows` is listed for that identity and not because anything here
    # measured one. `_diagram_from_columns` adds the key before calling
    # `_build_meta`, so in real construction an adapter-owned `clamped_rows`
    # overwrites a caller's key of that name -- which is `_build_meta`'s
    # documented rule, the adapter being the party that saw the backend's
    # output. Omitting it here made the preflight *stricter* than the
    # construction it stands in for: `provenance={"clamped_rows": <junk>}` is
    # accepted and overwritten by every other adapter, and was refused by this
    # one alone.
    _build_meta(
        backend="giotto",
        backend_version=version,
        provenance={
            "essential_bars": essential_bars,
            "essential_bars_source": essential_bars,
            "padding_removed": 0,
            "clamped_rows": 0,
            **source_dtype,
        },
        params={"reduced_homology": reduced_homology},
        meta=dict(meta),
    )

    diagrams = []
    trivial_seen = 0
    clamped_rows = clamped_total = 0
    clamped_worst = 0.0
    for i in range(int(arr.shape[0])):
        sample = arr[i]
        births, deaths, dims = sample[:, 0], sample[:, 1], sample[:, 2]

        # The degree column is validated for the whole sample *before* the
        # padding mask is applied, and the two lines together are the point.
        # Padding is a row giotto could have emitted, which means a row whose
        # degree is a homological degree; `(b, b, -1)`, `(b, b, 0.5)` and
        # `(b, b, nan)` are not padding but a corrupt array, and §3.1's one
        # answer to a violation is to surface it. Filtering first would let
        # `strip_padding=True` delete the evidence and count the deletion as
        # padding -- and would leave the three modes disagreeing about whether
        # the same array is acceptable input, which is exactly what §11.1's
        # "the caller decides" is not meant to mean. This is the same argument
        # the finiteness term below makes for births, applied to the third
        # column.
        dims = _as_dims(dims, xp)
        _reject_negative_degrees(dims, xp)

        # Finiteness is part of what makes a row padding. giotto pads with
        # `(b, b, dim)` for a finite `b` (§11.1, A.2), whereas `(inf, inf)` is
        # an I4 violation -- and §3.1 has one answer for a violation, which is
        # to surface it. Stripping it instead would delete the evidence and
        # count the deletion as padding.
        trivial = (births == deaths) & xp.isfinite(births)
        n_trivial = int(xp.sum(xp.astype(trivial, xp.int64)))
        trivial_seen += n_trivial

        removed = 0
        if strip_padding:
            keep = ~trivial
            births, deaths, dims = births[keep], deaths[keep], dims[keep]
            removed = n_trivial

        diagram, clamped = _diagram_from_columns(
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
                **source_dtype,
            },
            params={"reduced_homology": bool(reduced_homology)},
            meta=dict(meta),
        )
        diagrams.append(diagram)
        clamped_rows += clamped.rows
        clamped_total += clamped.total
        clamped_worst = max(clamped_worst, clamped.worst)

    # Once for the batch, not once per sample -- see `_warn_clamped`, and
    # §11.1's own "warn once" for the padding warning below.
    _warn_clamped(_Clamped(clamped_rows, clamped_total, clamped_worst))

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
