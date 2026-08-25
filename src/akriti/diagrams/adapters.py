"""`from_*` adapters: backend output in, `PersistenceDiagram` out. RFC-0001 §11.

Five adapters, one per source this project supports:

    from_gudhi(obj, *, dim=None, **meta)   -> PersistenceDiagram
    from_ripser(obj, **meta)               -> PersistenceDiagram
    from_giotto(arr, *, reduced_homology, infinity_values,
                strip_padding=None, **meta)
                                           -> DiagramBatch
    from_persim(obj, **meta)               -> PersistenceDiagram
    from_array(arr, *, columns=None, dim=None, **meta) -> PersistenceDiagram

`from_giotto` has four deliberate deviations (§11): `reduced_homology` is a
required keyword-only argument, because omitting it MUST be a `TypeError` at
the call site rather than a value that slips past inside `**meta` (§5.1);
`infinity_values` is required on the same grounds and admits only `inf`,
giotto's default of `None` writing a finite sentinel §5 refuses;
`strip_padding=None` exposes §11.1's three-valued padding decision explicitly;
and its return type is fixed at `DiagramBatch`, length one for a single sample,
because nothing about an adapter's return type may depend on how many samples
the particular call happened to carry (§4).

`infinity_values` was enforced here before §11 carried it; RFC-0001 entry 55
ratified it (§12.3 R5), closing `tasks/questions.md` C1. §11.2 carries the
refusal cases, and the narrowing remains compatible with §5.1 as written,
which fixes what is *derived* from `reduced_homology` rather than which inputs
are admissible.

`dim=` and `strip_padding=` are keyword-only bar-data controls rather than
`DiagramMeta` fields. The first defaults to §11's degree-carrying input
behaviour; the second defaults to §11.1's keep-and-warn padding mode.

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

**Imports.** Importing this module remains third-party-free (§3.3, §10.1
requirement 2). Three function-scoped, lazy paths are permitted: numpy for an
input carrying no array at all -- GUDHI's `persistence()` list or an empty
diagram list -- on the `akriti[numpy]` extra (see `_namespace_for_rows`);
`array-api-compat` through the shared namespace resolver when a caller's
backend has no native `__array_namespace__`, on `akriti[torch]`; and pyarrow
inside `to_parquet()`, on `akriti[parquet]`. None is reached by module import,
and each is confined to the function that needs it.
"""

from __future__ import annotations

import csv
import io
import itertools
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
    _parse_optional_version,
    namespace_of,
)

__all__ = [
    "from_array",
    "from_giotto",
    "from_gudhi",
    "from_persim",
    "from_ripser",
    "to_arrays",
    "to_csv",
    "to_parquet",
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
# absorbed only when it is within eight local downward float64 ULPs of the
# birth value. The spacing is computed with `nextafter` after conversion to
# float64, so the allowance follows the representable grid at every magnitude
# (including zero and subnormals) rather than adding an arbitrary absolute
# floor or using a broad relative tolerance. Anything larger is a backend bug
# and reaches §3.1's I6 check unmodified, which reports its magnitude.
_CLAMP_ULPS = 8
_FLOAT64_MIN_SUBNORMAL = float.fromhex("0x0.0000000000001p-1022")
_FLOAT64_SMALLEST_NORMAL = float.fromhex("0x1.0p-1022")

# I2 fixes `int32` as the storage dtype for degrees, so a degree outside this
# range is not one this type can hold. Named rather than inlined because two
# paths -- the degree column and a caller's `dim=` -- must refuse the same
# values, and a bound written twice is a bound that can drift.
_INT32_MIN = -(2**31)
_INT32_MAX = 2**31 - 1

# §8's reserved `provenance` keys, in full -- eight of them. Every one names
# the writer that measured it: two adapter-time counts (`clamped_rows`,
# `padding_removed`), a dtype (`source_dtype`), two source keys
# (`essential_bars_source`, `coeff_field_source`), and the three remaining
# `essential_bars*` keys whose writers §8 lists by name. None of those writers
# is a caller. They are refused in `_build_meta` on exactly the ground
# `backend` and `backend_version` already are.
#
# Named as a set rather than checked one adapter at a time because the defect
# this closes was that the refusal *was* per-adapter, by accident: a caller's
# key lost the merge wherever the adapter wrote one of its own and survived
# wherever it did not, so `from_persim` and `from_array` -- the two that record
# no essential-bar claim (§11) -- were the two that would accept one.
_ADAPTER_OWNED_PROVENANCE = frozenset(
    {
        "essential_bars",
        "essential_bars_dropped",
        "essential_bars_finitized_at",
        "essential_bars_source",
        "coeff_field_source",
        "source_dtype",
        "clamped_rows",
        "padding_removed",
    }
)

# §10.3's recognised `columns=` names, matched case-insensitively. `diagram_id`
# is deliberately absent and refused by name rather than falling in here: a
# table headed with it is a batch CSV wanting the `.akd` format, which is a
# different message from "unknown column name" (§10.1 requirement 1).
_COLUMN_NAMES = frozenset({"birth", "death", "dim"})

# §6.1 fixes `float64` as the storage dtype for coordinates, and float64 holds
# every integer up to 2**53 exactly and only some of them above it. Named for
# the same reason as the two bounds above: the array path and the row path
# must refuse the same values.
_FLOAT64_EXACT_INT = 2**53


# ---------------------------------------------------------------------------
# Namespace and dtype
# ---------------------------------------------------------------------------


def _is_row_sequence(obj: Any) -> bool:
    """Whether `obj` is a sequence *of rows*, which `str` and `bytes` never are.

    Spelled out rather than left to `isinstance(obj, Sequence)` because both
    are registered `Sequence`s: a bare `isinstance` gate admits `"hello"`, and
    the refusal that §11 owes the caller then arrives from inside the row loop
    as an exception about a single character.
    """
    return isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray))


def _is_degree_indexed_block_list(obj: Any) -> bool:
    """Whether `obj` is a list of `(n, 2)` blocks rather than of rows. §11.

    GUDHI's sklearn-compatible form (`RipsPersistence` and its siblings)
    returns, per sample, a `list[(n, 2)]`. This exists only so that omitting
    `homology_dimensions` with such a list raises the `TypeError` `N11-2`
    requires, naming the missing argument, rather than the `ValueError` about
    a mis-shaped `(dim, (birth, death))` row that `_columns_from_pairs` would
    otherwise produce -- true, and about the wrong thing.

    **This is not the discriminator between the two GUDHI forms.** §11 states
    that the sklearn shape is identical to Ripser's `Rips().fit_transform(X)`
    and to persim's input, so nothing structural can tell them apart; the
    presence of `homology_dimensions=` is what selects the form, and this
    predicate is never consulted when it was supplied. Reading the blocks to
    decide the form would be the guess §11 refuses.

    Rank is asked of the block object itself, which is what the backend
    returns: an `(n, 2)` array reports `ndim == 2`, and `persistence()`'s
    `(dim, (birth, death))` rows are Python tuples that report no `ndim` at
    all. An empty list is not this form -- it is `persistence()` on a
    filtration with no bars, which already constructs an empty diagram.
    """
    if not _is_row_sequence(obj) or len(obj) == 0:
        return False
    return all(getattr(block, "ndim", None) == 2 for block in obj)


def _is_coordinate_slot(obj: Any) -> bool:
    """Whether `obj` sits where a single filtration value belongs. §11.

    A scalar, of any type: what it *is* is `_as_coordinate`'s question. What
    this rules out is anything holding more than one value -- a sequence, or
    an array of rank one or more -- because that is where a `(birth, death)`
    pair and a `(dim, (birth, death))` row part company, and `_is_interval`
    needs exactly that distinction.

    `str` counts as a slot, `_is_row_sequence` having already excluded it from
    the sequences: `["a", "b"]` is then an interval whose coordinates are
    refused by name, rather than a non-interval whose row stops being a row.
    """
    return not _is_row_sequence(obj) and getattr(obj, "ndim", 0) == 0


def _is_interval(obj: Any) -> bool:
    """Whether `obj` is one `(birth, death)` pair, however it is spelled. §11.

    **Structural, and it has to be**, because the one thing it must never
    accept is a *row*: `_is_persistence_row` asks this of `obj[1]`, so a
    `(dim, (birth, death))` arriving here is what separates a two-row extended
    member from a single row, and the whole detector rests on that. A pair of
    coordinates is therefore a two-element sequence *both of whose entries
    hold one value* (`_is_coordinate_slot`), which is exactly that distinction
    and no more of one. "Neither entry is a sequence" is the same rule written
    one degree too weakly, and gets a member of two array-interval rows wrong:
    the row `[0, array([b, d])]` has no sequence in it either.

    Asking instead whether both entries were `numbers.Real` answered the same
    question by a stronger test, and paid for it on values rather than shape:
    `[0, ["a", "b"]]` stopped being a row, so four of them became "extended
    persistence" -- a message about scope for an input whose defect is a
    string where a filtration value belongs. Admissibility is
    `_columns_from_pairs`' question and it answers it by row index and field
    name.

    **A rank-1 two-element array is a pair too**, and leaving it out was the
    same bug from the other end. `numpy.ndarray` is not a registered
    `Sequence`, so `[dim, array([b, d])]` was not a row, four of them were
    four list members that are not rows, and `_is_extended_persistence` --
    written specifically so that acceptance would not depend on how many bars
    an input carried -- refused at four what it accepted at three and five. An
    extended member cannot be confused for one of these: a member holds rows,
    not coordinates, and no row is rank-1 of length two.
    """
    if _is_row_sequence(obj):
        return len(obj) == 2 and all(_is_coordinate_slot(value) for value in obj)
    return getattr(obj, "ndim", None) == 1 and tuple(getattr(obj, "shape", ())) == (2,)


def _is_persistence_row(obj: Any) -> bool:
    """Whether `obj` is one `(dim, (birth, death))` row. §11.

    Structural only, and deliberately so: it exists to tell a *row* from a
    *list of rows*, which is the one distinction `_is_extended_persistence`
    turns on. Whether the values in a row are admissible is
    `_columns_from_pairs`' question, and it answers it by row index and field
    name. So `[0, [True, False]]` is a row here and is refused there, with a
    message about a boolean where a filtration value belongs rather than about
    extended persistence. `_is_interval` carries what "a pair" means and why
    that reading is the structural one.
    """
    return _is_row_sequence(obj) and len(obj) == 2 and _is_interval(obj[1])


def _is_extended_persistence(obj: Any) -> bool:
    """Whether `obj` is GUDHI's four-element extended-persistence list. §11.

    §11's table lists two GUDHI forms and this is neither, so it is refused
    rather than attempted. It is also the one part of that exclusion an
    adapter can see: the call returns a *list* of exactly four
    `list[(dim, (birth, death))]` -- ordinary, relative, extended+ and
    extended- -- where `persistence()` returns one flat list of rows.

    The outer container is a list in GUDHI 3.13. A `persistence()` result of
    four bars is also four things long, so **length is not the discriminator
    and MUST NOT be used as one**: what separates the two is that an extended
    member is a *list of rows* and a `persistence()` member is a single
    `(dim, (birth, death))` row. Testing only that the members are lists
    refuses `[[0, [0.0, 1.0]], ...]` -- a four-bar `persistence()` result
    whose rows were spelled with lists, which every other bar count accepts --
    so the same input form would be admissible at three bars and at five and
    rejected at four. Cardinality-dependent acceptance is §4's
    shape-depends-on-what-else-was-there hazard in the adapter's own gate.

    That hazard is closed only as far as `_is_persistence_row` recognises a
    row, so **a spelling of a row it does not recognise reopens it**, at four
    members and nowhere else. `_is_interval` carries the two that were
    missing -- an array interval and a non-numeric one -- and why the reading
    it makes is structural.

    **The same argument binds the two container axes, which is what an earlier
    version got wrong.** Both the outer container and each member were pinned
    to `list` exactly, while `_is_persistence_row` accepted any row sequence --
    so of the twelve ways one extended result can be spelled, only the three
    with `list` in both container slots were named as out of scope. The other
    nine were refused with a message about *shape*, which §11 forbids in the
    same sentence that requires the rejection: "MUST name the scope exclusion
    rather than the shape". A caller told that row 0 is mis-shaped goes hunting
    for a typo in data that is exactly what GUDHI handed them, and a caller
    holding a four-element *tuple* was told this adapter rejects "a flat tuple
    of persistence rows", which is a different input form entirely.

    So both containers are read with `_is_row_sequence`, the same predicate the
    rows are. This cannot widen the detector onto ordinary output, because
    what discriminates is unchanged and is a property of the members rather
    than of any container: a `persistence()` result's members *are* rows, and
    an extended result's members are lists *of* rows. A flat tuple of four
    bars still reaches the flat-tuple refusal, and a four-bar `persistence()`
    list is still accepted whatever its rows are spelled as.

    GUDHI itself returns a list of lists of tuple rows, so this decides nothing
    about live backend output; it decides what happens to output that has been
    through a serializer, which is what §11.2's frozen fixtures are. JSON
    spells every container a `list` and the detector already worked there --
    it is the round trips that *preserve* tuples, or a caller who writes
    `tuple(st.extended_persistence())`, that this closes.

    A single member passed alone is not detected, being indistinguishable
    from ordinary output; `from_gudhi`'s docstring states that residual case.
    """
    return (
        _is_row_sequence(obj)
        and len(obj) == 4
        and all(
            _is_row_sequence(member) and not _is_persistence_row(member)
            for member in obj
        )
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

    **The namespace comes back through `namespace_of`, not off the probe.**
    §3.3 requires resolution to go through exactly one function, and the
    hazard it names is a codebase holding two namespace objects for one
    backend: `array_api_compat.array_namespace` on a NumPy array returns
    `array_api_compat.numpy` rather than `numpy` (A.7.5), and I7's `is` then
    raises on arrays that legitimately share a namespace. A second direct
    `__array_namespace__()` call agrees with the resolver today and is exactly
    the second spelling that rule exists to forbid.

    **The feature probe below must stay, and must stay ahead of that call.**
    It is what makes routing through the resolver safe rather than circular:
    `namespace_of` falls back to `array_api_compat` for any object without the
    method, so a numpy older than 2.0 would resolve to `array_api_compat.numpy`
    -- the two-objects hazard, reached through the very rule meant to prevent
    it -- or fail naming `akriti[torch]`, an extra that has nothing to do with
    this caller's problem. Refusing first turns both into the `ImportError`
    that names `akriti[numpy]`.
    """
    try:
        numpy = import_module("numpy")
    except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
        if exc.name != "numpy":
            raise
        raise ImportError(
            "this input carries no array to derive an array namespace from, "
            "so building one needs numpy, which is not installed. Install "
            "`akriti[numpy]`, or pass an array instead of a Python list."
        ) from exc

    try:
        installed_version = metadata.version("numpy")
    except metadata.PackageNotFoundError as exc:
        raise ImportError(
            "numpy distribution metadata is unavailable; install "
            "`akriti[numpy]` (numpy >=2.0)."
        ) from exc
    except ValueError as exc:
        raise ImportError(
            "numpy distribution metadata could not be read; install "
            "`akriti[numpy]` (numpy >=2.0)."
        ) from exc
    try:
        epoch, release, unstable = _parse_optional_version(installed_version)
    except (TypeError, ValueError) as exc:
        raise ImportError(
            f"could not parse the installed numpy version {installed_version!r}; "
            "install `akriti[numpy]` (numpy >=2.0)."
        ) from exc
    floor = (0, (2, 0, 0))
    if (epoch, release) < floor or ((epoch, release) == floor and unstable):
        raise ImportError(
            f"numpy >=2.0 is required for Python-row inputs (found "
            f"{installed_version!r}); install `akriti[numpy]`."
        )

    probe = numpy.empty(0)
    if not hasattr(probe, "__array_namespace__"):  # pragma: no cover - old numpy
        raise ImportError(
            "the installed numpy has no main-namespace array API "
            "(`__array_namespace__`), which is required from numpy >=2.0; "
            "install `akriti[numpy]`."
        )
    return namespace_of(probe)


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


def _require_float64_exact(column: Array, xp: Any, what: str) -> None:
    """Refuse an integral column `float64` cannot hold exactly. I2, §6.1.

    `_require_int32_range`'s argument, one column over. That guard exists
    because "the cast does not report"; neither does this one, and the
    consequence here is worse than a wrong value, because §3.1's invariants
    run on the *converted* columns. Above `2**53` the float64 grid is coarser
    than the integers, so a violation can be rounded away before the check
    that exists to catch it:

        birth = 2**53 + 1, death = 2**53      # I6: death < birth
        astype(float64)  ->  both 9007199254740992.0

    and a diagram that should have raised is stored as a zero-persistence bar
    -- clean, plausible and wrong, which is §9's category. The same pair one
    ULP lower raises correctly, so whether an invalid bar is caught would
    otherwise depend on its magnitude.

    **Integral dtypes only.** A float input has already been rounded to the
    grid by whoever built it, and there is nothing left for this to detect;
    `float(2**53 + 1)` is `2**53` in Python before any adapter is called.
    What this catches is an exactly-representable input we would be the ones
    to damage.

    The bounds are read with `int`, never `float`. `float(xp.max(column))` on
    the very value this refuses returns `2**53`, so the comparison would be
    made in the arithmetic whose limits it is testing and the check would pass
    itself.
    """
    if not xp.isdtype(column.dtype, "integral") or int(column.shape[0]) == 0:
        return
    low, high = int(xp.min(column)), int(xp.max(column))
    if low < -_FLOAT64_EXACT_INT or high > _FLOAT64_EXACT_INT:
        raise ValueError(
            f"{what} holds an integer outside +/-2**53, which is the range "
            "float64 represents exactly; §6.1 stores coordinates as float64, "
            f"so converting would round it. Its range is [{low}, {high}] -- "
            "pass float64 directly if the rounding is intended"
        )


def _as_float64(column: Array, xp: Any, what: str = "a coordinate column") -> Array:
    """I2, §6.1: storage is the namespace's own `float64`, whatever arrived."""
    _require_real(column, xp, what)
    _require_float64_exact(column, xp, what)
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
    # `_require_float64_exact`'s check on the one path that has no column to
    # check: GUDHI's `persistence()` form reaches storage as Python scalars,
    # and the `float()` below is where an exactly-representable integer would
    # be rounded onto the float64 grid -- silently, and before §3.1 sees the
    # row. Restricted to `Integral` for the reason given there.
    if isinstance(value, numbers.Integral) and abs(int(value)) > _FLOAT64_EXACT_INT:
        raise ValueError(
            f"{where} is {value}, an integer outside +/-2**53, which is the "
            "range float64 represents exactly; §6.1 stores coordinates as "
            "float64, so converting would round it"
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
    """Repair small representational `death < birth` rows. §3.1.

    Returns the (possibly repaired) deaths and a report of what was repaired:
    the count, which the caller records as `provenance["clamped_rows"]` (§8),
    and the largest gap absorbed, which `_warn_clamped` reports. §3.1 requires
    the warning -- a silent repair is a backend defect absorbed without trace
    -- but this function does not issue it, so that a batch adapter can warn
    once for the whole call.

    A positive gap is repaired only when it is at most eight local downward
    float64 ULPs of `birth`. The local spacing is
    `birth - nextafter(birth, -inf)` on finite rows after the coordinates have
    been converted to float64. This is a representational threshold, not a
    persistence-significance tolerance. Larger violations are left exactly as
    they arrived, so §3.1's I6 check raises on them and names the magnitude --
    "a backend that returns `death < birth` has a bug ... and we surface it
    rather than absorb it".
    """
    # Only a row with two finite coordinates can carry an I6 violation worth
    # repairing: an `inf` death is an essential bar (§5) and violates nothing,
    # and any other non-finite value violates I4 or I5, which are core's to
    # refuse rather than the adapter's to absorb. Both operands are masked to
    # zero rather than subtracted and filtered afterwards, because `inf - inf`
    # is `nan` and raises an invalid-operation warning on the way. Finding the
    # candidate first also keeps valid extreme rows out of every arithmetic
    # operation below.
    comparable = xp.isfinite(births) & xp.isfinite(deaths)
    candidate = comparable & (deaths < births)
    zero = xp.zeros_like(births)
    safe_birth = xp.where(candidate, births, zero)
    safe_death = xp.where(candidate, deaths, zero)

    # `nextafter` on zero or a subnormal produces a subnormal and can raise
    # under strict NumPy floating-point errors. Probe a benign normal value in
    # those lanes, then select the exact minimum-subnormal spacing they share.
    subnormal = xp.abs(safe_birth) <= _FLOAT64_SMALLEST_NORMAL
    spacing_probe = xp.where(subnormal, xp.ones_like(safe_birth), safe_birth)
    below = xp.nextafter(spacing_probe, xp.full_like(spacing_probe, -math.inf))
    normal_spacing = spacing_probe - below
    spacing = xp.where(
        subnormal,
        xp.full_like(safe_birth, _FLOAT64_MIN_SUBNORMAL),
        normal_spacing,
    )
    tolerance = xp.where(candidate, _CLAMP_ULPS * spacing, zero)

    # Compare only against the local tolerance; do not form a wide
    # `birth - death` on a candidate that will be refused. Clip the threshold
    # at the minimum finite float so a candidate one ULP above that endpoint
    # does not overflow while subtracting eight ULPs. The subtraction is zero
    # on non-candidates and bounded by the local tolerance elsewhere.
    minimum = -xp.finfo(safe_birth.dtype).max
    at_lower_edge = safe_birth <= minimum + tolerance
    threshold_birth = xp.where(at_lower_edge, zero, safe_birth)
    threshold = xp.where(
        at_lower_edge,
        xp.full_like(safe_birth, minimum),
        threshold_birth - tolerance,
    )
    repair = candidate & (safe_death >= threshold)
    n_repaired = int(xp.sum(xp.astype(repair, xp.int64)))
    total = int(births.shape[0])
    if n_repaired == 0:
        return deaths, _Clamped(0, total, 0.0)

    repaired_birth = xp.where(repair, births, zero)
    repaired_death = xp.where(repair, deaths, zero)
    gap = repaired_birth - repaired_death
    worst = float(xp.max(gap))
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

    The caller's `provenance` and `params` are kept, **except for §8's eight
    reserved `provenance` keys, which are refused outright**
    (`_ADAPTER_OWNED_PROVENANCE`). Each of those names the writer that measured
    it and none of those writers is a caller: `essential_bars` has two, "`from_giotto`
    at construction and `finitize()` later ... the only places that set this
    key"; `essential_bars_source` is "Written only by `from_*`"; the rest are
    counts and dtypes read off the backend's own output. `provenance` exists to
    be auditable rather than assertable, and a fact a caller can state is not
    one a reader can audit.

    **Refused rather than silently overwritten**, which is what stood here.
    Overwriting protects a key only where the adapter happens to write one of
    its own, so the protection was a property of which adapter was called
    rather than of the key: `from_persim` and `from_array` record no
    essential-bar claim (§11, D2), which made them the two adapters that would
    accept one from a caller and return a diagram carrying `essential_bars`
    with no `essential_bars_source` -- the pairing §11 requires in the same
    construction. Refusal is uniform, and it tells a caller their key went
    nowhere instead of discarding it in silence.

    The pairing cannot be enforced in `DiagramMeta` instead, which is why this
    is the adapter's boundary to hold: `finitize` (§5) legitimately writes
    `essential_bars` onto a diagram that has no source to inherit, so a
    constructor rule would refuse the one writer §8 requires.

    A caller with a genuine fact to record -- the host that captured a fixture,
    the dtype an array had before a JSON round trip -- has the rest of the
    mapping, which is open. What is closed is the eight names a reader trusts.

    **`params` is refused on a collision rather than on a name list**, and the
    difference is §8's own: "`provenance` is meant to be backend-agnostic;
    `params` is not". `provenance["essential_bars"]` means the same thing on
    every diagram, so the name can be reserved outright; `params` holds one
    backend's call parameters, so a key that one adapter measures is ordinary
    caller data at the other four. Reserving `reduced_homology` by name would
    refuse a `from_array` caller recording what produced the array they are
    adapting, which is exactly what `params` is for.

    So the rule is the narrower one: a caller may state any `params` key **the
    adapter did not measure in this call**. Today that refuses exactly
    `from_giotto`'s `reduced_homology` (§5.1 gives it one writer, and it is the
    argument, not the caller's mapping) and nothing else, since no other
    adapter writes `params` at all. It stays correct without a list to
    maintain if one later does.

    Refused rather than overwritten for the reason the provenance keys are: a
    caller who passes `params={"reduced_homology": False}` alongside
    `reduced_homology=True` has contradicted themselves, and the useful answer
    says so. Overwriting silently discards one of the two, and which one
    survives is invisible in the result.

    Both mappings are required to *be* mappings before anything is merged into
    them; `_as_metadata_mapping` carries the three ways the previous
    `dict(x or {})` got that wrong.

    `backend` and `backend_version` are refused outright rather than merged.
    An unknown field raises `TypeError` from `DiagramMeta` itself, naming the
    field, so a misspelled `filtraton=` cannot vanish into a diagram that
    reports nothing.

    `coeff_field` is checked here rather than only in `_coeff_field`, which
    the three adapters D17 excludes never call. §11 excuses `from_array`,
    `from_persim` and `from_giotto` from *recording* a field; it does not make
    `coeff_field="two"` admissible on them, and §8 types the field `int |
    None` for every diagram however it was built.

    The converted value is written *back*, which is the half an earlier
    version left out. `_require_coeff_field` admits any `numbers.Integral` so
    that a field read out of an array is not refused for being an `int64` --
    and then storing that `int64` unconverted puts a value in `coeff_field`
    that §8's `int | None` does not describe. `from_gudhi` and `from_ripser`
    already store a builtin `int`, having gone through `_coeff_field`;
    assigning here is what makes the other three agree rather than differ by
    which adapter was called.

    Since this line is a *narrowing* and not the check, an `int64` that
    reached `DiagramMeta` unconverted would be refused there rather than
    stored: the type validates its five scalar fields and both mapping fields
    at the public boundary. What the line buys is that a caller's `int64` is
    accepted at all, uniformly across the five adapters.
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

    caller_provenance = _as_metadata_mapping(
        meta.pop("provenance", None), what="provenance"
    )
    caller_params = _as_metadata_mapping(meta.pop("params", None), what="params")
    reserved_keys = sorted(_ADAPTER_OWNED_PROVENANCE.intersection(caller_provenance))
    if reserved_keys:
        raise TypeError(
            f"provenance[{reserved_keys[0]!r}] is recorded by the adapter and "
            "cannot be passed in: §8 reserves this key for the writer that "
            "measured it, and a fact a caller can state is not one a reader "
            f"can audit (RFC-0001 §8, §11). Refused: {reserved_keys}"
        )
    caller_provenance.update(provenance)

    measured_params = params or {}
    contested = sorted(set(measured_params).intersection(caller_params))
    if contested:
        raise TypeError(
            f"params[{contested[0]!r}] is measured by this adapter from its "
            "own argument, which is that key's one writer, so a second value "
            "for it in params= is a contradiction rather than a fact "
            "(RFC-0001 §8; §5.1 for this key). Pass it as the keyword "
            f"argument alone. Refused: {contested}"
        )
    caller_params.update(measured_params)

    return DiagramMeta(
        backend=backend,
        backend_version=backend_version,
        params=caller_params,
        provenance=caller_provenance,
        **meta,
    )


def _as_metadata_mapping(stated: Any, *, what: str) -> dict[str, Any]:
    """A caller's `params` or `provenance`, as the mapping §8 types it.

    `dict(stated or {})` stood here, and it answered one mistake three ways.
    A falsy non-mapping -- `0`, `False`, `""`, `[]`, `set()` -- was absorbed by
    the `or` and **silently discarded**, so `provenance=0` built a diagram
    that recorded nothing and said nothing, which is the outcome `_build_meta`
    refuses `backend=` and `DiagramMeta` refuses an unknown field to prevent.
    A truthy one reached `dict()` and raised its words -- `dictionary update
    sequence element #0 has length 1` -- naming neither this library, this
    adapter, nor the argument. And a sequence of pairs was accepted outright,
    storing a mapping built from an argument that is not one.

    `DiagramMeta(provenance=0)` raises, so the adapter was looser than the
    type it wraps on precisely the path §11 makes it the boundary for.

    **`None` is "stated nothing", not a bad mapping**, on `_coeff_field`'s
    reading of `coeff_field=None`: §8 makes every field optional and spells an
    absent value `None`, so a caller who has no `provenance` and passes the
    field anyway has said the same thing as a caller who omitted it.

    Only the container is checked here. Key and value admissibility is §8's
    JSON rule, which `DiagramMeta` enforces on the assembled mapping, after
    the adapter's own keys have been merged in.
    """
    if stated is None:
        return {}
    if not isinstance(stated, Mapping):
        raise TypeError(
            f"{what}= must be a str-keyed mapping (RFC-0001 §8); got "
            f"{type(stated).__name__}"
        )
    return dict(stated)


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
    """A caller's coefficient field, narrowed to §8's `int | None`.

    **This is a conversion, not the type check.** `DiagramMeta` refuses a
    `coeff_field` that is not an exact builtin `int`, so `coeff_field="two"`
    recorded with `coeff_field_source = "caller"` -- a provenance entry that
    reads as authoritative and describes no field at all, the outcome D17's
    source key exists to prevent -- is refused whether or not this function
    runs.

    What this adds is the widening the type deliberately does not carry:
    `numbers.Integral` rather than `int`, so that a field read out of an array
    is not refused for being an `int64`, converted here because storing that
    `int64` unconverted would put a value in `coeff_field` that §8's
    `int | None` does not describe and §10.2 cannot serialise. The widening
    belongs at the adapter boundary because that is where a caller's array
    scalar actually arrives.

    `bool` is excluded for `_as_degree`'s reason: the field of one element is
    not what a caller means by `coeff_field=True`. `DiagramMeta` excludes it
    too, so the two agree rather than one relying on the other.
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
            raise TypeError(
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


def _columns_from_gudhi_intervals(
    arr: Array, xp: Any, *, dim: int | None
) -> tuple[Array, Array, Array]:
    """Read GUDHI's interval-array form, which is rank-2 `(n, 2)` only. §11."""
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(
            "GUDHI's interval array form is only rank-2 (n, 2) with dim=; "
            f"got shape {tuple(arr.shape)}"
        )
    if dim is None:
        raise TypeError(
            "a GUDHI (n, 2) interval array carries no homological degree; pass dim=<k>"
        )
    degree = _as_degree(dim)
    return xp.full((arr.shape[0],), degree, dtype=xp.int32), arr[:, 0], arr[:, 1]


def _normalised_column_names(columns: Any) -> tuple[tuple[str, str], ...]:
    """The checks on `columns=` that need no array at all. §10.3.

    Returns each entry as `(as the caller spelled it, casefolded)`.

    Split from `_named_columns` so that §10.3's ordering rule -- "MUST raise on
    the argument, before `arr` is inspected, so the failure does not depend on
    the data" -- is what the call order expresses rather than a comment.
    **Everything decidable from `columns=` alone lives here**: its type, its
    entries' types, the `diagram_id` refusal, the recognised vocabulary, and
    §10.3's cardinality rule that `birth` and `death` appear exactly once each
    and `dim` at most once. Only the length agreement with `arr`'s width needs
    the array, and that is `_named_columns`.

    That split is what lets `columns=` settle §11's degree question by itself.
    With every name recognised, none repeated, and `birth`/`death` both
    present, a two-entry `columns` can only be `(birth, death)` and a
    three-entry one can only also name `dim` -- so the width never has to be
    consulted to know which case the caller is in.

    **`diagram_id` is refused here, ahead of the shape check, deliberately.** A
    table headed `diagram_id,dim,birth,death` is a batch CSV, and that caller
    needs the `.akd` format rather than the true and unhelpful news that
    `from_array` reads `(n, 2)` and `(n, 3)`. It is the one name whose meaning
    survives being the wrong width.
    """
    if isinstance(columns, (str, bytes, bytearray)) or not isinstance(
        columns, Sequence
    ):
        raise TypeError(
            "columns= must be a sequence of strings, one per array column; "
            f"got {type(columns).__name__}"
        )
    names: list[tuple[str, str]] = []
    for index, name in enumerate(columns):
        if not isinstance(name, str):
            raise TypeError(
                f"columns[{index}] must be a string column name; got "
                f"{type(name).__name__}"
            )
        normal = name.casefold()
        if normal == "diagram_id":
            raise TypeError(
                "column 'diagram_id' identifies a batch and cannot be read by "
                "from_array; use the normative .akd batch format"
            )
        # The caller's own spelling is carried alongside the folded one so that
        # a message about `columns=["Birth", "DEATH", "Xyz"]` quotes what they
        # typed. Matching is case-insensitive; complaining need not be.
        names.append((name, normal))

    # §10.3's vocabulary and cardinality rules, both decidable here. A name
    # that is not one of the three MUST raise rather than fall through to
    # position -- an unrecognised name is the one case where the positional
    # reading has been actively contradicted.
    seen: list[str] = []
    for index, (spelled, normal) in enumerate(names):
        if normal not in _COLUMN_NAMES:
            raise ValueError(
                f"unknown column name {spelled!r} at columns[{index}]; expected "
                "birth, death or dim"
            )
        if normal in seen:
            raise ValueError(f"duplicate column name {spelled!r} in columns=")
        seen.append(normal)

    # "`columns` MUST name `birth` and `death` exactly once each, and `dim` at
    # most once" (§10.3). The exactly-once half is already given by the
    # duplicate check above, so what is left is the presence of each. A
    # repeated name and a missing one are one defect seen from two ends --
    # `["birth", "birth", "dim"]` names two births and no death -- and neither
    # is resolvable by falling back to position, the argument having been
    # supplied precisely to override position.
    missing = [name for name in ("birth", "death") if name not in seen]
    if missing:
        raise ValueError(
            f"columns= is missing required column name(s): {', '.join(missing)}"
        )
    return tuple(names)


def _named_columns(
    names: tuple[tuple[str, str], ...], n_columns: int
) -> tuple[str, ...]:
    """The one `columns=` rule that needs `arr`: its width. §10.3, §11.

    Everything else was decided in `_normalised_column_names`, on the argument
    alone and before `arr` was touched, which is §10.3's ordering rule. What
    is left cannot be: "`columns` MUST have one entry per column of `arr`, and
    a length disagreement MUST raise" names both sides, so it is the one check
    the data participates in.

    `names` therefore already holds distinct recognised names including
    `birth` and `death`, so `n_columns` being 2 or 3 makes this a valid header
    for that width, and the caller's order is returned as-is.
    """
    if len(names) != n_columns:
        raise ValueError(
            f"columns= has length {len(names)}, but the array has {n_columns} columns"
        )
    return tuple(normal for _, normal in names)


def _columns_from_named_table(
    arr: Array, xp: Any, *, spelled: tuple[tuple[str, str], ...], dim: int | None
) -> tuple[Array, Array, Array]:
    """Read a named `(n,2)`/`(n,3)` array, with names winning over position.

    Takes `columns=` **already validated** -- `from_array` runs
    `_normalised_column_names` before it resolves a namespace, so that §10.3's
    "MUST raise on the argument, before `arr` is inspected" is expressed by
    where the call sits rather than asserted in a comment here. What is left is
    the order §10.3 and §11 put the remaining two steps in: §11's shape
    refusal, then the width agreement that only means anything once the width
    is one this adapter reads.
    """
    if arr.ndim != 2 or arr.shape[1] not in (2, 3):
        raise ValueError(
            "expected an array of shape (n, 2) or (n, 3) (RFC-0001 §11); got "
            f"shape {tuple(arr.shape)}"
        )
    names = _named_columns(spelled, int(arr.shape[1]))
    positions = {name: index for index, name in enumerate(names)}
    if len(names) == 3 and dim is not None:
        raise TypeError(
            "a named (n, 3) array already carries a dim column, so external "
            "dim= is a conflicting second source"
        )
    if len(names) == 2 and dim is None:
        raise TypeError(
            "a named (n, 2) array carries no homological degree; pass dim=<k>"
        )
    if len(names) == 2:
        degree = _as_degree(dim)
        dims = xp.full((arr.shape[0],), degree, dtype=xp.int32)
    else:
        dims = arr[:, positions["dim"]]
    return dims, arr[:, positions["birth"]], arr[:, positions["death"]]


def _is_array_block(obj: Any) -> bool:
    """Whether `obj` is array-shaped enough to ask for a namespace. §3.3.

    Duck-typed on the three attributes every path below reads -- `ndim` and
    `shape` for the `(n, 2)` check, `dtype` for `_as_float64` -- rather than
    on `__array_namespace__`, which torch does not expose and which
    `namespace_of` exists to work around (§3.3, A.7).
    """
    return all(hasattr(obj, name) for name in ("ndim", "shape", "dtype"))


def _first_array_block(blocks: Sequence[Any]) -> Any | None:
    """The first real array block, skipping Python row blocks. §3.3, §11.

    **Every block is checked on the way past, not just the one returned.** A
    degree list holds `(n, 2)` arrays or blocks of rows (§11); a member that
    is neither -- `None`, a bare number, a mapping -- used to reach
    `namespace_of` untouched and fail there, with `array_namespace requires at
    least one non-scalar array input`. That is the namespace's words for a
    refusal §11 owes the caller, naming neither the adapter, the argument, nor
    which block was wrong, and it is exactly what `_columns_from_degree_list`
    wraps `asarray` to prevent one path over. Both paths owe the same refusal;
    this is the third.

    It belongs here rather than in that loop because this function runs first:
    the namespace these adapters build with is resolved from what this returns,
    so `[3]` failed a line earlier than `[None]` did -- one on the resolution,
    one inside the loop -- for no reason a caller could see.

    `None` where every block is a Python row block, which is the array-less
    input `_namespace_for_rows` answers.
    """
    first: Any | None = None
    for degree, block in enumerate(blocks):
        if _is_row_sequence(block):
            continue
        if not _is_array_block(block):
            raise ValueError(
                f"diagram at index {degree} must have shape (n, 2) "
                f"(RFC-0001 §11); this block is neither an array nor a "
                f"sequence of rows: {_abbreviated(block)}"
            )
        if first is None:
            first = block
    return first


def _abbreviated(obj: Any, limit: int = 120) -> str:
    """`repr(obj)`, bounded, for a message that quotes a caller's block.

    A degree block can hold every bar of a large diagram, and an error that
    pastes forty thousand rows into the terminal is one the caller scrolls
    past rather than reads.
    """
    text = repr(obj)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _wrong_adapter_hint(block: Any) -> str:
    """Name `from_gudhi` when an unreadable block is a `persistence()` row.

    The one input form this failure has a specific cause for. GUDHI's
    `persistence()` returns `list[(dim, (birth, death))]` and Ripser's `dgms`
    is `list[(n, 2)]`; §11 accepts each at one adapter, and handing the first
    to `from_ripser` or `from_persim` reads a `(dim, (birth, death))` row as a
    degree block, which is not rectangular and cannot convert. The reverse
    mistake is already refused by name from `_columns_from_pairs`, so this
    closes the asymmetry rather than adding a rule.

    Silent on every other unreadable block: a jagged `(n, 2)` block is a
    mis-shaped block, not a misrouted call, and a hint naming another adapter
    would send that caller somewhere their data does not belong.
    """
    if not _is_persistence_row(block):
        return ""
    return (
        ". A (dim, (birth, death)) row is GUDHI's persistence() form, which "
        "this adapter does not accept -- pass it to from_gudhi"
    )


def _columns_from_degree_list(
    dgms: Sequence[Any], xp: Any, *, degrees: Sequence[int] | None = None
) -> tuple[Array, Array, Array]:
    """Stack `list[(n, 2)]`, where list position selects the degree. §11.

    Ripser's `dgms` and persim's input share this shape. Row order within a
    degree is preserved exactly; degrees follow the list, which is the
    backend's own order for the merged diagram.

    **`degrees` is what position means, and the two backends disagree about
    it** (D20). For Ripser and persim, position *is* the homological degree
    and `degrees` stays `None`. For GUDHI's sklearn-compatible form the
    position is an index into the `homology_dimensions` list the caller passed
    the transformer, which the returned object does not carry -- measured,
    `homology_dimensions=[2, 0]` returns H2 first and H0 second -- so
    `from_gudhi` passes that list here and position is resolved through it.
    Reading position as degree for that form would mislabel every diagram
    computed with a reordered or non-contiguous list, silently and plausibly.

    Errors name the **index**, not the degree, and for `degrees` they are two
    different numbers. The index is what the caller can point at in the object
    they passed; the degree is what this function decided it meant.

    Every block must share one namespace (I7). `core.py` checks the same thing
    across the diagrams of a batch and says why: without it, `xp.concat` would
    "either raise something opaque from the backend or silently coerce a
    foreign array", and a silently coerced diagram then validates cleanly with
    the mixed input already erased.
    """
    dim_blocks, birth_blocks, death_blocks = [], [], []
    for index, block in enumerate(dgms):
        degree = index if degrees is None else degrees[index]
        if not _is_row_sequence(block):
            block_xp = namespace_of(block)
            if block_xp is not xp:
                raise ValueError(
                    f"the diagram at index {index} has array namespace "
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
            #
            # Wrapped because a block whose rows are not all the same width
            # fails *inside* `asarray`, so the shape refusal below is never
            # reached and the caller gets the namespace's words instead:
            # NumPy's "setting an array element with a sequence. The requested
            # array has an inhomogeneous shape after 1 dimensions" names
            # neither the adapter, the argument, nor which block was wrong.
            # Both paths owe §11's refusal, and they give the same one.
            try:
                block = (
                    xp.empty((0, 2), dtype=xp.float64)
                    if len(block) == 0
                    else xp.asarray(block)
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"diagram at index {index} must have shape (n, 2) "
                    f"(RFC-0001 §11); these rows could not be read as an "
                    f"array at all: {_abbreviated(block)}" + _wrong_adapter_hint(block)
                ) from exc
        if block.ndim != 2 or block.shape[1] != 2:
            raise ValueError(
                f"diagram at index {index} must have shape (n, 2) "
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
    return {"source_dtype": str(arr.dtype)} if hasattr(arr, "dtype") else {}


def _require_infinite_infinity_values(value: object) -> None:
    """Refuse a giotto `infinity_values` that finitizes essential bars. §5.

    Takes `object` rather than the parameter's declared `float` so the runtime
    checks below are checks rather than assertions: the whole point is the
    argument a caller actually passed, which the annotation describes and does
    not enforce.

    **`None` is separated from every other rejected value because it is
    giotto's default.** A caller who passes `infinity_values=vr.infinity_values`
    off an unconfigured transformer lands there, and "not a real number" would
    describe the type rather than what went wrong with their data.

    `bool` is excluded ahead of `numbers.Real`, which it registers as, on
    `_as_coordinate`'s precedent: `infinity_values=True` is not a filtration
    value, and letting it through would report it as a finite sentinel of 1.0.

    `nan` and `-inf` fall out of the equality rather than needing their own
    branch -- `nan == inf` is `False` -- and both are rejected on the same
    ground, being deaths §5 does not recognise for a class that never dies.
    """
    if value is None:
        raise ValueError(
            "infinity_values=None is giotto's default, and it encodes classes "
            "still alive at the cutoff as a death of max_edge_length -- a "
            "finite sentinel, which RFC-0001 §5 forbids and which cannot be "
            "told apart from a bar that genuinely died at that value. "
            "Re-run the transformer with infinity_values=numpy.inf and adapt "
            "that output; this adapter will not guess which rows were "
            "essential."
        )
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(
            "infinity_values= must be the float you gave the transformer -- "
            "pass vr.infinity_values off the fitted transformer (RFC-0001 §5, "
            f"§11); got {value!r} of type {type(value).__name__}"
        )
    if float(value) != math.inf:
        raise ValueError(
            f"infinity_values={value!r} finitizes every class still alive at "
            "the cutoff, and RFC-0001 §5 stores essential bars as inf: a "
            "finite sentinel is 'unrecoverable ... indistinguishable from a "
            "genuine bar that happened to die at that value'. Only numpy.inf "
            "is accepted; re-run the transformer with it."
        )


def _reject_impossible_reduced_homology(
    dims: Array, deaths: Array, xp: Any, *, sample: int
) -> None:
    """Refuse a `reduced_homology=False` declaration the data contradicts. §11.

    Non-reduced H0 of a nonempty space carries a class that never dies, so a
    diagram declared both `reduced_homology=False` and `infinity_values=inf`
    whose degree-0 deaths are *all* finite is not merely unlikely but
    impossible: one of the two declarations is false. The adapter cannot tell
    which, so the error names both (`N11-9`, `N11-10`).

    **The predicate is three-termed, and the middle term is the one that took
    measuring.** "Every degree-0 death is finite" is a reduction over an empty
    selection, so it holds vacuously of a diagram with no degree-0 row at all
    -- and `homology_dimensions` excluding 0 is an ordinary giotto request
    rather than a perverse one. Appendix A.10 measures `(1, 2)`, `(1,)` and
    `(2,)` all returning non-empty, correct arrays with zero H0 rows. A check
    scoped only to non-empty diagrams refuses every one of them, which is why
    `N11-11` requires all three terms to hold before it raises:

    1. the diagram is non-empty;
    2. it carries at least one degree-0 row;
    3. every degree-0 death is finite.

    Term 1 is implied by term 2 -- a diagram with an H0 row has a row -- and is
    still spelled out, because the clause states it and a reader checking the
    code against the clause should find all three rather than have to
    reconstruct the implication.

    **This does not extend to `reduced_homology=True`**, where the essential H0
    class is dropped by design and its absence proves nothing. §11 takes that
    half on trust and says so; the caller guards this call accordingly.

    Called per sample rather than over the batch, `N11-11`'s subject being a
    diagram and a giotto batch being many of them. One impossible sample is
    enough to refuse the call: the diagram it would become carries
    `essential_bars="faithful"` (§5.1) over bars whose essential class was
    finitized upstream.

    Runs on the rows as giotto returned them, *before* the padding mask, on the
    same grounds the degree validation above it gives: padding is a row giotto
    could have emitted, and a mode that changed which arrays are acceptable
    input is exactly what §11.1's "the caller decides" does not mean.
    """
    # Term 1. A giotto batch may carry a sample with no bars at all (§4.2),
    # and §3.2 and §8.2 both treat an empty diagram as valid.
    if int(dims.shape[0]) == 0:
        return

    # Term 2. The H0 sub-diagram is what has to exist, not the diagram.
    degree_zero = dims == 0
    if not bool(xp.any(degree_zero)):
        return

    # Term 3. Spelled as "no H0 death is inf" rather than by masking the
    # deaths, so the array API standard's boolean-indexing rules are not
    # needed for a question that is a reduction over a conjunction.
    if bool(xp.any(degree_zero & xp.isinf(deaths))):
        return

    raise ValueError(
        f"sample {sample} declares reduced_homology=False and "
        "infinity_values=inf, but every one of its degree-0 deaths is finite. "
        "Non-reduced H0 of a nonempty space carries a class that never dies, "
        "so that combination is impossible and one of the two declarations is "
        "false (RFC-0001 §11, §5.1). Either the transformer was built with "
        "reduced_homology=True, or it was built with a finite "
        "infinity_values -- giotto's default of None finitizes the essential "
        "class to max_edge_length -- and this adapter cannot tell which from "
        "the array. Pass both values off the fitted transformer."
    )


def _source_dtype_of_blocks(blocks: Sequence[Any]) -> dict[str, str]:
    """§8's `source_dtype` over a per-degree list. §11.

    Ripser's `dgms` and persim's input are lists of one array per degree, and
    nothing requires those arrays to share a dtype. Reading the first block
    alone -- which is what this did -- records `"float32"` for a
    `[float32, float64]` input, a statement about degree 0 presented as one
    about the diagram.

    **A mixed list records no key rather than one of the two.** §8 gives
    `source_dtype` one slot and no vocabulary for a disagreement, and the
    alternatives are worse in the direction §8 exists to prevent: a compound
    `"float32,float64"` invents a spelling no reader is expecting, and picking
    either member is the bug being fixed. Absence already carries "no dtype
    could be determined" for the array-less case, and this widens it to "not
    one dtype" rather than adding a second way to be wrong.

    **The diagram itself is not refused.** The bars are valid whatever their
    incoming dtypes were; it is only the record about them that cannot be
    written, and §3.1's surface-a-violation rule is about invariants on data,
    not about metadata a backend never promised. Neither backend emits a mixed
    list in practice, so this describes a hand-assembled input.
    """
    dtypes = {str(b.dtype) for b in blocks if hasattr(b, "dtype")}
    return {"source_dtype": dtypes.pop()} if len(dtypes) == 1 else {}


# ---------------------------------------------------------------------------
# §11's five adapters
# ---------------------------------------------------------------------------


def _from_gudhi_sklearn(
    obj: Any,
    *,
    homology_dimensions: Sequence[int],
    dim: int | None,
    field: int | None,
    provenance: dict[str, Any],
    meta: dict[str, Any],
) -> PersistenceDiagram:
    """GUDHI's sklearn-compatible `list[(n, 2)]` for one sample. §11, D20.

    Split out of `from_gudhi` rather than written as a third arm inside it,
    because it is the arm with the argument checks: the other two dispatch on
    the object and this one dispatches on a keyword, and interleaving the two
    made the order of the refusals hard to read.

    **Every refusal here is on the arguments, before the blocks are looked
    at.** §10.3 imposes the same ordering on `from_array`'s `columns`, §5 on
    `finitize`'s `at`, and §6.3 on the cross-namespace check, all for one
    reason: a failure that depends on the data is one the caller reproduces
    only with that data in hand.
    """
    # `N11-3`. Refused first, being a contradiction in the call itself rather
    # than a disagreement with the object: a per-degree list carries every
    # degree at once, so a single `dim` is not a thing the caller can be
    # asserting about it. Same grounds as the `persistence()` list's refusal.
    if dim is not None:
        raise TypeError(
            "homology_dimensions= and dim= cannot both be given (RFC-0001 "
            "§11): GUDHI's sklearn-compatible output carries one block per "
            "requested degree and homology_dimensions already names all of "
            "them, so dim= would be a second source for one fact. dim= is "
            "for persistence_intervals_in_dimension(k), which is a single "
            "(n, 2) array stating no degree."
        )

    if not _is_row_sequence(homology_dimensions):
        raise TypeError(
            "homology_dimensions= must be the sequence of degrees you gave "
            "the transformer, in that order (RFC-0001 §11); got "
            f"{homology_dimensions!r} of type "
            f"{type(homology_dimensions).__name__}"
        )
    if not _is_row_sequence(obj):
        raise TypeError(
            "homology_dimensions= goes with GUDHI's sklearn-compatible "
            "output, which is a list of (n, 2) blocks, one per requested "
            f"degree (RFC-0001 §11); got {type(obj).__name__}. A single "
            "(n, 2) array from persistence_intervals_in_dimension(k) states "
            "no degree either, and takes dim=k instead."
        )

    # `N11-2`'s second half. A length disagreement is a `ValueError` rather
    # than a `TypeError` because the argument is of the right kind and the
    # wrong size, and it is checkable without reading a single bar: the
    # caller either passed a different transformer's dimension list or
    # indexed one nesting level too few, and both are worth naming.
    if len(homology_dimensions) != len(obj):
        raise ValueError(
            f"homology_dimensions= names {len(homology_dimensions)} degrees "
            f"but the diagram list holds {len(obj)} blocks (RFC-0001 §11); "
            "they index each other, so the two must agree. "
            "RipsPersistence().fit_transform(X) returns one such list *per "
            "sample* -- if you passed the whole result, index it first: "
            "from_gudhi(result[i], homology_dimensions=...)."
        )

    degrees = [
        _as_degree(
            value,
            where=f"homology_dimensions[{position}]",
        )
        for position, value in enumerate(homology_dimensions)
    ]

    first_array = next((b for b in obj if not _is_row_sequence(b)), None)
    xp = namespace_of(first_array) if first_array is not None else _namespace_for_rows()
    dims, births, deaths = _columns_from_degree_list(obj, xp, degrees=degrees)
    provenance.update(_source_dtype_of_blocks(obj))

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


def from_gudhi(
    obj: Any,
    *,
    dim: int | None = None,
    homology_dimensions: Sequence[int] | None = None,
    **meta: Any,
) -> PersistenceDiagram:
    """A GUDHI persistence result as a `PersistenceDiagram`. §11.

    Accepts all three measured input forms (§11, D20):

    - `SimplexTree.persistence()` -> `list[(dim, (birth, death))]`, carrying
      every degree at once. This form has no array, so the diagram is
      numpy-backed; see the module docstring.
    - `SimplexTree.persistence_intervals_in_dimension(k)` -> `(n, 2)`, which
      states no degree, so `dim=k` is required.
    - The sklearn-compatible family (`RipsPersistence` and its siblings) ->
      per sample, a `list[(n, 2)]`, which requires `homology_dimensions=`.

    **The sklearn form cannot identify itself, so `homology_dimensions` is
    what selects it** (D20). Its shape is identical to Ripser's
    `Rips().fit_transform(X)` and to persim's input -- §11 says so -- and it
    is nonetheless *not the same object*: Ripser's list position **is** the
    homological degree, while GUDHI's is a position in the
    `homology_dimensions` list the caller passed the transformer, which the
    returned value does not carry. Measured: `homology_dimensions=[2, 0]`
    returns H2 first and H0 second, and `[1]` returns a length-one list
    holding H1. An adapter reading position as degree would mislabel every
    diagram computed with a reordered or non-contiguous list -- silently,
    plausibly, and wrongly. So the discriminator is **the presence of the
    keyword**, never the shape, and the fact the object lacks is required
    from the caller, on §5.1's `reduced_homology` precedent.

    `dim=` alongside it raises `TypeError` on the same grounds the
    `persistence()` list is refused it: both carry every degree at once, and a
    single degree is not a thing the caller can be asserting about them. A
    `homology_dimensions` whose length disagrees with the outer list raises
    `ValueError`.

    **This adapter takes one sample, and `fit_transform` returns many.**
    `RipsPersistence().fit_transform(X)` gives `list[list[(n, 2)]]`, so the
    caller indexes -- `from_gudhi(result[i], homology_dimensions=[0, 1])` --
    and `DiagramBatch.from_diagrams` (§4.2) assembles the batch. `from_giotto`
    faces the same situation and resolves it the other way, because giotto's
    leading axis is unambiguous and a `list` of `list`s is not: a
    `list[(n, 2)]` for one sample over several degrees is structurally the
    same object as a list of several one-degree samples. Guessing which would
    be wrong on data rather than on type, so this adapter does not guess.

    **Extended persistence is out of scope, and only partly detectable.**
    `extended_persistence()` returns a four-element list of
    `list[(dim, (birth, death))]` -- ordinary, relative, extended+ and extended-
    -- which is structurally distinct from `persistence()`'s flat list, so the
    outer list itself is
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

    if homology_dimensions is not None:
        return _from_gudhi_sklearn(
            obj,
            homology_dimensions=homology_dimensions,
            dim=dim,
            field=field,
            provenance=provenance,
            meta=meta,
        )

    if isinstance(obj, tuple):
        raise TypeError(
            "from_gudhi rejects a flat tuple of persistence rows; pass the "
            "ordinary persistence() result as a list"
        )
    if not _is_row_sequence(obj) and not hasattr(obj, "ndim"):
        raise TypeError(
            "from_gudhi accepts SimplexTree.persistence() output "
            "(list[(dim, (birth, death))]) or a rank-2 interval array; got "
            f"{type(obj).__name__}"
        )
    # Two branches, not three: the gate above already refused everything that
    # is neither a row sequence nor array-shaped, so `_is_row_sequence` is a
    # partition here and a third arm would be unreachable code carrying a
    # message no input can produce.
    if not _is_row_sequence(obj):
        xp = namespace_of(obj)
        dims, births, deaths = _columns_from_gudhi_intervals(obj, xp, dim=dim)
        provenance.update(_source_dtype(obj))
    else:
        if dim is not None:
            raise TypeError(
                "a persistence() list already carries a degree per bar, so "
                "dim= would be a second source for one fact"
            )
        # `N11-2`. Reached only because `homology_dimensions` was omitted, so
        # a list of `(n, 2)` blocks is the sklearn form missing the one
        # argument that makes it readable -- not a malformed `persistence()`
        # result, which is what `_columns_from_pairs` would report it as.
        if _is_degree_indexed_block_list(obj):
            raise TypeError(
                "this looks like GUDHI's sklearn-compatible output "
                "(list[(n, 2)], one block per requested degree), which needs "
                "homology_dimensions= (RFC-0001 §11, D20). List position is "
                "an index into the homology_dimensions you gave the "
                "transformer, not the homological degree -- "
                "homology_dimensions=[2, 0] returns H2 first -- and the "
                "returned object does not carry it, so this adapter will not "
                "guess. Pass homology_dimensions= off the fitted transformer. "
                "If these blocks came from Ripser or persim, where position "
                "*is* the degree, use from_ripser or from_persim instead."
            )
        xp = _namespace_for_rows()
        dims, births, deaths = _columns_from_pairs(obj, xp)

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
    stated_filtration = meta.get("filtration")
    if stated_filtration not in (None, "rips"):
        raise TypeError(
            "from_ripser always records filtration='rips'; conflicting "
            f"filtration={stated_filtration!r} cannot be accepted"
        )
    meta["filtration"] = "rips"

    if isinstance(obj, Mapping):
        if "dgms" not in obj:
            raise ValueError(
                "a ripser() result carries its diagrams under 'dgms'; this "
                f"mapping has keys {sorted(map(str, obj.keys()))}"
            )
        dgms: Sequence[Any] = obj["dgms"]
        if not isinstance(dgms, list):
            raise TypeError(
                "a ripser() result carries its diagrams under 'dgms' as a "
                "list of (n, 2) arrays, degree by index (RFC-0001 §11); this "
                f"mapping's 'dgms' is {type(dgms).__name__}"
            )
    elif isinstance(obj, list):
        dgms = obj
    else:
        raise TypeError(
            "from_ripser accepts ripser(X) output (a dict with 'dgms') or "
            f"Rips().fit_transform(X) output (list[(n, 2)]); got "
            f"{type(obj).__name__}"
        )

    field, source = _coeff_field(meta, _RIPSER_DEFAULT_COEFF_FIELD)
    first_array = _first_array_block(dgms)
    xp = namespace_of(first_array) if first_array is not None else _namespace_for_rows()
    dims, births, deaths = _columns_from_degree_list(dgms, xp)

    provenance = {
        "essential_bars": "faithful",
        "essential_bars_source": "faithful",
        "coeff_field_source": source,
    }
    provenance.update(_source_dtype_of_blocks(dgms))

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
    if not isinstance(obj, list):
        raise TypeError(
            f"from_persim accepts list[(n, 2)], degree by index; got "
            f"{type(obj).__name__}"
        )

    first_array = _first_array_block(obj)
    xp = namespace_of(first_array) if first_array is not None else _namespace_for_rows()
    dims, births, deaths = _columns_from_degree_list(obj, xp)

    provenance = _source_dtype_of_blocks(obj)
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
    arr: Array,
    *,
    columns: Sequence[str] | None = None,
    dim: int | None = None,
    **meta: Any,
) -> PersistenceDiagram:
    """A raw array as a `PersistenceDiagram`. §11.

    Two accepted shapes:

    - `(n, 2)` with an explicit `dim=k`.
    - `(n, 3)` with columns `(birth, death, dim)` -- giotto's order, matched
      deliberately (§11). A supplied `columns=` sequence names columns in
      their actual order, case-insensitively, and wins over position (§10.3).

    The array's namespace is preserved (§3.3): an array from JAX gives a
    JAX-backed diagram. What is converted is dtype (§6.1, I2).

    **No backend, so no version and no essential-bar claim.**
    `backend_version` is `None`, and `provenance` records neither
    `essential_bars` nor `coeff_field_source`: nothing about a caller's array
    says what produced it, and §8's keys record what an adapter observed
    rather than what it hopes.
    """
    # §10.3: `columns=` is judged on the argument **before `arr` is inspected
    # at all** -- ahead of the row-sequence refusal below and ahead of
    # `namespace_of`, so an invalid header fails identically whatever it was
    # passed beside, including an object that would raise on attribute access.
    spelled = None if columns is None else _normalised_column_names(columns)
    if isinstance(arr, (list, tuple, str, bytes, bytearray)):
        raise TypeError(
            "from_array accepts an array object, not a Python row/list "
            f"({type(arr).__name__}); pass an array with an "
            "__array_namespace__"
        )
    xp = namespace_of(arr)
    if spelled is None:
        dims, births, deaths = _columns_from_table(arr, xp, dim=dim)
    else:
        dims, births, deaths = _columns_from_named_table(
            arr, xp, spelled=spelled, dim=dim
        )
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
    infinity_values: float,
    strip_padding: bool | None = None,
    **meta: Any,
) -> DiagramBatch:
    """A giotto-tda transform result as a `DiagramBatch`. §11, §11.1, §5.1.

    Input is `(n_samples, n_bars, 3)` with columns `(birth, death, dim)`.

    **`infinity_values` is required, and only `inf` is accepted.** giotto's
    own default is `None`, which assigns every class still alive at the cutoff
    a death of `max_edge_length` -- a finite sentinel, which §5 refuses in the
    first row of its table of rejected conventions: "Unrecoverable. The bar is
    now indistinguishable from a genuine bar that happened to die at that
    value." Nothing in the returned array separates that row from a bar that
    really died there, so the adapter cannot repair it and does not try. Pass
    `infinity_values=vr.infinity_values` off the fitted transformer, having
    constructed the transformer with `infinity_values=numpy.inf`.

    This is measured rather than inferred, and **both settings are committed**:
    `tests/fixtures/giotto_output.json` carries one capture per setting, from a
    single run in the pinned environment. Under `samples`
    (`infinity_values=inf`) the `reduced_homology=False` case has 40 H0 bars,
    exactly one death at `inf`, and no death equal to `max_edge_length`. Under
    `samples_default_infinity` (giotto's own `None`) the same call gives the
    same 40 bars with that one death sitting at `max_edge_length` instead --
    the finite sentinel this adapter refuses, and the array §11.2 requires the
    refusal to be tested against.

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
    recorded in `params`, being a raw fact of the original call (§5.1) -- so
    `params={"reduced_homology": ...}` is refused rather than merged, this
    argument being the one writer §5.1 gives that key.

    **That derivation is §5.1's, and it is only sound because `inf` is the one
    accepted `infinity_values`.** §5.1 states the rule as though
    `reduced_homology` were its only input; on giotto's default sentinel,
    `"faithful"` would describe a diagram whose essential bar had been
    finitized, and `"lost_upstream"` -- which §5.1 scopes to the H0 class
    reduced homology removes -- would say nothing about an essential H1 class
    finitized the same way. Narrowing the accepted input is what makes the
    unchanged rule true, rather than reinterpreting it.

    `infinity_values` is deliberately *not* recorded in `params`, unlike
    `reduced_homology`. §8 requires every `params` value to be
    JSON-representable and `inf` has no JSON spelling, and with one accepted
    value there is no fact left to record: every diagram this returns has
    `infinity_values=inf`, and its essential bars are visible as `inf` in the
    data itself.

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
    if isinstance(arr, (list, tuple, str, bytes, bytearray)):
        raise TypeError(
            "from_giotto accepts an array object, not a Python row/list "
            f"({type(arr).__name__}); pass an array with an "
            "__array_namespace__"
        )
    xp = namespace_of(arr)
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

    _require_infinite_infinity_values(infinity_values)

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
    # measured one: `_diagram_from_columns` adds the key before calling
    # `_build_meta`, so the mapping real construction passes carries it and the
    # mapping this preflight passes must too. The divergence it guards against
    # has outlived the rule that first caused it -- a caller's `clamped_rows`
    # used to be accepted and overwritten, and is now refused outright
    # (`_ADAPTER_OWNED_PROVENANCE`) -- because the refusal reads the *caller's*
    # mapping and this dict is the adapter's. Whichever way the rule falls,
    # a zero-sample batch must fall the same way as a one-sample batch.
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
        # Spelled with an index per axis rather than `arr[i]`. The array API
        # standard requires a rank-3 array be indexed by three indices or an
        # ellipsis; NumPy, torch and JAX all accept the short form anyway, and
        # `array_api_strict` -- the conformance backend this project is
        # written against (§3.3) -- raises `IndexError`. `from_giotto` was the
        # one adapter that could not take a strict array.
        sample = arr[i, :, :]
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
        # §11's impossibility check, run on the constructed diagram rather
        # than on the raw columns, and therefore *after* §3.1's invariants.
        # The ordering is the point: an array carrying a `nan` death or an
        # `-inf` birth is not a diagram whose H0 deaths can meaningfully be
        # called finite, and §3.1's one answer to a violation is to surface
        # it. Checking first would report an impossible *declaration* for an
        # array whose real defect is a malformed *coordinate*, sending the
        # caller to re-read their transformer's arguments over what is
        # actually a corrupt row.
        #
        # Clamping having already run changes nothing here: `_clamp_i6` moves
        # a death up to its birth, both finite, and never turns an `inf` into
        # a finite death or the reverse.
        if not reduced_homology:
            _reject_impossible_reduced_homology(
                diagram.dims, diagram.deaths, xp, sample=i
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


# ---------------------------------------------------------------------------
# Exporters (§10.3)
# ---------------------------------------------------------------------------


def _warn_export_loss(*, arrays: bool = False, batch: bool = False) -> None:
    """Warn once about information not represented by an exporter. §10.3."""
    losses = ["all DiagramMeta"]
    if arrays:
        losses.append("global inter-degree order")
    if batch:
        losses.append("empty-member/cardinality information")
    warnings.warn(
        "export discards " + " and ".join(losses) + ".",
        UserWarning,
        stacklevel=3,
    )


def _export_rows(
    obj: PersistenceDiagram | DiagramBatch,
) -> tuple[bool, list[tuple[int, int, float, float]]]:
    """Return ``(is_batch, rows)`` in stored order, without metadata. §10.3."""
    if isinstance(obj, PersistenceDiagram):
        return False, [
            (0, int(dim), float(birth), float(death))
            for dim, birth, death in zip(obj.dims, obj.births, obj.deaths, strict=True)
        ]
    if isinstance(obj, DiagramBatch):
        rows: list[tuple[int, int, float, float]] = []
        bounds = [int(offset) for offset in obj.offsets]
        for diagram_id, (lo, hi) in enumerate(itertools.pairwise(bounds)):
            rows.extend(
                (diagram_id, int(dim), float(birth), float(death))
                for dim, birth, death in zip(
                    obj.dims[lo:hi], obj.births[lo:hi], obj.deaths[lo:hi], strict=True
                )
            )
        return True, rows
    raise TypeError(
        f"exporters accept PersistenceDiagram or DiagramBatch; got {type(obj).__name__}"
    )


def to_arrays(obj: PersistenceDiagram) -> dict[int, Array]:
    """Export one diagram as degree-keyed ``(n, 2)`` arrays. §10.3.

    The input must be a ``PersistenceDiagram`` backed by one array namespace.
    Rows retain their within-degree order, duplicate multiplicity, float64
    dtype and essential ``+inf`` deaths. Metadata and global inter-degree row
    order are intentionally not represented.
    """
    if not isinstance(obj, PersistenceDiagram):
        raise TypeError(
            "to_arrays accepts PersistenceDiagram only; DiagramBatch has no "
            "single-diagram output shape"
        )
    degrees = sorted({int(dim) for dim in obj.dims})
    result = {
        degree: obj.xp.stack(
            (obj.births[obj.dims == degree], obj.deaths[obj.dims == degree]),
            axis=1,
        )
        for degree in degrees
    }
    _warn_export_loss(arrays=True)
    return result


def to_csv(obj: PersistenceDiagram | DiagramBatch) -> str:
    """Export bars as UTF-8-compatible CSV text with an LF header. §10.3.

    A diagram uses ``dim,birth,death``. A batch prepends ``diagram_id`` and
    preserves member and row order; empty members have no rows and therefore
    cannot be recovered from the CSV alone. Floats use Python's shortest
    round-trip spelling, including the literal ``inf`` and signed zero.
    """
    is_batch, rows = _export_rows(obj)
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        ("diagram_id", "dim", "birth", "death")
        if is_batch
        else ("dim", "birth", "death")
    )
    for diagram_id, dim, birth, death in rows:
        values = (
            (diagram_id, dim, repr(birth), repr(death))
            if is_batch
            else (dim, repr(birth), repr(death))
        )
        writer.writerow(values)
    text = output.getvalue()
    _warn_export_loss(batch=is_batch)
    return text


def _parquet_version_is_supported(version: str) -> bool:
    """Return whether a parsed pyarrow version meets the 25.0.0 floor."""
    epoch, release, unstable = _parse_optional_version(version)
    floor = (0, (25, 0, 0))
    return (epoch, release) > floor or ((epoch, release) == floor and not unstable)


def _load_pyarrow() -> Any:
    """Import and validate optional pyarrow on every call. §10.1, §10.3."""
    try:
        pyarrow = import_module("pyarrow")
    except ModuleNotFoundError as exc:
        if exc.name != "pyarrow":
            raise
        raise ImportError(
            "to_parquet requires pyarrow >=25.0.0; install `akriti[parquet]`"
        ) from exc
    try:
        version = metadata.version("pyarrow")
    except metadata.PackageNotFoundError as exc:
        raise ImportError(
            "pyarrow metadata is unavailable; install `akriti[parquet]`"
        ) from exc
    try:
        supported = _parquet_version_is_supported(version)
    except (TypeError, ValueError) as exc:
        raise ImportError(
            "could not parse the installed pyarrow version; install `akriti[parquet]`"
        ) from exc
    if not supported:
        raise ImportError(
            f"pyarrow >=25.0.0 is required (found {version!r}); "
            "install `akriti[parquet]`"
        )
    return pyarrow


def to_parquet(obj: PersistenceDiagram | DiagramBatch) -> Any:
    """Export bars as an explicit-schema ``pyarrow.Table``. §10.3.

    The optional pyarrow dependency is imported and version-checked only in
    this function. Metadata is not attached to the table. Diagram rows retain
    stored order, duplicates, signed zero and essential ``+inf`` deaths; batch
    rows carry an int64 ``diagram_id`` column.
    """
    is_batch, rows = _export_rows(obj)
    pyarrow = _load_pyarrow()
    if is_batch:
        schema = pyarrow.schema(
            [
                pyarrow.field("diagram_id", pyarrow.int64()),
                pyarrow.field("dim", pyarrow.int32()),
                pyarrow.field("birth", pyarrow.float64()),
                pyarrow.field("death", pyarrow.float64()),
            ]
        )
        columns = [
            pyarrow.array([row[0] for row in rows], type=pyarrow.int64()),
            pyarrow.array([row[1] for row in rows], type=pyarrow.int32()),
            pyarrow.array([row[2] for row in rows], type=pyarrow.float64()),
            pyarrow.array([row[3] for row in rows], type=pyarrow.float64()),
        ]
    else:
        schema = pyarrow.schema(
            [
                pyarrow.field("dim", pyarrow.int32()),
                pyarrow.field("birth", pyarrow.float64()),
                pyarrow.field("death", pyarrow.float64()),
            ]
        )
        columns = [
            pyarrow.array([row[1] for row in rows], type=pyarrow.int32()),
            pyarrow.array([row[2] for row in rows], type=pyarrow.float64()),
            pyarrow.array([row[3] for row in rows], type=pyarrow.float64()),
        ]
    table = pyarrow.Table.from_arrays(columns, schema=schema)
    _warn_export_loss(batch=is_batch)
    return table
