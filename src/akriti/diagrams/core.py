"""The canonical persistence diagram types. RFC-0001.

`PersistenceDiagram` (§3) and `DiagramBatch` (§4) are the interchange layer:
every backend adapter produces one of these, every `core/`/`castle/` routine
consumes one. This module and `adapters.py` carry no third-party dependency
required at import time: all namespace resolution goes through
`namespace_of`, with a lazy optional `array_api_compat` fallback. Everything
is written against the Python array API, never against `numpy` directly
(§3.3, §10.1 requirement 2).

Neither type has a mutating method (I8, and the equivalent reasoning for
`DiagramBatch`): every operation that looks like one constructs and returns a
new instance rather than modifying `self`. That is what makes
`DiagramBatch.__getitem__`'s zero-copy views (§4.2) safe -- nothing offered
here can write through one view and corrupt a sibling.

**Public construction copies caller-owned arrays.** `@dataclass(frozen=True)`
prevents rebinding a field, and the constructors copy each input buffer before
validation, so later mutation through a caller's original reference cannot
change a constructed object. The array API supplies no read-only array
contract, however: a caller can still write directly through `d.births[0]` (or
through a batch/view), and this module cannot prevent that residual mutation.
Its methods never write in place, which is the contract that makes the
deliberate zero-copy views safe.

**Which operations are available under `jax.jit` is settled in §3.3 and
§4.3, and is not re-argued here.** Each one says so in its own docstring;
the reasoning lives in the RFC, so there is one copy of it to keep true
rather than a second that can drift. As an index only:

- *Traceable*: `essential`, `persistence`, `canonical`, `n_bars` on a
  diagram; `essential`, `persistence`, `bar_counts`, `__len__` on a batch.
- *Eager-only*: `dim`, `dimensions`, `finite`; `finitize` in **every** mode,
  not only `at="drop"`; `DiagramBatch.__getitem__` and everything routed
  through it, `DiagramBatch.canonical()` included -- which is
  shape-preserving and still not traceable, the distinction §3.3 draws; and
  everything returning a Python `bool` or `str`.

`==` and `allclose` raise `ValueError` rather than answering when the two
sides are backed by different array namespaces (§6.3); see
`_require_shared_namespace` for the alternatives weighed and for what raising
costs. `same_provenance` touches no array and is deliberately exempt.
"""

from __future__ import annotations

import array
import hashlib
import itertools
import math
import operator
import re
import struct
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from importlib import import_module, metadata
from typing import Any, NoReturn, SupportsIndex, TypeAlias

# The Python array API standard's `Array` (§3): any object resolved by
# `namespace_of`, either through native `__array_namespace__` or the optional
# `array_api_compat` fallback, never `np.ndarray` specifically. There is no
# practical Protocol that captures the full surface (arithmetic, boolean
# indexing, `.shape`, `.dtype`, ...) without becoming its own maintenance
# burden, so this is a documented `Any`, not a structural type. Unrelated to
# the stdlib `array` module imported above, which appears only in
# `_big_endian_block`.
Array: TypeAlias = Any

__all__ = [
    "Array",
    "DiagramBatch",
    "DiagramMeta",
    "PersistenceDiagram",
    "namespace_of",
]

# Domain-separation tags for the two content hashes (§8.1, §8.2). Distinct
# tags are what make a one-diagram batch structurally unable to collide with
# the diagram it wraps, rather than merely unlikely to.
_DIAGRAM_HASH_TAG = b"akriti.PersistenceDiagram.v1\x00"
_BATCH_HASH_TAG = b"akriti.DiagramBatch.v1\x00"
_ARRAY_API_COMPAT_MIN_VERSION = (1, 15, 0)
_COMPAT_ARRAY_NAMESPACE: Any = None
_OPTIONAL_VERSION_RE = re.compile(
    r"^(?P<epoch>\d+!)?(?P<release>\d+(?:\.\d+)*)"
    r"(?P<pre>a|b|rc|alpha|beta|c|pre|preview)?(?P<pre_n>\d+)?"
    r"(?P<post>(?:\.post|post)\d+)?(?P<dev>(?:\.dev|dev)\d+)?"
    r"(?:\+[0-9A-Za-z]+(?:[._-][0-9A-Za-z]+)*)?$",
    re.IGNORECASE,
)


def _parse_optional_version(version: str) -> tuple[int, tuple[int, ...], bool]:
    """Parse the PEP 440 forms needed by optional dependency floor checks."""
    match = _OPTIONAL_VERSION_RE.fullmatch(version)
    if match is None:
        raise ValueError(version)
    epoch = int((match.group("epoch") or "0").rstrip("!"))
    release = tuple(int(part) for part in match.group("release").split("."))
    while len(release) > 3 and release[-1] == 0:
        release = release[:-1]
    if len(release) < 3:
        release += (0,) * (3 - len(release))
    unstable = match.group("pre") is not None or (
        match.group("dev") is not None and match.group("post") is None
    )
    return epoch, release, unstable


def _clear_namespace_cache() -> None:
    """Clear the validated optional-compat resolver cache (tests/internal)."""
    global _COMPAT_ARRAY_NAMESPACE
    _COMPAT_ARRAY_NAMESPACE = None


def namespace_of(x: Array) -> Any:
    """Resolve an array's Python array-API namespace. RFC-0001 §3.3.

    Native ``__array_namespace__`` is authoritative and is always preferred.
    Backends that conform in substance but do not expose that method (notably
    torch) use the optional ``array_api_compat`` fallback. The fallback import
    is deliberately function-scoped so the default install remains free of
    optional backend dependencies. A successfully imported and validated
    fallback is cached; metadata is checked against the ``>=1.15.0`` floor
    once per process. Missing dependencies, below-floor versions, and
    unparseable version metadata raise actionable ``ImportError`` exceptions.

    Assumes *x* is an array object or a backend object supported by
    ``array_api_compat``. When the fallback dependency is unavailable, callers
    receive an actionable error naming the extra that supplies it.
    """
    global _COMPAT_ARRAY_NAMESPACE
    native = getattr(x, "__array_namespace__", None)
    if native is not None:
        return native()
    if _COMPAT_ARRAY_NAMESPACE is not None:
        return _COMPAT_ARRAY_NAMESPACE(x)
    try:
        compat = import_module("array_api_compat")
    except ModuleNotFoundError as exc:
        if exc.name != "array_api_compat":
            raise
        raise ImportError(
            "array namespace resolution for this backend requires "
            "array-api-compat; install akriti[torch]"
        ) from exc
    try:
        installed_version = metadata.version("array-api-compat")
    except metadata.PackageNotFoundError as exc:
        raise ImportError(
            "array-api-compat is unavailable for this backend; install akriti[torch]"
        ) from exc
    try:
        epoch, release, unstable = _parse_optional_version(installed_version)
    except (TypeError, ValueError) as exc:
        # `TypeError` as well as `ValueError`: `_parse_optional_version` reaches
        # `fullmatch` on whatever the metadata held, which raises `TypeError`
        # rather than `ValueError` when that is not a `str`. Catching one of the
        # two let a malformed distribution escape as a bare `TypeError` naming
        # `re`, while `adapters.py`'s identical numpy check turned the same
        # input into the actionable `ImportError` below. Whichever way the
        # metadata is broken, both resolvers must name the extra that fixes it.
        raise ImportError(
            "could not parse the installed array-api-compat version; install "
            "akriti[torch]"
        ) from exc
    floor = (0, _ARRAY_API_COMPAT_MIN_VERSION)
    if (epoch, release) < floor or ((epoch, release) == floor and unstable):
        raise ImportError(
            "array-api-compat >=1.15.0 is required for this backend; install "
            "akriti[torch]"
        )
    _COMPAT_ARRAY_NAMESPACE = compat.array_namespace
    return _COMPAT_ARRAY_NAMESPACE(x)


class _FrozenList(list[Any]):
    """A JSON list that cannot be mutated after metadata construction. §8."""

    @staticmethod
    def _read_only() -> NoReturn:
        raise TypeError("metadata containers are read-only")

    def __setitem__(self, index: Any, value: Any) -> NoReturn:
        self._read_only()

    def __delitem__(self, index: Any) -> NoReturn:
        self._read_only()

    def append(self, value: Any) -> NoReturn:
        self._read_only()

    def extend(self, values: Any) -> NoReturn:
        self._read_only()

    def insert(self, index: SupportsIndex, value: Any) -> NoReturn:
        self._read_only()

    def pop(self, index: SupportsIndex = -1) -> NoReturn:
        self._read_only()

    def remove(self, value: Any) -> NoReturn:
        self._read_only()

    def clear(self) -> NoReturn:
        self._read_only()

    def reverse(self) -> NoReturn:
        self._read_only()

    def sort(self, *, key: Any = None, reverse: bool = False) -> NoReturn:
        self._read_only()

    # Mypy requires in-place operators to return Self even when they always
    # raise; NoReturn states the runtime contract more accurately.
    def __iadd__(self, values: Iterable[Any]) -> NoReturn:  # type: ignore[misc]
        self._read_only()

    def __imul__(self, count: SupportsIndex) -> NoReturn:
        self._read_only()


class _FrozenDict(dict[str, Any]):
    """A JSON mapping that cannot be mutated after metadata construction. §8."""

    @staticmethod
    def _read_only() -> NoReturn:
        raise TypeError("metadata containers are read-only")

    def __setitem__(self, key: str, value: Any) -> NoReturn:
        self._read_only()

    def __delitem__(self, key: str) -> NoReturn:
        self._read_only()

    def clear(self) -> NoReturn:
        self._read_only()

    def pop(self, key: str, default: Any = None) -> NoReturn:
        self._read_only()

    def popitem(self) -> NoReturn:
        self._read_only()

    def setdefault(self, key: str, default: Any = None) -> NoReturn:
        self._read_only()

    def update(self, *args: Any, **kwargs: Any) -> NoReturn:
        self._read_only()

    # See _FrozenList.__iadd__: this operator is deliberately non-returning.
    def __ior__(self, value: Any) -> NoReturn:  # type: ignore[misc]
        self._read_only()


def _freeze_json_value(value: Any) -> Any:
    """Recursively copy JSON-compatible containers, leaving scalars alone. §8."""
    if isinstance(value, list):
        return _FrozenList(_freeze_json_value(item) for item in value)
    if isinstance(value, Mapping):
        return _FrozenDict(
            (key, _freeze_json_value(item)) for key, item in value.items()
        )
    return value


def _freeze_json_mapping(mapping: Mapping[str, Any]) -> _FrozenDict:
    """Own and freeze one metadata mapping, including all nested containers. §8."""
    # `dict` retains the existing constructor behavior for mapping-like
    # inputs, while the value walk below closes the nested aliasing hole.
    copied = dict(mapping)
    return _FrozenDict(
        (key, _freeze_json_value(value)) for key, value in copied.items()
    )


def _require_json_representable(mapping: Mapping[str, Any], field: str) -> None:
    """Require §8 metadata values to survive the normative JSON round trip.

    The public ``DiagramMeta`` boundary owns this check so hand-built
    diagrams and every adapter receive the same contract. Values are limited
    to exact builtin JSON scalar types, lists, and recursively str-keyed
    mappings. Tuples, NumPy scalars, and non-finite floats are rejected.
    """
    scalars = (str, bool, int, float)

    def check_keys(m: Mapping[Any, Any], path: str) -> None:
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


def _validate_scalar_fields(meta: DiagramMeta) -> None:
    """§8's declared types for the five scalar fields, enforced here.

    `params` and `provenance` are walked by `_require_json_representable`;
    without this the five fields beside them were typed in the annotation and
    checked nowhere, so `DiagramMeta(filtration=3.5)` constructed and every
    `from_*` adapter (§11) passed `**meta` straight into it.

    The check belongs here for §3.1's reason and not in five adapters: a rule
    stated only as an obligation on writers is one every future writer has to
    remember, and the adapters are not the only writer -- `dataclasses.replace`
    and direct construction reach these fields too.

    **Exact builtin types, matching `_require_json_representable`'s house
    rule**, because §10.2 stores `meta.json` as UTF-8 JSON and the same reason
    applies one field over: a `Path` filtration or a NumPy `coeff_field` is a
    diagram that satisfies §3.1 and §8 completely and cannot be saved, failing
    arbitrarily far from whatever wrote it. Adapters may widen at their own
    boundary -- `adapters._require_coeff_field` admits any `numbers.Integral`
    and converts -- so the widening stays where a caller's array scalar
    actually arrives, and the stored value is a builtin either way.

    `bool` is refused for `coeff_field` on `adapters._as_degree`'s ground:
    `coeff_field=True` records the field of one element, which is a
    coincidence of the language rather than anything a caller means.
    """
    for name in ("filtration", "backend", "backend_version", "description"):
        value = getattr(meta, name)
        if value is not None and type(value) is not str:
            raise TypeError(
                f"{name} is {value!r} of type {type(value).__name__}; §8 "
                f"types it `str | None`, and §10.2 stores it as JSON"
            )
    # `type(...) is not int` already excludes `bool`, `type(True)` being
    # `bool`; spelled exactly rather than with `isinstance` for that reason.
    field_value = meta.coeff_field
    if field_value is not None and type(field_value) is not int:
        raise TypeError(
            f"coeff_field is {field_value!r} of type "
            f"{type(field_value).__name__}; §8 types it `int | None` -- the "
            "characteristic of the field homology was computed over (§9.3)"
        )


# §8's closed vocabulary for `provenance["essential_bars"]`, in document
# order. Two of the four are an adapter's verdict on what the backend gave it
# and two are `finitize`'s record of what it did (§5); `essential_bars_source`
# ranges over the first two alone, which is why that check spells its pair
# separately rather than slicing this tuple.
_ESSENTIAL_BARS_VALUES = (
    "faithful",
    "lost_upstream",
    "finitized_at",
    "finitized_dropped",
)


def _validate_provenance(provenance: Mapping[str, Any]) -> None:
    """§8's consistency rules for the reserved `essential_bars*` keys.

    §8 states these as requirements on *writers* ("MUST be kept consistent
    with it, not merely written alongside it"). They are enforced here, at
    construction, for §3.1's reason one type over: a requirement only the
    writer can violate is one every future writer has to remember, and
    `finitize` is not the only writer -- every `from_*` adapter (§11) sets
    these keys through this constructor and none of them passes through
    `finitize`'s code path.

    Four rules, all quoted directly from §8; nothing beyond them is checked.
    Unknown keys are left alone -- `provenance` is an open mapping and §8
    reserves names within it rather than closing it.
    """
    bars = provenance.get("essential_bars")
    dropped_present = "essential_bars_dropped" in provenance

    # §8 closes the vocabulary: "one of `faithful`, `lost_upstream`,
    # `finitized_at`, `finitized_dropped`". Checked before the two iff-rules
    # below, because both are stated against specific members of it -- a
    # misspelled `"finitized_droped"` would otherwise pass the dropped-count
    # rule by failing to be the value that requires a count, reporting nothing
    # while meaning nothing.
    # Keyed on presence rather than on `bars is not None`: an explicit
    # `{"essential_bars": None}` is a key that is there holding no verdict,
    # which is not one of the four and is not the same as saying nothing.
    if "essential_bars" in provenance and bars not in _ESSENTIAL_BARS_VALUES:
        raise ValueError(
            f"provenance['essential_bars'] is {bars!r}; §8 closes the "
            f"vocabulary to {', '.join(map(repr, _ESSENTIAL_BARS_VALUES))}"
        )

    # "present iff essential_bars == 'finitized_at'" (§8), and the value is
    # the substituted death, so it MUST be a finite real number: `inf` is the
    # thing finitizing removes, and `NaN` is not a death time. `bool` is
    # refused on `_validate_meta_types`' ground -- `True` is not a death.
    at_present = "essential_bars_finitized_at" in provenance
    if at_present and bars != "finitized_at":
        raise ValueError(
            "provenance['essential_bars_finitized_at'] is present but "
            f"provenance['essential_bars'] is {bars!r}; §8 admits the "
            "substituted death only alongside 'finitized_at'"
        )
    if bars == "finitized_at" and not at_present:
        raise ValueError(
            "provenance['essential_bars'] is 'finitized_at' but "
            "provenance['essential_bars_finitized_at'] is missing; §8 "
            "requires the substituted death alongside it"
        )
    if at_present:
        at_value = provenance["essential_bars_finitized_at"]
        if type(at_value) is bool or not isinstance(at_value, (int, float)):
            raise TypeError(
                "provenance['essential_bars_finitized_at'] is "
                f"{at_value!r} of type {type(at_value).__name__}; §8 types it "
                "as the finite numeric death `finitize` substituted"
            )
        # §8 also requires that death to be **finite**, and there is
        # deliberately no check for it here: §10.2 keeps non-finite floats out
        # of `provenance` entirely, and `_require_json_representable` runs
        # ahead of this function, so `inf` and `NaN` are already refused by a
        # strictly stronger rule. Restating it would add a branch nothing can
        # reach, which is worse than absent -- it reads as live defence and
        # rots untested. `test_a_non_finite_substituted_death_is_rejected`
        # asserts the outcome §8 wants against whichever layer delivers it.

    # "present iff essential_bars == 'finitized_dropped'" (§8). Both
    # directions matter: a stale count outliving a substitution is the failure
    # §8 names, and a drop that forgot its count is the same record half-made.
    if dropped_present and bars != "finitized_dropped":
        raise ValueError(
            "provenance['essential_bars_dropped'] is present but "
            f"provenance['essential_bars'] is {bars!r}; §8 admits the count "
            "only alongside 'finitized_dropped'"
        )
    if bars == "finitized_dropped" and not dropped_present:
        raise ValueError(
            "provenance['essential_bars'] is 'finitized_dropped' but "
            "provenance['essential_bars_dropped'] is missing; §8 requires the "
            "count alongside it"
        )

    # "'faithful' or 'lost_upstream', never a 'finitized_*' value" (§8): the
    # key records the adapter-time verdict, and no adapter emits a finitized
    # one. A copy-forward from `essential_bars` is the specific mistake §5
    # rejects, and this is where it surfaces.
    source = provenance.get("essential_bars_source")
    if source is not None and source not in ("faithful", "lost_upstream"):
        raise ValueError(
            f"provenance['essential_bars_source'] is {source!r}; §8 admits "
            "only 'faithful' or 'lost_upstream' -- it records the adapter's "
            "verdict at computation time, which no 'finitized_*' value can be"
        )


@dataclass(frozen=True)
class DiagramMeta:
    """Filtration, backend, and provenance facts about a diagram. §8.

        All fields are optional -- a diagram typed in by hand from a paper is a
        valid diagram -- but `from_*` adapters MUST populate `backend`,
        `backend_version`, and `provenance`.

    `params` and `provenance` own their complete value trees and are stored as
    read-only dict/list containers, checked for JSON-safe values, and
    `provenance` is checked against §8's reserved-key rules; see `__post_init__`
    and the internal validators.
    """

    filtration: str | None = None
    backend: str | None = None
    backend_version: str | None = None
    coeff_field: int | None = None
    params: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    description: str | None = None

    # Explicitly unhashable, and it has to be said rather than left to the
    # private metadata containers to enforce. `@dataclass(frozen=True, eq=True)`
    # generates
    # a `__hash__`, so without this line `DiagramMeta` would satisfy
    # `isinstance(m, Hashable)` and pass any `Hashable` guard, then raise
    # `TypeError: unhashable type: 'dict'` from inside the generated tuple
    # hash -- an error naming neither this class nor the field responsible.
    # Nothing here hashes a `DiagramMeta`: `content_hash` (§8.1) covers bars
    # only and deliberately excludes metadata. `==` is unaffected because the
    # private containers compare equal to their builtin counterparts.
    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Own, recursively freeze, and validate the two mapping fields. §8.

        `frozen=True` stops `meta.params = {...}`, while the private dict/list
        subclasses below stop ordinary mutation at every nested level. They
        retain builtin list/dict equality and `json.dumps` behavior, so the
        normalized JSON representation remains the RFC's representation.
        Construction copies before validation, so neither the caller's
        mapping nor any nested list or mapping can be aliased into the stored
        metadata. JSON validation precedes reserved-key validation so all
        metadata failures use the same field/path-aware contract.
        """
        _validate_scalar_fields(self)
        object.__setattr__(self, "params", _freeze_json_mapping(self.params))
        object.__setattr__(self, "provenance", _freeze_json_mapping(self.provenance))
        _require_json_representable(self.params, "params")
        _require_json_representable(self.provenance, "provenance")
        _validate_provenance(self.provenance)
        source_present = "coeff_field_source" in self.provenance
        source = self.provenance["coeff_field_source"] if source_present else None
        if source_present and source not in ("caller", "backend_default"):
            raise ValueError(
                f"provenance['coeff_field_source'] is {source!r}; §8 admits "
                "only 'caller' or 'backend_default'"
            )
        if source_present and self.coeff_field is None:
            raise ValueError(
                "provenance['coeff_field_source'] requires a non-None coeff_field"
            )


def _copy_array(x: Array, xp: Any) -> Array:
    """Copy a caller-owned array at a public construction boundary. §3.1."""
    return xp.asarray(x, copy=True)


def _validate_bar_arrays(dims: Array, births: Array, deaths: Array) -> Any:
    """I1-I7, I9: shared invariants for a diagram's coordinate arrays.

    Written once and shared by `PersistenceDiagram` and `DiagramBatch`
    (§4.1): a batch's concatenated buffer must satisfy the same per-bar
    invariants a single diagram does. Returns the shared namespace.

    I9 is checked before I1 deliberately: `shape[0]` on a rank-0 array raises
    `IndexError` rather than reporting the real problem.
    """
    if not (namespace_of(dims) is namespace_of(births) is namespace_of(deaths)):
        raise ValueError("dims, births, and deaths must share one array namespace (I7)")
    xp = namespace_of(dims)

    if not (dims.ndim == births.ndim == deaths.ndim == 1):
        raise ValueError(
            "dims, births, and deaths must be rank-1 (I9); got ndim "
            f"{dims.ndim}, {births.ndim}, {deaths.ndim}"
        )

    if not (dims.shape[0] == births.shape[0] == deaths.shape[0]):
        raise ValueError(
            "dims, births, and deaths must have equal length (I1); got "
            f"{dims.shape[0]}, {births.shape[0]}, {deaths.shape[0]}"
        )

    # Equality against the namespace's own dtype, not `xp.isdtype(...,
    # "real floating")` (§6.1): that predicate admits `float32`, which D3
    # rejects outright, so it would not be a check at all here.
    if births.dtype != xp.float64:
        raise ValueError(f"births must be float64 (I2); got {births.dtype}")
    if deaths.dtype != xp.float64:
        raise ValueError(f"deaths must be float64 (I2); got {deaths.dtype}")
    if dims.dtype != xp.int32:
        raise ValueError(f"dims must be int32 (I2); got {dims.dtype}")

    if bool(xp.any(dims < 0)):
        raise ValueError(
            f"dims must be non-negative (I3); minimum is {int(xp.min(dims))}"
        )

    # Each message names the offending value, not just the invariant. §3.1
    # anticipates I6 violations at the 1e-16 level from filtration arithmetic,
    # where the magnitude *is* the diagnostic -- it decides whether the
    # adapter should clamp (§3.1) or the backend has a real bug -- and a
    # validator that withholds it forces a print statement into the one place
    # this library exists to make unnecessary.
    if bool(xp.any(xp.isnan(births))):
        raise ValueError(f"births must not be NaN (I4); {_count(xp, xp.isnan(births))}")
    if bool(xp.any(xp.isinf(births))):
        raise ValueError(
            "births must be finite -- a class that is never born is not a "
            f"class (I4); {_count(xp, xp.isinf(births))}"
        )

    if bool(xp.any(xp.isnan(deaths))):
        raise ValueError(f"deaths must not be NaN (I5); {_count(xp, xp.isnan(deaths))}")
    # `deaths == -xp.inf`, not `xp.equal(deaths, -xp.inf)`: the standard has
    # guaranteed a Python scalar on the right of an array operator since
    # 2021.12, while accepting one as an argument to a two-array elementwise
    # *function* is a 2024.12 addition. The operator form costs nothing and
    # does not depend on which version a backend negotiates.
    if bool(xp.any(deaths == -xp.inf)):
        raise ValueError(
            f"-inf is not a valid death time (I5); {_count(xp, deaths == -xp.inf)}"
        )

    if not bool(xp.all(deaths >= births)):
        # The reductions below run only on the failure path, so the valid case
        # still costs exactly the one comparison above.
        gap = deaths - births
        raise ValueError(
            "deaths must be >= births elementwise (I6); "
            f"{_count(xp, gap < 0)}, worst by {float(xp.min(gap))}"
        )

    return xp


def _count(xp: Any, mask: Array) -> str:
    """`"k of n rows"` for an error message, from a boolean mask.

    Failure paths only. Kept out of the messages themselves so the six checks
    above stay one line each and read as the invariant list they are.
    """
    return f"{int(xp.sum(xp.astype(mask, xp.int64)))} of {mask.shape[0]} rows"


def _require_shared_namespace(a: Any, b: Any, op: str) -> None:
    """Reject a comparison spanning two array namespaces. §6.3.

    I7 constrains the three arrays *within* a diagram and says nothing across
    two, so a NumPy diagram and a torch diagram of the same bars are each
    valid and still not comparable: this module may not convert either one
    (§3.3 gives it no third-party import and no default backend), so the
    comparison has no well-defined answer to return. Checked before any
    length or value comparison, so the failure does not depend on the data --
    the same ordering rule §5 imposes on `finitize`'s `at`.

    Three alternatives were weighed, and §6.3 records the outcome; the third
    is the one worth naming here because it is the one Python itself would
    reach for.

    - **Pass the mismatch through to `xp.equal`.** Yields whatever the
      backend happens to raise, the opaque failure §4.2 rejects for `concat`.
    - **Return `False`.** A clean, plausible answer asserting the bars
      differ, which may be untrue; §9's whole category of bug is answers of
      that shape.
    - **Return `NotImplemented` from `__eq__`.** Not the same as returning
      `False`: it lets the right-hand side try its own `__eq__` first, and
      only then does Python fall back to identity, i.e. `False`. It is the
      language's own answer for "I cannot compare these", and it keeps `==`
      total, which the chosen behaviour does not.

    **The cost of raising is that `==` is no longer total, and that reaches
    ordinary container operations.** `d in [d1, d2]`, `list.index`,
    `list.remove`, `assertIn`, and any third-party container comparing
    elements will raise `ValueError` rather than answer, whenever a
    cross-namespace diagram is anywhere in the sequence. That is a real cost
    and it is accepted deliberately: a `False` from `in` would assert that a
    diagram with these exact bars is absent from a collection that may well
    contain it, which is the §9-shaped answer this check exists to refuse,
    and a container search is precisely where nobody would think to check.
    Callers who need a total comparison across backends convert at their own
    boundary first (§3.3) -- there is one obvious place to do it and this is
    not it.
    """
    if a.xp is not b.xp:
        # `getattr`, not `.__name__`: a namespace is a module in every backend
        # we have seen, but the standard only requires it to be an object with
        # the right attributes, and an AttributeError raised while building an
        # error message would bury the error it was reporting.
        left = getattr(a.xp, "__name__", repr(a.xp))
        right = getattr(b.xp, "__name__", repr(b.xp))
        raise ValueError(
            f"{op} cannot compare across array namespaces: one side is "
            f"{left}-backed and the other {right}-backed. Convert one side at "
            "the caller's boundary; this module does not (§3.3)."
        )


def _close_within(a: Array, b: Array, rtol: float, atol: float, xp: Any) -> Array:
    """Symmetric elementwise tolerance, broadcast to a pair matrix. §6.3.

    `|a - b| <= atol + rtol * max(|a|, |b|)`.

    **This deliberately diverges from `numpy.allclose`**, which computes
    `atol + rtol * |b|` -- scaling by its second argument alone, so that
    swapping the arguments can change the answer at the boundary. Two
    diagrams either agree to a tolerance or they do not; which one was
    written first is not part of the question, and an `allclose` whose
    result depends on it cannot be relied on by a caller who did not think
    about argument order. §6.3 requires the symmetric form.

    Purely elementwise, so it serves both shapes `allclose` needs: 1-D
    against 1-D for the paired check, and `(n, 1)` against `(1, m)` for the
    full pair matrix.
    """
    return xp.abs(a - b) <= (atol + rtol * xp.maximum(xp.abs(a), xp.abs(b)))


def _deaths_within(
    deaths_a: Array,
    inf_a: Array,
    deaths_b: Array,
    inf_b: Array,
    rtol: float,
    atol: float,
    xp: Any,
) -> Array:
    """`_close_within` for a death column, where `inf` is reachable. §5, §6.3.

    Two disjoint cases, and the tolerance formula runs in neither of them for
    an essential bar. `inf` matches `inf` exactly; a pair with an `inf` on
    exactly one side is incompatible whatever the formula would have said;
    everything else goes through the tolerance. That masking is what decides
    essential bars.

    The zeros substituted below decide nothing. They exist only to keep
    `inf - inf` -> `nan` and `inf - x` -> `inf` out of the arithmetic in
    cells the masks discard anyway. Any finite value would do, and the result
    does not depend on which -- relying instead on `nan` comparing False
    would reach the same answer by an accident of IEEE 754 that a reader has
    to verify rather than read.

    Elementwise, so it broadcasts the same two ways `_close_within` does.
    """
    both_inf = inf_a & inf_b
    both_finite = (~inf_a) & (~inf_b)
    finite_a = xp.where(inf_a, xp.zeros_like(deaths_a), deaths_a)
    finite_b = xp.where(inf_b, xp.zeros_like(deaths_b), deaths_b)
    return both_inf | (both_finite & _close_within(finite_a, finite_b, rtol, atol, xp))


def _perfect_matching_exists(neighbours: list[list[int]], n: int) -> bool:
    """Does a perfect matching exist in a bipartite graph on `n + n` vertices?

    Kuhn's augmenting-path algorithm, iterative. `neighbours[u]` lists the
    right-hand vertices `u` may be matched to.

    **Standard library only, deliberately (§6.3).** `scipy.sparse.csgraph.
    maximum_bipartite_matching` would do this, and §3.3 gives this module the
    standard library and the caller's namespace and nothing else. A lazy
    import on a *comparison* path is a worse trade than §10.1 requirement 2's
    narrow one at the `save`/`load` boundary: serialization is a boundary a
    caller crosses deliberately, while `allclose` is called from inside test
    suites and assertions where an ImportError is pure obstruction.

    Augmenting paths rather than Hopcroft-Karp: O(V*E) against O(E*sqrt(V)),
    on diagrams whose bar counts are in the hundreds to thousands, reached
    only after an O(n^2) edge construction that dominates anyway. `allclose`
    is a verification surface, not a numerical inner loop (§6.3).

    Iterative rather than recursive because the augmenting path can be as
    long as `n`, and a diagram with more bars than the recursion limit is an
    ordinary diagram, not a pathological one.

    **Deliberately no special case for a forced matching**, where every bar
    has exactly one candidate partner. It looks like the fast path worth
    adding and it buys nothing: with single-element adjacency lists, each
    root here either takes its one free partner or fails on the spot, so the
    loop is already linear and a separate path would only add a branch and a
    second thing to keep correct. The cases that *do* pay are upstream in
    `allclose`, where they can be answered with array reductions before any
    of this is built.
    """
    # For each right-hand vertex, the left-hand vertex holding it, or -1.
    held_by = [-1] * n
    for root in range(n):
        seen = bytearray(n)
        # Each frame is a left vertex and its unexplored neighbours. `path`
        # holds the tentative edges from `root` down to the current frame.
        stack: list[tuple[int, Any]] = [(root, iter(neighbours[root]))]
        path: list[tuple[int, int]] = []
        augmented = False
        while stack and not augmented:
            u, candidates = stack[-1]
            for v in candidates:
                if seen[v]:
                    continue
                seen[v] = 1
                path.append((u, v))
                if held_by[v] == -1:
                    # Free vertex: flip every edge along the path we came by.
                    for left, right in path:
                        held_by[right] = left
                    augmented = True
                else:
                    # Taken: recurse into its current holder to displace it.
                    stack.append((held_by[v], iter(neighbours[held_by[v]])))
                break
            else:
                # No candidate left for `u`; back out of the edge that led here.
                stack.pop()
                if path:
                    path.pop()
        if not augmented:
            # `root` cannot be matched even after exhausting every
            # alternating path, so no perfect matching exists. Halting here
            # is what makes the common "these diagrams differ" case cheap.
            return False
    return True


def _big_endian_block(values: Array, typecode: str) -> bytes:
    """One canonical-ordered column as big-endian machine values. §8.1.

    Two paths, which produce identical bytes by construction and MUST stay
    that way -- a divergence would make a published `content_hash` depend on
    which backend recomputed it:

    - The buffer-protocol path, taken when the backend exposes its memory as
      a C-contiguous buffer of exactly the expected format and item size (the
      NumPy case, and §10.1's reasoning already establishes NumPy as the
      ambient backend). It reinterprets the raw machine values and byte-swaps
      the whole block at once. For an IEEE 754 `double` or a two's-complement
      `int32` that is by definition what `struct.pack(">d"/">i", ...)` emits.
    - A per-element `struct.pack` fallback for every backend that does not,
      which includes `array_api_strict` and therefore CI.

    The fast path exists because the alternative is two Python-level scalar
    extractions per bar, each a device synchronisation on a torch- or
    JAX-backed diagram, on a code path `repro/` runs over every diagram in a
    table. `memoryview` is the stdlib buffer protocol, not a NumPy API, so
    §3.3's zero-third-party-dependency requirement is untouched.
    """
    itemsize = struct.calcsize(">" + typecode)
    try:
        view: Any = memoryview(values)
    except (TypeError, BufferError, ValueError):
        view = None

    if (
        view is not None
        and view.c_contiguous
        and view.format == typecode
        and view.itemsize == itemsize
        and array.array(typecode).itemsize == itemsize
    ):
        block = array.array(typecode)
        block.frombytes(view.tobytes())
        if sys.byteorder == "little":
            block.byteswap()
        return block.tobytes()

    fmt = ">" + typecode
    scalar = float if typecode == "d" else int
    return b"".join(struct.pack(fmt, scalar(values[i])) for i in range(values.shape[0]))


# `eq=False` keeps the dataclass from generating a field-by-field `__eq__`;
# §6.3's equality is over the multiset of bars, which no field comparison can
# express. Defining `__eq__` in the body also sets `__hash__` to `None`, so
# `PersistenceDiagram` is unhashable -- deliberately, and the same for
# `DiagramBatch`. Any `__hash__` would have to agree with `==`, and the only
# thing that does is `content_hash` (§8.1): a 64-character digest, not a
# machine word, and it costs a canonical sort to compute. Callers who want it
# ask for it by name.
@dataclass(frozen=True, eq=False)
class PersistenceDiagram:
    """A finite multiset of bars: `(dim, birth, death)` triples. §2, §3.

    Three parallel arrays, one row per bar. `PersistenceDiagram` is
    immutable (I8): every method that looks like a mutation constructs and
    returns a new instance rather than modifying `self`, which is what makes
    `DiagramBatch.__getitem__`'s zero-copy views (§4.2) safe. See the module
    docstring for the exact extent of that guarantee, and for which accessors
    are eager-only (§3.3). The array API cannot make arrays read-only, so a
    caller can still mutate a coordinate directly through `d.births[0]`.
    """

    dims: Array
    births: Array
    deaths: Array
    meta: DiagramMeta = field(default_factory=DiagramMeta)

    def __post_init__(self) -> None:
        """Enforce §3.1. An invalid instance MUST NOT be constructible.

        I1-I7 and I9 are the checkable ones and are checked here. I8 is
        immutability, which `frozen=True` and the absence of any mutating
        method carry structurally instead of at construction.
        """
        object.__setattr__(
            self, "dims", _copy_array(self.dims, namespace_of(self.dims))
        )
        object.__setattr__(
            self, "births", _copy_array(self.births, namespace_of(self.births))
        )
        object.__setattr__(
            self, "deaths", _copy_array(self.deaths, namespace_of(self.deaths))
        )
        _validate_bar_arrays(self.dims, self.births, self.deaths)

    @classmethod
    def _unchecked(
        cls, *, dims: Array, births: Array, deaths: Array, meta: DiagramMeta
    ) -> PersistenceDiagram:
        """Construct without re-running §3.1's validation. Internal only.

        Sound only for arrays already known to satisfy I1-I9: a slice, a
        boolean-mask subset, or a permutation of a diagram whose arrays have
        already been validated. I1, I2, I7 and I9 survive all three, and
        I3-I6 are elementwise, so they hold on any subset or reordering of
        rows that satisfied them. Anything computing *new* coordinate values
        -- `finitize`'s substituting modes -- MUST use the ordinary
        constructor: a substituted death can break I6.

        Exists because §4.2's zero-copy view is meant to be cheap. Validation
        is six array reductions over the segment, and `__getitem__` is on the
        inner loop of `canonical`, `==`, `allclose`, `same_provenance`, and
        `content_hash` at the batch level.
        """
        obj = object.__new__(cls)
        object.__setattr__(obj, "dims", dims)
        object.__setattr__(obj, "births", births)
        object.__setattr__(obj, "deaths", deaths)
        object.__setattr__(obj, "meta", meta)
        return obj

    @property
    def xp(self) -> Any:
        """The shared array namespace, derived from `dims`, not stored. §3.

        I7 already requires all three arrays to agree, so deriving it makes
        disagreement structurally impossible rather than merely prohibited,
        with nothing to keep in sync at the view construction sites §4.2
        introduces.
        """
        return namespace_of(self.dims)

    # -- §3.2 accessors ----------------------------------------------------

    def dim(self, k: int) -> PersistenceDiagram:
        """The sub-diagram of homological degree `k`. §3.2.

        Returns an *empty* diagram for a `k` not present, never raises: empty
        is a legitimate answer to "what are the 7-dimensional cycles" (§3.2).
        `meta` carries through unchanged, including `provenance` -- filtering
        by degree loses no bar the metadata was describing.

        Eager-only (§3.3): boolean-mask selection, so the output shape depends
        on the values and this is unavailable inside a compiled region.
        """
        # `self.dims == k`, not `xp.equal(self.dims, k)`; see the I5 check in
        # `_validate_bar_arrays` for why the operator form is the portable one.
        return self._masked(self.dims == k)

    @property
    def dimensions(self) -> Array:
        """Sorted unique degrees actually present. §3.2.

        Sorted explicitly because the standard leaves `unique_values`' output
        order unspecified; a diagram with a gap in its degrees (H0 and H2 but
        no H1) reports exactly that, with no sentinel (§3).

        Eager-only (§3.3): how many distinct degrees there are is a property
        of the values.
        """
        xp = self.xp
        return xp.sort(xp.unique_values(self.dims))

    @property
    def essential(self) -> Array:
        """Boolean mask over bars, `deaths == inf`. §3.2, §5.

        `isinf` rather than an explicit comparison against `+inf` is safe
        because I5 forbids `-inf` outright. Shape-preserving, so unlike
        `finite` this remains available under `jax.jit` (§3.3, §4.3).
        """
        return self.xp.isinf(self.deaths)

    @property
    def finite(self) -> PersistenceDiagram:
        """This diagram with its essential bars removed. §3.2.

        Not a finitization: bars are dropped, nothing is substituted, and
        nothing is recorded in `meta.provenance`. Use `finitize` (§5) when the
        loss has to be on the record.

        Eager-only (§3.3): boolean-mask selection.
        """
        return self._masked(~self.essential)

    @property
    def persistence(self) -> Array:
        """`deaths - births`, elementwise, `inf` for essential bars. §3.2.

        Shape-preserving, so available under `jax.jit` (§3.3).
        """
        return self.deaths - self.births

    @property
    def n_bars(self) -> int:
        """Number of bars, counted with multiplicity (§2). A Python `int`.

        The batch-level counterpart is `DiagramBatch.bar_counts`, an array,
        deliberately not sharing this name across the shape change (§4.3).
        """
        return int(self.dims.shape[0])

    def _masked(self, mask: Array) -> PersistenceDiagram:
        """Select bars by boolean mask, carrying `meta` through unchanged.

        The shared implementation behind `dim`, `finite`, and
        `finitize(at="drop")`. A subset of a valid diagram is valid, so this
        skips revalidation (see `_unchecked`).

        Eager-only (§3.3).
        """
        return PersistenceDiagram._unchecked(
            dims=self.dims[mask],
            births=self.births[mask],
            deaths=self.deaths[mask],
            meta=self.meta,
        )

    # -- §7 ordering ---------------------------------------------------------

    def canonical(self) -> PersistenceDiagram:
        """Sort by `(dim, birth, death)` ascending, stable. Presentation only.

        Composed from stable `argsort` passes, least significant key first
        (§7) -- `lexsort` is not part of the array API standard. Stability is
        what makes the composition correct: an unstable sort at any step
        loses the ordering the previous step established.

        Shape-preserving, so this is not one of §3.3's eager-only accessors.
        A permutation of a valid diagram is valid, so the result skips
        revalidation (see `_unchecked`).

        Canonical order is a presentation and determinism concern and MUST
        NOT be assumed by any numerical routine (§7).

        `meta` carries through unchanged. There is no order key to update:
        D15 removed `provenance["order"]` on the ground that canonical order
        is recoverable from the arrays in one pass, so a cached answer to it
        could only go stale -- a diagram carrying `"canonical"` while no
        longer being canonical is §9's clean-plausible-wrong shape, arrived
        at by our own hand (§7, §8, §12.2).
        """
        xp = self.xp
        order = xp.argsort(self.deaths, stable=True)
        order = xp.take(order, xp.argsort(xp.take(self.births, order), stable=True))
        order = xp.take(order, xp.argsort(xp.take(self.dims, order), stable=True))
        return PersistenceDiagram._unchecked(
            dims=xp.take(self.dims, order),
            births=xp.take(self.births, order),
            deaths=xp.take(self.deaths, order),
            meta=self.meta,
        )

    # -- §6.3 equality ---------------------------------------------------

    def __eq__(self, other: object) -> bool:
        """Exact: same multiset of bars, compared without tolerance.

        Order-insensitive (§7): both sides are canonicalised first, and a
        canonical ordering is a function of the multiset alone. `inf` compares
        equal to `inf`; `NaN` cannot occur (I4, I5). `meta` does not
        participate (§8) -- two diagrams with the same bars from different
        backends are the same diagram. Use `same_provenance` for the cases
        that care.

        "Exact" means "no tolerance", not "bit-identical" (§6.3): `-0.0 ==
        0.0` in IEEE 754, so two diagrams differing only in the sign of a zero
        are equal here while their raw bytes differ. That is why `content_hash`
        normalises the sign of zero (§8.1) -- without it, `d1 == d2` with
        differing digests would be reachable.

        This is the exact half of §6.3's split, for same-provenance
        round-trips: serialize, load, compare. Two diagrams of the same data
        from two different backends will *never* satisfy it (§6.2); that is
        what `allclose` is for.

        Raises `ValueError` for a diagram from a different array namespace,
        rather than returning `False`; see `_require_shared_namespace` for
        why, and for why RFC-0001 does not settle it. This is the one case
        where `==` here does not return a `bool`.
        """
        if not isinstance(other, PersistenceDiagram):
            return NotImplemented
        _require_shared_namespace(self, other, "==")
        if self.n_bars != other.n_bars:
            return False
        a, b = self.canonical(), other.canonical()
        xp = a.xp
        return (
            bool(xp.all(xp.equal(a.dims, b.dims)))
            and bool(xp.all(xp.equal(a.births, b.births)))
            and bool(xp.all(xp.equal(a.deaths, b.deaths)))
        )

    def allclose(
        self, other: PersistenceDiagram, *, rtol: float = 1e-9, atol: float = 0.0
    ) -> bool:
        """Approximate: for cross-backend agreement (§6.2). Order-insensitive.

        `inf` only compares close to `inf` -- a tolerance-based formula is
        undefined there (`inf - inf` is `nan`), so essential bars are matched
        exactly rather than run through `rtol`/`atol`. `dims` are compared
        exactly (§6.3).

        The default `rtol` is deliberately tighter than Ripser's `float32`
        reality: a cross-backend test MUST pass an explicit `rtol=1e-6` and
        thereby say in its own source that it is tolerating a precision
        difference (§6.2, §6.3).

        Eager-only (§3.3), for two independent reasons: the essential/finite
        split below is boolean-mask selection, and the return value is a
        Python `bool`.

        Raises `TypeError` for a non-diagram and `ValueError` for a diagram
        from a different array namespace, the same split `finitize` (§5) uses
        and for the same reason: the first is a call no diagram could make
        meaningful, the second a well-typed call this module cannot answer
        (`_require_shared_namespace`).

        **This is a matching, not a sorted pairwise comparison (§6.3).** Two
        diagrams are `allclose` iff there is a bijection between their bars
        under which every matched pair shares a `dim` exactly and agrees on
        both coordinates within tolerance. Equal bar counts are necessary and
        not sufficient.

        Sorting both sides canonically and comparing pairwise -- the obvious
        implementation, and the one this replaced -- is exact in the sort and
        approximate in the comparison, and the two do not compose. When two
        bars' births lie within tolerance *of each other*, two backends can
        canonicalise them into opposite orders, and the pairwise comparison
        then pairs each bar against the other's partner and reports `False`
        for diagrams that do have a bar-for-bar partner within `rtol`. §6.2
        measures GUDHI/Ripser disagreement at the magnitude that flips such a
        tie, so this is reachable on exactly the cross-backend comparison this
        method exists to serve. No sort key repairs it: the RFC's history
        document carries the argument that no total order is stable under
        perturbation.

        **The tolerance is symmetric**, `atol + rtol * max(|a|, |b|)`,
        diverging from `numpy.allclose` -- see `_close_within`.

        **This is not an equivalence relation.** It is reflexive and
        symmetric, but *not transitive*: `a.allclose(b)` and `b.allclose(c)`
        do not imply `a.allclose(c)`, since two tolerance-width steps span
        twice the tolerance. `==` is an equivalence relation and callers will
        reasonably assume the parity holds, so it is stated here. In
        particular, do not use `allclose` to deduplicate or to group
        diagrams: the result depends on the order they are visited in.

        **This is not bottleneck distance and MUST NOT be refactored into
        it** (§6.3, §9). "Does a perfect matching within threshold `t` exist"
        is the decision problem a bottleneck binary search calls repeatedly,
        so the two are adjacent by construction and this method will look like
        a starting point for one. It is not: `allclose` admits no diagonal
        projection and optimises nothing. `core/distances.py` delegates
        bottleneck distance and is not built on this method.
        """
        if not isinstance(other, PersistenceDiagram):
            raise TypeError(
                f"allclose expects a PersistenceDiagram, got {type(other)!r}"
            )
        _require_shared_namespace(self, other, "allclose")
        n = self.n_bars
        if n != other.n_bars:
            return False
        if n == 0:
            # The empty bijection is a bijection.
            return True

        xp = self.xp
        a, b = self.canonical(), other.canonical()
        inf_a, inf_b = xp.isinf(a.deaths), xp.isinf(b.deaths)

        # Fast path, and it is a *witness* rather than a decision. If the two
        # canonically sorted sequences agree bar for bar, then pairing them by
        # position is itself a bijection satisfying the definition, so `True`
        # here is `True` there. The converse does not hold -- that is the
        # whole of D14 -- so a failure falls through to the matching rather
        # than being reported. Nothing below may be skipped on its account.
        #
        # It earns its place because it is the common case: a round-trip, a
        # diagram against itself, or two backends with no near-tie all land
        # here in O(n log n) instead of building an O(n^2) pair matrix.
        if (
            bool(xp.all(xp.equal(a.dims, b.dims)))
            and bool(xp.all(_close_within(a.births, b.births, rtol, atol, xp)))
            and bool(
                xp.all(_deaths_within(a.deaths, inf_a, b.deaths, inf_b, rtol, atol, xp))
            )
        ):
            return True

        # Pair matrix: entry (i, j) asks whether a's bar i could be partnered
        # with b's bar j. Built with array operations, so the O(n^2) work
        # stays in the caller's namespace where it is vectorised.
        compatible = xp.equal(
            xp.expand_dims(a.dims, axis=1), xp.expand_dims(b.dims, axis=0)
        )
        compatible = compatible & _close_within(
            xp.expand_dims(a.births, axis=1),
            xp.expand_dims(b.births, axis=0),
            rtol,
            atol,
            xp,
        )
        compatible = compatible & _deaths_within(
            xp.expand_dims(a.deaths, axis=1),
            xp.expand_dims(inf_a, axis=1),
            xp.expand_dims(b.deaths, axis=0),
            xp.expand_dims(inf_b, axis=0),
            rtol,
            atol,
            xp,
        )

        # An unmatchable bar on either side rules out a bijection, and both
        # directions have to be asked: a row of `False` says one of a's bars
        # has no candidate, a column of `False` says one of b's does, and
        # neither implies the other. Two O(n^2) reductions inside the
        # namespace, taken before anything crosses into Python, so the common
        # "these diagrams differ" case never materialises an edge list.
        if not bool(xp.all(xp.any(compatible, axis=1))):
            return False
        if not bool(xp.all(xp.any(compatible, axis=0))):
            return False

        # The search itself cannot stay in the namespace: an augmenting path
        # is a sequential, data-dependent walk that rebinds one edge at a
        # time, which array operations do not express. So the graph crosses
        # into Python once, here.
        #
        # `nonzero` over the whole matrix rather than per row: one
        # data-dependent-shape call instead of `n` of them, and only the
        # surviving edges cross rather than all n^2 cells. The array API
        # offers no bulk conversion to Python objects, so per-edge `int()` is
        # the portable floor. Adjacency costs O(edges), bounded by the
        # O(n^2) `compatible` already built above, so it adds no order of
        # memory that the pair matrix had not already paid for.
        row_index, col_index = xp.nonzero(compatible)
        neighbours: list[list[int]] = [[] for _ in range(n)]
        # Indexed by position rather than iterated: the standard specifies
        # `__getitem__` on a 1-D array, not the iteration protocol, and
        # falling back to `__iter__` would be relying on something no backend
        # is required to provide (§3.3).
        for k in range(row_index.shape[0]):
            neighbours[int(row_index[k])].append(int(col_index[k]))
        return _perfect_matching_exists(neighbours, n)

    def same_provenance(self, other: PersistenceDiagram) -> bool:
        """Metadata comparison, excluded from `==`/`allclose` (§8).

        Provenance is recorded so a human can audit it, not so equality can
        reject on it; this is the accessor for the cases that genuinely care.
        Compares every `DiagramMeta` field except free-text `description`.

        Raises `TypeError` for a non-diagram rather than returning `False`,
        matching `allclose`: `==` may legitimately be asked to compare against
        anything, these two may not.

        No namespace check, deliberately, unlike `==` and `allclose`: this
        touches no array. A NumPy diagram and a torch diagram of the same
        bars have incomparable coordinates and perfectly comparable metadata,
        and answering the metadata question is the whole point of the method.
        """
        if not isinstance(other, PersistenceDiagram):
            raise TypeError(
                f"same_provenance expects a PersistenceDiagram, got {type(other)!r}"
            )
        # `description` is intentionally free text and excluded from every
        # provenance comparison (§8); all other metadata fields participate.
        return (
            self.meta.filtration,
            self.meta.backend,
            self.meta.backend_version,
            self.meta.coeff_field,
            self.meta.params,
            self.meta.provenance,
        ) == (
            other.meta.filtration,
            other.meta.backend,
            other.meta.backend_version,
            other.meta.coeff_field,
            other.meta.params,
            other.meta.provenance,
        )

    # -- §8.1 content hash -------------------------------------------------

    @property
    def content_hash(self) -> str:
        """sha256 over canonical-ordered coordinates + dims. §8.1.

        Bars only, never metadata (§8.1). This is what a paper pins, so the
        hashed message is specified rather than incidental: a domain tag, the
        bar count, then the three canonical-ordered columns as big-endian
        machine values, `int32` dims first, then `float64` births, then
        `float64` deaths.

            b"akriti.PersistenceDiagram.v1\\x00" || n(8 bytes, big-endian)
            || dims || births || deaths

        The tag and the length carry the same two guarantees §8.2 states for
        the batch hash, one level down. The tag keeps a diagram's digest
        structurally distinct from a batch's rather than incidentally so; the
        length keeps an empty diagram from hashing to `sha256(b"")`, a
        published constant indistinguishable from a bug that hashed nothing
        at all.

        Negative zero is normalised to `+0.0` before hashing. `-0.0 == 0.0`
        is true, so `==` (§6.3) calls two diagrams differing only in the sign
        of a zero equal, and `canonical()` cannot separate them either -- a
        stable sort leaves numerically equal keys in input order. Without the
        normalisation the digest of such a diagram would depend on the order
        the backend happened to emit its bars in, which §7 and §10.1's
        determinism requirement both rule out. Zero births are ubiquitous in
        H0 and `-0.0` is an ordinary product of filtration arithmetic, so this
        is not a hypothetical.
        """
        c = self.canonical()
        h = hashlib.sha256()
        h.update(_DIAGRAM_HASH_TAG)
        h.update(c.n_bars.to_bytes(8, "big"))
        h.update(_big_endian_block(c.dims, "i"))
        # `+ 0.0` is exact for every value I4/I5 admit -- there is no NaN, and
        # `inf + 0.0` is `inf` -- so its only effect is `-0.0` -> `+0.0`.
        h.update(_big_endian_block(c.births + 0.0, "d"))
        h.update(_big_endian_block(c.deaths + 0.0, "d"))
        return h.hexdigest()

    # -- §5 finitization -----------------------------------------------------

    def finitize(self, *, at: float | str) -> PersistenceDiagram:
        """Replace or drop essential bars. Explicit, never implicit (§5).

        `at="max_finite_death"` or `at=<float>` substitute a finite death
        time and record `provenance["essential_bars"] = "finitized_at"` plus
        `provenance["essential_bars_finitized_at"]`, the substituted death.
        `at="drop"` removes essential bars entirely -- a cardinality change,
        not a substitution -- and records `"finitized_dropped"` plus
        `provenance["essential_bars_dropped"]`, the count.

        A diagram with **no essential bars is returned unchanged, provenance
        included** (§5): nothing was substituted and nothing dropped, so
        there is nothing to record and recording anything would overwrite
        what `essential_bars` already said.

        **`essential_bars` is overwritten; `essential_bars_source` is the
        adapter's and is never touched here** -- not written, not read, not
        disturbed if already present (§5, §8).

        **Each mode clears the other's qualifier** (§8): a substitution drops
        `essential_bars_dropped` and a drop drops
        `essential_bars_finitized_at`, since §8 admits each only alongside the
        one `essential_bars` value it qualifies. Finitizing twice overwrites
        rather than accumulates, so the recorded death is always the one now
        in `deaths`. `_validate_provenance` enforces both pairings on the way
        out, in both directions.

        **`at` is validated in full -- type and finiteness, not only the two
        mode names -- before `essential` is looked at** (§5). A check placed
        after the no-essential-bars return above would be one the *data*
        decides whether to apply. Eager-only in every mode (§3.3), not only
        `at="drop"`.

        Raising splits by what is wrong with the call, not by which check
        caught it (§5), matching `allclose` and `same_provenance` one
        signature over:

        - `TypeError` for an `at` that is neither a mode name nor a number --
          `None`, a list, a `bytes`, a `bool`. Nothing about the diagram
          could make such a call meaningful.
        - `ValueError` for an `at` of the right type carrying a value this
          method cannot use: an unrecognised mode name, a non-finite float,
          `at="max_finite_death"` on a diagram with no finite death to take a
          maximum over, or a substituted death below some essential bar's
          birth. The latter check is performed here, after resolving either
          spelling of the substitution, so the error can name both values.
        """
        xp = self.xp
        # `at` is never rebound: a validated substitution lands in
        # `substitute`, and `mode` carries which of the three shapes the
        # argument had. Rebinding the parameter to a narrower type was how the
        # branches below ended up comparing a possibly-numeric `at` against a
        # mode name -- `np.float64(2.0) == "drop"` is a comparison this method
        # should not be making, whatever it happens to return.
        substitute: float | None = None
        if isinstance(at, str):
            if at not in ("drop", "max_finite_death"):
                raise ValueError(f"unknown finitize mode: {at!r}")
            mode = at
        elif isinstance(at, bool):
            # Before the numeric-protocol branch, which `bool` would otherwise
            # satisfy: it is an `int`, so `finitize(at=True)` would substitute
            # a death time of `1.0` and record `"finitized_at:1.0"` -- a
            # provenance entry confidently describing a substitution nobody
            # asked for. That is the same non-numeric-literal-with-a-numeric-
            # protocol hole the `bytes` case below rules out, and the argument
            # for closing it is this paragraph's own subject: `at` is
            # validated in full so a typo cannot survive to reach the data.
            raise TypeError(
                "finitize expects at='drop', at='max_finite_death', or a "
                f"finite float; got the boolean {at!r}"
            )
        elif hasattr(at, "__float__") or hasattr(at, "__index__"):
            # A substitution value. Requiring a numeric protocol rather than
            # just trying `float()` matters: `float(b"2.0")` succeeds, so a
            # `bytes` spelling a number would be accepted while `b"drop"`
            # raised. This admits a Python `int` or `float`, a `Decimal`, and
            # a backend's own float scalar.
            mode = "substitute"
            substitute = float(at)
            if not math.isfinite(substitute):
                raise ValueError(
                    "finitize(at=<float>) needs a finite substitution value; "
                    f"got {substitute!r}"
                )
        else:
            raise TypeError(
                "finitize expects at='drop', at='max_finite_death', or a "
                f"finite float; got {type(at)!r}"
            )

        essential = self.essential
        if not bool(xp.any(essential)):
            return self

        provenance = dict(self.meta.provenance)

        if mode == "drop":
            kept = self._masked(~essential)
            provenance["essential_bars"] = "finitized_dropped"
            provenance["essential_bars_dropped"] = self.n_bars - kept.n_bars
            # §8: each qualifier is present iff `essential_bars` holds the one
            # value it qualifies, so a drop over an already-substituted
            # diagram MUST clear the substituted death rather than leave it
            # describing a bar that is no longer here. The mirror of the pop
            # in the substitution branch below.
            provenance.pop("essential_bars_finitized_at", None)
            return PersistenceDiagram._unchecked(
                dims=kept.dims,
                births=kept.births,
                deaths=kept.deaths,
                meta=replace(self.meta, provenance=provenance),
            )

        if substitute is not None:
            value = substitute
        else:
            # `mode == "max_finite_death"`, the only case left.
            finite_deaths = self.deaths[~essential]
            if finite_deaths.shape[0] == 0:
                raise ValueError(
                    "finitize(at='max_finite_death') needs at least one finite death"
                )
            value = float(xp.max(finite_deaths))

        max_essential_birth = float(xp.max(self.births[essential]))
        if value < max_essential_birth:
            if substitute is not None:
                raise ValueError(
                    f"finitize(at={value!r}) is below the maximum essential "
                    f"birth {max_essential_birth!r}"
                )
            raise ValueError(
                "finitize(at='max_finite_death') computed "
                f"{value!r}, below the maximum essential birth "
                f"{max_essential_birth!r}"
            )

        # §8: the state is the bare enum member and the substituted death is a
        # separate numeric key. Not `f"finitized_at:{value}"` -- that spelling
        # made every reader parse a float back out of a string, and made
        # `essential_bars` an open vocabulary that `DiagramMeta` could not
        # validate. Repeated finitization overwrites both, so the value here
        # is always the death currently in `deaths`, never the first one.
        provenance["essential_bars"] = "finitized_at"
        provenance["essential_bars_finitized_at"] = value
        provenance.pop("essential_bars_dropped", None)

        fill = xp.asarray(value, dtype=self.deaths.dtype)
        new_deaths = xp.where(essential, fill, self.deaths)
        return PersistenceDiagram(
            dims=self.dims,
            births=self.births,
            deaths=new_deaths,
            meta=replace(self.meta, provenance=provenance),
        )


# Unhashable for the same reason `PersistenceDiagram` is; see the comment
# above that class.
@dataclass(frozen=True, eq=False)
class DiagramBatch:
    """A ragged sequence of diagrams in one shared, CSR-style buffer. §4, §4.2.

    Not a dense padded array -- padding is indistinguishable from genuine
    trivial bars (§4, Appendix A.2). `offsets` gives each diagram's row
    range into the concatenated `dims`/`births`/`deaths` buffers, matching
    §10.2's on-disk layout. `__getitem__` returns a view, safe only because
    `PersistenceDiagram` is immutable (I8); see the module docstring for the
    extent of that guarantee. The array API cannot make arrays read-only, so a
    caller can still mutate a coordinate directly through `b.dims[0]`.
    """

    dims: Array
    births: Array
    deaths: Array
    offsets: Array
    metas: Sequence[DiagramMeta]

    def __post_init__(self) -> None:
        """Enforce §3.1 over the shared buffer and B1-B7 over `offsets`.

        §4.1: a batch's concatenated buffer must satisfy the same per-bar
        invariants a single diagram does, so the bar checks are the same
        function `PersistenceDiagram` uses (I1-I7 and I9; I8 is structural,
        as it is for a single diagram). Without B1-B7, `__getitem__`'s slice
        arithmetic can silently read the wrong range, or past the end of the
        buffer -- the same category of silent-wrongness bug §9 exists to
        catch, self-inflicted.

        B6 is checked before B1 for I9's reason one field over: B1 reads
        `offsets.shape[0]`, which a rank-2 array answers happily.

        B2, B3 and B4 are checked against `_bounds()` rather than against
        `offsets` directly. Validation has to read every offset anyway to
        establish B4, and `_bounds()` is the read `__getitem__` will need
        regardless, so doing it here costs nothing and leaves the cache warm
        for the first indexing operation instead of paying for it twice.

        `metas` is normalised to a tuple, so a caller's list cannot be
        appended to behind the batch's back.
        """
        object.__setattr__(
            self, "dims", _copy_array(self.dims, namespace_of(self.dims))
        )
        object.__setattr__(
            self, "births", _copy_array(self.births, namespace_of(self.births))
        )
        object.__setattr__(
            self, "deaths", _copy_array(self.deaths, namespace_of(self.deaths))
        )
        object.__setattr__(
            self, "offsets", _copy_array(self.offsets, namespace_of(self.offsets))
        )
        xp = _validate_bar_arrays(self.dims, self.births, self.deaths)
        object.__setattr__(self, "metas", tuple(self.metas))

        if namespace_of(self.offsets) is not xp:
            raise ValueError("offsets must share the batch's array namespace (B5)")
        if self.offsets.ndim != 1:
            raise ValueError(
                f"offsets must be rank-1 (B6); got ndim {self.offsets.ndim}"
            )
        if self.offsets.dtype != xp.int64:
            raise ValueError(f"offsets must be int64 (B7); got {self.offsets.dtype}")
        if self.offsets.shape[0] != len(self.metas) + 1:
            raise ValueError(
                "len(offsets) must equal len(batch) + 1 (B1); got "
                f"{self.offsets.shape[0]} offsets for {len(self.metas)} diagrams"
            )

        bounds = self._bounds()
        if bounds[0] != 0:
            raise ValueError(f"offsets[0] must be 0 (B2); got {bounds[0]}")
        if bounds[-1] != self.dims.shape[0]:
            raise ValueError(
                f"offsets[-1] must equal total_bars (B3); got {bounds[-1]} "
                f"against {self.dims.shape[0]} bars"
            )
        for i in range(len(bounds) - 1):
            if bounds[i] > bounds[i + 1]:
                raise ValueError(
                    "offsets must be non-decreasing (B4); "
                    f"offsets[{i}]={bounds[i]} > offsets[{i + 1}]={bounds[i + 1]}"
                )

    @classmethod
    def _unchecked(
        cls,
        *,
        dims: Array,
        births: Array,
        deaths: Array,
        offsets: Array,
        metas: tuple[DiagramMeta, ...],
    ) -> DiagramBatch:
        """Construct without re-running §3.1/§4.2 validation. Internal only.

        Sound only when the arrays are already known to satisfy I1-I9 and
        B1-B7 -- in practice, a per-segment permutation of an existing batch,
        which changes no segment's length and so leaves `offsets` correct.
        `metas` MUST already be a tuple.
        """
        obj = object.__new__(cls)
        object.__setattr__(obj, "dims", dims)
        object.__setattr__(obj, "births", births)
        object.__setattr__(obj, "deaths", deaths)
        object.__setattr__(obj, "offsets", offsets)
        object.__setattr__(obj, "metas", metas)
        return obj

    def __len__(self) -> int:
        """Number of diagrams in the batch. §4.3.

        `len(offsets) - 1` by B1. Shape-preserving and data-independent, so
        available under `jax.jit`.
        """
        return int(self.offsets.shape[0]) - 1

    def __iter__(self) -> Iterator[PersistenceDiagram]:
        """Each diagram in turn, in batch order. §4.2.

        **Stated rather than inherited from `__getitem__`.** Python's legacy
        iteration protocol already made `for d in batch` work, so this adds no
        runtime capability -- but a protocol satisfied by accident is not part
        of an interface. Without `__iter__`, `isinstance(batch, Iterable)` is
        `False` on an object that iterates, which is the same shape of bug as
        a `Hashable` that raises when hashed: a guard branching on the check
        takes the path the object cannot follow. No type checker sees the
        legacy protocol either, so `[d.dimensions for d in b]` -- the
        expression §4.3 argues *from* when it declines to add a batch-level
        `dimensions` accessor -- does not type-check without this method.

        A generator, so each call starts over. Returning `self` would let the
        first consumer exhaust the batch for every later one, and nested loops
        over one batch are ordinary here.

        Yields the same zero-copy views `__getitem__` returns, on the same
        terms (§4.2).
        """
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, i: int) -> PersistenceDiagram:
        """The `i`-th diagram, as a zero-copy view into the shared buffer. §4.2.

        The returned `PersistenceDiagram`'s arrays are slices of this batch's
        own buffers, not copies. Safe only because neither type can be written
        through after construction (I8, module docstring). A slice of a
        validated buffer is itself valid, so no revalidation happens here --
        see `PersistenceDiagram._unchecked` for why that matters.

        Negative indices count from the end. Non-integers raise `TypeError`
        via `operator.index`, out-of-range raises `IndexError`. Slicing a
        batch is not part of §4's interface and is not supported; iteration
        works through the legacy `__getitem__` protocol.

        Eager-only (§3.3), and not a filtering operation; see `_bounds`.
        """
        index = operator.index(i)
        n = len(self)
        adjusted = index + n if index < 0 else index
        if not 0 <= adjusted < n:
            # Report what the caller passed, not the normalised form: an
            # `IndexError(-3)` for a `batch[-5]` names an index that was never
            # asked for and sends the reader looking for the wrong bug.
            raise IndexError(
                f"index {index} is out of range for a DiagramBatch of length {n}"
            )
        bounds = self._bounds()
        lo, hi = bounds[adjusted], bounds[adjusted + 1]
        return PersistenceDiagram._unchecked(
            dims=self.dims[lo:hi],
            births=self.births[lo:hi],
            deaths=self.deaths[lo:hi],
            meta=self.metas[adjusted],
        )

    def _bounds(self) -> tuple[int, ...]:
        """`offsets` as Python ints, read from the array once per batch.

        **Eager-only (§3.3), and this is the method that makes it so.** Every
        element of `offsets` is read at Python level, which concretizes a
        traced array exactly as `finitize`'s essential-bar branch does, so
        `__getitem__` and everything routed through it -- `canonical`, `==`,
        `allclose`, `same_provenance`, `content_hash` -- is unavailable inside
        a compiled region. `__len__` and `bar_counts` read `offsets.shape` and
        slice `offsets` without concretizing an element, and stay traceable.

        Cached because the uncached form is the expensive one, and expensively
        so at exactly the scale this type exists for. `__getitem__` reading
        `int(offsets[i])` and `int(offsets[i + 1])` per call costs two device
        synchronisations per diagram on a torch- or JAX-backed batch, and the
        five operations above each drive a Python loop over every diagram: a
        batch of N costs 2N synchronisations per operation, repeated per
        operation. That is the same cost `_big_endian_block`'s fast path
        exists to avoid one level down, and it is unbounded here where that
        one is not. Reading `offsets` once amortises it to N for the first
        indexing operation and zero for every later one.

        Caching derived data does not weaken I8. The arrays are untouched and
        no value observable through the public surface changes; what is stored
        is a function of `offsets`, which nothing in this module can rebind
        (`frozen=True`) or write through (I8, module docstring). A caller who
        retained a reference to the array they handed in and writes through it
        invalidates this cache, exactly as the module docstring already says
        such a write invalidates every view derived from it -- one guarantee,
        not a new exception to it.
        """
        cached: tuple[int, ...] | None = self.__dict__.get("_bounds_cache")
        if cached is None:
            offsets = self.offsets
            cached = tuple(int(offsets[i]) for i in range(offsets.shape[0]))
            object.__setattr__(self, "_bounds_cache", cached)
        return cached

    @property
    def xp(self) -> Any:
        """The shared array namespace, derived from `dims`, not stored. §4.3.

        Well-defined for `offsets` too, since B5 already requires it to share
        `dims`' namespace. Same derive-don't-store reasoning as
        `PersistenceDiagram.xp` (§3).
        """
        return namespace_of(self.dims)

    @classmethod
    def from_diagrams(
        cls, diagrams: Sequence[PersistenceDiagram], *, xp: Any = None
    ) -> DiagramBatch:
        """Concatenate ordinary adapter output into one batch. §4.2.

        The common path: N separate `from_gudhi`/`from_ripser` calls that have
        to become one batch for a bootstrap or permutation test. Input order
        is preserved -- batch position is meaningful (§6.3) -- and each
        diagram's `meta` lands at the matching index in `metas`.

        **All diagrams must share one array namespace.** I7 constrains the
        three arrays within a single diagram and says nothing across
        diagrams, so it is checked here. Without the check, `xp.concat` would
        either raise something opaque from the backend or silently coerce a
        foreign array, and the concatenated result would then validate
        cleanly -- the mixed input having already been erased.

        **`xp` is required only for an empty `diagrams`**, and rejected if it
        disagrees with a non-empty one. An empty batch is perfectly valid --
        §8.2 defines a hash for it, `canonical()` handles it, and a
        permutation test that filters down to zero diagrams needs to be able
        to build one -- but there is no diagram to derive a namespace from, and
        this type has no default backend to fall back on (§3.3). Pass it
        directly: `DiagramBatch.from_diagrams([], xp=some_namespace)`.
        """
        if diagrams:
            namespace = diagrams[0].xp
            if xp is not None and xp is not namespace:
                raise ValueError("xp does not match the array namespace of diagrams[0]")
            for i, d in enumerate(diagrams):
                if d.xp is not namespace:
                    raise ValueError(
                        f"diagrams[{i}] does not share diagrams[0]'s array "
                        "namespace; a batch has one namespace (I7, B5)"
                    )
            dims = namespace.concat([d.dims for d in diagrams])
            births = namespace.concat([d.births for d in diagrams])
            deaths = namespace.concat([d.deaths for d in diagrams])
        else:
            if xp is None:
                raise ValueError(
                    "from_diagrams([]) cannot derive an array namespace from an "
                    "empty sequence; pass xp=<namespace> to build an empty batch"
                )
            namespace = xp
            dims = namespace.asarray([], dtype=namespace.int32)
            births = namespace.asarray([], dtype=namespace.float64)
            deaths = namespace.asarray([], dtype=namespace.float64)

        boundaries = [0, *itertools.accumulate(d.n_bars for d in diagrams)]
        return cls._unchecked(
            dims=dims,
            births=births,
            deaths=deaths,
            offsets=namespace.asarray(boundaries, dtype=namespace.int64),
            metas=tuple(d.meta for d in diagrams),
        )

    # -- §4.3 accessors ------------------------------------------------------

    @property
    def essential(self) -> Array:
        """Boolean mask over the whole concatenated buffer, `deaths == inf`. §4.3.

        Elementwise, shape `(total_bars,)`, and correct by the same invariants
        that govern the stored fields (I4, I5). Shape-preserving, so it stays
        available under `jax.jit` where the diagram-level filtering accessors
        do not (§3.3, §4.3).
        """
        return self.xp.isinf(self.deaths)

    @property
    def persistence(self) -> Array:
        """`deaths - births` over the whole buffer, `inf` for essential bars. §4.3.

        Shape `(total_bars,)`, elementwise, and jit-safe for the same reason
        `essential` is.
        """
        return self.deaths - self.births

    @property
    def bar_counts(self) -> Array:
        """Bar count per diagram, shape `(len(b),)`. §4.3.

        `offsets[1:] - offsets[:-1]`, always non-negative by B4. Deliberately
        not named `n_bars`: `PersistenceDiagram.n_bars` is a scalar and this
        is an array, and reusing a name across a shape change is the same
        silent-wrongness class §9 rules out elsewhere. The batch total is
        `b.dims.shape[0]`, already available from the stored field.
        """
        return self.offsets[1:] - self.offsets[:-1]

    def canonical(self) -> DiagramBatch:
        """Sort within each `[offsets[i], offsets[i+1])` segment independently. §7.

        MUST NOT reorder segments -- diagram order is meaningful, bar order
        within a diagram is not. Reuses `PersistenceDiagram.canonical()` per
        segment rather than a flat sort, which would interleave rows across
        diagram boundaries and leave `offsets` pointing at the wrong ranges.

        Segment lengths are unchanged, so `offsets` and `metas` carry through
        and the result skips revalidation (see `_unchecked`). This is also
        what makes §10.1's determinism requirement true for a batch and not
        just for a single diagram. Each segment's `meta` carries through
        untouched, there being no order key to update (D15; see
        `PersistenceDiagram.canonical()`).

        **Eager-only (§3.3), unlike `PersistenceDiagram.canonical()`.** The
        sort is shape-preserving at both levels, but this one reaches its
        segments through `__getitem__`, which concretizes `offsets`. Shape-
        preserving and traceable are different properties; this pair is where
        they come apart at the batch level.
        """
        if len(self) == 0:
            return self
        xp = self.xp
        segments = [self[i].canonical() for i in range(len(self))]
        return DiagramBatch._unchecked(
            dims=xp.concat([d.dims for d in segments]),
            births=xp.concat([d.births for d in segments]),
            deaths=xp.concat([d.deaths for d in segments]),
            offsets=self.offsets,
            metas=tuple(self.metas),
        )

    # -- §6.3 equality: order-sensitive across diagrams -----------------------

    def __eq__(self, other: object) -> bool:
        """Exact, and order-sensitive across diagrams. §6.3.

        Requires `len(b1) == len(b2)` and `b1[i] == b2[i]` for every `i` in
        sequence. Diagram 3 in one batch is not interchangeable with diagram 5
        in another just because their bars match: batch position is a
        positional axis (§4's leading batch dimension), while bar order within
        a single diagram explicitly is not (§7). A batch is an order-sensitive
        container of order-insensitive things.

        The namespace check is hoisted here rather than left to the first
        `self[i] == other[i]`, so that a cross-namespace pair raises whatever
        the two lengths are, instead of returning `False` for mismatched
        lengths and raising for equal ones.
        """
        if not isinstance(other, DiagramBatch):
            return NotImplemented
        _require_shared_namespace(self, other, "==")
        if len(self) != len(other):
            return False
        return all(self[i] == other[i] for i in range(len(self)))

    def allclose(
        self, other: DiagramBatch, *, rtol: float = 1e-9, atol: float = 0.0
    ) -> bool:
        """Approximate, elementwise over batch positions. §6.3.

        The same order-sensitivity as `==`, with each pair compared by
        `PersistenceDiagram.allclose`: bar order within a diagram is not
        meaningful and diagram order within a batch is (§6.3, §7), so this is
        an order-sensitive comparison over an order-insensitive one. Its
        `ValueError` on a cross-namespace comparison is hoisted here for the
        reason `==` gives.

        Everything that method documents about itself carries over
        unchanged -- the matching semantics, the symmetric tolerance
        diverging from `numpy.allclose`, and that it is **not an equivalence
        relation**. The last is worth repeating at this level rather than
        left to a cross-reference: a batch of batches is exactly where
        someone would reach for transitivity.
        """
        if not isinstance(other, DiagramBatch):
            raise TypeError(f"allclose expects a DiagramBatch, got {type(other)!r}")
        _require_shared_namespace(self, other, "allclose")
        if len(self) != len(other):
            return False
        return all(
            self[i].allclose(other[i], rtol=rtol, atol=atol) for i in range(len(self))
        )

    def same_provenance(self, other: DiagramBatch) -> bool:
        """Metadata comparison, order-sensitive the same way `==` is. §8.

        Requires `len(b1) == len(b2)` and `b1[i].same_provenance(b2[i])` for
        every `i` in sequence. Raises `TypeError` for a non-batch, matching
        `allclose`.
        """
        if not isinstance(other, DiagramBatch):
            raise TypeError(
                f"same_provenance expects a DiagramBatch, got {type(other)!r}"
            )
        if len(self) != len(other):
            return False
        return all(self[i].same_provenance(other[i]) for i in range(len(self)))

    # -- §8.2 batch content hash ---------------------------------------------

    @property
    def content_hash(self) -> str:
        """Domain-separated composition of member `content_hash`es, in batch order.

        Composed from `b[i].content_hash`, never re-serialised from the raw
        concatenated buffer (§8.2): composing localises a mismatch, since
        comparing member hashes pinpoints which diagram changed.

        Order-sensitive -- `[A, B]` and `[B, A]` are different batches (§6.3)
        -- and domain-separated from `PersistenceDiagram.content_hash` by the
        `akriti.DiagramBatch.v1` tag, with the explicit `len(b)` keeping an
        empty or truncated batch from being confused with a prefix of another
        batch's concatenated hashes. Exact equality only; §8.2 offers no
        approximate counterpart, because no hash is consistent with a
        tolerance.
        """
        h = hashlib.sha256()
        h.update(_BATCH_HASH_TAG)
        h.update(len(self).to_bytes(8, "big"))
        for i in range(len(self)):
            h.update(bytes.fromhex(self[i].content_hash))
        return h.hexdigest()
