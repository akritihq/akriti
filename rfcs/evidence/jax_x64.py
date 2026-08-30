#!/usr/bin/env python3
"""Measure whether a JAX-backed diagram can satisfy RFC-0001 I2 and B7.

Run:  python rfcs/evidence/jax_x64.py          # JAX default config
      python rfcs/evidence/jax_x64.py --x64    # jax_enable_x64 = True

RFC-0001 sec 3.1 (I2) requires `births` and `deaths` to be the namespace's own
`float64` and `dims` its own `int32`, tested by equality against `xp.float64` /
`xp.int32`. Sec 4.2 (B7) requires `offsets` to be the namespace's own `int64`.
Sec 3.3 promises that "a diagram built from JAX arrays stays JAX-backed".

JAX defaults to `jax_enable_x64 = False`, under which a requested 64-bit dtype
is truncated to 32-bit. This script answers, by measurement rather than by
citation:

  1. What `jnp.asarray(..., dtype=jnp.float64)` and `dtype=jnp.int64` actually
     return under each config, and the exact warning text, if any.
  2. Whether `jnp.float64` exists as an attribute and whether it is `jnp.float32`
     -- i.e. whether I2's equality test is even well-formed on JAX.
  3. The same probes down the array-API path `core.py` actually uses:
     `xp = arr.__array_namespace__()`, then `xp.float64`, `xp.astype`,
     `xp.asarray(..., dtype=xp.float64)` and the `dtype ==` comparisons
     themselves. This path, not the `jnp.*` one, decides the finding.
  4. What `xp.__array_namespace_info__().dtypes()` advertises.
  5. Whether converting an *existing* 64-bit NumPy array warns or is silent.
     This is the adapter path, and it does not behave like the explicit one.
  6. Whether x64 can be enabled after arrays already exist, and whether
     `JAX_ENABLE_X64=1` is an equivalent lever. This decides whether the
     requirement can be written as "the library enables it" or must be written
     as "the caller enables it".
  7. Whether `jax_explicit_x64_dtypes`, a second and much narrower config flag,
     honours an explicit 64-bit request while `jax_enable_x64` stays False.
     This is the probe that decides whether F1 holds as stated.
  8. Whether `PersistenceDiagram` and `DiagramBatch` construct from JAX arrays,
     run against the repository's own `core.py` when it is importable.

Measured 2026-08-23 with jax 0.11.1, jaxlib 0.11.1, numpy 2.5.2,
CPython 3.12.13, CPU-only (no CUDA jaxlib installed).

Dependencies: jax, numpy, stdlib. Probes 7 and 8 additionally need `akriti` on
the path and report themselves as unmeasured when it is absent, rather than
skipping silently (RFC-0001 sec 9.2).
"""

from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

# Deliberately NOT suppressed, for the reason probe_backends.py records: this
# script exists to observe a warning, and the whole finding under measurement
# turns on whether one is emitted. JAX also warns at most once per call site
# under the default filters, so "always" is required for the second and later
# probes to see anything at all.
warnings.simplefilter("always")

# The exact text JAX 0.11.1 emits when an explicitly requested dtype is
# truncated. Kept as a template rather than four literals so a drift in the
# shared sentence is caught once, and reported against every probe that
# reconstructs it.
TRUNCATION_TEMPLATE = (
    "Explicitly requested dtype {requested} requested in {operation} is not "
    "available, and will be truncated to dtype {actual}. To enable more "
    "dtypes, set the jax_enable_x64 configuration option or the "
    "JAX_ENABLE_X64 shell environment variable. See "
    "https://github.com/jax-ml/jax#current-gotchas for more."
)


# Quoted exactly as JAX 0.11.1 emits it, missing space and wrong flag name
# included: the message names `allow_explicit_x64_dtypes`, while the flag is
# actually `jax_explicit_x64_dtypes`.
EXPLICIT_X64_ERROR = (
    "Explicitly requested dtype float64 requested in asarray is not "
    "available. To enable more dtypes, set the jax_enable_x64 or "
    "allow_explicit_x64_dtypes configuration options."
    "See https://github.com/jax-ml/jax#current-gotchas for more."
)


class ProbeDriftError(RuntimeError):
    """Raised when a measured RFC-0001 JAX claim has changed."""


def _fail(section: str, message: str) -> None:
    raise ProbeDriftError(f"{section} drift: {message}")


def _require(condition: bool, section: str, message: str) -> None:
    if not condition:
        _fail(section, message)


def rule(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def truncation_warning(*, requested: str, operation: str, actual: str) -> str:
    """The exact message JAX emits for one truncated dtype request."""
    return TRUNCATION_TEMPLATE.format(
        requested=requested, operation=operation, actual=actual
    )


def measure(operation: Callable[[], Any]) -> tuple[Any, list[warnings.WarningMessage]]:
    """Run one probe and return its value with every warning it emitted."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = operation()
    return value, list(caught)


def _require_warning(
    caught: Sequence[warnings.WarningMessage],
    expected: str | None,
    *,
    section: str,
    label: str,
) -> None:
    """Require an exact warning message, or require silence when None."""
    observed = [str(entry.message) for entry in caught]
    if expected is None:
        _require(
            not observed,
            section,
            f"{label} now warns where it did not: {observed!r}",
        )
        return
    _require(
        len(observed) == 1,
        section,
        f"{label} emitted {len(observed)} warnings, expected 1: {observed!r}",
    )
    _require(
        caught[0].category is UserWarning,
        section,
        f"{label} warning category changed: {caught[0].category!r}",
    )
    _require(
        observed[0] == expected,
        section,
        f"{label} warning text changed:\n  observed={observed[0]!r}\n"
        f"  expected={expected!r}",
    )


def _report(label: str, dtype: Any, caught: Sequence[warnings.WarningMessage]) -> None:
    note = "silent" if not caught else f"{len(caught)} UserWarning"
    print(f"  {label:<46s} -> {dtype!s:<8s}  ({note})")


def _dtype_probe(
    operation: Callable[[], Any],
    *,
    label: str,
    expected_dtype: str,
    expected_warning: str | None,
    section: str,
) -> Any:
    """Measure one dtype request: its result, its warning, and both expectations."""
    value, caught = measure(operation)
    _report(label, value.dtype, caught)
    _require(
        str(value.dtype) == expected_dtype,
        section,
        f"{label} returned {value.dtype}, expected {expected_dtype}",
    )
    _require_warning(caught, expected_warning, section=section, label=label)
    return value


def _subprocess_probe(program: str, env: dict[str, str] | None = None) -> str:
    """Run a short probe in a fresh interpreter and return its stdout.

    Probes 6a and 6b are about process-global configuration state, so they
    cannot be measured in this process without corrupting every probe that
    follows them. Running them out-of-process also makes both independent of
    which mode this script was invoked in.
    """
    child_env = dict(os.environ)
    child_env.pop("JAX_ENABLE_X64", None)
    child_env.pop("JAX_EXPLICIT_X64_DTYPES", None)
    if env:
        child_env.update(env)
    completed = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        env=child_env,
        check=False,
    )
    if completed.returncode != 0:
        _fail("X.6", f"subprocess probe failed:\n{completed.stderr}")
    return completed.stdout.strip()


LATE_FLIP_PROGRAM = """
import warnings
import jax
import jax.numpy as jnp

with warnings.catch_warnings(record=True) as before:
    warnings.simplefilter("always")
    early = jnp.asarray([0.0, 1.0], dtype=jnp.float64)
jax.config.update("jax_enable_x64", True)
with warnings.catch_warnings(record=True) as after:
    warnings.simplefilter("always")
    late = jnp.asarray([0.0, 1.0], dtype=jnp.float64)
    promoted = jnp.astype(early, jnp.float64)
print(early.dtype, len(before), late.dtype, len(after), promoted.dtype)
"""

ENV_VAR_PROGRAM = """
import jax
import jax.numpy as jnp

print(
    jax.config.jax_enable_x64,
    jnp.asarray([0.0, 1.0], dtype=jnp.float64).dtype,
    jnp.asarray([0, 1], dtype=jnp.int64).dtype,
    jnp.asarray([0.0, 1.0]).dtype,
)
"""


# The `allow` probe measures the whole chain in one child process: the dtype
# request, whether arithmetic on the result stays 64-bit, what the namespace
# advertises, whether plain conversion still truncates, and whether the
# repository's own constructor accepts the arrays. Anything less would leave
# open the obvious objection -- that the flag buys a float64 array which then
# decays on the first operation performed on it.
EXPLICIT_ALLOW_PROGRAM = """
import os
import jax
import jax.numpy as jnp
import numpy as np

if os.environ.get("PROBE_SET_CONFIG") == "1":
    jax.config.update("jax_explicit_x64_dtypes", "allow")

births = jnp.asarray([0.0, 0.5], dtype=jnp.float64)
offsets = jnp.asarray([0, 1, 2], dtype=jnp.int64)
xp = births.__array_namespace__()
deaths = jnp.asarray([1.0, jnp.inf], dtype=jnp.float64)
dims = jnp.asarray([0, 1], dtype=jnp.int32)

status = "unmeasured"
try:
    from akriti.diagrams.core import DiagramBatch, DiagramMeta, PersistenceDiagram
except ImportError:
    pass
else:
    try:
        diagram = PersistenceDiagram(dims=dims, births=births, deaths=deaths)
        batch = DiagramBatch(
            dims=dims,
            births=births,
            deaths=deaths,
            offsets=offsets,
            metas=[DiagramMeta(), DiagramMeta()],
        )
        status = "ok" if diagram.persistence.dtype == jnp.float64 else "lossy"
        status = status if batch.offsets.dtype == jnp.int64 else "batch-lossy"
    except ValueError as exc:
        status = "rejected:" + str(exc).split(";")[0].replace(" ", "_")

print(
    jax.config.jax_enable_x64,
    births.dtype,
    offsets.dtype,
    jnp.asarray([0.0, 1.0]).dtype,
    (deaths - births).dtype,
    "float64" in xp.__array_namespace_info__().dtypes(),
    jnp.asarray(np.array([0.0, 1.0], dtype=np.float64)).dtype,
    status,
)
"""

EXPLICIT_ERROR_PROGRAM = """
import jax
import jax.numpy as jnp

jax.config.update("jax_explicit_x64_dtypes", "error")
try:
    jnp.asarray([0.0, 1.0], dtype=jnp.float64)
except ValueError as exc:
    print(repr(str(exc)))
else:
    print(repr(""))
"""

# `jax_enable_x64` is explicitly removed from JAX's context-manager flags
# (`jax/_src/config.py`: `config._contextmanager_flags.remove('jax_enable_x64')`),
# so it has no scoped form. `jax_explicit_x64_dtypes` was never removed and
# does. The State object carrying it is private, which is the whole finding
# here: the scoped lever exists and is not public API.
EXPLICIT_SCOPED_PROGRAM = """
import jax.numpy as jnp
from jax._src import config

outside = jnp.asarray([0.0, 1.0], dtype=jnp.float64).dtype
with config.explicit_x64_dtypes("allow"):
    inside = jnp.asarray([0.0, 1.0], dtype=jnp.float64).dtype
after = jnp.asarray([0.0, 1.0], dtype=jnp.float64).dtype
import jax

print(outside, inside, after, hasattr(jax.config, "explicit_x64_dtypes"))
"""


def probe_environment(jax: Any, jaxlib: Any) -> None:
    rule("ENVIRONMENT")
    print(f"  python  {sys.version.split()[0]}  ({sys.implementation.name})")
    print(f"  jax     {jax.__version__}")
    print(f"  jaxlib  {jaxlib.__version__}")
    print(f"  numpy   {np.__version__}")
    print(f"  devices {[d.platform for d in jax.devices()]}")
    print(f"  jax_enable_x64 = {jax.config.jax_enable_x64}")


def probe_direct_requests(jnp: Any, *, x64: bool) -> None:
    """X.1 -- the `jnp.*` spelling of an explicit 64-bit request."""
    rule("X.1  EXPLICIT 64-BIT REQUESTS THROUGH `jax.numpy`")
    _dtype_probe(
        lambda: jnp.asarray([[0.0, 1.0]], dtype=jnp.float64),
        label="jnp.asarray([[0.0, 1.0]], dtype=jnp.float64)",
        expected_dtype="float64" if x64 else "float32",
        expected_warning=None
        if x64
        else truncation_warning(
            requested="float64", operation="asarray", actual="float32"
        ),
        section="X.1",
    )
    _dtype_probe(
        lambda: jnp.asarray([0, 1], dtype=jnp.int64),
        label="jnp.asarray([0, 1], dtype=jnp.int64)",
        expected_dtype="int64" if x64 else "int32",
        expected_warning=None
        if x64
        else truncation_warning(requested="int64", operation="asarray", actual="int32"),
        section="X.1",
    )
    if not x64:
        print("  => The request is honoured in name and not in dtype. It is NOT")
        print("     silent: each truncation carries one UserWarning naming both")
        print("     the requested and the delivered dtype, and the two levers")
        print("     that would fix it.")


def probe_dtype_identity(jnp: Any) -> None:
    """X.2 -- does I2's equality test have operands to compare?"""
    rule("X.2  DTYPE ATTRIBUTES -- IS `float64` EVEN PRESENT?")
    for name in ("float64", "int64", "float32", "int32"):
        _require(
            hasattr(jnp, name),
            "X.2",
            f"jax.numpy no longer exposes {name}",
        )
        print(f"  jnp.{name:<8s} = {getattr(jnp, name)!r}")
    aliased_float = jnp.float64 is jnp.float32
    aliased_int = jnp.int64 is jnp.int32
    print(f"  jnp.float64 is jnp.float32 -> {aliased_float}")
    print(f"  jnp.int64   is jnp.int32   -> {aliased_int}")
    _require(
        not aliased_float and not aliased_int,
        "X.2",
        "jax.numpy now aliases its 64-bit dtype attributes onto the 32-bit ones",
    )
    print(f"  np.dtype(jnp.float64)      -> {np.dtype(jnp.float64)}")
    print("  => The attributes exist and are distinct objects under both configs.")
    print("     I2's `births.dtype != xp.float64` is therefore a well-formed")
    print("     comparison on JAX, not an AttributeError. What varies with the")
    print("     config is which side of it an array can land on.")


def probe_array_api_path(jnp: Any, *, x64: bool) -> Any:
    """X.3 -- the path `core.py` actually takes. This one decides F1."""
    rule("X.3  THE ARRAY-API PATH (`__array_namespace__`) -- core.py's OWN PATH")
    seed = jnp.asarray([0.0, 1.0])
    xp = seed.__array_namespace__()
    print(f"  arr.__array_namespace__() is jax.numpy -> {xp is jnp}")
    print(f"  xp.__array_api_version__               -> {xp.__array_api_version__}")
    _require(xp is jnp, "X.3", "JAX no longer returns jax.numpy from the protocol")
    print(f"  default float dtype (no dtype= given)  -> {seed.dtype}")

    _dtype_probe(
        lambda: xp.astype(seed, xp.float64),
        label="xp.astype(arr, xp.float64)",
        expected_dtype="float64" if x64 else "float32",
        expected_warning=None
        if x64
        else truncation_warning(
            requested="float64", operation="astype", actual="float32"
        ),
        section="X.3",
    )
    births = _dtype_probe(
        lambda: xp.asarray([[0.0, 1.0]], dtype=xp.float64),
        label="xp.asarray(..., dtype=xp.float64)",
        expected_dtype="float64" if x64 else "float32",
        expected_warning=None
        if x64
        else truncation_warning(
            requested="float64", operation="asarray", actual="float32"
        ),
        section="X.3",
    )
    offsets = _dtype_probe(
        lambda: xp.asarray([0, 1], dtype=xp.int64),
        label="xp.asarray(..., dtype=xp.int64)",
        expected_dtype="int64" if x64 else "int32",
        expected_warning=None
        if x64
        else truncation_warning(requested="int64", operation="asarray", actual="int32"),
        section="X.3",
    )
    dims = _dtype_probe(
        lambda: xp.asarray([0, 1], dtype=xp.int32),
        label="xp.asarray(..., dtype=xp.int32)",
        expected_dtype="int32",
        expected_warning=None,
        section="X.3",
    )

    print("\n  I2 / B7 as core.py spells them:")
    checks = {
        "births.dtype == xp.float64  (I2)": bool(births.dtype == xp.float64),
        "dims.dtype   == xp.int32    (I2)": bool(dims.dtype == xp.int32),
        "offsets.dtype == xp.int64   (B7)": bool(offsets.dtype == xp.int64),
    }
    for label, result in checks.items():
        print(f"    {label} -> {result}")
    _require(
        checks["dims.dtype   == xp.int32    (I2)"],
        "X.3",
        "I2's int32 half no longer holds on JAX under either config",
    )
    _require(
        checks["births.dtype == xp.float64  (I2)"] is x64,
        "X.3",
        f"I2's float64 half returned {checks['births.dtype == xp.float64  (I2)']} "
        f"with x64={x64}",
    )
    _require(
        checks["offsets.dtype == xp.int64   (B7)"] is x64,
        "X.3",
        f"B7 returned {checks['offsets.dtype == xp.int64   (B7)']} with x64={x64}",
    )
    if not x64:
        print("  => I2's `dims` half is satisfiable; its `births`/`deaths` half is")
        print("     not, and neither is B7. The finding is about float64 and the")
        print("     `offsets` int64 -- NOT about `dims`, which I2 requires to be")
        print("     int32 and which default JAX supplies natively.")
    return xp


def probe_namespace_info(xp: Any, *, x64: bool) -> None:
    """X.4 -- what the namespace advertises about itself."""
    rule("X.4  `__array_namespace_info__().dtypes()` -- IS float64 ADVERTISED?")
    info = xp.__array_namespace_info__()
    dtypes = info.dtypes()
    defaults = info.default_dtypes()
    names = sorted(dtypes)
    print(f"  dtypes()          -> {names}")
    print(f"  default_dtypes()  -> { {k: str(v) for k, v in defaults.items()} }")
    advertises_f64 = "float64" in dtypes
    advertises_i64 = "int64" in dtypes
    print(f"  advertises float64 -> {advertises_f64}")
    print(f"  advertises int64   -> {advertises_i64}")
    _require(
        advertises_f64 is x64 and advertises_i64 is x64,
        "X.4",
        f"namespace advertises float64={advertises_f64} int64={advertises_i64} "
        f"with x64={x64}",
    )
    if not x64:
        print("  => The namespace does not merely decline the request, it declines")
        print("     to list the dtype at all. Under the default config a JAX")
        print("     namespace is one in which float64 does not exist, which is a")
        print("     stronger statement than a truncation: a caller inspecting the")
        print("     namespace before building arrays is told so up front.")


def probe_conversion(jnp: Any, jax: Any, *, x64: bool) -> None:
    """X.5 -- converting an existing 64-bit NumPy array. The adapter path."""
    rule("X.5  CONVERTING AN EXISTING NumPy float64 ARRAY (THE ADAPTER PATH)")
    source_f64 = np.array([0.0, 1.0], dtype=np.float64)
    source_i64 = np.array([0, 1], dtype=np.int64)
    print(f"  source dtypes: {source_f64.dtype}, {source_i64.dtype}")
    expected_f = "float64" if x64 else "float32"
    expected_i = "int64" if x64 else "int32"
    for label, operation, expected in (
        ("jnp.asarray(np_float64)", lambda: jnp.asarray(source_f64), expected_f),
        ("jnp.array(np_float64)", lambda: jnp.array(source_f64), expected_f),
        ("jax.device_put(np_float64)", lambda: jax.device_put(source_f64), expected_f),
        ("jax.device_put(np_int64)", lambda: jax.device_put(source_i64), expected_i),
    ):
        _dtype_probe(
            operation,
            label=label,
            expected_dtype=expected,
            expected_warning=None,
            section="X.5",
        )
    if not x64:
        print("  => This path IS silent. No dtype was explicitly requested, so the")
        print("     X.1 warning does not fire: an adapter handing a backend's")
        print("     float64 diagram to JAX loses 29 bits of mantissa without")
        print("     emitting anything. The explicit request in X.1 is the loud")
        print("     case; conversion is the quiet one.")


def probe_config_timing() -> None:
    """X.6 -- when can x64 be enabled, and by whom?"""
    rule("X.6  WHEN CAN x64 BE ENABLED? (fresh subprocesses)")

    early, before, late, after, promoted = _subprocess_probe(LATE_FLIP_PROGRAM).split()
    print("  6a. jax.config.update AFTER arrays already exist:")
    print(f"      array built before the flip  -> {early} (warnings: {before})")
    print(f"      array built after the flip   -> {late} (warnings: {after})")
    print(f"      astype() of the older array  -> {promoted}")
    _require(
        early == "float32" and late == "float64" and promoted == "float64",
        "X.6",
        f"late-flip behaviour changed: {early}/{late}/{promoted}",
    )
    print("      => The flip is accepted late and takes effect immediately for")
    print("         newly built arrays. Arrays that already exist keep the dtype")
    print("         they were built with, and can be promoted with astype()")
    print("         afterwards without a warning. So the requirement need NOT be")
    print("         'enable before importing jax'.")

    enabled, f64, i64, default_float = _subprocess_probe(
        ENV_VAR_PROGRAM, env={"JAX_ENABLE_X64": "1"}
    ).split()
    print("\n  6b. JAX_ENABLE_X64=1 in the environment, config untouched:")
    print(f"      jax.config.jax_enable_x64    -> {enabled}")
    print(f"      explicit float64 request     -> {f64}")
    print(f"      explicit int64 request       -> {i64}")
    print(f"      DEFAULT float dtype          -> {default_float}")
    _require(
        enabled == "True" and f64 == "float64" and i64 == "int64",
        "X.6",
        f"JAX_ENABLE_X64=1 no longer enables x64: {enabled}/{f64}/{i64}",
    )
    _require(
        default_float == "float64",
        "X.6",
        f"enabling x64 no longer changes the default float dtype: {default_float}",
    )
    print("      => The env var is an equivalent lever, and either lever changes")
    print("         the DEFAULT dtype for the whole process, not just explicit")
    print("         requests. That is why this is the caller's switch to throw")
    print("         and not a library's: flipping it changes the numerics of")
    print("         every other JAX computation in the program.")


def probe_explicit_x64(*, x64: bool) -> None:
    """X.7 -- the narrower lever. This is the probe that decides F1.

    `jax_enable_x64` is not JAX's only 64-bit switch. `jax_explicit_x64_dtypes`
    takes `warn` (the default), `error`, or `allow`, and under `allow` an
    explicitly requested 64-bit dtype is honoured with `jax_enable_x64` still
    False. Every probe above measured the default `warn`.
    """
    rule("X.7  THE NARROWER LEVER -- `jax_explicit_x64_dtypes` (subprocesses)")

    levers = (
        (
            "7a",
            "jax.config.update('jax_explicit_x64_dtypes', 'allow')",
            {"PROBE_SET_CONFIG": "1"},
        ),
        (
            "7b",
            "JAX_EXPLICIT_X64_DTYPES=allow in the environment",
            {"JAX_EXPLICIT_X64_DTYPES": "allow"},
        ),
    )
    for tag, label, env in levers:
        (
            enabled,
            births,
            offsets,
            default_float,
            arithmetic,
            advertised,
            converted,
            status,
        ) = _subprocess_probe(EXPLICIT_ALLOW_PROGRAM, env=env).split()
        print(f"  {tag}. {label}")
        print(f"      jax_enable_x64 stays          -> {enabled}")
        print(f"      explicit float64 births       -> {births}")
        print(f"      explicit int64 offsets        -> {offsets}")
        print(f"      DEFAULT float dtype           -> {default_float}")
        print(f"      deaths - births               -> {arithmetic}")
        print(f"      namespace advertises float64  -> {advertised}")
        print(f"      jnp.asarray(np_float64)       -> {converted}")
        print(f"      akriti construction           -> {status}")
        _require(
            enabled == "False",
            "X.7",
            f"the narrow flag now also flips jax_enable_x64: {enabled}",
        )
        _require(
            births == "float64" and offsets == "int64",
            "X.7",
            f"'allow' no longer honours explicit 64-bit requests: {births}/{offsets}",
        )
        _require(
            arithmetic == "float64",
            "X.7",
            f"64-bit arithmetic decays under 'allow': {arithmetic}",
        )
        _require(
            default_float == "float32",
            "X.7",
            f"'allow' now changes the process default float dtype: {default_float}",
        )
        _require(
            advertised == "False",
            "X.7",
            "the namespace now advertises float64 under 'allow' -- JAX has fixed "
            "the inconsistency this probe recorded",
        )
        _require(
            converted == "float32",
            "X.7",
            f"plain conversion is no longer truncated under 'allow': {converted}",
        )
        _require(
            status in {"ok", "unmeasured"},
            "X.7",
            f"akriti construction under 'allow' returned {status}",
        )

    # `repr()` on the child side, `literal_eval` here: the message contains a
    # URL and no space after a full stop, so it must not be split or stripped.
    message = ast.literal_eval(_subprocess_probe(EXPLICIT_ERROR_PROGRAM))
    print(f"\n  7c. jax_explicit_x64_dtypes = 'error':\n      ValueError: {message}")
    _require(
        message == EXPLICIT_X64_ERROR,
        "X.7",
        f"'error' mode message changed:\n  observed={message!r}\n"
        f"  expected={EXPLICIT_X64_ERROR!r}",
    )
    print("      => Note JAX's own message names `allow_explicit_x64_dtypes`,")
    print("         which is not the flag's name. The flag is")
    print("         `jax_explicit_x64_dtypes`. Quoted as measured.")

    outside, inside, after, public = _subprocess_probe(EXPLICIT_SCOPED_PROGRAM).split()
    print("\n  7d. scoped form:")
    print(f"      outside the context -> {outside}")
    print(f"      inside  the context -> {inside}")
    print(f"      after   the context -> {after}")
    print(f"      exposed on jax.config (public API)? -> {public}")
    _require(
        outside == "float32" and inside == "float64" and after == "float32",
        "X.7",
        f"scoped behaviour changed: {outside}/{inside}/{after}",
    )
    _require(
        public == "False",
        "X.7",
        "jax.config now exposes explicit_x64_dtypes publicly -- the scoped lever "
        "has become public API and this probe's caveat can be dropped",
    )
    if not x64:
        print("\n  => F1 AS STATED IS FALSE. A JAX-backed diagram CAN exist with")
        print("     jax_enable_x64 = False. `jax_explicit_x64_dtypes = 'allow'`")
        print("     honours I2's float64 and B7's int64, arithmetic stays 64-bit,")
        print("     and it does NOT change the process-wide default dtype the way")
        print("     jax_enable_x64 does (X.6b) -- so it is the narrower and safer")
        print("     of the two levers. Its scoped form is private API, and the")
        print("     namespace still under-reports its own dtypes (X.4) even while")
        print("     producing them.")


def probe_akriti(jnp: Any, *, x64: bool) -> None:
    """X.7 -- the repository's own constructor, against JAX arrays."""
    rule("X.8  akriti.diagrams.core AGAINST JAX ARRAYS")
    try:
        # Function-scoped on purpose: this section is optional, and the
        # script must still run in a JAX environment with no akriti in it.
        from akriti.diagrams.core import (
            DiagramBatch,
            DiagramMeta,
            PersistenceDiagram,
            namespace_of,
        )
    except ImportError as exc:
        print(f"  NOT MEASURED -- akriti is not importable here ({exc}).")
        print("  Re-run with PYTHONPATH=<repo>/src to measure this section")
        print("  (RFC-0001 sec 9.2: an unmeasured row is reported, not skipped).")
        return

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        dims = jnp.asarray([0, 1], dtype=jnp.int32)
        births = jnp.asarray([0.0, 0.5], dtype=jnp.float64)
        deaths = jnp.asarray([1.0, jnp.inf], dtype=jnp.float64)
        offsets = jnp.asarray([0, 1, 2], dtype=jnp.int64)

    print(f"  namespace_of(dims) is jax.numpy -> {namespace_of(dims) is jnp}")
    print(f"  bar dtypes: {dims.dtype}, {births.dtype}, {deaths.dtype}")
    print(f"  offsets dtype: {offsets.dtype}")
    _require(
        namespace_of(dims) is jnp,
        "X.8",
        "namespace_of no longer resolves a JAX array to jax.numpy",
    )

    try:
        diagram = PersistenceDiagram(dims=dims, births=births, deaths=deaths)
    except ValueError as exc:
        diagram = None
        print(f"  PersistenceDiagram -> ValueError: {exc}")
        _require(
            not x64,
            "X.8",
            f"PersistenceDiagram failed with x64 enabled: {exc}",
        )
        _require(
            str(exc) == "births must be float64 (I2); got float32",
            "X.8",
            f"I2 rejection message changed: {str(exc)!r}",
        )
    else:
        print(
            f"  PersistenceDiagram -> OK, {diagram.n_bars} bars, "
            f"xp is jax.numpy = {diagram.xp is jnp}"
        )
        _require(x64, "X.8", "PersistenceDiagram constructed with x64 disabled")

    try:
        batch = DiagramBatch(
            dims=dims,
            births=births,
            deaths=deaths,
            offsets=offsets,
            metas=[DiagramMeta(), DiagramMeta()],
        )
    except ValueError as exc:
        print(f"  DiagramBatch       -> ValueError: {exc}")
        _require(not x64, "X.8", f"DiagramBatch failed with x64 enabled: {exc}")
    else:
        print(
            f"  DiagramBatch       -> OK, len {len(batch)}, "
            f"offsets {batch.offsets.dtype}"
        )
        _require(x64, "X.8", "DiagramBatch constructed with x64 disabled")

    if x64:
        # B7 in isolation. Under the default config it is unreachable: the bar
        # arrays fail I2 first, so the batch never gets as far as its offsets,
        # and B7's message cannot be quoted from a default-config run at all.
        try:
            DiagramBatch(
                dims=dims,
                births=births,
                deaths=deaths,
                offsets=jnp.asarray([0, 1, 2], dtype=jnp.int32),
                metas=[DiagramMeta(), DiagramMeta()],
            )
        except ValueError as exc:
            print(f"  B7 in isolation    -> ValueError: {exc}")
            _require(
                str(exc) == "offsets must be int64 (B7); got int32",
                "X.8",
                f"B7 rejection message changed: {str(exc)!r}",
            )
        else:
            _fail("X.8", "an int32 `offsets` array no longer violates B7")
    else:
        print("  B7 in isolation    -> unreachable: I2 rejects the bar arrays")
        print("                        before __post_init__ reaches `offsets`.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--x64",
        action="store_true",
        help="enable jax_enable_x64 before any array is created",
    )
    args = parser.parse_args(argv)

    # Both flags are readable from the environment, and either one set there
    # decides the config before this script gets a say. X.7 measures what they
    # do; inheriting one silently would relabel an allow-mode run as the
    # default config and quietly invert every result below.
    for name in ("JAX_ENABLE_X64", "JAX_EXPLICIT_X64_DTYPES"):
        if os.environ.get(name) and not (name == "JAX_ENABLE_X64" and args.x64):
            print(
                f"{name} is set in the environment, which decides the config "
                f"before this script does. Unset it so the probes below measure "
                f"the configuration they claim to measure.",
                file=sys.stderr,
            )
            return 2

    # Imported here, after argv is parsed and before any array exists, so
    # `--x64` is applied to a process that has not yet built anything.
    import jax
    import jaxlib

    if args.x64:
        jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    x64 = bool(jax.config.jax_enable_x64)
    _require(
        x64 is args.x64,
        "setup",
        f"requested x64={args.x64} but jax reports {x64}",
    )

    probe_environment(jax, jaxlib)
    probe_direct_requests(jnp, x64=x64)
    probe_dtype_identity(jnp)
    xp = probe_array_api_path(jnp, x64=x64)
    probe_namespace_info(xp, x64=x64)
    probe_conversion(jnp, jax, x64=x64)
    probe_config_timing()
    probe_explicit_x64(x64=x64)
    probe_akriti(jnp, x64=x64)

    rule("VERDICT")
    if x64:
        print("  With jax_enable_x64 = True, every RFC-0001 dtype requirement is")
        print("  satisfiable on JAX: float64 bars (I2), int32 dims (I2), int64")
        print("  offsets (B7), and both types construct.")
    else:
        print("  With BOTH flags at their defaults (jax_enable_x64=False,")
        print("  jax_explicit_x64_dtypes='warn') no JAX-backed PersistenceDiagram")
        print("  and no DiagramBatch can be constructed: I2's float64 and B7's")
        print("  int64 are both unsatisfiable, and core.py rejects the arrays with")
        print("  `births must be float64 (I2); got float32` (X.8).")
        print()
        print("  But `jax_enable_x64` is not the only lever, so the claim that no")
        print("  JAX-backed diagram can exist without it is FALSE (X.7):")
        print("  jax_explicit_x64_dtypes='allow' honours both dtypes with")
        print("  jax_enable_x64 still False, and both types then construct.")
        print()
        print("  Three qualifications on the default-config failure:")
        print("   - I2's `dims` int32 half IS satisfiable. Default JAX supplies")
        print("     int32 natively; only float64 (I2) and int64 (B7) fail.")
        print("   - The truncation is NOT silent on an explicit request (X.1, X.3):")
        print("     it carries one UserWarning. It IS silent when an existing")
        print("     64-bit array is merely converted (X.5), which is the adapter")
        print("     path and the one F1 did not name.")
        print("   - B7 is unreachable under the default config: I2 rejects the bar")
        print("     arrays before __post_init__ ever reads `offsets` (X.8).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
