#!/usr/bin/env python3
"""Cost of resolving namespaces through `array-api-compat` — evidence for §3.3.

`torch.Tensor` does not implement `__array_namespace__`, so §3's definition of
`Array` excludes it and no diagram can currently be torch-backed. One way to
close that is to resolve namespaces through `array_api_compat.array_namespace`
rather than by calling `__array_namespace__` directly. This script measures
what that costs the two backends that *do* conform, NumPy and JAX, since a
resolver adopted for torch's sake would sit in front of them as well.

No decision row carries this yet; §12 has no entry for the torch gap.

Run:

    pip install "numpy>=2.0" array-api-compat    # jax optional, see part 5
    python rfcs/evidence/array_api_compat_overhead.py

Figures in the docstrings below were measured on CPython 3.14, numpy 2.5.1,
array-api-compat 1.15.0, best-of-7. They are one machine's numbers; the ratios
are what transfer, not the absolute nanoseconds.

Five findings, in descending order of how much they matter:

0. On numpy 2.5 the wrappers buy almost nothing (part 6). `device=`,
   `unique_values`, `cumulative_sum(include_initial=)`, `reshape(copy=)` and
   the 0-d `nonzero` rejection are all native now; most of part 1's eleven
   wrappers are vestigial, kept for older NumPy. The one live correction is
   finding 4's sort default, and `stable=True` buys that without the
   dependency. So compat's value here is the torch path, not conformance.

1. JAX pays nothing, structurally. array-api-compat has no JAX wrapper: a
   `jax.Array` routes to `jnp.empty(0).__array_namespace__()`, which is
   `jax.numpy` itself, and the dispatch reaching it is `lru_cache`d on the
   class. Every `xp.foo` afterwards is JAX's own function object.
2. NumPy pays a flat sub-microsecond frame on the 11-of-26 namespace
   functions that are actually wrapped, and nothing at all on the other 15,
   which are numpy's own objects by identity. Operators never touch the
   namespace. §7's `canonical()` is 1.16x at 40 bars and 1.00x from 100k up.
3. `array_namespace()` on a NumPy array returns `array_api_compat.numpy`,
   NOT `numpy`. A codebase that resolves natively in one place and through
   the helper in another gets two namespace objects for one backend, and I7's
   `is` then raises on arrays that legitimately share a namespace. Resolution
   has to go through exactly one function.
4. Unrelated to compat, and the reason part 4 exists: numpy's main namespace
   defaults `sort`/`argsort` to `stable=None` -> quicksort, where the array
   API standard specifies `stable=True`. §7 passes the keyword explicitly at
   all three call sites and is safe; a future call site that omits it gets an
   unstable sort on NumPy and a stable one under `array_api_strict`, which
   the conformance suite cannot catch because `array_api_strict` is the side
   that is correct. Same shape as §7's `lexsort` trap.
"""

from __future__ import annotations

import importlib.util
import operator
import sys
import time
from collections.abc import Callable
from functools import partial
from typing import Any

if importlib.util.find_spec("array_api_compat") is None:
    sys.exit("array-api-compat is not installed; see this file's docstring.")

import array_api_compat
import array_api_compat.numpy as cnp
import numpy as np

if int(np.__version__.split(".")[0]) < 2:
    sys.exit(f"numpy>=2.0 required for a native baseline; found {np.__version__}")

# The RFC's own 40-point circle (§5.1, A.1) through Appendix A.6's ~1M-bar
# scale. A.6 measures H0 == point count exactly, so these are cloud sizes too.
SIZES = (40, 1_000, 100_000, 1_000_000)

# Every namespace function this document names or that `core/` plausibly
# reaches for. `lexsort` is here because §7 forbids it and the inventory
# should show it is numpy's own, not something compat supplies.
SURFACE = (
    "sort",
    "argsort",
    "concat",
    "take",
    "max",
    "min",
    "sum",
    "any",
    "all",
    "isnan",
    "isinf",
    "isdtype",
    "asarray",
    "astype",
    "unique_values",
    "nonzero",
    "zeros",
    "empty",
    "equal",
    "abs",
    "where",
    "arange",
    "searchsorted",
    "cumulative_sum",
    "reshape",
    "lexsort",
)


def bench(fn: Callable[[], Any], reps: int, rounds: int = 7) -> float:
    """Best-of-`rounds` mean nanoseconds per call over `reps` calls."""

    def once() -> float:
        t0 = time.perf_counter_ns()
        for _ in range(reps):
            fn()
        return (time.perf_counter_ns() - t0) / reps

    return min(once() for _ in range(rounds))


def compare(
    label: str, native: Callable[[], Any], compat: Callable[[], Any], reps: int
) -> None:
    a, b = bench(native, reps), bench(compat, reps)
    print(f"  {label:<38} {a:>13,.0f} {b:>13,.0f}  {b / a:>5.2f}x")


def canonical(xp: Any, dims: Any, births: Any, deaths: Any) -> Any:
    """§7's three-pass composition, verbatim, as the sort-heaviest thing here."""
    order = xp.argsort(deaths, stable=True)
    order = xp.take(order, xp.argsort(xp.take(births, order), stable=True))
    return xp.take(order, xp.argsort(xp.take(dims, order), stable=True))


def part1_inventory() -> None:
    """Which namespace functions carry a wrapper at all.

    Measured: 11 wrapped, 15 identical to numpy's own object. Only the first
    group can cost anything, and what it costs is one Python frame.
    """
    print("\n1. Wrapped vs. passed through (`array_api_compat.numpy`)\n")
    wrapped, identical = [], []
    for name in SURFACE:
        c, n = getattr(cnp, name, None), getattr(np, name, None)
        if c is None:
            print(f"  absent from the compat namespace: {name}")
        elif c is n:
            identical.append(name)
        else:
            wrapped.append(name)
    print(f"  wrapped   ({len(wrapped):>2}): {', '.join(wrapped)}")
    print(f"  identical ({len(identical):>2}): {', '.join(identical)}")


def part2_resolution() -> None:
    """Resolution cost, and the identity fact that constrains the design.

    Measured: 309 ns native, 858 ns through the helper. The `is numpy` line
    below is the one to read — see finding 3 in the module docstring.
    """
    print("\n2. Namespace resolution                     native (ns)  compat (ns)\n")
    x = np.zeros(3)
    resolve = array_api_compat.array_namespace
    compare("array_namespace(x)", x.__array_namespace__, partial(resolve, x), 20_000)
    print()
    for label, held in (
        ("x.__array_namespace__() is numpy", x.__array_namespace__() is np),
        ("array_namespace(x) is numpy", resolve(x) is np),
        ("array_namespace(x) is compat.numpy", resolve(x) is cnp),
        (
            "identity across two arrays (I7)",
            resolve(np.zeros(1)) is resolve(np.ones(9)),
        ),
    ):
        print(f"  {label:<42} {held}")


def part3_per_op(rng: np.random.Generator) -> None:
    """Per-operation overhead at equal semantics, and §7 end-to-end.

    Both sides are asked for the same work: where the standard's default
    differs from numpy's, the keyword is passed explicitly, which is what §7
    already does. Measured: 1.16x on `canonical()` at 40 bars, 1.00x at 100k
    and 1M. Operators are 1.00x everywhere -- they never reach the namespace.
    """
    for n in SIZES:
        births = rng.random(n)
        deaths = births + rng.random(n)
        dims = rng.integers(0, 3, n).astype(np.int32)
        idx = np.arange(n)
        reps = 2_000 if n <= 1_000 else (100 if n <= 100_000 else 10)

        print(f"\n3. n = {n:>9,} bars                  native (ns)  compat (ns)\n")
        compare(
            "argsort(births, stable=True)",
            partial(np.argsort, births, stable=True),
            partial(cnp.argsort, births, stable=True),
            reps,
        )
        compare(
            "sort(births, stable=True)",
            partial(np.sort, births, stable=True),
            partial(cnp.sort, births, stable=True),
            reps,
        )
        compare(
            "take(births, idx)",
            partial(np.take, births, idx),
            partial(cnp.take, births, idx),
            reps,
        )
        compare(
            "concat((births, births))",
            partial(np.concatenate, (births, births)),
            partial(cnp.concat, (births, births)),
            reps,
        )
        compare("max(births)", partial(np.max, births), partial(cnp.max, births), reps)
        compare(
            "astype(dims, float64)",
            partial(np.astype, dims, np.float64),
            partial(cnp.astype, dims, cnp.float64),
            reps,
        )
        compare(
            "§7 canonical(), three passes",
            partial(canonical, np, dims, births, deaths),
            partial(canonical, cnp, dims, births, deaths),
            reps,
        )
        # Not a comparison: `deaths - births` dispatches on the array object and
        # never reaches a namespace, so no wrapper can sit in front of it. Shown
        # as one figure so its scale can be read against the rows above.
        op = bench(partial(operator.sub, deaths, births), reps)
        print(f"  {'operator: deaths - births (no namespace)':<38} {op:>13,.0f}")


def part4_stability(rng: np.random.Generator) -> None:
    """What stability itself costs, and NumPy's deviation from the standard.

    This is not a compat cost. Asking numpy for the standard's semantics
    (`stable=True`) against numpy's own default (`stable=None`, quicksort)
    measures ~9.6x on `sort` and ~2.9x on `argsort` at 1M elements. §7 already
    pays it deliberately, at all three call sites, and needs to: an unstable
    pass loses the ordering the previous pass established.

    The hazard is the default, not the price. A call site that omits the
    keyword is silently unstable on NumPy and stable under `array_api_strict`.
    """
    n = 1_000_000
    x = rng.random(n)
    print(f"\n4. Stability, numpy only, n = {n:,}    stable=None   stable=True\n")
    compare("np.sort(x)", partial(np.sort, x), partial(np.sort, x, stable=True), 5)
    compare(
        "np.argsort(x)", partial(np.argsort, x), partial(np.argsort, x, stable=True), 5
    )
    print("\n  The standard specifies stable=True as the default for both.")
    print("  numpy's main namespace defaults to stable=None -> quicksort.")
    print("  §7 passes the keyword explicitly; nothing enforces that it keeps")
    print("  doing so at a call site added later.")


def part5_jax() -> None:
    """JAX resolves to its own namespace, so there is nothing to measure.

    array-api-compat ships no JAX wrapper. `_cls_to_namespace` is `lru_cache`d
    on the class and routes `jax.Array` to `jnp.empty(0).__array_namespace__()`,
    which for jax>=0.4.32 is `jax.numpy` itself. This asserts that rather than
    timing it: a namespace that is the library's own module has no overhead to
    measure, and a future version that started wrapping JAX would fail here.
    """
    print("\n5. JAX\n")
    if importlib.util.find_spec("jax") is None:
        print("  jax not installed; skipping. array-api-compat ships no JAX")
        print("  wrapper, so array_namespace(jax_array) is jax.numpy itself.")
        return
    import jax.numpy as jnp

    resolved = array_api_compat.array_namespace(jnp.zeros(3))
    print(f"  array_namespace(jnp.zeros(3)) is jax.numpy  {resolved is jnp}")
    print(f"  resolved namespace                          {resolved.__name__}")
    if resolved is not jnp:
        print("  !! JAX is now wrapped; part 3's argument does not cover it.")


def part6_still_corrected(rng: np.random.Generator) -> None:
    """The other half of the decision: what compat still buys on this NumPy.

    A wrapper existing does not mean the deviation it patches still exists.
    Most of part 1's eleven are vestigial on numpy 2.5, kept for older NumPy:
    `device=`, `unique_values`, `cumulative_sum(include_initial=)`,
    `reshape(copy=)` and the 0-d `nonzero` rejection are all native now.

    One survives, and it is the one §7 depends on. This probes rather than
    asserts, so a later NumPy that closes the gap turns the OK column over
    and the conclusion updates itself.
    """
    print("\n6. What compat still corrects on this NumPy\n")
    x = np.array([3.0, 1.0, 1.0, 2.0])
    counts = np.array([2, 3, 1])
    probes: tuple[tuple[str, Callable[[], Any]], ...] = (
        ("zeros(3, device=)", partial(np.zeros, 3, device="cpu")),
        ("asarray(x, device=)", partial(np.asarray, x, device="cpu")),
        ("arange(3, device=)", partial(np.arange, 3, device="cpu")),
        ("unique_values(x)", partial(np.unique_values, x)),
        (
            "cumulative_sum(c, include_initial=True)",
            partial(np.cumulative_sum, counts, include_initial=True),
        ),
        ("reshape(x, (2,2), copy=True)", partial(np.reshape, x, (2, 2), copy=True)),
    )
    for label, fn in probes:
        try:
            fn()
            print(f"  native on numpy {np.__version__}: {label:<40} OK")
        except TypeError as exc:
            print(f"  native on numpy {np.__version__}: {label:<40} {exc}")

    # The standard requires stable=True by default; numpy defaults to
    # stable=None -> quicksort. Ties are the only place that is observable.
    keys = rng.integers(0, 4, 100_000).astype(np.float64)
    stable = np.argsort(keys, stable=True)
    print()
    print(
        f"  np.argsort(keys)  is the stable order       "
        f"{np.array_equal(np.argsort(keys), stable)}"
    )
    print(
        f"  cnp.argsort(keys) is the stable order       "
        f"{np.array_equal(cnp.argsort(keys), stable)}"
    )
    print("\n  §7's three-pass composition is incorrect under an unstable pass,")
    print("  so this is the one correction that still has a customer here --")
    print("  and passing stable=True buys it without the dependency.")


def main() -> None:
    rng = np.random.default_rng(0)
    print(
        f"numpy {np.__version__} | array-api-compat {array_api_compat.__version__}"
        f" | python {sys.version.split()[0]}"
    )
    part1_inventory()
    part2_resolution()
    part3_per_op(rng)
    part4_stability(rng)
    part5_jax()
    part6_still_corrected(rng)
    print("\n=> Adopting array-api-compat costs JAX nothing and NumPy one Python")
    print("   frame per wrapped call, and on this NumPy it buys them nothing")
    print("   they cannot have for free. Its value is the torch path alone.")
    print("   The design constraint is part 2's identity fact. RFC-0001 §3.3.")


if __name__ == "__main__":
    main()
