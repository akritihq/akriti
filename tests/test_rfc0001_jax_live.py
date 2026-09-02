"""Live JAX coverage for RFC-0001 §3.3's D23 64-bit configuration constraint.

§3.3 promises that a diagram built from JAX arrays stays JAX-backed, and D23
scopes that promise: JAX's defaults supply neither dtype the type requires, so
the promise holds only under a 64-bit configuration the *caller* sets and this
library must not. A support claim no test runs is what §9 exists to distrust,
so both halves are asserted here -- that the default configuration refuses the
diagram, and that `jax_explicit_x64_dtypes='allow'` is enough to build one.

JAX is not in the default test environment and does not enter the dependency
closure to satisfy this; the module skips where it is absent. Appendix A.11
carries the measurement and `rfcs/evidence/jax_x64.py` reproduces it.

**Ordering matters and is why the two halves cannot share a process.** Both
flags are process-global (§3.3), so a test that flipped one would leak the
change into every later test in the session. Each half therefore runs in a
subprocess with the configuration set from the environment, which is also the
only way a caller can set it before importing JAX.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("jax")

pytestmark = [pytest.mark.backend, pytest.mark.jax]


_BUILD = """
import jax.numpy as jnp

from akriti.diagrams import DiagramBatch, PersistenceDiagram

dims = jnp.asarray([0, 1], dtype=jnp.int32)
births = jnp.asarray([0.0, 0.2], dtype=jnp.float64)
deaths = jnp.asarray([float("inf"), 0.5], dtype=jnp.float64)
d = PersistenceDiagram(dims=dims, births=births, deaths=deaths)
b = DiagramBatch.from_diagrams([d])
print("BUILT", d.births.dtype, b.offsets.dtype, type(d.dims).__module__)
"""


def _run(source: str, **env_overrides: str) -> subprocess.CompletedProcess[str]:
    import os

    env = dict(os.environ)
    env.update(env_overrides)
    # `filterwarnings = ["error"]` is this suite's setting, not the child's;
    # the truncation warning is what we are measuring, not a failure.
    env.pop("PYTHONWARNINGS", None)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_default_jax_configuration_cannot_build_a_diagram() -> None:
    """§3.3, D23: I2 refuses the truncated bars, names the dtype, and points here.

    The failure a caller sees is I2's ordinary `ValueError` rather than
    anything JAX-specific, which is why §3.3 has to explain it: `float32` is
    what JAX handed back for an explicit `float64` request, not what the
    caller asked for. §3.3 requires the error to point at itself for that
    reason -- the dtype alone tells a caller what is wrong and nothing about
    a flag only they can set -- so the pointer is asserted alongside it.
    """
    result = _run(_BUILD)

    assert result.returncode != 0, result.stdout
    assert "float64" in result.stderr
    assert "float32" in result.stderr
    assert "I2" in result.stderr
    # The section number alone, not "§3.3": the child writes its traceback
    # through `sys.stderr`, which escapes a non-ASCII character rather than
    # failing where the process encoding cannot carry one, and the section
    # sign is the only part of the pointer that would not survive that.
    assert "3.3" in result.stderr


def test_explicit_x64_dtypes_allow_is_enough_to_build_one() -> None:
    """§3.3, D23: the narrow flag satisfies I2 and B7 without `enable_x64`.

    This is the half that makes D23 a scoping decision rather than a
    withdrawal: a JAX-backed diagram *is* constructible, and the flag that
    gets there changes only dtypes requested by name.
    """
    result = _run(_BUILD, JAX_EXPLICIT_X64_DTYPES="allow")

    assert result.returncode == 0, result.stderr
    built, births, offsets, module = result.stdout.split()
    assert built == "BUILT"
    assert births == "float64"
    assert offsets == "int64"
    assert module.startswith("jax")


def test_this_library_never_sets_either_flag() -> None:
    """§3.3, D23: both flags are process-global, so setting one reaches outside.

    A grep rather than a behavioural assertion, deliberately: the prohibition
    is on the source, and a runtime check would pass on a library that set the
    flag and then restored it -- which is the thing being prohibited.
    """
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "akriti"
    offenders = [
        path
        for path in src.rglob("*.py")
        if "jax_enable_x64" in path.read_text()
        or "jax_explicit_x64_dtypes" in path.read_text()
    ]
    assert offenders == []
