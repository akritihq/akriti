#!/usr/bin/env python3
"""Drive akriti: probe the environment, then run the real API through it.

Why this exists -- the environment matrix, and why a green `pytest` is not
enough -- is in `.claude/skills/run-akriti/SKILL.md`. This is the usage.

    python .claude/skills/run-akriti/driver.py probe
    python .claude/skills/run-akriti/driver.py smoke [--strict]
    python .claude/skills/run-akriti/driver.py closure
    python .claude/skills/run-akriti/driver.py matrix [--rows rips,jax,...]
    python .claude/skills/run-akriti/driver.py exec  'expr or file.py'
    python .claude/skills/run-akriti/driver.py all   [--strict]

Nothing here is a test. `tests/` is the specification's checker; this is the
handle you use when you want to watch the library actually do something, or
when you have changed one function and want to call it without writing a
preamble first (`exec`).
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
import traceback
from pathlib import Path

# .claude/skills/run-akriti/driver.py -> up three to the repo root. Every mode
# runs subprocesses with cwd=REPO and reads fixtures from it, so the driver
# works from any cwd -- but only while it sits where the skill puts it. A copy
# moved somewhere else would otherwise resolve REPO to an unrelated directory
# and fail later with a confusing missing-fixture or pip error.
REPO = Path(__file__).resolve().parents[3]
if not (REPO / "pyproject.toml").is_file() or not (REPO / "src" / "akriti").is_dir():
    sys.exit(
        f"driver.py expects to live at <repo>/.claude/skills/run-akriti/, which "
        f"would make the repo root {REPO} -- but that is not an akriti checkout. "
        f"Run the copy inside the repo."
    )

# The one directory name the driver owns. It starts with `.venv` so that in the
# repo root -- the default workdir -- .gitignore's existing `.venv*/` covers it
# and nothing the driver builds can be committed by accident.
SCRATCH = ".venv-driver"

# Extras that carry an import-checkable payload, mapped to the module that
# proves they are installed. pyproject is the authority on which extras exist;
# `probe` cross-checks this table against it, so a new extra fails there
# rather than becoming a row nobody probes.
EXTRAS = {
    "numpy": ("numpy", "array namespace, and the Python-row adapter fallback"),
    "rips": ("ripser", "Vietoris-Rips via Ripser"),
    "alpha": ("gudhi", "GUDHI (Rips/Alpha)"),
    "distances": ("persim", "diagram distances"),
    "parquet": ("pyarrow", "to_parquet()"),
    "torch": ("torch", "tensor namespace, via array-api-compat"),
    "jax": ("jax", "JAX namespace, under a caller-set x64 config (D23)"),
    "bio": ("anndata", "anndata interop"),
}

# Extras that name tooling rather than a backend, which the cross-check must
# not expect EXTRAS to carry. By-reference extras (`io`, `dev`) need no
# listing: they are recognised by their requirements.
TOOLING_EXTRAS = frozenset({"test", "lint"})


def _declared_extras() -> dict[str, list[str]] | None:
    """pyproject's `[project.optional-dependencies]`; None below 3.11 (no tomllib)."""
    try:
        import tomllib
    except ImportError:
        return None
    with (REPO / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)["project"]["optional-dependencies"]


GREEN, RED, DIM, BOLD, OFF = "\033[32m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"
if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
    GREEN = RED = DIM = BOLD = OFF = ""


class Report:
    """Ran / skipped, where a skip is a first-class result with a reason."""

    def __init__(self) -> None:
        self.ran: list[str] = []
        self.skipped: list[tuple[str, str]] = []
        self.failed: list[tuple[str, str]] = []

    def ok(self, what: str, detail: str = "") -> None:
        self.ran.append(what)
        tail = f"  {DIM}{detail}{OFF}" if detail else ""
        print(f"  {GREEN}ok{OFF}   {what}{tail}")

    def skip(self, what: str, why: str) -> None:
        self.skipped.append((what, why))
        print(f"  {DIM}skip {what}  -- {why}{OFF}")

    def fail(self, what: str, why: str) -> None:
        self.failed.append((what, why))
        print(f"  {RED}FAIL{OFF} {what}  -- {why}")

    def summary(self, *, strict: bool) -> int:
        print()
        print(
            f"{BOLD}{len(self.ran)} exercised, "
            f"{len(self.skipped)} not exercised, "
            f"{len(self.failed)} failed{OFF}"
        )
        for what, why in self.skipped:
            print(f"  {DIM}not exercised: {what} -- {why}{OFF}")
        for what, why in self.failed:
            print(f"  {RED}failed: {what} -- {why}{OFF}")
        if self.failed:
            return 1
        if strict and self.skipped:
            print(f"{RED}--strict: an unexercised path is a failure here.{OFF}")
            return 1
        return 0


def have(module: str) -> bool:
    """Importable AND actually usable.

    `find_spec` alone is not enough. A `pip install torch` that runs out of
    disk part-way leaves a package that imports fine and has no `__version__`
    -- every later call then fails somewhere deep with an error naming
    neither torch nor the install. Import it and look for the attribute.
    """
    try:
        if importlib.util.find_spec(module) is None:
            return False
        mod = importlib.import_module(module)
    except Exception:
        return False
    # array_api_compat is the one entry with no __version__ convention to lean
    # on; every other module in EXTRAS carries one.
    return module == "array_api_compat" or hasattr(mod, "__version__")


def version_of(module: str) -> str:
    try:
        return getattr(importlib.import_module(module), "__version__", "?")
    except Exception as exc:  # a broken install is not the same as an absent one
        return f"<import failed: {type(exc).__name__}>"


def broken(module: str) -> bool:
    """Present on disk but not usable -- worth naming rather than calling absent."""
    try:
        return importlib.util.find_spec(module) is not None and not have(module)
    except Exception:
        return False


def scratch(args: argparse.Namespace, mode: str) -> Path:
    """The one directory a mode may create and delete.

    Always a driver-owned child of the workdir, never the workdir itself, so
    `--workdir ~/.cache` cannot delete `~/.cache`.
    """
    parent = Path(args.workdir) if args.workdir else REPO
    return parent / SCRATCH / mode


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts" if os.name == "nt" else "bin") / "python"


def discard(path: Path) -> None:
    """Remove what a mode built, and the `.venv-driver` parent once it is empty."""
    shutil.rmtree(path, ignore_errors=True)
    with contextlib.suppress(OSError):
        path.parent.rmdir()


# --------------------------------------------------------------------------
# probe


def cmd_probe(args: argparse.Namespace) -> int:
    import akriti

    print(
        f"{BOLD}akriti{OFF} {akriti.__version__}   "
        f"{DIM}{Path(akriti.__file__).parent}{OFF}"
    )
    print(f"{BOLD}python{OFF} {sys.version.split()[0]}   {DIM}{sys.executable}{OFF}")
    print()
    print(f"{BOLD}extras{OFF}")
    rep = Report()
    for extra, (module, what) in EXTRAS.items():
        if have(module):
            rep.ok(f"akriti[{extra}]", f"{module} {version_of(module)} -- {what}")
        elif broken(module):
            rep.fail(
                f"akriti[{extra}]",
                f"{module} is on disk but unusable (a part-finished "
                f"install?) -- reinstall before trusting any {module} result",
            )
        else:
            rep.skip(f"akriti[{extra}]", f"{module} not installed -- {what}")

    declared = _declared_extras()
    if declared is None:
        print(f"  {DIM}not cross-checked against pyproject: tomllib needs 3.11+{OFF}")
    else:
        for extra, reqs in declared.items():
            by_reference = all(req.startswith("akriti[") for req in reqs)
            if extra in EXTRAS or extra in TOOLING_EXTRAS or by_reference:
                continue
            rep.fail(
                f"akriti[{extra}]",
                "declared in pyproject.toml, unknown to the driver's EXTRAS",
            )
        for extra in sorted(EXTRAS.keys() - declared.keys()):
            rep.fail(
                f"akriti[{extra}]",
                "in the driver's EXTRAS, not declared in pyproject.toml",
            )

    print()
    print(f"{BOLD}also named in CLAUDE.md{OFF}")
    for sub in ("akriti.core", "akriti.castle"):
        # find_spec, not have(): these subpackages carry no __version__, so
        # have() would call them absent on the day they land.
        found = importlib.util.find_spec(sub) is not None
        print(f"  {DIM}{'present' if found else 'absent '}  {sub}{OFF}")
    repro = (REPO / "repro").is_dir()
    print(f"  {DIM}{'present' if repro else 'absent '}  repro/{OFF}")
    # probe measures and exercises nothing, so --strict has nothing to bind to
    # here: an extra that is missing is a fact to read, and `smoke --strict` is
    # where an unexercised path becomes a failure.
    return rep.summary(strict=False)


# --------------------------------------------------------------------------
# smoke -- the real API, end to end, on whatever is installed


def _circle(n: int, noise: float, seed: int):
    import numpy as np

    r = np.random.default_rng(seed)
    t = r.uniform(0, 2 * np.pi, n)
    return np.c_[np.cos(t), np.sin(t)] + noise * r.standard_normal((n, 2))


def cmd_smoke(args: argparse.Namespace) -> int:
    import warnings

    rep = Report()

    if not have("numpy"):
        # Not a skip: without a namespace there is no flow to run at all.
        print(
            f"{RED}numpy is absent, so there is no array namespace to build a "
            f"diagram in.{OFF}"
        )
        print(
            "This is a legitimate install (the default closure is empty) but "
            "nothing below can run."
        )
        print("Check that boundary with `closure` instead, or install `.[test,rips]`.")
        return 1

    import numpy as np

    from akriti.diagrams import (
        DiagramBatch,
        from_array,
        from_gudhi,
        from_persim,
        from_ripser,
        load,
        save,
        to_arrays,
        to_csv,
    )

    print(f"{BOLD}adapters{OFF}")

    # from_array is the one adapter that needs no backend at all.
    # NOTE the column order: (birth, death, dim), giotto's, not ripser's.
    arr = np.array([[0.0, 1.0, 0], [0.0, np.inf, 0], [0.5, 2.0, 1]])
    d_arr = from_array(arr)
    rep.ok("from_array", f"{d_arr.n_bars} bars, {int(d_arr.essential.sum())} essential")

    diagrams: list = []

    if have("ripser"):
        import ripser

        d = from_ripser(ripser.ripser(_circle(60, 0.05, 0), maxdim=1))
        h1 = d.dim(1)
        rep.ok(
            "from_ripser",
            f"{d.n_bars} bars; H1 persistence "
            f"{np.round(np.asarray(h1.persistence), 3).tolist()}; "
            f"backend={d.meta.backend} {d.meta.backend_version}",
        )
        diagrams.append(d)
    else:
        rep.skip("from_ripser", "ripser not installed -- pip install -e '.[rips]'")

    if have("gudhi"):
        import gudhi

        st = gudhi.RipsComplex(
            points=_circle(50, 0.05, 1), max_edge_length=3.0
        ).create_simplex_tree(max_dimension=2)
        d = from_gudhi(st.persistence())
        rep.ok(
            "from_gudhi",
            f"{d.n_bars} bars, {int(d.essential.sum())} essential; "
            f"backend={d.meta.backend} {d.meta.backend_version}",
        )
        diagrams.append(d)
    else:
        rep.skip("from_gudhi", "gudhi not installed -- pip install -e '.[alpha]'")

    # No gate on persim: the adapter imports nothing and reads persim's version
    # from installed metadata, so it runs wherever numpy does. persim consumes
    # diagrams and computes no homology, so from_persim records no coeff_field
    # (RFC-0001 s11); what persim being installed changes is backend_version.
    d = from_persim([np.array([[0.0, 1.0], [0.0, 2.0]]), np.array([[0.5, 1.5]])])
    rep.ok(
        "from_persim",
        f"{d.n_bars} bars; coeff_field={d.meta.coeff_field!r}; "
        f"backend_version={d.meta.backend_version!r}",
    )

    giotto_fixture = REPO / "tests" / "fixtures" / "giotto_output.json"
    if giotto_fixture.is_file():
        from akriti.diagrams import from_giotto

        raw = json.loads(giotto_fixture.read_text(encoding="utf-8"))
        # `samples` is the infinity_values=inf capture -- the one setting
        # from_giotto accepts, so essential classes arrive as inf rather than
        # as a death at max_edge_length. Rebuilt dtype-and-shape exact, the
        # way tests/conftest.py's _array does it.
        spec = raw["samples"]["reduced_false"]["batch"]
        block = np.asarray(spec["data"], dtype=spec["dtype"]).reshape(spec["shape"])
        try:
            # strip_padding is passed explicitly. Left unset, from_giotto warns
            # that giotto pads a batch to a common row count with rows where
            # birth == death, which are indistinguishable from genuine
            # zero-persistence bars (A.2). Keeping them is the faithful
            # reading, but the choice has to be made rather than defaulted.
            batch = from_giotto(
                block,
                reduced_homology=False,
                infinity_values=float("inf"),
                strip_padding=False,
            )
            stripped = from_giotto(
                block,
                reduced_homology=False,
                infinity_values=float("inf"),
                strip_padding=True,
            )
            kept = int(np.asarray(batch.bar_counts).sum())
            # `padding_removed` is the adapter's own count of what
            # strip_padding=True dropped (s11.1). It counts trivial rows, not
            # padding: the adapter cannot tell the two apart and MUST NOT
            # guess, so neither does the label below.
            trivial = sum(m.provenance["padding_removed"] for m in stripped.metas)
            rep.ok(
                "from_giotto",
                f"{len(batch)} diagrams from the committed fixture "
                f"({raw['versions'].get('giotto-tda', '?')}); {kept} bars "
                f"kept, {trivial} trivial (birth == death; padding and "
                f"zero-persistence bars are indistinguishable, s11.1); "
                f"giotto is AGPLv3 and is never installed here",
            )
        except Exception as exc:
            rep.fail("from_giotto", f"{type(exc).__name__}: {exc}")
    else:
        rep.skip("from_giotto", f"fixture missing: {giotto_fixture}")

    if not diagrams:
        # No backend installed, so nothing here came out of a real persistence
        # computation -- but from_array's diagram is a valid one, and the batch
        # and io surface below is worth exercising on it rather than skipping
        # wholesale. Say which it is.
        rep.skip(
            "batch / io / export on real backend output",
            "no persistence backend installed; the section below runs on "
            "the from_array diagram instead",
        )
        diagrams = [d_arr]

    print()
    print(f"{BOLD}diagram surface{OFF}")
    d = diagrams[0]
    rep.ok("finite / essential", f"{d.finite.n_bars} finite of {d.n_bars}")
    rep.ok("content_hash", d.content_hash[:32] + "...")
    fz = d.finitize(at="max_finite_death")
    rep.ok(
        "finitize(at='max_finite_death')",
        f"essential now {int(fz.essential.sum())}; "
        f"provenance={fz.meta.provenance.get('essential_bars')!r}",
    )
    no_essential = d.finitize(at="drop")
    rep.ok(
        "finitize(at='drop')",
        f"{no_essential.n_bars} bars; "
        f"provenance={no_essential.meta.provenance.get('essential_bars')!r}",
    )
    rep.ok("canonical()", f"{d.canonical().n_bars} bars")
    with warnings.catch_warnings():
        # to_arrays warns that it discards DiagramMeta and inter-degree order.
        # That is the designed contract, and it is stated in the line below.
        warnings.simplefilter("ignore", UserWarning)
        degrees = sorted(to_arrays(d))
    rep.ok("to_arrays", f"degrees {degrees} (warns: discards DiagramMeta)")

    print()
    print(f"{BOLD}batch{OFF}")
    if have("ripser"):
        import ripser

        batch = DiagramBatch.from_diagrams(
            [
                from_ripser(ripser.ripser(_circle(50, 0.05, s), maxdim=1))
                for s in range(3)
            ]
        )
    else:
        batch = DiagramBatch.from_diagrams(diagrams)
    rep.ok(
        "DiagramBatch.from_diagrams",
        f"len={len(batch)}, bar_counts={np.asarray(batch.bar_counts).tolist()}",
    )
    rep.ok("batch.content_hash", batch.content_hash[:32] + "...")

    print()
    print(f"{BOLD}io and export{OFF}")
    # to_csv and to_parquet warn that they discard metadata. That is the
    # designed behaviour, not a problem -- but pytest runs under
    # filterwarnings=error, so a test calling these must expect the warning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        csv = to_csv(batch)
        rep.ok(
            "to_csv",
            f"{len(csv.splitlines())} lines; "
            f"header={csv.splitlines()[0]!r} "
            f"(warns: discards DiagramMeta)",
        )
        if have("pyarrow"):
            from akriti.diagrams import to_parquet

            table = to_parquet(batch)
            rep.ok(
                "to_parquet",
                f"{table.num_rows} rows, schema {[f.name for f in table.schema]}",
            )
        else:
            rep.skip(
                "to_parquet", "pyarrow not installed -- pip install -e '.[parquet]'"
            )

    tmp = scratch(args, "smoke")
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        path = tmp / "batch.akd"
        save(batch, path)
        rt = load(path)
        if rt == batch and rt.content_hash == batch.content_hash:
            rep.ok(
                "save/load round-trip",
                f"{path.stat().st_size} bytes; equal and hash-identical",
            )
        else:
            rep.fail("save/load round-trip", "round-trip is not equal to the original")
    finally:
        if not args.keep:
            discard(tmp)

    print()
    print(
        f"{BOLD}array namespaces{OFF} {DIM}(core/ is written against the "
        f"array API standard, not numpy){OFF}"
    )
    rep.ok("numpy", f"diagram.xp = {d_arr.xp.__name__}")

    if have("torch"):
        import torch

        t = torch.tensor([[0.0, 1.0, 0.0], [0.5, 2.0, 1.0]], dtype=torch.float64)
        dt = from_array(t)
        rep.ok(
            "torch",
            f"diagram.xp = {dt.xp.__name__} (the lazy array-api-compat fallback, D18)",
        )
    else:
        rep.skip("torch", "torch not installed -- pip install -e '.[torch]'")

    if have("jax"):
        import jax

        # D23: akriti MUST NOT set this; the caller does, and the driver is a
        # caller. A default JAX install truncates to float32 and from_array
        # raises I2's ValueError. This is the narrow lever s3.3 names, set
        # process-wide rather than in a scope: the diagram's arrays outlive
        # any scope that built them, and the driver reads them afterwards.
        jax.config.update("jax_explicit_x64_dtypes", "allow")
        import jax.numpy as jnp

        j = jnp.asarray([[0.0, 1.0, 0], [0.5, 2.0, 1]], dtype=jnp.float64)
        dj = from_array(j)
        rep.ok(
            "jax", f"diagram.xp = {dj.xp.__name__} (needs a caller-set x64 config, D23)"
        )
    else:
        rep.skip("jax", "jax not installed -- pip install -e '.[jax]'")

    print()
    return rep.summary(strict=args.strict)


# --------------------------------------------------------------------------
# closure -- the empty-default-install claim, proven rather than asserted


def cmd_closure(args: argparse.Namespace) -> int:
    uv = shutil.which("uv")
    venv = scratch(args, "closure")
    py = venv_python(venv)
    pip = py.with_name("pip")
    rep = Report()

    print(f"{BOLD}building a default-closure environment{OFF} {DIM}{venv}{OFF}")
    venv.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(venv, ignore_errors=True)
    try:
        if uv:
            subprocess.run(
                [uv, "venv", str(venv), "--python", args.python, "-q"],
                check=True,
                cwd=REPO,
            )
            subprocess.run(
                [uv, "pip", "install", "-q", "--python", str(py), "."],
                check=True,
                cwd=REPO,
            )
        else:
            subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
            subprocess.run([str(pip), "install", "--quiet", "."], check=True, cwd=REPO)

        # 1. The closure is empty. Not "permissive only": empty.
        listing = subprocess.run(
            [uv, "pip", "list", "--python", str(py)] if uv else [str(pip), "list"],
            capture_output=True,
            text=True,
            cwd=REPO,
        ).stdout
        third_party = [
            ln
            for ln in listing.splitlines()
            if ln
            and not ln.startswith(("Package", "---", "Using"))
            and ln.split()[0] not in {"akriti", "pip", "setuptools", "wheel"}
        ]
        if third_party:
            rep.fail(
                "empty default closure", f"unexpected distributions: {third_party}"
            )
        else:
            rep.ok("empty default closure", "akriti and nothing else")

        # 2. It imports with nothing installed. CI's test job runs this step.
        r = subprocess.run(
            [
                str(py),
                "-c",
                "import akriti, akriti.diagrams; print(akriti.__version__)",
            ],
            capture_output=True,
            text=True,
            cwd=REPO,
        )
        if r.returncode == 0:
            rep.ok("bare import", f"akriti {r.stdout.strip()}")
        else:
            rep.fail("bare import", r.stderr.strip().splitlines()[-1])

        # 3. The lazy numpy boundary reports itself properly rather than
        # crashing. Not a CI check: every CI environment that runs tests has
        # numpy, and the branch in adapters.py is `pragma: no cover`, so a bare
        # venv is the only place it is ever exercised.
        r = subprocess.run(
            [
                str(py),
                "-c",
                textwrap.dedent("""
                from akriti.diagrams import from_ripser
                try:
                    from_ripser([[(0.0, 1.0)]])
                    print("NO-ERROR")
                except ImportError as e:
                    print("ImportError:", e)
            """),
            ],
            capture_output=True,
            text=True,
            cwd=REPO,
        )
        out = r.stdout.strip()
        if out.startswith("ImportError:") and "akriti[numpy]" in out:
            rep.ok("lazy numpy boundary", "names the extra to install")
        else:
            rep.fail("lazy numpy boundary", out or r.stderr.strip())

        # 4. The licence gate CI's closure job runs against exactly this
        # environment.
        r = subprocess.run(
            [str(py), "tools/check_license_closure.py"],
            capture_output=True,
            text=True,
            cwd=REPO,
        )
        last = (r.stdout.strip().splitlines() or ["<no output>"])[-1]
        if r.returncode == 0:
            rep.ok("check_license_closure.py", last)
        else:
            rep.fail("check_license_closure.py", last)
    finally:
        if not args.keep:
            discard(venv)
    return rep.summary(strict=args.strict)


# --------------------------------------------------------------------------
# matrix -- the isolated optional rows, as CI builds them

CI_YML = REPO / ".github" / "workflows" / "ci.yml"

# CI installs torch from PyPI, whose default wheel drags CUDA in (~2.5 GB).
# CPU is what a correctness check needs, an order of magnitude smaller, and
# the one place the driver departs from ci.yml. Two uv traps decide the flags,
# both measured with `uv pip install --dry-run`: uv gives --extra-index-url
# priority over --index-url, and by default stops at the first index that
# carries a package. So with the CPU index primary, PyPI wins and the CUDA
# wheel is back; with it extra, every package the PyTorch index happens to
# mirror is pinned to the copy there -- `packaging<=24.1`, which hatchling
# cannot build under. `unsafe-best-match` considers every index and takes the
# highest version: `2.14.0+cpu` outranks `2.14.0` (a PEP 440 local label sorts
# above none) and PyPI's `packaging` outranks the mirror's. The "unsafe" is
# dependency confusion between indexes, which two trusted ones do not have.
TORCH_CPU_FLAGS = (
    "--extra-index-url",
    "https://download.pytorch.org/whl/cpu",
    "--index-strategy",
    "unsafe-best-match",
)

PASSED = re.compile(r"(\d+) passed")
_MATRIX_EXPR = re.compile(r"\$\{\{\s*matrix\.(\w+)\s*\}\}")
_IF_CLAUSE = re.compile(r"^matrix\.name == '([^']+)'$")


def _yaml_items(block: str, indent: int) -> list[dict[str, str]]:
    """The `- ` items at `indent` in a YAML block, each as a flat mapping.

    Enough YAML for ci.yml's `include:` rows and `steps:` and no more: plain
    `key: value` lines two deeper than the dash, `key: |` block scalars,
    comments and blank lines skipped, anything nested further (`with:`)
    dropped. No YAML library is a dependency here, and a parser this narrow
    fails visibly when the job's shape changes rather than reading it wrong.
    """
    items: list[dict[str, str]] = []
    key_indent = indent + 2
    block_key: str | None = None
    for line in block.splitlines():
        stripped = line.strip()
        lead = len(line) - len(line.lstrip(" "))
        if block_key is not None:
            if not stripped or lead > key_indent:
                items[-1][block_key] += line[key_indent + 2 :] + "\n"
                continue
            block_key = None
        if not stripped or stripped.startswith("#"):
            continue
        if lead == indent and stripped.startswith("- "):
            items.append({})
            stripped, lead = stripped[2:], key_indent
        if lead != key_indent or not items:
            continue
        key, _, value = stripped.partition(":")
        value = value.strip()
        if value == "|":
            block_key, items[-1][key] = key, ""
        else:
            items[-1][key] = value
    return items


def _ci_optional_job() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """ci.yml's `optional` job: its matrix rows and its steps, read live.

    Read rather than copied, so a row or step added to CI is one here the same
    day. Each row carries `name`, `extra`, `marker` and `imports`; each step
    `name`, optionally `if`, and `run` or `uses`.
    """
    text = CI_YML.read_text(encoding="utf-8")
    job = re.search(r"^  optional:\n(.*?)(?=^  [\w-]+:\n|\Z)", text, re.S | re.M)
    if job is None:
        raise ValueError(f"{CI_YML}: no `optional` job")
    parts = re.search(
        r"^        include:\n(.*?)^    steps:\n(.*)", job.group(1), re.S | re.M
    )
    if parts is None:
        raise ValueError(f"{CI_YML}: the optional job has no `include:` and `steps:`")
    rows = _yaml_items(parts.group(1), indent=10)
    steps = _yaml_items(parts.group(2), indent=6)
    columns = {"name", "extra", "marker", "imports"}
    if not rows or any(columns - row.keys() for row in rows):
        raise ValueError(f"{CI_YML}: optional matrix rows parsed wrong: {rows}")
    if not any(step.get("run", "").startswith("pytest") for step in steps):
        raise ValueError(f"{CI_YML}: the optional job has no pytest step: {steps}")
    return rows, steps


def _step_applies(condition: str | None, row: dict[str, str]) -> bool:
    """A step's `if:` for one row -- `matrix.name == '...'`, `||`-joined."""
    if condition is None:
        return True
    names = set()
    for clause in condition.split("||"):
        match = _IF_CLAUSE.match(clause.strip())
        if match is None:
            raise ValueError(f"ci.yml `if:` the driver cannot evaluate: {condition!r}")
        names.add(match.group(1))
    return row["name"] in names


def _step_commands(run: str, row: dict[str, str]) -> list[list[str]]:
    """A step's `run:` as argv lists, `${{ matrix.* }}` substituted from the row."""
    text = _MATRIX_EXPR.sub(lambda m: row[m.group(1)], run)
    text = re.sub(r"\\\n\s*", " ", text)  # shell line continuations
    lines = [
        ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")
    ]
    return [shlex.split(ln) for ln in lines]


def _mirror(argv: list[str], *, py: Path, uv: str, torch_row: bool) -> list[str]:
    """A CI command as the row's venv runs it: pip -> uv pip, python -> the venv's."""
    head, rest = argv[0], argv[1:]
    if head == "pip" and rest[:1] == ["install"]:
        cmd = [uv, "pip", "install", "-q", "--python", str(py), *rest[1:]]
        return [*cmd, *TORCH_CPU_FLAGS] if torch_row else cmd
    if head == "python":
        return [str(py), *rest]
    if head == "pytest":
        return [str(py), "-m", "pytest", "-q", "-rs", *rest]
    raise ValueError(f"ci.yml step the driver cannot mirror: {shlex.join(argv)}")


def _run_row(
    row: dict[str, str], steps: list[dict[str, str]], *, py: Path, uv: str, rep: Report
) -> None:
    name = row["name"]
    torch_row = "torch" in row["extra"].split(",")
    tail = ""
    for step in steps:
        # `uses:` steps are checkout and setup-python; the venv stands in.
        if "run" not in step or not _step_applies(step.get("if"), row):
            continue
        for argv in _step_commands(step["run"], row):
            label = step.get("name") or shlex.join(argv)
            r = subprocess.run(
                _mirror(argv, py=py, uv=uv, torch_row=torch_row),
                cwd=REPO,
                capture_output=True,
                text=True,
            )
            last_out = (r.stdout.strip().splitlines() or ["<no output>"])[-1]
            if r.returncode != 0:
                last_err = (r.stderr.strip().splitlines() or [last_out])[-1]
                rep.fail(f"row {name}", f"{label}: {last_err}")
                return
            if argv[0] == "pytest":
                # pytest exits 0 when every selected test skipped. CI's import
                # step is what stops that being a green row that ran nothing;
                # the driver replays that step and checks the count as well.
                passed = PASSED.search(last_out)
                if passed is None or int(passed.group(1)) == 0:
                    rep.fail(f"row {name}", f"nothing passed: {last_out}")
                    return
                tail = last_out
    rep.ok(f"row {name}", tail)


def cmd_matrix(args: argparse.Namespace) -> int:
    uv = shutil.which("uv")
    if not uv:
        print(f"{RED}matrix needs uv on PATH (it builds one venv per row).{OFF}")
        return 1
    rep = Report()
    rows, steps = _ci_optional_job()
    by_name = {row["name"]: row for row in rows}
    wanted = args.rows.split(",") if args.rows else list(by_name)
    base = scratch(args, "matrix")
    base.mkdir(parents=True, exist_ok=True)

    try:
        for name in wanted:
            row = by_name.get(name)
            if row is None:
                rep.fail(
                    name, f"not a row in ci.yml's optional matrix: {','.join(by_name)}"
                )
                continue
            venv = base / name
            print(
                f"{BOLD}row {name}{OFF} "
                f"{DIM}.[test,{row['extra']}] -m {row['marker']}{OFF}"
            )
            shutil.rmtree(venv, ignore_errors=True)
            try:
                subprocess.run(
                    [uv, "venv", str(venv), "--python", args.python, "-q"],
                    check=True,
                    cwd=REPO,
                )
                _run_row(row, steps, py=venv_python(venv), uv=uv, rep=rep)
            finally:
                if not args.keep:
                    shutil.rmtree(venv, ignore_errors=True)
    finally:
        if not args.keep:
            discard(base)
    return rep.summary(strict=args.strict)


# --------------------------------------------------------------------------
# exec -- call one function without writing a preamble


def _preamble() -> dict:
    """`np`, every name in `akriti.diagrams.__all__`, and its three modules.

    Read from `__all__` rather than listed here, so a name added to the public
    surface is in scope the same day.
    """
    import numpy as np

    from akriti import diagrams
    from akriti.diagrams import adapters, core, io

    ns: dict = {"np": np, "core": core, "adapters": adapters, "io": io}
    ns.update({name: getattr(diagrams, name) for name in diagrams.__all__})
    return ns


def cmd_exec(args: argparse.Namespace) -> int:
    """Most PRs here touch one function in diagrams/. This is how you call it."""
    src = args.code
    # os.path.isfile, not Path.is_file: before 3.13 the latter raises
    # ENAMETOOLONG on a snippet longer than one path component allows.
    body = Path(src).read_text() if os.path.isfile(src) else src
    try:
        ns = _preamble()
    except ImportError as exc:
        print(f"{RED}the preamble needs numpy: {exc}{OFF}")
        print("Install the test extra -- see SKILL.md's Setup.")
        return 1
    try:
        try:
            # An expression gets its value printed; a statement block does not.
            value = eval(compile(body, "<exec>", "eval"), ns)
            if value is not None:
                print(repr(value))
        except SyntaxError:
            exec(compile(body, "<exec>", "exec"), ns)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


# --------------------------------------------------------------------------


def cmd_all(args: argparse.Namespace) -> int:
    codes = []
    for name, fn in (
        ("probe", cmd_probe),
        ("smoke", cmd_smoke),
        ("closure", cmd_closure),
    ):
        print(f"\n{BOLD}=== {name} ==={OFF}")
        codes.append(fn(args))
    return max(codes)


def main() -> int:
    # The global flags are attached to every subparser as well as the root, so
    # both orders work. Left root-only, `driver.py closure --workdir X` is an
    # argparse error rather than a run, which is the first thing anyone types.
    # The defaults live on the namespace rather than on the arguments: a
    # subparser copies its own defaults over whatever the root already parsed,
    # so with ordinary defaults `driver.py --strict smoke` silently dropped the
    # flag. SUPPRESS never sets an attribute, so the pre-filled namespace
    # survives and a flag in either position wins.
    common = argparse.ArgumentParser(add_help=False, argument_default=argparse.SUPPRESS)
    common.add_argument(
        "--strict",
        action="store_true",
        help="an unexercised path is a failure (probe_backends' "
        "--require-giotto stance)",
    )
    common.add_argument(
        "--keep",
        action="store_true",
        help="keep the venvs and temp files the run builds",
    )
    common.add_argument(
        "--workdir",
        help=f"parent for what the driver builds; it creates and deletes only "
        f"<workdir>/{SCRATCH}/<mode> (default: the repo root, where "
        f".gitignore's .venv*/ covers it)",
    )
    common.add_argument(
        "--python",
        help="interpreter for venvs the driver builds (default 3.12, CI's "
        "optional rows)",
    )

    p = argparse.ArgumentParser(
        description=__doc__,
        parents=[common],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser(
        "probe",
        parents=[common],
        help="what is installed, and what that makes runnable",
    )
    sub.add_parser(
        "smoke", parents=[common], help="the real API end to end, on what is installed"
    )
    sub.add_parser(
        "closure", parents=[common], help="prove the empty default install closure"
    )
    m = sub.add_parser(
        "matrix", parents=[common], help="build the isolated optional rows, as CI does"
    )
    m.add_argument(
        "--rows", help="comma-separated row names from ci.yml's optional matrix"
    )
    e = sub.add_parser(
        "exec", parents=[common], help="run a snippet with the API pre-imported"
    )
    e.add_argument("code", help="a Python expression, statements, or a .py path")
    sub.add_parser("all", parents=[common], help="probe + smoke + closure")

    args = p.parse_args(
        namespace=argparse.Namespace(
            strict=False, keep=False, workdir=None, python="3.12"
        )
    )
    return {
        "probe": cmd_probe,
        "smoke": cmd_smoke,
        "closure": cmd_closure,
        "matrix": cmd_matrix,
        "exec": cmd_exec,
        "all": cmd_all,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
