---
name: run-akriti
description: Build, run, and drive akriti — the TDA library. Use when asked to run akriti, start or smoke-test it, build it, install its optional backends, run its tests or CI matrix, check the dependency closure, exercise a diagrams/ function, or confirm a change works in the real library rather than only in tests.
---

akriti is a Python library, so "running the app" means importing it and calling
it. Drive it with **`.claude/skills/run-akriti/driver.py`** — a probe, an
end-to-end smoke run, a closure check, a CI-matrix runner, and a one-liner
`exec` for poking a single function. All paths below are relative to the repo
root.

The thing that makes this non-trivial is the **environment matrix**. The
default install closure is empty — `pip install akriti` fetches nothing, not
even numpy — and every backend and array namespace is an opt-in extra. The test
suite marks each one, so it *skips* where the extra is absent. A green `pytest`
therefore proves nothing about the rows nobody installed. The driver reports
what it exercised **and what it could not, with the reason**; `--strict` makes
the second list a non-zero exit.

## Prerequisites

No system packages: every extra installs from wheels.

`uv` on PATH for `closure` and `matrix`, which build venvs (`closure` falls
back to `python -m venv` without it). A uv-made venv **has no pip**:
`.venv/bin/python -m pip` gives "No module named pip". Use
`uv pip --python .venv/bin/python ...`.

## Setup

```bash
uv venv .venv --python 3.14
uv pip install --python .venv/bin/python -e ".[dev]"
```

3.14 rather than the 3.10 floor because the `.akd` zstd path exists only
there — see Gotchas. CI's optional rows run 3.12, which is what the driver's
own venvs default to.

`dev` resolves to `test,lint,rips,alpha,distances`. It deliberately excludes
torch and jax — see Gotchas. To add them:

```bash
# torch and jax are large; keep them out of the shared dev env unless needed.
uv pip install --python .venv/bin/python -e ".[test,torch,jax,parquet,bio]"
```

There is no build step. It is a pure-Python hatchling package installed
editable.

## Run (agent path)

```bash
# what is installed, and what that makes runnable
python .claude/skills/run-akriti/driver.py probe

# the real API end to end on whatever is installed
python .claude/skills/run-akriti/driver.py smoke

# probe + smoke + closure
python .claude/skills/run-akriti/driver.py all
```

`smoke` runs a noisy circle point cloud through Ripser and GUDHI into
`from_ripser` / `from_gudhi`, plus `from_array`, `from_persim`, and
`from_giotto` against the committed fixture; then the diagram surface
(`finite`, `content_hash`, both `finitize` modes, `canonical`, `to_arrays`), a
`DiagramBatch`, `to_csv` / `to_parquet`, an `.akd` save/load round-trip checked
for equality **and** hash identity, and each array namespace present. The
summary line counts exercised, not exercised and failed.

Add `--strict` to turn any unexercised path into exit 1 — the stance
`rfcs/evidence/probe_backends.py --require-giotto` takes. It is the mode for
an all-extras environment; in `dev` it exits 1 on torch, jax and parquet.
`probe` ignores it: probe measures and exercises nothing.

### Direct invocation — most PRs need only this

Most PRs here touch one function in `src/akriti/diagrams/`. To call it without
writing an import preamble:

```bash
# an expression: its value is printed
python .claude/skills/run-akriti/driver.py exec \
  'from_array(np.array([[0.,1.,0],[0.,np.inf,0]])).content_hash'

# statements, reaching a private helper
python .claude/skills/run-akriti/driver.py exec '
d = from_array(np.array([[0.,1.,0],[0.5,2.,1]]))
print(adapters._as_degree(2), d.dim(1).persistence)
'

# or a file
python .claude/skills/run-akriti/driver.py exec /path/to/snippet.py
```

`np`, every public name from `akriti.diagrams`, and the `core`, `adapters` and
`io` modules are pre-imported.

### Proving the empty default closure

```bash
python .claude/skills/run-akriti/driver.py closure
```

Builds a throwaway venv holding **only** `akriti`, then checks four things: the
closure is empty, by `pip list` and by `tools/check_license_closure.py` (what
CI's `closure` job runs); `import akriti, akriti.diagrams` works on it (CI's
`test` job); and the lazy numpy boundary raises an `ImportError` that *names*
`akriti[numpy]`. The last is not a CI check — every CI environment that runs
tests has numpy, and the branch is `pragma: no cover` — so a bare venv is the
only place it is ever exercised.

### The CI optional matrix, locally

```bash
python .claude/skills/run-akriti/driver.py matrix                  # every row ci.yml declares
python .claude/skills/run-akriti/driver.py matrix --rows rips,jax  # some, by name
```

The rows and the steps are read from `.github/workflows/ci.yml`'s `optional`
job at run time, not copied into the driver, so a row or step added to CI is
one here. One isolated venv per row; each `run:` step is replayed in it with
`pip` → `uv pip`, `python` → the row's interpreter and `pytest` → `python -m
pytest`, and the venv stands in for the `uses:` steps (checkout,
setup-python). Two deliberate departures from CI: the resolver is uv, and the
torch row adds the PyTorch CPU wheel index (CI's PyPI torch drags CUDA in).
A row whose marked tests all skipped fails rather than passing — pytest exits
0 on that; CI catches it with the import step, and the driver replays that
step and checks the count as well.

Global flags (`--strict --keep --workdir DIR --python X.Y`) work before or
after the subcommand. Everything the driver builds goes under
`<workdir>/.venv-driver/<mode>`, and that child is the only path it deletes;
the default workdir is the repo root, where `.gitignore`'s `.venv*/` already
covers it. Venvs are removed after each row unless you pass `--keep`.

## Run (human path)

There is no application to launch — no CLI entry point, no server, no GUI. The
human path is a REPL:

```bash
python -c "
import numpy as np, ripser
from akriti.diagrams import from_ripser
X = np.random.default_rng(0).standard_normal((40, 2))
print(from_ripser(ripser.ripser(X, maxdim=1)))
"
```

## Test

```bash
pytest -m "not backend"    # the fast gate; what CI's test job runs
pytest                     # everything the environment supports
pytest -m rips             # one marker: rips alpha distances cross_backend
                           #             torch jax parquet backend
```

Markers are declared in `pyproject.toml` under `[tool.pytest.ini_options]`.

Lint, exactly as CI runs it:

```bash
ruff check . && ruff format --check . && mypy
```

**`ruff` covers `.claude/` too** — the driver is part of the lint surface, so
keep it formatted or CI breaks. `mypy` is scoped to `files = ["src/akriti"]`
and does not see the driver.

Reproduce the RFC-0001 appendix measurements:

```bash
python rfcs/evidence/probe_backends.py
python rfcs/evidence/probe_backends.py --require-giotto   # exit 1 wherever giotto is absent
```

## Gotchas

- **`from_array` on an `(n, 3)` array reads `(birth, death, dim)`** —
  giotto's order, deliberately, not Ripser's `(dim, birth, death)`. Passing
  ripser's order raises *"the degree column contains a non-finite value ...
  (I3)"*, which names the column you did not think you were writing to. Pass
  `columns=("dim","birth","death")` to be explicit.
- **Properties, not methods**: `content_hash`, `finite`, `essential`,
  `persistence`, `n_bars`, `bar_counts`, `xp`. `d.content_hash()` gives
  `TypeError: 'str' object is not callable`. `canonical()`, `dim(k)`,
  `finitize()`, `allclose()` *are* methods.
- **The finitize mode is `at="max_finite_death"`**, not `"max_finite"`. The
  other two forms are `at="drop"` and a float.
- **`filterwarnings = ["error"]` is set for pytest.** `to_csv`, `to_arrays` and
  `to_parquet` all emit a `UserWarning` by design ("export discards all
  DiagramMeta"), and `from_giotto` warns about giotto's `birth == death`
  padding rows unless you pass `strip_padding=` explicitly. A new test calling
  any of these fails unless it expects the warning. The driver passes
  `strip_padding=False` and reports the trivial-row count instead.
- **Never add numpy to `dependencies`** to fix an ImportError. The empty
  closure is load-bearing and enforced by `tools/check_license_closure.py` in a
  clean venv. The lazy boundary is designed: it raises an `ImportError` naming
  `akriti[numpy]`.
- **JAX needs a caller-set 64-bit flag (D23).** A default JAX install truncates
  to float32 and `from_array` raises *"births must be float64 (I2)"*. The
  caller sets it; akriti MUST NOT, and a test greps `src/` for either spelling.
  The driver sets `jax_explicit_x64_dtypes='allow'`, the narrow lever §3.3
  names, because the driver is a caller.
- **giotto-tda is AGPLv3 and must never enter a user-facing extra.**
  `from_giotto` is exercised against `tests/fixtures/giotto_output.json`,
  captured from a real giotto run. Only CI's `rfc-evidence` job installs
  giotto, with `--no-deps`. Do not read giotto source.
- **`repro/`, `akriti.core` and `akriti.castle` do not exist yet**; only
  `diagrams/` is implemented. `probe` prints all three so you do not go
  looking. Do not create any of them to satisfy a line in CLAUDE.md: each has
  an issue or PR that owns it, and what state those are in lives on GitHub,
  not here.
- **`ZIP_ZSTANDARD` exists only on Python 3.14+.** The `.akd` zstd branch
  silently never runs on 3.12, where one io test skips. The driver's venvs
  default to `--python 3.12`; pass `--python 3.14` to reach that path.
- **CI isolates the optional rows on purpose.** An all-extras environment
  passing proves something different from seven isolated rows — it cannot
  catch a module that only imports because some *other* extra happened to pull
  its dependency.
- **`rfcs/evidence/bar_counts.py` does not run from this repo.** It imports
  `lib.datasets` from the separate `classify` repo; its own docstring says so.
  `pypi_downloads.py` needs `pypistats.org`. Every other evidence script runs
  offline.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `.venv/bin/python: No module named pip` | The venv is uv-managed. Use `uv pip install --python .venv/bin/python ...`. |
| `Disk quota exceeded (os error 122)` installing torch | The workdir is on a small filesystem. Pass `--workdir` on a larger one; the driver creates and deletes only `<workdir>/.venv-driver/`. |
| `AttributeError: module 'torch' has no attribute '__version__'` | A part-finished install left a broken package that still imports. Delete the venv and reinstall. `probe` reports this as a failure rather than as "not installed". |
| `ValueError: the degree column contains a non-finite value ... (I3)` | Column order. `from_array` wants `(birth, death, dim)`. |
| `TypeError: 'str' object is not callable` on `content_hash` | It is a property. Drop the `()`. |
| `ValueError: unknown finitize mode: 'max_finite'` | It is `"max_finite_death"`. |
| `ValueError: births must be float64 (I2); got float32` | JAX without the x64 flag. See the D23 gotcha. |
| `ProbeDriftError: A.1 drift: giotto-tda stopped being importable` | Expected without giotto. Drop `--require-giotto`, or install the pinned giotto environment from CI's `rfc-evidence` job. |
| `ModuleNotFoundError: No module named 'lib'` from `bar_counts.py` | That script only runs from the `classify` repo. Not a bug. |
