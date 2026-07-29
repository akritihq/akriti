# Akriti — working conventions

Read automatically at the start of every session. Applies to all developers.
Keep it short and keep it current: when a convention changes, change it here in
the same commit.

## Position

Akriti is the entry point to TDA in Python. We **delegate** computation to
GUDHI / Ripser / persim and **own** the statistical inference layer. The model is
`scanpy`, not `giotto-tda` — own the surface, delegate the depths, implement only
what nobody else offers.

## Hard rules

- **NEVER reimplement persistence computation or bottleneck / Wasserstein
  distance.** Delegate. If a delegation looks inadequate, raise it — do not write
  our own.
- **NEVER add a dependency without asking.** The default install closure must
  stay permissive-only. Read `DEPENDENCIES.md` first; the reasons there are
  measured, not stylistic. Verify a package exists and is the one you mean before
  it enters `pyproject.toml` — models hallucinate package names and typosquatting
  against hallucinated names is a real attack.
- **NEVER read or reproduce giotto-tda source.** It is AGPLv3. The compat shim is
  clean-room from public API documentation only. This applies to prompting a
  model, not just to reading with your own eyes.
- **Numerical code in `core/` and `castle/` must trace to a specific equation in
  Papers I–IV.** Cite it in the docstring. Do not derive formulas.
- **Do not write a numerical function and its test in the same session.** A test
  written by whoever just wrote the function blesses the function's bugs.

## AI assistance — three tiers

**Free use.** Adapters, IO, packaging, CI, test scaffolding, type hints, error
messages, examples, first-draft docs, refactoring.

**Draft, then verify.** Anything numerical. Every formula traced by hand to its
equation in the papers.

**Human-derived.** The correctness-critical statistical core: null calibration,
certificate radii, selectors and WSI diagnostics, the sample-size formula. Not
because a model can't produce it, but because someone has to defend it to a
referee. A subtly wrong test that looks fine and runs clean is precisely the
failure mode we sell protection against.

### Four traps specific to this codebase

Left alone, a model will cheerfully:

1. Reimplement bottleneck distance instead of delegating.
2. Add `scipy` or `torch` to fix a small problem.
3. Invent a package name that doesn't exist.
4. Produce a statistically plausible test with the wrong null.

The first three are caught by review. The fourth is why the human-derived tier
exists.

## Style

- `core/` is written against the **Python array API standard**
  (`__array_namespace__`), not NumPy directly. The same code then runs on NumPy,
  PyTorch, or JAX later with no rewrite.
- **Every public function takes a leading batch dimension.** Not a Python loop
  over diagrams. This is giotto-tda's biggest UX weakness and it is very hard to
  remove later.
- **Every docstring states its assumptions.** A domain scientist and their agent
  both read these.
- PyTorch is never a hard dependency. It lives behind `akriti[torch]`.

## Specifications

Implementation follows the RFCs in `rfcs/`. `akriti.diagrams` is specified by
**RFC-0001**; read it before touching `diagrams/`. If the code and the RFC
disagree, one of them is a bug — say which.

Three consequences of RFC-0001 that are easy to get wrong:

- Essential bars are stored as `inf`. Never a sentinel, never dropped.
- Diagram batches are **ragged**. Never a dense padded array.
- Diagrams from different backends are never exactly equal — Ripser computes in
  single precision. Exact and approximate equality are separate methods.

## Verification

- **Acceptance = `repro/` reproduces the Paper IV tables.** If the tables
  reproduce, the code is right regardless of who or what typed it.
- **Property-based tests** for numerical code — stability bounds and
  invariances. They catch what example-based tests miss.
- Every tutorial executes in CI. A broken tutorial breaks the build. Stale
  documentation is how giotto-tda's tutorials became worthless.
- Optional backends are absent from the default test environment by design. Mark
  tests that need one with `@pytest.mark.backend`.

## Scope

Not in the first six months: GPU batching, PyTorch Geometric integration, a
Streamlit playground, Discord, an issue-response SLA, Mapper (patent-encumbered),
and anything resembling persistence computation.

Scope discipline is not pessimism. giotto-tda had a company behind it and still
went to zero.
