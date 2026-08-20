# Akriti — working conventions

Applies to all developers. When a convention changes, change it here in the same
commit.

## Position

Akriti is the entry point to TDA in Python. We **delegate** computation to
GUDHI / Ripser / persim and **own** the statistical inference layer. The model
is `scanpy`, not `giotto-tda` — own the surface, delegate the depths, implement
only what nobody else offers.

## Hard rules

- **NEVER reimplement persistence computation or bottleneck / Wasserstein
  distance.** Delegate. If a delegation looks inadequate, raise it.
- **NEVER add a dependency without asking.** Read `DEPENDENCIES.md` first — its
  reasons are measured, not stylistic. The default install closure stays
  permissive-only. Confirm a package exists and is the one you mean before it
  enters `pyproject.toml`; hallucinated names are a typosquatting target.
- **NEVER read or reproduce giotto-tda source** (AGPLv3), including via a model
  prompt. The compat shim is clean-room from public API docs only.
- **Numerical code in `core/` and `castle/` must cite a specific equation in
  Papers I–IV** in its docstring. Do not derive formulas.
- **Never write a numerical function and its test in the same session.** A test
  written by whoever just wrote the function blesses that function's bugs.

## AI assistance — three tiers

- **Free use.** Adapters, IO, packaging, CI, test scaffolding, type hints, error
  messages, examples, first-draft docs, refactoring.
- **Draft, then verify.** Anything numerical. Trace every formula by hand to its
  equation in the papers.
- **Human-derived.** Null calibration, certificate radii, selectors and WSI
  diagnostics, the sample-size formula. Someone has to defend these to a
  referee, and a statistically plausible test with the wrong null looks fine and
  runs clean — that failure mode is what we sell protection against.

## Style

- `core/` is written against the **Python array API standard**
  (`__array_namespace__`), not NumPy directly, so the same code runs on PyTorch
  or JAX later with no rewrite.
- **Every public function takes a leading batch dimension** — not a Python loop
  over diagrams. This is giotto-tda's biggest UX weakness and very hard to
  remove later.
- **Every docstring states its assumptions.** A domain scientist and their agent
  both read these.
- PyTorch is never a hard dependency; it lives behind `akriti[torch]`.

## Specifications

Implementation follows the RFCs in `rfcs/`. `akriti.diagrams` is specified by
**RFC-0001**; read it before touching `diagrams/`. If the code and the RFC
disagree, one of them is a bug — say which.

**Before reviewing an RFC or a change to one, read `REVIEWING.md`** -- what RFC review passes turned up that generalise.

Three consequences of RFC-0001 that are easy to get wrong:

- Essential bars are stored as `inf`. Never a sentinel, never dropped.
- Diagram batches are **ragged**. Never a dense padded array.
- Backends never agree exactly (Ripser computes in single precision). Exact and
  approximate equality are separate methods.

## Verification

- **Acceptance = `repro/` reproduces the Paper IV tables.** If the tables
  reproduce, the code is right regardless of who or what typed it.
- **Property-based tests** for numerical code — stability bounds and
  invariances. They catch what example-based tests miss.
- Every tutorial executes in CI. A broken tutorial breaks the build.
- Optional backends are absent from the default test environment by design. Mark
  tests that need one with `@pytest.mark.backend`.

## Scope

Not in the first six months: GPU batching, PyTorch Geometric integration, a
Streamlit playground, Discord, an issue-response SLA, Mapper
(patent-encumbered), and anything resembling persistence computation. Scope
discipline here is deliberate, not pessimism.
