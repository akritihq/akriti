# Adapter/RFC-0001 completion plan

## Goal

Bring the `adapters` branch into conformance with
`origin/rfc-0001-persistence-diagram-interchange-draft-revisions`, close the
adapter and constructor edge cases found in review, and prove the optional
backend paths independently in CI.

## Constraints and decisions

- Preserve the user's existing dirty files (`CLAUDE.md`, `TODO.md`,
  `AGENTS.md`, `tasks/progress.md`, `tasks/questions.md`, and `uv.lock`).
- Synchronize the two RFC-0001 documents from the named branch before using
  their requirements as the local source of truth.
- Correct the RFC's GUDHI extended-persistence shape from a 4-tuple to the
  observed/documented list of four lists.
- Keep the default dependency closure empty. Add `array-api-compat` only to
  `akriti[torch]` and `pyarrow` only to a new `akriti[parquet]` extra, both
  lazily imported and license-audited.
- Preserve public-constructor copying while retaining the explicitly internal
  zero-copy paths used by batch slicing.
- Write the floating-point clamp regression test and its implementation in
  separate agent sessions, per the project's numerical-code rule.

## Work plan

1. **Specification baseline**
   - Bring the named RFC revision and history into this branch.
   - Patch the GUDHI list-of-four wording and record the correction in history.
   - Confirm no unrelated working-tree file is overwritten.

2. **Core contracts (test first)**
   - Add failures for public `PersistenceDiagram`/`DiagramBatch` buffer
     aliasing and caller-owned `metas` mutation.
   - Add namespace-resolution tests for native-first behavior, missing-extra
     diagnostics, torch fallback, and accessor parity.
   - Add metadata tests for `description`, description-insensitive
     `same_provenance`, and `coeff_field_source` coherence.
   - Implement one shared `namespace_of()` resolver, constructor copying with
     internal unchecked/view paths preserved, and the revised metadata rules.

3. **Import adapters (test first)**
   - Cover `from_array(columns=...)`: case-insensitive reordering, length,
     duplicate/missing/unknown names, `diagram_id`, and `dim=` conflicts.
   - Cover adapter-owned Ripser filtration, explicit backend versions,
     `source_dtype`, row order/multiplicity, and empty/essential forms.
   - Cover real GUDHI list-of-four rejection, reject unlisted GUDHI `(n,3)`,
     and standardize degree-conflict exceptions to `TypeError`.
   - Implement the smallest matching adapter changes.

4. **Floating-point clamp (separate authorship)**
   - Add a failing large-magnitude/multi-ULP inversion regression test without
     changing production code.
   - In a separate session, replace the broad relative tolerance with a
     representational-noise/ULP-bounded rule and keep warning/provenance
     behavior intact.

5. **Interoperability exporters (test first)**
   - Test `to_arrays`, `to_csv`, and `to_parquet` for diagram and batch shapes,
     row order, duplicates, empty inputs, essential `inf`, warnings, headers,
     dtypes, `diagram_id`, metadata loss, lazy imports, and actionable extras.
   - Implement/export the three functions in `adapters.py`.
   - Add the `parquet` extra and dependency-policy documentation.

6. **Packaging and CI**
   - Add the verified `array-api-compat` floor to `akriti[torch]` and document
     it in `DEPENDENCIES.md`.
   - Split optional-backend CI so one install failure cannot suppress every
     live suite; add independently gated torch and parquet coverage.
   - Enforce permissive closure checks for `torch` and `parquet` where their
     upstream licenses permit it, and keep backend/copyleft audits explicit.

7. **Proof and review**
   - Run focused red/green tests after each task.
   - Run the complete pytest suite, Ruff check/format, mypy, build metadata,
     and dependency-closure checks available locally.
   - Inspect the final diff for scope, have an independent reviewer check RFC
     conformance and regression risk, and update this plan with evidence.

## Status

- [x] Audit and baseline verification: 202 focused tests and 299 full tests
  passed before changes; Ruff and mypy were clean.
- [x] User approved fixing all audited gaps.
- [x] Specification baseline — merged the named revision; corrected GUDHI's
  outer return container to a four-element list, made the row-sequence NumPy
  fallback explicit, and corrected `to_arrays()`'s outer-container claim in
  the RFC and history.
- [x] Core contracts — 114 focused tests, Ruff, and mypy pass; independent
  specification and quality review found no remaining Critical/Important issue.
- [x] Import adapters — 234 focused tests, Ruff, and mypy pass; the independent
  review's three boundary findings were corrected and the re-review verdict is
  Ready.
- [x] Floating-point clamp — separate test author reproduced both laundering
  paths. A separate implementation author now uses eight local downward
  float64 ULPs, with no absolute or broad relative floor. The three-scale
  regression plus strict-error extreme/zero/subnormal tests, 15 focused clamp
  tests, 240 adapter tests, Ruff, and mypy pass; independent re-review verdict
  is Ready.
- [x] Interoperability exporters — `to_arrays`, `to_csv`, and `to_parquet`
  preserve rows, multiplicity, infinities, signed zero, explicit schemas, and
  one loss warning per successful call. Live PyArrow 25.0.0 and PEP 440 floor
  cases pass; the independent exporter re-review verdict is Ready.
- [x] Packaging and CI — `array-api-compat>=1.15.0` is confined to
  `akriti[torch]`, `pyarrow>=25.0.0` to `akriti[parquet]`, and the default
  closure remains empty. Six isolated optional rows use precise markers and
  fatal import preflights; Parquet's clean closure is strictly Apache-2.0 and
  Torch's large closure is report-only. Documentation and the corrected RFC
  signature/history agree.
- [x] Final proof and review — late holistic review findings were closed for
  NumPy's declared version floor, mode-specific `finitize` validation,
  recursively owned/immutable JSON metadata, optional-row false greens, and
  stale RFC notes. Fresh evidence: 471 full tests; 450 backend-free tests;
  isolated rows rips 4, alpha 5, distances 3, cross-backend 4, torch 2, and
  parquet 3; Ruff check/format, mypy, and `git diff --check` pass; wheel and
  sdist build and pass Twine; the clean PyArrow 25.0.0 closure is
  permissive-only. Final independent verdict: Ready, with no remaining
  Critical or Important findings.

## Extension — RFC §10 / §11.2 `.akd` serialization

### Goal

Complete the serialization dependency that RFC §11.2 makes part of adapter
acceptance: deterministic `.akd` save/load for diagrams and ragged batches,
including exact metadata round-trips and strict format validation.

### Scope and decisions

- The reconciled working-tree RFC is normative. The divergent remote RFC
  branches are not merged into this dirty worktree.
- Add no dependency. NumPy remains lazy and function-scoped behind the already
  approved `akriti[io]` extra at its declared `numpy>=2.0` floor.
- Spell the lazy import as `importlib.import_module("numpy")`, matching the
  existing optional-boundary policy in `core.py`/`adapters.py` so mypy never
  resolves NumPy stubs into the default dependency-free check.
- Put serialization in `src/akriti/diagrams/io.py`; adapters remain focused on
  interchange imports and exports.
- Write and observe failing tests before production code. Serialization is not
  numerical code, so the separate numerical-author rule does not apply.
- Keep the guarded bottleneck wrapper, metadata-validation defect, `allclose`
  tolerance domain, and `finitize` identity tension out of this change; they
  are recorded in `tasks/questions.md`.

### Serialization design

- `save(obj, path)` accepts a `PersistenceDiagram` or `DiagramBatch` from any
  supported array namespace; `load(path)` returns a NumPy-backed object of the
  explicit saved `kind`.
- The outer ZIP contains exactly `meta.json`, then `bars.npz`. Both that ZIP
  and the nested NPZ use `ZIP_STORED` entries with timestamp
  `(1980, 1, 1, 0, 0, 0)`, Unix creator `3`, permissions `0o600 << 16`, zero
  flags, and empty entry/archive extras and comments. Their member order is
  fixed too, so determinism is deliberate rather than an incidental NumPy/ZIP
  default.
- Diagrams are canonicalized before writing. Batch diagram order and metadata
  order stay fixed while each ragged segment is canonicalized independently.
- Required NPY arrays use fixed little-endian on-disk dtypes (`<f8`, `<i4`,
  `<i8`) so deterministic bytes do not depend on the writer's CPU. Loading
  accepts either endian for the required widths and converts to native NumPy
  dtypes before public construction.
- Serialized coordinate buffers normalize negative zero to positive zero,
  because exact diagram equality treats their signs as equal and therefore
  identical diagrams must not produce different bytes. Inputs are never
  mutated.
- `meta.json` uses the RFC-pinned compact, sorted UTF-8 JSON encoding and
  carries all seven `DiagramMeta` fields. `bars.npz` carries births, deaths,
  dims, and batch offsets where applicable.
- The outer two-member archive is closed. Within a supported format version,
  unknown JSON envelope fields and unknown NPZ arrays are ignored as advisory
  additions; all required fields/arrays and every `kind` consistency rule are
  still validated. A `meta`/`metas` member itself remains the closed
  `DiagramMeta` dataclass schema: every item must be an object and unknown
  nested field names are malformed rather than advisory envelope keys.
- Loading routes through public constructors, never `_unchecked`, so I1–I9
  and B1–B8 are revalidated at the untrusted-file boundary.
- Before NumPy allocates a required array, loading validates unique logical
  NPZ member names and checks that each NPY header's declared shape/dtype has
  exactly the enclosing member's uncompressed byte length. Ambiguous JSON
  duplicate keys and malformed ZIP/NPY failures are normalized to
  `ValueError`.
- Missing or pre-2.0 NumPy raises actionable `ImportError` naming
  `akriti[io]`; malformed or unsupported archives raise `ValueError`.

### Work plan

8. **Record scope boundary**
   - Add the four non-adapter findings to `tasks/questions.md`.
   - Correct the stale `from_giotto` fixture description.
9. **Serialization contract tests — red first**
   - Add exact diagram and batch round-trips, metadata preservation,
     deterministic bytes, canonical bar order, and signed-zero/multiplicity
     cases.
   - Add archive/schema validation, `kind` dispatch, malformed-file cases,
     empty/ragged batch coverage, lazy-import/version diagnostics, and the
     required Hypothesis round-trip property.
10. **Minimal implementation**
    - Implement deterministic `meta.json` and nested `bars.npz` payloads.
    - Implement strict `load` validation and NumPy-backed reconstruction.
11. **Public surface and documentation**
    - Export `save` and `load` from `akriti.diagrams`.
    - Reconcile package/dependency documentation that still calls `.akd`
      planned, without changing the dependency closure.
12. **Proof and review**
    - Run focused red/green tests, the full suite, Ruff, mypy, build checks,
      and `git diff --check`.
    - Obtain independent specification and code-quality reviews; close every
      Important finding before completion.

### Extension status

- [x] User approved completing full RFC §11.2 adapter acceptance.
- [x] Non-adapter concerns recorded separately; stale Giotto documentation
  corrected and verified.
- [x] Exact archive, determinism, validation, namespace-boundary, and test
  contracts derived from RFC §§10.1, 10.2, and 11.2.
- [x] Serialization tests written, independently reviewed, and observed failing
  for the missing API without collection or setup errors.
- [x] `.akd` implementation and public exports complete.
- [x] Full verification and independent review complete.

### Errors encountered

| Error | Attempt | Resolution |
|---|---:|---|
| A focused Giotto pytest selector matched no tests (exit 5) | 1 | Located the assertion in `test_rfc0001_adapters.py`; the corrected exact test passed |
| `.venv/bin/python -m build` lacked the optional `build` frontend | 1 | Used the installed `uv build --offline` with a writable temporary cache; sdist and wheel both built |
