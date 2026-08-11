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
