# RFC-0001 review execution -- progress log

Live state for the work described in `rfc-0001-review-plan.md`. Written so the
pass can be resumed from a cold start: every row says what landed, on which
branch, and what proves it.

Started 2026-08-23. Base: `main` at `149e958`.

## Phase status

All eight phases executed 2026-08-23 on `main`, one branch per phase, merged
in the plan's order.

| # | Phase | Branch | State |
|---|---|---|---|
| 1 | Measurements | `rfc/0001/evidence-{jax-x64,giotto-h0-scope}` | done; A.1's environment settled by archaeology, no recapture needed |
| 2 | Tier 1 corrections | `rfc/0001/tier1-corrections` | done, six findings |
| 3 | `F25` in code | `diagrams/meta-propagation` | done, landed first |
| 4 | Tier 2 decisions | `rfc/0001/decisions-d23-d26` | D23 settled, D24 opened, `F5`/`F8` as clauses |
| 5 | Tier 3, then 4 and 5 | `rfc/0001/tier3-gaps`, `rfc/0001/tier4-5-editorial` | done |
| 6 | Structural | `rfc/0001/normative-index`, `rfc/0001/internal-reference-sweep` | done |
| 7 | Version and changelog | `rfc/0001/version-and-changelog` | 1.1.0, entries 68-76, renumbered behind main's 66-67. Rebased onto published `main` (#34); D24 settled in §12.2 |
| 8 | Proof | -- | pytest 1199 passed / 6 skipped / 1 failed, Ruff check and format, mypy clean. The failure is `test_smoke.py::test_runtime_and_distribution_versions_agree`, inherited from main: the venv's editable install still reports `0.0.1.dev0` against the released `0.1.0` |


| # | Phase | Branch(es) | State |
|---|---|---|---|
| 1 | Measurements (`F1`, `F2`, `F4`) | `rfc/0001/evidence-*` | `F4` archaeology done; `F1`, `F2` running |
| 2 | Tier 1 corrections | `rfc/0001/tier1-corrections` | `F3`, `F7`, `F19`, `F25`-spec landed; `F2`, `F4` held for measurement |
| 3 | `F25` in code | `diagrams/meta-propagation` | **done**, merged to `main` at `c265a42` |
| 4 | Tier 2 decisions | `rfc/0001/decisions-d23-d26` | not started |
| 5 | Tier 3, then Tier 4/5 | `rfc/0001/tier3-gaps`, `rfc/0001/tier4-5-editorial` | not started |
| 6 | Structural | `rfc/0001/normative-index`, `rfc/0001/internal-reference-sweep` | not started |
| 7 | Version and changelog | `rfc/publish-for-comment` | not started |
| 8 | Proof | -- | not started |

## Working areas

- Measurement environments and raw output live outside the repository, in
  `../.rfc0001-work/`. Nothing there is committed; the evidence scripts and
  their recorded figures are what land.

## Log

- 2026-08-23 -- Plan read. Environment surveyed: the project venv has
  numpy 2.5.1, scikit-learn 1.9.0, gudhi 3.13.0, ripser 0.6.15, persim 0.3.8;
  giotto-tda, jax and torch are absent. Measurements therefore need their own
  environments, built with `uv` and pinned per finding.

- 2026-08-23 -- `F25`'s code half landed on `diagrams/meta-propagation` and
  merged. `_masked` is now the degree-restricting path only; a new
  `_dropping_essential` is shared by `finite` and `finitize(at="drop")`. Eight
  regression tests, five of which failed before the change. Full pytest, Ruff
  and mypy clean.

- 2026-08-23 -- Tier 1's measurement-independent half landed on
  `rfc/0001/tier1-corrections`: `F7`, `F3` (plus `O9`'s `__iter__`), `F25`'s
  specification half, `F19`.

- 2026-08-23 -- **`F4` is not the finding it was written as.** The archaeology
  found that `rfcs/evidence/probe_backends.py` carries a scikit-learn
  compatibility shim (`patch_giotto`, `_make_check_array_shim`) that
  translates giotto's `force_all_finite` into `ensure_all_finite`, so
  giotto-tda 0.6.2 *does* run on scikit-learn 1.8.0 under that script and
  Appendix A's preamble does not contradict §9.2. The real defect is that the
  document has never mentioned the shim, and that
  `.github/workflows/ci.yml:185-187` runs a **live** giotto call with
  `--require-giotto` on an unpinned install while `pyproject.toml:119-121`
  says CI uses committed fixtures instead. Report in
  `../.rfc0001-work/evidence/appendix_a_history.md`. Two further defects fall
  out: A.4's Warnings column is provably from a 2026-07-30 rerun rather than
  the 2026-07-29 run the preamble names, and no run artifact has ever been
  committed, so the preamble's version list is attested by prose alone.

## Code follow-ups owed

Small and deliberately not mixed into an RFC-only branch:

- `DiagramBatch.__getitem__`'s docstring still says iteration "works through
  the legacy `__getitem__` protocol"; `__iter__` has existed since then.
- `DiagramBatch.__getitem__` is annotated `i: int` where it accepts, and §4.2
  now specifies, `SupportsIndex`.

## Judgement calls made without an owner

- **No new `R` rows in §12.3.** Four Tier 1 findings are "the document stated
  something false", which is what §12.3 is for, but that section's preamble
  scopes itself to the six found on branch `adapter2` and extending it means
  rewriting the framing. Appendix C carries the record instead, which is what
  the plan's phase 7 asks for. Cheap to reverse if the owner disagrees.
- **`O9`'s `__iter__` folded into `F3`.** The plan's tracker table says one
  amendment to §4's interface block closes both, and questions.md records the
  decision as already taken with the implementation complete.

## What is left

- **`rfc/publish-for-comment` has landed**, as #34 on `origin/main`, and this
  branch is rebased onto it -- 20 commits ahead, nothing behind. The plan's
  ordering constraint said `F6` and `F21` settle before it merges, because
  that commit reproduces §10.2's bump-rule text verbatim. It is discharged
  from the other side instead: the text is inherited rather than reproduced,
  and D24 settled in favour of keeping the rule, so the publication diff would
  not have moved either way.
- The dependency-closure check was not re-run in a clean environment: it
  exits 1 in the developer venv on the *unmodified* baseline tool too, gudhi
  and hopcroftkarp being installed there, and CI builds `.venv-closure`
  separately for it. This pass added no dependency -- the only
  `pyproject.toml` change is one pytest marker.

## Code follow-ups still owed

- `DiagramBatch.__getitem__`'s docstring still says iteration "works through
  the legacy `__getitem__` protocol"; `__iter__` has existed since then.
- `DiagramBatch.__getitem__` is annotated `i: int` where it accepts, and §4.2
  now specifies, `SupportsIndex`.
- `probe_backends.py` prints one version, conditionally, and only
  scikit-learn's. Appendix A's provenance problem was that no run artifact
  has ever been committed; having the script print every version it ran
  against is the cheap half of fixing that.
- §11.2's determinism case now requires asserting `ZipInfo.date_time` and
  `compress_type` directly (`F30f`); `tests/test_rfc0001_io.py` still sleeps.
- §10.1 now requires `save()` to refuse a non-host-resident array by name
  (`F29`); `io.py` does not yet check.
- §3.1 now requires the clamp target and the `clamped_rows` record (`F9`);
  `adapters.py` should be read against it.
- §11's three-termed impossibility check (`F2`) is still unimplemented in
  `adapters.py`, as it was before this pass.

## Revisions after the first pass

Four corrections, on the owner's reading:

- **The index is Appendix C and the changelog is Appendix D.** The changelog
  is removed when the comment window closes, by its own author's note, and
  the index is permanent -- so the permanent one goes first and the removal
  leaves no gap in the lettering. `tools/normative_index.py` inserts itself
  *before* the changelog rather than at the end, so the order cannot drift
  back. Two generator defects surfaced by the move are fixed: appendix
  subsection labels took the whole heading, and a bulleted requirement was
  glued to the paragraph above it. The index now covers §1-§11 only.
- **Changelog entries rewritten to length.** They were 181-289 words against
  a house median of 52 across entries 1-65; they are now 52-120, and nine
  entries rather than six, renumbered 68-76 -- behind, not ahead of,
  `rfc/publish-for-comment`'s own 66 and 67, which landed on `main` as #34
  before this branch was rebased onto it.
- **The document no longer refers to a review of it.** A reader cannot
  resolve a pointer to a document that is not this one.
- **Nor to components it does not affect.** The §1 gloss table answered the
  wrong question: the fix for an unresolvable reference is to remove it, not
  to define it. §1 now names only `diagrams/core.py`, `diagrams/adapters.py`,
  `diagrams/io.py` and `core/distances.py`, which §9.1 binds. `castle/`,
  `repro/`, `compat/`, the licensing files and the `classify` repository are
  gone from the body and from §12. **A.6's `classify` references are kept**:
  that is a dataset source rather than an Akriti component, and a bar count is
  only checkable against the data that produced it. D2 was quoting §4's old
  wording and was stale in any case.
