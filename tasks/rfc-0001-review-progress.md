# RFC-0001 review execution -- progress log

Live state for the work described in `rfc-0001-review-plan.md`. Written so the
pass can be resumed from a cold start: every row says what landed, on which
branch, and what proves it.

Started 2026-08-23. Base: `main` at `149e958`.

## Phase status

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
