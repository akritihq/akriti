# RFC-0001 review execution -- progress log

Live state for the work described in `rfc-0001-review-plan.md`. Written so the
pass can be resumed from a cold start: every row says what landed, on which
branch, and what proves it.

Started 2026-08-23. Base: `main` at `149e958`.

## Phase status

| # | Phase | Branch(es) | State |
|---|---|---|---|
| 1 | Measurements (`F1`, `F2`, `F4`) | `rfc/0001/evidence-*` | in progress |
| 2 | Tier 1 corrections | `rfc/0001/tier1-corrections` | not started |
| 3 | `F25` in code | `diagrams/meta-propagation` | not started |
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
