# RFC-0001 — Process History

Non-normative. This document holds the full narrative that the main RFC
(`0001-persistence-diagram-interchange.md`) now only points to. Nothing here
changes or adds to any requirement; every MUST, SHOULD and MAY lives in the
main document. This file exists so the audit trail survives being pruned out
of the RFC before publication, per the M1 target.

What is below, in order:

- **Full changelog** — every entry in full. The main RFC's Appendix B carries
  one line each.
- **Original "Note on Dx" text** — the explanations that used to sit below the
  §12 decision table (D1, D2, D6, D8), plus D9's and D12's, which were written
  here directly once that convention was retired.
- **Body narrative relocated in the 2026-08-02 pass** — superseded designs and
  first-draft corrections from §3, §4.1, §4.2 and §9.1.
- **Body narrative relocated in the 2026-08-05 pass** — the same, from §5,
  §6.1 and §6.3.
- **What did not move here** — the standing rule for what stays in the RFC.
- **D6, D9, D10, D11** — removed from RFC-0001's scope, their §12 table rows
  preserved verbatim, with the reasoning for removing rather than resolving
  them.

Seven passes have relocated or added material here, dated 2026-07-31 through
2026-08-05; the changelog entries record which did what. The seventh (entry
35) is the first to change `diagrams/core.py` as well as the RFC, so its
entry carries an implementation section the earlier ones had no need for.

---

## Full changelog

- **2026-07-29** — Initial draft.
- **2026-07-30** — Added §4.1, reconciling the two-type design
  (`PersistenceDiagram` / `DiagramBatch`) against a prior proposal for a single
  type with an internal dense, padded batch representation. Added a
  cross-reference note under D2. No normative requirement in §3 through §11
  changed; §4's existing rules were already sufficient, they just were not
  connected back to the prior discussion anywhere in the text.
- **2026-07-30 (2)** — Added §4.2, specifying `DiagramBatch`'s internal storage
  as a shared CSR buffer (concatenated `dims`/`births`/`deaths` plus
  `offsets`), matching §10.2's on-disk layout, with `__getitem__` returning a
  zero-copy view rather than an independently allocated object. Considered and
  rejected a further alternative, merging `PersistenceDiagram` and
  `DiagramBatch` into one CSR-backed type, over `DiagramMeta` (§8) and
  `content_hash` (§8.1) not merging cleanly across a batch. Updated the D2
  note accordingly.
- **2026-07-30 (3)** — Fixed a pre-existing contradiction: D1 recommended
  Parquet while §10.1/§10.2 rule it out and normatively specify `.akd`. D1 is
  now marked resolved by §10. Added `DiagramBatch.from_diagrams` to §4.2, the
  missing constructor from ordinary (non-giotto) adapter output. Added
  `DiagramBatch` equality rules to §6.3, order-sensitive across diagrams,
  unlike the order-insensitive multiset equality within one diagram. Added D7,
  an open, unresolved question on whether `DiagramBatch` needs its own
  `content_hash` for the table-reproduction commitment.
- **2026-07-30 (4)** — Added I8 to §3.1, requiring `PersistenceDiagram`
  immutability, which §4.2's view-based `DiagramBatch.__getitem__` already
  depended on without stating it. Added B1 through B5 to §4.2, invariants on
  `offsets` that the slice arithmetic in `__getitem__` silently assumed. Added
  a batch-canonicalization rule to §7: `DiagramBatch.canonical()` sorts within
  each `[offsets[i], offsets[i+1])` segment independently and MUST NOT reorder
  segments, since the flat three-pass `argsort` as originally written would
  interleave rows across diagram boundaries.
- **2026-07-30 (5)** — Added D8, proposing Parquet as a `pyarrow`-gated
  interoperability export in §10.3 alongside `to_arrays()`/`to_csv()`, kept
  separate from the `.akd` default rather than replacing it. Added D9,
  flagging that §10.1's "dependency-free" requirement is enforced elsewhere
  as "MIT/BSD-only," which may be an oversight given Akriti's own outbound
  license is Apache 2.0; not resolved here, and noted as possibly needing a
  call in the onboarding document rather than this RFC alone. D8 is marked as
  depending on D9, since `pyarrow` itself is Apache 2.0 licensed.
- **2026-07-30 (6)** — D9 resolved on new information: the onboarding
  document's own 2026-07-30 revision retracted its "MIT/BSD-only" language.
  Measurement showed GPLv3 (`hopcroftkarp`, via `persim`) and unlabeled
  GPL-marked code (the `gudhi` wheel) arrive transitively regardless of a
  backend's own license, so the real policy was never license-family-based;
  it keeps the default install dependency-free entirely and pushes every
  backend behind a labeled extra, any license, enforced by
  `tools/check_license_closure.py`. Corrected §10.1's requirement (2), which
  had restated the retracted "MIT/BSD-only" framing. Updated D8 to drop the
  now-void dependency on D9 and recommend `akriti[parquet]` following the
  existing extras pattern.
- **2026-07-30 (7)** — Corrected a second, related mistake in §10.1's
  requirement (2) and in D9's own resolution text: both had described the
  default install as "the interchange layer plus `numpy`," which contradicts
  §3/§3.4's array-API-generic mandate for `core/`. The default install MUST
  be solely the interchange layer (`core.py`, `adapters.py`), zero
  third-party dependencies including `numpy`; `numpy` is needed only inside
  `io.py`'s `save`/`load` for the NumPy-native `.npz` format, and MUST be a
  lazy, function-scoped import there rather than a package-level dependency.
  Added a sentence to §3.4 stating this explicitly. Added D10, the remaining
  open implementation choice (lazy import vs. formal `[io]` extra), with a
  recommendation for the lazy-import approach.
- **2026-07-30 (8)** — Corrected §5.1's account of giotto's essential-bar
  loss (review comment). The prior text implied `infinity_values` was simply
  ignored for essential bars; the actual cause is `reduced_homology=True`
  (giotto's default), which by definition omits the one class every diagram
  has, before `infinity_values` has any bearing on it. `infinity_values`
  correctly governs every other case. Updated the §5.1 table entry, the
  prose, and the adapter-consequence requirement (now conditional on the
  caller's `reduced_homology` setting rather than unconditional). Added a
  note to Appendix A.1 flagging that the measured table varies only
  `infinity_values`, not `reduced_homology`, and that a
  `reduced_homology=False` row should be added to `probe_backends.py`.
- **2026-07-30 (9)** — Reworked §5.1's adapter-consequence text: `giotto`'s
  `reduced_homology` is now a stored `params` fact rather than an inspected
  setting, `provenance["essential_bars"]` is explicitly derived from it
  (never independently set), the loss is scoped to H0 only (reduced homology
  has no effect on H1+), and the non-fabrication rule's justification was
  replaced with an elder-rule argument: the missing bar's birth is the
  minimum vertex birth over the whole point cloud, which the remaining rows
  cannot recover except by coincidence on an unweighted complex, and
  `VietorisRipsPersistence`'s `weights` parameter breaks that coincidence in
  the ordinary case. Added a note to §8 explaining why `essential_bars` is
  kept as a derived, single-sourced field rather than folded away as
  redundant with `params`: it is backend-agnostic where `params` is not, and
  `finitize()` needs to write the same field on diagrams that never had a
  `params["reduced_homology"]` to derive it from in the first place. Added
  `reduced_homology` to `params`'s example keys in the `DiagramMeta` code
  block.
- **2026-07-30 (10)** — `from_giotto`'s `reduced_homology` changed from an
  assumed-default-with-warning to a required, keyword-only parameter
  (`from_giotto(arr, *, reduced_homology, **meta)`); omitting it now raises
  rather than falling back to giotto's own default. §9.1's own evidence-script
  warning-suppression incident was the deciding precedent: this document has
  direct, measured proof within itself that a missed warning here would leave
  no trace. Updated §11's signature block to show the one non-uniform
  adapter, with a note explaining the deviation is deliberate.
- **2026-07-31 (11)** — Editorial pass (external review). Renumbered the
  section previously labeled §3.4 to §3.3 — no §3.3 existed anywhere in the
  document, a gap left over from an earlier edit — and updated every
  cross-reference to it (§3, I7, B5, §6.1, §9.2, §10.1, D6, D9, D10). Entries
  dated 2026-07-30 and earlier in this changelog still say "§3.4"; read that
  as "§3.3" under the current numbering. Reordered the I1–I8 invariant table
  into numeric order (I7 previously sat between I2 and I3, all eight rows
  otherwise unchanged). Replaced the stored `xp: Array` field on
  `PersistenceDiagram` (§3) with a derived `xp` property returning
  `self.dims.__array_namespace__()`: as a stored field it was a fourth piece
  of state with no invariant requiring it to agree with `dims`/`births`/
  `deaths`, and it was typed `Array` despite holding a namespace, not array
  data. I7 already requires the three stored arrays to agree; the property
  makes disagreement with `xp` impossible rather than merely unchecked.
  Specified §11.1's `provenance["padding_removed"]` value for
  `strip_padding=False` (`0`, previously unstated). Added D11, flagging that
  D9's "resolved" status rests on an onboarding-document retraction this RFC
  asserts but does not independently confirm; this needs checking against
  the actual onboarding document before M1, not taken on the RFC's own say-so.
  Added a note on D6, flagging that its "raise the floor to `numpy>=2.0`"
  recommendation is written as a packaging constraint but predates D10's
  later decision that no floor can be declared anywhere in this project's
  zero-dependency default install.
- **2026-07-31 (12)** — Applied D8: added `to_parquet()` to §10.3, a
  `pyarrow`-gated escape hatch (`akriti[parquet]`, Apache 2.0), same three
  columns and order as `to_csv()`, lazy-imported the way `save`/`load`
  already lazy-imports `numpy` (D10). `DiagramBatch` gets a `diagram_id`
  column instead of an `offsets` array, matching Parquet's row-oriented
  model rather than forcing the CSR layout `.akd` uses. Carries none of
  `DiagramMeta`, by design — a bars-only table for R/pandas/Polars,
  serving §1's R-bridging goal, not a `.akd` replacement. Added a
  one-line clarification to §10.1 that its Parquet exclusion concerns the
  default format only. Marked D8 applied in §12 and added a note flagging
  that this still rests on D9, which D11 has not yet confirmed.
- **2026-07-31 (13)** — Condensed §12 (open decisions) and Appendix B
  (this changelog) in the main document for readability. Every "Note on Dx"
  paragraph that previously sat below the §12 table (D1, D2, D6, D8) was
  merged into its own table cell and trimmed; where the reasoning was
  already fully carried by the body text (D1 by §10, D2 by §4.1/§4.2), the
  cell was reduced to a pointer. The full original notes are reproduced
  below for reference. No normative requirement changed.
- **2026-08-02 (14)** — Second condensation pass, this time on body
  narrative rather than process/meta content. Four passages moved to this
  document in full, replaced in the main RFC by their conclusion plus a
  pointer: the §3 "this was `np.ndarray` in the first draft" aside and the
  §3 "`xp` is a derived property, not a stored field" aside (both
  first-draft corrections); the §4.1 "Reconciling the two-type design"
  section (the rejected single-type-with-dense-padding proposal); the §4.2
  "why not merge `PersistenceDiagram` and `DiagramBatch` into one CSR-backed
  type" discussion, including the PyTorch Geometric precedent; and the §9.1
  "Correction, 2026-07-30" blockquote about persim's warning behavior.
  Unlike the 2026-07-31 pass, this one touches sections 3, 4.1, 4.2, and 9.1
  directly rather than only §12/Appendix B, so it required judgment about
  which prose was genuine hazard-documentation (kept in the body, untouched)
  versus process narrative about superseded drafts and rejected designs
  (moved here). No normative requirement changed; verified by diffing every
  I1-I8/B1-B5 invariant row, every MUST/MUST NOT/SHOULD sentence, the §11
  adapter signatures, and the four Appendix A tables against the prior
  revision.
- **2026-08-02 (15)** — Local tightening of redundant connective prose,
  nothing relocated to this document. §3.3's two "serialization is
  NumPy-bound" paragraphs merged into one, all three MUST clauses kept
  intact. §3.3's "conformance is tested" paragraph shortened to point at
  §3's (now relocated) first-draft note instead of restating the same
  fact. §9.1's three paragraphs on why persim's warning is a
  severity-mismatch bug rather than a silent one, merged to two. Appendix
  A.4's caption trimmed of a throat-clearing lead-in ("Note what the counts
  imply:"). One self-check finding worth recording: the first draft of this
  entry's own changelog line used the bare word "MUST" to describe the
  edit, which is the RFC's own RFC-2119 keyword, and tripped this
  document's MUST-count verification step even though no requirement had
  changed. Reworded to "normative clauses" before finalizing, specifically
  so the changelog doesn't itself read as asserting a new requirement.
- **2026-08-02 (16)** — Second §12 trim: D6, D8, and D9's cells shortened
  further, each to its status plus whatever is still live (D6's two
  reframing options, D8's outstanding D9/D11 dependency and TODOs, D9's
  "not yet independently verified" flag). D9 is the first of these without
  a pre-existing below-table note to fall back on; the full license
  reasoning it previously carried inline is preserved below as a new Note
  on D9. D11 was left untouched, it is already the terse action item
  itself, not narrative to condense. Converting the changelog list (this
  section) to a Date/Change table was considered and rejected: several
  entries run to full sentences, and a table would force those into single
  unwrapped cells, worse to read than a bulleted list — this document
  reserves tables for genuinely short, structured data (invariants,
  measured evidence), which a chronological narrative log is not. No
  normative content changed; same verification battery as entry 14, plus a
  full-document table-row diff to confirm no table cell anywhere, in either
  file, changed by so much as one entry.
- **2026-08-02 (17)** — Split §12 into §12.1 "Needs the lead before M1" and
  §12.2 "Settled," open rows first. This was a direct response to being
  asked whether resolved rows should simply be deleted to reduce noise;
  the answer given was no, reorder instead, for two reasons specific to
  this table: §1's own stated purpose for the RFC includes being the
  parallel-work contract and decision record, so a closed decision with its
  reasoning stripped invites someone to reopen it without knowing it was
  already settled; and this table has rows whose cells cross-reference each
  other by number (D8 and D11 both point at D9), so deleting a resolved row
  can orphan a live one's pointer. D-numbering was left untouched,
  including inside §12 itself; only grouping and row order moved. Two
  placement calls, both logged in the entry itself rather than left
  implicit: D9 went in the open group despite its "Resolved" status word,
  because it is explicitly contingent on D11 and would mislead if filed
  under "settled"; D5 stayed in the settled group even though the issues it
  calls for haven't been filed yet, because what was actually undecided
  (which course of action) is decided — filing them is engineering
  follow-up, not something still awaiting the lead's judgment. No normative
  content changed; verified that the same eleven rows, D1 through D11,
  appear exactly once each with byte-identical cell text before and after,
  via a sorted table-row diff rather than a positional one (positional
  diff is meaningless once order changes on purpose).
- **2026-08-02 (18)** — Moved the top-of-document "Note on this revision"
  callout into this appendix, at the request of a direct instruction rather
  than as part of a planned pass. It had drifted into restating, in
  miniature, what entries 13 through 17 already say in full, and it sat
  between the RFC-2119 keyword line and §1, meaning every reader had to
  pass a paragraph about the document's own edit history before reaching
  the first substantive section, on every visit, regardless of whether
  they cared. Its content became this appendix's opening sentences; no
  wording was newly invented, and no normative content changed.
- **2026-08-02 (19)** — Made outside the pass-by-pass process this document
  otherwise records: edited directly, then presented for review rather than
  planned and verified against this file first. Its own changelog line reads
  "Removed references to history unrelated to decision-making. The history
  doc exists for a reason," which covers exactly one of six actual changes —
  dropping the two `(History: ...)` parentheticals in §3. The other five,
  none of them history references: removed "NumPy is the expected and
  default backend; it is not the required one" from §3; removed the "so a
  NumPy-pinned container would make that unachievable..." rationale from the
  same paragraph; removed "never against NumPy dtype objects" from I2's
  invariant-table row (the same caution survives in §6.1's prose, so the
  point isn't lost from the document, only from the standalone I1-I8
  checklist); normalized §2's `+∞` to `` `+inf` `` (defensible on its own —
  it was the only place in the document using the math symbol instead of the
  code-styled token used everywhere else — but not a history reference
  either); and replaced §3.3's `` **`lexsort` is not in the standard`` item,
  including the `hasattr(xp, "lexsort")` false-positive explanation, with
  `` **`argsort` is in the standard.** `` This last one was the one worth
  stopping on: it broke §3.3's own "Three limits" framing (the replacement
  isn't a limit), and it emptied out what §7's `"not part of the array API
  standard (§3.3)"` cross-reference was pointing at. Entry 20 addresses it;
  the other four items are untouched as of this entry.
- **2026-08-02 (20)** — Resolved via direct discussion rather than
  independent judgment: asked whether the `hasattr(xp, "lexsort")` trap
  needed prose explanation at all, given §3 already explains why the array
  API standard is required in the first place. The two aren't the same
  claim — §3's reasoning is strategic (why be namespace-generic at all);
  the trap is a specific, narrow way to violate that goal while believing
  a defensive check protects you, which is a different and non-obvious
  fact, not a restatement. But its home didn't have to be a paragraph:
  the RFC's own "Conformance is tested, not intended" principle (§3.3)
  applies here as directly as anywhere else in the document, so the trap
  moved from prose into an actual test requirement instead. Concretely:
  §3.3's misfit item is removed outright, not replaced with another claim,
  and "Three limits" becomes "Two limits," genuinely descriptive again.
  §7's dangling `(§3.3)` pointer is dropped, since §3.3 no longer discusses
  lexsort in any form. §7's "Verified against `np.lexsort`..." sentence,
  previously a one-time verification note, is upgraded to a normative,
  requirement that the comparison stay in CI as a standing regression test,
  with the `hasattr` trap named in one clause so a future implementer still
  knows what the test is actually guarding against, without a full
  paragraph doing it. This is flagged in the main document's entry 20 as
  the one non-normative-neutral change in the run: requiring a test to stay
  in CI, rather than reporting that a check was run once, is new normative
  text, not a relocation or a trim, and this document's own standard is to
  say so plainly rather than let "no normative content changed" quietly
  stop being true. The other three items entry 19 introduced (the
  NumPy-default sentence, the NumPy-pinned-container rationale, I2's dtype
  caution) were raised but intentionally left untouched pending a separate
  decision, not overlooked.
- **2026-08-02 (21)** — Opened D12: whether `bars.npz` should remain the
  default array storage inside `.akd`, or whether a fully dependency-free
  format, stdlib `csv`/`tsv` or stdlib `sqlite3`, should replace it. This
  differs in kind from entries 13 through 20: those condensed, relocated,
  or corrected existing text; this one adds a genuinely new open question,
  surfaced by a gap in §10.1's own 2026-08-02 rewrite, which tested `.npz`
  against HDF5 and Parquet on all five requirements but never tested it
  against the two candidates that clear requirement 2 outright rather than
  through the lazy-import exception numpy needs. Added a scoping sentence
  to §10.1's concluding paragraph so its "therefore follows from
  requirement 5" claim reads as tested against binary alternatives only,
  not as having foreclosed CSV or SQLite; this is the same category of fix
  the Note on D1 describes, a conclusion overclaiming past what its own
  argument established, caught before publication rather than after. Added
  D12 to §12.1, following D9's entry-16 precedent: no below-table note in
  the main RFC first, full reasoning written directly here as a new Note
  on D12, since that pattern is already established for a decision opened
  after the below-table-note convention was retired. Updated §12's intro
  sentence from "Four" to "Five" and added D12 to the enumerated list,
  grouped with D6 and D7 as calls the document can't make for itself.
  Flagged in the note that resolving D12 toward CSV or SQLite requires
  coordinated edits to §4.2's CSR on-disk layout and §10.2's format
  specification, not a change to §10.1 alone, the same kind of cross-
  section dependency D8 already carries toward D9, and that it would also
  revisit D10's numpy-ubiquity justification, which stops being load-
  bearing if the default format no longer needs numpy at all. No normative
  content changed: §10.2's `bars.npz` specification, §4.2's storage
  requirements, and D10's lazy-import recommendation are all untouched
  pending the lead's call; verified by diffing every I/B invariant and
  every MUST/MUST NOT/SHOULD sentence outside §12 and §10.1's one new
  sentence, and confirming D1 through D11's cells are byte-identical
  before and after.
- **2026-08-03 (22)** — Added the rationale §10.1 requirement 4 had never
  stated, prompted by a direct question about whether byte-determinism is
  actually needed for round-tripping or for hashing. It is neither.
  `load(dump(d)) == d` (requirement 1) is a single save-then-load cycle and
  is satisfied regardless of whether two separate `save()` calls on the
  same diagram agree byte-for-byte; nothing in requirement 1 depends on
  requirement 4. `content_hash` (§8.1) is computed from the diagram's own
  canonical-ordered arrays in memory and never touches a serialized file,
  so it is already fully determined by §7's canonical ordering, independent
  of §10 entirely. The actual point of requirement 4, made explicit for the
  first time, is artifact-level reproducibility: the ability to verify with
  a checksum or `diff` alone, no library invocation required, that
  regenerating a `.akd` fixture from `repro/` reproduces exactly the file
  previously committed or published, the same audit-without-our-library
  spirit requirement 5 already states, and the reason §11.2 lists byte-
  determinism as its own dedicated test rather than folding it into the
  round-trip test. This also required rewriting §7's own sentence, which
  had read "the on-disk format is written in canonical order, so
  byte-identical diagrams produce byte-identical files and a content hash
  is meaningful," a phrasing that puts both effects in one causal chain
  when they are in fact two independent consequences of the same cause
  (canonical ordering), not one depending on the other. The rewrite states
  them as parallel: canonical ordering alone makes `content_hash`
  well-defined, and canonical ordering is one necessary but insufficient
  ingredient of file-level determinism, which also needs zip metadata
  pinned, a fact §10.1 already tracked as open but had not connected back
  to §7. Two implementation drafts of this entry's own text tripped the
  MUST/SHOULD count check by using the bare keyword pair to describe what
  was being verified, the identical trap entry 15 first documented and
  entry 21 re-tripped; both were caught by the same count comparison and
  reworded before being kept, in the main RFC's version, not here, since
  this document's own non-normative status means the keywords appearing in
  its prose was never the hazard. No normative content changed in either
  file; the main RFC's MUST/MUST NOT/SHOULD/MAY count is 55 before and
  after, matching entry 20's baseline, confirmed by direct recount rather
  than trusted from memory.
- **2026-08-03 (23)** — Added a disclosure paragraph and cross-reference list
  of every accessor to §3.2. `d.finitize()` is named explicitly as a
  deliberate exclusion, not an oversight, consistent with I8's existing
  rationale sentence, which already treats it separately from §3.2. No
  normative content changed.
- **2026-08-03 (24)** — Added §4.3, the `DiagramBatch` counterpart to §3.2.
  Core, self-contained list: `len(b)` and `b[i]`, everything specifiable from
  §4.2 alone. Cross-referenced, not restated: `b.canonical()` (§7) and
  `b1 == b2`/`b1.allclose()` (§6.3). Named `from_diagrams` as a deliberate
  exclusion (constructor, not accessor). Flagged two genuine gaps rather than
  papering over them: no batch-level `content_hash` (D7's open question, not a
  placement choice), and no `xp` or batch-level
  `dim(k)`/`dimensions`/`essential`/`finite`/`persistence`. Updated §3.2 to
  reflect the new section. No normative content changed.
- **2026-08-03 (25)** — Added four safe accessors to §4.3: `essential`,
  `persistence`, `bar_counts`, and `xp`. Left `dim(k)`, `finite`, and
  `dimensions` flagged-open rather than added, and `content_hash` pointing at
  D7, matching entry 24's existing (now trimmed) gap list. Added
  `b1.same_provenance(b2)` to §8, order-sensitive the same way `==` and
  `allclose` already are (§6.3), fixing an omission entry 24 itself had.
  Sharpened §4.1's claim ("DiagramBatch owns no invariant or numerical code of
  its own... persistence... written once, against PersistenceDiagram") to
  protect against any `DiagramBatch` accessor needing its own new rule, true of
  the four additions (mechanical consequences of I4, I5, B4, B5) and not true
  of `canonical()` or equality (each needed one, §7, §6.3). `bar_counts` was
  deliberately not named `n_bars`, since `PersistenceDiagram.n_bars` is a
  scalar and this is an array; reusing a name across a shape change is the same
  silent-wrongness class §9 targets elsewhere. No normative content changed.
- **2026-08-03 (26)** — Design-review pass. Normative changes: **(1)** §11's
  `from_giotto` now always returns `DiagramBatch`, length 1 when
  `n_samples == 1`; a return type that depends on the input's own batch
  dimension is the adapter-surface version of the
  shape-depends-on-what-else-was-there hazard §4/A.2 already rules out inside
  diagrams, and it's ruled out here the same way. Updated the §5.1 signature
  block and the §11 adapter table to match; added a paragraph to §11 explaining
  the change. **(2)** `finitize(at="drop")` (§5) now records
  `provenance["essential_bars"] = "finitized_dropped"` plus a new
  `provenance["essential_bars_dropped"]` count, rather than being silently
  unrepresentable in the `"finitized_at:<value>"` form that only fits the two
  substituting modes; added both to §8's reserved-key table. **(3)** §11.2's
  real-backend-output requirement now states explicitly that a frozen fixture
  captured from an actual backend call counts as real backend output,
  reconciling it with §9.2's stored-fixture requirement for giotto rather than
  leaving the two sections in unstated tension. **(4)** §10.2 now separates the
  container format (zip archive, JSON metadata split from binary array data —
  settled, MUST-level, does not wait on D12) from the `bars.npz` array-payload
  choice (now explicitly provisional pending D12, not MUST-level); no prior
  MUST-level text is weakened, this makes explicit what D12's own existence
  already implied. **(5)** §9.2 adds a status paragraph: `from_giotto` is a
  best-effort compatibility shim, not a peer of `from_gudhi`/`from_ripser`,
  justified by the audit's install-rate evidence for a stranded giotto userbase
  (roughly 6,391/month against zero commits in 52 weeks). This changes
  priority, not scope — §4/A.2, §5.1, §9.2, and §11.1's full
  giotto-interoperability spec are unchanged. §12: **removed D6, D9, D10, and
  D11** —
  dependency-and-licensing policy, out of this RFC's scope on review; full
  prior table text preserved below, along with a note on why removal rather
  than resolution. Updated D8 to no longer depend on the removed D9/D11.
  **Added D13** (multiparameter persistence representability). D-numbers are
  not renumbered to close the gap left by the four removals, consistent with
  D-numbers being stable identifiers rather than a dense sequence (already
  established when entry 17 reordered rows without renumbering them).
- **2026-08-03 (27)** — Added a Rationale column to §4.2's B1-B5 table,
  matching §3.1's I1-I8 table. No normative content changed.
- **2026-08-03 (28)** — Removed all references to specific papers, as this
  repository is meant to be universal.
- **2026-08-04 (29)** — Resolved D7 on the lead's guidance. Added §8.2
  (defining `DiagramBatch.content_hash`), referencing §10.1 requirement 4.
- **2026-08-04 (30)** — Trimmed changelog to avoid restatements. No normative
  content changed.
- **2026-08-04 (31)** — Added I9 (`dims`, `births`, `deaths` each rank-1) to
  §3.1's invariant table, closing a gap where a same-length-but-wrong-rank
  array could pass I1 unnoticed. Updated §4.2's cross-reference from "I1
  through I8" to "I1 through I9".
- **2026-08-05 (32)** — Normative changes: **(1)** §6.1 and I2 disagreed about
  how a dtype is checked, and the version I2 named was wrong —
  `xp.isdtype(a.dtype, "real floating")` is true of `float32`, which D3 rejects
  outright, so as a check on I2 it is no check at all. Both now require
  equality against the namespace's own `xp.float64` / `xp.int32`; `xp.isdtype`
  keeps its place for genuine kind-level questions, which I2 is not. **(2)**
  Added **B6** (`offsets` rank-1) and **B7** (`offsets` is `int64`), both
  already implied by the class body, and B1's `len(offsets)` reads `shape[0]`,
  which answers happily for a rank-2 array — entry 31's gap, one field over.
  Stated, not new. **(3)** §4.2's `from_diagrams` now checks that every input
  diagram shares one namespace (I7 constrains a single diagram's three arrays
  and says nothing across diagrams, and `concat` erases the evidence), and
  gains an `xp=` keyword required for, and only for, an empty `diagrams` — an
  empty batch is valid everywhere else in this document, so refusing to
  construct one left the private path as its only constructor. **(4)** §5:
  `finitize` on a diagram with no essential bars MUST return it unchanged,
  provenance included; recording `"finitized_dropped"` with a count of zero
  asserts a cardinality change that did not happen, and either mode's record
  would overwrite a `"lost_upstream"` with a claim about work the call did not
  do. The mode argument is still validated first, so a typo does not depend on
  the data to be caught. **(5)** §5, §5.1, §8 and §11: added reserved key
  `essential_bars_source`, `essential_bars` as the adapter recorded it, written
  by `from_*` at construction and by nothing afterwards. `essential_bars` is a
  single slot describing the bars as they now stand; a giotto-sourced
  diagram's `"lost_upstream"` is a claim about how they were *computed*, which
  no later transformation makes untrue, and one overwritable key cannot answer
  both questions, which is what a lossy diagram that is then finitized exposes.
  Having `finitize` copy the current value forward on its first call was the
  first form of this and is rejected in §5: only the adapter knows the answer,
  a copy-forward admits `"finitized_*"` values the key cannot legitimately hold
  (reachable through `load`, since §10.1 requirement 1 round-trips provenance),
  and it makes the key's absence ambiguous between "no adapter recorded one"
  and "never finitized". The key shares `essential_bars`' string vocabulary
  rather than being a boolean, so that then-versus-now is one question asked of
  two keys, and because the fifth backend §8 anticipates extends a string enum
  and cannot extend a boolean. **(6)** §8: keys qualifying `essential_bars` are
  kept consistent with it, not merely written alongside it — a substitution
  after a drop clears `essential_bars_dropped`, whose "present iff" the obvious
  merge-into-existing-provenance implementation breaks on the second call.
  **(7)** §8.1 now specifies the hashed message — domain tag, bar count, then
  the three canonical-ordered columns big-endian — rather than leaving the
  layout to the implementation, and requires `-0.0` to be normalised to `+0.0`
  before hashing. The normalisation is a correctness fix, not a formatting one:
  `-0.0 == 0.0`, so §6.3 calls two such diagrams equal and §7's stable sort
  cannot separate them, while their IEEE 754 bytes differ, which made the
  digest depend on backend row order and `d1 == d2` with differing hashes
  reachable. The tag and length give §8.1 the two properties §8.2 already
  required of the batch hash; §8.2's domain-separation paragraph is updated to
  note that separation now holds from both sides. **(8)** Opened **D14** in
  §12.1: §6.3 requires `allclose` to be approximate and order-insensitive but
  never says how bars are paired, and the obvious implementation — canonical
  sort, then pairwise comparison — is exact in the sort and approximate in the
  comparison, which do not compose. Bars whose births lie within tolerance of
  each other can canonicalise into different orders on two backends, and the
  comparison then fails for diagrams that do have a partner for every bar
  within `rtol`, at the `2.7e-8` magnitude Appendix A.3 measures, on exactly
  the cross-backend case §6.2 defines `allclose` for. D14 carries both that and
  whether the tolerance is symmetric. §6.3 gains a paragraph stating the gap
  and pointing at the row. Left open rather than resolved here because the
  error is conservative — a spurious `False`, never a spurious `True` — so
  nothing downstream is silently wrong while the lead decides between a real
  matching and an accepted-and-documented false negative. §12's count moves
  from nine decisions to ten, three of them open. **(9)** §3.3: `finitize` is
  eager-only in **every** mode, not only `at="drop"`. Each mode must decide
  whether the diagram has any essential bar, since item (4) above requires one
  answer when it does and another when it does not, and that branch
  concretizes a traced array whatever mode was asked for;
  `at="max_finite_death"` additionally masks before reducing. The substituting
  modes preserve the output *shape*, which is a claim about the result and not
  about traceability — the two were conflated in the first implementation of
  item (4), which documented the modes as shape-preserving and left a reader to
  conclude they were available under `jax.jit`. §3.3 now states the distinction
  and names the other eager-only operations that are not filtering ones (`==`,
  `allclose`, `same_provenance`, `content_hash`), with `n_bars` called out as
  the exception that reads a shape rather than values. **(10)** §5: the
  "validated first" rule of item (4) now covers the whole `at` argument rather
  than the two mode names alone, and `at=<float>` MUST be finite. Both close
  data-dependent holes in the same check: `at=None` raised on a diagram with
  essential bars and returned the diagram silently on one without — item (4)'s
  own failure mode, one argument domain over — and `at=+inf` substituted an
  infinity for an infinity, leaving every essential bar essential while
  `provenance["essential_bars"]` recorded `"finitized_at:inf"`. The second is
  the mirror of item (4): there the record described work that reached no bar,
  here it describes work that reached every essential bar and changed none of
  them, so a diagram's `essential` mask and its own provenance contradict each
  other, and §9.1's partitioning reader and §8's human auditor reach opposite
  conclusions from the same diagram without either misreading what it looked
  at. `at=nan` was already excluded by I5, but incidentally and with an error
  naming death times; it now raises on the argument. §5 also fixes which
  exception each case raises, since two checks were being added and the answer
  should not be whichever one caught it: `TypeError` for an `at` that is
  neither a mode name nor convertible to a float, no diagram making such a call
  meaningful, matching the `TypeError` §6.3's `allclose` and §8's
  `same_provenance` already raise on a wrong-typed argument; `ValueError` for a
  right-typed `at` carrying an unusable value — an unrecognised mode name, a
  non-finite float, `at="max_finite_death"` with no finite death, or an
  `at=<float>` below some essential bar's birth. **(11)** §6.3: "bit-identical
  coordinates" in the `==`/`allclose` block was false, and is corrected to
  "compared without tolerance". `-0.0 == 0.0` in IEEE 754, so diagrams
  differing only in the sign of a zero are equal under `==` while their bytes
  differ — which is not a defect in `==`, whose IEEE semantics match §7's sort
  and every backend's own comparisons, but is the reason item (7)'s hash-side
  normalisation is required. Items (7) and (11) are the same fact stated from
  the two ends; only §8.1 previously carried it, leaving §6.3 asserting
  something §8.1 already contradicted. Items (9) through (11) were all found by
  reviewing `diagrams/core.py` against this document, entry 26's design-review
  pass run in the other direction. **(12)** §5: recorded a second
  considered-and-rejected alternative alongside the copy-forward one — having a
  substitution keep the *smaller* of the previously recorded
  `"finitized_at:<value>"` and the new one, on the intuition that the more
  aggressive finitization is worth remembering. It is unreachable (a finite
  substitution leaves no essential bar, so item (4)'s return-unchanged rule
  makes the next call a no-op in either direction), it would misdescribe the
  bars if it were reachable (a minimum keeps `"finitized_at:3.0"` on a diagram
  whose essential bars now die at `7.0`, item (10)'s failure with the numbers
  swapped), and it has no ordering to apply to the slot's three non-numeric
  values. Recorded rather than left implicit because the intuition behind it is
  correct — an earlier verdict should survive a later call — and this
  document's answer to it is item (5)'s `essential_bars_source`, a second key
  with a single writer rather than an ordering imposed on the first. No
  behaviour changes; nothing implemented this and nothing should.
- **2026-08-05 (33)** — Entry 32's items (9) through (11) continued. Normative changes:
  **(1)** §3.3 and §4.3: `DiagramBatch.__getitem__` is eager-only, and so is
  every batch operation routed through it — `canonical()`, `==`, `allclose`,
  `same_provenance`, `content_hash`. It is neither a filtering operation nor a
  `bool`-returning one, the two categories §3.3 already named, but its slice
  bounds are `int(offsets[i])` and `int(offsets[i + 1])`, which concretize a
  traced array for the same reason §5's essential-bar branch does.
  `b.canonical()` is stated outright because `d.canonical()` genuinely is
  traceable and the batch version looks like the same operation: the sort is
  shape-preserving at both levels and only one is available under `jax.jit`.
  This is item (9)'s shape-preserving-is-not-traceable conflation at a second
  site, which item (9) closed for `finitize` without noticing generalised.
  `len(b)` and `bar_counts` do not index and stay available. **(2)** Opened
  **D15** in §12.1: §8 reserves `provenance["order"]` with two values and names
  a writer for only one of them. §7 forbids adapters from sorting, so
  `"backend"` is all any `from_*` can record, and `d.canonical()` — the
  operation that makes `"canonical"` true — carries `meta` through unchanged,
  so a sorted diagram still reports `"backend"` and nothing writes the other
  value at all. §7 and §8 both gain a paragraph stating the gap and pointing at
  the row. It is `essential_bars`' then-versus-now problem at a key nobody
  noticed it applied to, but it does not resolve the same way — `order` has no
  adapter-time verdict worth a second key — so it is opened rather than
  answered by analogy. `core.py` leaves the key untouched pending the call.
  **(3)** §4.2: `len(...)` of an array is shorthand for `shape[0]` and MUST be
  implemented as such; §4.2's `__len__` snippet and §4.3's batch total, which
  read `len(self.offsets)` and `len(b.dims)`, are corrected. The array API
  standard does not require an array object to implement `__len__`, so the
  snippets were NumPy habits inside the section arguing against exactly that;
  `core.py` had already been reading `.shape[0]` throughout, making this the
  one item in this entry where the document, not the implementation, was the
  thing that was wrong. I1, B1 and B3 keep their `len(...)` phrasing, now
  covered by the stated shorthand. Non-normative in `core.py`, also from this
  pass and recorded here only so the review's scope is on file: an unstated
  `ValueError` on comparing two diagrams backed by different array namespaces
  (§6.3 is silent, `xp.equal` would otherwise raise whatever the backend
  raises, and returning `False` would assert bars differ that may not — the
  argument §4.2 already makes for `from_diagrams`); the same `int(...)` reads
  by which `finitize(at="max_finite_death")` can reach I6, when the largest
  finite death falls below an essential bar's birth, added to §5's enumeration
  of what raises `ValueError`; and two elementwise calls narrowed from
  `xp.equal(a, scalar)` to `a == scalar`, scalar arguments to two-array
  elementwise functions being a 2024.12 addition while the operator form has
  been guaranteed since 2021.12.
- **2026-08-05 (34)** — Readability pass on the main RFC, the sixth of the
  kind entries 13, 14, 15, 16 and 17 record, and the first to treat the
  changelog itself as the primary offender: entries 32 and 33 had each grown
  to roughly two thousand words in a single bullet, which is a history
  document's job done in the wrong file. Entries 23 through 33 are now
  reproduced above in full — they had never been mirrored here at all, so the
  main RFC was the only copy — and the main RFC's Appendix B carries a single
  bullet each. ("One line each", as this entry originally read, was an
  overstatement: entries 26, 32 and 33 still run several sentences apiece. The
  condensation was real — entry 32 went from roughly two thousand words to
  about two hundred and fifty — but the summary claimed a finish line it did
  not reach. Corrected in entry 35, in both files.) Four further relocations, each replaced in the main RFC by its
  conclusion plus a pointer: §5's "keep the smaller of the two recorded
  values" rejected alternative (below, reproduced in full — it is a design
  that was never reachable and never implemented, which is exactly the
  category entry 14 established belongs here); §6.1's account of what an
  earlier revision of I2 said about `xp.isdtype`; §6.3's account of what an
  earlier revision said about "bit-identical"; and §12's account of *why* D6,
  D9, D10 and D11 were removed, which this document already carried in full
  under its own heading below, meaning the main RFC was restating it a second
  time. Three single-sourcing fixes with nothing relocated: §4.3's restatement
  of the `__getitem__` eager-only argument now points at §3.3, which states it
  once and in full; §8's second D15 paragraph is dropped in favour of D15's
  own §12.1 cell, which already carried the same three options verbatim; and
  §4.1's "no duplicated logic" claim now points at §4.3 rather than
  re-deriving the accessor-by-accessor case for it. §10.1's argument
  paragraphs were reflowed and given code formatting — the one stretch of
  unwrapped, un-backticked prose left in the document — with no claim altered.
  Also dropped: D15's cell reference to "entry 32's items (9) through (11)",
  which became unresolvable the moment the entry it named was condensed.

  **Entry 26's item (6) is removed, on the lead's confirmation that the change
  it described was never made.** It had claimed §9.1 resolves the cost of
  essential-to-essential matching left open by "delegate on the finite parts
  only"; §9.1 still reads "delegate on the finite parts only, handle +inf bars
  internally, and combine responsibly", which is the unresolved version. The
  removal is recorded here rather than made silently because a changelog
  asserting work that was never done is the same clean-plausible-wrong signal
  §9 exists to catch, pointed at this document instead of at a backend — and
  because the false claim, left standing, would have told a future reader that
  a genuine open question in §9.1 had already been answered. No text in the
  main RFC changes: entry 26's condensed one-line summary there never carried
  item (6), and §9.1 itself is untouched. What §9.1 does about that cost
  remains unspecified and unrecorded as a decision; it is not opened as a
  D-number here, since that is the lead's call and not this pass's business.

  No requirement changed. The main RFC's body (§1 through §12 plus Appendix A,
  excluding the changelog) counts 92 MUST / MUST NOT / SHOULD / SHOULD NOT /
  MAY tokens before this pass and 91 after, and the single difference is not a
  requirement: it is §8's *quotation* of §7's "every consumer MUST treat row
  order as arbitrary", removed with the duplicated paragraph around it. §7
  still states it, and D15's cell still reaches the same conclusion from it.
  Verified by extracting and diffing every keyword-bearing sentence in the
  body, which shows each surviving requirement matched one-to-one; separately,
  all 27 I1-I9, B1-B7 and D-row table rows and all four Appendix A tables are
  byte-identical across the pass, save D15's dropped entry reference.

- **2026-08-05 (35)** — Review of `diagrams/core.py` against this document,
  run in the direction entry 33 established, plus the first pass to change
  `core.py` as well as the RFC. Normative changes: **(1)** §10.1 requirement 1
  gains a second clause, `load(dump(d)).same_provenance(d)`. The first clause
  is stated in terms of `==`, and §8 requires `meta` to take no part in `==`,
  so a `load` that discarded every byte of `params` and `provenance` satisfied
  requirement 1 completely. §5 nonetheless cites requirement 1 for the
  opposite — its argument against `finitize` copying `essential_bars` forward
  turns on a diagram arriving from `load` already carrying a `"finitized_*"`
  value, which is unreachable unless `load` preserves provenance. One of the
  two sections was wrong about the other; requirement 1 was the one missing a
  clause. §11.2 now tests both. **(2)** §8 requires `params` and `provenance`
  values to be JSON-representable. §10.2 stores both as UTF-8 JSON, and
  nothing previously constrained the `Mapping[str, Any]`, so a diagram holding
  a NumPy scalar or a dtype object under `source_dtype` satisfied §3.1 and §8
  and could not be saved — the type admitting instances requirement 1 cannot
  round-trip, with the failure surfacing at `save()`, arbitrarily far from the
  adapter that wrote the value. **(3)** §8 makes `DiagramMeta` enforce the
  `essential_bars` qualifier rules at construction. They were stated as
  obligations on writers, and `finitize` was the only writer honouring them;
  every `from_*` adapter (§11) sets the same keys through the constructor and
  none passes through `finitize`'s code path, so the rule with the most
  writers had the fewest checks. This is §3.1's "an invalid instance MUST NOT
  be constructible" applied to the one part of the type it had not reached.
  Deliberately narrow: the two stated rules only, since §8 reserves names
  inside an open mapping rather than closing it. **(4)** §6.3 makes the
  cross-namespace `ValueError` normative, at both levels and in both methods.
  Entry 33 recorded it as a non-normative `core.py` addition on the grounds
  that §6.3 was silent; a behaviour that only the implementation carries is
  one the next implementer has no reason to reproduce, and this one is
  observable to every caller. §6.3 now also records the third alternative the
  original reasoning missed — returning `NotImplemented`, which is what Python
  itself would reach for — and states what raising costs, namely that `==` is
  no longer total and `d in [d1, d2]` raises rather than answering. The
  behaviour is unchanged; what changed is that the cost is now written down
  where a reader meets the method. **(5)** §11.2 requires `content_hash` to be
  tested against §8.1's byte layout across **both** paths an implementation
  may take to produce it. This is the finding with the widest blast radius and
  the least visible cause: an implementation that reinterprets a backend's
  buffer where one exists and falls back to per-element packing where it does
  not has two paths that must agree, and `array_api_strict` exposes no buffer,
  so a suite exercising the hash only under the conformance backend tests the
  fallback and never the path every NumPy-backed diagram takes. `core.py` had
  exactly this shape and `content_hash` had no test of any kind, so the value
  a paper pins was computed by a code path CI structurally could not reach.

  Corrections, none of them changing a requirement: §3.3's "Two limits" is
  three — entries 32 and 33 each added a bolded limit paragraph without
  touching the count, which is precisely the error entry 20 spent a pass
  fixing, recurring at the same sentence. §8's reserved-key table was broken
  markdown: a five-column header with three unnamed columns, and
  `essential_bars`' enum split across them by unescaped pipes, so the
  normative key table rendered as garbage in any viewer. Appendix A.1's bare
  `TODO` is replaced by a statement of what the table does not measure — it
  varies `infinity_values` and holds `reduced_homology` fixed, so it rules the
  first out and leaves the second an inference — and why the row is blocked,
  giotto being unrunnable per §9.2, which makes it a frozen-fixture capture
  under §11.2 rather than a live run.

  Opened **D16**: I7, B5 and §4.2's `from_diagrams` check are written as `is`
  on `__array_namespace__()`, and the standard does not require that method to
  return the same object on every call. It is §4.2's `len(...)`-versus-
  `shape[0]` finding one method over — a NumPy habit inside the section
  arguing against exactly that, invisible because NumPy and `array_api_strict`
  both return the module itself. Unlike that one it cannot be fixed by
  rewording, the standard offering nothing to compare namespaces with, so it
  is opened rather than answered. §12's count moves to twelve, five open.

  Entry 34's "one line each" is corrected to "a single bullet each" in both
  files; see the parenthetical in that entry.

  **Changes to `diagrams/core.py` in the same pass**, recorded here because
  this document is where the review's scope belongs on file. Behavioural:
  `DiagramBatch.__getitem__` normalised a negative index before the bounds
  check and then raised with the normalised value, so `batch[-5]` on a
  length-2 batch reported `IndexError(-3)` — an index the caller never passed;
  it now names the original index and the batch length. `DiagramMeta` set no
  `__hash__`, so `@dataclass(frozen=True, eq=True)` generated one: the class
  satisfied `isinstance(m, Hashable)` and then raised `TypeError: unhashable
  type: 'dict'` from inside the generated tuple hash, naming neither the class
  nor the field: `__hash__ = None` makes the documented intent true.
  `finitize(at=True)` was accepted and substituted a death time of `1.0`,
  `bool` being an `int` and satisfying the numeric-protocol gate that exists
  to reject `b"2.0"`; it now raises `TypeError`, the same non-numeric-literal
  hole one type over. Non-behavioural: `DiagramBatch` now reads `offsets` into
  Python ints once and caches them, where `__getitem__` previously read two
  elements per call and five batch operations each drive a loop over every
  diagram — 2N device synchronisations per operation on a torch- or
  JAX-backed batch, unbounded, and the same cost `_big_endian_block`'s fast
  path exists to avoid one level down where it is bounded. Validation reuses
  that read for B2-B4. The six §3.1 error messages now name the offending
  value; §3.1 anticipates I6 violations at the 1e-16 level, where the
  magnitude is what decides between an adapter clamp and a backend bug, and
  the validator was withholding it. Docstrings: the module docstring's four
  paragraphs re-deriving §3.3's eager-only argument, `canonical()`'s paragraph
  re-deriving D15, and `finitize`'s re-derivations of §5 are replaced by
  pointers, leaving one copy of each fact in the document that owns it. The
  same fact had five drift sites; §3.2's "one place to update beats two that
  can drift apart" is the principle, and the implementation had been violating
  it against the RFC itself. Tests added:
  `tests/test_rfc0001_content_hash.py` (§8.1/§8.2, including the two-path
  agreement, which deliberately does not `importorskip` anything so it runs in
  the default environment) and `tests/test_rfc0001_diagram_contract.py` (the
  enforcement points above). Also fixed: four stale `§3.4` references in
  `tests/` and `pyproject.toml`, left over from entry 11's renumbering.

  **Packaging: `pyproject.toml` was the side that was wrong, confirmed by the
  lead and corrected.** It declared `numpy>=2.0` in `dependencies` under a
  header reading "DEFAULT CLOSURE — permissive only. numpy and nothing else",
  against §10.1 requirement 2 and §3.3, which both say in MUST language that
  the default install carries no third-party dependency at all and that
  `numpy` is a lazy, function-scoped import inside `save`/`load` only. The
  declaration was a survival from the closure as it stood before entry 7's
  correction, which is the entry that removed `numpy` from the default install
  in this document and which `pyproject.toml` never followed. No RFC text
  changed: the specification was already right, and this is the implementation
  catching up to it. `dependencies` is now empty, `numpy>=2.0` moved to
  `akriti[test]` as the only place it is declared, and `pip install .` into a
  clean venv fetches exactly one distribution — `akriti` itself — with
  `import akriti.diagrams` succeeding and `numpy` absent from `sys.modules`.

  Three supporting changes, since a closure claim that nothing checks is the
  kind that decays: `tools/check_license_closure.py` now fails the `default`
  profile on a *non-empty* closure rather than only on a non-permissive one,
  the two having stopped being the same test the moment numpy left — a
  permissive but non-empty default would otherwise pass every check the tool
  ran while being exactly what the policy forbids. `tests/test_smoke.py` gains
  a check that importing `akriti` and `akriti.diagrams` does not pull in
  `numpy`, run in a subprocess because almost every other module in the suite
  imports it and an in-process check would pass for the wrong reason precisely
  when it mattered. `DEPENDENCIES.md`, `README.md` and the package docstring
  are corrected; the default-closure table now has no rows.

  The version floor is the one thing that had nowhere to go, which is the
  question the removed D6 raised and left open. A floor constrains a declared
  dependency and there is no longer one to constrain, so it is recorded in
  `DEPENDENCIES.md` as a statement about the caller's environment instead:
  array-API code paths need a caller-supplied `numpy>=2.0` or another
  array-API-native library, and an older numpy fails at the caller's own
  `__array_namespace__()` call rather than inside akriti. That is the
  supported-baseline reframing D6's note proposed, now that the packaging
  decision it was waiting on has actually been made.

- **2026-08-05 (36)** — Linted §12.1 and §12.2, and reordered D15 correctly before D16. No normative content changed.
- **2026-08-05 (37)** — Opened **D17**, found by auditing the main RFC's
  RFC-2119 keyword use rather than by review against `diagrams/core.py`, which
  is what entries 33 and 35 ran. The audit was prompted by a question about
  where the keyword definitions ought to live and turned up a distribution
  worth recording: 97 uppercase MUST against 12 lowercase "must", 2 SHOULD
  against 11 "should", 3 MAY against 11 "may", and 5 lowercase "required".
  Nearly all of the lowercase uses are a settled and good idiom — an
  adjectival lede naming the shape of a rule, followed by an all-caps modal
  carrying the obligation, as at §4.2's `xp`-for-empty-batches, §11's
  `reduced_homology`, and §8's own "All fields are optional … but `from_*`
  adapters MUST populate". Exactly one is not: §8's `DiagramMeta` block
  comments `coeff_field` with "affects the diagram, must be recorded", an
  obligation with no uppercase counterpart anywhere in the document.

  Three facts make it a decision rather than a typo. `coeff_field` occurs
  exactly once in 1731 lines, in that comment — no section, no MUST clause and
  no test requirement mentions it. The prose seven lines below it says the
  opposite, "All fields are optional", and the MUST-populate list names three
  fields that deliberately exclude it. §8.1's `content_hash` covers bars and
  never metadata, so nothing downstream depends on the value being present;
  `core.py` implements `coeff_field: int | None = None`, having followed the
  prose. The comment is therefore a requirement that no clause states, no test
  can check, and the implementation does not honour — but its claim is
  correct, since homology over ℤ/2 and ℤ/3 genuinely differ where there is
  torsion, which is the same criterion §8's opening sentence uses to justify
  recording `filtration` at all.

  What blocks resolving it here is that the three fields §8 does require are
  all derivable from the adapter itself, and this one is not. §11's five
  adapters take a computed result plus `**meta`, never the call that produced
  it, so whether an adapter can record a coefficient field depends on whether
  the backend's returned object carries it — a per-backend fact this RFC does
  not state, Appendix A does not measure, and `rfcs/evidence/probe_backends.py`
  does not probe. Two adapters are out of reach regardless: `from_array` has
  no backend, and `from_persim` consumes diagrams rather than computing them.
  If the value turns out to be unrecoverable from the returned objects, the
  only remaining route is the one §5.1 took for `reduced_homology` — put it in
  the signature and make omission raise — which is a signature change on up to
  three adapters and the lead's call, not this document's.

  §8 is left untouched on purpose. Both available edits, adding `coeff_field`
  to the MUST-populate list or striking the comment's normative clause, are
  the two answers to D17, so making either one pre-empts the decision; this
  follows D15 and D16, which likewise left the text as it stood. §12's count
  moves to thirteen, six open.

  One matter the audit raised and this pass did not act on: the keyword line
  at the top of the RFC cites RFC 2119 alone, without RFC 8174's "when, and
  only when, they appear in all capitals". BCP 14 has consisted of both since
  2017, and it is the caps rule that makes the lowercase idiom above formally
  non-normative rather than merely conventional — the distinction this entry's
  finding turns on, and the one entry 15 recorded hitting from the other
  direction when a bare "MUST" in a changelog line tripped the MUST-count
  check. Left for a separate pass, since it is a change to the document's
  interpretive frame rather than to a decision inside it.
- **2026-08-05 (38)** — The separate pass entry 37 asked for. The main RFC's
  keyword line cited RFC 2119 alone and said only that the five keywords "are
  to be interpreted as in" it. It now cites BCP 14 — RFC 2119 together with
  RFC 8174, which has been the other half of BCP 14 since 2017 — and carries
  8174's operative clause, that the keywords have their special meaning "when,
  and only when, they appear in all capitals". It also states that the
  remaining six BCP 14 keywords are deliberately unused, and why: SHALL is a
  bare synonym for MUST with no grammatical niche of its own, and REQUIRED,
  RECOMMENDED and OPTIONAL collide with three vocabularies already live in
  this document — Python parameter semantics (§11's required keyword-only
  arguments), dependency optionality (§10.1, §10.3), and §12's "No
  recommendation" decision status. Capitalising them would make every such use
  ambiguous between its ordinary sense and a normative one, and would also
  break the MUST-count verification this process relies on, which counts one
  token rather than three.

  **This entry is not normative-content-neutral**, in the same way entry 20
  was not: no requirement changed, but the rule for reading the document did.
  Before it, the body's 12 lowercase "must", 11 "should" and 11 "may" were
  non-normative by convention and by a reader's charity; after it they are
  non-normative by the document's own terms.

  Because that demotes text wholesale, the lowercase uses were audited before
  the line was changed rather than after. Nearly all of them are the same
  deliberate idiom — an adjectival or explanatory lede naming the shape of a
  rule, with an all-caps modal carrying the obligation nearby — and four
  candidates that read most like orphaned requirements were checked
  individually. §3.3's "a topological layer … must operate on the full arrays
  with a mask" and "*every* mode must decide whether the diagram has any
  essential bar" are both descriptive, of the caller's situation and of why
  the eager-only rule holds, with §3.3's "MUST be documented as such" carrying
  the requirement. §4.2's "row ranges must not overlap or invert" is the B4
  table's Rationale cell, the invariant column beside it being the normative
  one. §11's "omitting it must be a `TypeError` at the call site" looked like
  the second orphan but is not: it is a consequence of the keyword-only
  signature §11 already specifies normatively, and §5.1's "Omitting it MUST
  raise" carries the obligation — the `TypeError` at §5 is `finitize`'s `at`
  argument and unrelated. §8's `coeff_field` comment is the only genuine
  exception, which is D17; entry 37's row is updated to say that the caps rule
  settles the comment's normative status without touching the question it
  raises.

  One case is left deliberately unresolved rather than audited clean. §3.1's
  I8 note says the no-mutation rule "should be enforced the same way
  `DiagramMeta` already is (`@dataclass(frozen=True)`, §8), or documented as
  an equivalent guarantee if the array API standard's read-only view support
  is used instead". The MUST beside it carries the substance — every
  mutation-shaped method constructs and returns a new `PersistenceDiagram` —
  and what the lowercase "should" governs is the enforcement mechanism, for
  which the sentence already offers an explicit alternative. That is the shape
  of a SHOULD, and promoting it would be a normative change on its own merits
  rather than a consequence of this one, so it is recorded here and left as
  written.
- **2026-08-05 (39)** — Takes the decision entry 38 deferred, on the lead's
  instruction, and makes §3.1's I8 note the first SHOULD this document has
  ever contained. Before it, the body carried 93 MUST and MUST NOT and two MAY
  (§2's `death` MAY be `+inf`, §10.1's lazy-import exception); SHOULD and
  SHOULD NOT were declared on the keyword line and used nowhere, so the
  keyword line promised a vocabulary two-fifths of which the document never
  exercised.

  This is the right first one because it is the case the keyword exists for.
  The MUST beside it fixes the behaviour — every mutation-shaped method
  constructs and returns a new `PersistenceDiagram` — and what the old
  lowercase "should" governed was the enforcement *mechanism*, for which the
  sentence already named a legitimate alternative in its own second half. A
  requirement whose author can name the circumstance that justifies deviating,
  and does not want to close it off, is a SHOULD rather than a MUST that was
  hedged.

  The sentence is also restructured rather than merely capitalised, and that
  half is a change entry 38 did not anticipate. As written, "or documented as
  an equivalent guarantee if the array API standard's read-only view support
  is used instead" was plainly obligatory in substance while being lowercase
  prose — which entry 38's caps-only rule had just demoted to carrying no
  force at all. Capitalising only the first clause would have produced a
  SHOULD with an unenforced escape hatch: an implementation could take the
  read-only-views route and document nothing, and the document would have no
  words to object with. The clause is now a MUST attaching to the deviation.
  This follows entry 35's reasoning on the cross-namespace `ValueError` —
  a behaviour the next implementer has no reason to reproduce is one that
  needs to be written down where they will meet it — and it is what makes the
  deviation reviewable, documentation being checkable in a way that RFC 2119's
  "carefully weighed" is not.

  §12 is unaffected; no decision opened or closed.

---

## Original "Note on Dx" text

These paragraphs previously appeared directly below the §12 table in the main
RFC (D1, D2, D6, D8), or, for D9, inline in the table cell itself until the
2026-08-02 (16) pass. They are reproduced here verbatim for the audit trail;
the current §12 table cells for D1, D2, D6, D8, and D9 are the condensed
replacements. D12 (entry 21) is a further exception to the pattern this
section's name describes: it never had a location in the main RFC at all,
below-table or inline, the below-table convention was already retired by
the time it was opened, so its full reasoning was written directly here
from the outset, the same way D9's was once the table-cell location stopped
working for it.

**Note on D1.** This row previously recommended Parquet. That directly
contradicted §10.1, which rules Parquet out by name for failing the
dependency-free requirement, and §10.2, which normatively specifies `.akd`.
The contradiction predates the batch-design discussion below; it looks like
D1 was written before §10 was settled and never reconciled afterward. §10 is
MUST-level and D1 was framed as still-open, so the table has been corrected
to point at §10 rather than restate a decision.

**Note on D2.** Two alternatives to the two-type design were raised and
rejected. §4.1 records why a single type using dense, padded batch storage is
not adopted (Appendix A.2). §4.2 records why a single type using shared,
CSR-backed storage (batch-of-one as `offsets = [0, n]`) is also not adopted,
despite avoiding the padding problem, and specifies `DiagramBatch`'s own
storage as a concatenated buffer with `offsets`, matching §10.2, with
`__getitem__` returning a zero-copy view rather than an independently
allocated object.

**Note on D6.** This row is written in packaging language — "raise the
floor," "add `array-api-compat`... behind `[torch]`" — that assumes a
requirements file with version constraints to set. It predates D10, which
was added later (Appendix B) and settled on zero declared dependencies in
the default install, not even `numpy` behind an extra, only a lazy runtime
import with no pinned version at all. There is currently nowhere to declare
a `numpy>=2.0` floor. D6 needs to be either reframed as a *supported-baseline*
statement — e.g. documented as "array-API-generic code paths require a
caller-supplied `numpy>=2.0` or another array-API-native library; older
`numpy` fails at the caller's own `__array_namespace__()` call, not inside
`akriti`" — or the lead needs to decide a formal floor belongs somewhere
after all, which D10 as written does not provide for. Not resolved here.

**Note on D8.** Applying this still rests on D9's premise — that Apache 2.0
sits on equal footing with the extras' existing MIT/GPLv3-transitively
licenses — which D11 (above) flags as unverified against the actual
onboarding document. If the onboarding document turns out to still read
"MIT/BSD-only," that equal-footing argument needs to be re-examined, not
just relabeled. Applying D8 now doesn't discharge D11; it means
`to_parquet()` is specified and ready to implement while that check is
still outstanding.

**Note on D9.** Resolved by the onboarding document's 2026-07-30 revision,
which retracted "MIT/BSD-only": `ripser`/`persim` are MIT but pull in
`hopcroftkarp` (GPLv3) transitively, and the `gudhi` wheel bundles
GPL-marked CGAL modules with no license metadata of its own. GPLv3 arrives
regardless of a backend's own license, so license family was never the real
axis. Actual policy: zero third-party dependencies in the default install,
any license behind a labeled extra, enforced by
`tools/check_license_closure.py`. This resolution is what D8's Apache-2.0
`pyarrow` extra depends on, and it is exactly what D11 flags as not yet
independently checked against the onboarding document itself — this note
records the reasoning as currently believed, not as confirmed.

**Note on D12.** §10.1's 2026-08-02 rewrite tests `.npz` against HDF5 and
Parquet on all five requirements and concludes the format choice follows
from requirement 5 (inspectability), with numpy entering only as a
consequence of that. That test was never run against the two candidates
that would satisfy requirement 2 outright, with no lazy-import exception
needed at all: stdlib `csv`/`tsv` and stdlib `sqlite3`.

Requirement 1 (exact round-trip) does not discriminate. Python's `repr()`
for floats has round-tripped exactly since 3.1, and `float("inf")` parses
the literal token, so a `dim,birth,death` CSV is losslessly round-trippable
with `csv` and `float()` alone. `sqlite3`'s `REAL` columns store the raw
8-byte IEEE-754 double, so round-trip should hold there too, though this
has not been measured against a real SQLite version the way §6.2 measures
Ripser's precision, and it should be before this is relied on.

Requirement 3 (self-describing) favors SQLite over both CSV and `.npz`:
its schema is inspectable via `.schema` without any extra layer. All three
still need a JSON blob for `params`/`provenance`, arbitrary
`Mapping[str, Any]` doesn't fit a fixed schema or a flat CSV cleanly, so
the win is partial.

Requirement 4 (byte-determinism) is where the two candidates diverge most.
CSV is the easiest of any candidate to reason about: no compression flags,
no binary padding, determinism reduces to fixed float formatting, canonical
row order (already specified by §7), and a fixed line terminator. SQLite is
the hardest: file layout depends on page allocation, freelist state, and
the compiled library version, none of which are guaranteed stable across
writes without explicit control (fixed `page_size`, a `VACUUM`, a single
transaction). This is a real engineering cost specific to SQLite, not
shared by CSV.

Requirement 5 (readable without the library) is where CSV wins outright.
Literal text, `cat`-able, diffable in git, pastable into a spreadsheet, a
more direct claim than `.npz` itself satisfies, since `.npz` still needs
`numpy.load` or a binary-aware tool to interpret. SQLite satisfies a
related but weaker claim: not text-editor-readable, but queryable with
near-universal tooling (the `sqlite3` CLI, any of several free browsers),
arguably a *better* fit for §8's provenance-auditing goal than either CSV
or `.npz`, since it supports queries like counting essential bars directly,
but this is "inspectable with common tools," not the "unzip and read"
standard §10.1 was actually written against.

Both alternatives require giving up §4.2's CSR-buffer-on-disk layout for a
`diagram_id`-per-row scheme, the shape `to_parquet()` (§10.3) already uses,
so resolving D12 toward either one is not a §10.1 edit alone: it also
touches §4.2's storage specification and §10.2's format section directly,
and it revisits D10's "numpy is close to universal in this ecosystem"
justification, since that justification stops mattering if the default
format no longer needs numpy at all. One correctness trap specific to
SQLite is worth naming now regardless of how D12 resolves: a schema with a
`UNIQUE` constraint on `(dim, birth, death)` is an easy default to reach
for and would silently deduplicate bars, violating §2's "multiset, not
set" definition, the exact category of silent-wrongness bug §9 exists to
catch.

No recommendation here, matching D7's precedent for a real, non-stylistic
gap: this turns on a per-diagram and per-batch bar-count
distribution the document does not state a figure for, and that figure is
what decides whether CSV's size and parse-speed cost at scale, or SQLite's
determinism cost, is acceptable against the win of a fully dependency-free
default install. Needs the lead's call, and needs that number in hand
before the call can be made.

---

## Body narrative relocated in the 2026-08-02 pass

These four passages previously appeared in full in the main RFC's body.
Each is reproduced here verbatim; the current body text is the condensed
replacement (conclusion plus a pointer to this section).

**§3, on `Array` not being `np.ndarray`.**

> This was `np.ndarray` in the first draft, which was wrong. The onboarding
> document requires `core/` to be written against the array API rather than
> hard-coding NumPy, and `PersistenceDiagram` is the input to every function in
> `core/`. A container that pins NumPy makes a framework-agnostic `core/`
> unachievable no matter how `core/` itself is written, and retrofitting the
> container later is exactly the expensive case that requirement exists to avoid.
> §3.3 states what this does and does not promise.

**§3, on `xp` as a derived property rather than a stored field.**

> `xp` is a derived **property**, not a fourth stored field, deliberately. An
> earlier draft stored it alongside `dims`/`births`/`deaths`, which creates a
> fourth piece of state that has to be kept in sync with the other three at
> every construction site, including the views `DiagramBatch.__getitem__`
> returns (§4.2), with nothing enforcing the agreement. I7 already requires
> `dims`, `births`, and `deaths` to share one namespace; deriving `xp` from
> `dims` makes disagreement structurally impossible rather than merely
> prohibited. Call sites that want the short spelling get `d.xp`; nothing
> about validity depends on a value that could drift.

**§4.1, "Reconciling the two-type design," in full.**

> This section exists because an earlier design pass considered folding
> `PersistenceDiagram` and `DiagramBatch` into a single type, batch-of-one by
> default, with the batch realized as a dense `(batch, max_points, 2)` array plus
> a boolean mask. That proposal is not adopted, and the reason is Appendix A.2,
> not taste. A dense padded batch cannot represent giotto's own output without
> the padding rows becoming indistinguishable from genuine trivial bars, and the
> measured consequence, a diagram changing shape depending on what else is in the
> batch, is exactly the class of bug this document exists to prevent. So
> `DiagramBatch` is a ragged sequence, and it stays that way at the interchange
> boundary.
>
> The dense, padded representation is not wrong, only misplaced. The paragraph
> above already says a function needing a rectangular buffer must build it
> internally and return an explicit mask alongside. That is the padding+mask
> scheme, deliberately scoped to computation rather than storage. It is the right
> shape for a `castle/` routine feeding an array-API vectorized op, or a
> topological layer inside a network; it is the wrong shape for the type that
> `save`, `load`, and every adapter hand back.
>
> Nor does the two-type split reintroduce duplicated implementation.
> `DiagramBatch` owns no numerical or invariant logic of its own: `dim(k)`,
> `persistence`, equality, and invariants I1 through I7 are all written once,
> against `PersistenceDiagram`, and `DiagramBatch.__getitem__` returns a
> `PersistenceDiagram`, not a different type. A batch of one diagram is not a
> special case anywhere in `core/`; it is a `DiagramBatch` of length one,
> wrapping the same object every other code path uses. §4.2 specifies how that
> wrapping is implemented in memory.

**§4.2, "Why not go further and merge," in full, including the PyG precedent.**

> **Why not go further and merge `PersistenceDiagram` and `DiagramBatch` into one
> CSR-backed type**, with `offsets = [0, n]` as the single-diagram case? This was
> considered, since it satisfies onboarding §9.3's leading-batch-dimension rule
> more literally than a two-type split does, and it was rejected on two points
> specific to this RFC, not on API-surface taste:
>
> - **`DiagramMeta` (§8) is genuinely per-diagram.** `backend`,
>   `backend_version`, `params`, and `provenance` can all differ across a batch.
>   A merged type forces `meta` to become a sequence the moment `offsets` has
>   more than one row, so every consumer of `.meta` must branch on batch size.
>   Keeping `meta` a single dataclass, true only on the unbatched type, is a
>   property worth keeping.
> - **`content_hash` (§8.1) is defined over one diagram's canonical bars.** A
>   merged type has to either forbid calling it on a multi-diagram instance or
>   redefine it as a hash-of-hashes, and either choice is new specification this
>   RFC does not otherwise need.
>
> Precedent: PyTorch Geometric solves the identical problem, ragged, per-item
> structure needing efficient batched storage, with two types: `Data` for one
> graph, `Batch` for many, concatenated storage plus an index vector, rather
> than one self-batching type. `Batch` subclasses `Data` in PyG; we do not adopt
> that inheritance here, since a subclass that fails to override even one
> accessor (`persistence`, `==`) would silently compute it across batch
> boundaries instead of raising, which is precisely the "clean, plausible, wrong
> answer" failure category §9 exists to rule out. `DiagramBatch` and
> `PersistenceDiagram` are related by composition (the view relationship above),
> not inheritance.

**§9.1, the "Correction, 2026-07-30" note, in full.**

> **Correction, 2026-07-30.** The first draft of this section asserted that
> persim gave no warning at all. That was wrong: the measurement behind it was
> taken with warnings globally suppressed. The evidence script no longer
> suppresses warnings, and `tests/test_rfc0001_backend_claims.py` now asserts
> that the warning *is* raised, so this cannot drift again.

---

## Body narrative relocated in the 2026-08-05 pass (entry 34)

Each passage is reproduced verbatim; the current main-RFC text is the
condensed replacement, conclusion plus a pointer to here.

**§5, the "keep the smaller recorded value" alternative, in full.**

> **Also considered and rejected: having a substitution keep the smaller of the
> previously recorded `"finitized_at:<value>"` and the new one**, on the
> intuition that the more aggressive finitization is the one worth remembering.
> It is the copy-forward idea aimed at the state slot rather than the history
> slot, and it fails three times over.
>
> - **It is unreachable.** A finite substitution leaves no essential bar
>   behind — every `+inf` death becomes `at`, so `essential` is empty
>   afterwards and the return-unchanged rule above makes the next `finitize`
>   a no-op whichever direction its `at` points. `at="drop"` removes the bars
>   and `at="max_finite_death"` substitutes a finite death, so neither reaches
>   a second substitution either. There are never two values to compare.
> - **It would misdescribe the bars if it were reachable.** `essential_bars`
>   is one slot and §8 requires it to describe the diagram's current state. A
>   minimum keeps `"finitized_at:3.0"` on a diagram whose essential bars now
>   all die at `7.0` — a record naming a value no bar carries, the same
>   clean-plausible-wrong signal this section already rules out for
>   `"finitized_dropped"` with a count of zero and for the `at=+inf` case
>   above.
> - **It has no ordering to apply.** The slot's other legal values are
>   `"faithful"`, `"lost_upstream"`, and `"finitized_dropped"`. None of them
>   is greater or less than a float, so the rule would fall back to plain
>   overwrite for three of the four cases and buy a special case for the
>   fourth.

**§6.1, on what an earlier revision of I2 said.**

> An earlier revision of I2 and of this paragraph offered
> `xp.isdtype(a.dtype, "real floating")` as an equally acceptable form. It is
> not [...]. Both this paragraph and I2's own table row now say equality, and
> the two no longer disagree about it.

**§6.3, on what an earlier revision of the `==`/`allclose` block said.**

> An earlier revision of the block above said bit-identical, and that is false
> in one reachable case [...]. Without that normalisation, `d1 == d2` with
> differing `content_hash`es is reachable.

The reachability claim itself was not dropped from the main RFC, only its
second copy: §8.1 states it in full as the reason for the normalisation, and
§6.3 now points there instead of asserting it a second time.

---

## What did not move here

Sections 1, 2, 5 through 11 (less the §9.1 correction note above, and less
one scoping sentence added to §10.1 in the fourth pass, entry 21), and the
D3/D4/D5/D7/D10/D11 table cells in §12 stayed in the main document unchanged
in substance across all four passes. D9's cell was trimmed in the third
pass (2026-08-02, entry 16): its conclusion, status, and the still-live
"not yet independently verified — see D11" flag all stayed, only the
supporting license detail moved here. D12, added in the fourth pass, is not
something that moved out of §12 the way D1/D2/D6/D8/D9's notes did; it has
no prior location in the main RFC to have moved from, per the note above.
What moved was always either
meta-content about the RFC's own process (changelog prose, superseded
recommendations, first-draft corrections, designs considered and rejected)
or supporting detail for a conclusion the body already states in full.
Measured evidence (Appendix A), invariants (I1-I8, B1-B5), and every MUST /
MUST NOT / SHOULD requirement stayed in the main document, in full, across
every pass — that content is what the RFC exists to carry, and condensing
it would be cutting the document's purpose rather than its prose.

---

## D6, D9, D10, D11 — removed from RFC-0001 scope (2026-08-03)

Design review concluded all four were dependency-and-licensing policy
questions, not persistence-diagram-interchange questions, and moved them out
of RFC-0001 entirely rather than resolving them here. This section preserves
the §12 table rows verbatim as they stood immediately before removal, for the
audit trail. Their below-table "Note on Dx" prose (D6, D9, above) was already
here; D10 and D11 never had a separate note, their full reasoning lived
entirely in the table cell, so the cell text is the whole record for those
two.

| # | Question | Recommendation / status (at time of removal) |
|---|---|---|
| D6 | Array-API support (§3.3) needs a NumPy that has it. Raise the floor to `numpy>=2.0`, or add `array-api-compat` and keep `numpy>=1.24`? | Unresolved, and stale. Its "raise the floor" recommendation assumed a requirements file to pin it in; D10 later removed that (zero declared dependencies by default). Reframe as a supported-baseline statement, or the lead decides a floor belongs somewhere after all. |
| D9 | §10.1's dependency-free requirement is read as "MIT/BSD-only" elsewhere in the project (onboarding). Akriti's own outbound license is Apache 2.0. Should the dependency closure be MIT/BSD-only, or MIT/BSD/Apache-2.0-only? | Resolved, provisionally — the onboarding document's 2026-07-30 revision was believed to have retracted "MIT/BSD-only" once GPLv3 was found arriving transitively regardless of a backend's own license. Actual policy as understood: zero third-party dependencies by default, any license behind a labeled extra, enforced by `tools/check_license_closure.py`. Not independently verified against the onboarding document itself — see D11. |
| D10 | The default install is solely the interchange layer (§3.3/§10.1), zero third-party dependencies, `numpy` included. But `io.py`'s `save`/`load` genuinely needs `numpy` for `.npz`. Should that be a lazy, function-scoped import with a friendly `ImportError` if missing, or a formal extra (e.g. `akriti[io]`)? | Lazy import, not a formal extra. A friendly, actionable `ImportError` inside `save`/`load` costs nothing to implement and keeps `pip install akriti` genuinely dependency-free while not forcing serialization users through an extra for a package as close to universal in this ecosystem as `numpy`. `tools/check_license_closure.py` should still assert the *default* venv has zero third-party imports at `diagrams/core.py` / `diagrams/adapters.py` import time. |
| D11 | D9 was marked resolved on the premise that the onboarding document's 2026-07-30 revision retracted "MIT/BSD-only" for "zero dependencies by default, any license behind a labeled extra." That retraction was never independently checked against the onboarding document itself — this RFC was the only place that claimed it happened. If the onboarding document still reads "MIT/BSD-only... copyleft-dependent backends go behind extras," D9 was not actually resolved, and D8's Apache-2.0 `pyarrow` extra reasoning needed re-justifying under the older, narrower policy. | Never confirmed before this section was removed. Whoever owns the onboarding document's actual policy should re-derive D9's conclusion (and D8's dependence on it, since D8 stayed in RFC-0001) from whatever that policy actually says, independent of anything RFC-0001 assumed. |

**Why removal rather than resolution.** RFC-0001's own stated purpose (§1) is
a canonical diagram type, an on-disk format, and an adapter contract. None of
D6/D9/D10/D11 bear on any of those three; they bear on how `pyproject.toml`
declares dependencies and what license policy the project enforces at
install time, both owned by the onboarding document. Keeping them here meant
this RFC's own "ready for public comment" status was gated on a licensing
question (D11) about a *different* document that RFC-0001 had asserted, not
verified, the exact category of claim §1 and §9 exist to catch when a backend
makes it. The normative content these rows were protecting is untouched:
§3.3 and §10.1 already state the zero-dependency-by-default requirement and
`numpy`'s lazy-import behavior directly, in MUST language, and D8 (kept, see
RFC-0001 §12.2) no longer cites D9 or D11 for its own justification.
