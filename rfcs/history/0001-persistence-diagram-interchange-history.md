# RFC-0001 — Process History

Non-normative. This document holds the full narrative that the main RFC
(`0001-persistence-diagram-interchange.md`) now only points to. Five passes
have relocated or added material here: the original, un-condensed changelog
entries and the "Note on Dx" explanations that used to sit below the §12
decision table (2026-07-31); process narrative from §3, §4.1, §4.2, and
§9.1 — first-draft corrections, superseded designs, and precedent
discussion (2026-08-02); a second §12 trim, D6/D8/D9's fuller reasoning,
including a new Note on D9 that has no earlier counterpart (2026-08-02);
a new Note on D12, authored directly here rather than relocated from
anywhere in the main RFC, alongside the corresponding full changelog entry
(2026-08-02); and the fuller reasoning behind entry 22's requirement-4
rationale and §7 rewrite, again with no earlier main-RFC location to have
moved from (2026-08-03). Nothing here changes or adds to any MUST / SHOULD
/ MAY requirement; every requirement lives in the main document. This file
exists so the audit trail survives being pruned out of the RFC before
publication, per the M1 target.

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
