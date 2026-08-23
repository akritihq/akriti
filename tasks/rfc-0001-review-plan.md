# RFC-0001 review: findings and remediation plan

Source: full critical review of
`rfcs/0001-persistence-diagram-interchange.md` at version 0.3.0
(Last Edited 2026-08-20), read against `src/akriti/diagrams/` as it stands
on this branch.

## Goal

Close the defects and gaps found in that review before the 2026-08-23
publication date, and separate the ones that are corrections from the ones
that need an owner's decision. Thirty findings, `F1` through `F30`, keeping
the review's own numbering so the two documents cross-reference cleanly.

## Constraints and decisions

- **Publication is live-dated.** `origin/rfc/publish-for-comment` already
  exists as one commit flipping the document to `1.0.0` and the Status row to
  "Open for public comment -- opened 2026-08-23, closes 2026-10-16 (#31)". It
  is small and rebase-friendly by design, so it goes last. Tier 1
  lands before it.
- Tier 1 is what must be true at publication; nothing below Tier 1 blocks it.
- Two findings rest on library behaviour this document has not measured
  (`F1`, `F2`). They get a measurement before they get an edit, on the
  standard §9 and Appendix A already hold everything else to.
- Findings that change a BCP 14 clause move `spec_version`'s minor per
  §10.2, and `io.py`'s `_SPEC_VERSION` plus the four `spec_version` pins in
  the I/O tests follow. Expect one bump for the whole pass, not one per
  finding: `0.4.0` since Tier 1 precedes publication.
- Where the implementation is already correct and the RFC is wrong, the RFC
  moves. That is the case for `F3`, `F5`, `F8` and `F26`, and it is worth
  noting as a pattern: the spec has drifted behind working code in four
  places, all of them found by reading the two side by side.
- `F25` is the one finding that is wrong in both, and the only Tier 1 item
  needing a code change.

## How to read the tables

`Lands in` is where the fix goes: `RFC` for a specification edit only,
`RFC+code` where both move.
`Evidence` is `read` for a defect visible in the document, `diff` where the
implementation was checked and diverges, and `measure` where a run is owed
before acting.

---

## Tier 1 -- publication blockers

Six. Each is wrong rather than thin, and a reviewer who finds one will
reasonably doubt the rest.

| ID | Finding | Location | Lands in | Evidence | Fix |
|---|---|---|---|---|---|
| `F25` | `d.finite` performs the cardinality change `finitize(at="drop")` exists to record, and records nothing. Identical bars, contradictory provenance. `d.finite.finitize(at="drop")` then hits §5's return-unchanged rule and preserves `essential_bars = "faithful"` on a diagram whose essential set was stripped. `d.dim(k)` has the same defect one degree over, and `"lost_upstream"` is meaningless once H0 is filtered away | §3.2 [:227], §5 [:850] | RFC+code | diff -- `core.py:_masked` is shared by `dim`, `finite` and `finitize(at="drop")` and carries `meta` through unchanged, while the drop branch writes `"finitized_dropped"` after it | Give §3.2 a general `meta` propagation rule for every derived diagram. `canonical()` is currently the only accessor that has one. Cheapest coherent option: `d.finite` records `"finitized_dropped"` like its twin |
| `F2` | `from_giotto`'s impossibility check false-positives on ordinary input. A `VietorisRipsPersistence(homology_dimensions=(1, 2))` array has no degree-0 rows, so "all H0 deaths finite" is vacuously true and a valid call is refused | §11 [:2395] | RFC | measure, then read | Scope the predicate to "if any degree-0 rows are present". Not yet implemented on the adapter branch, so the specification can be corrected before code follows |
| `F4` | Appendix A's preamble puts A.1 through A.4 on `scikit-learn 1.8.0`; §9.2 says giotto raises `TypeError` on 1.8.0. The three `reduced_homology=True` rows and all of A.2 are giotto output. Only the `=False` rows carry the pinned-1.3.2 note | A [:2724], §9.2 [:1793] | RFC | read | Establish which environment actually produced those rows and state it. This is the worst possible place for the defect: §9's thesis is that unmeasured claims rot silently |
| `F3` | §4.2's `__getitem__` pseudocode is wrong for negative indices: `i = -1` reads `offsets[-1]` and `offsets[0]`, returning an empty slice with the correct `metas[-1]`. The last diagram silently becomes an empty diagram carrying the right metadata | §4.2 [:582] | RFC | diff -- `core.py` normalises correctly, so the shipping code already diverges from the normative sketch | Specify index normalisation, out-of-range `IndexError`, `operator.index` for non-integers, and whether slices are supported |
| `F7` | §2's "the diagonal is implicit and MUST NOT be stored" forbids trivial bars, which §11.2 requires as a test case and on which §4 and §11.1's entire padding argument depends | §2 [:94] | RFC | read | Rewrite as "the diagonal is not materialised as a multiset"; a `birth == death` bar is an ordinary bar |
| `F19` | §12's header contradicts §12.1 in one sentence: "one, D22, is open (§12.1) ... §12.1 is empty" | §12 [:2648] | RFC | read | Delete the stale clause, left over from changelog entry 45 |

---

## Tier 2 -- open decisions, need an owner

Four. None is an editing job; each is a call about what the type promises.

| ID | Finding | Location | Lands in | Evidence | Fix |
|---|---|---|---|---|---|
| `F1` | No JAX-backed diagram can exist. I2 requires `float64` and B7 `int64`; JAX defaults to `jax_enable_x64=False`, under which a requested `float64` warns and downcasts. JAX is the RFC's worked example throughout -- §3.3's traceability argument, `from_diagrams`' mixed-namespace case, §6.3's cross-namespace `ValueError`, and entry 45's decision to move the torch illustrations to JAX | §3.1 [:151], [:619] | RFC | measure | State the x64 requirement as a supported-backend constraint with a CI assertion, on the D16 pattern -- or stop using JAX as the illustrative backend. Deserves a D-row beside D22 rather than silence. Already tracked as `C4 / JAX` in questions.md, and absent from the RFC |
| `F5` | §3.1's "MUST enforce these at construction" and §3.3's "`d.canonical()` genuinely is traceable" are incompatible: I6 is a Python `bool`, which concretizes a traced array, so every operation returning a diagram is eager-only | §3.1 [:144], §3.3 [:309] | RFC | diff -- `core.py` resolves it with a private `_unchecked` used in eight places; §3.1 sanctions one bypass, and only on the copy rule | Add a clause: derivation from an already-valid instance MAY skip revalidation, with the soundness conditions. Without it, `canonical()`, `dim`, `finite` and `__getitem__` are all unimplementable as specified |
| `F6` | Requirement 4 is defeated by the file's own `spec_version`. §10.1's stated purpose is verifying a regenerated `repro/` fixture by checksum without the library; that fails on every minor bump, and the minor bumps on any altered BCP 14 clause. 0.1.0 to 0.3.0 in three weeks. §11.2's two-saves-in-one-process test cannot catch it | §10.1, §10.2 [:2113] | RFC | read -- `io.py:25` writes `_SPEC_VERSION = "0.3.0"` into every envelope | Either scope requirement 4 to one writer version, or move `spec_version` out of the byte-compared payload. Interacts with `F21` |
| `F8` | I8 covers arrays and the `metas` sequence but not `params` and `provenance`. A caller who retains the mapping mutates provenance after the constructor validated the `essential_bars` qualifier rules. Unlike the array case this is enforceable | §3.1 [:203], §8 | RFC | diff -- `core.py:_freeze_json_mapping` already does it | State the mapping rule in I8 alongside the `metas` rule it already has. The specification is behind the code in the one field §8 exists to make auditable |

---

## Tier 3 -- specification gaps, one editing pass

Eleven. Correct together; none needs a decision.

| ID | Finding | Location | Lands in | Fix |
|---|---|---|---|---|
| `F9` | `clamped_rows` is a reserved key with no writer, no threshold, no clamp target and no adapter obligation. Two mentions in the whole document. It also sits badly beside "I6 is checked exactly ... we surface it rather than absorb it" three lines above. This is the defect D15 removed `order` for | §3.1 [:162], §8 [:1364] | RFC | Specify the threshold, target and warning, or drop the key. Overlaps `O4`, which records that the adapter uses an eight-ULP clamp the RFC never sets |
| `F10` | The `array_api_compat` fallback is mis-scoped. `namespace_of` reaches it for anything lacking `__array_namespace__` but declares it `akriti[torch]` and calls it unreachable by default. A `numpy` 1.26 array from a transitive backend dependency yields `ImportError: install akriti[torch]` | §3.3 [:389] | RFC | Add the numpy-floor branch to `namespace_of`, not only to the io and adapter imports |
| `F11` | `d.dimensions` is missing from §3.3's eager-only enumeration. It uses `unique_values`, so its shape is data-dependent, but it is neither boolean-mask selection nor a `bool`/`str` return -- the two categories §3.3 names | §3.3 | RFC | Add it. `core.py`'s module docstring already lists it |
| `F12` | The `to_csv()`/`from_array` round-trip pair has no reader half. `from_array` takes an array, not a path; no `from_csv` is specified; on a zero-dependency install there is no way to turn `inf`-bearing text into a `float64` array. §11.2 nonetheless requires a test that reads the file back | §10.3, §11.2 | RFC | Add `from_csv`, or state that the caller supplies the parse and that it needs `akriti[numpy]` |
| `F13` | `from_gudhi`'s sklearn form ignores the outer per-sample axis. The RFC describes what `fit_transform` returns "per sample" but returns a scalar diagram, never saying the caller indexes. `from_giotto`, facing the identical situation, must return a batch | §11 [:2255] | RFC | Say what `from_gudhi` does with the whole `list[list[(n,2)]]`, which is structurally distinguishable, and reconcile the asymmetry. Also: `dim=` alongside the sklearn form should raise on the same grounds as the `list` form |
| `F14` | `__getitem__`'s "MUST return a view ... not a copy" is unsatisfiable on JAX, where slicing allocates | §4.2 | RFC | Rephrase as a prohibition on deep-copying the buffers. The safety argument survives -- a copy is strictly safer |
| `F16` | §9.1 derives "same dimension" from a bijection formula with no dimension in it. Degree preservation is imported silently -- the same defect the section attributes to persim. The `inf - inf = 0` convention for two essential bars is also unstated, while §6.3 spends a paragraph on `|inf - inf|` being `NaN` | §9.1 [:1721] | RFC | State degree preservation in the definition; the max-over-degrees decomposition then follows properly |
| `F26` | `@dataclass(frozen=True)` cannot coexist with §6.3's custom `__eq__`: the generated `__eq__` would compare arrays elementwise, and `frozen=True` generates a `__hash__` over unhashable fields | §3.1 [:180] | RFC | Specify `eq=False` and the `__hash__` decision, including whether `DiagramMeta` is hashable. `core.py` already uses `eq=False` on both types and `__hash__ = None` on `DiagramMeta` |
| `F27` | The `content_hash` fallback path is unspecified where it matters. §11.2 requires the buffer path and the per-element path to agree byte-for-byte on signed zero, `inf`, subnormals and the `int32` extremes, but §8.1 never says how a namespace with no exposed buffer produces big-endian bytes | §8.1, §11.2 | RFC | Specify the per-element packing (`>d`, `>i`) and where signed-zero normalisation sits relative to the float conversion, so the two paths are provably identical rather than tested into agreement |
| `F28` | §8's MUST-populate list is stale against §11, which adds `filtration`, `coeff_field`, `coeff_field_source` and `essential_bars_source`. This is the single-point-of-truth failure that produced D17 | §8 [:1338] | RFC | Single-source the list, or have §8 defer to §11 |
| `F29` | `save()` is unspecified for non-CPU arrays. `np.asarray` raises on a CUDA tensor or a GPU-resident JAX array. The RFC never mentions device, despite torch being supported behind `akriti[torch]` | §3.3, §10 | RFC | Say what the I/O boundary does about device, or require host residency and fail actionably |

---

## Tier 4 -- over-claims and judgement

| ID | Finding | Location | Fix |
|---|---|---|---|
| `F15` | `b.canonical()` is not inherently eager-only. That is a fact about routing it through `b[i]`, not about the operation. A segment key from `xp.searchsorted(offsets[1:], xp.arange(total_bars), side="right")` gives a fourth, most-significant sort key with no concretisation, and `searchsorted` is in A.7.1's own list. The document forecloses a traceable batch canonicalisation by describing one implementation as a property of the operation -- the "shape-preserving is not traceable" conflation §3.3 exists to fix, running the other way | §3.3 [:303] | Either implement the traceable form or state that the eager-only restriction is a choice of implementation rather than a property |
| `F17` | D21 understates its own cost. The requirement refuses `infinity_values=None`, and D21 itself concedes that under giotto's default `max_edge_length=inf` that resolves to `inf` and is safe. The practical cost is the most common giotto configuration, not the rare deliberate `99.0` the cost cell names, and the fix reaches back into how the transformer was constructed | §12.2 D21 | Name the right case in the cost cell. The trade may still be correct |
| `F18` | `strip_padding` keeps the "default and warn once" pattern §5.1 rejects on measured evidence and §9.1 argues against at length. The distinction is real -- trivial rows are detectable, so there is something to warn on -- but D21 makes that argument for `infinity_values` and §11.1 and D4 never make it for `strip_padding` | §11.1, D4 | State the distinction where the pattern is used |
| `F20` | Only `canonical()` has a stated `meta` propagation rule. The general form of `F25`, and worth fixing as one rule rather than case by case | §3.2, §7 [:1283] | Folded into `F25` |
| `F21` | The `spec_version` bump rule has become ceremony. "Any clause carrying a BCP 14 keyword" makes an editorial rewording of a MUST a minor bump, and entries 50 and 51 already track it by counting keywords as a proxy for semantic change. With `F6` and "`load` MUST NOT branch on it", the version is audit-only, costly, and harmful to determinism | §10.2 [:2124] | Bump on semantic change to a requirement. Decide jointly with `F6` |
| `F22` | `d.essential` (mask) and `d.finite` (diagram) read as complements and are not; the finite mask is `~d.essential`. §4.3 compounds it: `b.essential` is a mask and `b.finite` is listed as a gap | §3.2, §4.3 | `essential_mask` costs nothing now and cannot be renamed after M1 |

---

## Tier 5 -- editorial

`F30`, broken out. Each is a line or two.

| ID | Item | Location |
|---|---|---|
| `F30a` | §8's Unicode clause says "the five scalar fields"; four are strings and `coeff_field` is an `int` | §8 [:1374] |
| `F30b` | §8.2's "each member hash is already available" implies a cache §8.1 does not specify. Relatedly, `core.py`'s `_bounds` cache uses `object.__setattr__` on a frozen dataclass, which B8's "none may rebind them" gives no room for. If derived caches are legal, B8 should say so | §8.2, §4.2 |
| `F30c` | §10.2 uses "member" in three senses on one page: zip members, `.npz` arrays, and "`"batch"` without `offsets` in `bars.npz`" | §10.2 |
| `F30d` | `==` between a `PersistenceDiagram` and a `DiagramBatch` is unspecified. Should be `NotImplemented` then `False`, not §6.3's `ValueError` | §6.3 |
| `F30e` | I8's copy-on-construction is dead cost on immutable-array backends. "MUST NOT alias a caller-writable array" says what is meant | §3.1 |
| `F30f` | §11.2's determinism test sleeps 2.5 s per case. Asserting `ZipInfo.date_time == (1980, 1, 1, 0, 0, 0)` tests the mechanism directly, runs instantly, and does not depend on A.9's DOS-timestamp reasoning surviving a future `zipfile` | §11.2 |
| `F30g` | `from_diagrams(diagrams, xp=...)` must reject a disagreeing `xp` by `is`, per D16. The clause does not say so | §4.2 |

---

## Structural

Not findings against a clause, and the higher-value half of this review.

- `F23` -- **Not reviewable at this length by the audience §1 names.** About
  45,000 words carrying roughly 169 MUSTs, with no normative index and no
  conformance checklist. The R and `tdaverse` reader §1 is courting must read
  the whole document to find the obligations. Appendix B exists to hold the
  long arguments and holds four subsections, while §3.1 spends about 700
  words on array mutability, §5 about 1,500 on a three-mode function, and §8
  restates its `essential_bars_source` argument three times.
- `F24` -- **Internal references leak into a document meant for public
  comment**: `castle/`, `repro/`, `compat/`, `DEPENDENCIES.md`,
  `tools/check_license_closure.py`, the `classify` repository, branch
  `adapter2`, PR #10. §1 claims the document does not require adopting the
  rest of Akriti; `castle/` appears in a normative sentence at [:504] and is
  never defined.

**The recommendation both point at.** `F8`, `F9`, `F25` and `F28` are one
failure repeated: a rule argued exhaustively in one section and not
propagated to the places it binds. A normative-requirements table -- clause,
section, who enforces it, which test -- would have caught all four
mechanically. It is the single highest-value addition available before the
comment window opens, and it is worth more than any further prose.

---

## Work plan

Phases map one-to-one onto the branches below; the branch name is given
with each so the two sections cannot drift apart.

1. **Measure the two unmeasured findings.**
   Branches `rfc/0001/evidence-jax-x64`, `rfc/0001/evidence-giotto-h0-scope`,
   `rfc/0001/evidence-a1-environment`, concurrent.
   - `F1`: confirm that a `float64` request under default JAX config
     downcasts, and that `int64` offsets do the same, then record the
     environment as Appendix A already does elsewhere.
   - `F2`: confirm that `VietorisRipsPersistence(homology_dimensions=(1, 2))`
     emits no degree-0 rows, in the pinned scikit-learn 1.3.2 environment
     §9.2 forces.
   - Neither edit lands before its measurement does.

2. **Tier 1 corrections.** Branch `rfc/0001/tier1-corrections`.
   `F2`, `F3`, `F4`, `F7`, `F19`, and the `F25` specification half.
   Establish A.1 and A.2's true environment first, since `F4` may require
   a recapture rather than a wording change.

3. **`F25` in code.** Branch `diagrams/meta-propagation`. May run
   concurrently with phase 2, but lands before or with it, never after --
   see "Two ordering constraints" below. Decide the `meta` propagation
   rule, then apply it to `_masked`'s three callers. Test first, per the
   branch's convention: a diagram whose `essential_bars` survives
   `d.finite` is the regression.

4. **Tier 2 decisions.** Branch `rfc/0001/decisions-d23-d26`. Each needs
   an owner and lands differently: `F1` and `F6` as decision rows, `F5`
   and `F8` as clauses. `F6` and `F21` are one decision taken twice and
   should be settled together, before phase 7.

5. **Tier 3 in one editing pass**, then Tier 4 and Tier 5. Branches
   `rfc/0001/tier3-gaps` then `rfc/0001/tier4-5-editorial`.

6. **Structural.** Branches `rfc/0001/normative-index` then
   `rfc/0001/internal-reference-sweep`, in that order. Build the
   normative-requirements index (`F23`) and sweep the internal references
   (`F24`). The index is the artifact that keeps the Tier 3 class of
   defect from recurring, and it goes last so it does not index text the
   earlier branches are still changing.

7. **Version and changelog.** Rebase `rfc/publish-for-comment` last. One
   `spec_version` bump for the whole pass under the current rule, or under
   whatever `F6` and `F21` replace it with; `io.py`'s `_SPEC_VERSION` and
   the four I/O test pins follow. Changelog entries record what landed,
   verified against the document rather than against this plan.

8. **Proof.** Full pytest, Ruff check and format, mypy, and the dependency
   closure checks, on the branch's existing standard.

---

## Branches and ordering

### The governing constraint

Twenty-six of the thirty findings edit one 3,353-line markdown file. This
repository has already paid for parallelizing that: changelog entry 58
reconciled `adapter2` against the RFC branch after "entries 48-54 landed here
while six corrections landed against the pre-48 text", and §12.3's six `R`
rows are the permanent scar. Theming the RFC into concurrent branches buys
tidy pull-request titles and re-runs that reconciliation.

**So: branch by what leaves the RFC file, and serialize everything that does
not.** Markdown prose and tables three-way-merge badly, and the failure mode
is silent semantic loss in a document whose whole thesis is silent wrongness.

### Parallel-safe branches

Disjoint files, run concurrently off `main`.

| Branch | Findings | Files | Note |
|---|---|---|---|
| `rfc/0001/evidence-a1-environment` | `F4` | `rfcs/evidence/probe_backends.py`, fixtures | Long pole. Needs the pinned scikit-learn 1.3.2 environment; may force a recapture |
| `rfc/0001/evidence-jax-x64` | `F1` | new evidence script | Gates the `F1` decision row |
| `rfc/0001/evidence-giotto-h0-scope` | `F2` | evidence script, fixture | Confirms `homology_dimensions=(1, 2)` emits no degree-0 rows |
| `diagrams/meta-propagation` | `F25` code half | `src/akriti/diagrams/core.py`, tests | `_masked`'s three callers |

### The serialized RFC train

One file, strictly ordered. Each rebases on the one above.

| Branch | Findings | Why here |
|---|---|---|
| `rfc/0001/tier1-corrections` | `F2` `F3` `F4` `F7` `F19`, `F25` spec half | Surgical, ships first, blocks publication |
| `rfc/0001/decisions-d23-d26` | `F1` `F5` `F6` `F8` | Where new D-numbers are allocated, continuing §12's deliberately non-dense sequence |
| `rfc/0001/tier3-gaps` | `F9`-`F14`, `F16`, `F26`-`F29` | One editing pass |
| `rfc/0001/tier4-5-editorial` | `F15`, `F17`, `F18`, `F20`-`F22`, `F30a`-`F30g` | |
| `rfc/0001/normative-index` | `F23` | Additive appendix. Last, or it indexes text the branches above are still changing |
| `rfc/0001/internal-reference-sweep` | `F24` | Scattered single words across many sections: the worst conflict profile in the set |

### Two ordering constraints that are not obvious

- **`F25`'s code half lands before or with its specification half, never
  after.** §3.2 is currently *silent* on `d.finite`'s provenance while
  `core.py` actively produces the contradiction. Code first leaves the code
  correct and the specification merely incomplete; specification first leaves
  shipping code non-conforming.
- **`F6` and `F21` settle before `rfc/publish-for-comment` merges.** That
  commit hardcodes the `1.0.0` story and reproduces the bump-rule text
  verbatim, so a change to the rule changes the publication diff. `F6` also
  gets worse after publication: fixtures published at `1.0.0` will
  checksum-diff against every later `1.x`.

### What not to split

- **Tier 3 by theme.** Eleven findings across eight sections, each one to
  three sentences. Splitting costs more in rebases than it saves in review.
- **`F6` from `F21`.** One decision taken twice.
- **`F23` concurrently with the clause edits.** If it is wanted early, branch
  the generator and run it last.

---

## Relation to existing trackers

| Finding | Existing item | Relation |
|---|---|---|
| `F1` | `C4 / JAX` in questions.md | Same issue. The finding adds that it makes I2 and B7 unsatisfiable rather than merely untested, and that it belongs in the RFC |
| `F9` | `O4` in questions.md | Same issue from the other side: `O4` asks whether the eight-ULP threshold is normative, `F9` observes the key has no specified writer at all |
| `F3` | `O9` in questions.md | Adjacent: `O9` covers the missing `__iter__`, `F3` the broken indexing it would run over. One amendment to §4's interface block closes both |
| `F25` | adjacent to `O15` (no-op `finitize` returns `self`) | Different defect, same neighbourhood. Settle together |
| D22 | `O17` in questions.md | Already open in the RFC; untouched by this review |

Findings not listed above are new.

---

## Status

Executed 2026-08-23. See `rfc-0001-review-progress.md` for what landed on
which branch and what the measurements changed about the findings.

- [x] Measurements for `F1` and `F2` -- and a third, unplanned, for `F4`
- [x] Tier 1 corrections
- [x] `F25` in code, test first
- [x] Tier 2 decisions taken and recorded -- `F1` settled as D23, `F5` and
      `F8` as clauses, `F6`/`F21` **opened as D24 and left with the lead**
- [x] Tier 3 editing pass
- [x] Tier 4 and Tier 5
- [x] Normative-requirements index (`F23`) -- generated, with a standing test
- [x] Internal-reference sweep (`F24`)
- [x] Version bump to `0.4.0` and changelog entries 66-71
- [ ] `rfc/publish-for-comment` rebased -- **blocked on D24**, per this plan's
      own ordering constraint: that commit reproduces the bump-rule text
      verbatim, so settling `F6`/`F21` changes the publication diff
- [x] Full proof run

Two findings turned out to be wrong as written, and both were caught by
measuring rather than by reading:

- **`F1`'s headline.** JAX carries a second, narrower 64-bit lever, so a
  JAX-backed diagram is constructible; a *default* JAX install is what cannot
  build one. Also: I2's `int32` half is satisfied natively, B7 is unreachable
  because I2 raises first, and the truncation is not silent where it is
  explicit.
- **`F4`'s premise.** `probe_backends.py` has shimmed scikit-learn since the
  document's first commit, so the preamble and §9.2 were never in conflict.
  What was true, and worse, is that the shim is nowhere in the document.

Three items are open and named as such rather than closed quietly:

- **D24** (`F6`, `F21`) is with the lead, and blocks the publication rebase.
- **`F22`'s rename** of `d.essential` to `essential_mask` is not taken here:
  it is a public API change and pre-M1 is the last cheap moment for it.
- **`F9`'s clamp threshold** is stated as an unfixed gap with its cost named.
  Fixing it is `O4`.
