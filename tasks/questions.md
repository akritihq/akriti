# Questions raised while implementing the adapters

Branch `adapters`, RFC-0001 §11. Everything the specification did not settle,
in three groups: what was put to you and answered, what was decided here under
a stated assumption, and what is still open for whoever owns the RFC. A fourth
group holds what review found *settled and wrong*, which is a different thing
from unsettled.

Each entry says what was chosen and what it costs, so a later reader can
reopen one without re-deriving the argument.

---

## 1. Put to you, and answered

### Q1 — Where does the array namespace come from when the input carries no array?

§11 fixes the signatures as `from_gudhi(obj, **meta)`, with no namespace
argument. GUDHI's primary form, `SimplexTree.persistence()`, is a Python list
of tuples; an empty Ripser or persim diagram list is the same problem. There is
nothing to call `__array_namespace__` on.

**Answered: lazy numpy fallback on array-less inputs.** Rejected alternative: a
keyword-only `xp=`, on the `DiagramBatch.from_diagrams([], xp=...)` precedent
(§4.2), which keeps zero third-party imports literally true but puts
`xp=np` on every GUDHI call.

**What it costs.** A JAX or torch user who passes a Python list gets a
numpy-backed diagram instead of an error. Array inputs are unaffected — they
preserve the caller's namespace exactly, as §3.3 requires. See open question
O1: §3.3's text does not currently admit this import.

### Q2 — How much of an I6 violation should an adapter absorb?

§3.1 says floating-point `death < birth` violations "are a real occurrence at
the 1e-16 level", that the adapter is the right place to clamp, and that it
MUST warn. It fixes no threshold.

**Answered: absorb representational noise only, measured in ULPs.** A gap is
repaired when `birth - death` is at most `_CLAMP_ULPS = 8` local downward
float64 ULPs of `birth`, the spacing taken with `nextafter` after conversion to
float64 so the allowance follows the representable grid at every magnitude,
zero and subnormals included. Anything larger reaches §3.1's I6 check
unmodified and raises with its magnitude named.

An earlier revision of this branch used a fixed `_CLAMP_ATOL + _CLAMP_RTOL *
|birth|` with both at `1e-12`, and it is worth recording why that went: `1e-12`
is an absolute quantity where the thing being absorbed is a *representational*
one, so it was simultaneously far too generous near zero — swallowing real
inversions eleven orders above the float64 grid — and, at `1e12` and above,
tighter than a single ULP, refusing gaps that no float64 computation could
have avoided. The ULP rule has one meaning at every magnitude.

**What it costs.** The constant is ours, not the RFC's. A backend with a
systematic error eight ULPs wide would raise where a more generous threshold
would have absorbed it — deliberately, since §3.1 says a real violation is a
bug to surface. See open question O4.

### Q3 — Does the core half of D17 land in this branch?

TODO.md recorded that `DiagramMeta` does not validate
`provenance["coeff_field_source"]`, and that "the adapter half of D17 belongs
with `adapters.py`". Adapters are what make the key live.

**Answered: adapters only; the TODO stays open.** TODO.md now says the key is
written on every `from_gudhi`/`from_ripser` call, so the gap is a reserved key
nothing checks rather than a key nothing writes.

---

## 2. Decided here, under a stated assumption

None of these were worth blocking on; each is recorded in the code at the
point it binds, and each is cheap to reverse.

| # | Question | Decision | Where it lives |
|---|---|---|---|
| D1 | §11's table names `dim=` for `from_array` and for GUDHI's `(n,2)` form, but the signature block omits it. Is it a real parameter? | Keyword-only `dim: int \| None = None` on `from_gudhi` and `from_array`. `(n,2)` without it raises `TypeError`; `(n,3)` *with* it raises, two sources for one fact. | `adapters.py` |
| D2 | Do `from_persim` and `from_array` record `essential_bars`? | No — neither key. persim computes no homology (§5.1: "no opinion") and an array has no backend, so neither can certify what happened upstream. §11 makes the key conditional, so omitting is conforming. | `from_persim`, `from_array` |
| D3 | §8 says adapters MUST populate `backend_version`. What does `from_array` populate it with? | `None`. There is no backend and no version; inventing one would be the failure `provenance` exists to prevent. | `from_array` |
| D4 | Is `clamped_rows` recorded when nothing was clamped? | Always, as an `int`, on §11.1's precedent for `padding_removed`. Absence would then unambiguously mean "no adapter wrote this". | `_diagram_from_columns` |
| D5 | What happens to a caller's `provenance` key that collides with an adapter-measured one? | Merged; the adapter's value wins. It is the only party that saw the backend's output. `backend`/`backend_version` are refused outright with `TypeError`. | `_build_meta` |
| D6 | Should a misspelled metadata field (`filtraton=`) be silently kept? | No — `DiagramMeta` raises `TypeError` naming the field. `**meta` is §8's field set, not a free-form bag. | `_build_meta` |
| D7 | How does an adapter learn the backend's version without importing the backend? | `importlib.metadata.version(dist)`, `None` when the distribution is absent — which is ordinary, since §11.2's frozen fixtures are adapted with no backend installed. | `_installed_version` |
| D8 | §11.1 says "warn once". Once per call, per sample, or per process? | Per call: trivial rows are counted across every sample, then one warning is issued. Per-process dedup would depend on `warnings` filter state, which is not a property of the data. | `from_giotto` |
| D9 | How should §9.3's two coefficient-field defaults be asserted against the installed backend? | Signature introspection (`inspect.signature(...).parameters[...].default`), which is literally the default. The behavioural alternative needs a torsion-carrying complex — an RP² triangulation for GUDHI and no clean equivalent for Ripser. | `test_rfc0001_adapters_live.py` |
| D10 | GUDHI's own defaults drop the two cases §11.2 requires — `min_persistence=0` excludes zero-persistence bars, `persistence_dim_max=False` returns nothing at all on a 0-dimensional complex. Hand-write them? | No: §11.2 forbids it. Captured a *second real call* with `min_persistence=-1.0, persistence_dim_max=True`. Both are backend calls recorded verbatim. | `tools/capture_backend_fixtures.py` |
| D11 | Should `_namespace_of` in `adapters.py` implement §3.3's `array_api_compat` fallback now? | Superseded, and the migration has since happened: `core.namespace_of` is the single resolver (D18), `_namespace_of` is gone, and `adapters.py` holds no direct `__array_namespace__()` call — `_namespace_for_rows` routes its NumPy probe through the resolver too, so one backend cannot yield two namespace objects. | `core.namespace_of`, `_namespace_for_rows` |
| D12 | giotto needs rectangular `(n_samples, n_points, n_features)` input, so unequal bar counts cannot come from unequal cloud sizes. How is a padded fixture produced? | A circle and a gaussian blob, both 40 points: different topology, different bar counts, so the batch pads. Confirmed padded — 5 trivial rows in one sample, 0 in the other. | `tools/capture_giotto_fixture.py` |
| D13 | Should the five adapters be reachable as `akriti.diagrams.from_gudhi`? | Yes, with a test. §1 makes "diagrams in" the primary user path; an entry point reachable only by module path is an implementation detail. | `diagrams/__init__.py` |
| D14 | The lazy `import numpy` pulled numpy's stubs into `mypy`, where they fail to parse against the project's 3.10 floor. Raise `python_version`, or add a mypy override? | Neither: `import_module("numpy")`, so the checker never resolves a package this one does not depend on. `pyproject.toml` is untouched. Two `[tool.mypy.overrides]` spellings were tried first and neither prevented parsing. | `_namespace_for_rows` |

---

## 3. Still open

These need someone other than this branch — the RFC owner, or a later change.

### ~~O1 — §3.3's "single lazy import" clause is too narrow~~ — closed

The clause has been widened, in the direction this entry argued for: §3.3 now
sets the closure over what a caller can reach rather than over which files may
import what — "nothing third-party on any path reachable without a backend the
caller installed themselves, and every exception lazy, function-scoped,
declared as an install extra" — and names the row-sequence NumPy fallback as
one of the four cases meeting it, alongside the namespace resolution rule,
`save`/`load`, and §10.3's `to_parquet`. Code and specification agree.

`DEPENDENCIES.md` repeated the same narrow sentence and should be re-read
against the amended clause before its next hand-verification date; no
dependency changed, so nothing there is wrong about the closure itself.

Kept as a struck heading rather than deleted, on O2's convention.

### ~~O2 — The core half of D17 (deferred by Q3)~~ — closed

`DiagramMeta` now rejects a `coeff_field_source` that is neither `"caller"`
nor `"backend_default"`, and rejects it being present while `coeff_field` is
`None` (`DiagramMeta.__post_init__` in `core.py`; tests in
`test_rfc0001_diagram_contract.py`, `test_coeff_field_source_*`). Kept as a
struck heading rather than deleted so the numbering below does not shift under
anyone holding a reference to it.

### O3 — Appendix A.1 has no `reduced_homology=False` row, and now could

`tests/fixtures/giotto_output.json` measures it directly: 40 H0 bars against
39, zero non-finite entries, giotto-tda 0.6.2 on scikit-learn 1.3.2. §5.1's
adapter requirement rested on an inference; the measurement now exists. Adding
the row is an edit to a document under public comment, so it goes through
whoever owns the RFC.

~~Not measured: the `infinity_values` sweep repeated at
`reduced_homology=False`. A.1 shows `infinity_values` is inert at `True` and
nothing in §5.1 turns on the cross.~~ **This paragraph is wrong and C1 below
says why.** The cross is exactly where `infinity_values` stops being inert:
`reduced_homology=True` excludes the essential H0 class upstream of the flag,
which is *why* the flag looks inert at `True`, and the moment the class comes
back at `False` the flag governs its death value. The fixture was read for its
row count and not for the value in the death column, and "zero non-finite
entries" above was recorded as agreement with A.1 rather than as the anomaly
it is at `reduced_homology=False`.

### O4 — Should the clamping threshold be normative?

Q2's constants are a policy about what counts as floating-point noise, they
appear in every adapted diagram's `clamped_rows`, and two implementations of
this RFC choosing differently would disagree about which diagrams are
constructible. That is the shape of a thing §3.1 should fix rather than leave
to an implementation — but §3.1 declined to, and possibly deliberately.

### ~~O5 — Should `dim=` appear in §11's signature block?~~ — closed

The RFC changed. §11's signature block now spells `from_gudhi(obj, *,
dim=None, **meta)` and `from_array(arr, *, columns=None, dim=None, **meta)`,
so a reader of the signatures alone can call `from_array` on an `(n,2)` array.
Both are pinned by signature introspection in `test_rfc0001_adapters.py`
(`test_from_array_signature_has_columns_before_dim`,
`test_from_gudhi_dim_is_keyword_only`).

### O6 — `from_giotto` joins D17 when §9.2's shim is testable

§11 excludes it on evidence: A.5 could not measure giotto's default
coefficient field. The pinned environment built for O3
(`tools/capture_giotto_fixture.py`) is where that measurement could now be
taken, which would close the exclusion on the terms §11 itself sets.

---

## 4. Concerns from review — defects, not gaps

Raised by an external audit pass on 2026-08-10 against the working tree of
`adapters`, and each one re-derived here against the committed fixtures before
being written down. Section 3 lists things the specification left unsettled;
this section lists things that are settled and wrong. Everything below is
present in a tree where `pytest` reports **553 passed** and `ruff`, `mypy` and
`git diff --check` are clean, which is the point: no existing check fails.

Severity is this branch's judgement, not the auditor's; two entries are
recorded at a lower severity than they were reported, and one reported gap is
recorded as not a gap at all.

**They are not the same kind of thing, and the fix differs accordingly:**

| # | Kind | Where the mistake is |
|---|---|---|
| C1 | **Specification error** | §5.1 asserts a rule the committed evidence falsifies. `adapters.py:1430` implements it faithfully, so the code is conforming *and* wrong. |
| C2 | **Implementation oversight** | No RFC clause is involved. The float64 cast runs before §3.1's ordering check; the degree column already guards against exactly this and the coordinate columns do not. |
| C3a | **Spec design tension** | Nobody wrote a bug. RFC:1307 defines the field as "as reported by the backend at adapter time" and the code does that; §11.2 separately mandates fixture replay. Each clause is fine alone. |
| C3b | **Implementation oversight** | No RFC clause is involved. `source_dtype` is read off the first degree block and recorded for the whole diagram. |
| C4 | **Test coverage** | No defect. Two of the five items reported are not gaps. |

An earlier draft of this section filed C3a as an implementation bug. That was
wrong — RFC:1307's annotation is explicit about "at adapter time" — and the
entry below is written against the corrected reading.

### C1 — `from_giotto` labels a finite sentinel `"faithful"` — RESOLVED

> **Closed 2026-08-11 by RFC-0001 entry 55 (§12.3 R5).** The adapter fix
> landed first and this row's residual was that the RFC's signature block had
> never carried `infinity_values`, so the code enforced a requirement the
> specification did not state. Entry 55 promotes it: `infinity_values` is now
> required in both §8's and §11's signature blocks, admits only `inf`, and
> §11.2 requires the refusal cases to run against the real capture carrying
> the sentinel and forbids asserting `"faithful"` over it. **The fixture
> recapture below is still outstanding** — it is what would let the faithful
> path be exercised on real giotto output rather than a synthetic array.
> Original analysis follows.


**Measured, from `tests/fixtures/giotto_output.json` itself:**

```
reduced_false / single:  H0 n=40  n_inf=0  max_death=4.0  rows_at_4.0=1
reduced_false / batch:   both samples identical to the above
reduced_true  / single:  H0 n=39  n_inf=0  max_death=0.2656
```

`MAX_EDGE` is `4.0`. The unreduced capture returns the fortieth H0 bar — the
essential one — with a death of exactly `max_edge_length` and no `inf`
anywhere. `tools/capture_giotto_fixture.py` never passes `infinity_values`, so
it took giotto's default of `None`, which assigns surviving classes the
filtration cutoff.

`from_giotto` takes only `reduced_homology` and derives
`essential_bars = "faithful"` from `False` (`adapters.py:1430`). So the
adapter's own provenance certifies as faithful a diagram in which the
essential bar has been replaced by the maximum filtration value — the first
row of §5's table of rejected conventions, described there as
"Unrecoverable. The bar is now indistinguishable from a genuine bar that
happened to die at that value." Downstream, `d.essential` is all-`False` and
§9.1's partitioning treats the bar as an ordinary death.
`test_from_giotto_derives_faithful_when_homology_is_not_reduced`
(`test_rfc0001_adapters.py:1444`) asserts the label and never asserts an
infinite death, so it blesses this.

**Three things follow that the audit did not separate, and they change the
shape of the fix.**

1. **The RFC is the bug here, not only the code.** §5.1's derivation rule is
   normative — `"lost_upstream"` when `True`, `"faithful"` when `False` — and
   the adapter implements it exactly. Appendix A.1 flagged its own gap in
   advance (RFC:2482): *"This table does not measure the claim §5.1 rests on…
   a `reduced_homology=False` row would show the effect directly, and
   `probe_backends.py` MUST gain one before M1."* The `reduced_false` capture
   **is** that row, and it falsifies the conclusion the missing row was meant
   to confirm. Per CLAUDE.md's rule — when code and RFC disagree, say which is
   wrong — this one is the RFC's.

2. **The defect is orthogonal to `reduced_homology`, so `"lost_upstream"` is
   not a safe branch either.** RFC:1032 states that an essential H1 class
   "goes through the ordinary `infinity_values` path and is faithful
   regardless" of the H0 outcome. That holds for `infinity_values=inf` and
   fails for the default `None`, under which an essential H1 bar is finitized
   to `max_edge_length` while provenance reads `"lost_upstream"` — a label
   §5.1 explicitly scopes to H0. Both branches of the rule are affected, so
   this is not a correction to one arm of a conditional.

3. **Point 2 is argued, not measured.** The fixture's H1 columns carry no
   essential class (`n_inf=0`, max death `1.666` against a cutoff of `4.0`),
   so nothing committed demonstrates the H1 case. Establishing it needs a
   capture whose H1 class survives the cutoff — a cutoff below the cycle's
   death, or a sweep of `infinity_values` at `reduced_homology=False`, which
   is the sweep O3 above wrongly recorded as unnecessary.

**Decided, and implemented on branch `adapter2`: reject.** `infinity_values`
becomes a second required keyword on `from_giotto`, alongside
`reduced_homology` and for the same reason — it is a fact about the original
call that no property of the returned array recovers. Only `+inf` is accepted;
`None`, any finite value, `-inf` and `nan` raise, because a death equal to the
cutoff cannot be told from a bar that genuinely died there and §5's table calls
that outcome "Unrecoverable".

**Rejecting does not contradict §5.1, which is why this lands without an RFC
edit.** §5.1's MUST governs what is *derived* from `reduced_homology`; it says
nothing about which inputs an adapter must accept. Narrowing the accepted
domain to `infinity_values=inf` leaves the derivation rule
(`"lost_upstream"` / `"faithful"`) untouched and makes it true for the first
time, since within that domain a `reduced_homology=False` diagram really does
carry its essential bars as `inf`.

Two alternatives were weighed and dropped:

- **Accept and record `essential_bars="finitized_at:<value>"`.** Data-preserving
  and non-silent, but §8 restricts `essential_bars_source` to `"faithful"` or
  `"lost_upstream"` and forbids a `"finitized_*"` value there (RFC:1349,
  enforced at `core.py:409`), while RFC:1394 requires the adapter to write
  `source` with the same value as `essential_bars`. The two clauses cannot both
  hold for this case, so the option needs §8 widened as well as §5.1 amended.
- **Map `death == infinity_values` back to `inf`.** This is the first row of
  §5's rejected-conventions table read backwards, and it silently misconverts
  any genuine bar dying exactly at the cutoff.

**What is *not* fixed by this, and stays owed to the RFC owner:** Appendix A.1
still has no `reduced_homology=False` row (O3), RFC:1032's claim that an
essential H1 class is "faithful regardless" is still stale, and §5.1 still
reads as though `reduced_homology` were the only input the derivation needs.
The code no longer depends on any of the three being right, which is the point
of rejecting rather than reinterpreting, but the document is still wrong where
it stands.

### C2 — Integral coordinates above `2**53` lose the ordering check — low

`_as_float64` casts before §3.1's invariants run, so an integral input whose
`death < birth` violation lives below the float64 grid arrives as a valid bar:

```python
from_array(np.array([[2**53 + 1, 2**53]], dtype=np.int64), dim=0)
# accepted: births=9007199254740992.0  deaths=9007199254740992.0
from_array(np.array([[5, 3]], dtype=np.int64), dim=0)
# ValueError, as it should
```

Reproduces identically through `from_gudhi`. **Recorded as low, not high, and
the reason is worth keeping**: the float path collapses the same pair before
the adapter is ever called — `float(2**53 + 1)` *is* `2**53` in Python — so
only the integral-array path is the adapter's to catch, and no filtration value
in this domain approaches `9e15`.

What makes it worth fixing anyway is consistency rather than risk. The degree
column already refuses exactly this class of silent cast: `_require_int32_range`
(`adapters.py:295`) checks before `astype` "because the cast does not report".
The coordinate columns take an integral dtype under the same reasoning and
carry no equivalent guard. A pre-cast exactness check on integral coordinate
input would make the two columns agree, and would come with a test at the
`2**53` boundary, which nothing currently has.

### C3 — Two provenance keys can record something false

**C3a — `backend_version` on replayed output. Conforming; left as it is.**

`_installed_version` (`adapters.py:626`) reads the local environment, so output
captured elsewhere is attributed to whatever happens to be installed here, and
`backend_version=` is deliberately refused as a caller argument (`TypeError:
'backend_version' is recorded by the adapter and cannot be passed in`). The
audit filed this as a bug and it is not one: RFC:1307 annotates the field
`# as reported by the backend at adapter time`, which is exactly what the code
records. D7 above chose the mechanism and §8 chose the meaning.

What is real is a tension between two clauses that are each fine alone. §8
defines the field as a fact about the *adapting* environment; §11.2 requires
giotto to be tested against fixtures captured on a *different* environment.
Composed, a replayed diagram carries a version that had nothing to do with its
numbers, and a reader who takes `backend_version` to mean "the version that
computed this" — which is the natural reading of a field sitting beside
`backend` — is misled.

Latent today by luck: `backend_output.json` records gudhi 3.13.0, ripser
0.6.15 and persim 0.3.8, which match this environment exactly, and giotto-tda
is absent so `from_giotto` records `None`. It stops being latent the first time
a fixture outlives an upgrade.

**Not fixed on `adapter2`, deliberately.** Every available fix — permitting a
caller override with a `backend_version_source` key on D17's precedent, or
narrowing §8's annotation — changes what a normative field means, which is the
RFC owner's call and not a defect repair. Recorded here rather than acted on.

**C3b — `source_dtype` reads only the first degree block. Fixed on
`adapter2`.**

`from_ripser` and `from_persim` both took `_source_dtype(first_array)`, so a
per-degree list with mixed dtypes recorded one of them:

```python
from_ripser([zeros((1, 2), float32), zeros((1, 2), float64)])
# provenance: source_dtype="float32"
```

Replaced by `_source_dtype_of_blocks`, which records the dtype when every block
agrees and **omits the key when they do not**. §8 gives `source_dtype` one slot
and no vocabulary for a disagreement: a compound `"float32,float64"` invents a
spelling no reader expects, and picking either member is the bug. Absence
already carries "no dtype could be determined" for the array-less case.

The diagram itself is still built. The bars are valid whatever their incoming
dtypes were, and §3.1's surface-a-violation rule is about invariants on data,
not about metadata a backend never promised.

### C4 — Test coverage gaps, one of which is not a gap

- ~~**`from_persim` has no row-order test.**~~ Added
  (`test_from_persim_preserves_row_order_within_a_degree`). It shares
  `_columns_from_degree_list` with `from_ripser`, so this was low risk and is
  now simply covered.
- **No live JAX test at all**, and `jax` is not installed in this environment.
  §6.1's float64 storage is an invariant JAX does not satisfy by default —
  x64 is off unless enabled — so a JAX-backed diagram is promised by the RFC
  and unasserted anywhere. Either the x64 assumption gets stated in §3.3 or a
  marked live test asserts it, on the `@pytest.mark.backend` pattern. **Still
  open**: installing JAX is a dependency decision, not a defect repair.
- **Malformed-coordinate matrices are uneven across backends.** `from_array`
  is well covered; the backend-shaped entry points are not tested against the
  same table of bad inputs. **Still open.**
- **`from_giotto` row order is *not* a gap**, contrary to the audit.
  `test_from_giotto_reads_columns_as_birth_death_dim` asserts element-wise
  equality of all three columns against the raw fixture, which is strictly
  stronger than the named row-order tests `from_gudhi` and `from_ripser` have.

### C5 — Three refusals the adapters owed and did not give. All fixed.

Found by an audit against the `rfc-0001-persistence-diagram-interchange-draft-revisions`
RFC, which carries three commits `HEAD` does not. None was visible to coverage
— `adapters.py` was at 99% line and branch before the fix and is at 99% after
— because all three are cases where the code ran to completion, or failed with
somebody else's words, rather than cases it never reached.

**C5a — `_is_extended_persistence` was still cardinality-dependent, in the two
row spellings `_is_persistence_row` did not recognise.**

The detector's own docstring states the rule it exists to satisfy: an input
form must not be "admissible at three bars and at five and rejected at four".
That holds only as far as `_is_persistence_row` recognises a row, and it
recognised a row by asking whether `obj[1]` was a two-element `Sequence` whose
entries were both `numbers.Real`. Two spellings failed that:

```python
from_gudhi([[k, np.array([0.0, 1.0])] for k in range(n)])
# n=3 -> ok    n=4 -> TypeError: extended persistence is out of scope
# n=5 -> ok                       ^ ndarray is not a registered Sequence

from_gudhi([[0, ["a", "b"]] for _ in range(4)])
# TypeError: extended persistence ... — for an input whose defect is a string
# where a filtration value belongs, and only ever at four rows
```

The second is also the docstring's other claim failing: `_is_persistence_row`
is documented as structural, and `numbers.Real` is a question about values.

**Fixed** by splitting the pair test out as `_is_interval`, which admits a
rank-1 two-element array as well as a two-element sequence, and decides a
sequence by `_is_coordinate_slot` — every entry holds one value — rather than
by what type that value has. The weaker "neither entry is a sequence" spelling
was tried first and is wrong: the row `[0, array([b, d])]` contains no
sequence either, so a member of two such rows would read as a row and an
extended result built from them would be accepted. That regression was caught
by `test_extended_persistence_is_rejected_when_its_intervals_are_arrays`,
which was written as the guard before the change.

**C5b — a `params=` or `provenance=` that is not a mapping.**

`_build_meta` spelled both as `dict(stated or {})`, which answered one mistake
three different ways:

| Argument | Was | §8 says |
|---|---|---|
| `provenance=0`, `False`, `""`, `[]`, `set()` | **silently discarded** by the `or` | `Mapping[str, Any]` |
| `provenance={"a"}`, `1.5` | `dictionary update sequence element #0 has length 1` | — |
| `provenance=[("a", 1)]` | **accepted**, storing a mapping built from a non-mapping | — |

The first row is the one that matters: `from_array(arr, dim=0, provenance=0)`
built a diagram that recorded nothing and reported nothing, which is the
outcome `_build_meta` refuses `backend=` and `DiagramMeta` refuses an unknown
field to prevent. `DiagramMeta(provenance=0)` raises, so the adapter was
**looser than the type it wraps** on the one path §11 makes it the boundary
for.

**Fixed** by `_as_metadata_mapping`, which requires a `Mapping` and names the
argument. `None` stays "stated nothing" on `_coeff_field`'s reading of
`coeff_field=None`. Only the container is checked there; §8's JSON rule still
runs in `DiagramMeta` on the assembled mapping.

**Residual, and core's rather than the adapter's:** `DiagramMeta` itself still
coerces, so `DiagramMeta(provenance=[("a", 1)])` is accepted where
`from_array(..., provenance=[("a", 1)])` now raises. The adapter being stricter
than the type is the safe direction and matches §11's boundary posture, but the
two should agree. Not touched here — `core.py` is outside what this pass was
asked for.

**C5c — a degree-list block that is neither an array nor rows.**

```python
from_ripser({"dgms": [None]})
# TypeError: array_namespace requires at least one non-scalar array input
```

Which is precisely what `_columns_from_degree_list` wraps `asarray` to prevent
one path over — its comment reads "the caller gets the namespace's words
instead ... names neither the adapter, the argument, nor which block was
wrong". Two of the three ways a block can be mis-shaped were guarded and
indexed; the third was not. It also failed in two different places depending on
the block: `[3]` on the namespace resolution, `[None]` a line later inside the
loop, for no reason a caller could see.

**Fixed** in `_first_array_block`, which now checks every block on the way past
instead of returning the first non-row one. That is the right place because it
runs before the namespace is resolved from what it returns, so both cases now
raise the same indexed `ValueError` the other two paths give.

---

## 5. What `adapter2` changed, and what it left owed

Branched off `adapters` at `ec6566a` plus the uncommitted tree. Suite: **574
passed**, up from 553; `ruff format`, `ruff check`, `mypy` and
`git diff --check` all clean.

**Fixed.**

| # | Change | Where |
|---|---|---|
| C1 | `from_giotto` gains a required `infinity_values`, admitting only `inf`; `None`, finite values, `-inf` and `nan` all raise. `_require_infinite_infinity_values` holds the checks and takes `object`, so they are checks and not assertions. | `adapters.py` |
| C1 | `tools/capture_giotto_fixture.py` now passes `infinity_values=np.inf`, and carries the reason at the constant. | `tools/` |
| C2 | `_require_float64_exact` refuses an integral coordinate column outside ±2⁵³ *before* the `float64` cast; `_as_coordinate` refuses the same on GUDHI's row path. Bounds read with `int`, never `float`, or the check would be made in the arithmetic whose limits it is testing. | `adapters.py` |
| C3b | `_source_dtype_of_blocks` replaces first-block sampling in `from_ripser` and `from_persim`. | `adapters.py` |
| C4 | `from_persim` row-order test. | tests |
| C5a | `_is_interval` and `_is_coordinate_slot` replace the `Sequence`-and-`numbers.Real` interval test inside `_is_persistence_row`, closing the last two spellings for which `_is_extended_persistence` accepted at three and five bars and refused at four. | `adapters.py` |
| C5b | `_as_metadata_mapping` replaces `dict(stated or {})` in `_build_meta`, so a `params=`/`provenance=` that is not a mapping is refused by name instead of being silently discarded, silently accepted, or reported in `dict()`'s words. `None` still reads as "stated nothing". | `adapters.py` |
| C5c | `_first_array_block` checks every block rather than returning the first non-row one, so a block that is neither an array nor a sequence of rows gets §11's indexed refusal instead of `array-api-compat`'s. `_is_array_block` holds the duck-type. | `adapters.py` |

**New tests, all failing before the corresponding change.** C1's refusal on the
real fixture; the fixture's sentinel asserted directly so a recapture cannot
remove the evidence silently; `"faithful"` re-grounded on an array carrying a
genuine `inf`; A.1's 40-against-39 count split into its own test; the ±2⁵³
boundary from both directions plus the inclusive-bound and float-input cases;
mixed and uniform degree-block dtypes for both list adapters. C5 adds:
array-spelled intervals accepted at three, four and five bars; a four-row list
reporting its own bad coordinate; both metadata mappings against eight
non-mapping arguments and against `None`; and five junk degree blocks across
both list adapters, each asserted by index. Two guards were written before
their changes rather than after — extended persistence still refused when its
intervals are arrays, which caught a wrong first cut at C5a, and `None`
metadata still read as unstated. Suite: **606 passed**, up from 574;
`ruff format`, `ruff check` and `mypy` clean.

**Owed, and none of it repairable here.**

1. **The fixture must be recaptured.** `giotto_output.json` was captured
   without `infinity_values`, so its `reduced_homology=False` sample carries
   the sentinel and cannot be adapted at all now — correctly. Two consequences:
   the `"faithful"` branch is exercised against a synthetic giotto-shaped array
   rather than real backend output, which is weaker than §11.2 wants, and O3's
   Appendix A.1 row still has to come from a pinned-environment capture.
   giotto-tda is not installed here and §11.2 forbids hand-editing a fixture,
   which includes substituting `inf` for the sentinel.
2. **§11's signature block is now stale.** It spells
   `from_giotto(arr, *, reduced_homology, strip_padding=None, **meta)` at
   RFC:1009 and RFC:2187. O5 established that this block is maintained rather
   than illustrative, so it needs the new argument. The module docstring says
   so at the point it diverges rather than leaving it to be discovered.
3. **§5.1, RFC:1032 and Appendix A.1 are still wrong** in the three ways C1
   sets out. Nothing in the code depends on them any more, which is what
   rejecting bought; the document is still incorrect where it stands.
4. **C3a and the two remaining C4 items** are untouched, for the reasons given
   under each.
5. **C5b's residual belongs to `core.py`.** `DiagramMeta` still coerces a
   non-mapping `provenance`/`params` through `dict()`, so it accepts
   `[("a", 1)]` where the adapter now refuses it. Stricter-at-the-boundary is
   the safe direction and matches §11, but the two should agree.
6. **Two clauses the working-tree RFC dropped and the branch still carries**,
   both implemented and tested here: §11.2's `to_csv()`/`from_array`
   round-trip-through-the-header bullet
   (`test_to_csv_single_is_lf_header_round_trip_safe_and_warns_once`), and
   §10.3's rule that `columns` names `birth`/`death` exactly once and `dim` at
   most once, raising on the argument before `arr` is inspected
   (`test_from_array_rejects_invalid_columns`, which correctly uses an all-zero
   `arr` that would construct cleanly under the positional reading). Reconcile
   the two copies before merging or the requirement is lost with no test
   pointing at it.
7. **`test_from_array_rejects_three_columns_with_a_degree` accepts
   `(TypeError, ValueError)`.** §11 says the degree-carrying-input-with-`dim=`
   case MUST raise `TypeError`, and the code does; as written the test also
   passes against an implementation the RFC forbids. One-line tightening, left
   with the other RFC-conformance items rather than folded into C5.
- **GUDHI extended persistence is not a gap either.** The audit read a remote
  branch calling it a "4-tuple"; §1 of the RFC in this tree already says
  "**four** sub-diagrams in a four-element list" (RFC:65), which matches what
  GUDHI returns and what the code handles.

---

## 6. Second review pass — C6, C7, C8. All fixed.

A review on 2026-08-11 read `adapters.py` and its suite against §11 in **both**
copies of the RFC — the working tree's and
`rfc-0001-persistence-diagram-interchange-draft-revisions` — and re-derived
each finding by running it. Starting point: **619 passed**, `adapters.py` at
99% line and branch coverage, `ruff` and `mypy` clean.

Coverage found none of these, and could not have. All three are cases where
every line ran and the result was wrong, unhelpful, or unasserted — the same
category as C5, and the reason line coverage is not the instrument for this
layer.

| # | Kind | Where the mistake is |
|---|---|---|
| C6 | **Implementation oversight, normative** | §8 names a writer for each of its seven reserved `provenance` keys and none of them is a caller. `_build_meta` merged a caller's mapping and let the adapter's keys win, which protects a key only where the adapter writes one. |
| C7 | **Message quality** | No RFC clause is violated. `columns=`' vocabulary rules ran ahead of §11's shape refusal, so a header of the wrong width was reported by whichever name rule its extra entries tripped. |
| C8 | **Test coverage** | No defect. Torch is installed and D18's compat path is asserted only for `from_array`'s happy path. |

### C6 — a caller could forge §8's reserved provenance keys — fixed

Found on `from_persim`, which is where the accident shows:

```python
from_persim([array([[0., 1.]])], provenance={"essential_bars": "faithful"})
# provenance: {'essential_bars': 'faithful', 'clamped_rows': 0, ...}
```

That diagram claims its essential bars are faithful, from an adapter §11 puts
out of scope for the claim (D2 above: "persim computes no homology ... so it
cannot certify what happened upstream"), and it carries `essential_bars` with
**no** `essential_bars_source` — which §11 forbids in as many words: "An
adapter that records `provenance['essential_bars']` MUST record
`provenance['essential_bars_source']` with the same value in the same
construction." Reachable in reverse too, and on `from_array`, and with
`padding_removed` on an adapter that strips nothing.

**The rule was right and its enforcement was accidental.** `_build_meta`
merged the caller's `provenance` and let an adapter-measured key win. That
protects a key exactly where the adapter writes one of its own — so
`essential_bars` was safe on `from_gudhi`, `from_ripser` and `from_giotto`,
and forgeable on the two adapters that record no claim. Whether a fact could
be forged was a property of which adapter you asked.

**Fixed by refusing the seven keys outright**, on the ground `backend` and
`backend_version` were already refused: `_ADAPTER_OWNED_PROVENANCE` holds §8's
table in full, and `_build_meta` raises `TypeError` naming the key. Refusal
rather than overwrite for two reasons — it is uniform across adapters, and it
tells a caller their key went nowhere instead of discarding it silently. The
rest of `provenance` is untouched and still open; what is closed is the seven
names a reader trusts.

**The pairing cannot move to `DiagramMeta` instead**, which is worth recording
because it is the obvious alternative. `_validate_provenance` already enforces
§8's other two consistency rules at construction, and adding "`essential_bars`
implies `essential_bars_source`" there would refuse `finitize` (§5) on a
`from_array` diagram — which legitimately writes `essential_bars` onto a
diagram that has no source to inherit, and which §8 requires MUST NOT write
one. The caller is at the adapter boundary, so the refusal is too.

**Two existing tests asserted the weaker rule and were rewritten, not
deleted.** `test_caller_provenance_is_merged_and_never_overwrites_a_measured_fact`
becomes `..._and_a_measured_fact_is_refused`, keeping the half that still
holds: an unreserved key is kept. `test_from_giotto_overwrites_an_adapter_
owned_provenance_key_like_the_rest` becomes `..._refuses_...`; its real
property — the zero-sample preflight must not diverge from the construction it
stands in for, or whether a typo is caught depends on how many samples the
batch carried — survives the rule changing under it, and is what the rewritten
test still asserts. The preflight comment in `from_giotto` was reasoning
explicitly about the overwrite and now reasons about the refusal.

**Still not checked, and left alone deliberately:** `_validate_provenance`
validates `essential_bars_source` against §8's two-value vocabulary but never
validates `essential_bars` against its own four-value one, so
`DiagramMeta(provenance={"essential_bars": "banana"})` constructs. That is a
`core.py` gap on the same footing as item 5 of section 5 — the free-form
`<value>` in `"finitized_at:<value>"` means the check needs a prefix rule
rather than a set membership, which is a `core.py` decision and not an adapter
repair. **Open.**

### C7 — `columns=`' vocabulary outranked §11's shape refusal — fixed

```python
from_array(zeros((1, 5)), columns=["birth", "death", "dim", "x", "y"])
# ValueError: unknown column name 'x' at columns[3]
from_array(zeros((1, 1)), columns=["birth"])
# ValueError: columns= is missing required column name(s): death, dim
```

Both inputs are refused, so nothing is silently wrong; both messages send the
caller to fix a header whose real defect is the array beside it. §11 admits
`(n, 2)` and `(n, 3)`, and a five-column table is a shape error however it is
labelled.

**Fixed by splitting the argument-only checks from the width-dependent ones.**
`_normalised_column_names` holds what §10.3's ordering rule is actually about —
"MUST raise on the argument, before `arr` is inspected" — namely the type
check, the entry-type check and the `diagram_id` refusal. §11's shape refusal
runs next. `_named_columns` then applies length, unknown name, duplicate and
missing against a width this adapter can read. The call order in
`_columns_from_named_table` is now the rule rather than a comment about it.

**`diagram_id` stays ahead of the shape check, deliberately.** A four-column
table headed `diagram_id,dim,birth,death` is a batch CSV, and that caller needs
the `.akd` format, not the true and useless news that arrays are `(n, 2)` or
`(n, 3)`. `test_from_array_names_diagram_id_before_rejecting_four_column_shape`
covered that precedence and still does.

Also fixed in passing: the vocabulary messages quoted the casefolded name, so a
`columns=["Birth", "DEATH", "Xyz"]` was answered about `'xyz'`. Both spellings
are carried through now; matching is case-insensitive, complaining is not.

### C8 — torch exercised only `from_array`'s happy path — fixed

`test_rfc0001_torch_live.py` asserted D18's five accessors and nothing else,
and torch is the one namespace RFC-0001 §3.3 reaches through
`array-api-compat` rather than a native `__array_namespace__` (A.7). The code
most likely to find a shim gap was the code nothing ran there: `_clamp_i6`
(`nextafter`, `finfo`, `full_like`, `zeros_like`, and the subnormal branch that
swaps in a benign probe), `from_giotto` (per-axis rank-3 indexing, then boolean
row masking), `from_ripser`'s `concat`, and the exporters' `stack`.

**Six tests added, and all six passed on first run.** Recorded as such rather
than dressed up: nothing was broken, and this is new coverage rather than a
repair. What it buys is that a compat-shim regression now fails the build in
the environment that installs torch, instead of reaching a user. The clamp is
covered in all three of its outcomes — repaired at one ULP, repaired at zero
through the subnormal branch, and left for I6 to refuse when the gap is real.

**Owed, unchanged from section 5:** JAX is still installed nowhere and still
unasserted, which is a dependency decision rather than a defect repair.

### After

**669 passed**, up from 619. `adapters.py` holds 99% line and branch coverage
(the one uncovered pair is the pre-existing transitive-`ImportError` re-raise
in `_namespace_for_rows`). `ruff format`, `ruff check` and `mypy` clean.

Every C6 and C7 test was written before the change and run against the
unmodified module first: 41 failing, then passing. The C8 tests were not, for
the reason given under C8.

### What this pass did *not* touch

Everything in section 5's "Owed" list stands, and none of it moved: the giotto
fixture still needs recapturing, §11's signature block is still stale on
`infinity_values`, §5.1 and Appendix A.1 are still wrong in C1's three ways,
C3a and the two remaining C4 items are untouched, C5b's residual still belongs
to `core.py`, and item 6's two dropped clauses still need reconciling between
the two RFC copies before either is merged.

One correction to item 7: it is right that
`test_from_array_rejects_three_columns_with_a_degree` accepts
`(TypeError, ValueError)` where §11 mandates `TypeError`. Left as it was —
tightening it is an RFC-conformance item, and this pass deliberately did not
open that list.

**On the two RFC copies**, re-checked here because C6 and C7 both turn on §11
text that differs between them:

- The branch calls `extended_persistence()`'s result a **4-tuple**; the working
  tree calls it a four-element **list**. The working tree is right, and it is
  measured rather than argued: `test_live_gudhi_extended_persistence_is_
  rejected` asserts the shape against GUDHI 3.13 directly. Section 5's closing
  note already said so; this pass confirms it against the live backend. A
  4-tuple is still refused, but by the flat-tuple message rather than §11's
  "MUST name the scope exclusion" — a one-line fix if the branch wording is
  ever the one that survives.
- The two clauses the working tree dropped (item 6) are both still implemented
  and tested. Nothing in this pass changed that, and the reconciliation is
  still owed.
