# Open questions and owed follow-ups

This file contains only matters that do not yet have a final answer or a
completed implementation. Resolved questions, local implementation decisions,
fixed review findings, and chronological review notes have been removed.

Existing `O`/`C` identifiers are retained where another project file may refer
to them. “Owner” below means the person responsible for changing RFC-0001 or
making the project policy decision; it does not mean the adapter branch should
choose silently.

## Status

| ID | Open matter | Needed next |
|---|---|---|
| O3 / O10 | Giotto evidence and Appendix A.1 are not yet reconciled | Choose a comparable capture, amend the table, and correct the related claims |
| O4 | The adapter uses an eight-ULP clamp, but the RFC sets no threshold | Decide whether the exact threshold is normative |
| O6 | Giotto's coefficient-field default is unmeasured | Measure it, then decide whether `from_giotto` joins RFC D17 |
| O7 | Malformed backend inputs use two exception types | Specify or explicitly decline a normative taxonomy |
| O9 | The RFC relies on batch iteration without specifying `__iter__` | Amend the RFC interface block |
| C3a | Replayed output can record the local backend version instead of the producing version | Decide what `backend_version` is meant to identify |
| C4 / JAX | JAX namespace support is promised but not exercised | Decide the x64 requirement and optional test/dependency policy |
| C5b | Adapters require metadata mappings while `DiagramMeta` accepts iterable pairs | Choose and enforce one public boundary |
| O12 | RFC §9.1's guarded bottleneck wrapper is absent | Schedule separate numerical work with the required paper-equation citation and delegated finite-part computation |
| O13 | Explicit `essential_bars_source=None` is accepted | Add a core metadata regression test, then validate key presence rather than `.get()` |
| O14 | `allclose` has no tolerance domain | Specify finite non-negative tolerances and exception types before changing the core API |
| O15 | A no-op `finitize` returns `self` despite §3.1's new-object wording | Decide whether code or RFC expresses the intended identity contract |
| O16 | `DiagramMeta` admits Unicode surrogate code points that cannot be UTF-8 serialized | Define JSON-safe strings at core construction, then reject invalid scalars there |
| O17 | RFC §10 gives `load` no resource-budget contract for untrusted archives | Define a threat model without breaking round-trips for valid large diagrams |
| Test policy | Two normative docstring requirements have no regression tests | Decide whether and how to pin them |

---

## RFC-owner decisions

### O3 / O10 — How should Appendix A.1 record giotto's unreduced result?

#### Known

- With `reduced_homology=True`, the measured giotto result has 39 H0 bars and
  omits the essential H0 class.
- The committed `reduced_homology=False`, `infinity_values=inf` fixture has 40
  H0 bars, including one death at `inf`.
- Appendix A.1 and the committed fixture do **not** use the same point cloud.
  Appendix A.1 samples random angles with `rng.uniform`; the fixture uses
  evenly spaced angles with `np.linspace`.
- H0 bar counts agree across those constructions, but the finite H0 death
  coordinates and the H1 counts do not: the Appendix A.1 cloud has two H1 bars
  and the fixture cloud has one. The fixture's unreduced row therefore cannot
  be inserted into Appendix A.1 as though only the giotto setting changed.
- `from_giotto` now requires `infinity_values=inf`, so the adapter no longer
  accepts a finite sentinel as a faithful essential death.

#### Unanswered

1. Should giotto be recaptured on Appendix A.1's exact `uniform` cloud, or
   should Appendix A.1 explicitly present the unreduced row as evidence from a
   second cloud? Recapturing is the cleaner controlled comparison.
2. Once comparable evidence exists, how should the missing
   `reduced_homology=False` row be added to Appendix A.1?
3. RFC §5.1 still describes provenance as derived from
   `reduced_homology` alone, while safe interpretation also depends on
   `infinity_values=inf`. The nearby claim that an essential H1 class is
   “faithful regardless” also needs either narrower wording or direct evidence
   for the finite-sentinel cases. What exact correction should the RFC make?

### O4 — Should the eight-ULP clamping threshold be normative?

The implementation repairs an adapted row when `death` is below `birth` by at
most eight downward float64 ULPs, then records the repair and warns. A ULP is
the gap to an adjacent representable floating-point value at that magnitude.
Anything beyond eight ULPs reaches the core invariant unchanged and raises.

RFC §3.1 requires adapters to absorb observed floating-point noise but does
not define how much. Consequently, two otherwise conforming implementations
could disagree about whether a diagram is valid and how many rows were
clamped.

**Decision needed:** Should RFC §3.1 mandate the eight-ULP rule, mandate a
different rule, or explicitly allow implementation-defined thresholds? If the
threshold remains implementation-defined, the interoperability cost should be
stated.

### O6 — Should `from_giotto` join RFC D17?

RFC D17 requires GUDHI and Ripser adapters to record `meta.coeff_field` and
whether it came from the caller or a documented backend default. Giotto is
excluded because Appendix A.5 did not establish its default coefficient
field.

A pinned giotto-tda 0.6.2 / scikit-learn 1.3.2 environment now exists, so the
missing fact can be measured rather than inferred.

**Evidence and decision needed:** Measure giotto's default coefficient field
in the pinned environment. Then decide whether §11 should require
`from_giotto` to record `coeff_field` and `coeff_field_source` on the same
terms as GUDHI and Ripser.

### O7 — What exception taxonomy should malformed backend input use?

The RFC requires `TypeError` for several invalid argument combinations, but it
does not specify exception types for malformed backend result objects. Current
Ripser behavior illustrates the split:

```python
from_ripser({})  # ValueError: missing "dgms"
from_ripser({"dgms": "nope"})  # TypeError: wrong kind of value
```

Both interpretations are defensible under Python's usual distinction between
the wrong kind of object and an invalid value. Tests pin the current behavior
without declaring it normative.

**Decision needed:** Either define a stable `TypeError`/`ValueError` taxonomy
for malformed backend objects, or explicitly state that the RFC does not
standardize adapter exception types outside its named argument errors. Changing
an existing exception type is a public API change.

### O9 — Should `DiagramBatch.__iter__` be part of the RFC interface?

`DiagramBatch` now implements generator-based `__iter__`, with tests for
re-iteration, nested iteration, and the empty batch. RFC §4's interface block
lists `__len__` and `__getitem__` but omits `__iter__`, even though §4.3 relies
on expressions such as `[d.dimensions for d in batch]`.

**RFC change needed:** Add `__iter__` and its batch-order semantics to the
interface block. The implementation side is already complete.

### C3a — What does `backend_version` identify for replayed output?

The current RFC defines `backend_version` as the version available “at adapter
time.” Adapting a frozen fixture therefore records the locally installed
version—or `None`—even when a different version produced the stored numbers.
That is conforming today, but a reader can naturally interpret a version beside
`backend` as the producing version.

**Decision needed:** Choose one meaning and make it unmistakable. Options
include:

1. Keep adapter-time semantics and document that it does not identify the
   producer of replayed data.
2. Permit a producer-version override paired with a provenance source.
3. Store producing and adapting versions in separate fields.

---

## Open implementation and test work

### C4 / JAX — What JAX support is actually promised?

RFC §3.3 promises that a diagram created from JAX arrays remains JAX-backed,
while invariant I2 requires float64 coordinates. JAX normally disables x64
unless the user enables it. JAX is not installed in the current test
environment, and no live JAX test asserts the combination.

**Decision needed:** State whether JAX support requires x64 mode, then either:

- add JAX as an approved optional backend dependency and test the marked live
  path; or
- narrow the RFC's support claim until that configuration is tested.

Adding JAX requires explicit dependency approval under project policy.

### C5b — Should `DiagramMeta` require an actual mapping?

Adapters now require `params` and `provenance` to implement `Mapping`.
`DiagramMeta` still passes its inputs through `dict()`, so direct construction
accepts iterable pairs such as:

```python
DiagramMeta(provenance=[("source", "capture")])
```

The annotations say `Mapping[str, Any]`, making the adapter's strict behavior
the clearer current contract, but direct construction and adapter construction
do not agree.

**Decision needed:** Either make `DiagramMeta` reject non-mappings or document
and type a broader “mapping or iterable pairs” contract. Then add a direct-core
test for the chosen boundary.

### O12 — Implement the guarded bottleneck wrapper separately

RFC §9.1 requires `core/distances.py` to partition essential bars by
dimension before delegating: unequal essential counts return `+inf`; equal
counts are paired by sorted birth; only finite sub-diagrams reach persim; and
the two contributions combine with `max`. No distances module exists yet.

This is numerical core work, not adapter implementation. Project policy also
requires its docstring to cite a specific equation in Papers I–IV, and the
finite bottleneck computation must be delegated rather than reimplemented.

**Needed next:** Identify the governing paper equation, write property-based
tests in a separate session from the numerical implementation, then implement
the guard and delegation as its own reviewed change.

### O13 — Reject an explicitly null essential-bar source

`DiagramMeta(provenance={"essential_bars_source": None})` currently succeeds
because `_validate_provenance` reads the key with `.get()` and treats `None`
like absence. RFC §8 instead closes the vocabulary whenever the key is
present: only `"faithful"` and `"lost_upstream"` are valid.

This is a core metadata-validation defect. Adapters inherit the validation,
but none needs adapter-specific logic to fix it.

**Needed next:** Add the missing direct-constructor regression test, observe
it fail, and change the validator to distinguish key presence from an absent
key, matching the existing `coeff_field_source` pattern.

### O14 — Define the valid tolerance domain for `allclose`

`PersistenceDiagram.allclose` accepts negative, `NaN`, and infinite `rtol`
or `atol`. Those inputs can make a diagram fail comparison with itself,
contradicting the RFC and method documentation that call the relation
reflexive. Tests currently generate only finite non-negative tolerances.

The RFC specifies the symmetric formula but not the tolerance input domain or
the exception taxonomy, so changing behavior before choosing that contract
would silently create a public API rule.

**Decision needed:** Require finite non-negative `rtol` and `atol`, then choose
whether invalid values raise `ValueError` uniformly or split wrong types into
`TypeError` before adding tests and implementation.

### O15 — Resolve `finitize`'s no-op identity contract

RFC §3.1 says mutation-looking methods, explicitly including `finitize`, MUST
construct and return a new `PersistenceDiagram`. RFC §5 also says a diagram
with no essential bars is returned "unchanged." The implementation interprets
that as object identity and returns `self`; the current test requires `is`.

Both behaviors preserve values and provenance, so this is an identity-contract
tension rather than a numerical defect.

**Decision needed:** Either make no-op `finitize` return a distinct but equal
diagram and update the identity test, or amend §3.1 to state that a semantically
unchanged no-op may return `self`.

### O16 — Define JSON-safe Unicode at core construction

`DiagramMeta` currently accepts strings and mapping keys containing unpaired
UTF-16 surrogate code points, for example `description="\ud800"`. Such a
diagram passes the core's claimed JSON-safety validation but cannot satisfy
RFC §10's pinned `ensure_ascii=False` UTF-8 encoding: `save` raises
`UnicodeEncodeError`, so the universal metadata round-trip does not hold.

This is a core metadata-domain defect rather than an adapter conversion rule.
Fixing only `save` would make the error friendlier but would still leave a
publicly constructible diagram that the normative format cannot represent.

**Decision needed:** State that every metadata string and object key must be a
Unicode scalar-value sequence encodable as UTF-8, enforce it recursively in
`DiagramMeta`, and add the core regression in a separate session.

### O17 — Define an untrusted-archive resource policy

The `.akd` loader can reject inconsistent NPY headers before allocation, so a
tiny member may not merely *declare* an enormous array. A well-formed archive
can still legitimately contain a very large diagram or highly compressed
metadata/payload, however, and RFC §10.1 currently requires round-tripping
every admitted diagram without specifying byte, bar-count, nesting-depth, or
compression-ratio limits. An arbitrary fixed loader cap would therefore make
valid objects unsavable/unloadable and violate the present requirement.

**Decision needed:** Define whether `load` is intended for adversarial files
and, if so, where callers supply resource budgets or which limits become part
of the format contract. Until then, malformed length amplification is rejected
but no policy cap is invented in the adapter branch.

### Test policy — Should normative docstring requirements be pinned?

Two RFC `MUST`s are currently satisfied only by prose:

- §11 requires `from_gudhi`'s docstring to state the residual extended
  persistence case the adapter cannot identify.
- §3.1 requires the diagram class documentation to state the caller-side limit
  on immutability that Python cannot enforce.

Both statements are present today, but no test would notice their removal.
Literal docstring assertions can be brittle when wording changes.

**Decision needed:** Add semantic substring/structure tests for these required
disclosures, or explicitly treat them as documentation-review obligations
rather than automated conformance tests.
