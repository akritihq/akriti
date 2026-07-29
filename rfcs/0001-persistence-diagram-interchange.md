# RFC-0001 — Persistence Diagram Interchange

| | |
|---|---|
| **Status** | Draft — not yet open for public comment |
| **Author** | Sushovan Majhi |
| **Created** | 2026-07-29 |
| **Target** | M0 (2026-08-01) drafted · M1 (2026-09-15) published for comment |
| **Implements** | `akriti.diagrams` |

Key words **MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT**, **MAY** are to be
interpreted as in RFC 2119.

---

## 1. Purpose

Python has four widely used persistence backends and no agreement on what they
return. A user who computes a diagram with Ripser cannot hand it to a routine
written against GUDHI without knowing four undocumented conventions. This
document defines one canonical in-memory type, one on-disk format, and the
adapter contract between them and every backend we support.

It exists for three reasons, in order of importance:

1. **It is the contract that makes parallel work possible.** `diagrams/core.py`
   and `diagrams/adapters.py` are built by different people at the same time.
   This document is the interface between them.
2. **It is independently valuable to the community.** The R ecosystem solved
   interchange first (`phutil`, the `tdaverse` project's first R Consortium
   deliverable). Python has not. This is publishable, reviewable ecosystem work
   that does not require anyone to adopt the rest of Akriti.
3. **It is where the silent-wrongness bugs live.** Section 9 documents three
   cases where an existing backend returns a clean, plausible, wrong answer.
   Every one of them is invisible without a specification to violate.

**Non-goal.** This RFC does not specify vectorisations, distances, kernels, or
any statistical procedure. It specifies the *object* those consume.

---

## 2. Definitions

A **persistence diagram** is a finite multiset of **bars**. A bar is a triple
`(dimension, birth, death)` where:

- `dimension` is a non-negative integer — the homological degree.
- `birth`, `death` are real, with `birth <= death`. `death` MAY be `+∞`.

A bar with `death == +∞` is **essential**: the class it represents never dies.
A bar with `birth == death` is **trivial**: it has zero persistence.

*Multiset*, not set: two bars with identical coordinates are two bars, and the
multiplicity is meaningful. Any representation that deduplicates is wrong.

*Finite*: infinite diagrams (e.g. the full diagonal) are out of scope. The
diagonal is implicit and MUST NOT be stored.

---

## 3. The canonical type

```python
class PersistenceDiagram:
    births: np.ndarray  # shape (n,), float64
    deaths: np.ndarray  # shape (n,), float64, may contain +inf
    dims:   np.ndarray  # shape (n,), int32
    meta:   DiagramMeta
```

Three parallel arrays, one row per bar, all of length `n`. This is the
representation chosen in the execution plan (§2.4) and it is the right one:

- It slices cleanly (`d.dims == 1` is a mask, not a lookup).
- It survives conversion to any array library without restructuring.
- It has no per-dimension nesting, so a diagram with a gap in its dimensions
  (H0 and H2 but no H1) is representable without a sentinel.

### 3.1 Invariants

An instance is **valid** iff all of the following hold. `core.py` MUST enforce
these at construction and MUST NOT permit an invalid instance to exist.

| # | Invariant | Rationale |
|---|---|---|
| I1 | `len(births) == len(deaths) == len(dims)` | structural |
| I2 | `births`, `deaths` are `float64`; `dims` is `int32` | §6 |
| I3 | `dims >= 0` | homological degree |
| I4 | `births` are all finite and non-`NaN` | a class that is never born is not a class |
| I5 | `deaths` are non-`NaN`; `+inf` permitted; `-inf` forbidden | §5 |
| I6 | `deaths >= births` elementwise | definitional |

**I6 is checked exactly, not within tolerance.** A backend that returns
`death < birth` has a bug, and we surface it rather than absorb it. Observed
floating-point violations are a real occurrence at the 1e-16 level in some
filtration code; the adapter (not the core type) is the correct place to clamp,
and it MUST warn when it does.

### 3.2 Accessors

```python
d.dim(k)          # -> PersistenceDiagram, the sub-diagram of degree k
d.dimensions      # -> sorted unique degrees actually present
d.essential       # -> bool mask, deaths == inf
d.finite          # -> PersistenceDiagram, essential bars removed
d.persistence     # -> deaths - births  (inf for essential bars)
d.n_bars          # -> int
```

`d.dim(k)` is canonical. `d.h0` / `d.h1` are provided as **aliases that emit a
`DeprecationWarning` from the first release** — they are ergonomic for ordinary
persistence and meaningless for the multiparameter case (Paper V), and we are
not going to be able to remove them later if we ship them unmarked.

`d.dim(k)` for a `k` not present MUST return an empty diagram, not raise. Empty
is a legitimate answer to "what are the 7-dimensional cycles".

---

## 4. Batch semantics

Every numerical function in `core/` and `castle/` takes a **leading batch
dimension** (onboarding §9.3). For diagrams, the batch container is:

```python
class DiagramBatch:          # a sequence of PersistenceDiagram
    def __len__(self) -> int: ...
    def __getitem__(self, i) -> PersistenceDiagram: ...
```

**The batch MUST be ragged — a sequence of diagrams, not a dense padded
array.** Diagrams in a batch have different numbers of bars, and the only two
ways to make them rectangular are padding and truncation. Both are lossy, and
we have direct evidence of the damage:

> giotto-tda pads a batch to a common row count using rows of the form
> `(b, b, dim)` where `b` is an arbitrary real birth value drawn from the data.
> A padded row is therefore **byte-identical in form to a genuine
> zero-persistence bar** and cannot be distinguished from one. Measured
> consequence: a point cloud that yields 2 one-dimensional bars when transformed
> alone yields 11 when transformed in a batch alongside a second point cloud —
> 9 of them padding. *The diagram of a dataset depends on what else was in the
> batch.* See Appendix A.2.

That is a correctness bug that a dense representation makes almost inevitable.
We do not adopt the representation that causes it.

Functions that genuinely need a rectangular buffer (a vectorisation feeding a
tensor op) MUST perform the padding internally, at the point of use, with an
explicit mask returned alongside — never in the interchange type.

---

## 5. Infinite bars

**Essential bars are stored as `deaths[i] == np.inf`. A finite sentinel MUST NOT
be used, and the essential set MUST NOT be silently discarded.**

This is the single most consequential decision in the document, so the rationale
is spelled out.

The alternatives, and why they lose:

| Convention | Problem |
|---|---|
| Replace `inf` with the max filtration value | Unrecoverable. The bar is now indistinguishable from a genuine bar that happened to die at that value. |
| Replace `inf` with a large sentinel (`1e9`, `99`) | Same, plus it silently corrupts any distance or vectorisation that treats it as a real number. |
| Drop essential bars | Discards the rank of the homology of the underlying space — for H0, the number of connected components. |
| Separate `essential` array | Defensible, but it splits every operation into two code paths and makes `len()` ambiguous. |

`inf` is what GUDHI and Ripser both return natively (Appendix A.1), it is
representable in float64, it propagates correctly through comparisons, and it is
the only choice from which the other conventions can be *derived* on demand.

**Finitisation is an explicit, caller-requested operation:**

```python
d.finitize(at="max_finite_death")   # or at=<float>, or at="drop"
```

It returns a new diagram, records the substitution in `meta.provenance`, and is
never applied implicitly by an adapter, a constructor, or an I/O routine.

### 5.1 What backends actually do

Measured, not recalled (Appendix A.1):

| Backend | Essential bars |
|---|---|
| GUDHI | `inf` in the death column. Faithful. |
| Ripser | `inf` in the death column. Faithful. |
| persim | Consumes `(n,2)` arrays; no opinion. |
| **giotto-tda** | **Silently dropped.** |

The giotto behaviour is worth stating precisely because the compat shim (§8 of
the onboarding) has to decide what to do about it. On a 40-point noisy circle,
GUDHI and Ripser both return 40 H0 bars, exactly one essential.
`VietorisRipsPersistence` returns **39** H0 rows and zero non-finite entries.
This holds for `infinity_values` set to `None`, `inf`, and `99.0` alike — the
parameter is recorded in `infinity_values_` and does not change the output. The
essential class is not clamped, not flagged, not recoverable. It is gone.

**Adapter consequence.** `from_giotto` MUST set
`meta.provenance["essential_bars"] = "lost_upstream"` and MUST warn once per
call. It MUST NOT fabricate an essential bar to compensate — we do not know its
birth value in general, and inventing one is worse than recording the loss.

---

## 6. Dtype, precision, and equality

### 6.1 Storage

`float64` for coordinates, `int32` for dimensions. Not negotiable at the type
level: mixed-precision diagrams make cross-backend comparison undefined, and
`int32` for a homological degree is already absurdly generous.

Adapters MUST upcast `float32` input rather than reject it, and MUST record the
input dtype in `meta.provenance["source_dtype"]`.

### 6.2 Precision is not what it looks like

**Ripser returns `float64` arrays containing `float32`-precision values.**
Measured on the same point cloud, Ripser and GUDHI agree on the same H1 bars to
`2.7e-8` — consistent with `float32` epsilon at that scale (`2.0e-7`), and about
`1e8` times worse than `float64` epsilon. The arrays are `dtype('float64')`;
the *values* are not. Appendix A.3.

The consequence is unavoidable and MUST be documented at every comparison
surface: **two diagrams of the same data from two backends will never be
exactly equal.** Any test, any equality check, any deduplication that assumes
otherwise is broken by construction.

### 6.3 Equality

Two levels, and they MUST be separate methods with different names. Conflating
them is how a tolerance silently becomes load-bearing in a statistical test.

```python
d1 == d2               # exact: same multiset of bars, bit-identical coordinates
d1.allclose(d2, rtol=1e-9, atol=0.0)   # approximate
```

Both are **order-insensitive** (§7). Both compare `dims` exactly. `==` is for
same-provenance round-trips — serialize, load, compare. `allclose` is for
cross-backend agreement, and its default `rtol` is deliberately *tighter* than
Ripser's `float32` reality: a cross-backend test MUST pass an explicit
`rtol=1e-6` and thereby state in its own source that it is tolerating a
precision difference. A default that silently absorbs `1e-6` would hide genuine
disagreement everywhere else.

`inf == inf` compares equal at both levels. `NaN` cannot occur (I4, I5).

---

## 7. Ordering

**No ordering is guaranteed. Bars are a multiset.**

Backends disagree, and the disagreement is not stable. On identical input GUDHI
returns H1 bars sorted ascending by birth; Ripser returns the same two bars in
the opposite order (Appendix A.3). Neither documents an ordering guarantee, and
depending on either is a latent bug.

Therefore:

- Every consumer MUST treat row order as arbitrary.
- Equality (§6.3) MUST be computed on the multiset.
- Adapters MUST NOT sort. Preserving backend order costs nothing and makes
  adapter round-trip tests sharper.

**Canonical order** exists solely for serialization determinism and diffable
output:

```python
d.canonical()   # sort by (dim, birth, death) ascending; stable
```

The on-disk format (§10) is written in canonical order, so byte-identical
diagrams produce byte-identical files and a content hash is meaningful.
`canonical()` is a presentation concern and MUST NOT be assumed by any
numerical routine.

---

## 8. Metadata and provenance

Metadata is not decoration. A diagram whose filtration and scale are unknown
cannot be interpreted, and the reproducibility hash (execution plan §3.9) is a
stated adoption commitment.

```python
@dataclass(frozen=True)
class DiagramMeta:
    filtration:      str | None   # "rips" | "alpha" | "cubical" | "lower_star" | ...
    backend:         str | None   # "gudhi" | "ripser" | "giotto" | "persim" | "array"
    backend_version: str | None   # as reported by the backend at adapter time
    coeff_field:     int | None   # e.g. 2, 3 — affects the diagram, must be recorded
    params:          Mapping[str, Any]  # max_edge_length, max_dimension, ...
    provenance:      Mapping[str, Any]  # adapter-recorded facts; see below
    space:           str | None   # free-text description of the underlying data
```

All fields are optional — a diagram typed in by hand from a paper is a valid
diagram — but `from_*` adapters MUST populate `backend`, `backend_version`, and
`provenance`.

`provenance` is the honest-accounting channel. Reserved keys:

| Key | Meaning |
|---|---|
| `essential_bars` | `"faithful"` \| `"lost_upstream"` \| `"finitized_at:<value>"` |
| `source_dtype` | dtype of the input array |
| `clamped_rows` | count of `death < birth` rows the adapter repaired |
| `padding_removed` | count of trivial rows stripped as suspected batch padding |
| `order` | `"backend"` \| `"canonical"` |

**`meta` MUST NOT participate in `==` or `allclose`.** Two diagrams with the
same bars from different backends are the same diagram. Provenance is recorded
so a human can audit it, not so equality can reject on it. `d.same_provenance(e)`
is available for the cases that genuinely care.

### 8.1 Content hash

```python
d.content_hash   # -> str, sha256 over canonical-ordered coordinates + dims
```

Covers bars only, never metadata. This is what a paper pins.

---

## 9. Delegation hazards

Akriti delegates computation and owns inference. Delegation is only safe where
the delegate is correct, and two of ours are not, in ways that produce clean
plausible numbers rather than errors. These are recorded here because
`core/distances.py` is written against this document.

### 9.1 persim returns a finite bottleneck distance between diagrams that are infinitely far apart

Measured:

```python
persim.bottleneck([[0, inf], [0.1, 0.5]], [[0, 1], [0.1, 0.5]])  # -> 0.5
```

The bottleneck distance between a diagram with an essential class and one
without is **infinite** — there is no finite-cost matching, and the essential
bar cannot be matched to the diagonal. persim returns `0.5`, the cost of the
unrelated finite bar. `persim.wasserstein` behaves comparably
(`0.707` on the same input).

**This is exactly the failure mode the project exists to prevent**: no
exception, no warning, a number of entirely reasonable magnitude, and it is
wrong. A user comparing a connected sample against a disconnected one gets a
small distance and concludes they are similar.

**Requirement on `core/distances.py`.** Before delegating, it MUST partition
both diagrams by `essential`. If the essential-bar counts differ **per
dimension**, the distance is `+inf` and MUST be returned as such without calling
the backend. If they agree, delegate on the finite parts only, and document that
the returned distance is the finite-part distance. It MUST NOT pass a diagram
containing `inf` to persim.

This is a guardrail in the sense of onboarding §7 — a negative result about a
dependency, converted into a safety feature. It should be documented publicly;
it is a genuine contribution and costs us nothing.

### 9.2 giotto-tda 0.6.2 does not run on current scikit-learn

`VietorisRipsPersistence.fit_transform` raises

```
TypeError: check_array() got an unexpected keyword argument 'force_all_finite'
```

on scikit-learn 1.8.0. The keyword was renamed in scikit-learn 1.6 and removed
in 1.8; giotto-tda has not tracked it. **The most-installed general-purpose TDA
library in Python is currently unusable on a default `pip install` of its own
declared dependency.**

Consequences:

- `from_giotto` MUST be tested against *stored fixture arrays*, not a live
  giotto call, or CI will fail for reasons that have nothing to do with us.
  Fixtures are committed with the giotto and scikit-learn versions that produced
  them.
- giotto-tda MUST NOT enter the default dependency closure. Test-only, pinned,
  in its own extra.
- This materially strengthens the case for `compat/giotto`: their users are not
  merely unmaintained, they are *broken today*. Worth a sentence in the audit
  and in the AMS talk.

*Clean-room note (onboarding §8): giotto-tda is AGPLv3. The above was determined
by calling public API and reading a traceback. No giotto source has been read,
and none may be read while implementing `compat/`.*

---

## 10. Serialization

### 10.1 Requirements

1. Round-trips exactly: `load(dump(d)) == d`, including `inf` and multiplicity.
2. Dependency-free — `numpy` plus the standard library. The default install
   closure is MIT/BSD-only and a serialization format is not a good reason to
   widen it.
3. Self-describing and versioned.
4. Deterministic: identical diagrams produce identical bytes.
5. Readable enough to inspect without our library.

HDF5 (`h5py`) and Parquet (`pyarrow`) both fail (2). Bare `.npz` fails (3) and
has no metadata story. Plain JSON fails on `inf` — it is not valid JSON, and the
`Infinity` token Python emits is a non-standard extension other languages reject.

### 10.2 Format: `.akd`

A **zip archive** — `zipfile` is standard library — containing:

```
meta.json      UTF-8 JSON, sorted keys, the DiagramMeta plus a format version
bars.npz       npz with arrays: births, deaths, dims  (canonical order, §7)
```

For a `DiagramBatch`, `bars.npz` additionally carries `offsets`, an
`int64` array of length `len(batch)+1` giving the CSR-style row range of each
diagram, and `meta.json` carries a list of per-diagram metadata. Ragged, exact,
no padding.

`inf` lives in `bars.npz`, where NumPy represents it correctly, and never in the
JSON. This is the reason for the split.

```python
akriti.diagrams.save(d, "sample.akd")
d = akriti.diagrams.load("sample.akd")
```

### 10.3 Interoperable escape hatches

Non-normative, and both MUST warn about what they lose:

- `to_arrays()` → `dict[int, np.ndarray]`, degree to `(n,2)` array. This is the
  de-facto community format (what Ripser returns and persim consumes) and is
  what people will paste into other tools.
- `to_csv()` → three columns `dim,birth,death`, with `inf` written as the
  literal `inf`. For humans and for spreadsheets.

---

## 11. Adapter contract

Signature for all five:

```python
from_gudhi(obj, **meta)   -> PersistenceDiagram
from_ripser(obj, **meta)  -> PersistenceDiagram
from_giotto(arr, **meta)  -> PersistenceDiagram | DiagramBatch
from_persim(obj, **meta)  -> PersistenceDiagram
from_array(arr, **meta)   -> PersistenceDiagram
```

Every adapter MUST: validate against §3.1; populate `backend`,
`backend_version`, `provenance`; preserve backend row order; and never
finitize, sort, or deduplicate.

Measured input formats (Appendix A):

| Source | Accepted input | Notes |
|---|---|---|
| GUDHI | `SimplexTree.persistence()` → `list[(dim, (b, d))]`; also `persistence_intervals_in_dimension(k)` → `(n,2)` | `inf` faithful. Both forms MUST be accepted; the `list` form carries all degrees at once. |
| Ripser | `ripser(X)` → `dict` with `"dgms"`; `Rips().fit_transform(X)` → `list[(n,2)]` | Index in the list *is* the degree. `inf` faithful. `float32` precision (§6.2). |
| giotto | `(n_samples, n_bars, 3)` array, columns `(birth, death, dim)` | Essential bars lost (§5.1). Padding ambiguity (§4). Returns a batch when `n_samples > 1`. |
| persim | `list[(n,2)]`, degree by index | Same shape as Ripser's `dgms`. |
| array | `(n,2)` with explicit `dim=`, or `(n,3)` with `(birth, death, dim)` | The `(n,3)` column order matches giotto's, deliberately. |

### 11.1 The giotto padding decision

`from_giotto` cannot distinguish padding from genuine trivial bars (§4). It MUST
NOT guess. Behaviour:

- Default `strip_padding=None`: keep every row, warn once if any trivial rows
  are present, record the count in `provenance["padding_removed"] = 0`.
- `strip_padding=True`: drop trivial rows, record the count.
- `strip_padding=False`: keep silently.

Dropping trivial bars is *usually* right for giotto batches and *never*
guaranteed right, so the caller decides and the file records what happened.

### 11.2 Round-trip tests

Student B's round-trip tests MUST run against **real backend output**, not
hand-written arrays — the whole value of this layer is that it survives contact
with what the backends actually emit. The suite MUST include, at minimum:

- A diagram with essential bars (GUDHI, Ripser).
- An empty diagram, and a diagram empty in one degree but not another.
- A diagram with repeated identical bars — multiplicity MUST survive.
- A diagram with a genuine zero-persistence bar.
- Cross-backend agreement GUDHI vs Ripser on the same point cloud, with an
  explicit `rtol=1e-6` and a comment pointing at §6.2.
- `save`/`load` byte-determinism: dumping twice gives identical bytes.

Property-based tests (Hypothesis) for the invariants and for
`load(dump(d)) == d`; onboarding §10 requires them for the numerical layer and
they fit this layer unusually well.

---

## 12. Open decisions

These need a call before implementation starts. My recommendation is given but
the decision is the lead's.

| # | Question | Recommendation |
|---|---|---|
| D1 | File extension `.akd`, or plain `.npz` with our layout inside? | `.akd`. A distinct extension lets us version the container and stops people from `np.load`-ing it and getting a confusing partial answer. |
| D2 | Is `DiagramBatch` in scope for M1, or does M1 ship the single-diagram type only? | In scope. Retrofitting a batch container after `core/` is written against scalars is the expensive order, and §9.3 of the onboarding commits us to batch-shaped signatures. |
| D3 | Do we accept `float32` storage behind a flag for large-scale work? | No, not in v0. Revisit when a real memory complaint exists. |
| D4 | Should `from_giotto` default to `strip_padding=True`? | No. Defaulting to a lossy repair contradicts §5's whole argument. Warn and let the caller choose. |
| D5 | Does the RFC published at M1 include §9's delegation hazards, or do we raise them upstream first? | Raise upstream first — file the persim issue and the giotto scikit-learn issue, then publish citing our own reports. Costs two weeks, buys enormous goodwill, and turns a criticism into a contribution. |

---

## Appendix A — Measured evidence

Every claim in this document was measured on 2026-07-29 with
`gudhi 3.11.0`, `ripser 0.6.14`, `persim 0.3.8`, `giotto-tda 0.6.2`,
`numpy 2.4.4`, `scikit-learn 1.8.0`, Python 3.12.11. Reproduction script:
`rfcs/evidence/probe_backends.py`.

Input: 40 points sampled uniformly on the unit circle with Gaussian noise
`σ = 0.05`, `numpy` default_rng seed 0.

### A.1 Essential bars

| Backend | H0 bars | Essential | H1 bars |
|---|---|---|---|
| GUDHI (Rips) | 40 | 1 (`inf`) | 2 |
| Ripser | 40 | 1 (`inf`) | 2 |
| giotto (`infinity_values=None`) | 39 | 0 | 2 |
| giotto (`infinity_values=inf`) | 39 | 0 | 2 |
| giotto (`infinity_values=99.0`) | 39 | 0 | 2 |

GUDHI `persistence()` returns `list[tuple[int, tuple[float, float]]]`, e.g.
`(0, (0.0, inf))`. `persistence_intervals_in_dimension(k)` returns a C-contiguous
`(n,2)` `float64` array.

### A.2 giotto batch padding

| Transform | rows | H0 | H1 | trivial rows |
|---|---|---|---|---|
| `A` alone | 41 | 39 | 2 | 0 |
| `B` alone | 50 | 39 | 11 | 0 |
| `[A, B]` batched | 50 each | 39 | 11 | 9 (in `A`) |

Padding rows in `A` are `(0.09452353, 0.09452353, 1.0)` — a real birth value
from `A`'s own data, not a zero or a sentinel.

### A.3 Precision and ordering

Same H1 bars, both backends, raw order as returned:

```
ripser: [[0.52018976, 1.6952107 ],     gudhi: [[0.09452353, 0.09486296],
         [0.09452353, 0.09486296]]              [0.52018979, 1.69521069]]
```

Order differs. Max coordinate difference after sorting: `2.69e-8`.
`float32` eps at this scale: `2.02e-7`. `float64` eps at this scale: `3.76e-16`.

The difference is ~71× smaller than `float32` eps and ~7×10⁷ times larger than
`float64` eps. Ripser is computing in single precision.

### A.4 persim on essential and empty diagrams

| Inputs | `bottleneck` | `wasserstein` | Correct? |
|---|---|---|---|
| `[[0,inf],[.1,.5]]` vs itself | 0.0 | 0.0 | yes |
| `[[0,inf],[.1,.5]]` vs `[[0,1],[.1,.5]]` | **0.5** | **0.707** | **no — should be `inf`** |
| empty vs empty | 0.0 | 0.0 | yes |
| empty vs `[[0,1],[.1,.5]]` | 0.5 | 0.990 | yes |

---

## Appendix B — Changelog

- **2026-07-29** — Initial draft.
