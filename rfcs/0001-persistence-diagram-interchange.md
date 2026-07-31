# RFC-0001 — Persistence Diagram Interchange

|                 |                                                                 |
| -----------------| -----------------------------------------------------------------|
| **Status**      | Draft — not yet open for public comment                         |
| **Author**      | Sushovan Majhi                                                  |
| **Edited By**   | A. D. Silberman                                                 |
| **Created**     | 2026-07-29                                                      |
| **Last Edited** | 2026-07-31                                                      |
| **Target**      | M0 (2026-08-01) drafted · M1 (2026-09-15) published for comment |
| **Implements**  | `akriti.diagrams`                                               |

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
    dims:   Array
    births: Array      # shape (n,), float64
    deaths: Array      # shape (n,), float64, may contain +inf
    meta:   DiagramMeta

    @property
    def xp(self):
        return self.dims.__array_namespace__()
```

`Array` is **any object implementing `__array_namespace__`** — the Python array
API standard — not `np.ndarray`. NumPy is the expected and default backend; it
is not the required one.

This was `np.ndarray` in the first draft, which was wrong. The onboarding
document requires `core/` to be written against the array API rather than
hard-coding NumPy, and `PersistenceDiagram` is the input to every function in
`core/`. A container that pins NumPy makes a framework-agnostic `core/`
unachievable no matter how `core/` itself is written, and retrofitting the
container later is exactly the expensive case that requirement exists to avoid.
§3.3 states what this does and does not promise.

`xp` is a derived **property**, not a fourth stored field, deliberately. An
earlier draft stored it alongside `dims`/`births`/`deaths`, which creates a
fourth piece of state that has to be kept in sync with the other three at
every construction site, including the views `DiagramBatch.__getitem__`
returns (§4.2), with nothing enforcing the agreement. I7 already requires
`dims`, `births`, and `deaths` to share one namespace; deriving `xp` from
`dims` makes disagreement structurally impossible rather than merely
prohibited. Call sites that want the short spelling get `d.xp`; nothing
about validity depends on a value that could drift.

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
| I2 | `births`, `deaths` are `float64`; `dims` is `int32` — tested with `xp.isdtype`, never against NumPy dtype objects | §6 |
| I3 | `dims >= 0` | homological degree |
| I4 | `births` are all finite and non-`NaN` | a class that is never born is not a class |
| I5 | `deaths` are non-`NaN`; `+inf` permitted; `-inf` forbidden | §5 |
| I6 | `deaths >= births` elementwise | definitional |
| I7 | all three arrays share one namespace — `births.__array_namespace__() is deaths.__array_namespace__() is dims.__array_namespace__()` | §3.3 |
| I8 | `PersistenceDiagram` is immutable after construction — no method may write to `dims`, `births`, or `deaths` in place, and none may rebind them | §4.2 |

**I6 is checked exactly, not within tolerance.** A backend that returns
`death < birth` has a bug (excluding extended persistence), and we surface it 
rather than absorb it. Observed floating-point violations are a real occurrence 
at the 1e-16 level in some filtration code; the adapter (not the core type) is 
the correct place to clamp, and it MUST warn when it does.

**I8 exists because §4.2 already assumes it.** `DiagramBatch.__getitem__`
returns a `PersistenceDiagram` whose arrays are views into the batch's shared
buffer, not copies, and that is only safe if nothing can write through one
view and corrupt a sibling diagram or the batch itself. Every method that
looks like a mutation (`finitize`, anything in §3.2) MUST construct and return
a new `PersistenceDiagram` rather than modify `self`. This should be enforced
the same way `DiagramMeta` already is (`@dataclass(frozen=True)`, §8), or
documented as an equivalent guarantee if the array API standard's read-only
view support is used instead.

### 3.2 Accessors

```python
d.dim(k)          # -> PersistenceDiagram, the sub-diagram of degree k
d.dimensions      # -> sorted unique degrees actually present
d.essential       # -> bool mask, deaths == inf
d.finite          # -> PersistenceDiagram, essential bars removed
d.persistence     # -> deaths - births  (inf for essential bars)
d.n_bars          # -> int
```

`d.dim(k)` is canonical. If we later provide `d.h0` / `d.h1`, these would be
**aliases that emit a `DeprecationWarning` from the first release** — 
ergonomic for ordinary persistence alone and meaningless for the 
multiparameter case (Paper V), and we are not going to be able to remove them 
later if we ship them unmarked.

`d.dim(k)` for a `k` not present MUST return an empty diagram, not raise. Empty
is a legitimate answer to "what are the 7-dimensional cycles".

### 3.3 What array-API support does and does not promise

Being namespace-agnostic is not free, and promising more than the standard
delivers would be worse than promising nothing. Three limits, all measured
against `array_api_strict` 2.6.1, the conformance reference:

**`lexsort` is not in the standard.** NumPy has it, and because
`np.ndarray.__array_namespace__()` returns the `numpy` module itself, a naive
`hasattr(xp, "lexsort")` check passes and hides the problem. It is absent from
`array_api_strict`. The canonical ordering in §7 MUST therefore be built from
successive **stable** `argsort` passes on the least-significant key first
(`death`, then `birth`, then `dim`), not from a single `lexsort`.

**Filtering produces data-dependent shapes.** `d.finite`, `d.dim(k)` and any
boolean-mask selection give an output shape that depends on the *values* in the
array. The standard permits this on eager backends and explicitly does not
guarantee it on lazy or JIT ones — under `jax.jit` these operations fail. They
are therefore **eager-only accessors**, and MUST be documented as such. They are
not available inside a traced or compiled region.

This is a real constraint on the neural-network path, and it is better to know
now: a topological layer inside a network cannot call `d.finite`. It must
operate on the full arrays with a mask, which is why §5 keeps `essential` as a
derivable mask rather than splitting the storage.

**Serialization is NumPy-bound, deliberately.** `io.py` (§10) writes `.npz`, so
it converts at the I/O boundary via `np.asarray` and returns NumPy-backed
diagrams on load. Serialization is not a numerical kernel and there is nothing
to gain from making it generic. The conversion MUST be at the boundary only —
never in the constructor, and never in an adapter.

Being NumPy-bound at that boundary does not make `numpy` a dependency of the
package. `diagrams/core.py` and `diagrams/adapters.py` MUST import nothing
beyond the standard library; they operate entirely through
`__array_namespace__` on whatever the caller already has. `numpy` is used
**only** inside `save`/`load`, imported lazily at the top of those two
functions, not at module scope, so `import akriti.diagrams` and everything
except serialization work with zero third-party dependencies. If `numpy` is
absent when `save`/`load` is actually called, that MUST raise a clear,
actionable `ImportError` naming `numpy` as the missing piece, not a bare
traceback. See §10.1 and D10.

**Adapters preserve the input namespace.** `from_*` MUST NOT force-convert to
NumPy. A diagram built from torch tensors stays torch-backed. What adapters
convert is *dtype* (§6.1), not namespace.

**Conformance is tested, not intended.** CI runs the diagram test suite against
`array_api_strict`, which rejects any NumPy-only call. A requirement of this kind
that is merely written down decays within weeks; the first draft of this RFC
hard-coded NumPy while the onboarding document forbade it, and nobody noticed
until a reviewer read both.

---

## 4. Batch semantics

Every numerical function in `core/` and `castle/` takes a **leading batch
dimension** (onboarding §9.3). For diagrams, the batch container is:

```python
class DiagramBatch:          # a sequence of PersistenceDiagram
    def __len__(self) -> int: ...
    def __getitem__(self, i) -> PersistenceDiagram: ...
```

This is an interface contract, not a storage commitment. §4.2 specifies the
underlying representation.

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

### 4.1 Reconciling the two-type design

This section exists because an earlier design pass considered folding
`PersistenceDiagram` and `DiagramBatch` into a single type, batch-of-one by
default, with the batch realized as a dense `(batch, max_points, 2)` array plus
a boolean mask. That proposal is not adopted, and the reason is Appendix A.2,
not taste. A dense padded batch cannot represent giotto's own output without
the padding rows becoming indistinguishable from genuine trivial bars, and the
measured consequence, a diagram changing shape depending on what else is in the
batch, is exactly the class of bug this document exists to prevent. So
`DiagramBatch` is a ragged sequence, and it stays that way at the interchange
boundary.

The dense, padded representation is not wrong, only misplaced. The paragraph
above already says a function needing a rectangular buffer must build it
internally and return an explicit mask alongside. That is the padding+mask
scheme, deliberately scoped to computation rather than storage. It is the right
shape for a `castle/` routine feeding an array-API vectorized op, or a
topological layer inside a network; it is the wrong shape for the type that
`save`, `load`, and every adapter hand back.

Nor does the two-type split reintroduce duplicated implementation.
`DiagramBatch` owns no numerical or invariant logic of its own: `dim(k)`,
`persistence`, equality, and invariants I1 through I7 are all written once,
against `PersistenceDiagram`, and `DiagramBatch.__getitem__` returns a
`PersistenceDiagram`, not a different type. A batch of one diagram is not a
special case anywhere in `core/`; it is a `DiagramBatch` of length one,
wrapping the same object every other code path uses. §4.2 specifies how that
wrapping is implemented in memory.

### 4.2 `DiagramBatch` storage representation

The interface in §4 is silent on layout, and silence here has a cost: the
natural reading, N independently allocated `PersistenceDiagram` objects held
in a Python list, is also the one that will not scale well if a neural-network
path is built on top of `castle/` later. This section closes that gap.

**`DiagramBatch` MUST store its diagrams in one shared, concatenated buffer,
laid out identically to §10.2's on-disk format:**

```python
class DiagramBatch:
    dims:    Array   # shape (total_bars,), concatenated across diagrams
    births:  Array   # shape (total_bars,)
    deaths:  Array   # shape (total_bars,)
    offsets: Array   # shape (len(batch)+1,), int64, CSR row pointers
    metas:   Sequence[DiagramMeta]   # one per diagram

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, i) -> PersistenceDiagram:
        lo, hi = self.offsets[i], self.offsets[i + 1]
        # a view: dims[lo:hi], births[lo:hi], deaths[lo:hi], no copy
        ...
```

`__getitem__` MUST return a **view**: a `PersistenceDiagram` whose arrays are
slices into the batch's own buffers, not a copy. This is safe only because
`core.py` gives `PersistenceDiagram` no mutating methods (I8, §3.1); there is
no aliasing hazard from two objects sharing memory when neither can be written
to after construction.

**`offsets` has its own invariants, and `core.py` MUST enforce them at
`DiagramBatch` construction, the same way §3.1 enforces I1 through I8 for a
single diagram:**

| # | Invariant |
|---|---|
| B1 | `len(offsets) == len(batch) + 1` |
| B2 | `offsets[0] == 0` |
| B3 | `offsets[-1] == total_bars` (i.e. `len(dims)`) |
| B4 | `offsets` is non-decreasing |
| B5 | `offsets.__array_namespace__()` matches `dims`, `births`, `deaths` (§3.3) |

Without B1 through B5, `__getitem__`'s slice arithmetic can silently read the
wrong range, or read past the end of the buffer, which is the same category of
silent-wrongness bug §9 exists to catch, just self-inflicted this time rather
than a backend's fault.

One buffer, rather than N, is what keeps a future DL path cheap: building a
training minibatch or moving one to device becomes a small, fixed number of
array operations rather than a Python-level loop over N objects. It costs
nothing at TDA's usual per-diagram bar counts, and it is the same shape §10.2
already committed to for the on-disk format; this extends that commitment to
memory.

**Why not go further and merge `PersistenceDiagram` and `DiagramBatch` into one
CSR-backed type**, with `offsets = [0, n]` as the single-diagram case? This was
considered, since it satisfies onboarding §9.3's leading-batch-dimension rule
more literally than a two-type split does, and it was rejected on two points
specific to this RFC, not on API-surface taste:

- **`DiagramMeta` (§8) is genuinely per-diagram.** `backend`,
  `backend_version`, `params`, and `provenance` can all differ across a batch.
  A merged type forces `meta` to become a sequence the moment `offsets` has
  more than one row, so every consumer of `.meta` must branch on batch size.
  Keeping `meta` a single dataclass, true only on the unbatched type, is a
  property worth keeping.
- **`content_hash` (§8.1) is defined over one diagram's canonical bars.** A
  merged type has to either forbid calling it on a multi-diagram instance or
  redefine it as a hash-of-hashes, and either choice is new specification this
  RFC does not otherwise need.

Precedent: PyTorch Geometric solves the identical problem, ragged, per-item
structure needing efficient batched storage, with two types: `Data` for one
graph, `Batch` for many, concatenated storage plus an index vector, rather
than one self-batching type. `Batch` subclasses `Data` in PyG; we do not adopt
that inheritance here, since a subclass that fails to override even one
accessor (`persistence`, `==`) would silently compute it across batch
boundaries instead of raising, which is precisely the "clean, plausible, wrong
answer" failure category §9 exists to rule out. `DiagramBatch` and
`PersistenceDiagram` are related by composition (the view relationship above),
not inheritance.

**Construction from ordinary adapter output.** §11's adapters return one
`PersistenceDiagram` per call, except `from_giotto`, which is only pre-batched
because giotto's own input already is. The common path, N separate
`from_gudhi` / `from_ripser` calls that need to become one `DiagramBatch` for
a bootstrap or permutation test, has no constructor yet. `DiagramBatch` MUST
provide one:

```python
@classmethod
def from_diagrams(cls, diagrams: Sequence[PersistenceDiagram]) -> "DiagramBatch":
    ...  # concatenate dims/births/deaths, derive offsets from each length,
         # collect .meta into metas, in input order
```

This is also where the concatenate-and-derive-`offsets` logic from the class
body above is actually exercised; nothing else in this document currently
calls it.

---

## 5. Infinite bars

**Essential bars are stored as `deaths[i] == xp.inf`. A finite sentinel MUST NOT
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
| **giotto-tda** | **One H0 class dropped by design (`reduced_homology=True`, default).** |

The giotto behaviour is worth stating precisely because the compat shim (§8 of
the onboarding) has to decide what to do about it. On a 40-point noisy circle,
GUDHI and Ripser both return 40 H0 bars, exactly one essential.
`VietorisRipsPersistence` returns **39** H0 rows and zero non-finite entries.
This holds for `infinity_values` set to `None`, `inf`, and `99.0` alike, and
the reason is not that giotto mishandles infinite death times. `infinity_values`
governs those correctly. The cause is `reduced_homology`, `True` by default:
reduced homology is defined to omit exactly the one class every diagram has,
the class recording that the space is nonempty, and for H0 that is precisely
the single essential (never-dying) bar. `infinity_values` never gets a chance
to act on it, because it is excluded upstream of that logic entirely, not
because the flag is ignored for essential bars in general.

**Adapter consequence.** `reduced_homology` is a raw fact of the original
call, the same category as `max_edge_length` or `max_dimension`, so
`from_giotto` MUST record it as `meta.params["reduced_homology"]` rather than
folding it into `provenance` directly. `provenance["essential_bars"]` MUST
then be **derived** from that value, not set independently: `"lost_upstream"`
when `params["reduced_homology"]` is `True`, `"faithful"` when `False`. The
adapter cannot recover this from the output array alone, a filtration
truncated by `max_edge_length` can also show zero H0 essential bars under
`reduced_homology=False`, so it MUST come from the caller.

**`reduced_homology` is a required, keyword-only parameter, not a default
with a warning:**

```python
from_giotto(arr, *, reduced_homology, **meta) -> PersistenceDiagram | DiagramBatch
```

Omitting it MUST raise, not fall back to giotto's own default. A default of
`assume True, warn once` was considered and rejected: §9.1's own evidence
script suppressed a warning and produced a wrong conclusion until a reviewer
caught it, which is direct, measured proof within this document that a missed
warning here would leave no trace. The two ways a wrong assumption could go
are not symmetric, assuming `True` when the caller actually passed `False`
mislabels a faithful diagram as `"lost_upstream"`, which is over-cautious
rather than silently confident, and does not corrupt `core/distances.py`'s
`essential` partitioning (§9.1), since that reads the actual death values, not
`provenance`. But §8 defines provenance's entire purpose as being auditable by
a human, and §9.1 already treats a misleading-but-technically-defensible
signal as a real bug rather than a tolerable edge case, so letting this one
slide on the grounds that it fails safe would be inconsistent with that
standard. Raising costs the caller one line, `reduced_homology=vr.reduced_homology`
off the already-fitted transformer, and removes the failure mode entirely
rather than making it merely unlikely.

This loss is **H0-only.** Reduced homology is defined to remove exactly the
one class recording that the space is nonempty, and that class exists only in
dimension 0. A diagram can separately carry a genuine essential class in H1
(e.g. one that survives past a truncated `max_edge_length`), and that class is
untouched by `reduced_homology`; it goes through the ordinary `infinity_values`
path and is faithful regardless of the H0 outcome. `"lost_upstream"` is
therefore a claim about H0 specifically, not the whole diagram, and should be
read that way even though `provenance` records one value per diagram.

`from_giotto` MUST NOT fabricate an essential bar to compensate for a
`reduced_homology=True` loss. It is tempting to infer the missing bar's birth
as `0`, since that is what the measured example shows, but that is a
coincidence of the example, not a property of the mechanism. By the elder
rule, the surviving H0 class's birth is the **minimum vertex birth across the
whole point cloud**, and that vertex is exactly the one whose class was
dropped, so it is structurally absent from the 39 remaining rows; the minimum
of what remains is the second-smallest birth overall, not the true minimum,
whenever vertex births actually differ. They tie at `0` only for an unweighted
Rips complex. `VietorisRipsPersistence`'s `weights` parameter (DTM-based
weighting) breaks that tie in the ordinary case, and "reconstruct as `0`"
would then be silently wrong on the first weighted call, exactly the
clean-plausible-wrong failure category this document exists to catch.

---

## 6. Dtype, precision, and equality

### 6.1 Storage

`float64` for coordinates, `int32` for dimensions. Not negotiable at the type
level: mixed-precision diagrams make cross-backend comparison undefined, and
`int32` for a homological degree is already absurdly generous.

Dtypes are the **namespace's own** `xp.float64` and `xp.int32`, not
`np.float64` / `np.int32`. Checks MUST use `xp.isdtype(a.dtype, "real floating")`
or equality against `xp.float64`; comparing against a NumPy dtype object breaks
on every non-NumPy backend and is the most likely way for NumPy to creep back in
unnoticed.

Adapters MUST upcast `float32` input rather than reject it, and MUST record the
input dtype in `meta.provenance["source_dtype"]`. Upcasting is a *dtype*
conversion within the input's namespace, never a conversion to NumPy (§3.3).

### 6.2 Precision is not what it looks like

**Ripser returns `float64` arrays containing `float32`-precision values.**
Measured on the same point cloud, Ripser and GUDHI agree on the same H1 bars to
`2.7e-8` — consistent with `float32` epsilon at that scale (`2.0e-7`), and about
`1e8` times worse than `float64` epsilon. The arrays are `dtype('float64')`;
the *values* are not. Appendix A.3.

The consequence is unavoidable and MUST be documented at every comparison
surface: **two diagrams of the same data from two backends will never be
exactly equal.** Any test or equality check that assumes otherwise is broken 
  by construction.

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

**`DiagramBatch` equality is order-sensitive across diagrams**, unlike bar
equality within one diagram. `b1 == b2` requires `len(b1) == len(b2)` and
`b1[i] == b2[i]` (§6.3's exact form) for every `i` in sequence; `b1.allclose(b2, ...)`
is the same but per-element approximate. Diagram 3 in one batch is not
interchangeable with diagram 5 in another just because their bars match, since
batch position is meaningful (§4's "leading batch dimension" is a positional
axis, not a set), while bar order within a single diagram explicitly is not
(§7). Do not confuse the two: a batch is order-sensitive over an
order-insensitive thing.

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

**It MUST NOT be implemented with `lexsort`.** That function is not part of the
array API standard (§3.3) — it exists in NumPy, and the `hasattr` check that
would catch its absence passes spuriously because NumPy's `__array_namespace__`
returns NumPy itself. Compose it instead from stable `argsort` passes, least
significant key first:

```python
order = xp.argsort(deaths, stable=True)
order = xp.take(order, xp.argsort(xp.take(births, order), stable=True))
order = xp.take(order, xp.argsort(xp.take(dims,   order), stable=True))
```

Stability is what makes the composition correct; an unstable sort at any step
loses the ordering established by the previous one. Use `xp.take` rather than
integer-array indexing — both work under `array_api_strict`, but `take` is the
form the standard specifies for gathering.

Verified against `np.lexsort` on 200 bars with deliberate ties in every column:
identical ordering, valid permutation.

The on-disk format (§10) is written in canonical order, so byte-identical
diagrams produce byte-identical files and a content hash is meaningful.
`canonical()` is a presentation concern and MUST NOT be assumed by any
numerical routine.

**Canonicalizing a `DiagramBatch` is not the same operation applied to the
concatenated buffer.** The three-pass `argsort` above, run unmodified against
§4.2's shared `dims`/`births`/`deaths` arrays, would sort across diagram
boundaries: rows from different diagrams could interleave, and `offsets` would
no longer point at the correct ranges after reindexing. `DiagramBatch.canonical()`
MUST instead sort **within each segment independently** —
`[offsets[i], offsets[i+1])` for every `i` — and MUST NOT reorder the segments
themselves. Diagram order in a batch is meaningful (it is a dataset's item
order, §4's leading batch dimension) in a way bar order within one diagram
explicitly is not; canonicalization must not conflate the two. This is also
what makes §10.1's determinism requirement true for a batch, not just a single
diagram: "identical diagrams produce identical bytes" only holds if the sort
respects segment boundaries.

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
    params:          Mapping[str, Any]  # max_edge_length, max_dimension,
                                         # reduced_homology, ...
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

**`essential_bars` is derived, never independently authored.** For a
giotto-sourced diagram it is a pure function of
`params["reduced_homology"]` (§5.1), and it would be fair to ask why both are
stored when one determines the other. Two reasons, not one:

- **`provenance` is meant to be backend-agnostic; `params` is not.** §1's whole
  premise is that a consumer, or `core/distances.py` (§9.1), should be able to
  ask "can I trust this diagram's essential set" without knowing which of four
  backends produced it or which of that backend's flags is responsible. A
  fifth backend with its own idiosyncratic reason for dropping an essential
  bar will have its own `params` key, not `reduced_homology`; `essential_bars`
  is the field that means the same thing regardless.
- **`params` cannot represent every writer.** `d.finitize()` (§5) rewrites
  this same field on any diagram, including GUDHI- or Ripser-sourced ones that
  never had a `reduced_homology` key in the first place. `essential_bars` has
  to answer the same question for every diagram's whole life, not just at
  adapter time for one backend.

Both writers, `from_giotto` at construction and `finitize()` later, MUST be
the only places that set this key, so the derived value and its source never
have the chance to drift apart.

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

persim does **not** do this silently. It emits

```
UserWarning: dgm1 has points with non-finite death times;ignoring those points
```

and that changes how the problem should be described. It is not a
silent-wrongness bug; it is a **severity-mismatch** bug. The warning states the
mechanism accurately — points are being dropped — but not the consequence, which
is that the returned value is not the bottleneck distance and that no finite
value is. It reads as a routine preprocessing note rather than "this answer is
wrong."

It also travels badly. Being a `warnings` warning, it is shown once per location
under the default filter, is absent from most logs, and is erased entirely by
the `warnings.filterwarnings("ignore")` that sits near the top of a great many
scientific Python scripts — including, until this was caught, our own evidence
script.

So a user comparing a connected sample against a disconnected one still receives
a small distance and still concludes they are similar. They get one line on
stderr first, if nothing upstream has turned warnings off.

> **Correction, 2026-07-30.** The first draft of this section asserted that
> persim gave no warning at all. That was wrong: the measurement behind it was
> taken with warnings globally suppressed. The evidence script no longer
> suppresses warnings, and `tests/test_rfc0001_backend_claims.py` now asserts
> that the warning *is* raised, so this cannot drift again.

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
2. Dependency-free — the default install (`diagrams/core.py`,
   `diagrams/adapters.py`) is **solely the interchange layer**, written
   against `__array_namespace__`, and carries no third-party dependency at
   all, not even `numpy`. `numpy` is needed only inside `io.py`'s `save`/
   `load`, lazily imported there (§3.3), because `.npz` is a NumPy-native
   format with no array-API equivalent. A serialization format is not a good
   reason to widen the *package's* dependency closure beyond that boundary.
3. Self-describing and versioned.
4. Deterministic: identical diagrams produce identical bytes.
5. Readable enough to inspect without our library.

HDF5 (`h5py`) and Parquet (`pyarrow`) both fail (2) as the **default** format.
Bare `.npz` fails (3) and has no metadata story. Plain JSON fails on `inf` —
it is not valid JSON, and the `Infinity` token Python emits is a non-standard
extension other languages reject. (Parquet is still available as an
optional, `pyarrow`-gated escape hatch — §10.3, D8 — this requirement rules
out only what ships by default.)

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

Non-normative, and all three MUST warn about what they lose:

- `to_arrays()` → `dict[int, xp.array]`, degree to `(n,2)` array. This is the
  de-facto community format (what Ripser returns and persim consumes) and is
  what people will paste into other tools.
- `to_csv()` → three columns `dim,birth,death`, with `inf` written as the
  literal `inf`. For humans and for spreadsheets.
- `to_parquet()` → a `pyarrow.Table` with the same three columns and order as
  `to_csv()` (`dim` int32, `birth`/`death` float64), so `inf` round-trips
  exactly — Parquet's `double` is IEEE 754, unlike JSON's. Requires
  `pip install akriti[parquet]  # pyarrow (Apache 2.0)` (D8); `pyarrow` MUST
  be a lazy, function-scoped import inside `to_parquet()`, with a friendly
  `ImportError` if missing — the same pattern `save`/`load` already uses for
  `numpy` (§10.1, D10). For a `DiagramBatch`, an integer `diagram_id` column
  is prepended rather than an `offsets` array — Parquet's natural unit is the
  row, not a CSR buffer. Carries none of `DiagramMeta`: no `backend`, no
  `provenance`, no `params`. This is a bars-only interchange table for
  R/pandas/Polars pipelines (§1's R-bridging goal), not a `.akd` replacement.

---

## 11. Adapter contract

Signature for all five, with one deviation:

```python
from_gudhi(obj, **meta)   -> PersistenceDiagram
from_ripser(obj, **meta)  -> PersistenceDiagram
from_giotto(arr, *, reduced_homology, **meta)  -> PersistenceDiagram | DiagramBatch
from_persim(obj, **meta)  -> PersistenceDiagram
from_array(arr, **meta)   -> PersistenceDiagram
```

`from_giotto` alone takes a required keyword-only argument outside `**meta`.
This is deliberate, not an inconsistency to fix later: `reduced_homology`
determines whether the diagram is silently missing its H0 essential class
(§5.1), so omitting it must be a `TypeError` at the call site, not a value
that can slip past as an optional key in `**meta`.

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
- `strip_padding=False`: keep silently, no warning, and
  `provenance["padding_removed"] = 0` regardless of how many trivial rows are
  present — the key records what was actually removed, never what was merely
  observed, so its meaning does not change across the three modes.

Dropping trivial bars is *usually* right for giotto batches and *never*
guaranteed right, so the caller decides and the file records what happened.

### 11.2 Round-trip tests

Round-trip tests MUST run against **real backend output**, not
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

| #      | Question                                                                                                                               | Recommendation                                                                                                                                                                                                                                                                                                                                                                                                                           |
| --------| ----------------------------------------------------------------------------------------------------------------------------------------| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| D1     | File extension `.akd`, or plain `.npz` with our layout inside?                                                                         | **Resolved by §10, not open.** §10.1 rules out Parquet by name (fails the dependency-free requirement) and §10.2 normatively specifies `.akd`. This row previously recommended Parquet, contradicting both; see the note below the table. |
| D2     | Is `DiagramBatch` in scope for M1, or does M1 ship the single-diagram type only?                                                       | In scope. Retrofitting a batch container after `core/` is written against scalars is the expensive order, and §9.3 of the onboarding commits us to batch-shaped signatures.                                                                                                                                                                                                                                                              |
| D3     | Do we accept `float32` storage behind a flag for large-scale work?                                                                     | No, not in v0. Revisit when a real memory complaint exists.                                                                                                                                                                                                                                                                                                                                                                              |
| D4     | Should `from_giotto` default to `strip_padding=True`?                                                                                  | No. Defaulting to a lossy repair contradicts §5's whole argument. Warn and let the caller choose.                                                                                                                                                                                                                                                                                                                                        |
| D5     | Does the RFC published at M1 include §9's delegation hazards, or do we raise them upstream first?                                      | Raise upstream first — file the persim issue and the giotto scikit-learn issue, then publish citing our own reports. Costs two weeks, buys enormous goodwill, and turns a criticism into a contribution.                                                                                                                                                                                                                                 |
| **D6** | Array-API support (§3.3) needs a NumPy that has it. Raise the floor to `numpy>=2.0`, or add `array-api-compat` and keep `numpy>=1.24`? **Stale — see note below the table.** | **Raise the floor to `numpy>=2.0`.** Main-namespace array API support landed in NumPy 2.0; `1.24` cannot satisfy §3.3 at all. NumPy 2.0 is over two years old and adding a dependency to support a version that old contradicts our own closure discipline. `array-api-compat` (MIT, zero dependencies — verified) is the right answer *later*, behind `[torch]`, where it is genuinely needed because torch is not natively conformant. |
| **D7** | `content_hash` (§8.1) is defined only on `PersistenceDiagram`, and §4.2 argues against redefining it as a hash-of-hashes on `DiagramBatch`. But the onboarding acceptance bar is reproducing entire Paper IV tables, and a table's provenance is a batch-level claim. Does `DiagramBatch` need its own hash, defined some other way, or is per-diagram hashing sufficient for the reproducibility commitment? | **No recommendation.** This is a real gap, not a stylistic one, and it turns on how `repro/` actually pins its tables, which this document does not know. Needs the lead's call before M1, not mine to make unilaterally. |
| **D8** | ~~Should Parquet be offered anywhere, given §10.1 rules it out as the default (`.akd`) storage format?~~ **Applied — §10.3 now specifies `to_parquet()`.** | Added as a `pyarrow`-gated escape hatch alongside `to_arrays()`/`to_csv()`, requiring `pip install akriti[parquet]  # pyarrow (Apache 2.0)`, lazy-imported the same way `save`/`load` lazy-imports `numpy` (D10). §10.1 got a one-line clarification that its Parquet exclusion is about the *default* format only, not a blanket ban. `tools/check_license_closure.py` and `DEPENDENCIES.md` still need updating to cover the new extra — implementation work, not tracked further here. See the note below the table for the dependency this still carries. |
| **D9** | ~~§10.1's dependency-free requirement is read as "MIT/BSD-only" elsewhere in the project (onboarding). Akriti's own outbound license is Apache 2.0. Should the dependency closure be MIT/BSD-only, or MIT/BSD/Apache-2.0-only?~~ **Resolved by the onboarding document's 2026-07-30 revision.** | The premise was wrong, not just Apache 2.0's treatment of it. Onboarding's own "MIT/BSD-only" language has been retracted: `ripser` and `persim` are MIT but pull in `hopcroftkarp`, GPLv3, transitively; the `gudhi` wheel ships no license metadata at all while bundling GPL-marked CGAL modules. GPLv3 arrives regardless of the backend's own license, so "MIT/BSD-only" was never an achievable or even accurate description of the actual constraint. The real policy: the **default** install carries no third-party dependency at all, not `numpy` either; **every** backend, any license, sits behind a labeled extra (`akriti[rips]  # Ripser (MIT, GPLv3 transitively)`), enforced by `tools/check_license_closure.py` against a clean venv. License family was never the axis; presence in the default install was. Apache 2.0 was never actually excluded by anything, since nothing in the corrected policy restricts extras by license family, only requires they're disclosed at the install boundary. (This resolution originally restated the default install as "`akriti.diagrams` + `numpy`"; that repeated a mistake corrected in §3.3/§10.1/D10, see below.) |
| **D10** | The default install is solely the interchange layer (§3.3/§10.1), zero third-party dependencies, `numpy` included. But `io.py`'s `save`/`load` genuinely needs `numpy` for `.npz`. Should that be a lazy, function-scoped import with a friendly `ImportError` if missing, or a formal extra (e.g. `akriti[io]`)? | **Lazy import, not a formal extra.** A friendly, actionable `ImportError` inside `save`/`load` costs nothing to implement and keeps `pip install akriti` genuinely dependency-free while not forcing serialization users through an extra for a package as close to universal in this ecosystem as `numpy`. `tools/check_license_closure.py` should still assert the *default* venv has zero third-party imports at `diagrams/core.py` / `diagrams/adapters.py` import time, to keep this honest going forward rather than just at review time. |
| **D11** | D9 was marked resolved on the premise that the onboarding document's 2026-07-30 revision retracted "MIT/BSD-only" for "zero dependencies by default, any license behind a labeled extra." That retraction has not been independently checked against the onboarding document itself in this pass — this RFC is currently the only place that claims it happened. If the onboarding document still reads "MIT/BSD-only... copyleft-dependent backends go behind extras," D9 is not actually resolved, and D8's Apache-2.0 `pyarrow` extra reasoning, which depends entirely on D9's "license family was never the axis" framing, needs to be re-justified under the older, narrower policy. | **Confirm against the current onboarding document before M1.** If the retraction happened, this row can be struck. If it didn't, reopen D9 and re-derive D8 and D10 from whatever the actual current policy is — not from what this RFC assumed it had become. |

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

All three giotto rows were run with `reduced_homology=True`, giotto's default;
that parameter, not `infinity_values`, is what accounts for 39 rather than 40
H0 bars (§5.1). Varying only `infinity_values` was the original probe design
and it is why the table's own values don't distinguish the two causes; a
`reduced_homology=False` row is not yet in `probe_backends.py` and should be
added so this table shows the effect directly rather than requiring §5.1's
prose to carry it.

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

| Inputs | `bottleneck` | `wasserstein` | Correct? | Warnings |
|---|---|---|---|---|
| `[[0,inf],[.1,.5]]` vs itself | 0.0 | 0.0 | yes | **2** |
| `[[0,inf],[.1,.5]]` vs `[[0,1],[.1,.5]]` | **0.5** | **0.707** | **no — should be `inf`** | **1** |
| empty vs empty | 0.0 | 0.0 | yes | 0 |
| empty vs `[[0,1],[.1,.5]]` | 0.5 | 0.990 | yes | 0 |

The warning is
`UserWarning: dgm<n> has points with non-finite death times;ignoring those points`,
raised once per argument containing a non-finite death. `wasserstein` warns
identically.

Note what the counts imply: the **correct** row raises *two* warnings and the
**wrong** row raises *one*. The warning tracks whether an argument contained an
essential bar, not whether the result is meaningful. Row 1 is right only by
accident — dropping matching essential bars from both diagrams happens to
preserve a distance of zero. So the warning cannot be used to detect the failure,
and neither can its absence be used to certify a result.

---

## Appendix B — Changelog

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
  `content_hash` for the Paper IV table-reproduction commitment.
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
