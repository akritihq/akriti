# RFC-0001 — Persistence Diagram Interchange

| | |
|---|---|
| **Status** | Draft — not yet open for public comment |
| **Author** | Sushovan Majhi |
| **Edited By** | A. D. Silberman |
| **Created** | 2026-07-29 |
| **Last Edited** | 2026-08-06 |
| **Target** | M0 (2026-08-01) drafted · M1 (2026-09-15) published for comment |
| **Implements** | `akriti.diagrams` |

Key words **MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT**, **MAY** are to be
interpreted as described in BCP 14 (RFC 2119, RFC 8174) when, and only when,
they appear in all capitals. The other BCP 14 keywords — SHALL, REQUIRED,
RECOMMENDED, OPTIONAL and their negations — are deliberately unused; this
document uses "required" and "optional" in their ordinary Python senses.

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

**Non-goal: multiparameter persistence**, for this type and for this RFC.
`PersistenceDiagram` is single-parameter-shaped by construction: a multiset of
intervals, one scalar `dim`, one `birth` and one `death` per bar.
Multiparameter persistence modules do not decompose into intervals and admit
no complete discrete invariant, so the objects such a module requires are not
this type extended — a rank invariant, a fibered or signed barcode — and no
extension point specified now would fit one.

Should multiparameter support ever be built, it takes a parallel type and the
two coexist. This is not a deprecation path and MUST NOT be read as one. The
only forward-compatibility machinery warranted is §10.1 requirement 3's
format version, carried in §10.2's `meta.json`, which already exists.

---

## 2. Definitions

A **persistence diagram** is a finite multiset of **bars**. A bar is a triple
`(dimension, birth, death)` where:

- `dimension` is a non-negative integer — the homological degree.
- `birth`, `death` are real, with `birth <= death`. `death` MAY be `+inf`.

A bar with `death == +inf` is **essential**: the class it represents never dies.
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
API standard — not `np.ndarray`.

This is a hard requirement, not a preference: `core/` MUST be written against
the array API rather than hard-coded NumPy, and `PersistenceDiagram` is the
input to every function in `core/`. §3.3 states what this does and does not
promise. This is a strictly stronger obligation than §3.3's stdlib-only MUST,
not a restatement of it: importing nothing third-party forbids *calling*
NumPy, while writing against the array API additionally rules out
NumPy-shaped idioms applied to an array the caller handed in — which is what
§7's `lexsort` prohibition and §4.2's `len()`-versus-`shape[0]` rule are both
instances of.

`xp` is a derived **property**, not a fourth stored field: I7 already
requires `dims`, `births`, and `deaths` to share one namespace, so deriving
`xp` from `dims` makes disagreement structurally impossible rather than
merely prohibited, with nothing extra to keep in sync at every construction
site, including the views §4.2 returns.

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
| I2 | `births`, `deaths` are `float64`; `dims` is `int32` — tested by equality against the namespace's own `xp.float64` / `xp.int32` | §6.1 |
| I3 | `dims >= 0` | homological degree |
| I4 | `births` are all finite and non-`NaN` | a class that is never born is not a class |
| I5 | `deaths` are non-`NaN`; `+inf` permitted; `-inf` forbidden | §5 |
| I6 | `deaths >= births` elementwise | definitional |
| I7 | all three arrays share one namespace — `births.__array_namespace__() is deaths.__array_namespace__() is dims.__array_namespace__()` | §3.3; the `is` is identity by requirement (D16), verified in CI |
| I8 | `PersistenceDiagram` is immutable after construction — no method may write to `dims`, `births`, or `deaths` in place, and none may rebind them | §4.2 |
| I9 | `dims`, `births`, `deaths` are each rank-1 (`ndim == 1`) | §3, shape `(n,)` |

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
a new `PersistenceDiagram` rather than modify `self`. This SHOULD be enforced
the same way `DiagramMeta` already is (`@dataclass(frozen=True)`, §8); an
implementation that relies on the array API standard's read-only view support
instead MUST document that as an equivalent guarantee.

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
**aliases that emit a `DeprecationWarning` from the first release** — an alias
shipped unmarked is one we are not going to be able to withdraw later, and a
permanent second spelling of the canonical accessor is a permanent liability.

`d.dim(k)` for a `k` not present MUST return an empty diagram, not raise. Empty
is a legitimate answer to "what are the 7-dimensional cycles".

**The list above is deliberately narrow, not the complete read-only
surface.** Every item on it is specifiable from §2's definitions and this
section alone. `PersistenceDiagram`'s other read-only operations each depend
on machinery introduced later — equality on §6's precision discussion,
`canonical()` on §7's ordering discussion, `content_hash` on both — so each
is defined where that machinery lives, and cross-referenced here rather than
restated. One place to update beats two that can drift apart, the same reason
§8 keeps `essential_bars` single-sourced and §3 keeps `xp` a derived
property.

- `d.xp` — derived from `dims`, defined above in this section's parent, §3.
- `d1 == d2`, `d1.allclose(d2, ...)` — exact and approximate equality, §6.3.
- `d.canonical()` — presentation-ordering accessor, §7.
- `d.content_hash` — sha256 over canonical-ordered bars, §8.1.
- `d1.same_provenance(d2)` — provenance comparison, excluded from `==`, §8.
- `d.finitize(at=...)` — not an accessor, listed here only so its absence
  isn't mistaken for an oversight. It takes an argument and records the
  substitution in `meta.provenance`: a transformation, not a read (§5).

`DiagramBatch` has the analogous surface catalogued the same way in §4.3.

### 3.3 What array-API support does and does not promise

Being namespace-agnostic is not free, and promising more than the standard
delivers would be worse than promising nothing. Three limits, all measured
against `array_api_strict` 2.6.1, the conformance reference:

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

**Shape-preserving is not the same as traceable, and `finitize` is the case
that separates them.** All three of §5's modes are eager-only, not just
`at="drop"`. `at="max_finite_death"` masks before it reduces, but the
restriction does not depend on that: *every* mode must decide whether the
diagram has any essential bar at all, because §5 requires one answer when it
does and a different one when it does not, and that is a Python-level branch
taken on array values. It concretizes a traced array whichever mode was
asked for. The substituting modes do preserve the output shape, and stating
that alone — as a first draft of this paragraph's implementation did — reads
as a traceability claim it does not support. Documentation that reports the
first property as though it settled the second is worse than saying nothing,
since a reader checking whether an operation is available under `jax.jit` has
been answered, incorrectly, rather than left to check.

The same applies to everything returning a Python `bool` or `str`: `==`,
`allclose`, `same_provenance`, and `content_hash` are eager-only without
being filtering operations. `n_bars` is the one exception worth naming — it
reads `shape[0]`, a property of the shape rather than the values, and stays
available.

**`DiagramBatch.__getitem__` is the second case that separates the two
properties, and it is the one that propagates.** It is neither a filtering
operation nor a `bool`-returning one, but its slice bounds are
`int(offsets[i])` and `int(offsets[i + 1])`, which concretize a traced array
for the same reason `finitize`'s branch does, so it is eager-only and so is
every batch operation routed through it. `b.canonical()` is the case worth
stating outright, because the diagram-level `d.canonical()` genuinely is
traceable and the batch version looks like the same operation: the sort is
shape-preserving at both levels, and only one of the two is available under
`jax.jit`. §4.3 carries the full list.

**Serialization is NumPy-bound, deliberately.** `io.py` (§10) writes `.npz`,
converting at the I/O boundary via `np.asarray` and returning NumPy-backed
diagrams on load — serialization is not a numerical kernel, so there's
nothing to gain from making it generic, but the conversion MUST happen at
that boundary only, never in the constructor or an adapter. This does not
make `numpy` a dependency of the package: `diagrams/core.py` and
`diagrams/adapters.py` MUST import nothing beyond the standard library,
operating entirely through `__array_namespace__` on whatever the caller
already has. `numpy` is used **only** inside `save`/`load`, imported lazily
there rather than at module scope, so everything except serialization works
with zero third-party dependencies (§10.1 requirement 2).

**The floor is `numpy>=2.0`, and it is declared rather than assumed.**
Array-API support in NumPy's main namespace landed in 2.0; a `numpy` older
than that does not answer `__array_namespace__` and cannot serve the boundary
conversion above. `numpy` is therefore declared as `akriti[io]`, an extra
rather than a required dependency: `pip install akriti` resolves to nothing
third-party, and `pip install akriti[io]` resolves the floor at install
time, where an unsatisfiable version is a resolver error rather than a
runtime one.

The lazy import MUST therefore check the **version**, not merely presence,
and MUST distinguish the two failures:

- `numpy` absent — MUST raise a clear `ImportError`.
- `numpy` present but older than 2.0 — MUST raise a clear `ImportError`
  as well, rather than proceeding into an `AttributeError` on the first
  array-API call.

Both messages MUST name the extra — "install `akriti[io]`" — not the bare
package. A message naming `numpy` alone tells a user with `numpy` 1.24
already installed to install what they have; naming the extra states the
action that actually resolves the floor.

**Adapters preserve the input namespace.** `from_*` MUST NOT force-convert to
NumPy. A diagram built from torch tensors stays torch-backed. What adapters
convert is *dtype* (§6.1), not namespace.

**Namespaces are compared by identity, and that is a constraint this document
places on supported backends rather than a property the standard provides.**
I7, B5 and §4.2's `from_diagrams` check all compare with `is`. The standard
requires `__array_namespace__()` to return "an object representing the
namespace"; it does not require the same object on every call, and it takes an
`api_version` argument that a backend could legitimately answer with distinct
wrapper objects. NumPy and `array_api_strict` return the module itself, so
identity holds there and the assumption is invisible. **Akriti requires
`__array_namespace__()` to return a consistent object for a given backend**,
and states that here rather than claiming the standard guarantees it.

D16 settled this on the **direction of the failure**, not on the theory. `is`
fails by raising `ValueError` (§6.3) on arrays that legitimately share a
namespace: conservative, loud, immediately diagnosable, and impossible to
mistake for a correct answer. Every surrogate the standard leaves available —
comparing `__name__`, a sentinel dtype, anything else — is a weaker test that
can *match across genuinely different namespaces*, admitting a torch/NumPy mix
into one diagram, which is what I7 exists to prevent. That is the silent
direction. Given a conservative check carrying a documented constraint against
a clever one that fails open, this document takes the first.

**The constraint MUST be verified in CI, not assumed.** A test MUST assert,
for every backend this project supports, that `__array_namespace__()` called
on two separate arrays of that backend returns the same object. This is the
paragraph below applied to a promise the standard does not make, and it
follows §7's `lexsort` precedent: a trap a reader cannot see becomes a
standing regression test rather than prose. A backend that ever returns a
fresh wrapper per call then fails that test and reopens D16 as a real
decision, instead of breaking silently in someone's pipeline.

**Conformance is tested, not intended.** CI runs the diagram test suite
against `array_api_strict`, which rejects any NumPy-only call. A requirement
merely written down decays within weeks — the first-draft slip noted in §3
is direct evidence of that, and nobody caught it until a reviewer read both
documents.

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

Functions that genuinely need a rectangular buffer (e.g. a vectorisation feeding a
tensor op) MUST perform the padding internally, at the point of use, with an
explicit mask returned alongside — never in the interchange type.

### 4.1 Reconciling the two-type design

**Not adopted:** folding `PersistenceDiagram` and `DiagramBatch` into one
type, batch-of-one by default, backed by a dense `(batch, max_points, 2)`
array plus a boolean mask. Appendix A.2 is why, not taste: a dense padded
batch cannot represent giotto's own output without padding rows becoming
indistinguishable from genuine trivial bars, exactly the class of bug this
document exists to prevent. `DiagramBatch` stays ragged at the interchange
boundary; a dense buffer is still the right tool inside `castle/`, just built
internally with an explicit mask, never at the interchange layer (§4's
padding+mask scheme).

The split adds no duplicated logic in the sense that matters: no
`DiagramBatch` accessor needs its own new rule stated against
`PersistenceDiagram`'s, the two exceptions being `canonical()` and equality,
each kept in the one place it lives (§7, §6.3). §4.3 works through which is
which. A batch of one is still a length-one `DiagramBatch` wrapping the same
`PersistenceDiagram` type (§4.2). Full deliberation: history document.

### 4.2 `DiagramBatch` storage representation

The interface in §4 is silent on layout, and silence here has a cost: the
natural reading, N independently allocated `PersistenceDiagram` objects held
in a Python list, is also the one that will not scale well in a neural-network 
path. This section closes that gap.

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
        return self.offsets.shape[0] - 1

    def __getitem__(self, i) -> PersistenceDiagram:
        lo, hi = self.offsets[i], self.offsets[i + 1]
        # a view: dims[lo:hi], births[lo:hi], deaths[lo:hi], no copy
        ...
```

**`.shape[0]`, not `len(...)`, and that is not a stylistic choice.** The
array API standard does not require an array object to implement `__len__`,
so `len(offsets)` and `len(dims)` are NumPy habits that §3.3's whole argument
rules out; `core.py` reads `.shape[0]` throughout. Where this document writes
`len(...)` of an array — I1, B1, B3, and §4.3's batch total — it is
shorthand for `shape[0]` and MUST be implemented as such. `len(batch)` and
`len(diagrams)` are ordinary Python and unaffected.

`__getitem__` MUST return a **view**: a `PersistenceDiagram` whose arrays are
slices into the batch's own buffers, not a copy. This is safe only because
`core.py` gives `PersistenceDiagram` no mutating methods (I8, §3.1); there is
no aliasing hazard from two objects sharing memory when neither can be written
to after construction.

**`offsets` has its own invariants, and `core.py` MUST enforce them at
`DiagramBatch` construction, the same way §3.1 enforces I1 through I9 for a
single diagram:**

| # | Invariant | Rationale |
|---|---|---|
| B1 | `len(offsets) == len(batch) + 1` | fencepost: `n` diagrams need `n+1` boundaries |
| B2 | `offsets[0] == 0` | buffer has no unowned leading bars |
| B3 | `offsets[-1] == total_bars` (i.e. `len(dims)`) | buffer has no unowned trailing bars; bounds the last diagram's slice |
| B4 | `offsets` is non-decreasing | row ranges must not overlap or invert |
| B5 | `offsets.__array_namespace__()` matches `dims`, `births`, `deaths` | §3.3; "matches" is identity, required of supported backends (D16) |
| B6 | `offsets` is rank-1 (`ndim == 1`) | I9's rationale, applied to `offsets`: a wrong-rank array of the right length passes B1 unnoticed |
| B7 | `offsets.dtype` is the namespace's own `int64` | the class body above already says `int64`; stated as an invariant so it is enforced and citable like the rest |

B6 and B7 were implicit before: the class body declares `offsets` `int64` of
shape `(len(batch)+1,)`, and B1 through B5 then quietly assume both. B1's
`len(offsets)` reads `shape[0]`, which is happy to answer for a rank-2 array —
exactly the gap I9 was added to close for `dims`/`births`/`deaths`, and the
same gap, one field over. Neither is a new requirement, only a stated one.

Without B1 through B7, `__getitem__`'s slice arithmetic can silently read the
wrong range, or read past the end of the buffer, which is the same category of
silent-wrongness bug §9 exists to catch, just self-inflicted this time rather
than a backend's fault.

One buffer, rather than N, is what keeps a future DL path cheap: building a
training minibatch or moving one to device becomes a small, fixed number of
array operations rather than a Python-level loop over N objects. It costs
nothing at TDA's usual per-diagram bar counts, and it is the same shape §10.2
already committed to for the on-disk format; this extends that commitment to
memory.

**Why not merge `PersistenceDiagram` and `DiagramBatch` into one CSR-backed
type**, with `offsets = [0, n]` as the single-diagram case? Considered and
rejected on two points specific to this RFC: `DiagramMeta` (§8) is genuinely
per-diagram (`backend`, `params`, `provenance` can all differ across a
batch), and `content_hash` (§8.1) is defined over one diagram's canonical
bars, so a merged type would force `meta` to become a sequence or
`content_hash` to become a hash-of-hashes, both new specification this RFC
doesn't otherwise need. §8.2 does add that hash-of-hashes, resolving D7, but
as `DiagramBatch.content_hash`, a property of the composition type, not as a
redefinition that `PersistenceDiagram.content_hash` was ever forced into by
merging; keeping the two types separate is what let it be added without
touching §8.1. `DiagramBatch` and `PersistenceDiagram` are related
by composition, not inheritance, deliberately: a subclass that fails to
override even one accessor would silently compute it across batch
boundaries instead of raising, the same "clean, plausible, wrong answer"
category §9 exists to rule out. Precedent (PyTorch Geometric) and full
deliberation may be found in the history document.

**Construction from ordinary adapter output.** §11's adapters return one
`PersistenceDiagram` per call, except `from_giotto`, which is only pre-batched
because giotto's own input already is. The common path, N separate
`from_gudhi` / `from_ripser` calls that need to become one `DiagramBatch` for
a bootstrap or permutation test, has no constructor yet. `DiagramBatch` MUST
provide one:

```python
@classmethod
def from_diagrams(
    cls, diagrams: Sequence[PersistenceDiagram], *, xp=None
) -> "DiagramBatch":
    ...  # concatenate dims/births/deaths, derive offsets from each length,
         # collect .meta into metas, in input order
```

This is also where the concatenate-and-derive-`offsets` logic from the class
body above is actually exercised; nothing else in this document currently
calls it.

**Every input diagram MUST share one array namespace, and `from_diagrams`
MUST check it.** I7 constrains the three arrays *within* one diagram and says
nothing across diagrams, so a sequence mixing a torch-backed diagram with a
NumPy-backed one satisfies I7 term by term and still cannot become a batch.
The check has to happen here because it cannot happen afterwards: `concat`
either raises something opaque from the backend or silently coerces the
foreign array, and the concatenated buffer then satisfies I7 and B5 cleanly,
the mixed input having already been erased. Adapters preserve the input
namespace (§3.3) rather than normalising it, so a caller assembling diagrams
from two sources is the ordinary case, not a perverse one.

**`xp` is required for, and only for, an empty `diagrams`.** An empty batch is
a valid `DiagramBatch` — §8.2 defines a hash for it, §7's segment-wise
canonicalisation is vacuous on it, and a bootstrap or permutation test that
filters its inputs down to nothing needs to be able to build one rather than
special-case it. But there is no diagram to derive a namespace from, and this
type has no default backend to fall back on (§3.3 is explicit that NumPy is
not one), so the namespace MUST come from the caller in that one case:
`DiagramBatch.from_diagrams([], xp=xp)`. Passing `xp` alongside a non-empty
`diagrams` is permitted and MUST be rejected if it disagrees with the
diagrams' own namespace. Refusing to construct an empty batch at all was the
alternative, and it is worse: the type supports empty batches everywhere else,
so the only constructor for one would be the private path.

### 4.3 Accessors

```python
len(b)                # -> int, number of diagrams in the batch
b[i]                  # -> PersistenceDiagram, a zero-copy view (§4.2)
b.essential           # -> bool mask, shape (total_bars,), deaths == inf
b.persistence         # -> deaths - births, shape (total_bars,), inf for essential bars
b.bar_counts          # -> int array, shape (len(b),), bar count per diagram
b.xp                  # -> the shared array namespace, derived from dims
```

`essential` and `persistence` are elementwise over the whole concatenated
buffer, as cheap as the stored fields themselves and correct by the same
invariants that already govern `dims`/`births`/`deaths` (I4, I5).
`bar_counts` is `offsets[1:] - offsets[:-1]`, always non-negative by B4; it
is deliberately not named `n_bars`, since `PersistenceDiagram.n_bars` is a
scalar and this is an array, and reusing a name across a shape change is the
silent-wrongness §9 exists to rule out elsewhere. The batch total is
`b.dims.shape[0]` (§4.2 on why not `len`), already available from the stored
field and not worth a second name. `xp` is `self.dims.__array_namespace__()`,
the derive-don't-store reasoning §3 gives for `PersistenceDiagram.xp`,
well-defined because B5 already requires `offsets` to share `dims`'
namespace.

**Which of these are available under `jax.jit` is settled in §3.3, not
restated here.** In short: `essential`, `persistence`, `len(b)` and
`bar_counts` are. `b[i]` is not, and neither is anything reached through it —
`canonical()`, `==`, `allclose`, `same_provenance`, `content_hash` — for the
slice-bound reason §3.3 gives.

None of the four accessors above needed a new rule stated to be correct,
which is what §4.1's "no duplicated logic" claim actually protects.
`canonical()` and equality each did, and are kept in the one place each lives
(§7, §6.3).

This, plus the constructor and comparisons below, is the complete
self-contained surface, everything specifiable from §4.2 alone. It is
thinner than §3.2's, deliberately, not an incomplete version of the same
list:

- `b.canonical()` — §7. Segment-respecting, not the same operation as
  applying `d.canonical()`'s argsort to the concatenated buffer.
- `b1 == b2`, `b1.allclose(b2, ...)` — §6.3. Order-sensitive across
  diagrams, unlike bar order within one diagram, which explicitly is not.
- `b1.same_provenance(b2)` — §8. Same order-sensitive pattern as `==`.
- `b.content_hash` — §8.2. Composed from member `content_hash`es in batch
  order, domain-separated from `PersistenceDiagram.content_hash`, exact
  equality only. Resolves D7.
- `DiagramBatch.from_diagrams(diagrams)` — not an accessor, listed here only
  so its absence isn't mistaken for one. It's the constructor (§4.2), a read
  on nothing yet built, not a read on an existing batch.

Two absences below are gaps, not placement choices, and are stated as
such rather than left for a reader to discover by searching:

- **No batch-level `dim(k)` or `finite`.** Both would return a new
  `DiagramBatch`, filtered within each `[offsets[i], offsets[i+1])` segment
  the way `canonical()` is per-segment sorted, and would inherit the
  diagram-level versions' eager-only restriction (§3.3) since filtering
  changes shape. Straightforward generalizations; nobody has written them
  down yet.
- **No batch-level `dimensions`.** Unlike the gap above, this one has no
  single obvious generalization. The global union of degrees present
  anywhere in the batch, and the per-diagram list, already expressible as
  `[d.dimensions for d in b]` without any new accessor, are both defensible
  and don't reduce to each other. A design call, not an oversight to just
  fill in.

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

**Finitization is an explicit, caller-requested operation:**

```python
d.finitize(at="max_finite_death")   # or at=<float>, or at="drop"
```

It returns a new diagram, records the substitution in `meta.provenance`, and is
never applied implicitly by an adapter, a constructor, or an I/O routine.

**`at="drop"` is not a substitution and MUST NOT be recorded as one.** The
other two modes replace `inf` with a finite value in place, the bar survives,
only its death time changes, so `"finitized_at:<value>"` (§8) correctly
describes what happened. `at="drop"` removes the bar entirely: `n_bars`
shrinks, and there is no substituted value to name. Recording it as
`"finitized_at:<value>"` with some placeholder would misrepresent a cardinality
change as a value change, exactly the kind of clean-plausible-wrong signal §9
exists to rule out. `finitize(at="drop")` MUST instead set
`provenance["essential_bars"] = "finitized_dropped"` and
`provenance["essential_bars_dropped"]` to the count of bars removed (§8).

**A diagram with no essential bars MUST be returned unchanged, provenance
included.** No bar was substituted and none was dropped, so there is nothing
for `provenance` to record, and recording something anyway is the same
misrepresentation the previous paragraph rules out with the signs reversed:
`"finitized_dropped"` with `essential_bars_dropped = 0` asserts a cardinality
change that did not happen, and `"finitized_at:<value>"` names a substitution
that reached no bar. Both would also overwrite whatever `essential_bars`
already said — a `"lost_upstream"` erased by a call that did nothing is the
worst version of this. The `at` argument MUST still be validated first: an
unknown `at` raises whether or not the diagram has essential bars, since
otherwise a typo would be caught only on the diagrams that happen to have one.

**"Validated first" covers the whole argument, not just the two mode names.**
The obvious implementation checks `at` against `"drop"` and
`"max_finite_death"` when it is a string and lets everything else fall
through to the substitution path, which is where a float is finally required.
That check is data-dependent for exactly the values it was meant to catch:
`at=None` — the ordinary typo, and the one a `float | str` signature makes
easy to write — raises from the conversion on a diagram that has essential
bars, and returns the diagram unchanged and silently on one that does not.
That is the failure this paragraph already rules out, moved one argument
domain over. An `at` that is neither a mode name nor convertible to a float
MUST raise regardless of the data, and MUST raise `TypeError` rather than
`ValueError`: no diagram makes `finitize(at=None)` meaningful, which is the
distinction the two exceptions carry in every other Python API a caller of
this one will have used, and §6.3's `allclose` and §8's `same_provenance`
already raise `TypeError` on a wrong-typed argument one signature over.
`ValueError` is for an `at` of the right type carrying a value this method
cannot use — an unrecognised mode name, or the non-finite float the next
paragraph rules out. The split is by what is wrong with the call, not by
which check happened to catch it.

**`at=<float>` MUST be finite.** `at=+inf` substitutes an infinity for an
infinity: nothing changes, every essential bar is still essential, and
`provenance["essential_bars"]` now reads `"finitized_at:inf"`. The resulting
diagram's own `essential` mask contradicts its own provenance — one says the
essential set is intact, the other says it was substituted away — and the two
readers §8 anticipates split along exactly that line: `core/distances.py`
(§9.1) partitions on the mask and would correctly return `+inf`, while a
human auditing `provenance` reads that the diagram was finitized. Neither is
wrong about what it looked at, which is what makes it the
clean-plausible-wrong category §9 exists to catch rather than an ordinary
bug. It is also the mirror of the no-essential-bars case above: there the
record described work that reached no bar, here it describes work that
reached every essential bar and changed none of them. `at=nan` is excluded by
I5 in any case, but only incidentally, and with an error naming death times
rather than the argument the caller actually got wrong; both MUST raise on
the argument. `at="max_finite_death"` needs no such check — I5 already
guarantees the maximum it takes is over finite values.

**`finitize` overwrites `essential_bars`, and MUST NOT write
`provenance["essential_bars_source"]`, which is the adapter's (§8).**
`essential_bars` is a single slot and §8 requires it to describe the
diagram's current state, so finitizing a giotto-sourced diagram necessarily
overwrites `"lost_upstream"` — and that value is a claim about how the
diagram was *computed*, which no later transformation can make untrue. A
diagram reading `essential_bars = "finitized_at:1.7"` alone is
indistinguishable from one whose essential set was faithful before the
substitution, when in fact its H0 essential class was already missing (§5.1).
One key answers "what is true of these bars now", the other "what was true
when they were computed"; §8 argues at length that provenance must answer the
second for a diagram's whole life, and one overwritable key cannot do both.

The second key is therefore written **once, by the `from_*` adapter that
recorded `essential_bars` in the first place** (§8, §11), never later by
`finitize`. Having `finitize` copy the current value forward on its first
call instead was considered and rejected, on the same
derive-at-the-source ground §8 already uses for `essential_bars` itself:

- **Only the adapter knows the answer.** `essential_bars_source` means "the
  verdict at computation time". `finitize` can only read whatever
  `essential_bars` says *now*, which is the adapter's value only if nothing
  has happened to the diagram in between — precisely the assumption a
  provenance record exists to stop a reader from having to make.
- **A copy-forward admits values the key cannot legitimately hold.**
  `essential_bars_source` ranges over `"faithful"` and `"lost_upstream"`, the
  two an adapter can produce. A diagram that arrives already finitized — from
  `load` (§10.1 requirement 1 round-trips provenance) or from a caller who
  finitized in an earlier session — would have `"finitized_dropped"` or
  `"finitized_at:<value>"` copied into it, asserting an adapter-time verdict
  no adapter ever emits.
- **Absence stays unambiguous.** With the adapter as sole writer, no
  `essential_bars_source` means no adapter recorded one — a hand-built
  diagram (§8), or a non-conforming adapter. Under copy-forward it would mean
  that *or* "has never been finitized", two conditions a reader cannot tell
  apart from the mapping alone.

The cost is that finitizing a diagram whose adapter never wrote the key
destroys its `essential_bars` with nothing preserving it. That is the correct
outcome: a diagram with no recorded adapter-time verdict has none to preserve,
and inventing one from a mutable field is how the record starts lying.

**Also considered and rejected: having a substitution keep the smaller of the
previously recorded `"finitized_at:<value>"` and the new one**, on the
intuition that the more aggressive finitization is the one worth remembering.
It is unreachable — the return-unchanged rule above makes a second `finitize`
a no-op in either direction — and it would misdescribe the bars if it were
not. Full reasoning: history document. The instinct behind it is sound, and
this document already acts on it: something *should* preserve an earlier
verdict rather than let a later call erase it. That is
`essential_bars_source`, above — a second key with a single writer, not an
ordering imposed on the first.

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
when `params["reduced_homology"]` is `True`, `"faithful"` when `False`.
`provenance["essential_bars_source"]` MUST be set to the same value in the
same construction — it is the record of what this adapter concluded, and no
later writer can reconstruct it (§5, §8). The
adapter cannot recover this from the output array alone, a filtration
truncated by `max_edge_length` can also show zero H0 essential bars under
`reduced_homology=False`, so it MUST come from the caller.

**`reduced_homology` is a required, keyword-only parameter, not a default
with a warning:**

```python
from_giotto(arr, *, reduced_homology, **meta) -> DiagramBatch
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
`np.float64` / `np.int32`. Comparing against a NumPy dtype object breaks on
every non-NumPy backend and is the most likely way for NumPy to creep back in
unnoticed.

**I2 MUST be checked by equality against `xp.float64` / `xp.int32`, not by
`xp.isdtype`.** `xp.isdtype(a.dtype, "real floating")` is true of `float32`,
which D3 rejects outright and which §6.2's whole precision argument depends
on excluding, so as a check on I2 it is no check at all. `xp.isdtype` remains
the right tool for a genuine kind-level question; I2 is not one.

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
d1 == d2               # exact: same multiset of bars, compared without tolerance
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

**Comparing two diagrams backed by different array namespaces MUST raise
`ValueError`, at both levels and in both methods.** I7 constrains the three
arrays *within* one diagram and says nothing across two, so a NumPy diagram
and a torch diagram of the same bars are each valid and still not comparable:
`core.py` may not convert either one (§3.3 gives it no third-party import and
no default backend), so the comparison has no well-defined answer to return.
The check MUST happen before any length or value comparison, so the failure
does not depend on the data — §5 imposes the same ordering on `finitize`'s
`at`, for the same reason. `same_provenance` is exempt: it touches no array,
and answering the metadata question for two diagrams from different backends
is the whole point of the method.

Three alternatives were weighed. Letting the mismatch reach the backend's own
`equal` yields whatever that backend happens to raise, the opaque failure
§4.2 rejects for `concat`. Returning `False` is worse — a clean, plausible
answer asserting the bars differ when they may not, which is §9's whole
category. Returning `NotImplemented` from `__eq__` keeps `==` total and is
what Python itself would reach for; it is rejected because Python converts it
to `False` once the right-hand side also declines, arriving at the previous
option by a longer route. **The accepted cost is that `==` is not total**:
`d in [d1, d2]`, `list.index`, `list.remove` and any container comparing
elements will raise rather than answer when a cross-namespace diagram is in
the sequence. That is deliberate — a container search is exactly where a
wrong `False` would go unnoticed — and callers needing a total comparison
convert at their own boundary first (§3.3).

**"Exact" means "without tolerance", not "bit-identical".** The two come
apart in one reachable case: `-0.0 == 0.0` in IEEE 754, so two diagrams
differing only in the sign of a zero are equal under `==` while their raw
bytes differ. This is not a defect to be repaired here — IEEE equality is
what §7's sort and every backend's own comparisons already use, and a `==`
that distinguished the two zeros would disagree with both. It is instead why
§8.1 MUST normalise the sign of zero before hashing: `==` and `content_hash`
are two answers to the same question and only one of them sees raw bytes, so
the hash is the side that has to be told. §8.1 carries the case in full.

**`allclose` MUST be a matching, not a sorted pairwise comparison.** Two
diagrams are `allclose` iff there exists a bijection between their bars under
which every matched pair shares a `dim` exactly and agrees on both
coordinates within tolerance. Equal bar counts are necessary and not
sufficient.

The sorted-pairwise implementation is rejected. Sorting both sides into
canonical order (§7) and comparing pairwise is exact in the sort and
approximate in the comparison, and the two do not compose: when two bars'
births lie within tolerance *of each other*, two backends can canonicalise
them into different orders, and the pairwise comparison then reports `False`
for diagrams that do have a bar-for-bar partner within `rtol`. Appendix A.3
measures GUDHI/Ripser disagreement at `2.7e-8`, the magnitude that flips such
a tie, so this is reachable on exactly the cross-backend comparison
`allclose` exists to serve. No choice of sort key repairs it, and no total
order is stable under perturbation; the history document carries that
argument in full. Accepting the false negative was weighed and rejected — it
is conservative, never a spurious `True`, but the caller's remedy for a
spurious failure is to widen `rtol` until the comparison passes, which
reintroduces into user code, where nobody reviews it, the silent loosening
this section exists to prevent.

**The implementation MUST NOT introduce a dependency.** §3.3 and §10.1
requirement 2 give this module the standard library and the caller's
namespace, so `scipy.sparse.csgraph.maximum_bipartite_matching` is
unavailable, and a lazy import on a comparison path is a worse trade than
requirement 2's narrow one at the `save`/`load` boundary. An augmenting-path
matching is sufficient at the sizes this method serves; Hopcroft–Karp's
asymptotics are not needed. Edge construction is O(n²), and `allclose` is a
verification surface rather than a numerical inner loop.

**This is not bottleneck distance and MUST NOT be refactored into it.** "Does
a perfect matching within threshold `t` exist" is the decision problem a
bottleneck binary search calls repeatedly, so the two are adjacent by
construction. They are not the same question: `allclose` admits no diagonal
projection and optimises nothing. §9's delegation rule is unaffected in both
directions — this section implements no distance, and `core/distances.py`
MUST NOT be built on this method.

**The tolerance MUST be symmetric:**
`|a - b| <= atol + rtol * max(|a|, |b|)`. This deliberately diverges from
`numpy.allclose`, which scales `rtol` by its second argument alone and so
permits `d1.allclose(d2)` and `d2.allclose(d1)` to disagree at the boundary.
The divergence MUST be documented in the method's own docstring.

**`allclose` is reflexive and symmetric but not transitive, and MUST be
documented as not an equivalence relation.** `==` is one, and callers will
assume the parity holds.

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

**It MUST NOT be implemented with `lexsort`.** That function is not part of
the array API standard (§3.3). Compose it instead from stable `argsort` passes,
least significant key first:

```python
order = xp.argsort(deaths, stable=True)
order = xp.take(order, xp.argsort(xp.take(births, order), stable=True))
order = xp.take(order, xp.argsort(xp.take(dims,   order), stable=True))
```

Stability is what makes the composition correct; an unstable sort at any step
loses the ordering established by the previous one. Use `xp.take` rather than
integer-array indexing — both work under `array_api_strict`, but `take` is the
form the standard specifies for gathering.

Canonical ordering serves two separate purposes, not one causing the other.
It is what makes `content_hash` (§8.1) well-defined: computed over the
diagram's own canonical-ordered arrays in memory, it does not depend on
anything in §10, and is stable across calls regardless of how, or whether,
a diagram is ever serialized. It is also one necessary ingredient, not the
only one, for the on-disk format (§10) to satisfy §10.1's determinism
requirement: `bars.npz` is written in canonical order, but byte-identical
files additionally require the zip archive's own metadata, timestamps,
compression settings, to be pinned, which §10.1 tracks as a separate,
currently open, implementation obligation.
`canonical()` is a presentation concern and MUST NOT be assumed by any
numerical routine. It returns a new diagram carrying `meta` through unchanged;
D15 removed `provenance["order"]`, so there is no order key to update (§12.2).

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
| `essential_bars` | one of `"faithful"`, `"lost_upstream"`, `"finitized_at:<value>"`, `"finitized_dropped"` |
| `essential_bars_dropped` | count of essential bars removed by `finitize(at="drop")`; present iff `essential_bars == "finitized_dropped"` |
| `essential_bars_source` | `essential_bars` as the adapter recorded it — `"faithful"` or `"lost_upstream"`, never a `"finitized_*"` value. Written only by `from_*`, never by `finitize` (§5) |
| `source_dtype` | dtype of the input array |
| `clamped_rows` | count of `death < birth` rows the adapter repaired |
| `padding_removed` | count of trivial rows stripped as suspected batch padding |

Every value in `params` and `provenance` MUST be JSON-representable —
`str`, `int`, `float`, `bool`, `None`, or a list or `str`-keyed mapping of
those. §10.2 stores both as UTF-8 JSON in `meta.json`, so a mapping holding a
NumPy scalar, a `Path`, or a backend object is a diagram that satisfies §3.1
and §8 completely and cannot be saved. Without this constraint the type
admits diagrams that requirement 1 (§10.1) cannot round-trip, and the failure
surfaces at `save()` — arbitrarily far from the adapter that wrote the
offending value. Adapters MUST convert at the point of recording:
`str(arr.dtype)` for `source_dtype`, a Python `int` for the counts.

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

`finitize()` overwrites `essential_bars`, which is why `essential_bars_source`
exists: the first describes the bars as they now stand, the second how they
were computed, and the two questions have different answers from the moment a
lossy diagram is finitized. **`essential_bars_source` has one writer, and it
is not `finitize()`.** Every `from_*` adapter that records `essential_bars`
MUST record `essential_bars_source` with the same value in the same
construction, and it MUST NOT be written afterwards. It shares `essential_bars`'
vocabulary deliberately, so that "what does it say now" and "what did it say
then" are the same question asked of two keys rather than two encodings of one
concept; a boolean was considered and rejected on that ground, and because the
two legitimate values are a fact about the four backends this document covers
rather than about the key — the fifth backend the bullet above anticipates
extends a string enum and cannot extend a boolean. §5 carries the full
argument for why `finitize()` copying the value forward on its first call is
not an acceptable substitute for the adapter writing it.

**The keys that qualify `essential_bars` MUST be kept consistent with it, not
merely written alongside it.** `essential_bars_dropped` is specified as
present *iff* `essential_bars == "finitized_dropped"`, so a writer that
changes `essential_bars` to anything else MUST remove it in the same
operation. Merging a new value into an existing `provenance` mapping and
leaving the rest alone — the obvious implementation — breaks this on the
second call: `d.finitize(at="drop").finitize(at=1.0)` would otherwise leave a
diagram claiming both a cardinality change and a value substitution, with a
count belonging to neither. `essential_bars_source` is the deliberate
exception, and the only one: it is not a qualifier on the current value but a
record of the adapter-time one, which is exactly why it is a separate key rather
than another form `essential_bars` can take.

**`DiagramMeta` MUST enforce the two rules above at construction**, for the
reason §3.1 gives one type over: a rule stated only as an obligation on
writers is one every future writer has to remember independently, and
`finitize` is not the only writer — every `from_*` adapter (§11) sets these
keys through this constructor and none of them passes through `finitize`'s
code path. Concretely, constructing a `DiagramMeta` MUST raise `ValueError`
when `essential_bars_dropped` is present without
`essential_bars == "finitized_dropped"` or absent with it, and when
`essential_bars_source` holds anything but `"faithful"` or `"lost_upstream"`
— the copy-forward §5 rejects, caught where it would have to be written
rather than left to a reader to notice. Nothing else about `provenance` is
validated: §8 reserves names within an open mapping rather than closing it,
so unreserved keys and the free-form `<value>` in `"finitized_at:<value>"`
pass through untouched.

**There is no `order` key, and the reason bears on any future one.** D15
removed it: whether rows are in canonical order is recoverable from the arrays
themselves in one pass, which is true of no other reserved key here, and a
cached answer to an always-computable question can only go stale (§12.2). The
order fact that is *not* recoverable is whether the **backend's own output**
was already canonical — the GUDHI-versus-Ripser disagreement §7 documents and
Appendix A.3 measures, unrecoverable once anything sorts. `order` did not
capture it, recording where the ordering came from rather than what it was. If
order provenance is ever wanted, that is the key to build, written at adapter
time, and building it does not reopen D15.

**`meta` MUST NOT participate in `==` or `allclose`.** Two diagrams with the
same bars from different backends are the same diagram. Provenance is recorded
so a human can audit it, not so equality can reject on it. `d1.same_provenance(d2)`
is available for the cases that genuinely care, and `DiagramBatch` gets the
same method: `b1.same_provenance(b2)` requires `len(b1) == len(b2)` and
`b1[i].same_provenance(b2[i])` for every `i` in sequence, order-sensitive the
same way `==` and `allclose` are (§6.3).

### 8.1 Content hash

```python
d.content_hash   # -> str, sha256 over canonical-ordered coordinates + dims
```

Covers bars only, never metadata. This is what a paper pins — which is why the
hashed message is specified here rather than left to the implementation:

```python
d.content_hash = sha256(
    b"akriti.PersistenceDiagram.v1\x00"
    + d.n_bars.to_bytes(8, "big")
    + dims   # canonical order (§7), int32,   big-endian, 4 bytes each
    + births # canonical order (§7), float64, big-endian, 8 bytes each
    + deaths # canonical order (§7), float64, big-endian, 8 bytes each
).hexdigest()
```

The tag and the explicit length carry the same two guarantees §8.2 states for
the batch hash, one level down: the tag is what makes a diagram's digest
structurally distinct from a batch's rather than incidentally so, and the
length keeps an empty diagram from hashing to `sha256(b"")` — a published
constant, indistinguishable from a bug that hashed nothing at all. Columns
rather than interleaved rows so that an implementation can hash three
contiguous blocks instead of walking the diagram bar by bar in Python; the
layout is fixed either way, and nothing else in this document depends on which
was chosen.

**Negative zero MUST be normalised to `+0.0` before hashing.** `-0.0 == 0.0`
is true, so §6.3's `==` calls two diagrams differing only in the sign of a
zero equal, and §7's canonical order cannot separate them either: a stable
sort leaves numerically equal keys in the order they arrived. Their raw IEEE
754 bytes differ. Without the normalisation, such a diagram's digest depends
on the order the backend happened to emit its bars in — directly contradicting
§7's "stable across calls regardless of how, or whether, a diagram is ever
serialized", and making `d1 == d2` with differing `content_hash`es reachable.
Zero births are ubiquitous in H0 and `-0.0` is an ordinary product of
filtration arithmetic, so this is a live case rather than a curiosity. No
other value needs normalising: `NaN` cannot occur (I4, I5), and `+inf` has one
representation.

### 8.2 Batch content hash

```python
b.content_hash   # -> str, sha256, domain-separated composition of member
                  #    content_hashes, in batch order, plus len(b)
```

Resolves D7. `DiagramBatch` gets its own hash, defined separately from
`PersistenceDiagram.content_hash` rather than reusing it, but derived from it
rather than an independent computation. Two properties are new here that
§8.1 didn't need to state, because a single diagram has neither an order
across sibling objects nor a sibling type it could be confused with.

**Composed from member hashes, not re-serialized.** `b.content_hash` MUST be
built from `b[i].content_hash` for each `i`, not by re-running §7's
canonical sort and §8.1's hash over the batch's raw concatenated buffer.
Composing is cheaper, each member hash is already available or cheap to
recompute on its own, and it localizes a mismatch: if two batches' hashes
differ, comparing member hashes pinpoints which diagram changed, without
either side ever re-serializing the whole batch to find out.

**Order-sensitive.** `b.content_hash` MUST hash the member hashes in batch
order, `b[0], b[1], ..., b[len(b) - 1]`, never a sorted or otherwise
canonicalized order. §6.3 already makes `DiagramBatch` equality
order-sensitive across diagrams for the same reason: onboarding §9.3's
leading batch dimension is a positional axis, and `[A, B]` and `[B, A]` are
different batches even when they hold the same two diagrams. A hash that
ignored order would be answering a looser question than `==` already
commits to, the exact drift §6.3 warns a tolerance can introduce silently.

**Domain-separated from `PersistenceDiagram.content_hash`.** A batch of one
diagram MUST NOT hash to the same value as the diagram it contains, and this
MUST be structural, not incidental, the same "structurally impossible rather
than merely prohibited" standard §3's `xp` property and I7 hold elsewhere in
this document, not "a hash of hex digests happens in practice to differ from
a hash of raw coordinates." Concretely:

```python
b.content_hash = sha256(
    b"akriti.DiagramBatch.v1\x00"
    + len(b).to_bytes(8, "big")
    + b"".join(bytes.fromhex(b[i].content_hash) for i in range(len(b)))
).hexdigest()
```

The `b"akriti.DiagramBatch.v1"` tag and the explicit `len(b)` are what
guarantee, respectively, domain separation from `d.content_hash` and that an
empty batch, or a batch truncated to the wrong length, can't be confused with
a valid prefix or continuation of a different batch's concatenated hashes.
Both MUST be present in whatever the actual implementation does, in this form
or an equivalent one; neither is discretionary. §8.1 now carries its own tag
and length for the same two reasons, so domain separation holds from both
sides rather than resting entirely on this one: neither digest is a plain hash
of unframed bytes that the other could reproduce by accident.

**Exact equality only. No approximate form is offered.** §6.3 separates `==`
from `allclose` because conflating exact and tolerance-based comparison is
how a tolerance silently becomes load-bearing in a statistical test;
`content_hash` inherits that split rather than reopening it. `b1.content_hash
== b2.content_hash` implies `b1 == b2` (up to hash collision) and says
nothing about `b1.allclose(b2, ...)`. There is no approximate counterpart,
because no hash function is consistent with a tolerance: two diagrams a
`rtol=1e-6` apart hash differently, correctly, the same reason `d1 == d2`
correctly fails between a GUDHI diagram and its Ripser counterpart (§6.2)
while `d1.allclose(d2, rtol=1e-6)` correctly passes. A hash that tried to
absorb a tolerance would be exactly the kind of loosening §6.3 already rules
out for `PersistenceDiagram`; §8.2 does not reopen that question for the
batch case.

**Relationship to serialization determinism (§10.1 requirement 4).**
`b.content_hash` is computed purely from in-memory `content_hash` values
(§8.1) and is well-defined without ever touching a serialized file, the
same independence §10.1 already states for the single-diagram case.
Requirement 4 adds a second, external route to the same verdict: once
`.akd` byte-determinism holds, a checksum or `diff` on two `.akd` files
agrees with `b1.content_hash == b2.content_hash` without invoking `akriti`
at all. That equivalence is what lets `repro/`'s Paper-table reproduction
bar be checked either way, in Python via `content_hash` or outside it via a
file checksum, and it is why requirement 4 matters specifically at the
batch level: a table's provenance is a batch-level claim, the same framing
D7's original question opened with.

---

## 9. Delegation hazards

Akriti delegates computation and owns inference. Delegation is only safe where
the delegate is correct, and two of ours are not, in ways that produce clean
plausible numbers rather than errors. A third hazard is not a defect in any
delegate: two of them are each correct and disagree with one another, because
their defaults differ and neither records which default it used (§9.3). All
three are recorded here because `core/distances.py` is written against this
document.

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
silent-wrongness bug; it is a **severity-mismatch** bug: the warning states
the mechanism accurately (points are being dropped) but not the consequence
(the returned value is not the bottleneck distance, and no finite value is),
so it reads as a routine preprocessing note rather than "this answer is wrong."

It also travels badly — shown once per location under the default filter,
absent from most logs, and erased entirely by the
`warnings.filterwarnings("ignore")` that sits near the top of a great many
scientific Python scripts, including, until this was caught, our own
evidence script. Net effect: a user comparing a connected sample against a
disconnected one still gets a small distance and still concludes they are
similar, with one line on stderr first, if anything upstream hasn't already
turned warnings off.

**Requirement on `core/distances.py`.** Before delegating, it MUST partition
both diagrams by `essential`. If the essential-bar counts differ **per
dimension**, the distance is `+inf` and MUST be returned as such without calling
the backend. If they agree, delegate on the finite parts only, handle +inf bars
internally, and combine responsibly. It MUST NOT pass a diagram containing `inf`
to persim.

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

**Status of this adapter: best-effort compatibility shim, not a peer of
`from_gudhi`/`from_ripser`, and its spec stays exactly this large regardless.**
The audit puts giotto-tda's install rate at roughly 6,391/month against zero
commits in 52 weeks — the "slot is vacant, users are stranded" case this
project exists to catch. A material share of Akriti's earliest adopters will
be migrating pipelines that already have giotto's output shape baked into
them: batched arrays, `reduced_homology` defaults, padding rows included.
`from_giotto`'s contract is not held to giotto staying maintained (§9.2
already assumes it will not) and MUST NOT block on anything getting fixed
upstream. But that is a statement about *priority relative to the other four
adapters*, not about *scope*: the essential-bar handling (§5.1), padding
disambiguation (§4, Appendix A.2), and this section's frozen-fixture testing
requirement all stay as specified. Thinning any of it would just move the
cost from this project maintaining a shim once onto every migrating user
rediscovering the same three hazards on their own, which defeats the point of
a shim aimed at a stranded userbase.

*Clean-room note (onboarding §8): giotto-tda is AGPLv3. The above was determined
by calling public API and reading a traceback. No giotto source has been read,
and MUST NOT be read while implementing `compat/`.*

### 9.3 GUDHI and Ripser compute over different coefficient fields by default

| Backend | Parameter | Default |
|---|---|---|
| GUDHI | `SimplexTree.persistence(homology_coeff_field=...)` | **11** — ℤ/11 |
| Ripser | `ripser(..., coeff=...)` | **2** — ℤ/2 |

Measured; Appendix A.5 carries the run, the environment, and the finding that
**neither backend returns the field it used**, so nothing in either returned
object records the disagreement.

Neither backend is wrong, and that is what separates this from §9.1 and §9.2.
Persistent homology is computed with coefficients in a field, the two chose
different ones, and both document their choice. The hazard is ours, and it
arrives the moment we call both and compare the results. **Two diagrams of the
same point cloud from our two primary backends are not, by default, diagrams
of the same thing**: they agree wherever the data is torsion-free and differ
wherever it is not, and no property of either diagram tells a caller which
case they are in.

This is recorded independently of D17. D17 decides what the adapters *do*
about it — whether `coeff_field` becomes a required argument (§12.1); §9 is
where the fact is written down, and it is true whichever way that lands.

**The consequence lands on §6.3.** Cross-backend agreement is precisely what
`allclose` exists for: §6.2 defines its tolerance against Ripser's single
precision, and §11.2 requires a GUDHI-vs-Ripser comparison at `rtol=1e-6`.
Under default settings that comparison can be carefully matching bars between
objects that are not comparable in the first place — and it will return
`True`, because test data is usually torsion-free. An `allclose` of `True`
across these two backends is evidence that the bars agree, not that the two
computations answered the same question.

**Requirement on §11.2's cross-backend test.** It MUST pin the coefficient
field explicitly on both sides rather than take each backend's default, and
MUST carry a comment pointing here. Pinning is available on both — it is a
call parameter on each (A.5) — so this costs nothing, and it removes the one
place in this document's own test suite where the hazard is otherwise live.
This is not an answer to D17: it constrains a test this RFC already requires,
not the adapter signatures D17 is about.

---

## 10. Serialization

### 10.1 Requirements

The five requirements below are normative, and each states an obligation on
the on-disk format and on `save`/`load`.

1. **Round-trips exactly.** `load(dump(d)) == d` MUST hold, including `inf` and
   multiplicity, **and `load(dump(d)).same_provenance(d)` — metadata
   round-trips too.** The second clause is not redundant. §8 requires `meta`
   to take no part in `==`, so a `load` that silently discarded every byte of
   `params` and `provenance` would satisfy the first clause completely. §5
   depends on the second: its argument for why `finitize` must not
   copy `essential_bars` forward into `essential_bars_source` turns on a
   diagram arriving from `load` already carrying a `"finitized_*"` value, and
   there is no such diagram unless `load` preserves provenance. The two
   clauses cover the two halves of a diagram that `==` deliberately splits,
   and requirement 1 needs both.
2. **Zero-dependency by default, with a narrow, lazily-imported exception at
   the `save`/`load` boundary.** `diagrams/core.py` and
   `diagrams/adapters.py` carry no third-party dependency at all: importing
   the package, or constructing, inspecting, or comparing a diagram, never
   touches one. The single normative on-disk format's `save`/`load`
   implementation MAY depend on one third-party library, provided the import
   is lazy and function-scoped, confined to those two functions, and nothing
   outside `save`/`load` requires it. That library MUST be declared as an
   install extra carrying a version floor, and the lazy import MUST fail
   actionably on both absence and an unsatisfied floor.
3. **Self-describing and versioned.** The format MUST identify itself and
   carry the version of this specification that wrote it.
4. **Deterministic.** Identical diagrams MUST produce identical bytes.
5. **Readable enough to inspect without our library.** Metadata MUST be
   recoverable with no TDA-specific and no scientific-Python tooling.

**These five requirements do not independently eliminate every alternative.** Walked through against HDF5 (h5py) and Parquet (pyarrow) as candidate default formats:

**Requirement 1 does not discriminate.** All three candidates, `.npz`, HDF5,
and Parquet, store IEEE 754 doubles natively and round-trip `inf` and
duplicate rows without loss. §10.3's `to_parquet()` already concedes this for
Parquet; there is no reason HDF5 would fare worse.

**Requirement 3 does not discriminate either, though it looks like it
should.** None of the three formats has a native slot for
`DiagramMeta.params` or `.provenance`, both arbitrary `Mapping[str, Any]`.
HDF5's attribute system and Parquet's file-level key-value metadata are both
plausible homes for a serialized metadata blob, exactly as `meta.json` is
here. Whatever format is chosen, an arbitrary mapping gets JSON-encoded and
attached somewhere; this requirement is satisfied by that pattern, not by the
container format.

**Requirement 4 exists for artifact-level reproducibility**, not for
round-tripping and not for `content_hash`. `load(dump(d)) == d` (requirement
1) needs only one save-then-load cycle to recover the original diagram; it
says nothing about whether two separate calls to `save(d)` produce matching
bytes. `content_hash` (§8.1) is computed from the diagram's own
canonical-ordered arrays in memory, never from a serialized file, so it is
already well-defined by §7's canonical ordering independent of anything in
this section. What requirement 4 actually protects is the ability to verify,
with a checksum or `diff` alone, that regenerating a `.akd` fixture — from
`repro/`, for instance — reproduces exactly the file previously committed or
published, without invoking this library at all: the same
audit-without-our-library spirit as requirement 5. §11.2 tests it directly,
as its own case, separate from the round-trip and invariant tests.

**Requirement 4 is not currently satisfied by any candidate, including the
one chosen.** "Identical diagrams produce identical bytes" is stated as a
requirement but no mechanism is specified. Zip archives carry per-entry
metadata — timestamps, compression method flags — that `save()` MUST pin
explicitly (fixed `ZipInfo.date_time`, a fixed compression setting) or two
writes of the same diagram will differ in bytes. HDF5 has the same class of
exposure through superblock and library version headers; Parquet through
writer version strings and row-group layout. This is an open implementation
obligation for `save()`, not a property `.npz` gets for free, and MUST be
tracked as such rather than assumed solved by requirement 4's presence in
this list.

**Requirement 5 is the actual discriminator.** A zip container puts
`meta.json` in the archive as literal UTF-8 text: any unzip tool or `cat`
reads it, with no TDA-specific or even scientific-Python tooling involved.
HDF5 and Parquet do not offer this. Both are single opaque binary files where
metadata lives inside the same binary structure as the array data; inspecting
just the metadata requires `h5py` or `h5dump` for one, Arrow-aware tooling for
the other. This is a structural property of the container, not a byproduct of
which array library the RFC happened to want.

**The format choice therefore follows from requirement 5, and requirement 2's
role is downstream, not independent.** Once a zip-based container is chosen
for requirement 5's sake, `.npz` is the zip-compatible array format already
ambient in every backend adapter's upstream graph (§11: GUDHI, Ripser, and
persim all hand back or consume `numpy` arrays before a diagram reaches
`save`/`load`), and `numpy` follows as a consequence of that. Requirement 2
constrains how that dependency may enter the codebase — lazily,
function-scoped, confined to `save`/`load` — but does not explain why it is
`numpy` rather than something else. That explanation is requirement 5 plus
§11's adapter contract.

**This argument was run against binary containers only, and the text-based
comparison it left open has since been run.** HDF5 and Parquet are what it
tested. `csv`/`tsv` and `sqlite3` — fully dependency-free, clearing
requirement 2 outright rather than through this section's narrow lazy-import
exception — were measured separately, and **D12 resolved in favour of the
binary payload**. The reason is not that CSV loses on requirement 5; it wins
there, being readable without even `numpy.load`. It is that requirement 5 is
already satisfied without it, twice: once by `meta.json` sitting in the
archive as literal UTF-8 text, and once by §10.3's `to_csv()`, which exists to
be exactly this format's human-readable surface. **Requirement 5 does not need
satisfying a third time**, and paying CSV's cost on every `load()` — roughly
2× the bytes and, at the scale Appendix A.6 measures, close to two orders of
magnitude in load time — to duplicate an escape hatch this document already
ships is the wrong trade. Appendix A.6 carries the figures, the one argument
for CSV that survives them, and the condition under which D12 should be
reopened.

Plain JSON alone still fails outright, independent of all of the above: `inf`
is not valid JSON, and the `Infinity` token Python emits is a non-standard
extension other languages reject. Bare `.npz` alone still fails requirement
3; it has no metadata story. Parquet remains available as an optional,
`pyarrow`-gated escape hatch (§10.3, D8) — an extra, not the default, since
it still fails requirement 5's inspectability bar and does not sit in any
adapter's upstream graph.

### 10.2 Format: `.akd`

An `.akd` file is a zip archive, `zipfile` is standard library, that
separates human-readable metadata from binary array data. That split is what
requirement 5 (§10.1) actually turns on.

**The array payload is settled, and normative: `bars.npz` (D12).** §10.1's
requirement-5 argument originally tested `.npz` against HDF5 and Parquet only,
and left the two stdlib candidates that clear requirement 2 outright,
`csv`/`tsv` and `sqlite3`, unmeasured — with CSV a live contender to beat
`.npz` on requirement 5 itself. Appendix A.6 measures both; §10.1 carries the
resolution. `bars.npz` below is a requirement rather than a working default.

```
meta.json      UTF-8 JSON, sorted keys, the DiagramMeta plus a format version
               — the format version is what satisfies §10.1 requirement 3
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
  `numpy` (§10.1). For a `DiagramBatch`, an integer `diagram_id` column
  is prepended rather than an `offsets` array — Parquet's natural unit is the
  row, not a CSR buffer. Carries none of `DiagramMeta`: no `backend`, no
  `provenance`, no `params`. This is a bars-only interchange table for
  R/pandas/Polars pipelines (§1's R-bridging goal), not a `.akd` replacement.

---

## 11. Adapter contract

Signature for all five, with two deviations, both on `from_giotto`:

```python
from_gudhi(obj, **meta)   -> PersistenceDiagram
from_ripser(obj, **meta)  -> PersistenceDiagram
from_giotto(arr, *, reduced_homology, **meta)  -> DiagramBatch
from_persim(obj, **meta)  -> PersistenceDiagram
from_array(arr, **meta)   -> PersistenceDiagram
```

`from_giotto` alone takes a required keyword-only argument outside `**meta`.
This is deliberate, not an inconsistency to fix later: `reduced_homology`
determines whether the diagram is silently missing its H0 essential class
(§5.1), so omitting it MUST be a `TypeError` at the call site, not a value
that can slip past as an optional key in `**meta`. §5.1 requires omission to
raise; this clause fixes what it raises and where.

`from_giotto` alone also has a fixed return type rather than the scalar
`PersistenceDiagram` every other adapter returns. An earlier draft had it
return `PersistenceDiagram | DiagramBatch`, a scalar when `n_samples == 1`
and a batch otherwise, which makes the caller's own result type depend on a
runtime property of the data, exactly the shape-depends-on-what-else-was-there
hazard §4 and Appendix A.2 exist to rule out, just moved from inside an array
to the adapter's return type. `from_giotto` MUST instead always return a
`DiagramBatch`, of length one when `n_samples == 1`. The caller who knows
they passed a single sample can unwrap explicitly (`batch[0]`); nothing about
the adapter's own return type is allowed to depend on how many samples the
particular call happened to carry.

Every adapter MUST: validate against §3.1; populate `backend`,
`backend_version`, `provenance`; preserve backend row order; and never
finitize, sort, or deduplicate.

An adapter that records `provenance["essential_bars"]` MUST record
`provenance["essential_bars_source"]` with the same value in the same
construction (§5, §8). The adapter is the only writer that can: the key means
"the verdict at computation time", and every later writer sees only a field
that may already have been overwritten. `finitize` (§5) MUST NOT write it.

Measured input formats (Appendix A):

| Source | Accepted input | Notes |
|---|---|---|
| GUDHI | `SimplexTree.persistence()` → `list[(dim, (b, d))]`; also `persistence_intervals_in_dimension(k)` → `(n,2)` | `inf` faithful. Both forms MUST be accepted; the `list` form carries all degrees at once. |
| Ripser | `ripser(X)` → `dict` with `"dgms"`; `Rips().fit_transform(X)` → `list[(n,2)]` | Index in the list *is* the degree. `inf` faithful. `float32` precision (§6.2). |
| giotto | `(n_samples, n_bars, 3)` array, columns `(birth, death, dim)` | Essential bars lost (§5.1). Padding ambiguity (§4). Always returns a `DiagramBatch`, length 1 when `n_samples == 1`. |
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
with what the backends actually emit. **A frozen fixture counts as real
backend output** provided it was captured from an actual call to the backend
and committed verbatim; "real" is about provenance, whether a real backend
produced the array, not about whether the call happens live in this run.
§9.2's requirement that `from_giotto` be tested against stored fixtures rather
than a live call is this rule applied to a backend that is currently broken
on install, not an exception to it: the fixture was still real giotto output
when it was captured, and remains so. A hand-written `(n,2)` array typed in to
match what a backend is believed to return does not qualify either way. The
suite MUST include, at minimum:

- A diagram with essential bars (GUDHI, Ripser).
- An empty diagram, and a diagram empty in one degree but not another.
- A diagram with repeated identical bars — multiplicity MUST survive.
- A diagram with a genuine zero-persistence bar.
- Cross-backend agreement GUDHI vs Ripser on the same point cloud, with an
  explicit `rtol=1e-6` and a comment pointing at §6.2. The coefficient field
  MUST be pinned explicitly on both sides, with a comment pointing at §9.3:
  the two backends default to different fields, so a comparison taking the
  defaults is matching bars across two different homology theories.
- **`allclose` on two diagrams that are within tolerance of each other but
  whose canonical orders differ because of that tolerance**, asserting
  `True`. Two bars in one dimension whose births are tied to within `rtol`
  and whose deaths are far apart: a bijection within tolerance exists, and
  the tie is what lets §7's sort place the two bars in opposite orders on the
  two sides. The rejected sorted-pairwise form pairs each bar against the
  other's partner and returns `False`; the matching §6.3 requires returns
  `True`. This is the case that motivated the matching, and a suite without
  it passes identically against either implementation.
- `save`/`load` byte-determinism: dumping twice gives identical bytes.

Property-based tests (Hypothesis) for the invariants and for both clauses of
§10.1 requirement 1 — `load(dump(d)) == d` *and*
`load(dump(d)).same_provenance(d)`, since the first passes on a `load` that
drops metadata entirely; onboarding §10 requires them for the numerical layer
and they fit this layer unusually well.

`content_hash` (§8.1, §8.2) MUST be tested against the byte layout this
document specifies, not against whatever the implementation currently emits,
and the test MUST cover **both** paths an implementation may take to produce
those bytes. An implementation that reinterprets a backend's own buffer where
one is available, and falls back to per-element packing where it is not, has
two code paths that must agree; `array_api_strict` exposes no buffer, so a
suite that exercises the hash only under the conformance backend tests the
fallback and never the path every NumPy-backed diagram takes. A divergence
between them would make a published digest depend on which backend recomputed
it, which is precisely what §8.1 fixes the message layout to prevent. The
minimum is: the two paths agree byte-for-byte on `±0.0`, `+inf`, subnormals
and the `int32` extremes; an empty diagram matches §8.1's tag-plus-length
message rather than `sha256(b"")`; a one-diagram batch does not collide with
its member; and the same bars hash identically under two namespaces.

---

## 12. Open decisions

Fifteen decisions are on record: D1-D8 and D12-D18. **Two still need the
lead's judgment before M1** (§12.1); the other thirteen are settled (§12.2),
each stating the outcome and pointing at the section that carries the
normative requirement. Superseded recommendations are logged in the history
document rather than repeated here.

**D6 is reinstated as superseded, not deleted.** It was removed with D9, D10
and D11 in the pass below, and its resolution has since been reversed: the
row is restored in §12.2 carrying both its original resolution and the one
that replaced it.

**D9, D10, and D11 were removed from this RFC** as
dependency-and-licensing policy questions rather than interchange ones; that
policy belongs to the onboarding document, which owns it. Nothing normative
went with them — §3.3 and §10.1 state the zero-dependency-by-default
requirement and `numpy`'s lazy-import behaviour directly, in MUST language,
and never depended on a table row to carry it. Their prior text and the
reasoning for removing them are in the history document. D-numbers are not
renumbered to close the gap; they are stable identifiers, not a dense
sequence.

### 12.1 Needs the lead before M1

| # | Question | Recommendation / status |
|---|---|---|
| **D17** | §8's `DiagramMeta` block annotates `coeff_field` with "affects the diagram, must be recorded", and that comment is the only place the field appears anywhere in this document. The prose immediately below it says the opposite — "All fields are optional", with `from_*` adapters required to populate `backend`, `backend_version` and `provenance` and nothing else — and §8.1's `content_hash` covers bars and never metadata, so no other clause depends on the value being present either. The comment's underlying claim is sound: homology over ℤ/2 and ℤ/3 differ wherever there is torsion, so a diagram whose coefficient field is unknown is uninterpretable in the way §8's opening sentence says one whose filtration is unknown is. But the three fields §8 does require are all derivable from the adapter itself, and this one is not — §11's adapters receive a computed result plus `**meta`, not the call that produced it. Does `coeff_field` become a required keyword-only argument on the adapters whose backend takes a coefficient parameter, on the `from_giotto`/`reduced_homology` precedent (§5.1, §11); does §8 require it only where the backend's returned object exposes it; or does the comment's normative clause go, leaving the field optional as the prose already has it? | **Narrowed to two options by Appendix A.5; still needs the lead.** Found by auditing this document's RFC-2119 keyword use rather than against `core.py`. It was recorded as the sole lowercase obligation without an uppercase counterpart; two later sweeps found more, all now promoted, and this remains the only one whose *underlying obligation* is in question rather than its register. The keyword line's caps-only rule settles its normative status — the comment states no requirement — but not the question it raises, which is whether the obligation it describes ought to exist; it still contradicts, as plain prose, the paragraph seven lines below it. Which of the three options is even available turned on a per-backend fact this RFC did not state; **Appendix A.5 now measures it.** No backend returns the coefficient field it was computed with — it is a call parameter on GUDHI, Ripser and giotto alike and is absent from every returned object — so the middle option, requiring it only where the returned object exposes it, applies to nothing and is out. The defaults disagree besides (GUDHI ℤ/11, Ripser ℤ/2), so an unrecorded value is unknown rather than conventionally ℤ/2. **That disagreement is now recorded as a delegation hazard in its own right, §9.3**, independently of how this row lands: §9 records that it is true, D17 decides what the adapters do about it. Two of the five adapters are out of reach whatever the answer: `from_array` has no backend, and `from_persim` consumes diagrams rather than computing them. The value having proved unrecoverable, this is the `reduced_homology` question again, and §5.1 answered that one by putting the parameter in the signature and making omission raise — a signature change on up to three adapters, not something to take on this document's own initiative. What remains is a judgment, not a further measurement: whether the obligation should exist at the cost of that signature change, or whether the comment's normative clause goes and the field stays optional as the prose already has it. §8's field list is not itself in question: `coeff_field` sitting top-level alongside whatever backend-specific key `params` carries is the same arrangement §8 defends at length for `provenance["essential_bars"]` against `params["reduced_homology"]`. The comment is left exactly as it stands pending this call, since rewording it in either direction answers the question. |
| **D18** | `torch.Tensor` does not implement `__array_namespace__`. array-api-compat's own documentation says so outright — "we do not wrap the `torch.Tensor` object. It is missing the `__array_namespace__` and `to_device` methods, so the corresponding helper functions… should be used instead" — PyTorch's compatibility tracker (gh-58743, open, last touched 2026-03-16) holds the attribute back deliberately as "the attribute that declares compliance", to be added at near-full conformance, and it is absent from the torch 2.13 `Tensor` reference. §3 defines `Array` as any object implementing that method, so **no diagram can currently be torch-backed**, and four sites in this document illustrate a second namespace with torch counterfactually: §3.3's "a diagram built from torch tensors stays torch-backed", D16's own "admitting a torch/NumPy mix into one diagram", §4.2's mixed-input `from_diagrams` check, and §6.3's "a NumPy diagram and a torch diagram… are each valid and still not comparable". D16's CI test is written against "every backend this project supports" and `akriti[torch]` is one; for torch it cannot fail the way D16 describes — it raises `AttributeError` before reaching the identity question. JAX is unaffected: `jax.Array` has implemented the method natively since 0.4.32. Resolving namespaces through `array_api_compat.array_namespace` closes the gap. Does that resolver go in front of **(1) every backend**, or **(2) torch alone**, behind `akriti[torch]`, with the native method preferred wherever it exists? | **Recommendation: option 2.** What decides it is §10.1 requirement 2, not performance. Under option 1 `core.py` cannot resolve *any* namespace without array-api-compat, so it becomes a required dependency and `pip install akriti` no longer resolves to nothing third-party; under option 2 the import is lazy and fires only when the native method is absent, which is the `numpy`-in-`io.py` pattern §3.3 already uses and the placement `pyproject.toml` already anticipates. **Performance does not discriminate** (Appendix A.7): JAX pays nothing structurally, since array-api-compat ships no JAX wrapper and a `jax.Array` resolves to `jax.numpy` itself; NumPy pays one Python frame on the 11 of 26 namespace functions that carry a wrapper and nothing on the other 15, which are numpy's own objects by identity, putting §7's `canonical()` at 1.17x at 40 bars and 1.00x from 100k up. **Conformance does not discriminate either, which is the finding that removes option 1's best argument**: on `numpy` 2.5 the wrappers are largely vestigial — `device=`, `unique_values`, `cumulative_sum(include_initial=)`, `reshape(copy=)` and the 0-d `nonzero` rejection are all native — leaving one live correction, the `sort`/`argsort` stable default, which §7 already buys by passing `stable=True` explicitly. **The constraint either option must respect** is that `array_namespace()` on a NumPy array returns `array_api_compat.numpy`, *not* `numpy`: resolution MUST go through exactly one function, or one backend yields two namespace objects and I7's `is` raises on arrays that legitimately share a namespace — the loud direction D16 chose, but fired by our own inconsistency rather than a backend's. A third arrangement, preferring compat when it happens to be installed, is excluded outright: it makes `d.xp` depend on the environment rather than the input. The four counterfactual illustrations and D16's test scope are left exactly as they stand pending this call, on D17's precedent — option 1 makes them true as written, option 2 requires substituting JAX, so editing them in either direction answers the question. |


### 12.2 Settled

| # | Question | Recommendation / status |
|---|---|---|
| D1 | File extension `.akd`, or plain `.npz` with our layout inside? | **Resolved by §10.** §10.1/§10.2 normatively specify `.akd`; Parquet is excluded as the *default* format. This row originally recommended Parquet, contradicting both; the correction is logged in the history document. |
| D2 | Is `DiagramBatch` in scope for M1, or does M1 ship the single-diagram type only? | **In scope.** Retrofitting a batch container after `core/` is written against scalars is the expensive order, and onboarding §9.3 commits us to batch-shaped signatures. §4.2's CSR storage is chosen deliberately ahead of the neural-network path that needs it, a committed direction, not a hypothetical one; see §4.1 (why not dense-padded) and §4.2 (the CSR design itself). |
| D3 | Do we accept `float32` storage behind a flag for large-scale work? | No, not in v0. Revisit when a real memory complaint exists. |
| D4 | Should `from_giotto` default to `strip_padding=True`? | No. Defaulting to a lossy repair contradicts §5's whole argument. Warn and let the caller choose. |
| D5 | Does the RFC published at M1 include §9's delegation hazards, or do we raise them upstream first? | Raise upstream first — file the persim issue and the giotto scikit-learn issue, then publish citing our own reports. Costs two weeks, buys enormous goodwill, and turns a criticism into a contribution. |
| **D6** | Array-API support (§3.3) needs a NumPy that has it. Raise the floor to `numpy>=2.0`, or add `array-api-compat` and keep `numpy>=1.24`? | **Superseded, and reinstated here rather than deleted.** Original resolution: raise the floor to `numpy>=2.0`, main-namespace array API support having landed in NumPy 2.0, declared in `pyproject.toml` as a required dependency. That resolution was merged. It is now superseded, not withdrawn: `numpy` leaves the required closure — `pip install akriti` resolves to nothing third-party (§10.1 requirement 2) — but moves to `akriti[io]` rather than vanishing, so the `>=2.0` floor stays declared and resolvable at install time. D6's purpose is what makes the extra necessary rather than optional: undeclared, a user on `numpy` 1.24 gets an `AttributeError` at run time instead of a resolver error at install time, because a presence-only lazy import never fires on a version that is merely too old. §3.3 carries the version check and the two failure paths, both naming the extra; §10.1 requirement 2 carries the declare-an-extra-with-a-floor obligation. |
| **D7** | Does `DiagramBatch` need its own `content_hash`, and if so, defined how? | §8.2 defines `DiagramBatch.content_hash`: composed from member `PersistenceDiagram.content_hash`es in batch order, not re-serialized from the raw buffer; domain-separated from `PersistenceDiagram.content_hash` by a type tag, so a one-element batch cannot collide with the diagram it wraps; and exact-equality only, no approximate form, per §6.3's exact/approximate split. §4.2's and §4.3's cross-references updated to point at §8.2 rather than flag the gap. |
| **D8** | Should Parquet be offered anywhere, given §10.1 rules it out as the default (`.akd`) storage format? | §10.3's `to_parquet()` (`akriti[parquet]`, Apache 2.0, lazy-imported per §10.1's pattern). Previously logged as resting on D9/D11; those rows are now out of this RFC's scope entirely (see above), and this row no longer depends on them. Whatever the project's actual license-family policy turns out to be is a packaging-level check against `tools/check_license_closure.py`, not something this RFC re-litigates. `tools/check_license_closure.py` and `DEPENDENCIES.md` still need updating for the new extra. |
| **D12** | §10.2 specifies `bars.npz` as the default array storage inside `.akd`, defended in §10.1 on requirement 5 (inspectability) against HDF5 and Parquet. Two stdlib alternatives that clear requirement 2 outright rather than through the lazy-import exception, `csv`/`tsv` and `sqlite3`, were never run through the same test — and CSV plausibly satisfies requirement 5 *better* than `.npz` does, being directly readable without even `numpy.load`. Does `.npz` remain the default, or should one of these replace it? | **`bars.npz` stays, and §10.2's payload is now normative rather than provisional.** Resolved on measurement (Appendix A.6), which supplies the bar-count figure this row said it turned on: H0 bar count equals the input point count *exactly*, so bars scale linearly in cloud size, and a 5,000-sample batch is around 4.7 million bars. At 1M bars CSV costs ~2.1× the bytes and ~78× the load time; sqlite3 ~1.3× and ~42×. **The reason is not that CSV loses on inspectability — it wins there.** It is that requirement 5 is already satisfied twice without it, by `meta.json` as literal archive text and by §10.3's `to_csv()`, which exists to be the human-readable surface. Requirement 5 does not need satisfying a third time, and paying that cost on every `load()` to duplicate an escape hatch this document already ships is the wrong trade. **One argument for CSV survives, and is recorded rather than waved away:** a stdlib payload would let the `[io]` extra be dropped altogether, `meta.json` being stdlib `json` already, and "zero dependencies, including serialization" is a stronger claim than the one this document makes. It is not worth 78× load time at four-million-bar scale — but **that is the condition to reopen D12 against**, together with a use case where batches are small and dependency-freedom outweighs load time. **sqlite3 is closed out** on four counts: larger, far slower, not inspectable without a separate tool, and its files carry internal page state that makes requirement 4's byte-determinism harder rather than easier. A.6 records the caveat that belongs with the figures: two datasets, both alpha-complex, both low-dimensional. They fix the order of magnitude and the linear scaling, not a distribution over what users will store — and the decision is robust to that, because every direction the bar count moves makes CSV worse. |
| **D13** | `PersistenceDiagram` (§3) is single-parameter-persistence-shaped: one scalar `dim`, `birth`, `death` per bar. §3.2 referenced "the multiparameter case" in passing, but nothing in this RFC said whether that module reuses this type, needs a parallel type, or forces a breaking change to this one once it exists. Does `PersistenceDiagram` need a version boundary, an extension point, or an explicit non-goal statement now, before adapters and `core/` are written against its current shape? | **Explicit non-goal, stated in §1** — normative scope rather than an aside in §3. No extension point and no new version machinery. Multiparameter persistence modules do not decompose into intervals and admit no complete discrete invariant, so a multiparameter "diagram" is not this type with an extra column but a different object — a rank invariant, a fibered or signed barcode — and no extension point designed now would fit a shape nobody can yet specify. An extension point would instead put a case into every adapter, accessor and invariant check to serve a module that is not on the roadmap through M4. If it is ever built it takes a parallel type and the two coexist; this is not a deprecation path. The forward-compatibility cost is already paid by §10.1 requirement 3's format version, and that is the whole of the version boundary this needs. §3.2's `d.h0`/`d.h1` note no longer leans on the multiparameter case: `d.dim(k)` is canonical and an unwithdrawable alias is a permanent liability regardless. |
| **D14** | §6.3 required `allclose` to be approximate and order-insensitive but did not say how bars are paired. Sorting both sides canonically (§7) and comparing pairwise is exact in the sort and approximate in the comparison, and the two do not compose: bars whose births lie within tolerance of each other can canonicalise into different orders on two backends, and the comparison then returns `False` for diagrams that do have a partner for every bar within `rtol` — at the `2.7e-8` magnitude Appendix A.3 measures, on exactly the cross-backend case §6.2 defines `allclose` for. Does `allclose` become a matching over the multiset, or does §6.3 accept the conservative false negative and require it to be documented? And is the tolerance symmetric, or does `rtol` scale `other` as `numpy.allclose` does? | **Both resolved in §6.3.** `allclose` MUST be a matching: a bijection under which every matched pair shares a `dim` exactly and agrees on both coordinates within tolerance; equal bar counts are necessary and not sufficient. The tolerance MUST be symmetric, `atol + rtol * max(abs(a), abs(b))`, deliberately diverging from `numpy.allclose` and documented as such in the method's own docstring. `allclose` is reflexive and symmetric but not transitive, and MUST be documented as not an equivalence relation. Accepting the false negative was weighed and rejected: the caller's remedy for a spurious failure is to widen `rtol` until it passes, which relocates §6.3's silent loosening into user code where nobody reviews it. No sort key repairs the pairwise form — the history document carries that argument. No new dependency: an augmenting-path matching in the standard library is sufficient at these sizes. Not bottleneck distance, and §9's delegation rule is unaffected in both directions — `core/distances.py` MUST NOT be built on this method. |
| **D15** | §8 reserves `provenance["order"]` with values `"backend"` and `"canonical"`, but names no writer for the second. §7 forbids adapters from sorting, so every `from_*` adapter records `"backend"`; `d.canonical()` is the operation that makes `"canonical"` true of a diagram, and §7 has it carry `meta` through unchanged, so a sorted diagram still reports `"backend"` and nothing ever writes the other value at all. Does `canonical()` become a second writer of `order`, does `save` write it (§10.2 emits `bars.npz` in canonical order regardless of what the in-memory diagram reports), or does the key not earn its place now that §7 makes row order advisory to a reader and load-bearing for nobody? | **The key goes.** §8's reserved-key table drops `order`; §7 and `core.py` drop their references to it. What decides it is not the missing writer but a property every other reserved key has and this one does not: **the rest record facts that vanish if unrecorded.** `source_dtype` is lost the moment the array is upcast, `clamped_rows` the moment the rows are repaired, `padding_removed` the moment they are stripped; `essential_bars_source` records a then-versus-now distinction no later inspection can recover. Whether rows are in canonical order is recoverable from the arrays themselves, exactly, in one pass, forever. `order` is a cached answer to an always-computable question, and one that goes stale: anything reordering rows while carrying `meta` through unchanged leaves a diagram reporting `"canonical"` while not being canonical — clean, plausible and wrong, §9's own category, arrived at by our hand rather than a dependency's. **Having `save` write it is worse rather than better**: §10.2 emits `bars.npz` in canonical order unconditionally, so the key would be a constant in every file on disk — no information, and a claim about the in-memory diagram it came from that may be false. And §7 leaves no consumer who could act on it either way, every consumer being required to treat row order as arbitrary, so a reader seeing `"canonical"` is forbidden to do anything differently. **The one genuine order fact is a different key**, and §8 records it so this is not reopened: whether the *backend's own output* was already canonical is not derivable, is exactly the GUDHI-versus-Ripser disagreement §7 documents and A.3 measures, and is unrecoverable once anything sorts. `order` as specified never captured it — every adapter records `"backend"` regardless of what the backend did — so it is a key to build at adapter time if order provenance is ever wanted, not this one repaired. |
| **D16** | I7, B5 and §4.2's `from_diagrams` check are all written as `is` on `__array_namespace__()`, and `core.py` implements them that way. The array API standard requires that method to return "an object representing the namespace"; it does not require the same object on every call, and it takes an `api_version` argument that a backend could legitimately answer with different wrapper objects. NumPy and `array_api_strict` return the module itself, so identity holds there and the assumption is invisible. Does the RFC require namespace *identity*, or a weaker equivalence — and if weaker, what is the portable test, given the standard defines no namespace equality? | **Require identity, state it as a supported-backend constraint, and verify it in CI (§3.3).** What decides it is the **direction of the failure**, not the theory — the standard does not guarantee identity, and no portable surrogate exists, both as this row observed. `is` fails by raising `ValueError` on arrays that legitimately share a namespace: conservative, loud, immediately diagnosable, impossible to mistake for a correct answer. Every available surrogate — `__name__`, a sentinel dtype, anything else the standard leaves — is a weaker test that can *match across genuinely different namespaces*, admitting a torch/NumPy mix into one diagram, which is the silent direction and defeats exactly what I7 exists to prevent. Given a conservative check with a documented constraint against a clever one that fails open, take the first. §3.3 therefore states plainly that akriti requires `__array_namespace__()` to return a consistent object for a given backend — a constraint on supported backends, not a property claimed of the standard. **Made verified rather than assumed:** a CI test MUST assert identity holds for each supported backend, which is §3.3's own "conformance is tested, not intended" and follows entry 20's precedent, where the `hasattr(xp, "lexsort")` trap became a standing regression test rather than prose. If a backend ever returns a fresh wrapper per call the test fails and D16 reopens as a real decision, instead of breaking silently in someone's pipeline. I7, B5 and §4.2's `from_diagrams` check are unchanged; only their rationale cells now cite the requirement rather than the open question. |

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

All three giotto rows were run with `reduced_homology=True` (giotto's
default); that parameter, not `infinity_values`, accounts for 39 rather than
40 H0 bars (§5.1).

**This table does not measure the claim §5.1 rests on.** It varies
`infinity_values` across three settings and holds `reduced_homology` fixed at
`True`, so it establishes that `infinity_values` is *not* the cause and
leaves `reduced_homology` as the inference. A `reduced_homology=False` row
would show the effect directly, and `probe_backends.py` MUST gain one before
M1. It has not been run because giotto-tda 0.6.2 does not execute on the
scikit-learn in this environment (§9.2), which is the same reason §9.2
requires `from_giotto` to be tested against frozen fixtures — so the row will
have to come from a pinned-environment capture, committed the way §11.2
accepts a fixture as real backend output.

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

The counts are the tell: the **correct** row raises *two* warnings, the
**wrong** row raises *one* — the warning tracks whether an argument contained
an essential bar, not whether the result is meaningful. Row 1 is right only
by accident (dropping matching essential bars from both diagrams happens to
preserve a distance of zero), so neither the warning's presence nor its
absence can be used to certify a result.

### A.5 Coefficient field — recoverability from backend output

Measured 2026-08-06, not on the 2026-07-29 run above: `gudhi 3.13.0`,
`ripser 0.6.15`, `persim 0.3.8`, `numpy 2.5.1`, `scikit-learn 1.9.0`.
giotto-tda is not installed in this environment and its row is not measured
(§9.2).

The question is D17's: not whether a backend *accepts* a coefficient field,
but whether the object it hands back carries the value it was computed with.

| Backend | Parameter | Default | Carried on the returned object? |
|---|---|---|---|
| GUDHI | `SimplexTree.persistence(homology_coeff_field=...)` | **11** | **No.** Returns `list[(dim, (b, d))]`; `SimplexTree` exposes no attribute naming a coefficient field. |
| Ripser | `ripser(..., coeff=...)` | **2** | **No.** Returned `dict` keys are `cocycles`, `dgms`, `dperm2all`, `idx_perm`, `num_edges`, `r_cover`. |
| giotto | `VietorisRipsPersistence(coeff=...)` | 2 | Not measured. The value sits on the estimator; `from_giotto` (§11) receives the `(n_samples, n_bars, 3)` array, which has no slot for it. |
| persim | — | — | Consumes diagrams, computes no homology. |
| array | — | — | No backend. |

**No backend returns the coefficient field it computed with.** On every one it
is a call parameter and is absent from the returned object, so an adapter
cannot recover it from its input — the value exists only in the caller's own
call.

**The defaults also disagree: GUDHI computes over ℤ/11, Ripser over ℤ/2.** An
unrecorded `coeff_field` is therefore not conventionally ℤ/2; it is genuinely
unknown, and two diagrams of the same data from these two backends differ
wherever the data has torsion. This is D17's option 2 — "require it only
where the returned object exposes it" — measured out of existence, and it
narrows D17 to two options rather than three. Which of those two is right is
a judgment about whether the obligation should exist, not a further fact, and
§12.1 still carries it.

**The default disagreement is a hazard on its own, and §9.3 records it as
one.** It does not wait on D17: whichever way that lands, GUDHI and Ripser are
computing different homology theories on the same input unless a caller
intervenes, and §6.3's cross-backend `allclose` is the place that bites.

### A.6 Bar counts and array-payload format comparison

D12's evidence. **Measured by the project lead and reported on PR #10**, not
on either run above; reproduced here as the record, since a decision log that
cites numbers it does not carry is one nobody can check later.

Alpha complex over two datasets from the `classify` repository. The cached
diagrams there are truncated to `top_n=50` and diagonal-padded, so they
saturate and measure nothing; these are recomputed from the point clouds.

| Dataset | Cloud size | Median bars/diagram | Breakdown |
|---|---|---|---|
| orbit5k | 500 pts | **936** | H0 500, H1 436 |
| synthetic single-cell | 150 pts | **499** | H0 150, H1 257, H2 94 |

**The structural point matters more than the medians: H0 equals the point
count exactly**, so bar count scales linearly in cloud size. 150 and 500
points are small, and real inputs go up from here rather than down. At batch
scale it compounds — `orbit5k_full` is 5,000 samples, so one `DiagramBatch` is
around **4.7 million bars**.

Format comparison at 1M bars:

| Payload | Size | Load |
|---|---|---|
| `bars.npz` | 20.0 MB | 0.02 s |
| `bars.csv` | 41.1 MB | 1.19 s — **78×** |
| `bars.db` (sqlite3) | 26.5 MB | 0.65 s — **42×** |

The ratios are against the unrounded `.npz` time; 0.02 s is that figure
rounded, so dividing the displayed column reproduces them only approximately.
Scaled to a realistic batch, CSV is ~190 MB and ~5.6 s per `load()` against
~94 MB and ~0.1 s. **Correctness does not discriminate between the three** —
CSV round-trips `float64` exactly through `repr` and `inf` survives as the
literal — so the comparison is decided on cost alone. §10.1 and §12.2 carry
the resolution and the one argument for CSV that survives it.

**Caveat, recorded with the figures rather than below them:** these are two
datasets, both alpha-complex, both low-dimensional, and both the lead's own.
They establish the order of magnitude and the linear-in-point-count scaling,
not a distribution over what users will actually store. D12 is robust to that,
because every direction the bar count moves makes CSV worse.

Reproduction: `rfcs/evidence/bar_counts.py` for the bar-count table. **The
format-benchmark script is not checked in** — its numbers were reported
without it, and the lead has offered it; until it lands, the second table is
the only claim in this appendix that cannot be re-run from this repository.

### A.7 `array-api-compat` — what it costs, and what it still corrects

D18's evidence. Measured on 2026-08-09 with `numpy 2.5.1`,
`array-api-compat 1.15.0`, CPython 3.14.4, best-of-7 — not the 2026-07-29
environment this appendix's preamble names, and `jax` is not installed in it
(A.7.3 is asserted from the library's own dispatch rather than timed).
Reproduction: `rfcs/evidence/array_api_compat_overhead.py`.

**A.7.1 — Wrapper coverage.** Of 26 namespace functions this document names
or `core/` plausibly reaches for, 11 carry a wrapper and 15 are numpy's own
objects by identity, so no wrapper can sit in front of them:

| Group | Functions |
|---|---|
| Wrapped (11) | `sort`, `argsort`, `asarray`, `astype`, `unique_values`, `nonzero`, `zeros`, `empty`, `arange`, `cumulative_sum`, `reshape` |
| Identical (15) | `concat`, `take`, `max`, `min`, `sum`, `any`, `all`, `isnan`, `isinf`, `isdtype`, `equal`, `abs`, `where`, `searchsorted`, `lexsort` |

**A.7.2 — Cost, at equal semantics.** Both sides asked for the same work;
where the standard's default differs from numpy's, the keyword is passed
explicitly, which is what §7 already does. §7's three-pass `canonical()` is
the sort-heaviest operation this document specifies:

| n bars | native | compat | Ratio |
|---|---|---|---|
| 40 | 10.2 µs | 12.0 µs | 1.17× |
| 1,000 | 129 µs | 131 µs | 1.02× |
| 100,000 | 29.7 ms | 29.7 ms | 1.00× |
| 1,000,000 | 389 ms | 389 ms | 1.00× |

Namespace resolution is 275 ns native against 837 ns through the helper. The
wrapped functions cost a flat 200-800 ns each; the overhead is per call, not
per element, which is why it disappears by 1,000 bars. Operators never reach
a namespace at all — `deaths - births` dispatches on the array object — so no
wrapper can sit in front of the elementwise work §3.2 and §4.3 are made of.

**A.7.3 — JAX pays nothing, structurally.** array-api-compat ships no JAX
wrapper. A `jax.Array` routes to `jnp.empty(0).__array_namespace__()`, which
is `jax.numpy` itself for `jax>=0.4.32`, through a dispatch cached on the
class. Every subsequent `xp.foo` is JAX's own function object, so there is no
overhead to measure rather than a small one.

**A.7.4 — What it still corrects on `numpy` 2.5, which is nearly nothing.**
A wrapper existing does not mean the deviation it patches survives. Probed
natively:

| Standard behaviour | `numpy` 2.5.1 |
|---|---|
| `device=` on `zeros` / `asarray` / `arange` | native |
| `unique_values` | native |
| `cumulative_sum(include_initial=True)` | native |
| `reshape(copy=True)` | native |
| `nonzero` rejects 0-d input | native |
| `sort` / `argsort` default to `stable=True` | **deviates** — defaults to `stable=None`, quicksort |

One live correction remains, and §7 already buys it by passing `stable=True`.
**The deviation is worth recording independently of D18**, because it is
invisible from inside this document's own test strategy: it is observable
only on tied keys, and against numpy's default the standard's semantics cost
**9.70× on `sort` and 2.89× on `argsort`** at 1M elements — a price §7 pays
deliberately, since an unstable pass loses the ordering the previous pass
established. §3.3's conformance suite cannot catch a call site that omits the
keyword, because `array_api_strict` is the side that behaves correctly; the
lapse would appear only on the backend every user actually runs. That is
§7's `lexsort` trap one function over, and it wants the same standing test.

**A.7.5 — The constraint on any resolver.** `array_namespace()` on a NumPy
array returns `array_api_compat.numpy`, **not** `numpy` — measured, not
inferred. Resolution MUST therefore go through exactly one function. A
codebase calling `__array_namespace__` directly in one place and the helper
in another gets two namespace objects for one backend, and I7's `is` then
raises on arrays that legitimately share a namespace: D16's loud direction,
fired by our own inconsistency rather than a backend's.

**Caveat.** One machine, one NumPy, one array-api-compat. The ratios are what
transfer; the absolute nanoseconds are not a claim about anyone else's
hardware. A.7.4 probes rather than asserts, so a later NumPy that closes the
sort gap turns that table over without an edit here.

---

## Appendix B — Changelog

- **2026-07-29** — Initial draft.
- **2026-07-30** — Added §4.1 (two-type design vs. dense padded batch).
- **2026-07-30 (2)** — Added §4.2 (`DiagramBatch` CSR storage); rejected a merged CSR type.
- **2026-07-30 (3)** — Resolved the D1/§10 Parquet contradiction; added `DiagramBatch.from_diagrams`; added batch equality to §6.3; opened D7.
- **2026-07-30 (4)** — Added I8 (immutability) and B1–B5 (`offsets` invariants); added the batch-canonicalization rule to §7.
- **2026-07-30 (5)** — Opened D8 (Parquet export) and D9 (MIT/BSD-only vs. dependency-free).
- **2026-07-30 (6)** — Resolved D9: onboarding document retracted "MIT/BSD-only"; corrected §10.1.
- **2026-07-30 (7)** — Removed `numpy` from the "dependency-free" default install (§10.1, D9); added D10.
- **2026-07-30 (8)** — Corrected §5.1: giotto's H0 loss traced to `reduced_homology`, not `infinity_values`.
- **2026-07-30 (9)** — Reworked §5.1/§8: `essential_bars` explicitly derived from `params["reduced_homology"]`; loss scoped to H0.
- **2026-07-30 (10)** — Made `from_giotto`'s `reduced_homology` a required keyword-only argument.
- **2026-07-31 (11)** — Editorial pass: renumbered §3.4 → §3.3; reordered I1–I8; `xp` changed from stored field to derived property; added D11.
- **2026-07-31 (12)** — Applied D8: added `to_parquet()` to §10.3.
- **2026-07-31 (13)** — Condensed §12 and this appendix for readability; moved full narrative to the history document. No normative content changed.
- **2026-08-02 (14)** — Condensed §3, §4.1, §4.2, and §9.1 to their conclusions with pointers; moved full narrative to the history document. No normative content changed.
- **2026-08-02 (15)** — Local tightening, nothing relocated: merged §3.3's two "serialization is NumPy-bound" paragraphs into one (all three normative clauses kept verbatim in substance); pointed §3.3's "conformance is tested" paragraph at §3's first-draft note instead of restating it; merged §9.1's severity-mismatch/travels-badly/consequence paragraphs from three into two; trimmed throat-clearing from Appendix A.4's caption. No normative content changed.
- **2026-08-02 (16)** — Trimmed D6, D8 and D9 to status plus still-live caveat or option; full reasoning moved to the history document. No normative content changed.
- **2026-08-02 (17)** — Split §12 into §12.1 (needs the lead) and §12.2 (settled). D-numbering unchanged; only grouping and row order moved. Resolved rows were kept rather than deleted, since §1 makes this RFC the record of what was decided and why, and several cells cross-reference each other by number. No normative content changed.
- **2026-08-02 (18)** — Moved the top-of-document "Note on this revision" callout into the history document. No normative content changed.
- **2026-08-02 (19)** — Removed references to history unrelated to decision-making.
- **2026-08-02 (20)** — Follow-up to entry 19, which undersold its own scope: that pass also dropped four items from §3, I2 and §2 that were not history references, and swapped §3.3's `lexsort` item for an unrelated `argsort` statement. §3.3's "Three limits" is now genuinely two — the misfit item is removed rather than replaced, on the view that the `hasattr(xp, "lexsort")` trap needs a test, not prose — and §7's one-time `np.lexsort` verification is upgraded to a standing CI regression-test requirement naming that trap. **The one entry in this run that is not normative-content-neutral**, since requiring a test to stay in CI is new normative text rather than a relocation.
- **2026-08-02 (21)** — Opened D12 (`.npz` vs. stdlib `csv`/`tsv` or `sqlite3`) in §12.1, and scoped §10.1's "follows from requirement 5" paragraph so it reads as tested against binary alternatives only. No normative content changed.
- **2026-08-03 (22)** — Gave §10.1 requirement 4 the rationale it had never stated: artifact-level reproducibility, not round-tripping (requirement 1) and not `content_hash` (§8.1). Rewrote §7's sentence that had implied file-level determinism is what makes a content hash meaningful; canonical ordering does that on its own. No normative content changed.
- **2026-08-03 (23)** — Added §3.2's accessor cross-reference list, naming `d.finitize()` as a deliberate exclusion rather than an oversight. No normative content changed.
- **2026-08-03 (24)** — Added §4.3, the `DiagramBatch` counterpart to §3.2, flagging two genuine gaps rather than papering over them. No normative content changed.
- **2026-08-03 (25)** — Added `essential`, `persistence`, `bar_counts` and `xp` to §4.3, and `b1.same_provenance(b2)` to §8; sharpened §4.1's "no duplicated logic" claim to the form §4.3 now carries. No normative content changed.
- **2026-08-03 (26)** — Design-review pass. Normative: `from_giotto` always returns a `DiagramBatch` (§11); `finitize(at="drop")` records `"finitized_dropped"` plus a count (§5, §8); §11.2 accepts a frozen fixture as real backend output; §10.2 separates the settled container format from the provisional `bars.npz` payload; §9.2 states `from_giotto`'s shim status. §12: removed D6, D9, D10 and D11 as dependency-and-licensing policy rather than interchange questions; added D13. D-numbers are not renumbered to close the gap, being stable identifiers rather than a dense sequence.
- **2026-08-03 (27)** — Added a Rationale column to §4.2's B-invariant table, matching §3.1's. No normative content changed.
- **2026-08-03 (28)** — Removed all references to specific papers; this repository is meant to be universal.
- **2026-08-04 (29)** — Resolved D7 on the lead's guidance: added §8.2, `DiagramBatch.content_hash`.
- **2026-08-04 (30)** — Trimmed changelog restatements. No normative content changed.
- **2026-08-04 (31)** — Added I9 (`dims`, `births`, `deaths` each rank-1) to §3.1, closing a gap where a same-length-but-wrong-rank array passed I1 unnoticed.
- **2026-08-05 (32)** — Normative: I2 and §6.1 now agree that dtypes are checked by equality against `xp.float64`/`xp.int32`, `xp.isdtype("real floating")` being true of the `float32` D3 rejects; added B6 and B7; `from_diagrams` checks namespace agreement across inputs and takes `xp=` for the empty case; `finitize` returns a diagram with no essential bars unchanged, validates the whole `at` argument rather than the two mode names, and requires `at=<float>` to be finite; added reserved key `essential_bars_source` (§5, §5.1, §8, §11) and the rule that qualifier keys stay consistent with `essential_bars` (§8); §8.1 specifies the hashed message and requires `-0.0` normalisation; §3.3 states that every `finitize` mode is eager-only and that shape-preserving is not traceable; §6.3's "bit-identical" corrected to "without tolerance". Opened D14.
- **2026-08-05 (33)** — Normative: `DiagramBatch.__getitem__` is eager-only and so is everything routed through it, `b.canonical()` included (§3.3, §4.3) — entry 32's shape-preserving-is-not-traceable conflation at a second site; `len(...)` of an array is shorthand for `shape[0]` and MUST be implemented as such (§4.2). Opened D15.
- **2026-08-05 (34)** — Readability pass, the first to treat this changelog as the main offender: entries 32 and 33 had each grown to roughly two thousand words in one bullet. Entries 16-33 are now a single bullet each, the full text living in the history document, which had never carried 23-33 at all. (Entry 35 corrects this line's original "one line each", which entries 26, 32 and 33 do not meet and did not meet when it was written.) Relocated with a pointer left behind: §5's unreachable "keep the smaller recorded value" alternative, §6.1's and §6.3's accounts of superseded revisions, and §12's restatement of why D6/D9/D10/D11 were removed. Single-sourced: §4.3 → §3.3 (eager-only), §8 → §12.1 (D15's options), §4.1 → §4.3 ("no duplicated logic"). Reflowed §10.1's argument paragraphs. No requirement changed; the body's one lost keyword is §8's *quotation* of a §7 MUST, dropped with the duplicated paragraph around it.
- **2026-08-05 (35)** — Review of `diagrams/core.py` against this document, run in the direction entry 33 established. Normative: §10.1 requirement 1 gains a second clause requiring metadata to round-trip, since §8 excludes `meta` from `==` and §5 nonetheless cites requirement 1 for exactly that (§11.2 updated to test both clauses); §8 requires `params` and `provenance` values to be JSON-representable, without which the type admits diagrams §10.2 cannot save; §8 makes `DiagramMeta` enforce the `essential_bars` qualifier rules at construction rather than stating them as writer obligations only; §6.3 makes the cross-namespace `ValueError` normative at both levels, previously a `core.py`-only behaviour recorded in entry 33 as non-normative; §11.2 requires `content_hash` to be tested against §8.1's byte layout on **both** an implementation's buffer and per-element paths, the buffer path being the one `array_api_strict` cannot reach and every NumPy diagram takes. Corrections: §3.3's "Two limits" is three, entries 32 and 33 having each added one — entry 20's own fix, recurred; §8's reserved-key table was broken markdown, its enum split across four unnamed columns; Appendix A.1's bare `TODO` now states what the table does not measure and why the row is blocked. Opened **D16** (the `is` in I7/B5 assumes a namespace identity the standard does not require); §12's count moves to twelve, five open. Entry 34's "one line each" corrected to "a single bullet each" here and in the history document.
- **2026-08-05 (36)** — Linted §12.1 and §12.2, and reordered D15 correctly before D16. No normative content changed.
- **2026-08-05 (37)** — Opened **D17**: §8's `coeff_field` comment asserts an obligation ("must be recorded") that no clause states and that the paragraph below it contradicts. §12's count moves to thirteen, six open. No normative content changed — §8 is untouched deliberately, the comment being the subject of the decision rather than a defect to fix ahead of it.
- **2026-08-05 (38)** — Acts on what entry 37 left open. The keyword line now cites BCP 14 (RFC 2119 **and** RFC 8174) and binds the keywords to all-capital use only, and records the other six BCP 14 keywords as deliberately unused, "required" and "optional" being ordinary Python vocabulary throughout this document. **Not normative-content-neutral**: every lowercase "must", "should" and "may" in the body becomes formally non-normative rather than conventionally so. Audited before the change rather than after — §3.3's two, §4.2's B4 rationale cell and §11's `TypeError` consequence all either restate an uppercase clause or are descriptive; §8's `coeff_field` comment is the sole lowercase obligation without an uppercase counterpart, and is D17.
- **2026-08-05 (39)** — Normative, and supersedes entry 38's decision to leave §3.1's I8 note as written. Its lowercase "should be enforced" is promoted, making it the only clause in the document to carry the weaker of the two obligation keywords: `@dataclass(frozen=True)`, as §8 already does for `DiagramMeta`, becomes the preferred enforcement of the no-mutation rule rather than an unmarked suggestion. The alternative the same sentence names — the array API standard's read-only view support — gains a full obligation to document itself as an equivalent guarantee, rather than sitting in prose that entry 38's caps-only rule had just drained of force. Preferred mechanism on the weaker keyword, the requirement attaching to the sanctioned deviation on the stronger one.
- **2026-08-05 (40)** — Linted all twelve tables into one compact form, completing what entry 36 began on §12.1 and §12.2. Cell text is untouched. No normative content changed.
- **2026-08-06 (41)** — Review pass on PR #10. **Normative in five places.** Resolved **D14**: §6.3's `allclose` MUST be a matching over the multiset — a bijection sharing `dim` exactly and agreeing on both coordinates within tolerance — with a symmetric tolerance `atol + rtol * max(abs(a), abs(b))` diverging from `numpy.allclose` and documented as such, no new dependency, no refactor into bottleneck distance, and documented as not an equivalence relation; §11.2 gains the test that separates it from the rejected sorted-pairwise form. Resolved **D13**: multiparameter persistence is an explicit non-goal, stated in §1 as normative scope; §3.2's `d.h0`/`d.h1` note drops its multiparameter clause. Reinstated **D6** in §12.2 as superseded rather than deleted: `numpy` stays out of the required closure but moves to `akriti[io]` at `>=2.0`, so §10.1 requirement 2 now requires a lazily-imported library to be a declared extra with a floor, and §3.3 requires the lazy import to check the version and name the extra on both failure paths. Completed entry 38's keyword sweep: §3's array-API rule, §11's `TypeError`, §10.1's five requirements and requirement 4's mechanism paragraph, §8's `essential_bars_source` writer prohibition and §9.2's clean-room note all take uppercase keywords. Entry 38 found one orphan of six; the sweep was run three times by three readers and did not converge on the first pass. **D17** probed, not resolved: new Appendix A.5 measures that no backend returns the coefficient field it computed with, which removes one of the three options and leaves a judgment the lead still owes. §12's count moves to fourteen, four open.
- **2026-08-07 (42)** — Second review pass on PR #10, acting on the lead's three replies. **Resolves D12, D15 and D16; §12.1 is down to D17 alone.** **D12** — `bars.npz` stays, and §10.2's payload stops being provisional. Resolved on the bar-count figure the row said it turned on, now Appendix A.6: H0 equals the input point count exactly, so bars scale linearly in cloud size and a 5,000-sample batch is ~4.7M bars; at 1M bars CSV costs ~2.1× the bytes and ~78× the load, sqlite3 ~1.3× and ~42×. The reason is *not* that CSV loses on inspectability — it wins there — but that requirement 5 is already satisfied by `meta.json` and §10.3's `to_csv()`, and does not need satisfying a third time. The one surviving argument for CSV (a stdlib payload would drop the `[io]` extra altogether) is recorded in §12.2 as the condition to reopen against, rather than waved away; sqlite3 is closed out. **D15** — `provenance["order"]` is removed. Every other reserved key records a fact that vanishes if unrecorded; canonical order is recoverable from the arrays in one pass, so `order` is a cached answer to an always-computable question and one that goes stale. §8 keeps a short note naming the order fact that is *not* derivable — whether the backend's own output was already canonical — as a different key to build if it is ever wanted, so this is not reopened. **D16** — namespace comparison is identity, stated in §3.3 as a constraint on supported backends rather than a property of the standard, and **verified in CI** rather than assumed. Decided on the direction of the failure: `is` fails loudly on arrays that legitimately share a namespace, while every available surrogate can match across genuinely different ones, which is the silent direction I7 exists to prevent. I7, B5 and §4.2's check are unchanged; only their rationale cells move from citing an open question to citing a requirement. **New §9.3**, at the lead's request and independent of D17: GUDHI defaults to ℤ/11 and Ripser to ℤ/2, neither returns the field it used (A.5), so two diagrams of the same cloud from our two primary backends are not diagrams of the same thing — and §6.3's cross-backend `allclose` is where that bites, returning `True` on torsion-free data. §9's preamble now counts three hazards, the third being a disagreement between correct delegates rather than a defect in one. **One addition beyond what was asked**, flagged for the lead to strike: §9.3 puts a MUST on §11.2's cross-backend test to pin the coefficient field on both sides. It constrains a test this RFC already requires rather than the adapter signatures D17 is about, and pinning is a call parameter on both backends, so it costs nothing — but it is new normative text the lead did not request. **New Appendix A.6** carries D12's figures, the lead's own measurements rather than a run of ours, with his caveat that two alpha-complex datasets fix an order of magnitude and not a distribution. `rfcs/evidence/bar_counts.py` is his script, committed verbatim in behaviour; the format-benchmark script was reported without its source and is **not** checked in.
- **2026-08-09 (43)** — Opened **D18** and added **Appendix A.7**, its evidence; §12's count moves to fifteen, two open. No other content changed. `torch.Tensor` does not implement `__array_namespace__` — array-api-compat's documentation says so, PyTorch's tracker holds the attribute back until near-full conformance, and it is absent from the torch 2.13 reference — so §3's definition of `Array` excludes torch tensors and no diagram can currently be torch-backed. D18 asks whether `array_api_compat.array_namespace` goes in front of every backend or torch alone. **This does not reopen D16**, whose identity requirement stands; what is wrong is its reach, since its CI test names a backend on which it raises `AttributeError` before reaching the identity question, and four sites illustrate a second namespace with torch counterfactually. Those four and the test's scope are left as they stand on D17's precedent, editing them in either direction being the answer. `rfcs/evidence/array_api_compat_overhead.py` carries the measurements, which are committed because two arguments that look decisive are not: JAX pays nothing (array-api-compat ships no JAX wrapper) and NumPy pays one Python frame on the 11 of 26 wrapped functions, §7's `canonical()` measuring 1.17x at 40 bars and 1.00x from 100k up; and on `numpy` 2.5 the wrappers are largely vestigial, leaving the `sort`/`argsort` stable default as the one live correction, which §7 already buys explicitly. §10.1 requirement 2 is what decides it. **Appendix A.7** carries the figures, on A.6's precedent that a decision log citing numbers it does not hold is one nobody can check later. Recorded in A.7.4 and the history document, not raised as a row: numpy's `sort`/`argsort` default to quicksort where the standard specifies stable, `array_api_strict` behaves correctly and so cannot catch a call site that omits the keyword, and §7 is therefore correct by discipline rather than construction — entry 20's `lexsort` precedent applies.
