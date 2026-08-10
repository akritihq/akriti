# RFC-0001 — Persistence Diagram Interchange

| | |
|---|---|
| **Status** | Draft — not yet open for public comment |
| **Version** | 0.1.0 — `major.minor.patch`; what §10.2 writes as `spec_version` into every file |
| **Author** | Sushovan Majhi |
| **Edited By** | A. D. Silberman |
| **Created** | 2026-07-29 |
| **Last Edited** | 2026-08-09 |
| **Target** | M0 (2026-08-01) drafted · published for comment 2026-08-23, ahead of M1 |
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

**Non-goal: extended persistence**, on the same terms and for a nearer reason.
GUDHI ships `extended_persistence()`, so this is a boundary a `from_gudhi`
caller reaches with one documented call rather than a hypothetical one. Its
output is not this type, and it is worth being exact about why, because the
reason is not uniform across what that call returns. Extended persistence has
no essential classes — pairing them off is the point of it — and returns
**four** sub-diagrams: ordinary, relative, extended+ and extended−. The
relative and extended− bars carry `death < birth` by construction rather than
as a backend defect, which I6 (§3.1) forbids exactly. The ordinary and
extended+ bars satisfy every invariant in this document and are still not
interchangeable with an ordinary diagram's, because nothing in this type
records which of the four families a bar came from.

Support therefore requires a fourth per-bar field carrying that family, and
every invariant, accessor and adapter in this document would have to say what
it means for each. A parallel type again, not a column. §11 states what
`from_gudhi` can and cannot do about it.

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
        return namespace_of(self.dims)   # §3.3's single resolution rule
```

`Array` is **any object carrying an array-API namespace** — the Python array
API standard — not `np.ndarray`. §3.3 states how that namespace is obtained:
from the object's own `__array_namespace__` where it has one, and through
`array_api_compat` where a backend conforms in substance but has not yet
exposed the method.

This is a hard requirement, not a preference: `core/` MUST be written against
the array API rather than hard-coded NumPy, and `PersistenceDiagram` is the
input to every function in `core/`. §3.3 states what this does and does not
promise. This is a strictly stronger obligation than the dependency closure
§10.1 requirement 2 sets, not a restatement of it: keeping third-party
libraries off a caller's path forbids *calling* NumPy, while writing against
the array API additionally rules out NumPy-shaped idioms applied to an array
the caller handed in — which is what §7's `lexsort` prohibition and §4.2's
`len()`-versus-`shape[0]` rule are both instances of.

`xp` is a derived **property**, not a fourth stored field: I7 already
requires `dims`, `births`, and `deaths` to share one namespace, so deriving
`xp` from `dims` makes disagreement structurally impossible rather than
merely prohibited, with nothing extra to keep in sync at every construction
site, including the views §4.2 returns.

Three parallel arrays, one row per bar, all of length `n`. This is the right
representation, for three reasons:

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
| I7 | all three arrays share one namespace — `namespace_of(births) is namespace_of(deaths) is namespace_of(dims)` | §3.3; resolved by the one rule, never by calling `__array_namespace__` here (D18); the `is` is identity by requirement (D16), verified in CI |
| I8 | `PersistenceDiagram` is immutable after construction — no method may write to `dims`, `births`, or `deaths` in place, and none may rebind them; public construction copies rather than aliasing the caller's arrays | §4.2; the three obligations and the one that stays a caller contract are below |
| I9 | `dims`, `births`, `deaths` are each rank-1 (`ndim == 1`) | §3, shape `(n,)` |

**I6 is checked exactly, not within tolerance.** A backend that returns
`death < birth` has a bug, and we surface it rather than absorb it. Observed
floating-point violations are a real occurrence at the 1e-16 level in some
filtration code; the adapter (not the core type) is the correct place to clamp,
and it MUST warn when it does. Extended persistence is the one setting where
`death < birth` is not a bug — its relative and extended− bars carry it by
construction — and it is out of scope (§1).

**I8 exists because §4.2 already assumes it, and it takes three separate
obligations to deliver.** `DiagramBatch.__getitem__` returns a
`PersistenceDiagram` whose arrays are views into the batch's shared buffer, not
copies, and that is only safe if nothing can write through one view and corrupt
a sibling diagram or the batch itself.

- **No method of either type may mutate.** Every method that looks like a
  mutation (`finitize`, anything in §3.2) MUST construct and return a new
  `PersistenceDiagram` rather than modify `self`. `DiagramBatch` is under the
  same rule, stated as B8 (§4.2): its buffers *are* every member diagram's
  arrays, so a batch-level in-place write is a write to N diagrams at once.
- **`@dataclass(frozen=True)` is the mechanism, and it is not sufficient by
  itself.** Both types SHOULD be frozen, as `DiagramMeta` already is (§8).
  What frozen buys is that `d.births = other` raises; what it does not touch
  is `d.births[0] = 5.0`, which reaches the array through an attribute the
  dataclass never sees assigned. **The array API standard supplies nothing
  that closes the gap, having considered it and declined.** It specifies no
  read-only array, no writeability flag and no immutable view; it lists
  `__setitem__` among the array object's methods; and its
  copies-views-and-mutation topic records read-only views as a rejected option,
  "hard to implement" and a backward-compatibility problem "for current strided
  array libraries", offering in their place the advice to avoid mutating
  anything that might be a view.
- **Public construction MUST NOT alias an array the caller retains.** This is
  the hole the other two leave open, and the one I8 was added for: a caller who
  passes an array to `from_array` (§11) or to the constructor, keeps their own
  reference, and writes through it afterwards has mutated a constructed
  diagram, without any method of ours having run. Every **public** construction
  path — the `PersistenceDiagram` constructor, every `from_*` adapter, and
  `DiagramBatch.from_diagrams` — MUST therefore copy the arrays it is given
  rather than store them. The cost is one copy at a boundary that is already
  copying in the common case (§6.1 has adapters upcast `float32`, and
  `from_diagrams` concatenates), and it is what makes the immutability the rest
  of this section reasons from actually true of a constructed object.
  **`DiagramBatch` MUST copy the `metas` sequence on the same rule**, B1 (§4.2)
  being stated over its length: a caller who appends to a list they passed in
  would otherwise break an invariant after the construction that enforced it.
  The elements need no copying — `DiagramMeta` is frozen (§8) — only the
  sequence holding them.

**`DiagramBatch.__getitem__` is the deliberate exception to the third rule and
MUST NOT be implemented through the public path.** It aliases on purpose (§4.2)
and is safe for precisely the reason a caller-supplied array is not: both ends
of the alias are objects this document forbids anyone to write to, and neither
is reachable by a caller who held a reference before the batch existed.

**What remains unenforceable MUST be documented rather than implied.** After
all three rules, `d.births[0] = 5.0` still runs on every backend, because no
array library in scope offers a way to stop it. I8 is a contract on callers at
that last step, not a guarantee the type can make, and the class docstring MUST
say so — a reader who has been told the type is immutable and finds that a
subscript assignment works will conclude the immutability is nominal
everywhere, including at the two places above where it is real.

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
delivers would be worse than promising nothing. What follows is what it does
and does not buy; the limits stated as measurements were measured against
`array_api_strict` 2.6.1, the conformance reference:

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
make `numpy` a dependency of the package. **What constrains `core.py` and
`adapters.py` is §10.1 requirement 2**, which sets the closure over what a
caller can reach rather than over which files may import what: nothing
third-party on any path reachable without a backend the caller installed
themselves, and every exception lazy, function-scoped, declared as an install
extra, and failing actionably on both absence and an unsatisfied floor. §3.3's
namespace resolution rule, `save`/`load`'s `numpy` and §10.3's `to_parquet`
are the three that currently meet it. `numpy` is used **only** inside
`save`/`load`, imported lazily there rather than at module scope, so
everything except serialization works with zero third-party dependencies.

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
NumPy. A diagram built from JAX arrays stays JAX-backed. What adapters
convert is *dtype* (§6.1), not namespace.

**Namespace resolution goes through exactly one function, and its answer
depends on the input and never on the environment.** `d.xp`, I7, B5 and §4.2's
`from_diagrams` check are all defined over what that one function returns:

```python
def namespace_of(x):
    ns = getattr(x, "__array_namespace__", None)
    if ns is not None:
        return ns()                              # numpy, jax, array_api_strict
    import array_api_compat                      # lazy; akriti[torch]
    return array_api_compat.array_namespace(x)
```

The native method MUST be preferred wherever it exists, and the fallback MUST
be reached only where it does not. **This is not a lower bar for backends that
skip the method.** `array_api_compat` supplies a conforming namespace for a
backend that conforms in substance without having declared it; a backend that
does not conform is not rescued by being wrapped. Exactly one backend is in
that state today — `torch.Tensor` implements no `__array_namespace__`, PyTorch
withholding it deliberately as the attribute that declares conformance
(gh-58743) — and without the fallback `Array` excludes torch tensors outright,
so no diagram could be torch-backed at all (D18). The arrangement expires on
its own: when the declaration lands, the first branch takes torch and the
second stops being reached.

`array-api-compat` MUST therefore be declared in the `akriti[torch]` extra
with a version floor, on the terms §10.1 requirement 2 sets. The import MUST
be lazy and function-scoped. It fires only for a caller who has handed in a
tensor from a backend they installed themselves, so it is unreachable on the
default install and cannot widen that closure.

**Preferring the wrapper whenever it happens to be installed is excluded**,
and that exclusion is what makes the rule single-valued. A NumPy array
resolved through the helper gives `array_api_compat.numpy`, not `numpy`
(A.7.5), so a codebase calling both would hold two namespace objects for one
backend and the identity comparison below would raise on arrays that
legitimately share a namespace — D16's loud failure, fired by our own
inconsistency rather than a backend's. Under the rule as written the same
array resolves the same way whether or not `array-api-compat` is installed.

**The resolver serves every namespace call, and reaches nothing on the array
object. The second half is the one that matters here.** For a resolved
backend, `d.xp` is the wrapper for the diagram's whole life and every `xp.*`
call goes through it — this is a binding, not a one-time check. But
`array_api_compat` returns the backend's own object rather than a wrapped one,
and its conformance record against the standard's own test suite
(`torch-xfails.txt`, under the heading *"We cannot wrap the tensor object"*)
lists `__eq__`, `__sub__`, `__add__` and `__truediv__` as failing for torch,
along with `__getitem__`, `__setitem__` and masked `__getitem__` under
*"Indexing does not support negative step"* and *"Masking doesn't support 0
dimensions in the mask"*.

**Those are the operations §3.2 and §4.3 are made of**, not an obscure corner:
`deaths == xp.inf` is `d.essential`, `dims == k` is `d.dim(k)`,
`deaths - births` is `d.persistence`, `offsets[1:] - offsets[:-1]` is
`b.bar_counts`, and `deaths[~xp.isinf(deaths)]` is `d.finite`. A.7.2 recorded
that operators never reach a namespace at all, as an observation about cost;
on a backend that fails conformance on those operators it is one about
correctness.

**This document does not assert that the divergence misses it.** The plausible
argument is that I2 and §6.1 fix every operand's dtype, so no mixed-dtype
arithmetic occurs here and the standard's promotion rules have nothing to
disagree about — but the xfails are structural, arising because the object is
unwrapped rather than because of any particular dtype pair, and this document
has measured none of it (A.7). An unverified argument that a known
non-conformance happens to miss us is exactly the reasoning §9 exists to
distrust. **A test MUST therefore assert that `essential`, `persistence`,
`bar_counts`, `dim(k)` and `finite` return the same values under two
namespaces for the same bars**, and until it runs against torch, a
torch-backed diagram is namespace-correct and not established as
object-correct.

Two further torch xfails do not reach this document, and are named so the
list above is not read as exhaustive: `unique_all` is unimplementable on
torch's `unique` and is unused here, while `unique_values` and `unique_counts`
are xfailed for complex dtypes only, which I2 and §6.1 exclude — `d.dimensions`
uses `unique_values` on `int32` and is unaffected.

**Namespaces are compared by identity, and that is a constraint this document
places on supported backends rather than a property the standard provides.**
I7, B5 and §4.2's `from_diagrams` check all compare with `is`. The standard
requires `__array_namespace__()` to return "an object representing the
namespace"; it does not require the same object on every call, and it takes an
`api_version` argument that a backend could legitimately answer with distinct
wrapper objects. NumPy and `array_api_strict` return the module itself, so
identity holds there and the assumption is invisible. **Akriti requires the
resolution rule above to return a consistent object for a given backend** —
which for a backend answering natively is a requirement on
`__array_namespace__` itself — and states that here rather than claiming the
standard guarantees it.

D16 settled this on the **direction of the failure**, not on the theory. `is`
fails by raising `ValueError` (§6.3) on arrays that legitimately share a
namespace: conservative, loud, immediately diagnosable, and impossible to
mistake for a correct answer. Every surrogate the standard leaves available —
comparing `__name__`, a sentinel dtype, anything else — is a weaker test that
can *match across genuinely different namespaces*, admitting a JAX/NumPy mix
into one diagram, which is what I7 exists to prevent. That is the silent
direction. Given a conservative check carrying a documented constraint against
a clever one that fails open, this document takes the first.

**The constraint MUST be verified in CI, not assumed.** A test MUST assert,
for every supported backend that implements `__array_namespace__` natively,
that the method called on two separate arrays of that backend returns the same
object. This is "conformance is tested, not intended" below, applied to a
promise the standard does not make, and it follows §7's `lexsort` precedent:
a trap a reader cannot see becomes a standing regression test rather than
prose. A backend that ever returns a fresh wrapper per call then fails that
test and reopens D16 as a real decision, instead of breaking silently in
someone's pipeline.

The test is scoped to backends that implement the method because a backend
resolved through the fallback has no method to call, and its namespace is a
module — identical across calls because Python caches imports, not because the
backend promises anything. **What MUST be pinned for those is which branch of
the resolution rule they take.** A test MUST assert that a `torch.Tensor` does
not implement `__array_namespace__` and therefore resolves through
`array_api_compat`, marked `@pytest.mark.backend` on the `akriti[torch]`
extra. gh-58743 is open and the attribute is being withheld until near-full
conformance, which means it lands eventually; on the release that adds it, a
torch tensor stops resolving through the wrapper and `d.xp` changes from
`array_api_compat.torch` to `torch`, so two diagrams built under adjacent
torch versions fail the identity comparison while being the same kind of
thing. That MUST break the build rather than reach a user, and when it does,
torch enters the identity test above on the same terms as every other backend.
This is §9.3's treatment of the coefficient-field defaults applied to a fact
about a backend that is expected to change.

**Conformance is tested, not intended.** CI runs the diagram test suite
against `array_api_strict`, which rejects any NumPy-only call. A requirement
merely written down decays within weeks, and this document is its own
evidence: the first draft specified `np.ndarray` for the three arrays despite
the array-API rule §3 states, and nothing caught it until a reader checked the
type against that rule by hand.

---

## 4. Batch semantics

Every numerical function in `core/` and `castle/` takes a **leading batch
dimension** rather than expecting a Python loop over diagrams. That is a
project-wide commitment made for its own reasons — a looping API is very hard
to withdraw once published — and it is why this document specifies a batch
type at all instead of leaving callers to assemble one. For diagrams, the
batch container is:

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
`len(...)` of an array — I1, B3, B1's `offsets`, and §4.3's batch total — it is
shorthand for `shape[0]` and MUST be implemented as such. `len(batch)`,
`len(diagrams)` and B1's `len(metas)` are ordinary Python over ordinary
sequences and unaffected; B1 is the one invariant spanning both, and each side
takes the spelling its own operand requires.

`__getitem__` MUST return a **view**: a `PersistenceDiagram` whose arrays are
slices into the batch's own buffers, not a copy. This is safe on three
conditions (§3.1): neither type has a method that writes in
place (I8, B8), a batch's buffers were copied at construction rather than
aliased from a caller's arrays (I8's third rule, which `from_diagrams` gets for
free from `concat`), and the residual hole — a caller subscript-assigning into
`b.dims` or `b[i].births`, which no array library in scope can prevent — is
documented as a contract rather than presented as a guarantee.

**`offsets` has its own invariants, and `core.py` MUST enforce them at
`DiagramBatch` construction, the same way §3.1 enforces I1 through I9 for a
single diagram:**

| # | Invariant | Rationale |
|---|---|---|
| B1 | `len(offsets) == len(metas) + 1` | fencepost: `n` diagrams need `n+1` boundaries. Stated against `metas`, not `len(batch)`, for the reason below |
| B2 | `offsets[0] == 0` | buffer has no unowned leading bars |
| B3 | `offsets[-1] == total_bars` (i.e. `len(dims)`) | buffer has no unowned trailing bars; bounds the last diagram's slice |
| B4 | `offsets` is non-decreasing | row ranges must not overlap or invert |
| B5 | `namespace_of(offsets)` matches `dims`, `births`, `deaths` | §3.3; resolved by the one rule (D18); "matches" is identity, required of supported backends (D16) |
| B6 | `offsets` is rank-1 (`ndim == 1`) | I9's rationale, applied to `offsets`: a wrong-rank array of the right length passes B1 unnoticed |
| B7 | `offsets.dtype` is the namespace's own `int64` | the class body above already says `int64`; stated as an invariant so it is enforced and citable like the rest |
| B8 | `DiagramBatch` is immutable after construction — no method may write to `dims`, `births`, `deaths` or `offsets` in place, none may rebind them, and `metas` is not mutated | I8 one field-set over; these buffers *are* the member diagrams' arrays, so one batch-level write is N diagram-level writes |

B6 and B7 were implicit before: the class body declares `offsets` `int64` of
shape `(len(batch)+1,)`, and B1 through B5 then quietly assume both. B1's
`len(offsets)` reads `shape[0]`, which is happy to answer for a rank-2 array —
exactly the gap I9 was added to close for `dims`/`births`/`deaths`, and the
same gap, one field over. Neither is a new requirement, only a stated one.

**B1 is stated against `metas` because the `len(batch)` form was a check that
could not fail.** `__len__` is defined in the class body above as
`offsets.shape[0] - 1`, so `len(offsets) == len(batch) + 1` expands to
`offsets.shape[0] == offsets.shape[0]`: a fencepost invariant with no
fencepost in it. `metas` is what the fencepost actually binds. It is the one
stored field no other invariant reaches — B2 through B7 are all about
`offsets` and the buffers — so a three-diagram batch carrying two
`DiagramMeta`s satisfies every one of them, and `b[2].meta` then raises
`IndexError` from somewhere deep in `__getitem__`, or, on any implementation
that zips or cycles, `b[1].meta` quietly returns the wrong diagram's
provenance. That is §9's silent-wrongness category arriving by our own hand, in
the field §8 exists to make auditable. Where this document writes `len(batch)`
elsewhere — §4.3's `bar_counts` shape, §8.2's hash length, §10.2's `offsets`
length — it is naming a derived quantity, not restating B1.

**B8 is what this section has been assuming throughout**: every argument here
for why a zero-copy `__getitem__` is safe is an argument about what cannot be
written to, and it is `DiagramBatch` that is on the other end of those views.
§3.1's I8 discussion carries the enforcement mechanisms for both types,
including which part of the guarantee is a caller contract rather than
something either type can enforce.

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
nothing across diagrams, so a sequence mixing a JAX-backed diagram with a
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
field and not worth a second name. `xp` is §3.3's resolution rule applied to
`self.dims`, the derive-don't-store reasoning §3 gives for
`PersistenceDiagram.xp`,
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
cannot use — an unrecognised mode name, the non-finite float the next paragraph
rules out, a substituted death below an essential bar's birth that the paragraph
after it rules out, or `at="max_finite_death"` on a diagram that has no finite
death to take a maximum over. The split is by what is wrong with the call, not
by which check happened to catch it.

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
the argument.

**The substituted death MUST NOT fall below the birth of any bar it replaces**,
and `finitize` MUST raise `ValueError` when it does. This binds **both**
substituting modes, not only the explicit float. I6 is checked exactly (§3.1),
so a substituted death less than some essential bar's birth produces
`death < birth` and no valid diagram exists to return: `finitize` either raises
or is unimplementable. The check is
`substituted >= xp.max(births[essential])`, over the essential bars alone, since
every other bar keeps the death it already had and already satisfies I6. It is
vacuous on a diagram with no essential bars, which the return-unchanged rule
above has already returned before this point is reached.

The two modes reach it differently and the error MUST say which. For
`at=<float>` this is not a hazard a caller can be expected to see coming: `at`
is naturally read as "the value I want infinity replaced by", and on a
filtration whose births are all positive, a plausible `at=0.0` is a
`ValueError` rather than the drop-to-the-diagonal it looks like. For
`at="max_finite_death"` the offending value is one this method computed itself,
so the message MUST name the computed maximum as well as the birth it fell
below — a caller who passed no number has none to correlate a complaint
against. That mode reaches the condition whenever the longest-lived finite bar
dies before the last essential bar is born: a Rips filtration truncated at
`max_edge_length` with an H1 cycle still open at the cutoff is the ordinary
case, the finite deaths all sitting below the cycle's birth. Leaving this to
I6 at construction was the alternative and it is the one §5 already rejects for
`at=nan` — an error naming death times rather than the argument the caller
actually got wrong.

**`at="max_finite_death"` MUST also raise `ValueError` on a diagram whose bars
are all essential**, with an error naming the mode and the absence of any finite
death. This is the separate case where there is no maximum to take rather than
one that lands too low. I5 guarantees no `NaN` and no `-inf`; it does not
guarantee that a finite death exists at all. `d.dim(k)` on a degree whose only
class is essential reaches this directly, and so does any H0 diagram of a
filtration whose complex is connected from its first value — a cubical or
lower-star filtration on connected data, not a constructed edge case.

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

The giotto behaviour is worth stating precisely because the compat shim (§9.2)
has to decide what to do about it. On a 40-point noisy circle, GUDHI and
Ripser both return 40 H0 bars, exactly one essential.
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

`inf == inf` compares equal at both levels. `NaN` cannot occur (I4, I5). `==`
gets this from IEEE equality directly; `allclose` gets it from an explicit
clause below, its tolerance formula being unable to deliver it.

**Comparing two diagrams backed by different array namespaces MUST raise
`ValueError`, at both levels and in both methods.** I7 constrains the three
arrays *within* one diagram and says nothing across two, so a NumPy diagram
and a JAX diagram of the same bars are each valid and still not comparable:
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
unavailable — requirement 2 names comparing a diagram among the paths that
MUST NOT reach a third-party library, so this is a path already closed rather
than a trade to weigh. An augmenting-path
matching is sufficient at the sizes this method serves; Hopcroft–Karp's
asymptotics are not needed. Edge construction is $O(n^{2})$, and `allclose` is a
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

**`+inf` deaths are matched exactly and MUST NOT be put through that formula.**
A matched pair agrees on `death` iff both deaths are `+inf`, or both are finite
and satisfy the tolerance; exactly one `+inf` never agrees, at any `atol` or
`rtol`. Without this clause the formula contradicts §6.3's own "`inf == inf`
compares equal at both levels" three paragraphs up, and does so on the common
case rather than a corner: `|inf - inf|` evaluates to `NaN`, every comparison
against `NaN` is `False`, and a diagram carrying an essential bar is therefore
not `allclose` to itself. `births` need no counterpart clause — I4 makes every
birth finite — and `NaN` cannot occur on either coordinate (I4, I5), so this is
the whole of the non-finite handling `allclose` requires. `==` needs no clause
at all: IEEE equality gives `inf == inf` directly, which is why the
contradiction was reachable only on the approximate side.

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

Metadata is not decoration. A diagram records the outcome of a computation and
not the computation: two diagrams with identical bars can come from different
filtrations, different backends, different scale parameters and different
coefficient fields (§9.3), and nothing in the bars distinguishes them. This
document also commits twice over to a diagram being checkable after the fact —
§8.1's content hash, and §10.1 requirement 4's byte-determinism — neither of
which is worth much attached to an artifact whose provenance was never
recorded.

```python
@dataclass(frozen=True)
class DiagramMeta:
    filtration:      str | None   # "rips" | "alpha" | "cubical" | "lower_star" | ...
    backend:         str | None   # "gudhi" | "ripser" | "giotto" | "persim" | "array"
    backend_version: str | None   # as reported by the backend at adapter time
    coeff_field:     int | None   # e.g. 2, 11 — the field homology was computed over (§9.3)
    params:          Mapping[str, Any]  # max_edge_length, max_dimension,
                                         # reduced_homology, ...
    provenance:      Mapping[str, Any]  # adapter-recorded facts; see below
    description:     str | None   # free-text description of the underlying data
```

All fields are optional — a diagram typed in by hand from a paper is a valid
diagram — but `from_*` adapters MUST populate `backend`, `backend_version`, and
`provenance`.

**An adapter MUST also populate `filtration` where its own input form
determines it, and MUST NOT guess otherwise.** One adapter is in that
position: `ripser(X)` and `Rips().fit_transform(X)` compute a Vietoris–Rips
filtration and nothing else, so `from_ripser` MUST record
`filtration = "rips"`, which is a fact about the call it was handed rather
than an inference about what the caller meant. The others cannot: a GUDHI
`SimplexTree` carries no record of what built it — Rips, alpha, cubical and
lower-star all arrive as the same object — `from_giotto` receives a bare array,
and `from_array` and `from_persim` have no backend to ask. Those MUST leave
`filtration` at whatever the caller passed through `**meta`, `None` included.
This is the same rule §11 applies to `coeff_field`, one field over: record what
the adapter knows, mark what it assumed, invent nothing.

**`description` is caller-supplied free text and carries no obligation on
anyone.** No adapter can write it — none of the five receives the underlying
data, only a computed result — no clause in this document reads it, and it
takes no part in `==`, `allclose`, `content_hash`, or `same_provenance`, which
excludes it by name for this reason (below). It is here because a diagram typed
in from a paper has somewhere to say what it is a diagram *of*, and for nothing
else. It MUST NOT acquire a machine-readable meaning — an enum, a parsed
grammar, an equality that turns on it — without a format-version bump (§10.1
requirement 3). It is the only `DiagramMeta` field no comparison in this
document reads, and `save`/`load` still round-trip it like any other.

`provenance` is the honest-accounting channel. Reserved keys:

| Key | Meaning |
|---|---|
| `essential_bars` | one of `"faithful"`, `"lost_upstream"`, `"finitized_at:<value>"`, `"finitized_dropped"` |
| `essential_bars_dropped` | count of essential bars removed by `finitize(at="drop")`; present iff `essential_bars == "finitized_dropped"` |
| `essential_bars_source` | `essential_bars` as the adapter recorded it — `"faithful"` or `"lost_upstream"`, never a `"finitized_*"` value. Written only by `from_*`, never by `finitize` (§5) |
| `coeff_field_source` | where `meta.coeff_field` came from — `"caller"` if the caller stated it, `"backend_default"` if the adapter recorded the backend's documented default (§9.3, §11) |
| `source_dtype` | dtype of the input array |
| `clamped_rows` | count of `death < birth` rows the adapter repaired |
| `padding_removed` | count of trivial rows stripped as suspected batch padding |

Every value in `params` and `provenance` MUST be JSON-representable —
`str`, `int`, **finite** `float`, `bool`, `None`, or a list or `str`-keyed
mapping of those. Non-finite floats are excluded by that word and not by
implication: `inf` and `NaN` are Python `float`s with no JSON spelling, and
`max_edge_length` on an untruncated Rips filtration is an `inf` every GUDHI
caller has. §10.2 stores both as UTF-8 JSON in `meta.json`, so a mapping holding a
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
rather than left to a reader to notice. It MUST likewise raise `ValueError`
when `coeff_field_source` holds anything but `"caller"` or
`"backend_default"`, and when it is present while `coeff_field` is `None`: a
source describing no value is not a weaker record but an incoherent one. The
converse is deliberately legal — a hand-built diagram MAY state a
`coeff_field` and no source, since §8's opening concession is that a diagram
typed in from a paper is a valid diagram and there is no adapter to have
formed a verdict. §11 is where the obligation to record both lands, and it
binds adapters only. Nothing else about `provenance` is validated: §8 reserves
names within an open mapping rather than closing it, so unreserved keys and the
free-form `<value>` in `"finitized_at:<value>"` pass through untouched.

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

**`coeff_field_source` is that key built for the coefficient field, and D17
resolves by building it rather than by adding an argument.** `coeff_field`
stays optional, as the prose above already has it; what changes is that an
adapter is no longer allowed to leave it silent. Where the backend takes a
coefficient parameter, §11 requires the adapter to record the value the caller
passed, or the backend's own documented default if the caller passed nothing —
GUDHI's $\mathbb{Z}/11$, Ripser's $\mathbb{Z}/2$ (§9.3) — and to say which of the two it did. The
condition D15 tested `order` against is what `order` failed and this passes:
**the backend's default is a fact the adapter knows, the caller may not, and no
later inspection can recover** — A.5 measures that no backend returns the field
it computed with, so an unrecorded value is unknown rather than conventionally
$\mathbb{Z}/2$, and a value recorded without its source is a number a reader cannot tell
was chosen or assumed. Recording both costs no signature change and no
friction in the common case, and it leaves the diagram never *silently*
ambiguous, which is the whole of what a required argument would have bought.
§12.2 carries the full argument and what was rejected.

**`meta` MUST NOT participate in `==` or `allclose`.** Two diagrams with the
same bars from different backends are the same diagram. Provenance is recorded
so a human can audit it, not so equality can reject on it. `d1.same_provenance(d2)`
is available for the cases that genuinely care, and `DiagramBatch` gets the
same method: `b1.same_provenance(b2)` requires `len(b1) == len(b2)` and
`b1[i].same_provenance(b2[i])` for every `i` in sequence, order-sensitive the
same way `==` and `allclose` are (§6.3).

**`same_provenance` compares every `DiagramMeta` field except `description`,
each by `==`, and this MUST be specified rather than left to a dataclass's
generated comparison.** `filtration`, `backend`, `backend_version` and
`coeff_field` compare as scalars; `params` and `provenance` compare as
mappings, which is well-defined because §8 already requires every value in
them to be JSON-representable, and exact because it is the same equality a
`load` has to reproduce for §10.1 requirement 1's second clause. The method
returns `bool`, raises `TypeError` on an argument that is not a diagram (§5
relies on this), and is exempt from §6.3's cross-namespace `ValueError`,
touching no array.

**`description` is excluded, and the exclusion is what keeps §8's own rule
true.** This section requires `description` to acquire no machine-readable
meaning without a format-version bump; a field that can flip
`same_provenance` from `True` to `False` has one. Two diagrams recording the
same filtration, backend, version, field, parameters and provenance differ in
nothing a reader is entitled to act on, and the free text one of them carries
about what the data *was* is not a sixth fact about the computation. A caller
who wants it compared has the field and can compare it.

The default a frozen dataclass would generate — compare all seven — is
therefore wrong here in exactly one field, which is the reason to write the
rule down rather than inherit it.

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
order-sensitive across diagrams for the same reason: §4's leading batch
dimension is a positional axis, and `[A, B]` and `[B, A]` are different
batches even when they hold the same two diagrams. A hash that
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
the backend. If they agree, it MUST delegate the finite parts, and it MUST NOT
pass a diagram containing `inf` to persim under any circumstances.

**The essential part is then computed here rather than delegated — the one
place this document requires a consumer to implement part of a distance rather
than call one (D19).** The bottleneck distance is

$$
d_{B}(D_{1}, D_{2}) = \inf_{\gamma} \sup_{p \in D_{1}} \lVert p - \gamma(p) \rVert _{\infty}
$$

over bijections $\gamma : D_{1} \cup \Delta \to D_{2} \cup \Delta$, where the
cost of sending a bar $p = (b, d)$ to the diagonal is

$$
\lVert p - \bar{p} \rVert _{\infty} = \frac{d - b}{2}
$$

That cost is $+\infty$ for an essential bar, and so is
$\lVert p - q \rVert _{\infty}$ for $p$ essential and $q$ finite. An essential
bar can therefore be matched only to another essential bar, at cost
$\lvert b_{1} - b_{2} \rvert$. Within each dimension:

- **Essential bars pair by sorted birth.** Sort each side's essential births
  ascending, $\beta^{1}_{1} \le \cdots \le \beta^{1}_{m}$ and
  $\beta^{2}_{1} \le \cdots \le \beta^{2}_{m}$ — equal in count, or the
  `+inf` above already returned — and pair them by index. Sorted pairing
  minimises the largest $\lvert \beta^{1}_{i} - \beta^{2}_{i} \rvert$ over
  one-dimensional data, so no search is needed:

$$
d_{\mathrm{ess}} = \max_{1 \le i \le m} \lvert \beta^{1}_{i} - \beta^{2}_{i} \rvert
$$

  with $d_{\mathrm{ess}} = 0$ when $m = 0$.

- **Combine with $\max$, never a sum:**

$$
d_{B}(D_{1}, D_{2}) = \max \left( d_{\mathrm{ess}}, \; d_{B}\left( D_{1}^{\mathrm{fin}}, D_{2}^{\mathrm{fin}} \right) \right)
$$

  the second term being persim's answer on the two finite sub-diagrams.
  Bottleneck is a supremum over a single matching, and the essential pairing
  being forced makes the two sub-problems independent, so minimising each and
  taking the larger is the same as minimising the whole.

This is a guardrail: a negative result about a dependency, converted into a
safety feature, worth documenting publicly. It is also a named exception to the
project's delegation rule rather than a drift away from it, which is why it
carries a decision row; the implementation owes the equation citation
`CLAUDE.md` requires of numerical code in `core/`, and its tests owe the
separate-session rule that file states for numerical code.

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

*Clean-room note: giotto-tda is AGPLv3. The above was determined by calling
public API and reading a traceback. No giotto source has been read, and MUST
NOT be read while implementing `compat/`.*

### 9.3 GUDHI and Ripser compute over different coefficient fields by default

| Backend | Parameter | Default |
|---|---|---|
| GUDHI | `SimplexTree.persistence(homology_coeff_field=...)` | **11** — $\mathbb{Z}/11$ |
| Ripser | `ripser(..., coeff=...)` | **2** — $\mathbb{Z}/2$ |

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

This is recorded independently of D17, and survived it. D17 decided what the
adapters *do* about the fact — `from_gudhi` and `from_ripser` record the field,
falling back to these two defaults and saying so (§8, §11, §12.2); §9 is where
the fact itself is written down, and it was true under every option D17 had.

**The two defaults in the table above are now load-bearing and MUST be
asserted in CI**, one test per backend, against the installed version. §11 has
the adapters write these numbers into the provenance of diagrams that never
stated one, so a change to either default upstream stops being a documentation
drift and becomes silently wrong provenance on every diagram recorded
afterwards. A measured fact a requirement leans on belongs in a standing test
rather than in prose — the same argument D16 made for verifying namespace
identity instead of assuming it, and the one §7 makes for its `lexsort`
regression test. A backend that changes its default breaks the build, which is
where a claim about a dependency should fail.

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

**This clause does not add a test; it makes a test §11.2 already requires
test what it claims.** Unpinned, that comparison sets GUDHI's $\mathbb{Z}/11$ against
Ripser's $\mathbb{Z}/2$ — two homology theories, not one computation done twice. On
torsion-free input, which synthetic test data almost always is, they agree
anyway, so the test passes, establishes nothing, and would go on passing
through a genuine regression in either adapter. Pinning both sides is what
makes a green result mean something.

It was also independent of D17 and remains independent of how D17 landed,
which is why it was written before that row closed: pinning is a call
parameter the test itself controls, not a claim about what the returned object
carries or an obligation on a caller of ours. §11's recording requirement and
this one meet nowhere — one governs what an adapter writes down about a
diagram it is handed, the other what our own test asks the backends for.

---

## 10. Serialization

### 10.1 Requirements

The five requirements below are normative, and each states an obligation on
the on-disk format and on `save`/`load`.

1. **Round-trips exactly.** With `save(d, p)` having written the file,
   `load(p) == d` MUST hold, including `inf` and multiplicity, **and
   `load(p).same_provenance(d)` — metadata
   round-trips too.** The second clause is not redundant. §8 requires `meta`
   to take no part in `==`, so a `load` that silently discarded every byte of
   `params` and `provenance` would satisfy the first clause completely. §5
   depends on the second: its argument for why `finitize` must not
   copy `essential_bars` forward into `essential_bars_source` turns on a
   diagram arriving from `load` already carrying a `"finitized_*"` value, and
   there is no such diagram unless `load` preserves provenance. The two
   clauses cover the two halves of a diagram that `==` deliberately splits,
   and requirement 1 needs both.

   **Both clauses bind `DiagramBatch` on the same terms**: with `save(b, p)`
   having written the file, `load(p) == b` and `load(p).same_provenance(b)`
   MUST hold for every batch this document admits, the empty one included
   (§4.2). Batch equality and `same_provenance` are order-sensitive across
   diagrams (§6.3, §8), so a `load` that recovered every diagram intact and
   in the wrong order satisfies neither clause, which is the property that
   makes this worth stating separately rather than reading as implied by the
   single-diagram case: `offsets` and the `metas` list are the two things a
   batch round-trip can lose, and neither exists in the previous case. §11.2
   tests the batch path as its own case.

   **The `==` clause is stated over NumPy-backed objects, because that is what
   `load` returns** (§3.3). §6.3 makes a cross-namespace `==` raise
   `ValueError` deliberately, so `load(p) == d` on a JAX-backed `d` raises
   rather than answering `False` — the comparison is not weaker for other
   namespaces, it is unavailable, and requirement 1 would otherwise be a MUST
   no conforming implementation could satisfy. `save` MUST still accept a
   diagram or batch backed by any namespace, converting at the I/O boundary
   only (§3.3); what a caller wanting the comparison does is convert at their
   own boundary first, which §6.3 already names as the remedy. **The
   `same_provenance` clause carries no such restriction** and binds every
   namespace: §6.3 exempts it from the cross-namespace raise, it touching no
   array. §11.2's round-trip cases are NumPy-backed for this reason, and are
   the one part of that suite §3.3's `array_api_strict` run cannot cover;
   the invariant, accessor and `content_hash` tests are unaffected and stay
   under it.
2. **Zero-dependency by default, with narrow, lazily-imported exceptions.**
   `pip install akriti` MUST resolve to nothing third-party, and importing the
   package, or constructing, inspecting, or comparing a diagram, MUST NOT
   reach a third-party library on any path a caller can take without having
   installed a backend themselves. Beyond that, a function MAY depend on a
   third-party library where **every** one of the following holds: the import
   is lazy and function-scoped; nothing outside the functions that need it
   requires it; the library is declared as an install extra carrying a version
   floor; and the import fails actionably on both absence and an unsatisfied
   floor. `numpy` in `save`/`load` (`akriti[io]`), `pyarrow` in `to_parquet`
   (`akriti[parquet]`, §10.3), and `array-api-compat` in §3.3's namespace
   resolution (`akriti[torch]`, reached only for an array from a backend the
   caller installed) are what currently meets it.
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
round-tripping and not for `content_hash`. `load(p) == d` (requirement
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
meta.json      UTF-8 JSON, the schema below
bars.npz       npz with arrays: births, deaths, dims  (canonical order, §7)
```

An `.akd` archive MUST contain exactly these two members, under exactly these
names, written in this order. `inf` lives in `bars.npz`, where NumPy
represents it correctly, and never in the JSON. This is the reason for the
split.

```python
akriti.diagrams.save(d, "sample.akd")     # d: PersistenceDiagram or DiagramBatch
obj = akriti.diagrams.load("sample.akd")  # -> whichever kind was saved
```

**`meta.json` schema.** Requirement 3 turns on this file, so it is specified
here rather than left to the implementation, the same reason §8.1 fixes a hash
message byte for byte:

```json
{
  "format": "akriti.diagrams.akd",
  "format_version": 0,
  "spec": "RFC-0001",
  "spec_version": "0.1.0",
  "kind": "diagram",
  "meta": { "filtration": "rips", "backend": "ripser", "...": "..." }
}
```

| Key | Type | Rule |
|---|---|---|
| `format` | `str` | Exactly `"akriti.diagrams.akd"`. This is requirement 3's self-identification, and it MUST be a fixed string rather than anything derived, so a reader can recognise the file without parsing the rest |
| `format_version` | `int` | The version of *this layout*, currently `0` — pre-publication. Incremented whenever a change would make an older `load` misread a newer file. The one version key `load` is allowed to branch on |
| `spec` | `str` | Which specification defines the file: `"RFC-0001"`. Separate from `format` so that a format defined by some later RFC is distinguishable from a later revision of this one |
| `spec_version` | `str` | Which revision of that specification the writer implemented, `major.minor.patch`, `"0.1.0"` at time of writing. A string rather than a number because `0.10.0` follows `0.2.0` and the float ordering says otherwise. Recorded for audit; `load` MUST NOT branch on it — a spec revision that changes what `load` must do is a `format_version` bump by definition, and one that does not is a revision older readers are entitled to ignore |
| `kind` | `str` | `"diagram"` or `"batch"`. Nothing else is valid |
| `meta` | object | Present iff `kind == "diagram"`: one `DiagramMeta` as a JSON object, its own keys being the field names of §8's dataclass |
| `metas` | array | Present iff `kind == "batch"`: the per-diagram `DiagramMeta` objects, in batch order |

**`load` MUST dispatch on `kind`, and MUST NOT infer the type from the
payload.** Deciding by whether `bars.npz` happens to contain an `offsets`
array reads a fact about the arrays as if it were a fact about what was saved,
so a diagram file that acquires an `offsets` key loads as a batch and a batch
file that loses one loads as a diagram — clean, plausible and wrong in both
directions, from a file the reader cannot see is malformed.

`load` MUST also, before returning anything:

- reject a `format` that is not the exact identifier, a `spec` that is not
  `"RFC-0001"`, and a `format_version` it does not implement, in all three
  cases by raising rather than attempting a best-effort read — `spec` exists
  to distinguish a file some later RFC defines from a later revision of this
  one, which it can only do if `load` acts on it;
- reject a `kind` inconsistent with the members present — `"diagram"` with
  `metas`, `"batch"` without `offsets` in `bars.npz`, either with both `meta`
  and `metas`;
- check B1 for a batch (`len(metas) == offsets.shape[0] - 1`) before
  constructing one, since §4.2's invariants are enforced at construction and a
  file is exactly the untrusted input they exist for;
- ignore unrecognised keys within a `format_version` it does implement, which
  is what lets a later revision add an advisory field without a version bump.

**Serialization of `meta.json` is pinned, not left to a JSON library's
defaults**, because requirement 4 makes the bytes load-bearing: UTF-8, keys
sorted at every level, no insignificant whitespace, and non-finite floats
rejected rather than emitted. In CPython that is
`json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False,
separators=(",", ":")).encode("utf-8")`. `allow_nan=False` is a backstop
rather than the rule: §8 excludes non-finite floats from `params` and
`provenance` outright, and adapters MUST convert such a value to a string at
the point of recording, on the rule §8 already states for `source_dtype`. What
`allow_nan=False` adds is that a mapping reaching `save` any other way fails
there, rather than emitting the `Infinity` token §10.1 rejects JSON for.

For a `DiagramBatch`, `bars.npz` additionally carries `offsets`, an `int64`
array of length `len(batch)+1` giving the CSR-style row range of each diagram.
Ragged, exact, no padding.

### 10.3 Interoperable escape hatches

**All three live in `adapters.py`, not in `io.py` and not in `core.py`.** They
are the export direction of the job §11 does on the import side — translating
between this type and a representation someone else defined — and
`to_arrays()` returns the exact format `from_persim` and `from_ripser`
consume. `to_csv()` and `from_array` matter most: the column-name rule below
makes them a round-trip pair, and a pair split across two modules is two
things that can drift apart. `io.py` keeps the one format this document makes
normative, `.akd` and its `save`/`load`. §10.1 requirement 2 binds the lazy
`pyarrow` import wherever the function sits, so the placement costs nothing
against the closure. `core.py` MAY expose them as methods (`d.to_csv(...)`)
delegating to `adapters.py`.

Non-normative, and all three MUST warn about what they lose:

- `to_arrays()` → `dict[int, xp.array]`, degree to `(n,2)` array. This is the
  de-facto community format (what Ripser returns and persim consumes) and is
  what people will paste into other tools.
- `to_csv()` → three columns `dim,birth,death`, with `inf` written as the
  literal `inf`, **preceded by a header row naming them**. For humans and for
  spreadsheets.
- `to_parquet()` → a `pyarrow.Table` with the same three columns and order as
  `to_csv()` (`dim` int32, `birth`/`death` float64), so `inf` round-trips
  exactly — Parquet's `double` is IEEE 754, unlike JSON's. Requires
  `pip install akriti[parquet]  # pyarrow (Apache 2.0)` (D8); `pyarrow` MUST
  be a lazy, function-scoped import inside `to_parquet()`, on §10.1
  requirement 2's terms. For a `DiagramBatch`, an integer `diagram_id` column
  is prepended rather than an `offsets` array — Parquet's natural unit is the
  row, not a CSR buffer — and `to_csv()` prepends the same column for the same
  reason. Carries none of `DiagramMeta`: no `backend`, no `provenance`, no
  `params`. This is a bars-only interchange table for R/pandas/Polars
  pipelines (§1's R-bridging goal), not a `.akd` replacement.

**The header row is what carries the column order back to `from_array`.**
`to_csv()` writes `dim,birth,death`, the order a human wants;
`from_array`'s `(n,3)` is `(birth, death, dim)`, matching giotto deliberately
(§11). Handing one to the other transposes two columns and reports nothing —
§10.1 requirement 5 and D12 both lean on `to_csv()` being this format's
human-readable surface, and a surface that cannot come back in through the
library's own array adapter is a weaker thing than either argument assumed.
**`from_array` MUST therefore take the column order from a `columns` argument
wherever the caller has one** — a sequence of strings naming `arr`'s columns in
order, which is exactly what a header row is and exactly what `to_csv()` now
writes:

```python
from_array(arr, *, columns=None, dim=None, **meta)   -> PersistenceDiagram
```

Recognised names are `birth`, `death` and `dim`, matched case-insensitively.
`columns` MUST have one entry per column of `arr`, and a length disagreement
MUST raise. A name that is not one of the three MUST raise rather than fall
through to position — a name that went unrecognised is the one case where the
positional reading has been actively contradicted.

**A separate argument, not names carried on the array itself.** A NumPy
structured or record array is the obvious alternative and MUST NOT be the
mechanism: the array API standard defines no structured dtype and no way to ask
an array for its field names, so recognising one means reaching for
`.dtype.names` — a NumPy-shaped idiom applied to an array the caller handed in,
which is what §3 forbids `adapters.py`, and which no other backend here
answers. Loaders also differ in whether they carry names at all, which is why
this is an argument rather than something inferred from the input.

**`diagram_id` is deliberately not among the recognised names, so a batch
written by `to_csv()` does not read back through this adapter.** `from_array`
returns a `PersistenceDiagram` (§11), and a return type that became a
`DiagramBatch` on the presence of a column is exactly the hazard §11 rules out
for `from_giotto`: the adapter's own result type depending on a runtime
property of the data. A `columns` entry naming `diagram_id` MUST raise naming
the column and pointing at `.akd`, which is the batch round trip (§10.1
requirement 1). The asymmetry is deliberate and bounded — `to_csv()` on a batch
exports to other people's tools, and this document's own batch round trip is
the normative format.

**`columns=None` keeps `(birth, death, dim)`, and §3.1 is what makes that
safe to default to.** It is the documented order and giotto's, so it is the
right reading for a bare `(n,3)` array of bars from anywhere. Where the default
is wrong — a headerless file in `to_csv()`'s own order — the `death` column
lands in `dims`, and a column holding `inf`, or any non-integral value, fails
I2 and I3 at construction rather than producing a diagram. That is not a proof:
a diagram with no essential bars whose coordinates are all small non-negative
integers survives the transposition and constructs cleanly. It is D16's trade
taken in D16's direction — a default that fails loudly on nearly every real
input beats refusing every nameless array — and the residual case is why the
header is a MUST on the writing side rather than a convenience.

---

## 11. Adapter contract

Signature for all five, with five deviations across three adapters:

```python
from_gudhi(obj, *, dim=None, **meta)                  -> PersistenceDiagram
from_ripser(obj, **meta)                              -> PersistenceDiagram
from_giotto(arr, *, reduced_homology, **meta)         -> DiagramBatch
from_persim(obj, **meta)                              -> PersistenceDiagram
from_array(arr, *, columns=None, dim=None, **meta)    -> PersistenceDiagram
```

**`dim` is keyword-only on the two adapters whose input may carry no degree,
and MUST be supplied exactly when it does not.** Both adapters
MUST raise `TypeError` when handed a degreeless input without `dim=`, and MUST
raise `TypeError` when handed a degree-carrying input *with* it — GUDHI's
`list` form and `from_array`'s `(n,3)` both carry every degree already, so a
`dim=` alongside one is either redundant or a contradiction and neither is
worth resolving by guessing.

`**meta` is not the channel for this and MUST NOT be used as one. It populates
`DiagramMeta` (§8), and `dim` is bar data rather than metadata; routed through
`**meta` a misspelling becomes an accepted metadata key rather than a
`TypeError`, and the degree — what I3 constrains and `d.dim(k)` selects on
— would arrive by the same door as a free-text `description`.

**`columns` is `from_array`'s second deviation, and it is bar data on the same
grounds**: a sequence of strings naming `arr`'s columns in order, defaulting to
`None` for the positional `(birth, death, dim)` reading. §10.3 specifies it,
because what it exists for is reading `to_csv()` output back in.

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
`backend_version`, `provenance`, and `filtration` where its own input form
determines it (§8 — `from_ripser` is the one that does); preserve backend row
order; and never finitize, sort, or deduplicate.

Every adapter MUST also reject an input form not in the table below rather than
attempt it. `from_gudhi` is where this bites, and the rejection it can perform
is narrower than the exclusion it enforces. `extended_persistence()` returns a
**4-tuple** of `list[(dim, (b, d))]` (§1), structurally distinct from
`persistence()`'s flat list, so `from_gudhi` MUST reject that tuple outright
and MUST name the scope exclusion rather than the shape.

**One of its four members, passed alone, is indistinguishable from ordinary
output, and this document does not require an adapter to detect what it cannot
see.** The relative and extended− members raise at construction on I6, with an
error about death times rather than about scope; the ordinary and extended+
members satisfy every invariant and construct cleanly into a diagram whose bars
this type cannot mark as extended. `from_gudhi`'s docstring MUST state this
residual case, since it is the one place in the adapter contract where a
rejected input form can arrive undetected. Closing it needs the fourth per-bar
field §1 rules out, so it stays open by the same decision.

An adapter that records `provenance["essential_bars"]` MUST record
`provenance["essential_bars_source"]` with the same value in the same
construction (§5, §8). The adapter is the only writer that can: the key means
"the verdict at computation time", and every later writer sees only a field
that may already have been overwritten. `finitize` (§5) MUST NOT write it.

**`from_gudhi` and `from_ripser` MUST record the coefficient field** (D17).
Each MUST set `meta.coeff_field` and `provenance["coeff_field_source"]` in the
same construction: to the caller's value with `"caller"` if one arrived in
`**meta`, and otherwise to that backend's documented default —
GUDHI 11, Ripser 2 (§9.3, A.5) — with `"backend_default"`. `coeff_field`
remains optional on the type and no adapter signature changes; what is
forbidden is an adapter leaving the field silent when it knows what the
backend would have done.

**`"backend_default"` is an assumption, and recording it as one is the point.**
No backend returns the field it computed with (A.5), so an adapter cannot
verify the caller left the default in place: a caller who passed
`homology_coeff_field=3` to GUDHI and did not pass `coeff_field=3` on to
`from_gudhi` gets a diagram recording 11. That is a marked assumption rather
than a silent one — the source key is what a reader checks before trusting the
value — and it is strictly better than the alternative it replaces, which is a
diagram carrying nothing and a reader defaulting to $\mathbb{Z}/2$ on a backend that uses
$\mathbb{Z}/11$. Adapter documentation SHOULD tell callers to pass the field through
whenever they set it on the backend.

`from_array` and `from_persim` are out of scope for this clause, having no
backend and computing no homology respectively. **`from_giotto` is excluded
for now on evidence, not on principle:** A.5 records giotto's default as
unmeasured (§9.2 — it does not currently run on installed scikit-learn), and
this document does not assert a backend default it has not measured. When
§9.2's shim is testable again, `from_giotto` joins this clause on the same
terms.

Measured input formats (Appendix A):

| Source | Accepted input | Notes |
|---|---|---|
| GUDHI | `SimplexTree.persistence()` → `list[(dim, (b, d))]`; also `persistence_intervals_in_dimension(k)` → `(n,2)` with explicit `dim=k` | `inf` faithful. Both forms MUST be accepted; the `list` form carries all degrees at once and MUST be rejected if given `dim=`. `extended_persistence()`'s 4-tuple is a third form and MUST be rejected outright; note: a single member of it passed alone is undetectable (§1, §11). |
| Ripser | `ripser(X)` → `dict` with `"dgms"`; `Rips().fit_transform(X)` → `list[(n,2)]` | Index in the list *is* the degree. `inf` faithful. `float32` precision (§6.2). The one adapter that knows its own `filtration` (§8). |
| giotto | `(n_samples, n_bars, 3)` array, columns `(birth, death, dim)` | Essential bars lost (§5.1). Padding ambiguity (§4). Always returns a `DiagramBatch`, length 1 when `n_samples == 1`. |
| persim | `list[(n,2)]`, degree by index | Same shape as Ripser's `dgms`. |
| array | `(n,2)` with explicit `dim=`, or `(n,3)` with `(birth, death, dim)` | The `(n,3)` column order matches giotto's, deliberately, and is what `columns=None` reads. A `columns=` sequence names the array's columns instead and wins over position (§10.3), which is what makes `to_csv()` output readable back in. |

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
- **The two coefficient-field defaults §9.3 tabulates, asserted against the
  installed backend** — GUDHI 11, Ripser 2 — one test per backend, marked
  `@pytest.mark.backend`. §11 writes these numbers into diagrams that stated
  no field, so an upstream change to either MUST break the build rather than
  reach a user's provenance.
- **`coeff_field` and `provenance["coeff_field_source"]` recorded by
  `from_gudhi` and `from_ripser` in both directions** (§11): passing the field
  gives `"caller"` and the value passed; omitting it gives
  `"backend_default"` and that backend's default. A suite testing only the
  omitted case passes on an adapter that ignores the argument outright.
- **`allclose` on two diagrams that are within tolerance of each other but
  whose canonical orders differ because of that tolerance**, asserting
  `True`. Two bars in one dimension whose births are tied to within `rtol`
  and whose deaths are far apart: a bijection within tolerance exists, and
  the tie is what lets §7's sort place the two bars in opposite orders on the
  two sides. The rejected sorted-pairwise form pairs each bar against the
  other's partner and returns `False`; the matching §6.3 requires returns
  `True`. This is the case that motivated the matching, and a suite without
  it passes identically against either implementation.
- **`allclose` on essential bars, both directions** (§6.3): a diagram carrying
  an essential bar is `allclose` to itself, at the default tolerance and at
  every other; and two diagrams alike but for one death being `+inf` on one
  side and finite on the other are not `allclose`, at any `atol` or `rtol`
  however wide. The first is the case §6.3's tolerance formula gets wrong on
  its own — `|inf - inf|` is `NaN` and every comparison against `NaN` is
  `False` — so a suite without it passes against an implementation that has no
  `+inf` clause at all, on the diagrams I5 makes ordinary.
- `save`/`load` byte-determinism: dumping twice gives identical bytes.
- **A `DiagramBatch` round-trip, as its own case** — both clauses of §10.1
  requirement 1, with `save(b, p)`, `load(p) == b` and
  `load(p).same_provenance(b)`. The suite MUST cover a batch whose diagrams
  have different bar counts, so `offsets` is exercised rather than degenerate;
  one containing an empty diagram, so a zero-length segment is; one of length
  zero (§4.2's `xp= `constructor); and one whose diagrams are in an order that
  no sort would produce, since both comparisons are order-sensitive across
  diagrams (§6.3, §8) and a `load` that recovered every diagram into the wrong
  slot passes a test built from identical members.
- **`load` dispatches on `kind`, not on the payload** (§10.2): a saved diagram
  loads as a `PersistenceDiagram` and a saved batch as a `DiagramBatch`,
  including a length-one batch, which holds exactly the bars its member does
  and is the case an `offsets`-sniffing implementation gets wrong. A file whose
  `format` string, `format_version` or `kind` is unrecognised, and one whose
  `kind` disagrees with the members present, MUST each raise rather than load.

Property-based tests (Hypothesis) for the invariants and for both clauses of
§10.1 requirement 1 — with `save(d, p)`, `load(p) == d` *and*
`load(p).same_provenance(d)`, since the first passes on a `load` that
drops metadata entirely. They fit this layer unusually well: §3.1's and
§4.2's invariants are universally quantified over every diagram the type
admits, which is what a property-based test states directly and an
example-based one only samples.

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

## 12. Decisions

Sixteen decisions are on record: D1-D8 and D12-D19. **All sixteen are
settled** (§12.2), each stating the outcome and pointing at the section that
carries the normative requirement; §12.1 is empty. Superseded recommendations
are logged in the history document rather than repeated here.

**D6 is reinstated as superseded, not deleted.** It was removed with D9, D10
and D11 in the pass below, and its resolution has since been reversed: the
row is restored in §12.2 carrying both its original resolution and the one
that replaced it.

**D9, D10, and D11 were removed from this RFC** as dependency-and-licensing
policy questions rather than interchange ones, and this document does not set
that policy. Nothing normative went with them — §3.3 and §10.1 state the
zero-dependency-by-default requirement and `numpy`'s lazy-import behaviour
directly, in MUST language, and never depended on a table row to carry it.
Their prior text and the reasoning for removing them are in the history
document. D-numbers are not renumbered to close the gap; they are stable
identifiers, not a dense sequence.

### 12.1 Open

**None.** Every decision this document opened has been resolved; the rows are
in §12.2. The section is kept rather than deleted so that a later question has
a place to open into, and so a reader arriving here from a cross-reference
finds the state of the log stated rather than inferred from an absence.


### 12.2 Settled

| # | Question | Recommendation / status |
|---|---|---|
| D1 | File extension `.akd`, or plain `.npz` with our layout inside? | **Resolved by §10.** §10.1/§10.2 normatively specify `.akd`; Parquet is excluded as the *default* format. This row originally recommended Parquet, contradicting both; the correction is logged in the history document. |
| D2 | Is `DiagramBatch` in scope for M1, or does M1 ship the single-diagram type only? | **In scope.** §4 requires every numerical function in `core/` and `castle/` to take a leading batch dimension rather than expecting a Python loop over diagrams, so the container those signatures consume has to exist before they are written — and retrofitting one after `core/` is written against scalars is the expensive order. That commitment is not made for this layer's convenience: a looping API is very hard to withdraw once published, and it is the usability complaint most often made of the library this project is positioned against. |
| D3 | Do we accept `float32` storage behind a flag for large-scale work? | No, not in v0. Revisit when a real memory complaint exists. |
| D4 | Should `from_giotto` default to `strip_padding=True`? | No. Defaulting to a lossy repair contradicts §5's whole argument. Warn and let the caller choose. |
| D5 | Does the RFC published at M1 include §9's delegation hazards, or do we raise them upstream first? | Raise upstream first — file the persim issue and the giotto scikit-learn issue, then publish citing our own reports. Costs two weeks, buys enormous goodwill, and turns a criticism into a contribution. |
| **D6** | Array-API support (§3.3) needs a NumPy that has it. Raise the floor to `numpy>=2.0`, or add `array-api-compat` and keep `numpy>=1.24`? | **Superseded, and reinstated here rather than deleted.** Original resolution: raise the floor to `numpy>=2.0`, main-namespace array API support having landed in NumPy 2.0, declared in `pyproject.toml` as a required dependency. That resolution was merged. It is now superseded, not withdrawn: `numpy` leaves the required closure — `pip install akriti` resolves to nothing third-party (§10.1 requirement 2) — but moves to `akriti[io]` rather than vanishing, so the `>=2.0` floor stays declared and resolvable at install time. D6's purpose is what makes the extra necessary rather than optional: undeclared, a user on `numpy` 1.24 gets an `AttributeError` at run time instead of a resolver error at install time, because a presence-only lazy import never fires on a version that is merely too old. §3.3 carries the version check and the two failure paths, both naming the extra; §10.1 requirement 2 carries the declare-an-extra-with-a-floor obligation. |
| **D7** | Does `DiagramBatch` need its own `content_hash`, and if so, defined how? | §8.2 defines `DiagramBatch.content_hash`: composed from member `PersistenceDiagram.content_hash`es in batch order, not re-serialized from the raw buffer; domain-separated from `PersistenceDiagram.content_hash` by a type tag, so a one-element batch cannot collide with the diagram it wraps; and exact-equality only, no approximate form, per §6.3's exact/approximate split. §4.2's and §4.3's cross-references updated to point at §8.2 rather than flag the gap. |
| **D8** | Should Parquet be offered anywhere, given §10.1 rules it out as the default (`.akd`) storage format? | §10.3's `to_parquet()` (`akriti[parquet]`, Apache 2.0, lazy-imported per §10.1's pattern). Previously logged as resting on D9/D11; those rows are now out of this RFC's scope entirely (see above), and this row no longer depends on them. Whatever the project's actual license-family policy turns out to be is a packaging-level check against `tools/check_license_closure.py`, not something this RFC re-litigates. `tools/check_license_closure.py` and `DEPENDENCIES.md` still need updating for the new extra. |
| **D12** | §10.2 specifies `bars.npz` as the default array storage inside `.akd`, defended in §10.1 on requirement 5 (inspectability) against HDF5 and Parquet. Two stdlib alternatives that clear requirement 2 outright rather than through the lazy-import exception, `csv`/`tsv` and `sqlite3`, were never run through the same test — and CSV plausibly satisfies requirement 5 *better* than `.npz` does, being directly readable without even `numpy.load`. Does `.npz` remain the default, or should one of these replace it? | **`bars.npz` stays, and §10.2's payload is now normative rather than provisional.** Resolved on measurement (Appendix A.6), which supplies the bar-count figure this row said it turned on: H0 bar count equals the input point count *exactly*, so bars scale linearly in cloud size, and a 5,000-sample batch is around 4.7 million bars. At 1M bars CSV costs ~2.1× the bytes and ~1.2 s to load where `.npz` costs ~0.01 s — two orders of magnitude — and sqlite3 ~1.3× and ~0.8 s. **A.6 records absolute times and no multiplier deliberately**, the ratio being a quotient by the fastest thing in the table and therefore sensitive to how that baseline is sampled; the order of magnitude is what survives re-measurement, and it is what this row rests on. **The reason is not that CSV loses on inspectability — it wins there.** It is that requirement 5 is already satisfied twice without it, by `meta.json` as literal archive text and by §10.3's `to_csv()`, which exists to be the human-readable surface. Requirement 5 does not need satisfying a third time, and paying that cost on every `load()` to duplicate an escape hatch this document already ships is the wrong trade. **One argument for CSV survives, and is recorded rather than waved away:** a stdlib payload would let the `[io]` extra be dropped altogether, `meta.json` being stdlib `json` already, and "zero dependencies, including serialization" is a stronger claim than the one this document makes. It is not worth two orders of magnitude on load at four-million-bar scale — but **that is the condition to reopen D12 against**, together with a use case where batches are small and dependency-freedom outweighs load time. **sqlite3 is closed out** on four counts: larger, far slower, not inspectable without a separate tool, and its files carry internal page state that makes requirement 4's byte-determinism harder rather than easier. A.6 records the caveat that belongs with the figures: two datasets, both alpha-complex, both low-dimensional. They fix the order of magnitude and the linear scaling, not a distribution over what users will store — and the decision is robust to that, because every direction the bar count moves makes CSV worse. |
| **D13** | `PersistenceDiagram` (§3) is single-parameter-persistence-shaped: one scalar `dim`, `birth`, `death` per bar. §3.2 referenced "the multiparameter case" in passing, but nothing in this RFC said whether that module reuses this type, needs a parallel type, or forces a breaking change to this one once it exists. Does `PersistenceDiagram` need a version boundary, an extension point, or an explicit non-goal statement now, before adapters and `core/` are written against its current shape? | **Explicit non-goal, stated in §1** — normative scope rather than an aside in §3. No extension point and no new version machinery. Multiparameter persistence modules do not decompose into intervals and admit no complete discrete invariant, so a multiparameter "diagram" is not this type with an extra column but a different object — a rank invariant, a fibered or signed barcode — and no extension point designed now would fit a shape nobody can yet specify. An extension point would instead put a case into every adapter, accessor and invariant check to serve a module that is not on the roadmap through M4. If it is ever built it takes a parallel type and the two coexist; this is not a deprecation path. The forward-compatibility cost is already paid by §10.1 requirement 3's format version, and that is the whole of the version boundary this needs. §3.2's `d.h0`/`d.h1` note no longer leans on the multiparameter case: `d.dim(k)` is canonical and an unwithdrawable alias is a permanent liability regardless. |
| **D14** | §6.3 required `allclose` to be approximate and order-insensitive but did not say how bars are paired. Sorting both sides canonically (§7) and comparing pairwise is exact in the sort and approximate in the comparison, and the two do not compose: bars whose births lie within tolerance of each other can canonicalise into different orders on two backends, and the comparison then returns `False` for diagrams that do have a partner for every bar within `rtol` — at the `2.7e-8` magnitude Appendix A.3 measures, on exactly the cross-backend case §6.2 defines `allclose` for. Does `allclose` become a matching over the multiset, or does §6.3 accept the conservative false negative and require it to be documented? And is the tolerance symmetric, or does `rtol` scale `other` as `numpy.allclose` does? | **Both resolved in §6.3.** `allclose` MUST be a matching: a bijection under which every matched pair shares a `dim` exactly and agrees on both coordinates within tolerance; equal bar counts are necessary and not sufficient. The tolerance MUST be symmetric, `atol + rtol * max(abs(a), abs(b))`, deliberately diverging from `numpy.allclose` and documented as such in the method's own docstring. `allclose` is reflexive and symmetric but not transitive, and MUST be documented as not an equivalence relation. Accepting the false negative was weighed and rejected: the caller's remedy for a spurious failure is to widen `rtol` until it passes, which relocates §6.3's silent loosening into user code where nobody reviews it. No sort key repairs the pairwise form — the history document carries that argument. No new dependency: an augmenting-path matching in the standard library is sufficient at these sizes. Not bottleneck distance, and §9's delegation rule is unaffected in both directions — `core/distances.py` MUST NOT be built on this method. |
| **D15** | §8 reserves `provenance["order"]` with values `"backend"` and `"canonical"`, but names no writer for the second. §7 forbids adapters from sorting, so every `from_*` adapter records `"backend"`; `d.canonical()` is the operation that makes `"canonical"` true of a diagram, and §7 has it carry `meta` through unchanged, so a sorted diagram still reports `"backend"` and nothing ever writes the other value at all. Does `canonical()` become a second writer of `order`, does `save` write it (§10.2 emits `bars.npz` in canonical order regardless of what the in-memory diagram reports), or does the key not earn its place now that §7 makes row order advisory to a reader and load-bearing for nobody? | **The key goes.** §8's reserved-key table drops `order`; §7 and `core.py` drop their references to it. What decides it is not the missing writer but a property every other reserved key has and this one does not: **the rest record facts that vanish if unrecorded.** `source_dtype` is lost the moment the array is upcast, `clamped_rows` the moment the rows are repaired, `padding_removed` the moment they are stripped; `essential_bars_source` records a then-versus-now distinction no later inspection can recover. Whether rows are in canonical order is recoverable from the arrays themselves, exactly, in one pass, forever. `order` is a cached answer to an always-computable question, and one that goes stale: anything reordering rows while carrying `meta` through unchanged leaves a diagram reporting `"canonical"` while not being canonical — clean, plausible and wrong, §9's own category, arrived at by our hand rather than a dependency's. **Having `save` write it is worse rather than better**: §10.2 emits `bars.npz` in canonical order unconditionally, so the key would be a constant in every file on disk — no information, and a claim about the in-memory diagram it came from that may be false. And §7 leaves no consumer who could act on it either way, every consumer being required to treat row order as arbitrary, so a reader seeing `"canonical"` is forbidden to do anything differently. **The one genuine order fact is a different key**, and §8 records it so this is not reopened: whether the *backend's own output* was already canonical is not derivable, is exactly the GUDHI-versus-Ripser disagreement §7 documents and A.3 measures, and is unrecoverable once anything sorts. `order` as specified never captured it — every adapter records `"backend"` regardless of what the backend did — so it is a key to build at adapter time if order provenance is ever wanted, not this one repaired. |
| **D16** | I7, B5 and §4.2's `from_diagrams` check are all written as `is` on `__array_namespace__()`, and `core.py` implements them that way. The array API standard requires that method to return "an object representing the namespace"; it does not require the same object on every call, and it takes an `api_version` argument that a backend could legitimately answer with different wrapper objects. NumPy and `array_api_strict` return the module itself, so identity holds there and the assumption is invisible. Does the RFC require namespace *identity*, or a weaker equivalence — and if weaker, what is the portable test, given the standard defines no namespace equality? | **Require identity, state it as a supported-backend constraint, and verify it in CI (§3.3).** What decides it is the **direction of the failure**, not the theory — the standard does not guarantee identity, and no portable surrogate exists, both as this row observed. `is` fails by raising `ValueError` on arrays that legitimately share a namespace: conservative, loud, immediately diagnosable, impossible to mistake for a correct answer. Every available surrogate — `__name__`, a sentinel dtype, anything else the standard leaves — is a weaker test that can *match across genuinely different namespaces*, admitting a JAX/NumPy mix into one diagram, which is the silent direction and defeats exactly what I7 exists to prevent. Given a conservative check with a documented constraint against a clever one that fails open, take the first. §3.3 therefore states plainly that akriti requires `__array_namespace__()` to return a consistent object for a given backend — a constraint on supported backends, not a property claimed of the standard. **Made verified rather than assumed:** a CI test MUST assert identity holds for each supported backend, which is §3.3's own "conformance is tested, not intended" and the same treatment §7's `hasattr(xp, "lexsort")` trap gets — a standing regression test rather than prose. If a backend ever returns a fresh wrapper per call the test fails and D16 reopens as a real decision, instead of breaking silently in someone's pipeline. I7, B5 and §4.2's `from_diagrams` check were unchanged by this row; only their rationale cells moved from citing an open question to citing a requirement. D18 has since respelled I7's and B5's comparison over §3.3's resolution rule rather than over `__array_namespace__` directly, which is a change of mechanism and not of this requirement: what `is` compares moved, what it means did not. |
| **D17** | §8's `DiagramMeta` block annotates `coeff_field` with "affects the diagram, must be recorded", and that comment is the only place the field appears anywhere in this document. The prose immediately below it says the opposite — "All fields are optional", with `from_*` adapters required to populate `backend`, `backend_version` and `provenance` and nothing else — and §8.1's `content_hash` covers bars and never metadata, so no other clause depends on the value being present either. The comment's underlying claim is sound: homology over $\mathbb{Z}/2$ and $\mathbb{Z}/3$ differ wherever there is torsion, so a diagram whose coefficient field is unknown is uninterpretable in the way §8's opening sentence says one whose filtration is unknown is. But the three fields §8 does require are all derivable from the adapter itself, and this one is not — §11's adapters receive a computed result plus `**meta`, not the call that produced it. Does `coeff_field` become a required keyword-only argument on the adapters whose backend takes a coefficient parameter, on the `from_giotto`/`reduced_homology` precedent (§5.1, §11); does §8 require it only where the backend's returned object exposes it; or does the comment's normative clause go, leaving the field optional as the prose already has it? | **Resolved on a fourth option this row did not frame: record the field, do not require it.** `coeff_field` stays optional and no adapter signature changes. §11 requires `from_gudhi` and `from_ripser` to record it in the same construction as the rest of the provenance — the caller's value where one was passed, that backend's documented default (GUDHI 11, Ripser 2) where none was — together with a new reserved `provenance["coeff_field_source"]` of `"caller"` or `"backend_default"` (§8). §8's comment loses its normative clause, so "All fields are optional" is now true as written rather than contradicted seven lines above, and what survives of the obligation is an adapter recording requirement rather than a demand on our callers. **Option 2 was measured out of existence by Appendix A.5**: no backend returns the field it computed with, so "require it where the returned object exposes it" would require it nowhere. **Option 1 is rejected on severity, not on shape.** The `reduced_homology` precedent is exact in shape and the difference in severity is what decides it — `reduced_homology` guards a demonstrated failure, giotto returning 39 H0 bars where GUDHI and Ripser return 40 (A.1, §5.1), while a coefficient field only bites where the data carries torsion, which for the domains this library targets is close to never: dynamical orbits in $\mathbb{R}^{2}$, single-cell blobs and rings in $\mathbb{R}^{3}$, molecular graphs. Torsion in low degrees wants projective planes, Klein bottles, lens spaces. Option 1 breaks three signatures and puts a mandatory argument on every `from_gudhi` and `from_ripser` call to guard something most users cannot reach — friction charged to everyone and repaid to almost nobody. **Option 3 is rejected because A.5 made the comment's underlying claim stronger than it looked**: an unrecorded field is not conventionally $\mathbb{Z}/2$, since the two backends we lean on hardest disagree by default and nothing in the artifact says which produced it. That is a diagram uninterpretable in the way §8's opening sentence describes, not one missing a nice-to-have. **The pattern is §8's own** — `essential_bars` / `essential_bars_source`, applied to the same then-versus-now problem. D15 tested `provenance["order"]` against this shape and correctly failed it, §7 fixing every adapter's answer at `"backend"` so that there was no adapter-time verdict worth a second key; here there is one, because the backend's default is a fact the adapter knows, the caller may not, and A.5 shows nothing recovers it afterwards. **One risk, pinned rather than accepted:** recording a backend default is a claim about a backend, and claims about backends go stale, which is what §9 is a whole section about. §9.3 now requires both defaults to be asserted in CI and §11.2 carries that test, so an upstream change breaks the build instead of silently corrupting the provenance of every diagram recorded afterwards. **Two residual limits, recorded rather than left to be found later.** `"backend_default"` is an assumption and is marked as one: a caller who sets a non-default field on GUDHI and does not pass it on to `from_gudhi` gets a diagram recording 11 (§11), which is a marked assumption where the status quo was a silent one. And `from_giotto` is excluded for now on evidence rather than principle — A.5 could not measure giotto's default (§9.2) and this document does not assert a backend default it has not measured; `from_array` and `from_persim` are out of reach permanently, having no backend and computing no homology respectively. **The condition to reopen against** is where the argument is least confident: it turns on torsion being rare in this library's target data, which is a judgment about users we do not have yet rather than a measurement. If §1's general-purpose framing means the projective-plane user should be assumed to exist, option 1 gets much stronger — three signatures once against a wrong default forever. |
| **D18** | `torch.Tensor` does not implement `__array_namespace__`. array-api-compat's own documentation says so outright — "we do not wrap the `torch.Tensor` object. It is missing the `__array_namespace__` and `to_device` methods" — PyTorch's tracker (gh-58743, open, last touched 2026-03-16) holds the attribute back deliberately as "the attribute that declares compliance", to be added at near-full conformance, and it is absent from the torch 2.13 `Tensor` reference. §3 defined `Array` as any object implementing that method, so **no diagram could be torch-backed**; four sites illustrated a second namespace with torch counterfactually; and D16's CI test was written against "every backend this project supports", of which `akriti[torch]` is one, where it raises `AttributeError` before reaching the identity question. JAX is unaffected — `jax.Array` has implemented the method natively since 0.4.32. Resolving through `array_api_compat.array_namespace` closes the gap. Does that resolver go in front of **(1) every backend**, or **(2) torch alone**, behind `akriti[torch]`, with the native method preferred wherever it exists? | **Option 2: the resolver in front of torch alone, the native method preferred wherever it exists.** §3.3 carries the rule as a single resolution function, and §3's `Array` definition now turns on that rule rather than on the method directly. **What decides it is §10.1 requirement 2, not performance.** Under option 1 `diagrams/core.py` cannot resolve *any* namespace without array-api-compat, so it stops being an extra and `pip install akriti` stops resolving to nothing third-party — undoing the closure D6 was superseded to establish, weeks after establishing it. Under option 2 the import is lazy, function-scoped and reachable only by a caller who passed in an array from a backend they installed themselves, which is the `numpy`-in-`io.py` pattern §3.3 already uses. **The two arguments that look decisive were measured and neither discriminates** (Appendix A.7): array-api-compat ships no JAX wrapper, so JAX pays nothing structurally; NumPy pays one Python frame on the 11 of 26 namespace functions carrying a wrapper and nothing on the other 15, putting §7's `canonical()` at 1.17× at 40 bars and 1.00× from 100k up; and on `numpy` 2.5 the wrappers are largely vestigial, leaving one live correction, the `sort`/`argsort` stable default, which §7 already buys by passing `stable=True` explicitly. **The constraint binding either option** is measured rather than argued: `array_namespace()` on a NumPy array returns `array_api_compat.numpy`, *not* `numpy` (A.7.5), so resolution MUST go through exactly one function — otherwise one backend yields two namespace objects and I7's `is` raises on arrays that legitimately share a namespace, D16's loud failure fired by our own inconsistency rather than a backend's. **A third arrangement, preferring compat wherever it happens to be installed, is excluded outright**: `d.xp` MUST depend on the input and never on the environment. **The fallback is not a lower bar for backends that skip the method** — it supplies a conforming namespace for a backend that conforms in substance without having declared it, and the arrangement expires by itself, since the first branch takes torch the moment the declaration lands. **One risk, pinned rather than accepted.** That landing is expected, not hypothetical, and on that release `d.xp` changes from `array_api_compat.torch` to `torch` on a version bump, so two diagrams built under adjacent torch versions fail `is` while being by any reasonable reading the same kind of thing. §3.3 therefore requires a CI test asserting which branch torch takes, so the release that adds the attribute breaks the build rather than quietly changing what `d.xp` returns — the shape of D16's identity assertion and §9.3's coefficient-field defaults, and the same reasoning, that a fact about a backend is exactly what changes without telling us. **Consequential edits this resolution forces**, none of which option 1 would have: the counterfactual torch illustrations move to JAX at §3.3, §4.2, §6.3 and D16's own cell — five sites rather than the four the open row named, §3.3 restating D16's illustration as well as carrying its own; D16's identity test narrows to backends implementing the method natively, a backend reached through the fallback having no method to call and a namespace that is a module, identical across calls because Python caches imports rather than because the backend promises anything; and §10.1 requirement 2 generalises from one lazily-imported exception to the class of them. **The resolver's reach is bounded, and the bound is where this document lives.** `array_api_compat` returns the backend's own object rather than a wrapped one, and its conformance record against the standard's test suite (`torch-xfails.txt`, A.7.6) lists `__eq__`, `__sub__`, `__getitem__` and masked indexing as failing for torch under the heading "We cannot wrap the tensor object". Those are §3.2's and §4.3's accessors exactly — `essential`, `dim(k)`, `persistence`, `bar_counts`, `finite`. **This row does not claim the divergence misses us.** The plausible argument that it does — I2 and §6.1 fix every operand's dtype, so nothing here has promotion rules to disagree about — is unverified, and the xfails are structural rather than dtype-specific. §3.3 therefore requires a cross-namespace test of the five accessors, on the `stable=True` precedent, and states that until it runs a torch-backed diagram is namespace-correct and not established as object-correct. **The evidence is weaker here than elsewhere in this document and A.7.6 says so**: nothing in that appendix was run against torch. **The condition to reopen against** is a second backend needing the fallback. The argument rests on the cost falling on `akriti[torch]` alone and on the fallback being a shim for one missing declaration; two backends in that state is a different question from this one. |
| **D19** | §9.1 requires `core/distances.py` to compute the essential part of the bottleneck distance itself. `CLAUDE.md` forbids that outright — never reimplement bottleneck or Wasserstein distance, delegate — and §6.3 restates the rule from the other side. Is this an exception, or does the requirement go? | **An exception, named and bounded here so it is not read as drift.** Delegating the whole distance means delegating to a known-wrong answer: persim drops essential bars and returns a finite number for diagrams that are infinitely far apart (§9.1, A.4). What §9.1 requires instead is not a persistence computation but a one-dimensional matching on birth values, whose optimum is a sort. **The exception is bounded to exactly that**: matched essential bars cost the largest birth gap, unequal essential counts per dimension cost `+inf`, and the finite part is still persim's. §6.3's rule is unaffected in both directions — `allclose` implements no distance, and `core/distances.py` MUST NOT be built on `allclose`. **Two obligations ride with it**: the implementation cites the equation `CLAUDE.md` requires of numerical code in `core/`, and its tests are written in a separate session from it. **Reopen when persim handles `inf` correctly** — the issue D5 requires us to file is the same one that would close this row — or if any second formula is ever proposed for `core/` on this row's precedent, which is the drift this row exists to make visible. |

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

The question was D17's: not whether a backend *accepts* a coefficient field,
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

**The defaults also disagree: GUDHI computes over $\mathbb{Z}/11$, Ripser over $\mathbb{Z}/2$.** An
unrecorded `coeff_field` is therefore not conventionally $\mathbb{Z}/2$; it is genuinely
unknown, and two diagrams of the same data from these two backends differ
wherever the data has torsion. This is D17's option 2 — "require it only
where the returned object exposes it" — measured out of existence.

**This table is what D17 resolved against, and two of its columns are now
normative.** The `Default` column supplies the values §11 has `from_gudhi` and
`from_ripser` record when a caller states none, and §9.3 requires both to be
asserted in CI against the installed backend rather than trusted to this run
(§12.2). The `Carried on the returned object?` column is why a second key is
needed at all: a value the adapter cannot read back is one it can only have
been told or have assumed, and `provenance["coeff_field_source"]` is what
distinguishes the two.

The giotto row stays unmeasured, and that has a consequence rather than being
a gap in the record: §11 excludes `from_giotto` from the recording requirement
for exactly this reason, since a default this appendix has not measured is not
one the adapter may assert.

**The default disagreement is a hazard on its own, and §9.3 records it as
one.** It did not wait on D17 and did not depend on how D17 landed: GUDHI and
Ripser compute different homology theories on the same input unless a caller
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

Format comparison at 1M bars, best of 3 loads, seed 0:

| Payload | Size | Load | Exact |
|---|---|---|---|
| `bars.npz` | 20.0 MB | 0.0083 s | yes |
| `bars.csv` | 41.1 MB | 1.2366 s | yes |
| `bars.db` (sqlite3) | 26.5 MB | 0.8245 s | yes |

**Absolute times, and no multiplier.** A ratio here is a quotient by the
fastest thing in the table, so it moves with how the `.npz` baseline is
sampled while nothing about the formats changes; rerunning the script on a
second machine gives byte-identical sizes and a load ratio less than half the
one above. The sizes and the order of magnitude transfer, the multiplier does
not, and §12.2 names this appendix as the thing to reopen D12 against. **The
durable claim is that CSV costs ~2.1× the bytes and takes ~1.2 s where `.npz`
takes ~0.01 s: two orders of magnitude on load.**

Scaled to a realistic batch, CSV is ~190 MB and a few seconds per `load()`
against ~94 MB and a few hundredths of a second. **Correctness does not
discriminate between the three.** The `Exact` column is the script's own
assertion of `float64` round-trip and `inf` preservation, per format and per
run, so a payload that silently lost precision cannot contribute a size and a
time to this comparison; it is checked rather than argued. The decision is
therefore made on cost alone. §10.1 and §12.2 carry the resolution and the one
argument for CSV that survives it.

**Caveat, recorded with the figures rather than below them:** these are two
datasets, both alpha-complex, both low-dimensional, and both the lead's own.
They establish the order of magnitude and the linear-in-point-count scaling,
not a distribution over what users will actually store. D12 is robust to that,
because every direction the bar count moves makes CSV worse.

Reproduction: `rfcs/evidence/bar_counts.py` for the bar-count table, which
runs from the `classify` repository rather than this one, and
`rfcs/evidence/payload_formats.py` for the format comparison.

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

**A.7.6 — Nothing in this appendix was run against torch.** A.7.1's coverage
and A.7.2's timings are NumPy's and A.7.3 is asserted from JAX's own dispatch
rather than timed. Every torch finding in D18 and §3.3 is documentary: the
missing `__array_namespace__` from `array-api-compat`'s documentation and
PyTorch's gh-58743, and the unwrapped-object consequences from
`array-api-compat`'s `torch-xfails.txt`, its own record of what still fails
the standard's conformance suite with the wrapper installed. That file is
better evidence than prose documentation, being the output of a test run
rather than a claim, and it is checkable by anyone; it is still not a run of
ours against the operations this document specifies. The relevant entries:

| Heading in `torch-xfails.txt` | Entries this document reaches |
|---|---|
| We cannot wrap the tensor object | `__eq__`, `__sub__`, `__add__`, `__truediv__`, and `__array_namespace__`/`to_device` as missing methods |
| Indexing does not support negative step | `__getitem__`, `__setitem__` |
| Masking doesn't support 0 dimensions in the mask | masked `__getitem__` |
| These functions do not yet support complex numbers | `unique_values`, `unique_counts` — excluded by I2 and §6.1 |

Closing this gap needs torch installed, which the default and `dev`
environments deliberately do not have (§3.3), so it belongs in the
`@pytest.mark.backend` suite rather than in this appendix.

---

## Appendix B — Changelog

- **2026-07-29** — Initial draft.
- **2026-07-30** — Added §4.1 (two-type design vs. dense padded batch).
- **2026-07-30 (2)** — Added §4.2 (`DiagramBatch` CSR storage); rejected a merged CSR type.
- **2026-07-30 (3)** — Resolved the D1/§10 Parquet contradiction; added `DiagramBatch.from_diagrams`; added batch equality to §6.3; opened D7.
- **2026-07-30 (4)** — Added I8 (immutability) and B1–B5 (`offsets` invariants); added the batch-canonicalization rule to §7.
- **2026-07-30 (5)** — Opened D8 (Parquet export) and D9 (MIT/BSD-only vs. dependency-free).
- **2026-07-30 (6)** — Resolved D9 on a revised project licensing policy retracting "MIT/BSD-only"; corrected §10.1.
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
- **2026-08-07 (42)** — Second review pass on PR #10, acting on the lead's three replies. **Resolves D12, D15 and D16; §12.1 is down to D17 alone.** **D12** — `bars.npz` stays, and §10.2's payload stops being provisional. Resolved on the bar-count figure the row said it turned on, now Appendix A.6: H0 equals the input point count exactly, so bars scale linearly in cloud size and a 5,000-sample batch is ~4.7M bars; at 1M bars CSV costs ~2.1× the bytes and ~78× the load, sqlite3 ~1.3× and ~42×. The reason is *not* that CSV loses on inspectability — it wins there — but that requirement 5 is already satisfied by `meta.json` and §10.3's `to_csv()`, and does not need satisfying a third time. The one surviving argument for CSV (a stdlib payload would drop the `[io]` extra altogether) is recorded in §12.2 as the condition to reopen against, rather than waved away; sqlite3 is closed out. **D15** — `provenance["order"]` is removed. Every other reserved key records a fact that vanishes if unrecorded; canonical order is recoverable from the arrays in one pass, so `order` is a cached answer to an always-computable question and one that goes stale. §8 keeps a short note naming the order fact that is *not* derivable — whether the backend's own output was already canonical — as a different key to build if it is ever wanted, so this is not reopened. **D16** — namespace comparison is identity, stated in §3.3 as a constraint on supported backends rather than a property of the standard, and **verified in CI** rather than assumed. Decided on the direction of the failure: `is` fails loudly on arrays that legitimately share a namespace, while every available surrogate can match across genuinely different ones, which is the silent direction I7 exists to prevent. I7, B5 and §4.2's check are unchanged; only their rationale cells move from citing an open question to citing a requirement. **New §9.3**, at the lead's request and independent of D17: GUDHI defaults to $\mathbb{Z}/11$ and Ripser to $\mathbb{Z}/2$, neither returns the field it used (A.5), so two diagrams of the same cloud from our two primary backends are not diagrams of the same thing — and §6.3's cross-backend `allclose` is where that bites, returning `True` on torsion-free data. §9's preamble now counts three hazards, the third being a disagreement between correct delegates rather than a defect in one. **One addition beyond what was asked**, flagged for the lead to strike: §9.3 puts a MUST on §11.2's cross-backend test to pin the coefficient field on both sides. It constrains a test this RFC already requires rather than the adapter signatures D17 is about, and pinning is a call parameter on both backends, so it costs nothing — but it is new normative text the lead did not request. **New Appendix A.6** carries D12's figures, the lead's own measurements rather than a run of ours, with his caveat that two alpha-complex datasets fix an order of magnitude and not a distribution. `rfcs/evidence/bar_counts.py` is his script, committed verbatim in behaviour; the format-benchmark script was reported without its source and is **not** checked in.
- **2026-08-09 (43)** — Opened **D18** and added **Appendix A.7**, its evidence; §12's count moves to fifteen, two open. No other content changed. `torch.Tensor` does not implement `__array_namespace__` — array-api-compat's documentation says so, PyTorch's tracker holds the attribute back until near-full conformance, and it is absent from the torch 2.13 reference — so §3's definition of `Array` excludes torch tensors and no diagram can currently be torch-backed. D18 asks whether `array_api_compat.array_namespace` goes in front of every backend or torch alone. **This does not reopen D16**, whose identity requirement stands; what is wrong is its reach, since its CI test names a backend on which it raises `AttributeError` before reaching the identity question, and four sites illustrate a second namespace with torch counterfactually. Those four and the test's scope are left as they stand on D17's precedent, editing them in either direction being the answer. `rfcs/evidence/array_api_compat_overhead.py` carries the measurements, which are committed because two arguments that look decisive are not: JAX pays nothing (array-api-compat ships no JAX wrapper) and NumPy pays one Python frame on the 11 of 26 wrapped functions, §7's `canonical()` measuring 1.17x at 40 bars and 1.00x from 100k up; and on `numpy` 2.5 the wrappers are largely vestigial, leaving the `sort`/`argsort` stable default as the one live correction, which §7 already buys explicitly. §10.1 requirement 2 is what decides it. **Appendix A.7** carries the figures, on A.6's precedent that a decision log citing numbers it does not hold is one nobody can check later. Recorded in A.7.4 and the history document, not raised as a row: numpy's `sort`/`argsort` default to quicksort where the standard specifies stable, `array_api_strict` behaves correctly and so cannot catch a call site that omits the keyword, and §7 is therefore correct by discipline rather than construction — entry 20's `lexsort` precedent applies.
- **2026-08-09 (44)** — Third review pass on PR #10, acting on the lead's four comments. **Resolves D17 on a fourth option the row did not frame — record the coefficient field, do not require it.** `coeff_field` stays optional and no adapter signature changes; §11 has `from_gudhi` and `from_ripser` record it in every construction, the caller's value where one was passed and that backend's documented default (GUDHI 11, Ripser 2) where none was, alongside a new reserved `provenance["coeff_field_source"]` of `"caller"` or `"backend_default"` (§8). §8's `coeff_field` comment loses its normative clause, so "All fields are optional" is true as written for the first time since entry 37 opened the contradiction. Option 1 is rejected on severity rather than shape — `reduced_homology` guards a demonstrated H0 loss where a coefficient field bites only on torsion, which this library's target domains do not carry — and option 3 on A.5, which makes an unrecorded field genuinely unknown rather than conventionally $\mathbb{Z}/2$. `DiagramMeta` validates the new key at construction on the `essential_bars_source` pattern; §9.3 requires both backend defaults to be **asserted in CI**, since recording a default is a claim about a backend and claims about backends go stale; §11.2 carries that test and a both-directions test of the recording itself. §12.1 is down to D18 alone. **Two additions beyond what was asked**, flagged for the lead to strike: `from_giotto` is excluded from the recording requirement because A.5 could not measure giotto's default and this document does not assert a backend default it has not measured; and `"backend_default"` is documented as a marked assumption, since a caller who sets a non-default field on the backend and does not pass it to the adapter gets the default recorded. **A.6 stops reporting multipliers.** The 78× and 42× were a quotient by the fastest thing in the table and moved with how the `.npz` baseline was sampled — best-of-3 reads 149× and 99× on the lead's machine and 65× and 46× on a second — so the appendix now records absolute times, the sizes, and "two orders of magnitude", all of which survive re-measurement, and D12's row follows. A.6's table gains the script's own `Exact` column asserting `float64` round-trip and `inf` preservation per format, so correctness is checked in the table rather than asserted in prose. **`rfcs/evidence/payload_formats.py` is checked in**, the lead's script, verbatim in behaviour; every figure in A.6 now re-runs and the format table re-runs from this repository directly. **Three citations to an unpublished internal document are removed** — §3's and §8's, plus `.github/CODEOWNERS` — found by the lead on the grounds that this RFC is about to go out for public comment and `(§2.4)` reads as self-reference until a reader checks. §3's three bullets already carried their own argument; §8's claim is lifted inline onto §8.1 and §10.1 requirement 4; CODEOWNERS states the rule it was citing. A repository-wide grep confirms none remain.
- **2026-08-09 (45)** — Fourth review pass on PR #10, acting on the lead's two comments. **Resolves D18 on option 2 — `array_api_compat.array_namespace` in front of torch alone, behind `akriti[torch]`, with the native method preferred wherever it exists. §12.1 is empty; all fifteen decisions are settled.** §3.3 carries the rule as a single resolution function whose answer depends on the input and never on the environment, and §3 defines `Array` against that rule rather than against `__array_namespace__` directly, so a diagram can be torch-backed for the first time. **§10.1 requirement 2 decides it**: under option 1 `diagrams/core.py` cannot resolve any namespace without array-api-compat, which stops being an extra, and `pip install akriti` stops resolving to nothing third-party — undoing the closure D6 was superseded to establish. Performance and conformance were both measured and neither discriminates (A.7). Preferring the wrapper wherever it happens to be installed is excluded: `array_namespace()` on a NumPy array returns `array_api_compat.numpy`, so a codebase holding both objects fires I7's `is` on arrays that legitimately share a namespace (A.7.5). The fallback is a shim for a missing declaration rather than a lower bar, and it expires by itself when gh-58743 lands. **That landing is pinned rather than awaited**: `d.xp` would change from `array_api_compat.torch` to `torch` on a version bump, so §3.3 requires a CI test asserting which branch torch takes, on §9.3's precedent for facts about backends. D16's identity test narrows to backends implementing the method natively; requirement 2 generalises from one lazily-imported exception to the class of them; and the counterfactual torch illustrations move to JAX at §3.3, §4.2, §6.3 and D16's own cell — **five sites, not the four the open row named**, §3.3 restating D16's illustration as well as carrying its own. **The resolver's reach is bounded in the text rather than left to be discovered**: `array_api_compat` returns the backend's own object, and its `torch-xfails.txt` lists `__eq__`, `__sub__`, `__getitem__` and masked indexing as failing the standard's conformance suite for torch under "We cannot wrap the tensor object" — which is §3.2's and §4.3's accessor surface exactly. §3.3 records the argument that I2 and §6.1 make this harmless, records that it is unverified, and requires a cross-namespace test of the five accessors on the `stable=True` precedent. New **A.7.6** states that nothing in the appendix was run against torch and tabulates the documentary evidence that stands in for it. **I7 and B5 moved too**, both having spelled the comparison as a direct `__array_namespace__()` call, which raises `AttributeError` on a torch tensor before reaching the identity question — D18's own finding about D16's test, sitting in the invariants rather than in a test and absent from the open row's list. **Nine citations to an unpublished internal document are removed** — §4, §5.1, §8.2, §9.1, §9.2, §11.2, §12's preamble, §12.2's D2 and this appendix's entry 6 — entry 44 having raised them and the lead having since confirmed that document does not publish alongside this one. Each took the highest of drop, lift inline, or state the rule that the site allowed; D2's is lifted, being the entire justification for `DiagramBatch` sitting in M1 scope. Six further sites outside this document are fixed in the same push and recorded in the history document. §3.3's "Three limits" count is deleted rather than corrected to a fourth wrong number, this pass having added a paragraph to a section that has carried a stale count since entry 35.
- **2026-08-09 (46)** — Header only. The **Target** row read `M1 (2026-09-15) published for comment`, a date the project's scheduling has since retired: publication and M1 are independent, publishing turning on the spec being right rather than on `diagrams/` being finished, and 15 September leaves eighteen days before the 3 October AMS sessions — too few for maintainers to read, react and comment, on a comment window that is funding-critical rather than a documentation detail. The row now reads `published for comment 2026-08-23, ahead of M1`, decoupling the two dates in the header the way the schedule already decouples them. No normative content changed.
- **2026-08-09 (47)** — Acting on items 1-13 of `RFC-0001-review.md`, all normative defects. Normative: §6.3's `allclose` matches `+inf` deaths outside the tolerance formula; §5's `finitize` rejects any substituted death below an essential bar's birth, in both substituting modes, and `at="max_finite_death"` on an all-essential diagram; B1 is restated against `metas`; **B8** states `DiagramBatch` immutability; **I8** requires public construction not to alias a caller's arrays or `metas`, and documents the subscript assignment nothing in scope can prevent; `from_gudhi` and `from_array` take a keyword-only `dim=`, and `from_array` a `columns=` sequence naming its input's columns; §8 requires `filtration` where the adapter's input form determines it, excludes non-finite floats from `params`/`provenance`, and defines what `same_provenance` compares, `description` excluded; §9.1 carries the essential-bar matching and combination rules; §10.1 requirement 1 binds `DiagramBatch`, its `==` clause scoped to NumPy-backed objects; requirement 2 generalises to any lazily-imported extra and replaces §3.3's rule naming two filenames; §10.2 specifies the `meta.json` schema and a `load` that dispatches on `kind` and rejects an unrecognised `spec`; §10.3 refuses `diagram_id` and moves the exporters to `adapters.py`; §11.2 gains the batch round-trip, `load`-dispatch and `allclose`-on-essential-bars cases. Scope: **extended persistence is a §1 non-goal** — no essential classes, four sub-diagrams, I6 violated only by the relative and extended− bars — with §11 rejecting the 4-tuple and documenting the member that arrives undetected. Opened and resolved **D19**, §9.1's departure from the delegation rule; sixteen decisions, all settled. Also: `DiagramMeta.space` becomes `description`, the header gains a **Version** row, a `dump()` that never existed is spelled `save`/`load`, and mathematical notation is LaTeX throughout. Full narrative: history document.
