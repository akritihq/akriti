# TODO

Outstanding obligations that are not yet code and not yet issues.

Notes here are meant to be short-lived. Anything that belongs in a
specification goes in the relevant RFC instead; anything already tracked as an
open decision (RFC-0001 §12) stays there. Each entry names the requirement,
where it is written down, and what would close it.

## Closed

Both entries this file opened are now closed; kept for one revision so the
claim they made can be checked against what actually landed, then delete.

**`diagrams/core.py` is never exercised under `array_api_strict`.**
`tests/test_array_api_conformance.py` now builds a `PersistenceDiagram` and a
`DiagramBatch` out of strict arrays and drives them through construct,
validate, slice, canonicalise, compare and hash. The four specific gaps it
named are covered: `_big_endian_block`'s fallback (both through strict arrays
and directly, via a stand-in that refuses the buffer protocol), `births + 0.0`
and the other array-plus-Python-scalar operations, `from_diagrams`'
empty-batch path including the `int32`/`int64` dtypes it constructs itself,
and `float()`/`int()` on the 0-d arrays the fallback indexes out.

**`content_hash` must agree across array namespaces.** Closed by
`tests/test_rfc0001_content_hash.py`, with the fixture this file asked for:
`-0.0`, `+inf`, an empty diagram, and repeated identical bars, the last
asserted to change the digest so multiplicity is shown to reach it. The
byte-level two-path agreement test deliberately does not `importorskip`
anything — it runs in the default environment, since the buffer path is the
one `array_api_strict` structurally cannot reach and skipping it there would
restore exactly the hole this entry described. RFC-0001 §11.2 now requires
both paths to be tested, so the obligation lives in the spec rather than here.

## The conformance module still skips as one unit

*`tests/test_array_api_conformance.py` — needs a CI guarantee, not a test.*

The entry above noted in passing that the whole module `importorskip`s on
`array_api_strict` and that this is "quiet enough to read past". That is still
true, and now matters more: the module is no longer a handful of facts about
the standard, it is where the diagram types get their only non-NumPy coverage.
A CI job that silently skipped it would leave §3.3's enforcement mechanism
running on nothing while reporting green.

`array_api_strict` is in `akriti[test]`, so the test job installs it and the
skip should never fire there. Nothing checks that. To close: fail CI if the
conformance module skips — a `--strict-markers`-style guard, an explicit
`-p no:cacheprovider` run that asserts a nonzero collected count, or simply
importing `array_api_strict` unconditionally in a CI-only conftest so its
absence is an error rather than a skip.

## `allclose`'s matching has no test, and must not get one from this session

*`src/akriti/diagrams/core.py` — RFC-0001 §6.3, D14, §11.2.*

`PersistenceDiagram.allclose` was rewritten from a sorted pairwise comparison
into a matching over the multiset in the same commit that opened this entry.
**It is untested.** The existing suite exercises `allclose` only through
`tests/test_array_api_conformance.py`'s single `a.allclose(b)`, which passes
identically against both implementations and therefore tests none of what
changed.

Unwritten deliberately, not overlooked: CLAUDE.md forbids writing a numerical
function and its test in the same session, because a test written by whoever
just wrote the function blesses the function's bugs. That applies with unusual
force here — the rejected implementation was plausible enough to ship and
survive review, so a test author reasoning from the same assumptions as the
implementer would very likely write assertions both versions satisfy.

The requirement is in the specification rather than here: §11.2 requires a
case where two diagrams are within tolerance of each other but their canonical
orders differ *because* of that tolerance, asserting `True`. §6.3 carries the
three properties that need separate coverage — the bijection, the symmetric
tolerance diverging from `numpy.allclose`, and that the relation is reflexive
and symmetric but **not transitive**, so not an equivalence relation.

To close: a separate session, deriving the cases from §6.3 and §11.2 rather
than from `core.py`. One check worth building in whatever gets written — the
motivating case is only a test of anything while the two canonical orders
genuinely differ, so that premise should be asserted rather than assumed, or
a later change to §7's sort key would leave it green and empty.

## `probe_backends.py` has no `reduced_homology=False` row

*`rfcs/evidence/` — RFC-0001 Appendix A.1.*

Appendix A.1 varies `infinity_values` across three settings and holds
`reduced_homology` fixed at `True`, so it establishes that `infinity_values`
is not the cause of giotto's H0 loss and leaves `reduced_homology` as an
inference. §5.1's adapter requirement rests on that inference.

Blocked rather than merely unwritten: giotto-tda 0.6.2 does not run on current
scikit-learn (§9.2), so the row cannot be produced by a live call in this
environment. To close: capture it in a pinned environment and commit it as a
frozen fixture, which §11.2 accepts as real backend output, then add the row
to the appendix table.

## `io.py` does not exist

*RFC-0001 §10 — `save`/`load`, `.akd`, and the byte-determinism test.*

§10.2 specifies the format and §11.2 requires a byte-determinism test
(dumping twice gives identical bytes), which §10.1 requirement 4 notes is not
satisfied for free by any candidate: zip entry metadata has to be pinned
explicitly. Nothing is implemented yet.

Two constraints to carry in when it is.

**The lazy import checks the version, not just presence, and both failure
paths name the extra.** `numpy` is imported lazily and function-scoped inside
`save`/`load` only (§3.3). It is now declared — `akriti[io]`, resolving to
`akriti[numpy]` at `numpy>=2.0` (D6, `DEPENDENCIES.md`) — and the floor is the
point: an import guarded on presence alone fires when numpy is absent and
stays silent when it is present but too old, which turns a resolver error at
install time into an `AttributeError` at run time. So `save`/`load` must raise
a clear `ImportError` in *both* cases, and both messages must say "install
`akriti[io]`" rather than "install numpy" — the second is not an instruction a
user who already has numpy 1.24 can act on.

**Requirement 1 has two clauses**, `load(dump(d)) == d` *and*
`load(dump(d)).same_provenance(d)`, because `==` excludes metadata by §8 and
the first clause alone is satisfied by a `load` that discards provenance
entirely.
