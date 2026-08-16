# TODO

Outstanding obligations that are not yet code and not yet issues.

Notes here are meant to be short-lived. Anything that belongs in a
specification goes in the relevant RFC instead; anything already tracked as an
open decision (RFC-0001 §12) stays there. Each entry names the requirement,
where it is written down, and what would close it.

## Closed

Kept for one revision each, so the claim an entry made can be checked against
what actually landed, then deleted.

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

**`allclose`'s matching has no test.** Closed by
`tests/test_rfc0001_allclose.py`, in a separate session from the commit that
wrote the matching and derived from §6.3 and §11.2 rather than from `core.py`:
the relation is checked against a brute-force `itertools.permutations` oracle
transcribed from §6.3's own sentence, so the test agrees with the definition
rather than with the optimisation of it. The check this entry asked to be
built is there — §11.2's motivating case asserts that the two canonical orders
genuinely differ before asserting `True`, so a later change to §7's sort key
fails it rather than leaving it green and empty. §6.3's three properties each
have their own coverage: the bijection, including the case where every bar on
both sides has a candidate and no matching exists; the symmetric tolerance, on
a pair where `numpy.allclose` contradicts itself under argument order; and
non-transitivity. The generator behind the property tests was tuned by
mutating `core.py` until each mutation was caught, and the measurements that
drove that are recorded in the file, since the tuning is what makes those
tests worth their runtime.

**`core.py` predates D17 and D18.** All five parts landed. `namespace_of` is
the single resolver and the only caller of `__array_namespace__`; the last
direct call was `adapters._namespace_for_rows`, on its own NumPy probe, which
now routes through the resolver like everything else — a second spelling
agreeing with the resolver today is exactly what §3.3's "exactly one function"
forbids, since `array_api_compat.numpy` and `numpy` are the two objects I7's
`is` would then raise between (A.7.5). `array-api-compat>=1.15.0` is declared
on `akriti[torch]`. `DiagramMeta`
validates `coeff_field_source` against §8's two values and against a `None`
`coeff_field`, and now type-checks the five scalar fields §8 declares, so
`filtration=3.5` and `description=42` are refused at the type rather than at
`save()`; `adapters._require_coeff_field` still widens to `numbers.Integral`
at the adapter boundary and converts, which is where a caller's array scalar
actually arrives. The three CI tests are in
`tests/test_rfc0001_torch_live.py` and `tests/test_array_api_conformance.py`;
D16's identity assertion now runs on **two separate arrays** per natively
implementing backend, numpy and `array_api_strict`, since the constraint §3.3
states is call-to-call consistency and a single array proves only that one
call returns the module.

**`adapters.py` carries a lazy import §3.3 does not admit.** The RFC was the
side that needed the edit and got it: §3.3 now sets the closure over what a
caller can reach rather than over which files may import what, and names the
row-sequence NumPy fallback as one of four cases that meet it, alongside the
namespace resolution rule, `save`/`load`, and §10.3's `to_parquet`.

**§11's signature block omits two arguments it requires.** Closed by the RFC,
which now reads `from_gudhi(obj, *, dim=None, **meta)`,
`from_giotto(arr, *, reduced_homology, strip_padding=None, **meta)` and
`from_array(arr, *, columns=None, dim=None, **meta)`, and specifies `columns=`
against §10.3 in the same pass. Code and specification agree, and
`adapters.py`'s remaining use of "deviation" is §11's own — the six across
three adapters that depart from the common signature, not a divergence
between the two documents.

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

**The measurement now exists; the appendix row does not.** The adapter branch
needed real giotto output and hit the same wall, so
`tools/capture_giotto_fixture.py` builds the pinned environment the paragraph
above describes — giotto-tda 0.6.2 on scikit-learn 1.3.2, numpy 1.26.4,
CPython 3.11 — and `tests/fixtures/giotto_output.json` is what it captured.
That file carries the same 40-point circle at `reduced_homology=True` **and**
`False`: 39 H0 rows against 40, zero non-finite entries either way. The
inference §5.1 rests on is therefore measured rather than inferred, on this
project's own machine, with the environment recorded in the fixture.

Two things remain, and neither is an adapter's to do. **A.1's table has no
row for it** — adding one is an edit to a document under public comment, so
it goes through whoever owns the RFC, not through a branch that happened to
need the number. And **the `infinity_values` sweep is not repeated at
`reduced_homology=False`**: the capture varies one flag, so it shows the H0
loss tracking `reduced_homology` and does not re-establish that
`infinity_values` is inert on the other branch. A.1 already shows the latter
at `True`, and nothing in §5.1 turns on the cross, which is why this is a
note rather than a second blocked entry.

## `.akd` I/O (implemented)

*RFC-0001 §10 — `save`/`load`, `.akd`, and the byte-determinism test.*

§10.2 specifies the format and §11.2 requires a byte-determinism test
(dumping twice gives identical bytes), which §10.1 requirement 4 notes is not
satisfied for free by any candidate: zip entry metadata has to be pinned
explicitly. `src/akriti/diagrams/io.py` now implements this contract.

Two implementation constraints are now enforced.

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
