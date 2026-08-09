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

## `core.py` predates D17 and D18

*`src/akriti/diagrams/core.py`, `pyproject.toml` — RFC-0001 §3.3, §8, §12.2.*

Both decisions resolved in #10 after this implementation was written, so the
spec moved underneath it. D18's half is still unreachable — no diagram can
currently be torch-backed — but **D17's half is now live**: `adapters.py`
exists and writes `provenance["coeff_field_source"]` on every `from_gudhi`
and `from_ripser` call, so the missing validation below is a reserved key
nothing checks rather than a key nothing writes.

**Namespace resolution goes through one function (D18, §3.3).** §3.3 requires
`namespace_of(x)` — the native `__array_namespace__` where it exists, and
`array_api_compat.array_namespace` where it does not, which today is torch
alone. `core.py` calls `__array_namespace__()` directly at five sites:
`_validate_bar_arrays` (I7, and the `xp` it returns),
`PersistenceDiagram.xp`, `DiagramBatch.__post_init__` (B5), and
`DiagramBatch.xp`. `adapters.py` adds a sixth, `_namespace_of`, written that
way deliberately and for this entry's own reason: an adapter resolving
through the compat shim would hand `core.py` a diagram whose namespace
`core.py` cannot re-derive, so the two have to move together rather than one
first. That spelling is exactly what D18 identified as broken —
it raises `AttributeError` on a torch tensor before reaching the identity
question I7 and B5 exist to ask. To close: add the resolver, route all five
through it, and keep it the only caller, since resolving two ways yields
`array_api_compat.numpy` alongside `numpy` for one backend and fires I7's
`is` on arrays that legitimately agree (A.7.5).

**`array-api-compat` is undeclared.** §3.3 requires it in the `akriti[torch]`
extra with a version floor, on §10.1 requirement 2's terms — lazy,
function-scoped, unreachable on the default install. The extra is still
`torch = ["torch>=2.0"]` and the string occurs nowhere in the repository.

**`coeff_field_source` is unvalidated (D17, §8).** `DiagramMeta` must raise
`ValueError` when it holds anything but `"caller"` or `"backend_default"`,
and when it is present while `coeff_field` is `None` — a source describing no
value being incoherent rather than merely weak. `_validate_provenance` covers
`essential_bars_dropped` and `essential_bars_source` and has no branch for
the new key. The adapter half of D17 is done —
`tests/test_rfc0001_adapters.py` covers both directions on both backends, and
`tests/test_rfc0001_adapters_live.py` asserts the two defaults against the
installed backend as §9.3 requires — so what remains is the core half alone,
deliberately left for a separate change.

**Three CI tests §3.3 now requires and nothing provides:** which resolution
branch a `torch.Tensor` takes, marked on the `akriti[torch]` extra, so that
the release closing gh-58743 breaks the build rather than quietly changing
what `d.xp` returns; namespace identity across two arrays of each natively
implementing backend (D16, narrowed by D18 to those backends); and a
cross-namespace check that `essential`, `persistence`, `bar_counts` and
`dim(k)` agree, since those accessors are built from operators the resolver
does not reach and are safe only because I2 and §6.1 fix every operand's
dtype.

## `adapters.py` carries a lazy import §3.3 does not admit

*`src/akriti/diagrams/adapters.py`, RFC-0001 §3.3 — the RFC is the side that
needs the edit.*

§3.3 requires `diagrams/core.py` and `diagrams/adapters.py` to "import nothing
beyond the standard library, save for the single lazy import §3.3's namespace
resolution rule carries". `adapters.py` has a second one:
`_namespace_for_rows` imports numpy, function-scoped, when the input carries
no array to derive a namespace from.

**The code is right and the RFC's clause is too narrow**, for a reason §11
creates itself. §11 fixes the adapter signatures with no namespace argument,
and GUDHI's primary form — `SimplexTree.persistence()` — is a Python list of
tuples with no array anywhere in it. Something has to supply a namespace, and
the three candidates are: a namespace keyword on the adapters, which §11's
signatures rule out; refusing list input, which §11's own table requires be
accepted; or a lazy numpy import, which is what §3.3's existing exception
already does for `io.py` and which is reachable only through an input shaped
that way. The last was chosen deliberately, with the trade recorded here: a
JAX or torch user who passes a Python list gets a numpy-backed diagram rather
than an error telling them to pass an array. `from_array` and every array
form are unaffected — those preserve the caller's namespace exactly, as
§3.3's "adapters preserve the input namespace" requires.

To close: amend §3.3's clause to describe the class of lazily-imported
exceptions rather than "the single lazy import", which §10.1 requirement 2
already generalised to for D18's sake, and state the list-input case as the
second member. Until that lands, code and specification disagree in text
while agreeing in substance, and this entry is the record of which is which.

The version is checked rather than merely presence, on §3.3's own terms for
`io.py`: numpy below 2.0 has no main-namespace array API, so both failure
paths raise `ImportError` naming `akriti[numpy]` rather than proceeding into
an `AttributeError`.

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
