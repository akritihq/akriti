# RFC-0002 — CASTLE Inference Tools

| Field | Value |
|---|---|
| **Status** | Draft |
| **Version** | 0.1.0 — `major.minor.patch`, on RFC-0001 §10.2's bump condition |
| **Authors** | Sushovan Majhi |
| **Created** | 2026-09-08 |
| **Last Edited** | 2026-09-08 |
| **Target** | Tool 1 signature frozen 2026-09-14 (D1); Tool 1 live for AMS 2026-10-03 |
| **Implements** | `akriti.castle` |

Key words **MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT**, **MAY** are to be
interpreted as described in BCP 14 (RFC 2119, RFC 8174) when, and only when,
they appear in all capitals.

---

## 1. Why this document exists

`akriti.castle` ships four tools and a reporting card. Its statistical content
is Paper III's; its *library* content is a set of choices Paper III does not
make — which protocol is the default, which feature space, what a result object
returns, and what a caller is entitled to say about the number they got back.

Those choices had a home in a planning note for a paper that no longer exists.
The note's own header now reads "Not maintained". A shipping module whose
specification lives in an abandoned document is the failure RFC-0001 was written
to prevent for `diagrams/`, and this document is the same instrument applied to
`castle/`.

**The immediate reason is sharper than tidiness.** Paper III's final form
withdrew results that an earlier draft of this module was built on, and the
scaffold in #15 still implements the withdrawn version. §9 is the list of claims
that died. It is placed last because it is the section that has to be read
before any body is written, and a reader who stops early should not stop before
it.

### 1.1 Scope

This document specifies the four tools, the protocol they share, the objects
they return, and the claims each field supports. It does not restate Paper III's
theorems; it names which result each obligation rests on, so that a change in
the paper has one place to land.

### 1.2 Non-goals

**Selecting the configuration $\nu$.** Every tool here takes $\nu$ as fixed on a
pilot split. Choosing it well is an open research direction and not a library
feature; §2.2 states why fitting it on the inference sample is prohibited rather
than discouraged.

**Covariates and $k$-sample designs.** Two groups, no adjustment. A third group
is not a loop over pairs, and saying so here is cheaper than withdrawing it
later.

**Reimplementing persistence or distances.** RFC-0001 §9's delegation rule holds
unchanged. Diagrams arrive as `DiagramBatch`; distances, where needed, go
through `core/distances.py`.

---

## 2. The protocol every tool shares

Paper III's template, and the thing a caller most often gets wrong.

1. Fix $\nu$ on a **pilot split**.
2. Embed the **inference split** with the raw additive $\Phi(\cdot;\nu)$.
3. Do the statistics in Hilbert space.
4. Translate to diagram space **only** through the transfer interface.

### 2.1 The split is the default protocol (D1)

Every tool **MUST** accept diagrams already partitioned into a pilot and an
inference split, or perform the partition itself from a caller-supplied
fraction and seed. The split is the only protocol under which every line of the
reporting card is simultaneously valid, and the pilot is the same sample Tool 2
needs anyway.

**Refit-within-permutation is the small-sample fallback** and **MUST** return a
$p$-value only. A result object produced under it **MUST NOT** carry an effect
size, a certificate, or a sample-size number: those rest on $\nu$ being fixed
independently of the data they are computed from, and the fallback breaks
exactly that.

### 2.2 $\nu$ **MUST NOT** be fitted on the inference sample

Not a preference. For any bounded feature map the plug-in
$\|\hat\mu_+-\hat\mu_-\|^2$ carries bias
$\operatorname{tr}\Sigma_+/n_+ + \operatorname{tr}\Sigma_-/n_-$, so a
configuration chosen to maximise separation on the inference sample is chosen
partly for its ambient scale. The published selector inversions on $K$ and
$\sigma$ are that bias, observed.

An implementation **MUST** raise rather than warn when asked to do it. A warning
is the wrong instrument: the result is not degraded, it is invalid, and a
caller who filters warnings gets a number with no property at all.

### 2.3 Raw $\Phi$ is the default feature space (D2)

The raw additive map is the default. The Gaussian gram remains available and,
when selected, the result object **MUST** record that no geometric certificate
is available under it — the transfer interface is stated for the additive map.

---

## 3. Tool 1 — two-sample test

### 3.1 Signature

```python
def two_sample(
    group_a: DiagramBatch,
    group_b: DiagramBatch,
    *,
    alpha: float = 0.05,
    nu: Configuration | None = None,      # fitted on the pilot if None
    pilot_fraction: float = 1 / 3,
    feature_space: Literal["raw", "gram"] = "raw",
    calibration: Literal["both", "spectral", "permutation"] = "both",
    n_permutations: int = 1000,
    seed: int | None = None,
) -> TwoSampleResult: ...
```

### 3.2 The statistic

$T_{m,m'} = N_{\mathrm{eff}}\,\|\hat\delta_{m,m'}\|_2^2$ on the inference split,
with $N_{\mathrm{eff}} = mm'/(m+m')$.

### 3.3 Calibration — both, and both reported

**Spectral**: the quantile of $\sum_j \hat\gamma_j Z_j^2$, resting on Paper III's
null distribution for the mean-embedding distance together with its covariance
consistency result.

**Permutation**: shuffle the diagram labels of the inference split with $\nu$
held fixed. Exact under $P = Q$.

The default reports **both**, and a conforming implementation **MUST** report
both when `calibration="both"` rather than choosing between them. They answer
slightly different questions and their disagreement is information: the
permutation $p$ is exact under the sharp null, the spectral one is asymptotic
and survives unequal covariances.

### 3.4 The primary output is the effect-size interval, not the $p$-value

The $(1-\alpha)$ interval $[\underline\Delta, \overline\Delta]$ for
$\Delta_\nu(P,Q)$, from Paper III's two-sample confidence ball. **This is the
primary inferential output** in the paper's own words, and the API **SHOULD**
present it that way: a caller who reads only the first field of the result
should be reading the interval.

### 3.5 The certificate, and what it is a statement about

$$W_{\infty,0}(\bar\mu_P, \bar\mu_Q) \;\ge\; \underline\Delta / (n L_\nu)$$

with $L_\nu$ the point-level Lipschitz constant of the PALACE map and $n$ the
cardinality bound of $\mathcal{D}_n$.

Three things this is **not**, each of which an earlier draft claimed:

- It is **not** a bound on the bottleneck distance between the two
  distributions. The transport quantity is $W_{\infty,0}$ between **padded mean
  measures**.
- It is **not** a statement about exemplar diagrams. There are no two diagrams
  it says are far apart.
- It is **not** $T/L_\Phi$. That form rested on a population converse to the
  lower-distortion floor, and no such converse exists.

The field carrying it **MUST** be named for the quantity it bounds rather than
for "bottleneck", and its docstring **MUST** state that it is a confidence
statement about mean measures.

**Whether it appears on the reporting card is D3 and is still open**: if its
magnitude remains at the $10^{-4}$ scale it stays in the documentation as a
qualitative, sign-only guarantee and comes off the card. A certificate a
practitioner cannot act on is worse on a card than absent, because a card
implies every line was worth printing.

### 3.6 The estimand line is mandatory

Every `TwoSampleResult` **MUST** carry, and every rendering of it **MUST**
display:

> tests equality of mean embeddings under $\nu$; not an omnibus test

A rejection means the mean embeddings differ under this configuration. It does
not mean the distributions differ, and the distance between those two readings
is where a practitioner will over-claim if the object lets them.

Consistency against any mean-embedding alternative follows from Paper III's
mean-distance consistency result. **The word "minimax" MUST NOT appear**: see
§9.

---

## 4. Tool 2 — sample-size calculator

Two numbers, both from one pilot embedding, **both returned and both labelled**.
Returning one is the failure mode this tool exists to avoid.

**Guarantee (conservative).**
$N_{\mathrm{eff}} \ge \hat v^2 \Delta^{-2}(\alpha^{-1/2} + \beta^{-1/2})^2$ with
$\hat v^2 = \operatorname{tr}\hat\Sigma_{\mathrm{pool}}$. Valid, and loose by the
Chebyshev-type constants.

**Planning number (oracle).**
$N_{\mathrm{eff}} \approx (z_\alpha + z_\beta)^2\,\hat\omega_h^2/\Delta^2$. It
presumes the direction is known or pilot-estimated, and the docstring **MUST**
say so. One $z$ convention **MUST** be fixed and named in the docstring; the
existing script uses the two-sided $z_{\alpha/2}$.

$\Delta$ enters as a pilot estimate or a prespecified target in embedding units.
A pilot estimate **MUST** use the unbiased $\hat\Delta^2_U$: the biased plug-in
carries $\operatorname{tr}\Sigma_c/n_c$ per class, which is §2.2's bias arriving
by a second route.

The post-hoc inversion — *was this study powered?* — is the same formula and
**SHOULD** be exposed, since it is what a case study actually asks.

---

## 5. Tool 3 — per-region significance map

Per-coordinate two-sample $z$-tests on the raw $\Phi$ coordinates of the
inference split, Benjamini–Hochberg across the $K$ landmarks. The theory is
Paper III's finite-coordinate Berry–Esseen result applied coordinate-wise, or
permutation of diagram labels with $\nu$ fixed.

**Attribution of a flagged coordinate to a birth–death region requires
near-disjoint landmark supports.** That is a condition on the FPS placement and
**MUST** be reported as a condition rather than as a certificate. Where supports
overlap materially, the map localises to a union of regions and the result
object **MUST** say which.

A simultaneous band over all coordinates is a v2 feature and is out of scope
here.

---

## 6. Tool 4 — robustness certificate

$\epsilon_{\max} = \Delta T / (L_\nu L^{\mathrm{filt}})$, with $\Delta T$ the
margin of $T_{m,m'}$ over the calibrated threshold.

**The soundness statement is conditional and MUST be reported as such.**
Measured survival is $0.9997$ above the median margin and $0.812$ in the lowest
decile, the failures being permutation-null boundary effects. An implementation
**MUST** return the margin alongside the certificate, because the certificate's
reliability is a function of it and a caller cannot otherwise tell which regime
they are in.

---

## 7. The reporting card

Five lines, every one computed. No line **MAY** be rendered from a stored
constant or an example value.

1. Estimand and configuration — mean embeddings under $\nu$; pilot size; $K$
2. $p$, spectral and permutation, at level $\alpha$
3. $\Delta$ interval $[\underline\Delta, \overline\Delta]$; the $W_{\infty,0}$
   certificate if non-vacuous (D3)
4. $n$ per group against Tool 2's guarantee and planning number
5. $\epsilon_{\max}$ with its margin

---

## 8. Relationship to RFC-0001

Diagrams enter as `DiagramBatch` (RFC-0001 §4). This module is a **caller** of
the interchange layer and adds no requirement to it, with one exception already
recorded there: RFC-0001 §9.1 binds `core/distances.py`, which this module uses
and does not implement.

`castle/` is NumPy-backed by the dated deviation of 2026-08-09; `diagrams/`
remains array-API-pure. Nothing here relaxes RFC-0001 §3.3.

---

## 9. Claims that did not survive Paper III's final form

The section to read before writing any body. Each row is a claim an earlier
draft made, and an implementation that reproduces any of them is wrong in a way
tests will not catch — the code runs, the number is plausible, and the sentence
attached to it is false.

| # | The withdrawn claim | What replaces it |
|---|---|---|
| 1 | Tool 1 certifies $d_B(P,Q) \ge T/L_\Phi$, "in the sense that there exist exemplar diagrams" | No such object. Population transfer goes through the **upper** bound only, and the lower-distortion floor gives no population converse. The certified output is a lower **confidence** bound on $W_{\infty,0}$ between padded mean measures, $\underline\Delta/(nL_\nu)$ |
| 2 | The test is minimax-optimal; "matching minimax lower bound" | Optimality over $\tau$-separated laws is **withdrawn** — no test has uniform power there. What survives is a conditional rate over structured classes. The honest line is consistency against any mean-embedding alternative, with the $\Delta^{-2}$ sample-size rate |
| 3 | A bottleneck-distance guarantee between distributions | A transport statement between mean measures, at confidence $1-\alpha$ |

**The naming consequence.** A field called `certified_bottleneck` is wrong even
if its arithmetic is right, because the name is the claim for every caller who
does not read the docstring. §3.5 names the field for the quantity.

---

## 10. Open decisions

| # | Decision | Needed by | Position |
|---|---|---|---|
| D1 | Split as the default protocol | **2026-09-14**, Tool 1 signature freeze | **Resolved: split.** §2.1 |
| D2 | Raw $\Phi$ or Gaussian gram as the default | after E0 | **Raw**, unless E0 shows a power loss the certificate does not buy back |
| D3 | Does the $W_{\infty,0}$ line stay on the card? | after E0 | Off the card if its magnitude remains unusable; documentation keeps it as a sign-only guarantee |
| D4 | Package identity | resolved | `akriti.castle`, Apache-2.0 |
| D5 | Does Tool 1 ship the fixed-direction $Z_{m,m'}(h)$ with $h$ pilot-estimated? | Tool 1 freeze | Include as an option, not a default. Cheap, and the natural directional mode once a pilot exists |
| D6 | What a result object does when the certificate is vacuous | Tool 1 freeze | Open. Returning a number a caller must know to ignore is the failure mode §9 is about |

---

## 11. What is measured and what is not

Stated in RFC-0001 §9's spirit: a support claim no test runs is a claim that
rots silently.

| Component | Evidence today |
|---|---|
| Tool 1, permutation calibration | Power results exist **for the Gaussian gram only** |
| Tool 1, raw map | **Nothing.** No power results, no interval coverage, no spectral calibration |
| Tool 2 | Validation requires families where $\Delta$ is known in closed form; the current synthetic estimates $\Delta$ and cannot test the claim |
| Tool 3 | Power $1.0$, FDR $0.03$–$0.04$ at $q=0.05$, median localisation error $0.042$ at tolerance $0.15$. **Missing:** a $\delta=0$ run under the global null, and the re-run under the split protocol |
| Tool 4 | $1{,}529$ certified rejections; conditional survival as in §6. Done modulo the re-run |

**The gap that matters is Tool 1's.** The default feature space is the one with
no evidence behind it, and the interval — the primary output — has never had its
coverage measured. Both are prerequisites to a release that makes the claim, and
neither is a prerequisite to freezing the signature.
