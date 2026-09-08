# RFC-0002 — CASTLE Inference Tools

| Field | Value |
|---|---|
| **Status** | Draft |
| **Version** | 0.2.0 — `major.minor.patch`, on RFC-0001 §10.2's bump condition |
| **Authors** | Sushovan Majhi |
| **Created** | 2026-09-08 |
| **Last Edited** | 2026-09-08 |
| **Target** | Tool 1 signature frozen 2026-09-14 (D1); Tool 1 live for AMS 2026-10-03 |
| **Implements** | `akriti.castle` |
| **Rests on** | Paper III, arXiv version. Results are cited by label; numbers are not quoted because they move with every edit |

Key words **MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT**, **MAY** are to be
interpreted as described in BCP 14 (RFC 2119, RFC 8174) when, and only when,
they appear in all capitals.

---

## 1. Why this document exists

`akriti.castle` ships four tools and a reporting card. Its statistical content
is Paper III's; its *library* content is a set of choices Paper III does not
make — which protocol is the default, which calibration, what a result object
returns, and what a caller is entitled to say about the number they got back.

Those choices had a home in a planning note for a paper that no longer exists,
and that note's own header now reads "Not maintained". A shipping module whose
specification lives in an abandoned document is the failure RFC-0001 was written
to prevent for `diagrams/`.

**This revision is written against Paper III's arXiv version rather than against
the note.** That changed the specification in both directions: §9 lists claims
that did not survive, and §3.7 and §4.2 record two results the note treated as
lost that are in the paper and are usable.

### 1.1 Scope

The four tools, the protocol they share, the objects they return, and the claims
each field supports. This document does not restate Paper III's theorems; it
names which result each obligation rests on, so a change in the paper has one
place to land.

### 1.2 Non-goals

**Selecting the configuration $\nu$.** Every tool takes $\nu$ as fixed on a
pilot split. §2.2 states why fitting it on the inference sample is prohibited
rather than discouraged.

**Covariates and $k$-sample designs.** Two groups, no adjustment.

**Reimplementing persistence or distances.** RFC-0001 §9's delegation rule holds
unchanged.

---

## 2. The protocol every tool shares

1. Fix $\nu$ on a **pilot split**.
2. Embed the **inference split** with the additive $\Phi(\cdot;\nu)$, truncated
   at $K$ coordinates.
3. Do the statistics in Hilbert space.
4. Translate to diagram space **only** through `prop:mean-embedding-transfer`.

### 2.1 The estimand is $\Delta_\nu^{\Psi}$, and the certificate is about $\Delta_\nu$

This distinction is the one most likely to be lost in implementation, and it is
not cosmetic.

The confidence interval of `cor:two-sample-confidence-ball` is an interval for
$\Delta_\nu^{\Psi}(P,Q)$ — the separation **in the truncated coordinate system
actually computed**. The transport certificate of
`prop:mean-embedding-transfer` is stated for $\Delta_\nu$. Truncation error
enters explicitly as $\epsilon_K$ (`thm:finite-approximation-three-way`), and a
result object **MUST** carry $K$ so that a reader can tell which estimand a
number refers to.

A conforming implementation **MUST NOT** describe the interval as an interval
for "the separation between the two populations" without qualification. It is
the separation their mean embeddings exhibit under this configuration at this
truncation.

### 2.2 The split is the default protocol (D1)

Every tool **MUST** accept diagrams already partitioned into a pilot and an
inference split, or perform the partition itself from a caller-supplied fraction
and seed.

**Refit-within-permutation is the small-sample fallback** and **MUST** return a
$p$-value only. A result produced under it **MUST NOT** carry an effect size, a
certificate, or a sample-size number: those rest on $\nu$ being fixed
independently of the data they are computed from.

### 2.3 $\nu$ **MUST NOT** be fitted on the inference sample

For any bounded feature map the plug-in $\|\hat\mu_+-\hat\mu_-\|^2$ carries bias
$\operatorname{tr}\Sigma_+/n_+ + \operatorname{tr}\Sigma_-/n_-$, so a
configuration chosen to maximise separation on the inference sample is chosen
partly for its ambient scale.

An implementation **MUST** raise rather than warn. The result is not degraded,
it is invalid, and a caller who filters warnings gets a number with no property
at all.

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
    truncation: int | None = None,        # K; selected on the pilot if None
    pilot_fraction: float = 1 / 3,
    calibration: Literal["spectral", "permutation", "chebyshev", "all"] = "all",
    n_permutations: int = 1000,
    seed: int | None = None,
) -> TwoSampleResult: ...
```

### 3.2 The statistic

$\|\hat\delta_{m,m'}\|$ on the inference split, with
$N_{\mathrm{eff}} = mm'/(m+m')$.

### 3.3 Three calibrations, and one of them has a caveat that **MUST** be surfaced

**Spectral** — the quantile of the weighted chi-square limit
(`thm:mean-distance-null`, with `prop:covariance-consistency` for the estimated
weights). Asymptotic; valid under the mean null.

**Chebyshev / plug-in** — $\phi_{\alpha,v}$ rejects when
$\|\hat\delta_{m,m'}\| > v/\sqrt{\alpha N_{\mathrm{eff}}}$
(`prop:finite-sample-power`). Level at most $\alpha$ under
$H_0^{\mathrm{mean}}$, non-asymptotically, using only a second-moment bound. It
is conservative by construction and it is the only one of the three with a
finite-sample level guarantee.

**Permutation** — shuffle diagram labels of the inference split with $\nu$ held
fixed.

> **Ordinary permutation calibration does not generally control level under the
> weaker null of equal mean embeddings.** It is exact under the sharp null
> $P = Q$ and is not, in general, valid under $H_0^{\mathrm{mean}}$.

That sentence is Paper III's, and it is the single most important thing this
module can get wrong. The estimand of the whole tool is a mean embedding, so the
null a user cares about is usually the weaker one, and the calibration most
users will reach for is the one that does not cover it. A `TwoSampleResult`
**MUST** record which null each reported $p$-value is valid under, and a
rendering that prints a permutation $p$ without that qualification is
non-conforming.

### 3.4 The interval is the primary output

$[\underline\Delta_{m,m'}^{\alpha}, \overline\Delta_{m,m'}^{\alpha}]$ for
$\Delta_\nu^{\Psi}(P,Q)$, from `cor:two-sample-confidence-ball`:
asymptotically valid and **potentially conservative**, which the docstring
**MUST** say. Paper III calls it the primary inferential output where the
structured constants are unavailable, and the API **SHOULD** present it that
way.

The two endpoints answer different questions and both are reported: the lower
says how much separation the data support, the upper how large it could still
be after sampling uncertainty.

### 3.5 The lower endpoint — transport certificate

$$\Pr\Big\{W_{\infty,0}(\bar\mu_P,\bar\mu_Q) \ge \underline\Delta_{m,m'}^{\alpha}/(n L_\nu)\Big\} \gtrsim 1-\alpha$$

**This requires only the additive Lipschitz interface** — neither
`ass:diagram-distortion-floor` nor the structured population model, and no
restriction on $P$ and $Q$. It is the general certificate.

Three things it is not, each of which an earlier CASTLE draft claimed:

- Not a bound on the **bottleneck distance between the two distributions**. The
  quantity is $W_{\infty,0}$ between **padded mean measures**.
- Not a statement about exemplar diagrams. There are no two diagrams it says are
  far apart.
- Not $T/L_\Phi$. That rested on a uniform population converse to the
  lower-distortion floor, and `ex:pop-obstruction` shows none exists.

The field carrying it **MUST** be named for the quantity it bounds rather than
for "bottleneck", and its docstring **MUST** state that it is a confidence
statement about mean measures.

**Paper III states the sharpness caveat itself** and the library **MUST**
propagate it: the strength of the certificate depends on $nL_\nu$, a large
cardinality bound or a conservative Lipschitz constant makes it small, and **a
zero or small certificate is inconclusive rather than evidence of geometric
closeness**. That is D3's answer in the paper's own words — the number is
reported with its interpretation attached, not suppressed.

### 3.6 The upper endpoint — structured exclusion

The note treated the upper endpoint as unused. It is not.

For a structured class $\mathfrak A_r(\tau)$ with $g_r(\tau) > 0$, membership
requires $\Delta_\nu(P,Q) \ge g_r(\tau)$ (`ass:uniform-structured-alternatives`
with `thm:population-lower-bound`). So an observed
$\overline\Delta_{m,m'}^{\alpha} < g_r(\tau)$ **supports exclusion of that class
at pointwise asymptotic level $\alpha$**.

Two constraints the object **MUST** carry:

- It is a **compatibility certificate, not a uniform test** over
  $\mathfrak A_r(\tau)$. A uniform guarantee would need uniform Gaussian
  approximation and covariance-estimation results, which are not available.
- It is usable **only when the constants defining $g_r(\tau)$ are scientifically
  defensible and numerically available.** Where they are not, the interval is
  the output and the exclusion is not reported at all.

Exposing this is optional in v1; claiming it without both constraints is not.

### 3.7 The population converse exists — under structure

`thm:population-lower-bound` gives population lower bounds on $\Delta_\nu(P,Q)$
under the template-thinning model and either prevalence regime, each of the form
*signal minus perturbation penalty*, positive when the structural signal exceeds
the penalty.

So the correct statement is **not** "there is no population converse". It is
that there is no **uniform** converse over $\mathcal P_\tau$, and there is a
structured one whose constants a user must be able to defend. §9 row 1 is scoped
accordingly.

### 3.8 The estimand line is mandatory

Every `TwoSampleResult` **MUST** carry, and every rendering **MUST** display:

> tests equality of mean embeddings under $\nu$ at truncation $K$; not an
> omnibus test

Consistency against any mean-embedding alternative follows from
`cor:mean-distance-consistency`. **The word "minimax" MUST NOT appear** in any
user-facing string: see §9.

---

## 4. Tool 2 — sample-size calculator

### 4.1 Three numbers, all labelled

**Guarantee (finite-sample, conservative).** From `prop:finite-sample-power`:

$$N_{\mathrm{eff}} \ge \frac{v^2}{\{\Delta_\nu^{\Psi}(P,Q)\}^2}\left(\alpha^{-1/2}+\beta^{-1/2}\right)^2$$

with $v^2$ a bound on $\operatorname{tr}\Sigma^{\Psi}$. **This is the sample
size for $\phi_{\alpha,v}$**, the Chebyshev test of §3.3 — not for the spectral
or permutation calibration. The docstring **MUST** name the test it applies to;
a user who plans with this number and then runs a different calibration has not
planned for what they ran.

**Uniform structured guarantee.** From `thm:finite-approximation-three-way`,
power at least $1-\beta$ **uniformly over $\mathfrak A_r(\tau)$**, with
truncation entering explicitly:

$$N_{\mathrm{eff}} \ge \frac{v^2}{g_r(\tau)^2 - 4\epsilon_K^2}\left(\alpha^{-1/2}+\beta^{-1/2}\right)^2, \qquad 2\epsilon_K < g_r(\tau)$$

This is the strongest of the three and the note omitted it. The side condition
$2\epsilon_K < g_r(\tau)$ **MUST** be checked and reported; below it the bound
does not apply and the tool **MUST** say so rather than returning a number.

**Planning number (oracle).** The local-alternative benchmark. It presumes the
direction is known or pilot-estimated, and the docstring **MUST** say so. One
$z$ convention **MUST** be fixed and named.

### 4.2 $\Delta$ enters unbiased

A pilot estimate **MUST** use the unbiased $\hat\Delta^2_U$; the biased plug-in
carries $\operatorname{tr}\Sigma_c/n_c$ per class, which is §2.3's bias by a
second route.

The post-hoc inversion — *was this study powered?* — is the same formula and
**SHOULD** be exposed.

---

## 5. Tool 3 — per-region significance map

Per-coordinate two-sample $z$-tests on the $\Phi$ coordinates of the inference
split, Benjamini–Hochberg across the $K$ landmarks, resting on
`thm:finite-coordinate-berry-esseen` applied coordinate-wise, or on permutation
with $\nu$ fixed — subject to §3.3's caveat, which applies here too.

**Attribution of a flagged coordinate to a birth–death region requires
near-disjoint landmark supports.** A condition on the placement, and it **MUST**
be reported as a condition rather than as a certificate. Where supports overlap
materially the map localises to a union of regions and the object **MUST** say
which.

A simultaneous band over all coordinates is out of scope here.

---

## 6. Tool 4 — robustness certificate

$\epsilon_{\max} = \Delta T/(L_\nu L^{\mathrm{filt}})$, with $\Delta T$ the
margin over the calibrated threshold. The centroid variant is
`prop:centroid-robustness-certificate`.

**The soundness statement is conditional and MUST be reported as such**, with
the margin returned alongside the certificate: reliability is a function of the
margin, and a caller cannot otherwise tell which regime they are in.

---

## 7. The reporting card

Five lines, every one computed. No line **MAY** be rendered from a stored
constant or an example value.

1. Estimand and configuration — mean embeddings under $\nu$; truncation $K$;
   pilot size
2. $p$ per calibration, **each labelled with the null it is valid under**
3. $\Delta^{\Psi}$ interval; the transport certificate with its sharpness
   caveat
4. $n$ per group against Tool 2's guarantee, uniform guarantee, and planning
   number
5. $\epsilon_{\max}$ with its margin

---

## 8. Relationship to RFC-0001

Diagrams enter as `DiagramBatch` (RFC-0001 §4). This module is a **caller** of
the interchange layer and adds no requirement to it, with one exception recorded
there: RFC-0001 §9.1 binds `core/distances.py`, which this module uses and does
not implement.

`castle/` is NumPy-backed by the dated deviation of 2026-08-09; `diagrams/`
remains array-API-pure.

---

## 9. Claims that did not survive

Each row is a claim an earlier CASTLE draft made. An implementation reproducing
any of them is wrong in a way tests will not catch: the code runs, the number is
plausible, and the sentence attached to it is false.

| # | The withdrawn claim | What is true |
|---|---|---|
| 1 | A **uniform** population converse: separation of mean measures implies mean-embedding separation, for all $P,Q$ | False, by `ex:pop-obstruction` — rare separated mass keeps $W_{\infty,0}$ at $\tau$ while $\Delta_\nu \to 0$. The **upper** transfer holds with no restriction, and a **structured** converse holds under `thm:population-lower-bound` |
| 2 | $d_B(P,Q) \ge T/L_\Phi$, "in the sense that there exist exemplar diagrams" | No such object. A lower **confidence** bound on $W_{\infty,0}$ between **padded mean measures**, §3.5 |
| 3 | The test is minimax-optimal over $\tau$-separated laws | Withdrawn, and replaced by a sharper negative result: $\inf_{\mathcal P_\tau}\Delta = 0$, so no test on the embedded samples has power bounded away from $\alpha$ uniformly over $\mathcal P_\tau$, at any sample size (`prop:no-uniform-power-unrestricted`). $\mathcal P_\tau$ was the wrong alternative class. A minimax rate survives over **structured** prevalence classes (`thm:structured-prevalence-minimax-rate`) |
| 4 | Permutation calibration validates the test | Exact under $P=Q$ only; **not** generally valid under the mean null, which is this module's own estimand (§3.3) |

**The naming consequence.** A field called `certified_bottleneck` is wrong even
where its arithmetic is right, because the name is the claim for every caller
who does not read the docstring.

**What is not on this list.** The certified-inference claim itself. A rejection
certifies separation of the padded mean measures, with no structural assumption.
That direction is what inference needs and it was never in doubt.

---

## 10. Open decisions

| # | Decision | Needed by | Position |
|---|---|---|---|
| D1 | Split as the default protocol | **2026-09-14**, Tool 1 freeze | **Resolved: split** (§2.2) |
| D2 | Default feature space | after E0 | Additive $\Phi$; the transfer interface is stated for it, and the gram has no geometric certificate |
| D3 | Certificate on the card when small | resolved by the paper | **On the card, with its caveat.** Paper III states that a small certificate is inconclusive rather than evidence of closeness; the library propagates that rather than suppressing the number (§3.5) |
| D4 | Package identity | resolved | `akriti.castle`, Apache-2.0 |
| D5 | Ship the fixed-direction statistic with $h$ pilot-estimated? | Tool 1 freeze | Option, not default |
| D6 | Is structured exclusion (§3.6) exposed in v1? | Tool 1 freeze | Open. It needs $g_r(\tau)$'s constants to be defensible and numerically available, and where they are not the feature cannot be offered at all |
| D7 | How is $K$ selected, and does the object expose $\epsilon_K$? | Tool 1 freeze | Open, and §4.1's side condition makes it consequential rather than presentational |

---

## 11. What is measured and what is not

RFC-0001 §9's rule: a support claim no test runs is a claim that rots silently.

| Component | Evidence today |
|---|---|
| Tool 1, permutation calibration | Power results exist **for the Gaussian gram only** |
| Tool 1, additive map — **the default** | **Nothing.** No power results, no interval coverage, no spectral calibration |
| Tool 1, interval coverage | Paper III measures coverage of its confidence balls in simulation; the **library's** interval has never been checked against it |
| Tool 2, uniform guarantee | Untested, and the $2\epsilon_K < g_r(\tau)$ side condition has no test at all |
| Tool 3 | Power $1.0$, FDR $0.03$–$0.04$ at $q = 0.05$. **Missing:** a $\delta = 0$ null run and the re-run under the split protocol |
| Tool 4 | Conditional survival as in §6; done modulo the re-run |

**The gap that matters is Tool 1's.** The default feature space is the one with
no evidence behind it, and the interval §3.4 calls the primary output has never
had its coverage measured in this implementation. Neither blocks freezing the
signature by D1; both block a release that makes the claim.
