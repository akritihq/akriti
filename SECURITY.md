# Security policy

## Reporting a vulnerability

**Please do not open a public issue for a security vulnerability.**

Report privately through GitHub's [private vulnerability
reporting](https://github.com/akritihq/akriti/security/advisories/new), or by
email to **security@akriti.io**.

Include, as far as you can: what the issue is, how to reproduce it, the version
and platform, and what an attacker could achieve. A proof of concept is helpful
but not required — a clear description of the mechanism is enough.

## What to expect

| Stage | Target |
|---|---|
| Acknowledgement of your report | 5 working days |
| Initial assessment and severity | 10 working days |
| Fix or documented mitigation | 90 days |

Akriti is a small academic project. These are honest targets rather than a
contractual SLA, and we will tell you promptly if something is going to take
longer.

**Embargo.** We ask for 90 days from acknowledgement before public disclosure,
or until a fix ships, whichever is sooner. If you need a different timeline, say
so in your report and we will agree one. We will not take legal action against
anyone acting in good faith under this policy.

**Credit.** Reporters are credited in the advisory and release notes by default.
Tell us if you would rather stay anonymous.

## Scope

In scope:

- The `akriti` package and anything under this repository.
- Our release and distribution pipeline — build workflows, signing, published
  artifacts.
- Dependency-confusion, typosquatting, or namespace attacks targeting `akriti`
  on PyPI.

Out of scope:

- Vulnerabilities in GUDHI, Ripser, persim, giotto-tda, NumPy or any other
  upstream project. Please report those to their maintainers. If an upstream
  issue is reachable through our API in a way that is worse than upstream's own
  exposure, that *is* in scope — tell us.
- Statistical or numerical incorrectness. That is a serious bug and we want to
  hear about it, but it belongs in a public issue where it can be discussed
  openly, not under embargo.

## Supply chain

The project's security posture is deliberately front-loaded, because it is far
cheaper now than retrofitted:

- **Trusted Publishing** to PyPI — no long-lived API tokens.
- **Sigstore signing** of release artifacts.
- **SBOM** generated per release.
- **OpenSSF Scorecard** in CI.
- **Enforced 2FA** across the `akritihq` organisation.
- **Pinned, audited dependency closure.** The default install is
  permissive-licensed only, and `tools/check_license_closure.py` fails CI if a
  copyleft or unlicensed package enters it. See
  [DEPENDENCIES.md](DEPENDENCIES.md).

### A note on dependencies

We take dependency provenance seriously and publish what we find, including
about our own closure. Two examples currently documented in
[DEPENDENCIES.md](DEPENDENCIES.md): a GPLv3 package with no release since 2019
sits in the transitive closure of two major TDA libraries, and one widely used
backend ships wheels with no licence metadata at all. Neither is a
vulnerability. Both are the kind of thing that becomes one.

Contributors: never add a dependency without verifying the package exists and is
the one you meant. Package-name hallucination followed by typosquatting against
the hallucinated name is an established attack, and this project uses AI
assistance by policy.
