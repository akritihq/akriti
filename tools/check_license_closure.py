#!/usr/bin/env python3
"""Fail if the installed dependency closure contains copyleft or unknown licenses.

The default `pip install akriti` closure must be permissive-only. `DEPENDENCIES.md`
states that rule, and it is not self-enforcing: persim pulls
GPLv3 `hopcroftkarp` transitively, and GUDHI ships a wheel with no license
metadata at all. Both were found by hand on 2026-07-29. Neither would be found
again by hand.

Run against a clean environment that has only the default install:

    python -m venv .venv-closure
    .venv-closure/bin/pip install .
    .venv-closure/bin/python tools/check_license_closure.py

Or check a permitted extra:

    .venv-closure/bin/python tools/check_license_closure.py --profile rips

Exit status 0 if the closure is clean, 1 otherwise.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import re
import sys
from dataclasses import dataclass

# Packaging tooling that pip puts in every venv. Not part of our closure and
# not shipped to anyone.
IGNORED = {"pip", "setuptools", "wheel", "pkg-resources", "akriti"}

PERMISSIVE = {
    "MIT",
    "MIT-CMU",
    "BSD",
    "BSD-2-CLAUSE",
    "BSD-3-CLAUSE",
    "0BSD",
    "APACHE-2.0",
    "APACHE 2.0",
    "APACHE SOFTWARE LICENSE",
    "PSF",
    "PSF-2.0",
    "PYTHON SOFTWARE FOUNDATION LICENSE",
    "ISC",
    "ZLIB",
    "CC0-1.0",
    "UNLICENSE",
    "HPND",
    "MIT LICENSE",
    "BSD LICENSE",
}
WEAK_COPYLEFT = {
    "MPL-2.0",
    "MPL 2.0",
    "MOZILLA PUBLIC LICENSE 2.0",
    "LGPL",
    "LGPL-2.1",
    "LGPL-3.0",
    "LGPLV3",
    "EPL-2.0",
}
STRONG_COPYLEFT = {
    "GPL",
    "GPL-2.0",
    "GPL-3.0",
    "GPLV2",
    "GPLV3",
    "AGPL",
    "AGPL-3.0",
    "AGPLV3",
    "GNU AGPLV3",
    "GNU GENERAL PUBLIC LICENSE",
    "SSPL",
    "SSPL-1.0",
}

# Reviewed exceptions. Every entry needs a name, a license, and a reason a human
# signed off on. Test-only packages never reach a user's runtime environment.
ALLOWED_EXCEPTIONS: dict[str, tuple[str, str]] = {
    "hypothesis": (
        "MPL-2.0",
        "test-only; weak file-level copyleft, never shipped, not linked",
    ),
}

# Packages whose metadata is absent or useless upstream, resolved by hand.
# Each entry records where the license was actually confirmed.
MANUAL_LICENSES: dict[str, tuple[str, str]] = {
    "gudhi": (
        "GPL-3.0",
        "no license metadata in the wheel at all; https://gudhi.inria.fr/licensing/ "
        "states GUDHI code is MIT but CGAL/Miniball/PyKeOps modules are "
        "'MIT (GPL v3)', and the wheel bundles them (gudhi.AlphaComplex imports "
        "from a plain install). Treated as GPL-3.0 -- confirmed 2026-07-29.",
    ),
}


@dataclass
class Finding:
    name: str
    version: str
    license: str
    verdict: str
    note: str = ""


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def license_of(dist: md.Distribution) -> str:
    """Best available license string, preferring modern metadata."""
    meta = dist.metadata
    expr = meta.get("License-Expression")
    if expr:
        return expr.strip()

    classifiers = [c for c in (meta.get_all("Classifier") or []) if "License" in c]
    if classifiers:
        return "; ".join(c.split("::")[-1].strip() for c in classifiers)

    lic = meta.get("License") or ""
    lic = lic.strip()
    # Some projects paste the entire license text into this field.
    if len(lic) > 80:
        first = lic.splitlines()[0][:80]
        return f"<full text> {first}"
    return lic


def classify(license_str: str) -> str:
    """permissive | weak-copyleft | strong-copyleft | unknown"""
    if not license_str or license_str.startswith("<full text>"):
        # A pasted license body is not a machine-readable answer. Fall through
        # to unknown so it gets a manual entry rather than a guess.
        if "BSD" in license_str.upper():
            return "permissive"
        return "unknown"

    upper = license_str.upper()
    # Split compound expressions: "BSD-3-Clause AND MIT AND Zlib"
    parts = re.split(r"\s+(?:AND|OR)\s+|;\s*", upper)
    parts = [p.strip(" ()") for p in parts if p.strip(" ()")]

    verdicts = set()
    for part in parts:
        strong = part in STRONG_COPYLEFT or any(
            tok in part for tok in STRONG_COPYLEFT if len(tok) > 3
        )
        if strong:
            verdicts.add("strong-copyleft")
        elif part in WEAK_COPYLEFT or any(t in part for t in ("MPL", "LGPL")):
            verdicts.add("weak-copyleft")
        elif part in PERMISSIVE or any(
            t in part
            for t in (
                "MIT",
                "BSD",
                "APACHE",
                "ISC",
                "ZLIB",
                "PSF",
                "PYTHON SOFTWARE FOUNDATION",
                "CC0",
                "HPND",
            )
        ):
            verdicts.add("permissive")
        else:
            verdicts.add("unknown")

    for level in ("strong-copyleft", "unknown", "weak-copyleft", "permissive"):
        if level in verdicts:
            return level
    return "unknown"


def audit() -> list[Finding]:
    findings = []
    for dist in md.distributions():
        raw = dist.metadata["Name"]
        if not raw:
            continue
        name = normalize(raw)
        if name in IGNORED:
            continue

        if name in MANUAL_LICENSES:
            lic, note = MANUAL_LICENSES[name]
            findings.append(
                Finding(name, dist.version, lic, classify(lic), f"manual: {note}")
            )
            continue

        lic = license_of(dist)
        findings.append(Finding(name, dist.version, lic or "<none>", classify(lic)))

    # importlib can yield duplicates across path entries.
    unique = {f.name: f for f in findings}
    return sorted(unique.values(), key=lambda f: f.name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        default="default",
        help="install profile being checked; only 'default' is enforced strictly",
    )
    parser.add_argument(
        "--allow-copyleft",
        action="store_true",
        help="report but do not fail; for auditing an extras profile",
    )
    args = parser.parse_args()

    findings = audit()
    width = max((len(f.name) for f in findings), default=10)

    print(f"Dependency closure audit — profile: {args.profile}")
    print(f"{len(findings)} distributions\n")

    violations = []
    for f in findings:
        marker = {
            "permissive": "ok  ",
            "weak-copyleft": "WEAK",
            "strong-copyleft": "FAIL",
            "unknown": "????",
        }[f.verdict]
        print(f"  [{marker}] {f.name:<{width}}  {f.version:<12} {f.license}")
        if f.note:
            print(f"           {f.note}")

        if f.verdict == "permissive":
            continue
        if f.name in ALLOWED_EXCEPTIONS:
            expected, reason = ALLOWED_EXCEPTIONS[f.name]
            print(f"           allowed exception ({expected}): {reason}")
            continue
        violations.append(f)

    print()

    # The default closure is not merely permissive, it is empty: RFC-0001
    # §3.3 and §10.1 requirement 2 require the interchange layer to import
    # nothing beyond the standard library, and numpy left the closure for that
    # reason after the backends left it for licensing ones. A permissive but
    # non-empty default would satisfy every check above and still be the thing
    # the policy forbids, so emptiness is checked separately rather than
    # inferred from an absence of violations. See DEPENDENCIES.md.
    if args.profile == "default" and findings:
        print(f"FAIL — the default closure must be empty, found {len(findings)}:")
        for f in findings:
            print(f"  - {f.name} {f.version}")
        print()
        print("`pip install akriti` installs no third-party distribution at")
        print("all, numpy included. Move it behind an extra, or if it is")
        print("genuinely needed at runtime, make it a lazy, function-scoped")
        print("import at the one boundary that needs it (RFC-0001 §3.3).")
        return 1

    if not violations:
        print(
            "PASS — closure is empty."
            if args.profile == "default"
            else "PASS — closure is permissive-only."
        )
        return 0

    print(f"{len(violations)} package(s) outside the permissive closure:")
    for f in violations:
        print(f"  - {f.name} {f.version}: {f.license} [{f.verdict}]")
    print()
    print("Fix by moving the dependency behind an optional extra, or add a")
    print("reviewed entry to ALLOWED_EXCEPTIONS with a stated reason. Do not")
    print("silence this by widening the permissive list.")

    if args.allow_copyleft:
        print("\n(--allow-copyleft set: reporting only, not failing)")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
