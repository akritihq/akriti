#!/usr/bin/env python3
"""Fail if the installed dependency closure contains copyleft or unknown licenses.

The default `pip install akriti` closure must be permissive-only. `DEPENDENCIES.md`
states that rule, and it is not self-enforcing: persim pulls
GPLv3 `hopcroftkarp` transitively, and GUDHI ships a wheel with no license
metadata at all. Both were found by hand on 2026-07-29. Neither would be found
again by hand.

Run the strict checks in a clean environment. Check the empty default closure
before installing the permissive-only `io` extra:

    python -m venv .venv-closure
    .venv-closure/bin/pip install .
    .venv-closure/bin/python tools/check_license_closure.py
    .venv-closure/bin/pip install ".[io]"
    .venv-closure/bin/python tools/check_license_closure.py --profile io

The development environment intentionally contains reviewed copyleft
dependencies. Audit it for visibility in explicit report-only mode:

    python tools/check_license_closure.py --profile dev --allow-copyleft

In strict mode, exit status is 0 if the closure is clean and 1 otherwise.
For non-default profiles, `--allow-copyleft` reports the same findings but
returns 0. The default profile's separate empty-closure rule always fails when
any third-party distribution is present.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass

# Packaging tooling that pip puts in every venv. Not part of our closure and
# not shipped to anyone.
IGNORED = {"pip", "setuptools", "wheel", "pkg-resources", "akriti"}

SUPPORTED_PROFILES = (
    "default",
    "extras",
    "rips",
    "alpha",
    "distances",
    "numpy",
    "io",
    "torch",
    "bio",
    "test",
    "lint",
    "dev",
)

MAX_LICENSE_LENGTH = 4096
MAX_LICENSE_NESTING = 64

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
    "MIT-0",
    "3-CLAUSE BSD LICENSE",
}
WEAK_COPYLEFT = {
    "MPL-2.0",
    "MPL 2.0",
    "MOZILLA PUBLIC LICENSE 2.0",
    "MOZILLA PUBLIC LICENSE 2.0 (MPL 2.0)",
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
    "GNU GENERAL PUBLIC LICENSE V3 (GPLV3)",
    "SSPL",
    "SSPL-1.0",
}


@dataclass(frozen=True)
class ExceptionPolicy:
    expected_license: str
    profiles: frozenset[str]
    reason: str


# Reviewed exceptions. Every entry needs an exact license, an explicit profile
# scope, and a reason a human signed off on. Test-only packages never reach a
# user's runtime environment.
ALLOWED_EXCEPTIONS: dict[str, ExceptionPolicy] = {
    "hypothesis": ExceptionPolicy(
        "MPL-2.0",
        frozenset({"test", "dev"}),
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
    """Classify exact aliases and explicit compounds; unknown input fails closed."""
    if not license_str or not license_str.isascii():
        # A pasted license body is not a machine-readable answer.
        return "unknown"
    if len(license_str) > MAX_LICENSE_LENGTH:
        return "unknown"

    normalized = _normalize_license(license_str)
    if (
        not normalized
        or len(normalized) > MAX_LICENSE_LENGTH
        or normalized.startswith("<FULL TEXT>")
    ):
        # A pasted license body is not a machine-readable answer.
        return "unknown"

    valid, verdicts = _parse_expression(normalized)
    if not valid:
        return "unknown"

    for level in ("strong-copyleft", "unknown", "weak-copyleft", "permissive"):
        if level in verdicts:
            return level
    return "unknown"


def _normalize_license(license_str: str) -> str:
    """Normalize only case and repeated whitespace for exact alias matching."""
    return re.sub(r"\s+", " ", license_str.strip().upper())


def _parse_expression(value: str, nesting: int = 0) -> tuple[bool, set[str]]:
    """Parse a bounded expression into verdicts, rejecting malformed syntax."""
    value = value.strip()
    if not value or nesting > MAX_LICENSE_NESTING:
        return False, set()
    if re.match(r"^(?:AND|OR)(?:\s|$)", value) or re.search(r"\s(?:AND|OR)$", value):
        return False, set()

    parts, separators, balanced = _split_top_level(value)
    if not balanced or len(parts) != len(separators) + 1:
        return False, set()
    if separators:
        verdicts: set[str] = set()
        for part in parts:
            valid, part_verdicts = _parse_expression(part, nesting)
            if not valid:
                return False, set()
            verdicts.update(part_verdicts)
        return True, verdicts

    verdict = _LICENSE_VERDICTS.get(value)
    if verdict is not None:
        return True, {verdict}

    inner = _outer_parenthesized_content(value)
    if inner is not None:
        return _parse_expression(inner, nesting + 1)
    if "(" in value or ")" in value:
        return False, set()
    return True, {"unknown"}


def _split_top_level(value: str) -> tuple[list[str], list[str], bool]:
    """Split only top-level operators while validating balanced parentheses."""
    parts: list[str] = []
    separators: list[str] = []
    start = 0
    depth = 0
    index = 0
    while index < len(value):
        character = value[index]
        if character == "(":
            depth += 1
            if depth > MAX_LICENSE_NESTING:
                return [], [], False
        elif character == ")":
            depth -= 1
            if depth < 0:
                return [], [], False
        elif depth == 0:
            if character == ";":
                parts.append(value[start:index])
                separators.append(";")
                start = index + 1
            else:
                for operator in ("AND", "OR"):
                    end = index + len(operator)
                    if (
                        value.startswith(operator, index)
                        and index > 0
                        and value[index - 1].isspace()
                        and end < len(value)
                        and value[end].isspace()
                    ):
                        parts.append(value[start:index])
                        separators.append(operator)
                        start = end
                        index = end - 1
                        break
        index += 1
    if depth != 0:
        return [], [], False
    parts.append(value[start:])
    return parts, separators, True


def _outer_parenthesized_content(value: str) -> str | None:
    if not value.startswith("(") or not value.endswith(")"):
        return None
    depth = 0
    for index, character in enumerate(value):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                if index != len(value) - 1:
                    return None
                return value[1:-1].strip()
    return None


_LICENSE_VERDICTS = {
    **dict.fromkeys(PERMISSIVE, "permissive"),
    **dict.fromkeys(WEAK_COPYLEFT, "weak-copyleft"),
    **dict.fromkeys(STRONG_COPYLEFT, "strong-copyleft"),
}


_EXCEPTION_LICENSE_ALIASES = {
    "MPL-2.0": "MPL-2.0",
    "MPL 2.0": "MPL-2.0",
    "MOZILLA PUBLIC LICENSE 2.0": "MPL-2.0",
    "MOZILLA PUBLIC LICENSE 2.0 (MPL 2.0)": "MPL-2.0",
}


def _canonical_exception_license(license_str: str) -> str | None:
    """Return a reviewed exception licence's canonical name, if exact."""
    if not license_str.isascii():
        return None
    normalized = _normalize_license(license_str)
    return _EXCEPTION_LICENSE_ALIASES.get(normalized)


def _validate_exception_configuration(
    exceptions: dict[str, ExceptionPolicy] | None = None,
) -> None:
    """Ensure every configured exception policy is explicit and supported."""
    configured = ALLOWED_EXCEPTIONS if exceptions is None else exceptions
    for package, policy in configured.items():
        if not isinstance(policy, ExceptionPolicy):
            raise ValueError(f"{package}: exception must be an ExceptionPolicy")
        if not isinstance(policy.expected_license, str):
            raise ValueError(f"{package}: expected licence must be a string")
        expected_canonical = _canonical_exception_license(policy.expected_license)
        if expected_canonical is None:
            raise ValueError(
                f"{package}: unsupported expected licence {policy.expected_license!r}; "
                "update _EXCEPTION_LICENSE_ALIASES"
            )
        if policy.expected_license != expected_canonical:
            raise ValueError(
                f"{package}: expected licence {policy.expected_license!r} must use "
                f"canonical spelling {expected_canonical!r}"
            )
        if not isinstance(policy.profiles, frozenset) or not policy.profiles:
            raise ValueError(f"{package}: exception profiles must be nonempty")
        if not all(isinstance(profile, str) for profile in policy.profiles):
            raise ValueError(f"{package}: exception profiles must contain only strings")
        unsupported = set(policy.profiles) - set(SUPPORTED_PROFILES)
        if unsupported:
            raise ValueError(
                f"{package}: unsupported profiles {sorted(unsupported)!r}; "
                f"choose from {SUPPORTED_PROFILES!r}"
            )
        if not isinstance(policy.reason, str) or not policy.reason.strip():
            raise ValueError(f"{package}: exception reason must be nonempty")


_validate_exception_configuration()


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--profile",
        default="default",
        choices=SUPPORTED_PROFILES,
        help="install profile being checked; 'default' also requires an empty closure",
    )
    parser.add_argument(
        "--allow-copyleft",
        action="store_true",
        help="report but do not fail; for development and non-permissive extras",
    )
    args = parser.parse_args(argv)

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
        exception = ALLOWED_EXCEPTIONS.get(normalize(f.name))
        if exception:
            expected = exception.expected_license
            actual_canonical = _canonical_exception_license(f.license)
            expected_canonical = _canonical_exception_license(expected)
            if args.profile not in exception.profiles:
                profiles = ", ".join(sorted(exception.profiles))
                print(
                    f"           exception not allowed for profile {args.profile!r}; "
                    f"allowed profiles: {profiles}"
                )
                violations.append(f)
                continue
            if actual_canonical is not None and actual_canonical == expected_canonical:
                print(f"           allowed exception ({expected}): {exception.reason}")
                continue
            print(
                f"           exception not allowed: detected {f.license}; "
                f"expected {expected}; update _EXCEPTION_LICENSE_ALIASES if "
                "this is a valid metadata alias"
            )
            violations.append(f)
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
