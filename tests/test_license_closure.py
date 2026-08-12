"""Focused regression tests for the release licence-closure gate."""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
import tempfile
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "check_license_closure",
    Path(__file__).parents[1] / "tools/check_license_closure.py",
)
assert _SPEC is not None
assert _SPEC.loader is not None
license_closure = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = license_closure
_SPEC.loader.exec_module(license_closure)


def _extract_optional_profile_keys(section_body: str) -> set[str]:
    profiles = set()
    for line_number, line in enumerate(section_body.splitlines(), start=1):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        if stripped.startswith(('"', "'")):
            quoted_key = re.match(
                r"""^(?:"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')"""
                r"(?:\s*\.\s*[A-Za-z0-9_-]+)*\s*=",
                stripped,
            )
            if quoted_key:
                raise ValueError(
                    f"unrecognized optional-dependency key on line {line_number}: "
                    f"{stripped.split('=', 1)[0].strip()!r}"
                )
            continue
        assignment = re.match(r"^\s*([^=]+?)\s*=", line)
        if assignment is None:
            continue
        key = assignment.group(1).strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]+", key):
            raise ValueError(
                f"unrecognized optional-dependency key on line {line_number}: {key!r}"
            )
        profiles.add(key)
    return profiles


@pytest.mark.parametrize("license_name", ["", "<full text> BSD 3-Clause License"])
def test_empty_and_full_text_are_unknown(license_name: str) -> None:
    assert license_closure.classify(license_name) == "unknown"


@pytest.mark.parametrize(
    ("license_name", "expected"),
    [
        ("MIT-0", "permissive"),
        ("3-Clause BSD License", "permissive"),
        ("mit   license", "permissive"),
        ("LGPL", "weak-copyleft"),
        ("LGPL-2.1", "weak-copyleft"),
        ("LGPL-3.0", "weak-copyleft"),
        ("LGPLv3", "weak-copyleft"),
        ("GNU General Public License v3 (GPLv3)", "strong-copyleft"),
        ("M\u0131T", "unknown"),
        ("B\u017fD", "unknown"),
        ("Moz\u0131lla Publ\u0131c L\u0131cense 2.0", "unknown"),
    ],
)
def test_exact_verified_aliases_classify_without_substring_matching(
    license_name: str, expected: str
) -> None:
    assert license_closure.classify(license_name) == expected


@pytest.mark.parametrize(
    "license_name",
    [
        "MIT-like proprietary license",
        "BSD-3-Clause with non-commercial restriction",
        "Business Source License 1.1 (converts to MIT after 4 years)",
        "MIT AND unknown-license",
        "MIT OR",
        "AND MIT",
        "MIT;",
        "GPL-3.0;",
        "; GPL-3.0",
        "GPL-3.0 AND (MIT) trailing",
        "MIT)",
        "(MIT",
        "MIT (proprietary addendum)",
    ],
)
def test_malformed_restricted_and_unknown_licenses_fail_closed(
    license_name: str,
) -> None:
    assert license_closure.classify(license_name) == "unknown"


@pytest.mark.parametrize(
    ("license_name", "expected"),
    [
        ("MIT AND BSD-3-Clause", "permissive"),
        ("Apache-2.0 OR MIT", "permissive"),
        ("LGPL-3.0; MIT", "weak-copyleft"),
        ("MIT OR GPL-3.0", "strong-copyleft"),
        ("GPL-3.0 AND Proprietary", "strong-copyleft"),
        ("BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0", "permissive"),
        ("MIT AND (BSD-3-Clause OR Apache-2.0)", "permissive"),
        ("Mozilla Public License 2.0 (MPL 2.0); MIT", "weak-copyleft"),
        ("GPL-3.0 OR OR MIT", "unknown"),
        ("(MIT) AND (GPL-3.0)", "strong-copyleft"),
        ("(MIT AND BSD-3-Clause)", "permissive"),
    ],
)
def test_explicit_compound_licenses_classify_all_components(
    license_name: str, expected: str
) -> None:
    assert license_closure.classify(license_name) == expected


@pytest.mark.parametrize(
    "license_name",
    [
        "MPL-2.0",
        "MPL 2.0",
        "Mozilla Public License 2.0",
        "Mozilla Public License 2.0 (MPL 2.0)",
    ],
)
@pytest.mark.parametrize("profile", ["test", "dev"])
def test_hypothesis_exception_accepts_reviewed_mpl_aliases(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    license_name: str,
    profile: str,
) -> None:
    finding = license_closure.Finding(
        "Hypothesis", "1.0", license_name, license_closure.classify(license_name)
    )
    monkeypatch.setattr(license_closure, "audit", lambda: [finding])

    assert license_closure.main(["--profile", profile]) == 0
    assert "allowed exception" in capsys.readouterr().out


def test_hypothesis_exception_rejects_unicode_confusable_alias(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    license_name = "Moz\u0131lla Publ\u0131c L\u0131cense 2.0"
    finding = license_closure.Finding(
        "hypothesis", "1.0", license_name, license_closure.classify(license_name)
    )
    monkeypatch.setattr(license_closure, "audit", lambda: [finding])

    assert license_closure.main(["--profile", "test"]) == 1
    output = capsys.readouterr().out
    assert "allowed exception" not in output
    assert f"detected {license_name}" in output


@pytest.mark.parametrize(
    "license_name",
    [
        "(" * 100 + "MIT" + ")" * 100,
        "MIT " * 2000,
    ],
)
def test_overlong_or_deeply_nested_license_is_unknown(license_name: str) -> None:
    assert license_closure.classify(license_name) == "unknown"


def test_raw_overlong_whitespace_collapsed_license_is_unknown() -> None:
    license_name = "MIT" + " " * (license_closure.MAX_LICENSE_LENGTH + 1) + "LICENSE"

    assert license_closure.classify(license_name) == "unknown"


@pytest.mark.parametrize("profile", ["io", "rips"])
def test_hypothesis_mpl_exception_is_rejected_outside_test_and_dev_profiles(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    profile: str,
) -> None:
    finding = license_closure.Finding(
        "hypothesis", "1.0", "MPL-2.0", license_closure.classify("MPL-2.0")
    )
    monkeypatch.setattr(license_closure, "audit", lambda: [finding])

    assert license_closure.main(["--profile", profile]) == 1
    output = capsys.readouterr().out
    assert f"profile {profile!r}" in output
    assert "dev, test" in output


@pytest.mark.parametrize(
    "license_name",
    [
        "AGPL-3.0",
        "GPL-3.0",
        "LGPL-3.0",
        "MPL-2.0 AND GPL-3.0",
        "MPL-2.0 OR MIT",
        "MPL-2.0; MIT",
        "<full text> Mozilla Public License 2.0",
    ],
)
def test_hypothesis_exception_rejects_changed_or_compound_licences(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    license_name: str,
) -> None:
    finding = license_closure.Finding(
        "hypothesis", "1.0", license_name, license_closure.classify(license_name)
    )
    monkeypatch.setattr(license_closure, "audit", lambda: [finding])

    assert license_closure.main(["--profile", "test"]) == 1
    output = capsys.readouterr().out
    assert "allowed exception" not in output
    assert f"detected {license_name}" in output
    assert "expected MPL-2.0" in output


def test_exception_package_with_permissive_relicense_needs_no_waiver(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    finding = license_closure.Finding("hypothesis", "1.0", "MIT", "permissive")
    monkeypatch.setattr(license_closure, "audit", lambda: [finding])

    assert license_closure.main(["--profile", "rips"]) == 0
    output = capsys.readouterr().out
    assert "allowed exception" not in output
    assert "exception not allowed" not in output


def test_mismatched_exception_remains_report_only_when_explicitly_allowed(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    finding = license_closure.Finding(
        "hypothesis", "1.0", "AGPL-3.0", "strong-copyleft"
    )
    monkeypatch.setattr(license_closure, "audit", lambda: [finding])

    assert license_closure.main(["--profile", "extras", "--allow-copyleft"]) == 0
    assert "reporting only" in capsys.readouterr().out


def test_invalid_profile_is_rejected_before_audit(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        license_closure,
        "audit",
        lambda: pytest.fail("audit must not run for an invalid profile"),
    )

    with pytest.raises(SystemExit) as exc_info:
        license_closure.main(["--profile", "defualt"])

    assert exc_info.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_supported_profiles_match_pyproject_optional_dependency_keys() -> None:
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    contents = pyproject.read_text(encoding="utf-8")
    section = re.search(
        r"(?ms)^\[project\.optional-dependencies\]\s*\n(?P<body>.*?)(?=^\[|\Z)",
        contents,
    )
    assert section is not None
    optional_profiles = _extract_optional_profile_keys(section.group("body"))

    assert set(license_closure.SUPPORTED_PROFILES) == optional_profiles | {
        "default",
        "extras",
    }


def test_optional_profile_keys_accept_hyphenated_bare_keys() -> None:
    assert _extract_optional_profile_keys("gpu-tools = []\n") == {"gpu-tools"}


def test_optional_profile_keys_reject_unrecognized_assignment() -> None:
    with pytest.raises(ValueError, match="unrecognized optional-dependency key"):
        _extract_optional_profile_keys("gpu.tools = []\n")


def test_optional_profile_keys_reject_quoted_dotted_assignment() -> None:
    with pytest.raises(ValueError, match="unrecognized optional-dependency key"):
        _extract_optional_profile_keys('"gpu".tools = []\n')


def test_profile_choice_help_preserves_declared_order(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        license_closure,
        "audit",
        lambda: pytest.fail("audit must not run for an invalid profile"),
    )

    with pytest.raises(SystemExit):
        license_closure.main(["--profile", "invalid"])

    error = capsys.readouterr().err
    declared = license_closure.SUPPORTED_PROFILES
    repr_choices = ", ".join(map(repr, declared))
    usage_choices = "{" + ",".join(declared) + "}"
    assert f"choose from {repr_choices}" in error or usage_choices in error


@pytest.mark.parametrize(
    "profile",
    [
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
    ],
)
def test_all_supported_profiles_are_accepted(
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
) -> None:
    monkeypatch.setattr(license_closure, "audit", lambda: [])

    assert license_closure.main(["--profile", profile]) == 0


def test_exception_configuration_rejects_unsupported_expected_license() -> None:
    exceptions = {
        "somepkg": license_closure.ExceptionPolicy(
            "LGPL-2.1", frozenset({"test"}), "reviewed"
        )
    }

    with pytest.raises(
        ValueError, match=r"somepkg.*LGPL-2.1.*_EXCEPTION_LICENSE_ALIASES"
    ):
        license_closure._validate_exception_configuration(exceptions)


def test_exception_configuration_requires_canonical_expected_license() -> None:
    exceptions = {
        "somepkg": license_closure.ExceptionPolicy(
            "MPL 2.0", frozenset({"test"}), "reviewed"
        )
    }

    with pytest.raises(ValueError, match=r"somepkg.*MPL 2\.0.*MPL-2\.0"):
        license_closure._validate_exception_configuration(exceptions)


@pytest.mark.parametrize(
    ("profiles", "reason", "message"),
    [
        (frozenset(), "reviewed", "nonempty"),
        (frozenset({"not-a-profile"}), "reviewed", "unsupported profile"),
        (frozenset({"test", 1}), "reviewed", "only strings"),
        (frozenset({"test"}), " ", "reason"),
    ],
)
def test_exception_configuration_rejects_invalid_policy_fields(
    profiles: frozenset[str], reason: str, message: str
) -> None:
    policy_type = getattr(license_closure, "ExceptionPolicy", None)
    assert policy_type is not None
    policy = policy_type("MPL-2.0", profiles, reason)
    with pytest.raises(ValueError, match=message):
        license_closure._validate_exception_configuration({"somepkg": policy})


def test_exception_policy_is_immutable() -> None:
    policy_type = getattr(license_closure, "ExceptionPolicy", None)
    assert policy_type is not None
    policy = policy_type("MPL-2.0", frozenset({"test"}), "reviewed")

    with pytest.raises(FrozenInstanceError):
        policy.reason = "changed"  # type: ignore[misc]


def test_mismatched_exception_points_to_license_aliases(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    finding = license_closure.Finding(
        "hypothesis", "1.0", "AGPL-3.0", "strong-copyleft"
    )
    monkeypatch.setattr(license_closure, "audit", lambda: [finding])

    assert license_closure.main(["--profile", "test"]) == 1
    assert "_EXCEPTION_LICENSE_ALIASES" in capsys.readouterr().out


def test_invalid_static_exception_fails_during_module_import() -> None:
    source = Path(__file__).parents[1] / "tools/check_license_closure.py"
    contents = source.read_text(encoding="utf-8")
    original = '"hypothesis": ExceptionPolicy(\n        "MPL-2.0",'
    replacement = '"hypothesis": ExceptionPolicy(\n        "LGPL-2.1",'
    assert contents.count(original) == 1

    with tempfile.TemporaryDirectory() as directory:
        copy = Path(directory) / "check_license_closure.py"
        copy.write_text(contents.replace(original, replacement), encoding="utf-8")
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import importlib.util, sys; "
                    "spec = importlib.util.spec_from_file_location("
                    "'closure', sys.argv[1]); "
                    "module = importlib.util.module_from_spec(spec); "
                    "sys.modules['closure'] = module; spec.loader.exec_module(module)"
                ),
                str(copy),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

    assert result.returncode != 0
    assert "hypothesis" in result.stderr
    assert "LGPL-2.1" in result.stderr
    assert "_EXCEPTION_LICENSE_ALIASES" in result.stderr


def test_empty_default_closure_still_passes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(license_closure, "audit", lambda: [])

    assert license_closure.main([]) == 0
    assert "PASS — closure is empty." in capsys.readouterr().out
