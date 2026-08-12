"""Focused regression tests for the release licence-closure gate."""

from __future__ import annotations

import importlib.util
import sys
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


@pytest.mark.parametrize("license_name", ["", "<full text> BSD 3-Clause License"])
def test_empty_and_full_text_are_unknown(license_name: str) -> None:
    assert license_closure.classify(license_name) == "unknown"


@pytest.mark.parametrize(
    "license_name",
    [
        "MPL-2.0",
        "MPL 2.0",
        "Mozilla Public License 2.0",
        "Mozilla Public License 2.0 (MPL 2.0)",
    ],
)
def test_hypothesis_exception_accepts_reviewed_mpl_aliases(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    license_name: str,
) -> None:
    finding = license_closure.Finding(
        "Hypothesis", "1.0", license_name, license_closure.classify(license_name)
    )
    monkeypatch.setattr(license_closure, "audit", lambda: [finding])

    assert license_closure.main(["--profile", "rips"]) == 0
    assert "allowed exception" in capsys.readouterr().out


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

    assert license_closure.main(["--profile", "rips"]) == 1
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
    exceptions = {"somepkg": ("LGPL-2.1", "reviewed")}

    with pytest.raises(
        ValueError, match=r"somepkg.*LGPL-2.1.*_EXCEPTION_LICENSE_ALIASES"
    ):
        license_closure._validate_exception_configuration(exceptions)


def test_mismatched_exception_points_to_license_aliases(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    finding = license_closure.Finding(
        "hypothesis", "1.0", "AGPL-3.0", "strong-copyleft"
    )
    monkeypatch.setattr(license_closure, "audit", lambda: [finding])

    assert license_closure.main(["--profile", "rips"]) == 1
    assert "_EXCEPTION_LICENSE_ALIASES" in capsys.readouterr().out


def test_empty_default_closure_still_passes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(license_closure, "audit", lambda: [])

    assert license_closure.main([]) == 0
    assert "PASS — closure is empty." in capsys.readouterr().out
