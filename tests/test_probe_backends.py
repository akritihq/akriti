"""Focused tests for the RFC-0001 evidence probe's release gates."""

from __future__ import annotations

import ast
import importlib.util
import inspect
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_PROBE_PATH = _ROOT / "rfcs/evidence/probe_backends.py"
_WORKFLOW_PATH = _ROOT / ".github/workflows/ci.yml"
_SPEC = importlib.util.spec_from_file_location("probe_backends", _PROBE_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
probe_backends = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(probe_backends)


def test_probe_sources_are_readable_after_chdir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Release-gate source paths remain readable independently of the CWD."""
    monkeypatch.chdir(tmp_path)

    assert _PROBE_PATH.read_text(encoding="utf-8")
    assert _WORKFLOW_PATH.read_text(encoding="utf-8")


def test_every_read_text_call_declares_utf8_encoding() -> None:
    """Test source reads must not depend on the host locale encoding."""
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    read_text_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "read_text"
    ]

    assert read_text_calls
    assert all(
        any(
            keyword.arg == "encoding"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == "utf-8"
            for keyword in call.keywords
        )
        for call in read_text_calls
    )


def test_workflow_requires_giotto_and_does_not_suppress_evidence_failures() -> None:
    """The evidence job must fail when installation or the probe drifts."""
    workflow = _WORKFLOW_PATH.read_text(encoding="utf-8")
    evidence_job = workflow.split("  rfc-evidence:\n", 1)[1].split("\n  build:", 1)[0]

    assert "continue-on-error: true" not in evidence_job
    assert "probe_backends.py --require-giotto" in evidence_job


def test_probe_has_one_shared_max_edge_and_uses_it_for_both_backends() -> None:
    """A.1/A.5 must not silently compare different filtration cutoffs."""
    tree = ast.parse(_PROBE_PATH.read_text(encoding="utf-8"))

    def keyword_uses_max_edge(call: ast.Call, keyword: str) -> bool:
        value = next((kw.value for kw in call.keywords if kw.arg == keyword), None)
        return isinstance(value, ast.Name) and value.id == "MAX_EDGE"

    gudhi_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "RipsComplex"
    ]
    ripser_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ripser"
    ]

    assert probe_backends.MAX_EDGE == 4.0
    assert len(gudhi_calls) == 2
    assert all(keyword_uses_max_edge(call, "max_edge_length") for call in gudhi_calls)
    assert len(ripser_calls) == 2
    assert all(keyword_uses_max_edge(call, "thresh") for call in ripser_calls)


def test_shape_guard_fails_before_comparison_with_section_diagnostic() -> None:
    """Unequal H1 shapes are drift, never broadcastable comparison input."""
    with pytest.raises(probe_backends.ProbeDriftError, match=r"A\.3.*shape"):
        probe_backends._require_same_shape(
            np.zeros((2, 2)), np.zeros((1, 2)), section="A.3"
        )


@pytest.mark.parametrize("shape", [(2,), (2, 3)])
def test_diagram_shape_guard_rejects_malformed_equal_shapes(shape) -> None:
    """Equal backend shapes are insufficient unless each diagram is `(n, 2)`."""
    malformed = np.zeros(shape)

    with pytest.raises(probe_backends.ProbeDriftError, match=r"A\.3.*shape"):
        probe_backends._require_array_shape(
            malformed,
            columns=2,
            section="A.3",
            label="Ripser H1",
        )


def test_giotto_batch_guard_fails_before_sample_indexing() -> None:
    """Malformed batches fail diagnostically before `[0]` or `[:, 2]`."""
    with pytest.raises(probe_backends.ProbeDriftError, match=r"A\.2.*shape"):
        probe_backends._require_batch_shape(
            np.zeros((2, 3)),
            samples=2,
            section="A.2",
            label="giotto batch",
        )


def test_trivial_rows_require_exact_zero_persistence() -> None:
    """Small positive persistence is drift, not giotto padding."""
    diagram = np.array([[1.0, 1.0, 0.0], [1.0, 1.0 + 1e-12, 0.0]])

    assert probe_backends._trivial_mask(diagram).tolist() == [True, False]


def test_float32_value_gate_rejects_true_float64_precision() -> None:
    """A declared float64 array is not evidence that its values are single precision."""
    float32_values = np.array([np.float64(np.float32(0.1))])
    float64_value = np.array([np.nextafter(1.0, 2.0)])

    probe_backends._require_float32_values(
        float32_values,
        section="A.3",
        label="Ripser H1",
    )
    with pytest.raises(probe_backends.ProbeDriftError, match=r"A\.3.*float32"):
        probe_backends._require_float32_values(
            float64_value,
            section="A.3",
            label="Ripser H1",
        )


def test_warning_capture_handles_an_empty_warning_list() -> None:
    """A warning-free backend case is measured without indexing a missing warning."""
    value, caught = probe_backends._measure_with_warnings(
        lambda _left, _right: 0.0,
        np.zeros((0, 2)),
        np.zeros((0, 2)),
    )

    assert value == 0.0
    assert caught == []


def test_warning_gate_checks_the_expected_message_multiset() -> None:
    """Duplicate warnings for one argument cannot stand in for both arguments."""
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(2):
            warnings.warn(
                "dgm1 has points with non-finite death times;ignoring those points",
                UserWarning,
                stacklevel=1,
            )

    expected = [
        "dgm1 has points with non-finite death times;ignoring those points",
        "dgm2 has points with non-finite death times;ignoring those points",
    ]
    with pytest.raises(probe_backends.ProbeDriftError, match="messages changed"):
        probe_backends._require_warnings(
            caught,
            expected,
            section="A.4",
            operation="bottleneck",
        )


def test_warning_gate_fails_diagnostically_before_indexing_an_empty_list() -> None:
    """Warning disappearance is reported as drift rather than IndexError."""
    with pytest.raises(probe_backends.ProbeDriftError, match="stopped warning"):
        probe_backends._require_warnings(
            [],
            ["expected warning"],
            section="A.4",
            operation="bottleneck",
        )


def test_a4_empty_finite_wasserstein_measurement_is_pinned_tightly() -> None:
    """The gate pins the measured value, not a broad range around another value."""
    assert (
        pytest.approx(0.9899494936611666, rel=0, abs=1e-15)
        == probe_backends.EMPTY_FINITE_WASSERSTEIN
    )
    assert probe_backends.A4_RTOL <= 1e-12
    assert probe_backends.A4_ATOL <= 1e-12


def test_coefficient_carrier_scan_detects_field_aliases() -> None:
    """A.5 must catch any returned attribute or key naming the field."""

    class Result:
        coeff_field = 3

    class Metadata:
        def __init__(self) -> None:
            self.prime = 3

    class NestedResult:
        def __init__(self) -> None:
            self.metadata = Metadata()

    class ArrayResult(np.ndarray):
        pass

    Record = namedtuple("Record", ["bars", "characteristic"])
    array_result = np.zeros((1, 2)).view(ArrayResult)
    array_result.metadata = Metadata()

    assert probe_backends._coefficient_carriers(Result()) == ["coeff_field"]
    assert probe_backends._coefficient_carriers({"coefficient": 3}) == ["coefficient"]
    assert probe_backends._coefficient_carriers([Record([], 3)]) == [
        "[0].characteristic"
    ]
    assert probe_backends._coefficient_carriers({"metadata": {"prime": 3}}) == [
        "metadata.prime"
    ]
    assert probe_backends._coefficient_carriers(NestedResult()) == ["metadata.prime"]
    assert probe_backends._coefficient_carriers(array_result) == ["metadata.prime"]

    with pytest.raises(probe_backends.ProbeDriftError, match=r"A\.5.*metadata\.prime"):
        probe_backends._require_no_coefficient_carriers(
            array_result,
            section="A.5",
            label="giotto returned value",
        )


def test_parameter_default_guard_fails_before_signature_lookup() -> None:
    """A removed backend parameter produces an A.5 diagnostic, not KeyError."""
    with pytest.raises(probe_backends.ProbeDriftError, match=r"A\.5.*parameter"):
        probe_backends._parameter_default(
            lambda: None,
            "coeff",
            section="A.5",
            label="Ripser",
        )


def test_attribute_guard_fails_before_backend_attribute_lookup() -> None:
    """A removed fitted attribute produces drift rather than AttributeError."""
    with pytest.raises(probe_backends.ProbeDriftError, match=r"A\.1.*infinity_values_"):
        probe_backends._required_attribute(
            object(),
            "infinity_values_",
            section="A.1",
            label="giotto estimator",
        )


@pytest.mark.parametrize(
    ("factory", "expected"),
    [
        (
            lambda seen: (
                lambda _array, *, force_all_finite: seen.update(
                    force_all_finite=force_all_finite
                )
            ),
            {"force_all_finite": False},
        ),
        (
            lambda seen: (
                lambda _array, *, ensure_all_finite: seen.update(
                    ensure_all_finite=ensure_all_finite
                )
            ),
            {"ensure_all_finite": False},
        ),
        (
            lambda seen: (
                lambda _array, *, force_all_finite, ensure_all_finite=True: seen.update(
                    force_all_finite=force_all_finite,
                    ensure_all_finite=ensure_all_finite,
                )
            ),
            {"force_all_finite": False, "ensure_all_finite": True},
        ),
    ],
    ids=["old-only", "new-only", "both-keywords"],
)
def test_check_array_shim_supports_public_signature_variants(factory, expected) -> None:
    """The compatibility shim must translate only when the new keyword exists."""
    seen: dict[str, object] = {}
    original = factory(seen)
    shim = probe_backends._make_check_array_shim(original)

    shim([[1.0]], force_all_finite=False)

    assert seen == expected
    assert inspect.signature(original) is not None


def test_check_array_shim_rejects_conflicting_old_and_new_values() -> None:
    """Translation must never silently overwrite a caller's new-keyword value."""

    def new_only(_array, *, ensure_all_finite):
        return ensure_all_finite

    shim = probe_backends._make_check_array_shim(new_only)

    with pytest.raises(TypeError, match="conflicting values"):
        shim(
            [[1.0]],
            force_all_finite=False,
            ensure_all_finite=True,
        )


@pytest.mark.backend
def test_main_returns_zero_on_a_clean_optional_backend_probe() -> None:
    """The probe exposes a testable success return while giotto remains optional."""
    assert probe_backends.main([]) == 0
