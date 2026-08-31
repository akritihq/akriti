"""Live torch coverage for RFC-0001 §3.3's D18 compatibility path."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
array_api_compat = pytest.importorskip("array_api_compat")

from akriti.diagrams.adapters import (  # noqa: E402
    from_array,
    from_giotto,
    from_ripser,
    to_arrays,
    to_csv,
)
from akriti.diagrams.core import DiagramBatch, namespace_of  # noqa: E402

pytestmark = [pytest.mark.backend, pytest.mark.torch]

_ROWS = [[0.0, float("inf"), 0.0], [0.2, 0.5, 1.0], [0.1, 0.4, 0.0]]


def _diagram_table() -> torch.Tensor:
    """Return `(birth, death, dim)` rows with one essential bar."""
    return torch.tensor(_ROWS, dtype=torch.float64)


def test_torch_namespace_uses_array_api_compat_fallback() -> None:
    """Torch lacks the declaration, so RFC-0001 §3.3 resolves through compat."""
    values = _diagram_table()

    assert not hasattr(torch.Tensor, "__array_namespace__")
    resolved = namespace_of(values)
    expected = array_api_compat.array_namespace(values)

    assert resolved is expected


def test_torch_from_array_and_d18_accessors_use_the_resolved_namespace() -> None:
    """D18's five affected accessors match under torch and NumPy namespaces."""
    values = _diagram_table()
    diagram = from_array(values)
    reference = from_array(np.asarray(_ROWS, dtype=np.float64))
    namespace = namespace_of(values)

    assert diagram.xp is namespace
    assert reference.xp is np
    assert diagram.xp is not reference.xp
    assert diagram.births.dtype == torch.float64
    assert (
        diagram.essential.tolist()
        == reference.essential.tolist()
        == [
            True,
            False,
            False,
        ]
    )
    assert diagram.persistence.tolist() == reference.persistence.tolist()

    degree_zero = diagram.dim(0)
    reference_degree_zero = reference.dim(0)
    assert (
        (
            degree_zero.dims.tolist(),
            degree_zero.births.tolist(),
            degree_zero.deaths.tolist(),
        )
        == (
            reference_degree_zero.dims.tolist(),
            reference_degree_zero.births.tolist(),
            reference_degree_zero.deaths.tolist(),
        )
        == ([0, 0], [0.0, 0.1], [float("inf"), 0.4])
    )

    finite = diagram.finite
    reference_finite = reference.finite
    assert (
        (
            finite.dims.tolist(),
            finite.births.tolist(),
            finite.deaths.tolist(),
        )
        == (
            reference_finite.dims.tolist(),
            reference_finite.births.tolist(),
            reference_finite.deaths.tolist(),
        )
        == ([1, 0], [0.2, 0.1], [0.5, 0.4])
    )

    batch = DiagramBatch.from_diagrams([diagram, degree_zero])
    reference_batch = DiagramBatch.from_diagrams([reference, reference_degree_zero])
    assert batch.xp is namespace
    assert batch.bar_counts.tolist() == reference_batch.bar_counts.tolist() == [3, 2]


# ---------------------------------------------------------------------------
# The adapter paths that touch the namespace hardest, under the compat shim
# ---------------------------------------------------------------------------
#
# Everything above exercises `from_array` on a well-formed table, which is the
# happy path and the one least likely to find a shim gap. The clamp reaches for
# `nextafter`, `finfo`, `full_like` and `zeros_like`; `from_giotto` indexes a
# rank-3 array per axis and then masks rows out with a boolean array; the
# exporters iterate arrays and call `stack`. None of that was asserted under
# any namespace but NumPy's and `array_api_strict`'s, and torch is the one
# backend RFC-0001 §3.3 reaches through `array-api-compat` rather than through
# a native `__array_namespace__` (D18, A.7).


def _ulp_below(value: float) -> float:
    """The next float64 below `value`, which is one local ULP of slack."""
    return float(np.nextafter(np.float64(value), np.float64(-np.inf)))


def test_torch_clamps_a_representational_i6_violation() -> None:
    """§3.1: the adapter absorbs a sub-ULP `death < birth` and MUST warn.

    `_clamp_i6` is the densest array-API code in the module. Under torch it
    runs through the compat shim rather than a native namespace, so this is
    where a missing or differently-spelled `nextafter` would surface."""
    values = torch.tensor([[1.0, _ulp_below(1.0), 0.0]], dtype=torch.float64)

    with pytest.warns(UserWarning, match="clamped 1 of 1 rows"):
        diagram = from_array(values)

    assert diagram.xp is namespace_of(values)
    assert diagram.deaths.tolist() == [1.0]
    assert diagram.meta.provenance["clamped_rows"] == 1


def test_torch_clamps_at_zero_through_the_subnormal_branch() -> None:
    """The branch that swaps in a benign probe rather than calling `nextafter`
    on zero, then selects the minimum-subnormal spacing back."""
    smallest_subnormal = float.fromhex("0x0.0000000000001p-1022")
    values = torch.tensor([[0.0, -smallest_subnormal, 0.0]], dtype=torch.float64)

    with pytest.warns(UserWarning, match="clamped"):
        diagram = from_array(values)

    assert diagram.deaths.tolist() == [0.0]


def test_torch_leaves_a_real_i6_violation_for_the_type_to_refuse() -> None:
    """§3.1: anything larger than the representational allowance is a backend
    bug, surfaced rather than absorbed -- under torch as under NumPy."""
    values = torch.tensor([[1.0, 0.5, 0.0]], dtype=torch.float64)

    with pytest.raises(ValueError, match="death"):
        from_array(values)


def test_torch_from_giotto_strips_padding_and_keeps_the_namespace() -> None:
    """§11.1 under torch: per-axis rank-3 indexing and boolean row masking.

    `strip_padding=True` is the mode that actually removes rows, so it is the
    one that exercises the mask. It also warns about nothing, which keeps this
    a test about the namespace.

    **Sample 0's H0 bar is essential, and must stay that way.** This fixture
    declares `reduced_homology=False` with `infinity_values=inf`, and §11's
    `N11-11` refuses that pair on a sample whose degree-0 deaths are all
    finite: non-reduced H0 of a nonempty space carries a class that never
    dies, so the two declarations cannot both be true. An earlier revision
    gave the bar a death of `1.0` and was written before that clause existed;
    it made this test raise `ValueError` from `_reject_impossible_reduced_homology`
    rather than measure anything. Shortening the death back is not a
    simplification, it is a return to an impossible diagram -- change
    `reduced_homology` instead if this ever needs a finite-death H0 row.
    """
    values = torch.tensor(
        [
            [[0.0, float("inf"), 0.0], [0.5, 0.5, 0.0]],
            [[0.0, float("inf"), 1.0], [2.0, 2.0, 1.0]],
        ],
        dtype=torch.float64,
    )

    batch = from_giotto(
        values,
        reduced_homology=False,
        infinity_values=float("inf"),
        strip_padding=True,
    )

    assert batch.xp is namespace_of(values)
    assert batch.bar_counts.tolist() == [1, 1]
    assert batch[0].meta.provenance["padding_removed"] == 1
    assert bool(batch[1].essential.tolist()[0])


def test_torch_from_ripser_stacks_a_degree_list() -> None:
    """§11: degree by list position, with `concat` running under the shim."""
    dgms = [
        torch.tensor([[0.0, float("inf")], [0.0, 0.5]], dtype=torch.float64),
        torch.tensor([[0.3, 0.9]], dtype=torch.float64),
    ]

    diagram = from_ripser(dgms)

    assert diagram.xp is namespace_of(dgms[0])
    assert diagram.dims.tolist() == [0, 0, 1]
    assert diagram.deaths.tolist() == [float("inf"), 0.5, 0.9]
    assert diagram.meta.provenance["source_dtype"] == str(torch.float64)


def test_torch_exporters_read_a_torch_backed_diagram() -> None:
    """§10.3: `to_arrays` stays in the diagram's namespace; `to_csv` leaves it
    for text, and must still spell `inf` and the degrees correctly."""
    diagram = from_array(_diagram_table())

    with pytest.warns(UserWarning, match="export discards"):
        grouped = to_arrays(diagram)
    with pytest.warns(UserWarning, match="export discards"):
        text = to_csv(diagram)

    assert sorted(grouped) == [0, 1]
    assert grouped[0].shape == (2, 2)
    assert namespace_of(grouped[0]) is namespace_of(diagram.births)
    assert text == "dim,birth,death\n0,0.0,inf\n1,0.2,0.5\n0,0.1,0.4\n"
