"""Invariants that must hold whatever the goldens say.

Each class below covers one invariant in both directions (a value that must
pass, and a value that must fail), and every "must fail" test asserts the
returned violation list contains *exactly* that one check and no other. That
is the independence evidence the task asked for: a check that only fires
when several things are wrong at once is not five invariants, it is one, and
these tests would catch that by finding more than one `Violation` (or the
wrong one) in the list.
"""

import math
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from mermin_corpus.invariants import Violation, check_invariants


def fake_result(
    *,
    coherence=0.5,
    area=1.0,
    n_cells=3,
    n_nuclei=3,
    correlation_length=4.0,
    charges=(0.5, -1.0),
):
    """A result that satisfies all five invariants by default, so a caller
    need only override the one field under test."""
    return SimpleNamespace(
        fields={"coherence": np.full((4, 4), coherence)},
        cells=pl.DataFrame({"area": [area] * n_cells}),
        segmentation={"n_nuclei": n_nuclei},
        correlations={"correlation_length": correlation_length},
        defects=[{"charge": charge} for charge in charges],
    )


class TestCheckInvariantsBaseline:
    def test_a_healthy_result_has_no_violations(self):
        assert check_invariants(fake_result()) == []

    def test_violation_is_a_frozen_dataclass_with_check_and_detail(self):
        v = Violation(check="x", detail="y")
        assert (v.check, v.detail) == ("x", "y")
        with pytest.raises(Exception):
            v.check = "z"


class TestCoherenceRange:
    """coherence is |psi_k| by construction, so it must lie in [0, 1]."""

    def test_zero_is_the_lower_boundary_and_passes(self):
        assert check_invariants(fake_result(coherence=0.0)) == []

    def test_one_is_the_upper_boundary_and_passes(self):
        assert check_invariants(fake_result(coherence=1.0)) == []

    def test_just_below_zero_fails_and_only_that_check_fires(self):
        violations = check_invariants(fake_result(coherence=-1e-9))
        assert [v.check for v in violations] == ["coherence_range"]

    def test_just_above_one_fails_and_only_that_check_fires(self):
        violations = check_invariants(fake_result(coherence=1 + 1e-9))
        assert [v.check for v in violations] == ["coherence_range"]


class TestAreaFloor:
    """The smallest area observed anywhere in the corpus was 0.5 square
    pixels; 0.25 is the floor a degenerate contour must not clear."""

    def test_exactly_the_floor_passes(self):
        assert check_invariants(fake_result(area=0.25)) == []

    def test_just_below_the_floor_fails_and_only_that_check_fires(self):
        violations = check_invariants(fake_result(area=0.1))
        assert [v.check for v in violations] == ["area_floor"]

    def test_zero_area_fails_and_only_that_check_fires(self):
        violations = check_invariants(fake_result(area=0.0))
        assert [v.check for v in violations] == ["area_floor"]

    def test_no_cells_is_vacuously_fine(self):
        assert check_invariants(fake_result(n_cells=0, n_nuclei=0)) == []


class TestCellCountMatchesNuclei:
    """A nucleus whose watershed catchment came out empty would vanish from
    `result.cells` with no error unless the counts are checked against each
    other."""

    def test_equal_counts_pass(self):
        assert check_invariants(fake_result(n_cells=5, n_nuclei=5)) == []

    def test_one_fewer_cell_than_nucleus_fails_and_only_that_check_fires(self):
        violations = check_invariants(fake_result(n_cells=4, n_nuclei=5))
        assert [v.check for v in violations] == ["cell_count_matches_nuclei"]

    def test_zero_of_both_is_vacuously_fine(self):
        assert check_invariants(fake_result(n_cells=0, n_nuclei=0)) == []


class TestCorrelationLengthNeverNaN:
    """Infinity is a legitimate correlation length for a uniform director
    field; NaN is not a result at all."""

    def test_infinite_passes(self):
        assert check_invariants(fake_result(correlation_length=math.inf)) == []

    def test_nan_fails_and_only_that_check_fires(self):
        violations = check_invariants(fake_result(correlation_length=math.nan))
        assert [v.check for v in violations] == [
            "correlation_length_finite_or_infinite"
        ]


class TestDefectChargesAreMultiplesOf1OverK:
    """A holonomy-derived charge cannot take any value other than a multiple
    of 1/k."""

    def test_half_and_minus_one_pass_at_k_2(self):
        assert check_invariants(fake_result(charges=(0.5, -1.0))) == []

    def test_a_third_fails_at_k_2_and_only_that_check_fires(self):
        violations = check_invariants(fake_result(charges=(0.3,)))
        assert [v.check for v in violations] == [
            "defect_charge_multiple_of_1_over_k"
        ]

    def test_no_defects_is_vacuously_fine(self):
        assert check_invariants(fake_result(charges=())) == []

    def test_a_third_passes_at_k_3(self):
        result = fake_result(charges=(1 / 3, -2 / 3))
        assert check_invariants(result, k=3) == []

    def test_a_half_fails_at_k_3_and_only_that_check_fires(self):
        violations = check_invariants(fake_result(charges=(0.5,)), k=3)
        assert [v.check for v in violations] == [
            "defect_charge_multiple_of_1_over_k"
        ]


class TestMultipleViolationsAllReport:
    def test_two_broken_invariants_both_appear(self):
        result = fake_result(coherence=-1e-9, area=0.0)
        checks = {v.check for v in check_invariants(result)}
        assert checks == {"coherence_range", "area_floor"}
