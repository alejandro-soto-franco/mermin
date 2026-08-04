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
    areas=None,
    n_cells=3,
    n_nuclei=3,
    correlation_length=4.0,
    charges=(0.5, -1.0),
    coherence_field=None,
):
    """A result that satisfies all five invariants by default, so a caller
    need only override the one field under test. `areas` and
    `coherence_field` let a test give per-row / per-pixel values, for
    checking that a violation's detail locates the offender."""
    cell_areas = list(areas) if areas is not None else [area] * n_cells
    field = coherence_field if coherence_field is not None else np.full((4, 4), coherence)
    return SimpleNamespace(
        fields={"coherence": field},
        cells=pl.DataFrame({"area": cell_areas}),
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


class TestKValidation:
    """`k` is a caller-supplied argument, not a property of the result, so a
    bad `k` is a programming error and raises rather than reporting a
    `Violation`."""

    def test_k_zero_raises_a_clear_error_naming_k(self):
        with pytest.raises(ValueError, match="k"):
            check_invariants(fake_result(), k=0)

    def test_k_negative_raises(self):
        with pytest.raises(ValueError, match="k"):
            check_invariants(fake_result(), k=-1)

    def test_k_non_integer_raises(self):
        with pytest.raises(ValueError, match="k"):
            check_invariants(fake_result(), k=2.5)

    def test_k_2_and_k_3_are_accepted(self):
        assert check_invariants(fake_result(), k=2) == []
        assert check_invariants(fake_result(charges=(1 / 3, -2 / 3)), k=3) == []


class TestViolationDetailsAreLocatable:
    """`area_floor` and `defect_charge_multiple_of_1_over_k` must name where
    the offending value lives, not just what it was, so a corpus entry with
    hundreds of cells or defects has a locator rather than a bare count."""

    def test_area_floor_names_the_offending_row_index(self):
        result = fake_result(areas=[1.0, 0.1, 1.0])
        violations = check_invariants(result)
        detail = next(v.detail for v in violations if v.check == "area_floor")
        assert "(1, 0.1)" in detail

    def test_area_floor_names_every_offending_row_when_more_than_one(self):
        result = fake_result(areas=[0.1, 1.0, 0.2])
        violations = check_invariants(result)
        detail = next(v.detail for v in violations if v.check == "area_floor")
        assert "(0, 0.1)" in detail
        assert "(2, 0.2)" in detail

    def test_defect_charge_names_the_offending_list_index(self):
        result = fake_result(charges=(0.5, 0.3, -1.0))
        violations = check_invariants(result)
        detail = next(
            v.detail
            for v in violations
            if v.check == "defect_charge_multiple_of_1_over_k"
        )
        assert "(1, 0.3)" in detail

    def test_coherence_detail_names_the_field_location_above_one(self):
        field = np.array([[0.5, 0.5], [1.2, 0.5]])
        result = fake_result(coherence_field=field)
        violations = check_invariants(result)
        detail = next(v.detail for v in violations if v.check == "coherence_range")
        assert "(1, 0)" in detail
        assert "above the upper bound 1" in detail

    def test_coherence_detail_names_the_field_location_below_zero(self):
        field = np.array([[0.5, -0.2], [0.5, 0.5]])
        result = fake_result(coherence_field=field)
        violations = check_invariants(result)
        detail = next(v.detail for v in violations if v.check == "coherence_range")
        assert "(0, 1)" in detail
        assert "below the lower bound 0" in detail
