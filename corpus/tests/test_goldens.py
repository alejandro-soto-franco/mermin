"""Golden record construction and comparison."""

import math
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from mermin_corpus.goldens import (
    LOOSE,
    TIGHT,
    Difference,
    build_record,
    compare,
)


def fake_result(*, n_cells=3, area_mean=10.0, splay=1.25, corr_len=4.0,
                 r_bins=(1.0,), g_values=(0.9,), defect_charges=(0.5,),
                 theta=None, coherence=None, optimal_sigma=None):
    # Every column the real pipeline (`mermin.pipeline.analyze`) puts on
    # `cells`, not just the four the original golden happened to name.
    # Values differ per row and per column so a scaling attack on any one
    # column is detectable in its own min/max/mean rather than accidentally
    # cancelling.
    cells = pl.DataFrame(
        {
            "label": list(range(1, n_cells + 1)),
            "centroid_x": [10.0 * (i + 1) for i in range(n_cells)],
            "centroid_y": [20.0 * (i + 1) for i in range(n_cells)],
            "area": [area_mean] * n_cells,
            "perimeter": [12.0 * (i + 1) for i in range(n_cells)],
            "shape_index": [3.8 + i for i in range(n_cells)],
            "convexity": [0.9 - 0.01 * i for i in range(n_cells)],
            "elongation": [0.4 + 0.01 * i for i in range(n_cells)],
            "elongation_angle": [0.1 * (i + 1) for i in range(n_cells)],
            "nuclear_aspect_ratio": [1.2 + 0.05 * i for i in range(n_cells)],
            "nuclear_angle": [0.2 * (i + 1) for i in range(n_cells)],
        },
        schema={
            "label": pl.Int64,
            "centroid_x": pl.Float64,
            "centroid_y": pl.Float64,
            "area": pl.Float64,
            "perimeter": pl.Float64,
            "shape_index": pl.Float64,
            "convexity": pl.Float64,
            "elongation": pl.Float64,
            "elongation_angle": pl.Float64,
            "nuclear_aspect_ratio": pl.Float64,
            "nuclear_angle": pl.Float64,
        },
    )
    if theta is None:
        theta = np.full((8, 8), 0.5)
    if coherence is None:
        coherence = np.full((8, 8), 0.25)
    if optimal_sigma is None:
        optimal_sigma = np.full((8, 8), 2.0)
    return SimpleNamespace(
        cells=cells,
        fields={
            "theta": theta,
            "coherence": coherence,
            "optimal_sigma": optimal_sigma,
        },
        defects=[{"position": (1, 1), "charge": c, "angle": 0.0} for c in defect_charges],
        correlations={"correlation_length": corr_len, "r_bins": list(r_bins), "g_values": list(g_values)},
        frank={"splay": splay, "bend": 0.75, "ratio": 1.5},
        ldg_params={"a": 1.0, "b": 2.0, "c": 3.0, "k_elastic": 4.0},
        ingest={
            "roles": {"nuclear": {"index": 0, "mechanism": "emission", "evidence": "470.0"}},
            "pixel_size_um": 0.5,
            "projection": "single",
        },
        segmentation={
            "backend": "threshold",
            "version": "threshold-1",
            "config": {"sigma": 2.0, "min_size": 25, "min_distance": 5},
            "mechanism": "explicit",
            "cache": "disabled",
            "n_nuclei": n_cells,
        },
    )


def record(**kwargs):
    return build_record(
        fake_result(**kwargs),
        entry="fake-entry",
        invocation={"segmentation": "threshold", "pixel_size_um": None},
        environment={"mermin": "0.5.0", "scikit-image": "0.26.0"},
    )


def record_from(result):
    return build_record(
        result,
        entry="fake-entry",
        invocation={"segmentation": "threshold", "pixel_size_um": None},
        environment={"mermin": "0.5.0", "scikit-image": "0.26.0"},
    )


_UNPINNED_CELL_COLUMNS = (
    "centroid_x",
    "centroid_y",
    "convexity",
    "elongation_angle",
    "nuclear_aspect_ratio",
    "nuclear_angle",
)


class TestBuildRecord:
    def test_carries_the_entry_and_schema(self):
        r = record()
        assert r["entry"] == "fake-entry"
        assert r["schema_version"]

    def test_counts_are_present_and_integral(self):
        r = record(n_cells=7)
        assert r["counts"] == {"nuclei": 7, "cells": 7, "defects": 1}
        assert all(isinstance(v, int) for v in r["counts"].values())

    def test_cells_are_summarised_not_enumerated(self):
        """A per-row golden over hundreds of cells is unreadable as a diff and
        fails on a single boundary pixel."""
        r = record(n_cells=200)
        summary = r["numerics"]["cells"]["area"]
        assert set(summary) == {"min", "max", "mean"}

    def test_every_non_identifier_column_is_summarised(self):
        """Every column `cells` carries is summarised, not the four the
        original golden happened to name. `label` is excluded deliberately:
        it is an arbitrary watershed-assigned integer already pinned exactly
        by `schema` and `counts.cells`, and its min/max/mean say nothing
        about the segmentation."""
        r = record(n_cells=5)
        summary = r["numerics"]["cells"]
        assert set(summary) == {
            "area", "centroid_x", "centroid_y", "convexity", "elongation",
            "elongation_angle", "nuclear_aspect_ratio", "nuclear_angle",
            "perimeter", "shape_index",
        }
        assert "label" not in summary

    def test_round_trips_through_json(self):
        import json

        r = record()
        assert json.loads(json.dumps(r)) == r


class TestCompare:
    def test_identical_records_have_no_differences(self):
        assert compare(record(), record()) == []

    def test_a_changed_count_is_an_exact_difference(self):
        diffs = compare(record(n_cells=3), record(n_cells=4))
        paths = {d.path for d in diffs}
        assert "counts.cells" in paths
        assert all(d.kind == "exact" for d in diffs if d.path.startswith("counts"))

    def test_a_mask_independent_quantity_is_tight(self):
        """`frank` comes from the structure tensor of the fibre plane and never
        touches a mask, so a change of 1e-9 relative is real."""
        diffs = compare(record(splay=1.25), record(splay=1.25 * (1 + 1e-9)))
        assert any(d.path == "numerics.frank.splay" for d in diffs)

    def test_a_mask_dependent_quantity_absorbs_floating_point_noise(self):
        """`LOOSE` no longer exists to absorb a scikit-image version-boundary
        sliver (that scenario sits below the supported `>=0.25.2` floor,
        where results are bit-exact by measurement); it absorbs the much
        smaller floating-point non-associativity a different BLAS build or
        thread count could introduce for an otherwise bit-exact computation."""
        assert compare(record(area_mean=10.0), record(area_mean=10.0 * (1 + 1e-7))) == []

    def test_a_mask_dependent_quantity_no_longer_absorbs_a_version_boundary_sliver(self):
        """Mutation evidence for tightening `LOOSE` from `1e-3` to `1e-6`: a
        1e-5 relative change is representative of the pre-`0.25.2` watershed
        boundary drift the old tolerance was built to absorb (and did). It
        no longer passes."""
        diffs = compare(record(area_mean=10.0), record(area_mean=10.0 * (1 + 1e-5)))
        assert any(d.path == "numerics.cells.area.mean" for d in diffs)

    def test_a_mask_dependent_quantity_still_catches_a_real_change(self):
        diffs = compare(record(area_mean=10.0), record(area_mean=11.0))
        assert any(d.path == "numerics.cells.area.mean" for d in diffs)

    def test_environment_differences_are_reported_not_failed(self):
        a = record()
        b = record()
        b["environment"]["scikit-image"] = "0.25.2"
        diffs = compare(a, b)
        assert [d.kind for d in diffs] == ["environment"]

    def test_infinity_compares_equal_to_infinity(self):
        assert compare(record(corr_len=math.inf), record(corr_len=math.inf)) == []

    def test_nan_is_always_a_difference(self):
        """NaN never equals itself, and a NaN appearing where a number was is a
        defect, so it must surface rather than compare equal."""
        diffs = compare(record(corr_len=4.0), record(corr_len=math.nan))
        assert any("correlation_length" in d.path for d in diffs)

    def test_a_missing_key_is_a_difference_not_a_crash(self):
        a = record()
        b = record()
        del b["numerics"]["frank"]["splay"]
        diffs = compare(a, b)
        assert any("splay" in d.path for d in diffs)

    def test_a_missing_section_renders_the_sentinel_as_readable(self):
        """A missing section is the acceptance-critical path: `--check`
        fails there most often when a golden predates a schema change. The
        message must read as `<missing>`, not a raw `<object object at
        0x...>` sentinel address."""
        a = record()
        b = record()
        del b["segmentation"]
        diffs = compare(a, b)
        d = next(d for d in diffs if d.path == "segmentation")
        text = str(d)
        assert "<missing>" in text
        assert "object at 0x" not in text

    def test_difference_carries_both_values(self):
        diffs = compare(record(n_cells=3), record(n_cells=4))
        d = next(d for d in diffs if d.path == "counts.cells")
        assert (d.golden, d.current) == (3, 4)

    def test_an_extra_top_level_section_is_a_difference(self):
        """A section `build_record` gains later must be unprotected for zero
        commits, not until someone remembers to teach `compare` about it."""
        a = record()
        b = record()
        b["extra_section"] = {"foo": "bar"}
        diffs = compare(a, b)
        assert any(d.path == "extra_section" for d in diffs)

    def test_a_missing_top_level_section_is_a_difference(self):
        a = record()
        b = record()
        del b["segmentation"]
        diffs = compare(a, b)
        assert any(d.path == "segmentation" for d in diffs)

    def test_a_count_as_a_float_is_a_difference_even_when_numerically_equal(self):
        """`counts` is exact-compared, and Python's `3 == 3.0`, so without a
        type check a regressed `int()` cast in `build_record` would pass
        silently."""
        a = record()
        b = record()
        b["counts"]["cells"] = float(b["counts"]["cells"])
        diffs = compare(a, b)
        assert any(d.path == "counts.cells" for d in diffs)


class TestZeroCells:
    """A segmentation that finds nothing is a plausible real input, not a
    malformed record: `build_record` must not crash on it."""

    def test_build_record_does_not_crash_on_zero_cells(self):
        r = record(n_cells=0)
        assert r["counts"]["cells"] == 0
        assert r["numerics"]["cells"]["area"] is None

    def test_two_zero_cell_records_compare_equal(self):
        assert compare(record(n_cells=0), record(n_cells=0)) == []

    def test_a_zero_cell_record_differs_from_a_populated_one(self):
        diffs = compare(record(n_cells=0), record(n_cells=3))
        assert any(d.path == "numerics.cells.area" for d in diffs)


class TestMalformedNumerics:
    def test_a_numerics_section_that_is_not_a_mapping_is_a_difference_not_a_crash(self):
        a = record()
        b = record()
        b["numerics"] = None
        diffs = compare(a, b)
        assert any(d.path == "numerics" for d in diffs)
        assert all(d.kind == "exact" for d in diffs if d.path == "numerics")


class TestSchemaSection:
    def test_the_column_set_is_recorded(self):
        r = record()
        assert r["schema"] == sorted(
            [
                "label", "centroid_x", "centroid_y", "area", "perimeter",
                "shape_index", "convexity", "elongation", "elongation_angle",
                "nuclear_aspect_ratio", "nuclear_angle",
            ]
        )

    def test_a_dropped_column_is_a_difference(self):
        """A column vanishing from `cells` on both sides is invisible to the
        per-column summary loop, so the column list itself must be pinned."""
        a = record_from(fake_result())
        result = fake_result()
        result.cells = result.cells.drop("elongation")
        b = record_from(result)
        diffs = compare(a, b)
        assert any(d.path == "schema" for d in diffs)


class TestCorrelationCurve:
    def test_the_curve_shape_is_summarised(self):
        r = record(r_bins=(1.0, 2.0, 3.0), g_values=(0.9, 0.5, 0.1))
        curve = r["numerics"]["correlations"]
        assert curve["g_values"] == {"min": 0.1, "max": 0.9, "mean": pytest.approx(0.5)}
        assert curve["r_bins"] == {"first": 1.0, "last": 3.0}

    def test_a_changed_curve_with_unchanged_length_and_bin_count_is_a_difference(self):
        a = record(r_bins=(1.0, 2.0, 3.0), g_values=(0.9, 0.5, 0.1), corr_len=4.0)
        b = record(r_bins=(1.0, 2.0, 3.0), g_values=(0.9, 0.5, 0.9), corr_len=4.0)
        diffs = compare(a, b)
        assert any(d.path == "numerics.correlations.g_values.mean" for d in diffs)

    def test_the_fallback_case_is_null_not_a_crash(self):
        r = record(r_bins=(), g_values=())
        curve = r["numerics"]["correlations"]
        assert curve["g_values"] is None
        assert curve["r_bins"] is None

    def test_two_fallback_records_compare_equal(self):
        a = record(r_bins=(), g_values=())
        b = record(r_bins=(), g_values=())
        assert compare(a, b) == []

    def test_a_fallback_record_differs_from_a_real_one(self):
        a = record(r_bins=(), g_values=())
        b = record(r_bins=(1.0,), g_values=(0.9,))
        diffs = compare(a, b)
        paths = {d.path for d in diffs}
        assert "numerics.correlations.g_values" in paths
        assert "numerics.correlations.r_bins" in paths


class TestCellColumnCoverage:
    """Reviewer's own attack: scaling `label`, `centroid_x`, `centroid_y`,
    `convexity`, `elongation_angle`, `nuclear_aspect_ratio` and
    `nuclear_angle` by -7 produced no differences at all, because only
    `area`, `perimeter`, `shape_index` and `elongation` were summarised.
    `nuclear_aspect_ratio` and `nuclear_angle` are an advertised README
    feature."""

    @pytest.mark.parametrize("column", _UNPINNED_CELL_COLUMNS)
    def test_scaling_a_previously_unpinned_column_is_a_difference(self, column):
        a = record(n_cells=5)
        mutated = fake_result(n_cells=5)
        mutated.cells = mutated.cells.with_columns((pl.col(column) * -7).alias(column))
        b = record_from(mutated)
        diffs = compare(a, b)
        assert any(d.path.startswith(f"numerics.cells.{column}.") for d in diffs)

    def test_scaling_label_changes_nothing(self):
        """`label` is deliberately excluded from the numeric summary: it is
        an arbitrary watershed-assigned identifier, already pinned exactly
        by `schema` and `counts.cells`, and its min/max/mean carry no
        segmentation information. Scaling it must not manufacture a
        difference where the actual measurements are unchanged."""
        a = record(n_cells=5)
        mutated = fake_result(n_cells=5)
        mutated.cells = mutated.cells.with_columns((pl.col("label") * -7).alias("label"))
        b = record_from(mutated)
        diffs = compare(a, b)
        assert diffs == []


class TestFieldQuadrants:
    """Reviewer's own attack: reversing `theta`/`coherence`/`optimal_sigma`
    with `[::-1, ::-1]` left the whole-field mean and standard deviation
    unchanged, so the comparison passed at `TIGHT` with zero differences
    even though every value moved to a different pixel."""

    def _asymmetric_fields(self):
        base = np.arange(64, dtype=float).reshape(8, 8)
        return base * 0.01, base * 0.02 + 0.1, base * 0.03 + 1.0

    def test_quadrant_means_are_recorded(self):
        theta, coherence, optimal_sigma = self._asymmetric_fields()
        r = record(theta=theta, coherence=coherence, optimal_sigma=optimal_sigma)
        quadrants = r["numerics"]["fields"]["theta_quadrants"]
        assert set(quadrants) == {"top_left", "top_right", "bottom_left", "bottom_right"}
        assert quadrants["top_left"] != quadrants["bottom_right"]

    def test_reversing_a_field_spatially_is_a_difference(self):
        theta, coherence, optimal_sigma = self._asymmetric_fields()
        a = record(theta=theta, coherence=coherence, optimal_sigma=optimal_sigma)
        b = record(
            theta=theta[::-1, ::-1],
            coherence=coherence[::-1, ::-1],
            optimal_sigma=optimal_sigma[::-1, ::-1],
        )
        # Whole-field aggregates are order-independent (a reversal is just a
        # permutation of the same values), circular or not, so neither
        # catches the reversal alone -- that is exactly why the quadrant
        # breakdown exists.
        assert a["numerics"]["fields"]["theta_mean"] == b["numerics"]["fields"]["theta_mean"]
        assert (
            a["numerics"]["fields"]["theta_resultant_length"]
            == b["numerics"]["fields"]["theta_resultant_length"]
        )
        diffs = compare(a, b)
        paths = {d.path for d in diffs}
        assert any("_quadrants" in p for p in paths)


class TestCircularStatistics:
    """`theta`, `elongation_angle` and `nuclear_angle` are director/axis
    angles identified modulo pi (confirmed from the Rust source: see the
    module docstring). A plain arithmetic mean is not merely fragile for
    such an angle, it is the wrong statistic: averaging a value just under
    pi with a value just over 0 -- physically the same orientation --
    produces something near pi/2. An all-declared-floors run surfaced
    exactly this instability in `elongation_angle.mean`."""

    def test_a_wrap_straddling_field_recovers_the_true_orientation(self):
        """Two half-fields at 0.01 rad and pi - 0.01 rad are the same
        physical orientation to within 0.02 rad. A plain mean of the raw
        values reports ~pi/2 (physically nonsensical); the circular mean
        must recover ~0 (equivalently ~pi), not ~pi/2."""
        theta = np.full((8, 8), 0.01)
        theta[4:, :] = math.pi - 0.01

        plain_mean = float(theta.mean())
        assert plain_mean == pytest.approx(math.pi / 2, abs=0.05)

        r = record(theta=theta)
        circular_mean = r["numerics"]["fields"]["theta_mean"]
        # Wrapped into [0, pi): either just above 0 or just below pi are the
        # same physical answer under the modulo-pi identification.
        wrapped_distance_from_zero = min(circular_mean, math.pi - circular_mean)
        assert wrapped_distance_from_zero < 0.05
        assert abs(circular_mean - plain_mean) > 1.0

    def test_a_uniform_field_has_resultant_length_one(self):
        r = record(theta=np.full((8, 8), 1.0))
        assert r["numerics"]["fields"]["theta_resultant_length"] == pytest.approx(1.0)

    def test_a_uniformly_spread_field_has_a_small_resultant_length(self):
        # Angles evenly spaced over exactly one modulo-pi period cancel
        # perfectly in the doubled-angle sum.
        theta = np.linspace(0.0, math.pi, num=64, endpoint=False).reshape(8, 8)
        r = record(theta=theta)
        assert r["numerics"]["fields"]["theta_resultant_length"] == pytest.approx(0.0, abs=1e-9)

    def test_elongation_angle_column_uses_the_circular_mean(self):
        """Same wrap-straddling construction as the field test, applied to a
        `cells` column."""
        mutated = fake_result(n_cells=4)
        mutated.cells = mutated.cells.with_columns(
            pl.Series("elongation_angle", [0.01, 0.01, math.pi - 0.01, math.pi - 0.01])
        )
        r = record_from(mutated)
        summary = r["numerics"]["cells"]["elongation_angle"]
        assert set(summary) == {"mean", "resultant_length"}
        wrapped_distance_from_zero = min(summary["mean"], math.pi - summary["mean"])
        assert wrapped_distance_from_zero < 0.05
        assert summary["mean"] != pytest.approx(math.pi / 2, abs=0.1)

    def test_nuclear_angle_column_uses_the_circular_mean(self):
        mutated = fake_result(n_cells=4)
        mutated.cells = mutated.cells.with_columns(
            pl.Series("nuclear_angle", [0.01, 0.01, math.pi - 0.01, math.pi - 0.01])
        )
        r = record_from(mutated)
        summary = r["numerics"]["cells"]["nuclear_angle"]
        assert set(summary) == {"mean", "resultant_length"}
        wrapped_distance_from_zero = min(summary["mean"], math.pi - summary["mean"])
        assert wrapped_distance_from_zero < 0.05

    def test_angular_columns_are_still_caught_by_compare(self):
        """The reviewer's column-coverage attack, re-run specifically for
        the two angular columns after their statistic changed shape (from
        min/max/mean to mean/resultant_length): still caught."""
        a = record(n_cells=5)
        mutated = fake_result(n_cells=5)
        mutated.cells = mutated.cells.with_columns(
            (pl.col("elongation_angle") + 1.7).alias("elongation_angle")
        )
        b = record_from(mutated)
        diffs = compare(a, b)
        assert any(d.path.startswith("numerics.cells.elongation_angle.") for d in diffs)


class TestDefectCharges:
    """Reviewer's own attack: negating every detected charge on
    `phantom-radial` and rebuilding the record produced zero differences,
    because only `counts.defects` (a count, not a charge) was pinned."""

    def test_charge_sum_min_and_max_are_recorded(self):
        r = record(defect_charges=(0.5, -0.5, 1.0))
        assert r["numerics"]["defects"] == {
            "charge_sum": 1.0,
            "charge_min": -0.5,
            "charge_max": 1.0,
        }

    def test_zero_defects_records_a_zero_sum_and_null_extrema(self):
        r = record(defect_charges=())
        assert r["numerics"]["defects"] == {
            "charge_sum": 0.0,
            "charge_min": None,
            "charge_max": None,
        }

    def test_negating_every_charge_is_a_difference(self):
        a = record(defect_charges=(0.5, -0.5, 0.5, 0.5))
        b = record(defect_charges=(-0.5, 0.5, -0.5, -0.5))
        diffs = compare(a, b)
        assert any(d.path == "numerics.defects.charge_sum" for d in diffs)

    def test_a_changed_charge_with_an_unchanged_count_is_a_difference(self):
        """`counts.defects` alone would not catch this: the same number of
        defects, one different value."""
        a = record(defect_charges=(0.5, -0.5, 0.5))
        b = record(defect_charges=(0.5, -0.5, 1.0))
        diffs = compare(a, b)
        assert "counts.defects" not in {d.path for d in diffs}
        assert any(d.path == "numerics.defects.charge_max" for d in diffs)

    def test_charge_content_is_compared_exactly_not_under_a_tolerance(self):
        """A relative difference far below `LOOSE`, and even below `TIGHT`,
        must still fail: charge is discrete and quantised, so there is no
        boundary-pixel noise here for a tolerance to absorb."""
        a = record(defect_charges=(0.5,))
        b = record(defect_charges=(0.5 * (1 + 1e-13),))
        diffs = compare(a, b)
        matches = [d for d in diffs if d.path == "numerics.defects.charge_max"]
        assert matches
        assert matches[0].tolerance is None
