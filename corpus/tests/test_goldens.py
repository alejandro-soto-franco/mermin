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
                 r_bins=(1.0,), g_values=(0.9,)):
    cells = pl.DataFrame(
        {
            "label": list(range(1, n_cells + 1)),
            "area": [area_mean] * n_cells,
            "perimeter": [12.0] * n_cells,
            "shape_index": [3.8] * n_cells,
            "elongation": [0.4] * n_cells,
        },
        schema={
            "label": pl.Int64,
            "area": pl.Float64,
            "perimeter": pl.Float64,
            "shape_index": pl.Float64,
            "elongation": pl.Float64,
        },
    )
    return SimpleNamespace(
        cells=cells,
        fields={
            "theta": np.full((8, 8), 0.5),
            "coherence": np.full((8, 8), 0.25),
            "optimal_sigma": np.full((8, 8), 2.0),
        },
        defects=[{"position": (1, 1), "charge": 0.5, "angle": 0.0}],
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

    def test_a_mask_dependent_quantity_absorbs_a_boundary_pixel(self):
        """A watershed boundary sliver moving between scikit-image versions
        must not fail the suite."""
        assert compare(record(area_mean=10.0), record(area_mean=10.0 * (1 + 1e-5))) == []

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
            ["label", "area", "perimeter", "shape_index", "elongation"]
        )

    def test_a_dropped_column_is_a_difference(self):
        """A column vanishing from `cells` on both sides is invisible to the
        per-column summary loop, so the column list itself must be pinned."""
        a = build_record(
            fake_result(),
            entry="fake-entry",
            invocation={"segmentation": "threshold", "pixel_size_um": None},
            environment={"mermin": "0.5.0", "scikit-image": "0.26.0"},
        )
        result = fake_result()
        result.cells = result.cells.drop("elongation")
        b = build_record(
            result,
            entry="fake-entry",
            invocation={"segmentation": "threshold", "pixel_size_um": None},
            environment={"mermin": "0.5.0", "scikit-image": "0.26.0"},
        )
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
