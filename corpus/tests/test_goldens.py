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


def fake_result(*, n_cells=3, area_mean=10.0, splay=1.25, corr_len=4.0):
    cells = pl.DataFrame(
        {
            "label": list(range(1, n_cells + 1)),
            "area": [area_mean] * n_cells,
            "perimeter": [12.0] * n_cells,
            "shape_index": [3.8] * n_cells,
            "elongation": [0.4] * n_cells,
        }
    )
    return SimpleNamespace(
        cells=cells,
        fields={
            "theta": np.full((8, 8), 0.5),
            "coherence": np.full((8, 8), 0.25),
            "optimal_sigma": 2.0,
        },
        defects=[{"position": (1, 1), "charge": 0.5, "angle": 0.0}],
        correlations={"correlation_length": corr_len, "r_bins": [1.0], "g_values": [0.9]},
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
