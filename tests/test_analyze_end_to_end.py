"""End-to-end smoke test for `analyze()` against a real, built `_native`.

Nothing else in the suite calls `analyze()` against actual image data: the
existing pipeline-contract tests check signatures and provenance plumbing
with the ingest layer mocked, and `test_smoke.py`/`test_tissue_analysis.py`
exercise individual `_native` functions in isolation. None of that would
have caught the Stage 5-7 call sites in `pipeline.py` passing Python lists
where the PyO3 bindings require numpy arrays, because none of them drove a
real image through the whole pipeline. This test does.

Skips cleanly when `mermin._native` is not built (the ephemeral `uv run`
environment used for Tasks 1-4), and runs for real in CI, which does build
it.
"""

import numpy as np
import pytest
import tifffile

pytest.importorskip(
    "mermin._native", reason="mermin._native is not built in this checkout"
)

from mermin.pipeline import analyze


def _write_synthetic_two_channel_tif(path):
    """A CYX tif with a nuclear channel holding six discs and a striped
    fibre channel, wavelength metadata so role resolution needs no explicit
    channel mapping.

    Six discs, not three: `analyze()` only takes the `orientational_correlation`
    branch in `pipeline.py` when `len(labels) >= 3`, and a fixture holding
    exactly three cells covers that branch by a margin of one. Six keeps the
    coverage clear of the threshold rather than incidental to it.
    """
    h, w = 96, 96
    nuclear = np.zeros((h, w), dtype=np.uint16)
    rng = np.random.default_rng(0)
    centres = [
        (20, 20), (20, 50), (20, 80),
        (70, 20), (70, 50), (70, 80),
    ]
    for cy, cx in centres:
        yy, xx = np.ogrid[:h, :w]
        disc = (yy - cy) ** 2 + (xx - cx) ** 2 <= 8**2
        nuclear[disc] = 4000 + rng.integers(0, 200)
    nuclear += rng.integers(0, 50, size=(h, w)).astype(np.uint16)

    x = np.arange(w)
    fibre = np.tile(np.sin(2 * np.pi * x / 12) * 2000 + 2000, (h, 1)).astype(
        np.uint16
    )

    data = np.stack([nuclear, fibre], axis=0)
    labels = [
        '<MetaData><PlaneInfo><prop id="wavelength" type="float" value="405"/>'
        "</PlaneInfo></MetaData>",
        '<MetaData><PlaneInfo><prop id="wavelength" type="float" value="568"/>'
        "</PlaneInfo></MetaData>",
    ]
    tifffile.imwrite(
        path,
        data,
        imagej=True,
        metadata={"axes": "CYX", "Labels": labels},
    )
    return path


def test_analyze_runs_end_to_end_on_a_synthetic_image(tmp_path):
    path = _write_synthetic_two_channel_tif(tmp_path / "synthetic.tif")

    result = analyze(path, pixel_size_um=0.25, segmentation="threshold")

    expected_columns = {
        "label",
        "centroid_x",
        "centroid_y",
        "area",
        "perimeter",
        "shape_index",
        "convexity",
        "elongation",
        "elongation_angle",
        "nuclear_aspect_ratio",
        "nuclear_angle",
    }
    assert expected_columns.issubset(set(result.cells.columns))

    # Six nuclei, comfortably above the `len(labels) >= 3` threshold that
    # gates the `orientational_correlation` branch in `pipeline.py`, so this
    # assertion states the branch coverage rather than leaving it incidental.
    assert len(result.cells) >= 6

    assert result.segmentation["backend"] == "threshold"

    assert result.fields["theta"].shape == (96, 96)

    # The correlation branch ran: real bins came back from
    # `_native.orientational_correlation`, not the `len(labels) < 3` fallback
    # dict of empty `r_bins`/`g_values` and an infinite correlation length.
    # (The synthetic fibre field is uniform, so an infinite correlation
    # length is itself a correct fit result here and is not a useful signal
    # of which branch ran.)
    assert len(result.correlations["r_bins"]) > 0
    assert len(result.correlations["g_values"]) > 0

    assert set(result.frank.keys()) == {"splay", "bend", "ratio"}
    assert set(result.ldg_params.keys()) == {"a", "b", "c", "k_elastic"}
