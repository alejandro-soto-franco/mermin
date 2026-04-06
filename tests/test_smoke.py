"""Smoke test: verify the full pipeline runs on synthetic data."""

import numpy as np
import pytest


def test_shape_analysis_synthetic():
    """Test shape analysis on a synthetic hexagonal contour."""
    from mermin._native import analyze_shape

    pts = np.array(
        [[np.cos(i * np.pi / 3), np.sin(i * np.pi / 3)] for i in range(6)]
    )
    result = analyze_shape(pts)
    assert result["area"] > 0
    assert result["perimeter"] > 0
    assert result["shape_katic"].shape == (4,)


def test_structure_tensor_synthetic():
    """Test structure tensor on a synthetic striped image."""
    from mermin._native import compute_structure_tensor

    x = np.arange(100)
    image = np.tile(np.sin(2 * np.pi * x / 10), (100, 1))
    result = compute_structure_tensor(image, 3.0)
    assert result["theta"].shape == (100, 100)
    assert result["coherence"].shape == (100, 100)


def test_nuclear_ellipse_synthetic():
    """Test nuclear ellipse fitting on a synthetic circular mask."""
    from mermin._native import fit_nuclear_ellipses

    mask = np.zeros((100, 100), dtype=np.int32)
    for r in range(100):
        for c in range(100):
            if (r - 50) ** 2 + (c - 50) ** 2 <= 400:
                mask[r, c] = 1
    results = fit_nuclear_ellipses(mask)
    assert len(results) == 1
    assert abs(results[0]["aspect_ratio"] - 1.0) < 0.2
