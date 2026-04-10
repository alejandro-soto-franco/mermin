"""Tests for tissue-level analysis functions added in v0.2.0."""

import numpy as np
import pytest
from scipy.spatial import Delaunay

from mermin._native import (
    compute_cell_mean_coherence,
    compute_cell_orientations,
    detect_defects_delaunay,
    frank_energy_delaunay,
    nematic_order,
)


class TestCellOrientations:
    def test_uniform_field(self):
        h, w = 50, 50
        mask = np.zeros((h, w), dtype=np.int32)
        mask[5:20, 5:20] = 1
        mask[25:40, 25:40] = 2
        theta = np.full((h, w), 0.7)
        coherence = np.ones((h, w))
        results = compute_cell_orientations(mask, theta, coherence)
        assert len(results) == 2
        for r in results:
            assert abs(r["theta"] - 0.7) < 0.01

    def test_distinct_orientations(self):
        h, w = 50, 50
        mask = np.zeros((h, w), dtype=np.int32)
        mask[5:20, 5:20] = 1
        mask[25:40, 25:40] = 2
        theta = np.full((h, w), 0.3)
        theta[25:40, 25:40] = 1.5
        coherence = np.ones((h, w))
        results = compute_cell_orientations(mask, theta, coherence)
        assert abs(results[0]["theta"] - 0.3) < 0.01
        assert abs(results[1]["theta"] - 1.5) < 0.01


class TestCellMeanCoherence:
    def test_uniform_coherence(self):
        h, w = 30, 30
        mask = np.zeros((h, w), dtype=np.int32)
        mask[5:15, 5:15] = 1
        coherence = np.full((h, w), 0.6)
        results = compute_cell_mean_coherence(mask, coherence)
        assert len(results) == 1
        assert abs(results[0]["coherence"] - 0.6) < 0.01

    def test_heterogeneous_coherence(self):
        h, w = 40, 40
        mask = np.zeros((h, w), dtype=np.int32)
        mask[0:10, 0:10] = 1
        mask[20:30, 20:30] = 2
        coherence = np.full((h, w), 0.2)
        coherence[0:10, 0:10] = 0.95
        results = compute_cell_mean_coherence(mask, coherence)
        assert abs(results[0]["coherence"] - 0.95) < 0.01
        assert abs(results[1]["coherence"] - 0.2) < 0.01


class TestNematicOrder:
    def test_perfectly_aligned(self):
        thetas = np.array([0.5] * 200)
        result = nematic_order(thetas)
        assert abs(result["s"] - 1.0) < 1e-10

    def test_isotropic(self):
        thetas = np.linspace(0, np.pi, 2000, endpoint=False)
        result = nematic_order(thetas)
        assert abs(result["s"]) < 0.01

    def test_too_few_raises(self):
        with pytest.raises(ValueError):
            nematic_order(np.array([0.5]))


class TestDelaunayDefects:
    def _make_mesh(self, centroids, thetas):
        tri = Delaunay(np.array(centroids))
        simplices = [[int(s) for s in sim] for sim in tri.simplices.tolist()]
        return simplices

    def test_uniform_no_defects(self):
        centroids = [[0, 0], [10, 0], [5, 8.66], [15, 8.66]]
        thetas = [0.5, 0.5, 0.5, 0.5]
        simplices = self._make_mesh(centroids, thetas)
        defects = detect_defects_delaunay(centroids, thetas, simplices, 100.0)
        assert len(defects) == 0

    def test_edge_exclusion(self):
        centroids = [[0, 0], [200, 0], [100, 173]]
        thetas = [0.0, np.pi / 4, np.pi / 2]
        simplices = self._make_mesh(centroids, thetas)
        defects = detect_defects_delaunay(centroids, thetas, simplices, 50.0)
        assert len(defects) == 0


class TestDelaunayFrankEnergy:
    def _make_mesh(self, centroids):
        tri = Delaunay(np.array(centroids))
        return [[int(s) for s in sim] for sim in tri.simplices.tolist()]

    def test_uniform_zero_energy(self):
        centroids = [[0, 0], [10, 0], [5, 8.66], [15, 8.66]]
        thetas = [0.7, 0.7, 0.7, 0.7]
        simplices = self._make_mesh(centroids)
        energy = frank_energy_delaunay(centroids, thetas, simplices, 100.0)
        assert energy["splay"] < 1e-10
        assert energy["bend"] < 1e-10
        assert energy["n_triangles"] == 2

    def test_gradient_nonzero_energy(self):
        centroids = [[0, 0], [10, 0], [5, 8.66]]
        thetas = [0.0, np.pi / 4, np.pi / 2]
        simplices = self._make_mesh(centroids)
        energy = frank_energy_delaunay(centroids, thetas, simplices, 100.0)
        assert energy["splay"] + energy["bend"] > 0

    def test_ratio_is_finite(self):
        """A smooth linear gradient should produce a finite, positive ratio."""
        # Grid of cells with theta varying linearly in x
        centroids = []
        thetas = []
        for i in range(5):
            for j in range(5):
                centroids.append([i * 10.0, j * 10.0])
                thetas.append((i * 0.1) % np.pi)
        simplices = self._make_mesh(centroids)
        energy = frank_energy_delaunay(centroids, thetas, simplices, 20.0)
        assert energy["ratio"] > 0, f"ratio should be positive, got {energy['ratio']}"
        assert np.isfinite(energy["ratio"]), f"ratio should be finite, got {energy['ratio']}"
