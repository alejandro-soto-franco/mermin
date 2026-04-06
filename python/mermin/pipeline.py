"""End-to-end analysis pipeline."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from mermin.io import load_tiff
from mermin.segment import (
    build_neighbor_graph,
    extract_contours,
    segment_cell_bodies,
    segment_nuclei,
)


@dataclass
class AnalysisResult:
    """Complete analysis result for a single image."""

    cells: pl.DataFrame
    fields: dict[str, np.ndarray]
    defects: list[dict]
    correlations: dict[str, Any]
    frank: dict[str, float]
    ldg_params: dict[str, float]
    persistence: dict[str, Any]

    def summary(self) -> str:
        n = len(self.cells)
        n_def = len(self.defects)
        mean_psi2 = (
            self.cells["internal_katic_k2"].mean()
            if "internal_katic_k2" in self.cells.columns
            else 0.0
        )
        return (
            f"mermin analysis: {n} cells, {n_def} defects, "
            f"mean |psi_2| = {mean_psi2:.3f}, "
            f"Frank ratio = {self.frank.get('ratio', 0):.2f}"
        )


def analyze(
    path: str | Path,
    channels: dict[str, int] | None = None,
    pixel_size_um: float = 0.345,
    k_values: list[int] | None = None,
    structure_tensor_scales: list[float] | None = None,
    cellpose_diameter: float | None = None,
) -> AnalysisResult:
    """Run the full mermin analysis pipeline on a single image.

    Args:
        path: Path to multi-frame TIFF.
        channels: Channel mapping. Default: {"dapi": 0, "vimentin": 1}.
        pixel_size_um: Physical pixel size in micrometers.
        k_values: k-atic symmetry orders to analyze. Default: [1, 2, 4, 6].
        structure_tensor_scales: Gaussian sigma values in pixels.
            Default: [1, 2, 4, 8, 16, 32].
        cellpose_diameter: Nuclear diameter for Cellpose. None for auto.

    Returns:
        AnalysisResult with all measurements.
    """
    if k_values is None:
        k_values = [1, 2, 4, 6]
    if structure_tensor_scales is None:
        structure_tensor_scales = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]

    from mermin import _native

    # Stage 1: Load and preprocess
    images = load_tiff(path, channels)
    dapi = images["dapi"]
    vimentin = images["vimentin"]

    # Stage 2: Segmentation
    nuclear_mask = segment_nuclei(dapi, diameter=cellpose_diameter)
    cell_mask = segment_cell_bodies(vimentin, nuclear_mask)
    contours = extract_contours(cell_mask, pixel_size_um)

    # Nuclear ellipses
    nuclear_ellipses = _native.fit_nuclear_ellipses(nuclear_mask)

    # Build neighbor graph
    labels = sorted(contours.keys())
    centroids_px = []
    for label in labels:
        c = contours[label].mean(axis=0)
        centroids_px.append(c)
    centroids_arr = np.array(centroids_px) if centroids_px else np.zeros((0, 2))

    # Stage 3: Shape analysis (Rust)
    contour_arrays = [contours[label] for label in labels]
    shape_results = (
        _native.analyze_shapes_batch(contour_arrays) if contour_arrays else []
    )

    # Stage 4: Orientation (Rust)
    ms_result = _native.compute_multiscale_structure_tensor(
        vimentin, structure_tensor_scales
    )

    # Stage 5: Defect detection (Rust)
    mid_idx = len(structure_tensor_scales) // 2
    theta_field = ms_result["scale_results"][mid_idx]["theta"]
    ny, nx = theta_field.shape

    defects = _native.detect_defects(
        theta_field.ravel().tolist(), nx, ny, 2, np.pi / 2
    )

    # Stage 6: Correlations (Rust)
    if len(labels) >= 3:
        cell_thetas = []
        for i, label in enumerate(labels):
            row = int(centroids_px[i][1] / pixel_size_um)
            col = int(centroids_px[i][0] / pixel_size_um)
            row = min(row, ny - 1)
            col = min(col, nx - 1)
            cell_thetas.append(float(theta_field[row, col]))

        correlations = _native.orientational_correlation(
            centroids_arr.tolist(),
            cell_thetas,
            2,
            max(nx, ny) * pixel_size_um * 0.5,
            20,
        )
    else:
        correlations = {
            "r_bins": [],
            "g_values": [],
            "correlation_length": float("inf"),
        }

    # Stage 7: Frank energy + theory (Rust)
    frank = _native.frank_energy(
        theta_field.ravel().tolist(), nx, ny, pixel_size_um
    )

    xi = correlations.get("correlation_length", 10.0)
    s_values = [sr.get("elongation", 0.0) for sr in shape_results]
    ldg = _native.estimate_ldg_params(s_values, xi, pixel_size_um)

    # Build per-cell DataFrame
    records = []
    for i, label in enumerate(labels):
        sr = shape_results[i] if i < len(shape_results) else {}
        ne = next((e for e in nuclear_ellipses if e["label"] == label), {})
        records.append(
            {
                "label": label,
                "centroid_x": centroids_px[i][0] if i < len(centroids_px) else 0.0,
                "centroid_y": centroids_px[i][1] if i < len(centroids_px) else 0.0,
                "area": sr.get("area", 0.0),
                "perimeter": sr.get("perimeter", 0.0),
                "shape_index": sr.get("shape_index", 0.0),
                "convexity": sr.get("convexity", 0.0),
                "elongation": sr.get("elongation", 0.0),
                "elongation_angle": sr.get("elongation_angle", 0.0),
                "nuclear_aspect_ratio": ne.get("aspect_ratio", 0.0),
                "nuclear_angle": ne.get("angle", 0.0),
            }
        )

    cells_df = pl.DataFrame(records) if records else pl.DataFrame()

    persistence = {"pairs": []}

    return AnalysisResult(
        cells=cells_df,
        fields={
            "theta": theta_field,
            "coherence": ms_result["scale_results"][mid_idx]["coherence"],
            "optimal_sigma": ms_result["optimal_sigma"],
        },
        defects=defects,
        correlations=correlations,
        frank=frank,
        ldg_params=ldg,
        persistence=persistence,
    )


@dataclass
class Experiment:
    """Batch analysis with condition comparison."""

    pixel_size_um: float = 0.345
    conditions: dict[str, list[str]] = field(default_factory=dict)

    def add_condition(self, name: str, paths: list[str]):
        self.conditions[name] = paths

    def run(self) -> "ComparisonResult":
        results = {}
        for cond, paths in self.conditions.items():
            results[cond] = [
                analyze(p, pixel_size_um=self.pixel_size_um) for p in paths
            ]
        return ComparisonResult(results)


@dataclass
class ComparisonResult:
    """Result of comparing multiple conditions."""

    results: dict[str, list[AnalysisResult]]

    def report(self, output_dir: str | Path):
        """Generate summary report."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        summary = {}
        for cond, res_list in self.results.items():
            summary[cond] = {
                "n_images": len(res_list),
                "summaries": [r.summary() for r in res_list],
            }

        with open(out / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
