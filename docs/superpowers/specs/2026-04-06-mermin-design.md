# mermin: k-atic alignment analysis of fluorescence microscopy

**Date**: 2026-04-06
**Status**: Design approved, pending implementation plan

## Overview

mermin is an open-source tool for evaluating k-atic alignment of cells in fluorescence TIFF images. Named after N. David Mermin, whose 1979 Reviews of Modern Physics paper "The topological theory of defects in ordered media" provides the mathematical framework for classifying orientational order and topological defects that this tool implements on experimental microscopy data.

The tool takes raw multi-channel fluorescence TIFFs and produces a complete physical and numerical analysis of cell alignment, shape, orientational order, topological defects, and continuum-theory parameter estimates.

## Architecture

Three-tier design: Python layer (IO, segmentation, visualization), Rust core (numerics via PyO3), cartan ecosystem (differential geometry).

### Rust Workspace (8 crates)

| Crate | Purpose | Dependencies |
|-------|---------|-------------|
| `mermin-core` | Types (`CellRecord`, `ImageField`, `BoundaryContour`), traits (`KAticAnalysis`, `ShapeDescriptor`), error types, coordinate systems | nalgebra, thiserror |
| `mermin-shape` | Minkowski tensors W_1^{s,0}, boundary Fourier decomposition, shape index p_0 = P/sqrt(A), cell morphometrics | mermin-core, nalgebra |
| `mermin-orient` | Multiscale structure tensor J_sigma, k-atic order parameter psi_k(x), per-cell internal alignment, scale-space coherence | mermin-core, ndarray, rayon |
| `mermin-topo` | Defect detection (wraps cartan-geo holonomy), charge classification, Poincare-Hopf validation, persistent homology | mermin-core, cartan-geo, cartan-manifolds |
| `mermin-stats` | G_k(r) correlation functions, Ripley's K, spatial block bootstrap, permutation tests | mermin-core, rand, rayon |
| `mermin-theory` | Frank energy decomposition, Landau-de Gennes fitting, activity estimation, volterra-compatible parameter output | mermin-core, mermin-orient, cartan-optim |
| `mermin-py` | PyO3 bindings exposing all crates as `mermin._native` | all above, pyo3, numpy |
| `mermin` | Facade crate re-exporting everything | all above |

### Python Package

| Module | Purpose |
|--------|---------|
| `mermin.io` | TIFF loading, channel separation, pixel size extraction, batch discovery |
| `mermin.segment` | Cellpose wrapper (nuclei), marker-controlled watershed (cell bodies), boundary contour extraction |
| `mermin.viz` | Orientation overlays, defect maps, Minkowski ellipse overlays, correlation plots, persistence diagrams, HTML/PDF report |
| `mermin.pipeline` | End-to-end orchestration, condition-aware batch mode, automatic statistical comparison |

### PyO3 Boundary

| Python passes | Rust receives | Rust returns |
|--------------|---------------|-------------|
| Vimentin channel (numpy f64) | `PyReadonlyArray2<f64>` | Structure tensor fields, psi_k fields as numpy arrays |
| Cell mask (numpy i32) | `PyReadonlyArray2<i32>` | Per-cell CellRecord list (converted to polars DataFrame Python-side) |
| Boundary contours (list of Nx2 arrays) | `Vec<PyReadonlyArray2<f64>>` | Minkowski tensors, Fourier coefficients per cell |
| Director field + Delaunay mesh | Arrays + index arrays | Defect list, Frank energies, correlation functions |
| Q-tensor field (numpy) | `PyReadonlyArray2<f64>` | LdG fitted parameters |

## Analysis Pipeline (per-image)

### Stage 1: Preprocessing (Python: mermin.io)

- Load 16-bit multi-frame TIFF, separate DAPI (frame 0) and vimentin (frame 1)
- Background subtraction: rolling-ball or morphological opening
- Contrast normalization: percentile-based stretch to [0, 1] float
- Optional denoising: non-local means on vimentin channel (preserves fiber structure)

### Stage 2: Segmentation (Python: mermin.segment)

- **Nuclei**: Cellpose `nuclei` model on DAPI channel, instance segmentation
- **Cell bodies**: marker-controlled watershed on vimentin, seeded by nuclear centroids, distance-transform energy landscape
- **Boundary extraction**: marching squares on each cell mask, subpixel refinement via gradient interpolation, ordered contour points
- **Neighbor graph**: Delaunay triangulation of centroids, adjacency list, edge lengths

### Stage 3: Shape Analysis (Rust: mermin-shape)

- Per-cell boundary contour processed into **Minkowski tensors**:
  - W_0 = area
  - W_1 = perimeter
  - W_2 = Euler characteristic (= 1 for simply connected)
  - W_1^{1,1} = rank-2 orientation tensor (eigenvalues give elongation magnitude and orientation)
  - W_1^{s,0} for s = 2, 3, 4, 6 (k-atic shape modes)
- **Fourier boundary decomposition** as secondary descriptor: a_k = (1/N) sum r_i exp(-ik theta_i)
- Shape index p_0 = P/sqrt(A), convexity = A/A_convex_hull, solidity
- All stored in CellRecord struct

### Stage 4: Orientation Extraction (Rust: mermin-orient)

- **Multiscale structure tensor** on vimentin at sigma in {1, 2, 4, 8, 16, 32} pixels:
  - J_sigma(x) = G_sigma * (grad I tensor grad I)
  - Eigendecompose: theta(x, sigma), coherence C(x, sigma) = (lambda_1 - lambda_2)/(lambda_1 + lambda_2)
- **k-atic order parameter field**: psi_k(x, sigma) = C(x, sigma) * exp(ik theta(x, sigma)) for k = 1, 2, 4, 6
- **Per-cell internal alignment**: average psi_k over each cell territory at each scale
- **Scale-space summary**: per cell, the scale sigma* maximizing |psi_2| (characteristic alignment length)
- **Nuclear ellipse fitting**: moments of inertia of nuclear mask, axis ratio and orientation (independent of vimentin)

### Stage 5: Topological Analysis (Rust: mermin-topo)

- Coarse-grain director field onto Delaunay mesh (one director per cell)
- **Defect detection** via cartan-geo `scan_disclinations()`:
  - Nematic (k=2): +/- 1/2 charge defects
  - Polar (k=1): +/- 1 charge defects
  - Hexatic (k=6): +/- 1/6 charge defects
- **Poincare-Hopf validation**: sum of charges = chi (Euler characteristic of domain)
- **Persistent homology**: filtration of Delaunay complex by ascending |psi_k|
  - H_0 persistence: robust alignment domains
  - H_1 persistence: orientational vortices
  - Computed via boundary matrix reduction

### Stage 6: Statistical Analysis (Rust: mermin-stats)

- **Orientational correlation function**: G_k(r) = <cos(k(theta(0) - theta(r)))>, binned by pairwise centroid distance, exponential fit for correlation length xi_k
- **Ripley's K-function** for defect point patterns, with Besag's L(r) = sqrt(K(r)/pi) - r, edge-corrected
- **Spatial block bootstrap** (block size ~ xi_k) for confidence intervals on population statistics
- **Permutation tests**: shuffle condition labels across wells, recompute statistic, build null distribution. p-values for: mean |psi_k|, defect density, xi_k, Frank energy ratio

### Stage 7: Continuum Theory (Rust: mermin-theory)

- **Frank energy**: splay = (div n)^2, bend = |n cross curl n|^2 on Delaunay mesh using cartan-dec operators. Total F_splay, F_bend, ratio
- **Landau-de Gennes fitting**: minimize integral [F_bulk(Q) + (K/2)|grad Q|^2] dA over (a, b, c, K) using cartan-optim Riemannian trust region
- **Activity estimation**: zeta_eff ~ K * rho_defect^2 (mean-field active nematic theory)
- **Output**: parameter set {a, b, c, K, zeta_eff} in volterra-compatible JSON

## Python API

```python
import mermin

# Single image
result = mermin.analyze(
    "path/to/d03_dv.tif",
    channels={"dapi": 0, "vimentin": 1},
    pixel_size_um=0.325,
    k_values=[1, 2, 4, 6],
    structure_tensor_scales=[1, 2, 4, 8, 16, 32],
)

# result.cells         -> polars DataFrame (per-cell measurements)
# result.fields        -> dict of numpy arrays
# result.defects       -> list of Defect(charge, position, k)
# result.correlations  -> dict of G_k(r) with xi_k fits
# result.frank         -> FrankEnergy(splay, bend, ratio)
# result.ldg_params    -> LandauDeGennes(a, b, c, K, zeta_eff)
# result.persistence   -> PersistenceDiagram(birth, death, dimension)
# result.summary()     -> human-readable summary

# Batch with condition comparison
experiment = mermin.Experiment(pixel_size_um=0.325)
experiment.add_condition("ctrl",        ["d01_dv.tif", "d02_dv.tif", "d03_dv.tif"])
experiment.add_condition("estradiol",   ["d04_dv.tif", "d05_dv.tif", "d06_dv.tif"])
experiment.add_condition("tgfb1",       ["d07_dv.tif", "d08_dv.tif", "d09_dv.tif"])
experiment.add_condition("rocki_tgfb1", ["d10_dv.tif", "d11_dv.tif", "d12_dv.tif"])

comparison = experiment.run()
comparison.report("output/")
```

## CLI

```bash
mermin analyze d03_dv.tif --pixel-size 0.325 --output results/
mermin batch experiment.toml --output results/ --report
```

## Output Artifacts

| Artifact | Format | Contents |
|----------|--------|----------|
| Per-cell table | CSV/Parquet | Area, perimeter, shape index, Minkowski eigenvalues, elongation angle, internal psi_k at each scale, nuclear ellipticity, neighbor count |
| Field data | NPY/Zarr | Structure tensor field, psi_k fields, director field, coherence maps at each scale |
| Population statistics | JSON | G_k(r), xi_k, defect census (+/- charges), Ripley's K, persistence diagrams, Frank energies, LdG parameters |
| Condition comparison | JSON | Permutation test p-values, effect sizes, bootstrap CIs per metric pair |
| Report | HTML/PDF | All figures, condition comparison tables, statistical test results, parameter estimates |

## Key Design Decisions

1. **Minkowski tensors over pure Fourier decomposition**: handles non-convex and irregular cell boundaries correctly, which Fourier r(theta) does not. Fourier retained as secondary descriptor for comparison.

2. **Multiscale structure tensor**: no single sigma captures the full picture. Subcellular (1-3 um), cellular (10-20 um), and tissue-level (50-100 um) alignment are distinct phenomena. ROCKi may destroy cellular alignment while preserving subcellular fiber bundles.

3. **Three independent k-atic measurements per cell**: shape (Minkowski), internal (structure tensor), collective (neighbor correlations). Agreement/disagreement between layers is itself diagnostic.

4. **Persistent homology**: topologically robust characterization of orientation field structure that correlation functions cannot capture. Noise-invariant.

5. **Spatial block bootstrap**: ordinary bootstrap assumes independence; cells in a tissue are spatially correlated. Block size matched to correlation length.

6. **cartan as geometry backend**: holonomy-based defect detection, Riemannian optimization for LdG fitting, DEC operators for Frank energy. Avoids reimplementing differential geometry.

7. **No volterra/pathwise dependency in v1**: mermin extracts parameters, outputs volterra-compatible JSON. Forward modeling is a documented future direction.

## Future Directions

### Stochastic Forward Modeling (mermin + volterra + pathwise)

The parameters extracted by mermin (a, b, c, K, zeta_eff) fully specify a Beris-Edwards active nematic model. The research pipeline:

1. **mermin** extracts continuum parameters from experimental microscopy
2. **volterra** runs deterministic Beris-Edwards simulation with those parameters
3. **pathwise-geo** adds stochastic perturbations via Riemannian SDEs on the QTensor manifold (cartan provides the geometry)
4. **Ensemble forecasting**: Monte Carlo ensemble of stochastic trajectories gives probability distributions over future orientation field states

This closes the experiment-theory-prediction loop: measure cell alignment, fit a physical model, forecast evolution with uncertainty quantification. Applications include predicting tissue remodeling under pharmacological intervention (e.g., will ROCKi treatment lead to isotropic remodeling or retain residual alignment domains?).

The mathematical framework: the orientation field Q(x,t) evolves as a stochastic PDE on the QTensor3 manifold. pathwise-geo provides manifold-valued SDE integrators (Euler-Maruyama, Milstein, Taylor 1.5 on Riemannian manifolds). volterra provides the deterministic drift (Beris-Edwards equations). The noise term models thermal fluctuations and active noise from molecular motors.

### Additional Future Work

- 3D z-stack analysis (volumetric k-atic fields, disclination lines via cartan-geo 3D detection)
- Time-lapse support (temporal correlation functions, defect tracking via cartan-geo DisclinationEvent)
- Additional fluorescent channels (actin for stress fiber analysis, alpha-SMA for myofibroblast confirmation)
- GPU acceleration via wgpu or CUDA for structure tensor on large images
- Integration with napari for interactive visualization

## Test Dataset

See `mermin-tests/README.md` for full metadata on the collaborator-provided human vaginal fibroblast dataset (8 TIFFs, 4 conditions x 2 substrates, DAPI + vimentin).
