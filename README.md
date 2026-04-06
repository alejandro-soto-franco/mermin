[![CI](https://github.com/alejandro-soto-franco/mermin/actions/workflows/ci.yml/badge.svg)](https://github.com/alejandro-soto-franco/mermin/actions/workflows/ci.yml)
[![Python](https://github.com/alejandro-soto-franco/mermin/actions/workflows/python.yml/badge.svg)](https://github.com/alejandro-soto-franco/mermin/actions/workflows/python.yml)
[![Crates.io](https://img.shields.io/crates/v/mermin.svg)](https://crates.io/crates/mermin)
[![PyPI](https://img.shields.io/pypi/v/mermin.svg)](https://pypi.org/project/mermin/)
[![docs.rs](https://docs.rs/mermin/badge.svg)](https://docs.rs/mermin)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![MSRV](https://img.shields.io/badge/MSRV-1.85-blue.svg)](Cargo.toml)

# mermin

**k-atic alignment analysis of fluorescence microscopy.**

Named after [N. David Mermin](https://en.wikipedia.org/wiki/N._David_Mermin), whose 1979 *Reviews of Modern Physics* paper "The topological theory of defects in ordered media" provides the mathematical framework this tool implements on experimental microscopy data.

mermin takes raw multi-channel fluorescence TIFFs (e.g. DAPI + vimentin) and produces a complete physical analysis of cell alignment: shape descriptors, orientational order parameters, topological defects, spatial statistics, and continuum theory parameter estimates.

## Features

- **Minkowski tensor shape analysis**: W_0 (area), W_1 (perimeter), W_1^{1,1} (elongation tensor), W_1^{s,0} (k-atic shape modes for k = 1, 2, 4, 6)
- **Fourier boundary decomposition**: secondary shape descriptor for comparison with Minkowski tensors
- **Multiscale structure tensor**: orientation and coherence fields at logarithmically spaced scales (subcellular to tissue-level)
- **k-atic order parameter fields**: psi_k(x, sigma) = C * exp(ik * theta) for arbitrary k
- **Nuclear ellipse fitting**: aspect ratio and orientation from DAPI masks via moments of inertia
- **Topological defect detection**: half-integer and integer charge defects via [cartan](https://crates.io/crates/cartan) holonomy, with Poincare-Hopf validation
- **Persistent homology**: boundary matrix reduction on Delaunay filtration by ascending alignment magnitude
- **Orientational correlation functions**: G_k(r) with exponential fit for correlation length xi_k
- **Ripley's K-function**: spatial clustering analysis for defect point patterns
- **Spatial block bootstrap**: confidence intervals that respect spatial autocorrelation
- **Permutation tests**: condition comparison with proper null distribution
- **Frank elastic energy**: splay and bend decomposition from the director field
- **Landau-de Gennes parameter fitting**: extract (a, b, c, K) from experimental Q-tensor fields
- **Activity estimation**: zeta_eff from defect density via mean-field active nematic theory
- **Volterra-compatible output**: fitted parameters exported as JSON for forward simulation with [volterra](https://crates.io/crates/volterra-nematic)

## Install

### Python (recommended for end users)

```bash
pip install mermin
```

Requires Python 3.10+. The Rust extension is compiled automatically via [maturin](https://www.maturin.rs/).

### Rust (for library use)

```toml
[dependencies]
mermin = "0.1"
```

### Build from source

```bash
git clone https://github.com/alejandro-soto-franco/mermin.git
cd mermin
pip install maturin
maturin develop --release
```

## Quick Start

### Python

```python
import mermin

# Single image analysis
result = mermin.analyze(
    "path/to/image.tif",
    channels={"dapi": 0, "vimentin": 1},
    pixel_size_um=0.345,
)

print(result.summary())
# mermin analysis: 847 cells, 12 defects, mean |psi_2| = 0.412, Frank ratio = 1.31

# Per-cell measurements as a polars DataFrame
result.cells.head()

# Batch experiment with condition comparison
experiment = mermin.Experiment(pixel_size_um=0.345)
experiment.add_condition("ctrl", ["d01.tif", "d02.tif", "d03.tif"])
experiment.add_condition("tgfb1", ["d07.tif", "d08.tif", "d09.tif"])
comparison = experiment.run()
comparison.report("output/")
```

### Rust

```rust
use mermin::shape::{minkowski_w0, minkowski_w1_tensor, elongation_from_w1_tensor};
use mermin::orient::{structure_tensor, katic_order_field};
use mermin::topo::detect_defects;
use mermin::{BoundaryContour, ImageField, Point2};

// Shape analysis on a cell boundary
let contour = BoundaryContour::new(points)?;
let area = minkowski_w0(&contour);
let tensor = minkowski_w1_tensor(&contour);
let (elongation, angle) = elongation_from_w1_tensor(&tensor);

// Orientation field from vimentin channel
let st = structure_tensor(&vimentin_field, 4.0);
let psi2 = katic_order_field(&st, 2);

// Defect detection
let defects = detect_defects(&cell_thetas, nx, ny, 2, std::f64::consts::FRAC_PI_2);
```

## Crate Structure

| Crate | Description |
|-------|-------------|
| **mermin** | Facade crate, re-exports everything |
| **mermin-core** | `CellRecord`, `ImageField`, `BoundaryContour`, `KValue`, error types |
| **mermin-shape** | Minkowski tensors, Fourier decomposition, shape index, convexity |
| **mermin-orient** | Multiscale structure tensor, k-atic order parameter fields, nuclear ellipse fitting |
| **mermin-topo** | Defect detection (via [cartan-geo](https://crates.io/crates/cartan-geo) holonomy), Poincare-Hopf validation, persistent homology |
| **mermin-stats** | Orientational correlation G_k(r), Ripley's K, spatial block bootstrap, permutation tests |
| **mermin-theory** | Frank energy, Landau-de Gennes fitting, activity estimation, volterra-compatible JSON output |
| **mermin-py** | PyO3 bindings exposing all crates to Python |

## Analysis Pipeline

```
TIFF (DAPI + Vimentin)
  |
  +-- 1. Preprocessing ---- background subtraction, contrast normalization
  |
  +-- 2. Segmentation ----- Cellpose (nuclei), watershed (cell bodies), Delaunay graph
  |
  +-- 3. Shape analysis ---- Minkowski tensors, Fourier modes, morphometrics
  |
  +-- 4. Orientation ------- multiscale structure tensor, k-atic fields, nuclear ellipse
  |
  +-- 5. Topology ---------- defect detection (holonomy), Poincare-Hopf, persistence
  |
  +-- 6. Statistics -------- G_k(r), Ripley's K, block bootstrap, permutation tests
  |
  +-- 7. Theory ------------ Frank energy, Landau-de Gennes fit, activity estimation
  |
  +-- Output: per-cell CSV, field arrays, JSON statistics, HTML report
```

## Three Independent k-atic Measurements

mermin extracts three independent orientational measurements per cell, each with distinct physical meaning:

| Measurement | Source | What it captures |
|-------------|--------|-----------------|
| **Shape k-atic** | Minkowski tensors W_1^{s,0} on cell boundary | How the cell is shaped (elongation, polygonality) |
| **Internal k-atic** | Structure tensor of vimentin within cell territory | How cytoskeletal fibers are organized inside |
| **Collective k-atic** | Neighbor correlations on Delaunay graph | How aligned the cell is with its neighbors |

Agreement or disagreement between these layers is itself diagnostic. A TGF-beta-treated myofibroblast shows concordance across all three. A ROCK-inhibited cell may show a round shape (low shape k=2) but residual internal fiber alignment (higher internal k=2).

## Performance

All numerics run in Rust with rayon parallelization. Benchmarked against scikit-image, scipy, shapely, and numpy on a 16-thread AMD Ryzen CPU:

| Operation | mermin | Reference | Speedup |
|-----------|--------|-----------|---------|
| Structure tensor (1000x1000, sigma=4) | 31 ms | scikit-image 73 ms | **2.4x faster** |
| Multiscale structure tensor (1000x1000, 6 scales) | 234 ms | scikit-image 823 ms | **3.5x faster** |
| Structure tensor at microscopy scale (4015x4015) | 688 ms | scikit-image 1793 ms | **2.6x faster** |
| Nuclear ellipse fitting (200 nuclei) | 0.7 ms | scikit-image regionprops 22 ms | **31x faster** |
| Orientational correlation G_k(r) (1000 cells) | 4.8 ms | numpy 426 ms | **88x faster** |
| Shape analysis (500 cells, full descriptors) | 51 ms | shapely (area+perim only) 13 ms | 4x slower, but computes 10x more per cell |

The structure tensor pipeline (Scharr gradient, fused triple Gaussian blur, eigendecomposition) is parallelized end-to-end via rayon with `unsafe` interior-pixel fast paths. Shape analysis computes Minkowski tensors, Fourier spectrum, convexity, and k-atic modes for every cell in a single pass.

Precision: polygon area and perimeter match shapely to machine precision (rel err = 0). Frank energy splay/bend agreement with numpy at rel err < 1e-14.

## Dependencies

mermin builds on the [cartan](https://crates.io/crates/cartan) ecosystem for differential geometry:
- **cartan-geo**: holonomy-based topological defect detection
- **cartan-optim**: Riemannian trust region for Landau-de Gennes fitting

Python dependencies: cellpose, scikit-image, scipy, polars, matplotlib, tifffile.

## License

MIT
