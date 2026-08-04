[![CI](https://github.com/alejandro-soto-franco/mermin/actions/workflows/ci.yml/badge.svg)](https://github.com/alejandro-soto-franco/mermin/actions/workflows/ci.yml)
[![Python](https://github.com/alejandro-soto-franco/mermin/actions/workflows/python.yml/badge.svg)](https://github.com/alejandro-soto-franco/mermin/actions/workflows/python.yml)
[![Crates.io](https://img.shields.io/crates/v/mermin.svg)](https://crates.io/crates/mermin)
[![PyPI](https://img.shields.io/pypi/v/mermin.svg)](https://pypi.org/project/mermin/)
[![docs.rs](https://docs.rs/mermin/badge.svg)](https://docs.rs/mermin)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![MSRV](https://img.shields.io/badge/MSRV-1.85-blue.svg)](Cargo.toml)

# mermin

**$k$-atic alignment analysis of fluorescence microscopy.**

Named after [N. David Mermin](https://en.wikipedia.org/wiki/N._David_Mermin), whose 1979 *Reviews of Modern Physics* paper "The topological theory of defects in ordered media" provides the mathematical framework this tool implements on experimental microscopy data.

mermin takes multi-channel fluorescence microscopy images (TIFF, OME-TIFF, OME-Zarr, and other formats [bioio](https://github.com/bioio-devs/bioio) supports) and analyses cell alignment: per-cell shape descriptors, an orientation field with topological defect detection, an orientational correlation function, and Frank energy and Landau-de Gennes parameter estimates. The nuclear and fibre channel roles are resolved from file metadata, or supplied explicitly.

## Features

- **Minkowski tensor shape analysis**: $W_0$ (area), $W_1$ (perimeter), $W_1^{1,1}$ (elongation tensor), $W_1^{s,0}$ ($k$-atic shape modes for $k = 1, 2, 4, 6$)
- **Fourier boundary decomposition**: secondary shape descriptor for comparison with Minkowski tensors
- **Multiscale structure tensor**: orientation $\theta(\mathbf{x})$ and coherence $C(\mathbf{x})$ fields at logarithmically spaced scales (subcellular to tissue-level)
- **$k$-atic order parameter fields**: $\psi_k(\mathbf{x}, \sigma) = C \cdot e^{ik\theta}$ for arbitrary $k$
- **Nuclear ellipse fitting**: aspect ratio and orientation from DAPI masks via moments of inertia
- **Topological defect detection**: half-integer and integer charge defects via [cartan](https://crates.io/crates/cartan-geo) SO(3) holonomy. Poincar&eacute;--Hopf validation is implemented in `mermin-topo` and exposed to Python, but `analyze()` does not call it yet.
- **Persistent homology**: boundary matrix reduction on Delaunay filtration by ascending alignment magnitude, implemented in `mermin-topo` and exposed to Python; `analyze()` does not call it yet, so `AnalysisResult.persistence` is always empty
- **Orientational correlation functions**: $G_k(r) = \langle \cos k(\theta_i - \theta_j) \rangle$ with exponential fit for correlation length $\xi_k$
- **Ripley's $K$-function**: spatial clustering analysis for defect point patterns
- **Spatial block bootstrap**: confidence intervals that respect spatial autocorrelation
- **Permutation tests**: condition comparison with proper null distribution
- **Frank elastic energy**: splay $(\nabla \cdot \hat{\mathbf{n}})^2$ and bend $|\hat{\mathbf{n}} \times \nabla \times \hat{\mathbf{n}}|^2$ decomposition from the director field
- **Landau--de Gennes parameter fitting**: extract $(a, b, c, K)$ from experimental $\mathbf{Q}$-tensor fields
- **Activity estimation**: $\zeta_{\text{eff}}$ from defect density via mean-field active nematic theory
- **Volterra-compatible output**: fitted parameters exported as JSON for forward simulation with [volterra](https://crates.io/crates/volterra-nematic)

## Install

### Python (recommended for end users)

```bash
pip install mermin
```

Segmentation runs on a threshold and watershed backend by default, which needs
no extra dependency and is deterministic. Cellpose 4 is optional:

    uv add "mermin[cellpose]"

It brings torch, torchvision and a model checkpoint downloaded on first use.
With it installed, `mermin.analyze` selects it; without it, the threshold
backend runs and a warning names the fallback. Pass `segmentation="threshold"`
to choose it outright and silence the warning.

Requires Python 3.11+. The Rust extension is compiled automatically via [maturin](https://www.maturin.rs/).

### Rust (for library use)

```toml
[dependencies]
mermin = "0.5"
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

# Explicit role mapping and pixel size. `channels` maps roles, not names,
# to channel indices: "nuclear" and "fibre" are the only two roles.
result = mermin.analyze(
    "path/to/image.tif",
    channels={"nuclear": 0, "fibre": 1},
    pixel_size_um=0.69,
)

print(result.summary())
# mermin analysis: 847 cells, 12 defects, Frank ratio = 1.31

# Per-cell measurements as a polars DataFrame
result.cells.head()

# Metadata-driven: with neither `channels` nor `pixel_size_um` given, roles
# and pixel size are both read from the file's own metadata. An image with
# no calibration raises `mermin.ingest.PixelSizeError` naming the file.
result = mermin.analyze("path/to/image.tif")

# Batch analysis across conditions. `report()` writes each image's summary
# to a per-condition JSON file; it performs no statistical comparison.
experiment = mermin.Experiment(pixel_size_um=0.69)
experiment.add_condition("ctrl", ["d01.tif", "d02.tif", "d03.tif"])
experiment.add_condition("tgfb1", ["d07.tif", "d08.tif", "d09.tif"])
comparison = experiment.run()
comparison.report("output/")
```

## Breaking changes (0.5.0)

- `cellpose_diameter` is removed from `analyze()`. Backend parameters now
  live on the backend itself: `segmentation=mermin.backends.CellposeBackend(diameter=30.0)`.
- Cellpose is no longer a base dependency. It is the optional extra
  `mermin[cellpose]`, and must be version 4 or newer. Without it,
  segmentation runs on the threshold backend and a warning names the
  fallback.
- Python 3.10 is no longer supported. The minimum is 3.11, because
  `bioio-ome-zarr` requires it in every release mermin can build against.
- `analyze()` gains `segmentation` and `mask_cache` parameters, and
  `AnalysisResult` gains `segmentation`, recording which backend ran.

## Breaking changes (0.4.0)

- `channels` maps roles to channel indices, not channel names to indices.
  `channels={"dapi": 0, "vimentin": 1}` no longer resolves; pass
  `channels={"nuclear": 0, "fibre": 1}`, or omit `channels` entirely to
  resolve roles from file metadata.
- `pixel_size_um` no longer defaults, in `analyze`, `open_image` and
  `Experiment` alike. Pass it explicitly or rely on the file's own
  calibration; an image with neither raises `mermin.ingest.PixelSizeError`
  naming the file, rather than assuming a value.
- `io.py`, `load_tiff` and `discover_tiffs` are removed, with no
  compatibility shim. Use `mermin.ingest.open_image`, which resolves roles
  from metadata and reads TIFF, OME-TIFF and OME-Zarr via bioio.

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
| **mermin-orient** | Multiscale structure tensor, $k$-atic order parameter fields, nuclear ellipse fitting |
| **mermin-topo** | Defect detection (via [cartan-geo](https://crates.io/crates/cartan-geo) holonomy), Poincar&eacute;--Hopf validation, persistent homology |
| **mermin-stats** | Orientational correlation $G_k(r)$, Ripley's $K$, spatial block bootstrap, permutation tests |
| **mermin-theory** | Frank energy, Landau--de Gennes fitting, activity estimation, volterra-compatible JSON output |
| **mermin-py** | PyO3 bindings exposing all crates to Python |

## Analysis Pipeline

```
TIFF, OME-TIFF, OME-Zarr (nuclear + fibre channels)
  |
  +-- 1. Preprocessing ---- percentile contrast normalisation
  |
  +-- 2. Segmentation ----- backend protocol (nuclei), watershed (cell bodies)
  |
  +-- 3. Shape analysis ---- Minkowski tensors, Fourier modes, morphometrics
  |
  +-- 4. Orientation ------- multiscale structure tensor, k-atic fields, nuclear ellipse
  |
  +-- 5. Topology ---------- defect detection (holonomy)
  |
  +-- 6. Statistics -------- G_k(r), Ripley's K, block bootstrap, permutation tests
  |
  +-- 7. Theory ------------ Frank energy, Landau-de Gennes fit, activity estimation
  |
  +-- Output: `AnalysisResult` (per-cell polars DataFrame, field arrays, in memory).
      `Experiment.report()` writes a JSON summary of per-image results.
```

`analyze()` returns per-cell shape, orientation, defect, correlation and
theory measurements. Neighbour-graph construction (`build_neighbor_graph`),
Poincar&eacute;--Hopf validation and persistent homology are implemented but not
yet called from `analyze()`; `AnalysisResult.persistence` is always empty.
Of stage 6, `analyze()` calls only `orientational_correlation`: Ripley's $K$,
block bootstrap and permutation tests are implemented in `mermin-stats` and
exposed to Python, but `analyze()` does not call them yet. Of stage 7,
`analyze()` calls only `frank_energy` and `estimate_ldg_params`: activity
estimation is implemented in `mermin-theory` and exposed to Python, but
`analyze()` does not call it yet. Plotting and HTML report generation
(`mermin.viz`) are not yet implemented.

## Three Independent $k$-atic Measurements

mermin's Rust crates compute three independent orientational measurements per cell, each with distinct physical meaning. In 0.5.0, `analyze()` does not yet assemble any of the three into its per-cell `cells` table: `result.fields["theta"]` and `result.fields["coherence"]` hold the underlying orientation field, and `result.correlations` holds the population-level $G_k(r)$, from which a caller can derive them directly.

| Measurement | Source | What it captures |
|-------------|--------|-----------------|
| **Shape $k$-atic** | Minkowski tensors $W_1^{s,0}$ on cell boundary | How the cell is shaped (elongation, polygonality) |
| **Internal $k$-atic** | Structure tensor of vimentin within cell territory | How cytoskeletal fibres are organised inside |
| **Collective $k$-atic** | Neighbour correlations on Delaunay graph | How aligned the cell is with its neighbours |

Agreement or disagreement between these layers is itself diagnostic. A TGF-$\beta$-treated myofibroblast shows concordance across all three. A ROCK-inhibited cell may show a round shape (low shape $k{=}2$) but residual internal fibre alignment (higher internal $k{=}2$).

Shape $k$-atic modes are computed per cell in `mermin-shape`. Internal and collective $k$-atic values are computed by `mermin-orient` and `mermin-stats` respectively, and are exposed to Python.

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

Python dependencies: numpy, polars, scikit-image, scipy, tifffile, and [bioio](https://github.com/bioio-devs/bioio) (with the `bioio-ome-tiff`, `bioio-ome-zarr` and `bioio-tifffile` plugins) for file I/O. Cellpose 4 is an optional extra, `mermin[cellpose]`.

## License

MIT
