# mermin

Facade crate re-exporting all [mermin](https://github.com/alejandro-soto-franco/mermin) subcrates for k-atic alignment analysis of fluorescence microscopy.

## Sub-crates

| Crate | Purpose |
|-------|---------|
| [`mermin-core`](https://crates.io/crates/mermin-core) | Core types, traits, and error handling |
| [`mermin-shape`](https://crates.io/crates/mermin-shape) | Minkowski tensors and cell shape descriptors |
| [`mermin-orient`](https://crates.io/crates/mermin-orient) | Multiscale structure tensor and order parameter fields |
| [`mermin-topo`](https://crates.io/crates/mermin-topo) | Topological defect detection and persistent homology |
| [`mermin-stats`](https://crates.io/crates/mermin-stats) | Spatial statistics and correlation functions |
| [`mermin-theory`](https://crates.io/crates/mermin-theory) | Frank energy, Landau-de Gennes fitting, activity estimation |
| [`mermin-py`](https://crates.io/crates/mermin-py) | PyO3 Python bindings |

## Usage

```rust
use mermin::{BoundaryContour, CellRecord, ImageField, K_NEMATIC};
use mermin::shape::analyze_shape;
use mermin::orient::structure_tensor;
use mermin::topo::detect_defects;
```

## License

MIT
