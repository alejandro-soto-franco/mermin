# mermin-core

Core types, traits, and error handling for the [mermin](https://crates.io/crates/mermin) k-atic alignment analysis library.

## Types

| Type | Purpose |
|------|---------|
| `CellRecord` | Per-cell measurement struct (shape, orientation, topology fields) |
| `ImageField` | Row-major 2D scalar field on a regular grid |
| `BoundaryContour` | Ordered boundary polygon with area, perimeter, centroid |
| `KValue` | Validated k-atic symmetry order (k=1 polar, k=2 nematic, k=4 tetratic, k=6 hexatic) |
| `Point2` | 2D coordinate with distance computation |
| `MerminError` | Unified error enum for all fallible operations |

## Usage

This crate is re-exported by the `mermin` facade crate. You typically do not depend on it directly.

```rust
use mermin::{BoundaryContour, CellRecord, ImageField, Point2, K_NEMATIC};
```

## License

MIT
