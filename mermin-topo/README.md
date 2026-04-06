# mermin-topo

Topological defect detection, Poincare-Hopf validation, and persistent homology. Part of the [mermin](https://crates.io/crates/mermin) library.

## Features

- **Director field embedding**: 2D director angles embedded as 3D SO(3) frames for compatibility with [cartan-geo](https://crates.io/crates/cartan-geo) holonomy machinery
- **Defect detection**: wraps cartan's `scan_disclinations()` to detect topological defects on coarse-grained cell-level orientation fields
  - Nematic (k=2): +/- 1/2 charge defects
  - Polar (k=1): +/- 1 charge defects
  - Hexatic (k=6): +/- 1/6 charge defects
- **Poincare-Hopf validation**: verifies that the sum of defect charges equals the Euler characteristic of the domain (sanity check for detection completeness)
- **Persistent homology**: boundary matrix reduction on Delaunay complex filtrations by ascending alignment magnitude
  - H_0 features: robust alignment domains
  - H_1 features: orientational vortices

## Defect Detection Method

Defects are detected via holonomy, not zero-crossing. The SO(3) transition matrix is computed along each plaquette boundary on the Delaunay mesh. If the accumulated rotation angle exceeds a threshold (default: pi/2), the plaquette contains a defect. This method is insensitive to core regularisation and works directly on discrete orientation data.

## Usage

```rust
use mermin::topo::{detect_defects, validate_poincare_hopf, compute_persistence};

let defects = detect_defects(&cell_thetas, nx, ny, 2, std::f64::consts::FRAC_PI_2);
let (charge_sum, chi, valid) = validate_poincare_hopf(&defects, 1, 0.1);
```

## License

MIT
