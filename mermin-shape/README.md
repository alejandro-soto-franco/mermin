# mermin-shape

Minkowski tensors, Fourier boundary decomposition, and morphometric shape descriptors for cell boundaries. Part of the [mermin](https://crates.io/crates/mermin) library.

## Features

- **Minkowski scalar functionals**: W_0 (area), W_1 (perimeter)
- **Minkowski tensor W_1^{1,1}**: rank-2 orientation tensor giving cell elongation magnitude and direction
- **k-atic shape amplitudes**: W_1^{s,0} for k = 1, 2, 4, 6 via boundary normal integration
- **Fourier boundary decomposition**: a_k = (1/N) sum r_i exp(-ik theta_i) in polar coordinates relative to centroid
- **Morphometrics**: shape index p_0 = P/sqrt(A), convexity = A/A_hull (Graham scan convex hull)

## Why Minkowski tensors over Fourier?

Fourier decomposition of r(theta) assumes a well-defined centroid and breaks down for non-convex or highly irregular cell shapes. Minkowski tensors handle arbitrary boundary geometry correctly because they integrate over edge normals, not radial coordinates. Both are provided for comparison.

## Usage

```rust
use mermin::shape::{minkowski_w1_tensor, elongation_from_w1_tensor, katic_shape_spectrum};
use mermin::BoundaryContour;

let contour = BoundaryContour::new(points)?;
let tensor = minkowski_w1_tensor(&contour);
let (elongation, angle) = elongation_from_w1_tensor(&tensor);
let [q1, q2, q4, q6] = katic_shape_spectrum(&contour);
```

## License

MIT
