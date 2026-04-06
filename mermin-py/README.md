# mermin-py

[PyO3](https://pyo3.rs/) bindings exposing all mermin analysis crates to Python. Built with [maturin](https://www.maturin.rs/).

## Exposed Functions

| Python function | Rust source | Purpose |
|----------------|-------------|---------|
| `analyze_shape(contour)` | mermin-shape | Minkowski tensors, Fourier spectrum, morphometrics for one cell |
| `analyze_shapes_batch(contours)` | mermin-shape | Batch shape analysis for all cells |
| `compute_structure_tensor(image, sigma)` | mermin-orient | Single-scale structure tensor (theta + coherence fields) |
| `compute_multiscale_structure_tensor(image, sigmas)` | mermin-orient | Multi-scale with optimal scale map |
| `fit_nuclear_ellipses(mask)` | mermin-orient | Nuclear aspect ratio and orientation from labeled DAPI mask |
| `detect_defects(thetas, nx, ny, k, threshold)` | mermin-topo | Topological defect detection via holonomy |
| `validate_poincare_hopf(defects, chi, tol)` | mermin-topo | Charge sum validation |
| `compute_persistence(vertices, edges, triangles)` | mermin-topo | Persistent homology of simplicial filtration |
| `orientational_correlation(centroids, thetas, k, max_r, n_bins)` | mermin-stats | G_k(r) with correlation length fit |
| `ripley_k(points, bbox, r_values)` | mermin-stats | Ripley's K-function for point patterns |
| `permutation_test(values_a, values_b, n_perms, seed)` | mermin-stats | Two-sample permutation test (difference of means) |
| `frank_energy(thetas, nx, ny, dx)` | mermin-theory | Splay + bend decomposition |
| `estimate_ldg_params(s_values, xi, pixel_size)` | mermin-theory | Landau-de Gennes parameter estimation |
| `build_volterra_params(a, b, c, k_elastic, ...)` | mermin-theory | Volterra-compatible parameter set |

## Building

```bash
pip install maturin
cd mermin-py
maturin develop --release
```

The compiled extension is imported as `mermin._native` by the pure-Python `mermin` package.

## License

MIT
