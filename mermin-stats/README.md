# mermin-stats

Spatial statistics, correlation functions, and hypothesis testing for orientational order. Part of the [mermin](https://crates.io/crates/mermin) library.

## Features

- **Orientational correlation function**: G_k(r) = <cos(k * (theta_i - theta_j))> binned by pairwise centroid distance, with exponential fit for correlation length xi_k
- **Ripley's K-function**: spatial clustering analysis for defect point patterns, with Besag's L(r) = sqrt(K(r)/pi) - r transformation and isotropic edge correction
- **Spatial block bootstrap**: confidence intervals that respect spatial autocorrelation (block size matched to correlation length, unlike ordinary bootstrap which assumes independence)
- **Permutation tests**: two-sample comparison of conditions with proper null distribution (shuffles condition labels, not individual cells)

## Why Spatial Block Bootstrap?

Cells in a tissue are spatially correlated. Ordinary bootstrap treats each cell as independent, underestimating uncertainty. The spatial block bootstrap divides the domain into blocks of size comparable to the correlation length and resamples blocks with replacement, preserving within-block correlations.

## Usage

```rust
use mermin::stats::{orientational_correlation, permutation_test};

let corr = orientational_correlation(&centroids, &thetas, 2, 500.0, 20);
println!("correlation length: {:.1} um", corr.correlation_length);

let result = permutation_test(&ctrl_values, &treated_values, |a, b| {
    let ma: f64 = a.iter().sum::<f64>() / a.len() as f64;
    let mb: f64 = b.iter().sum::<f64>() / b.len() as f64;
    ma - mb
}, 9999, 42);
println!("p = {:.4}", result.p_value);
```

## License

MIT
