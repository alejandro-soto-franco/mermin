// mermin-stats/src/bootstrap.rs

use mermin_core::Real;
use rand::prelude::*;

/// Spatial block bootstrap for computing confidence intervals on a statistic.
///
/// Divides the spatial domain into blocks of size `block_size` and resamples
/// blocks with replacement. This preserves spatial autocorrelation within blocks.
///
/// `values`: the per-cell measurements.
/// `positions`: (x, y) positions of each cell.
/// `statistic`: function that computes the statistic from a sample.
/// `block_size`: side length of spatial blocks (should be ~ correlation length).
/// `n_bootstrap`: number of bootstrap resamples.
/// `seed`: random seed for reproducibility.
///
/// Returns sorted bootstrap distribution of the statistic.
pub fn spatial_block_bootstrap<F>(
    values: &[Real],
    positions: &[(Real, Real)],
    statistic: F,
    block_size: Real,
    n_bootstrap: usize,
    seed: u64,
) -> Vec<Real>
where
    F: Fn(&[Real]) -> Real,
{
    let n = values.len();
    assert_eq!(n, positions.len());

    // Assign each cell to a block
    let x_min = positions
        .iter()
        .map(|p| p.0)
        .fold(Real::INFINITY, Real::min);
    let y_min = positions
        .iter()
        .map(|p| p.1)
        .fold(Real::INFINITY, Real::min);

    let block_of = |i: usize| -> (i64, i64) {
        let bx = ((positions[i].0 - x_min) / block_size).floor() as i64;
        let by = ((positions[i].1 - y_min) / block_size).floor() as i64;
        (bx, by)
    };

    // Group cell indices by block
    let mut blocks: std::collections::HashMap<(i64, i64), Vec<usize>> =
        std::collections::HashMap::new();
    for i in 0..n {
        blocks.entry(block_of(i)).or_default().push(i);
    }
    let block_list: Vec<Vec<usize>> = blocks.into_values().collect();
    let n_blocks = block_list.len();

    if n_blocks == 0 {
        return vec![];
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut distribution = Vec::with_capacity(n_bootstrap);

    for _ in 0..n_bootstrap {
        // Resample blocks with replacement
        let mut sample = Vec::with_capacity(n);
        for _ in 0..n_blocks {
            let block_idx = rng.gen_range(0..n_blocks);
            for &cell_idx in &block_list[block_idx] {
                sample.push(values[cell_idx]);
            }
        }
        distribution.push(statistic(&sample));
    }

    distribution.sort_by(|a, b| a.partial_cmp(b).unwrap());
    distribution
}

/// Compute confidence interval from a sorted bootstrap distribution.
/// Returns (lower, upper) for the given confidence level (e.g., 0.95).
pub fn confidence_interval(sorted_distribution: &[Real], confidence: Real) -> (Real, Real) {
    let n = sorted_distribution.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let alpha = (1.0 - confidence) / 2.0;
    let lo_idx = (alpha * n as Real).floor() as usize;
    let hi_idx = ((1.0 - alpha) * n as Real).ceil() as usize;
    (
        sorted_distribution[lo_idx.min(n - 1)],
        sorted_distribution[hi_idx.min(n - 1)],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bootstrap_mean_ci() {
        let values: Vec<Real> = (0..100).map(|i| i as Real).collect();
        let positions: Vec<(Real, Real)> = (0..100)
            .map(|i| ((i % 10) as Real * 10.0, (i / 10) as Real * 10.0))
            .collect();

        let dist = spatial_block_bootstrap(
            &values,
            &positions,
            |s| s.iter().sum::<Real>() / s.len() as Real,
            20.0,
            500,
            42,
        );

        let (lo, hi) = confidence_interval(&dist, 0.95);
        let true_mean = 49.5;
        assert!(
            lo < true_mean && hi > true_mean,
            "95% CI [{lo}, {hi}] should contain true mean {true_mean}"
        );
    }
}
