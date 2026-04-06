// mermin-stats/src/correlation.rs

use mermin_core::{Point2, Real};

/// Result of computing the orientational correlation function G_k(r).
#[derive(Debug, Clone)]
pub struct CorrelationResult {
    /// Bin centers (distances in physical units).
    pub r_bins: Vec<Real>,
    /// G_k(r) values at each bin center.
    pub g_values: Vec<Real>,
    /// Number of pairs contributing to each bin.
    pub counts: Vec<usize>,
    /// Fitted correlation length xi_k from exponential decay G_k ~ exp(-r/xi).
    pub correlation_length: Real,
}

/// Compute the orientational correlation function
///   G_k(r) = <cos(k * (theta_i - theta_j))>
/// binned by pairwise centroid distance.
///
/// `centroids`: cell centroid positions.
/// `thetas`: per-cell director angle in [0, pi).
/// `k`: k-atic symmetry order.
/// `max_r`: maximum distance to compute.
/// `n_bins`: number of distance bins.
pub fn orientational_correlation(
    centroids: &[Point2],
    thetas: &[Real],
    k: u32,
    max_r: Real,
    n_bins: usize,
) -> CorrelationResult {
    let bin_width = max_r / n_bins as Real;
    let mut sums = vec![0.0; n_bins];
    let mut counts = vec![0usize; n_bins];
    let kf = k as Real;
    let n = centroids.len();

    for i in 0..n {
        for j in (i + 1)..n {
            let dist = centroids[i].distance_to(centroids[j]);
            if dist >= max_r || dist < 1e-15 {
                continue;
            }
            let bin = (dist / bin_width) as usize;
            if bin < n_bins {
                let dtheta = thetas[i] - thetas[j];
                sums[bin] += (kf * dtheta).cos();
                counts[bin] += 1;
            }
        }
    }

    let r_bins: Vec<Real> = (0..n_bins).map(|i| (i as Real + 0.5) * bin_width).collect();

    let g_values: Vec<Real> = sums
        .iter()
        .zip(counts.iter())
        .map(|(&s, &c)| if c > 0 { s / c as Real } else { 0.0 })
        .collect();

    // Fit correlation length: G_k(r) ~ exp(-r/xi)
    // Linear regression on log(G_k) vs r for bins with G_k > 0.01 and count > 5.
    let correlation_length = fit_exponential_decay(&r_bins, &g_values, &counts);

    CorrelationResult {
        r_bins,
        g_values,
        counts,
        correlation_length,
    }
}

/// Fit xi from G(r) ~ exp(-r/xi) using least squares on ln(G) vs r.
fn fit_exponential_decay(r: &[Real], g: &[Real], counts: &[usize]) -> Real {
    let mut sum_r = 0.0;
    let mut sum_lng = 0.0;
    let mut sum_r2 = 0.0;
    let mut sum_r_lng = 0.0;
    let mut n = 0.0;

    for i in 0..r.len() {
        if g[i] > 0.01 && counts[i] >= 5 {
            let lng = g[i].ln();
            sum_r += r[i];
            sum_lng += lng;
            sum_r2 += r[i] * r[i];
            sum_r_lng += r[i] * lng;
            n += 1.0;
        }
    }

    if n < 2.0 {
        return Real::INFINITY; // Cannot fit
    }

    // slope = (n * sum(r*lng) - sum(r)*sum(lng)) / (n * sum(r^2) - sum(r)^2)
    let denom = n * sum_r2 - sum_r * sum_r;
    if denom.abs() < 1e-15 {
        return Real::INFINITY;
    }
    let slope = (n * sum_r_lng - sum_r * sum_lng) / denom;

    // G ~ exp(-r/xi) => ln(G) ~ -r/xi => slope = -1/xi
    if slope >= 0.0 {
        Real::INFINITY // Not decaying
    } else {
        -1.0 / slope
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfectly_aligned_cells() {
        // All cells at theta = 0.5, random positions
        let n = 50;
        let centroids: Vec<Point2> = (0..n)
            .map(|i| Point2::new((i % 10) as Real * 10.0, (i / 10) as Real * 10.0))
            .collect();
        let thetas = vec![0.5; n];

        let result = orientational_correlation(&centroids, &thetas, 2, 50.0, 10);
        // All G_k(r) should be 1.0 (perfect alignment)
        for (i, &g) in result.g_values.iter().enumerate() {
            if result.counts[i] > 0 {
                assert!(
                    (g - 1.0).abs() < 1e-10,
                    "aligned cells: G_k(r={:.1}) = {g}, expected 1.0",
                    result.r_bins[i]
                );
            }
        }
    }
}
