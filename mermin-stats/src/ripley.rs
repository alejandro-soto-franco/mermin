// mermin-stats/src/ripley.rs

use mermin_core::{Point2, Real};

/// Result of Ripley's K-function analysis.
#[derive(Debug, Clone)]
pub struct RipleyResult {
    /// Evaluation distances.
    pub r_values: Vec<Real>,
    /// K(r) values.
    pub k_values: Vec<Real>,
    /// Besag's L(r) = sqrt(K(r)/pi) - r. Zero under CSR (complete spatial randomness).
    pub l_values: Vec<Real>,
}

/// Compute Ripley's K-function for a 2D point pattern.
///
/// K(r) = (A/n^2) * sum_{i != j} 1(d(i,j) <= r) * w(i,j)
///
/// where A is the study area and w(i,j) is the edge correction weight.
/// Uses the isotropic (Ripley) edge correction.
///
/// `points`: point positions.
/// `bbox`: bounding box [x_min, y_min, x_max, y_max].
/// `r_values`: distances at which to evaluate K.
pub fn ripley_k(
    points: &[Point2],
    bbox: [Real; 4],
    r_values: &[Real],
) -> RipleyResult {
    let n = points.len();
    let area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]);
    let nf = n as Real;

    let mut k_values = vec![0.0; r_values.len()];

    for i in 0..n {
        for j in (i + 1)..n {
            let dist = points[i].distance_to(points[j]);
            // Simple edge correction: 1 / fraction of circle within bbox
            let weight = edge_correction(points[i], dist, &bbox);

            for (ri, &r) in r_values.iter().enumerate() {
                if dist <= r {
                    k_values[ri] += 2.0 * weight; // count both (i,j) and (j,i)
                }
            }
        }
    }

    // Normalize: K(r) = A / n^2 * count
    for kv in &mut k_values {
        *kv *= area / (nf * nf);
    }

    let l_values: Vec<Real> = k_values
        .iter()
        .zip(r_values.iter())
        .map(|(&k, &r)| (k / std::f64::consts::PI).sqrt() - r)
        .collect();

    RipleyResult {
        r_values: r_values.to_vec(),
        k_values,
        l_values,
    }
}

/// Isotropic edge correction: 1 / (fraction of circle of radius r centered at p
/// that falls within the bounding box). Approximated by the proportion of
/// the circle in each quadrant.
fn edge_correction(p: Point2, r: Real, bbox: &[Real; 4]) -> Real {
    if r < 1e-15 {
        return 1.0;
    }
    let dx_min = p.x - bbox[0];
    let dx_max = bbox[2] - p.x;
    let dy_min = p.y - bbox[1];
    let dy_max = bbox[3] - p.y;

    // Fraction of the circle within the rectangle
    // Simple approximation: if all distances to edges > r, fraction = 1
    let min_dist = dx_min.min(dx_max).min(dy_min).min(dy_max);
    if min_dist >= r {
        return 1.0;
    }

    // Rough correction: proportion based on how much of the circle is clipped
    // This is a simplification; exact correction involves arc length integrals.
    // For interior points (min_dist > r), weight = 1.
    // For edge points, weight > 1 to compensate.
    let frac = (min_dist / r).max(0.25); // clamp to avoid extreme weights
    1.0 / frac
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn k_grows_with_r() {
        // Grid of points: K(r) should increase monotonically
        let mut points = Vec::new();
        for i in 0..10 {
            for j in 0..10 {
                points.push(Point2::new(i as Real * 10.0, j as Real * 10.0));
            }
        }
        let bbox = [0.0, 0.0, 90.0, 90.0];
        let r_values: Vec<Real> = (1..=5).map(|i| i as Real * 15.0).collect();
        let result = ripley_k(&points, bbox, &r_values);

        for i in 1..result.k_values.len() {
            assert!(
                result.k_values[i] >= result.k_values[i - 1],
                "K(r) should be non-decreasing"
            );
        }
    }
}
