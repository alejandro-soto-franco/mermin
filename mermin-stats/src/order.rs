// mermin-stats/src/order.rs

use mermin_core::Real;

/// Result of computing the global nematic order parameter.
#[derive(Debug, Clone, Copy)]
pub struct NematicOrder {
    /// Scalar order parameter S in [0, 1].
    /// S = 1 for perfect alignment, S = 0 for isotropic.
    pub s: Real,
    /// Mean director angle in radians [0, pi).
    pub mean_angle: Real,
}

/// Compute the global nematic order parameter S = <cos 2(theta - theta_mean)>.
///
/// `cell_thetas`: per-cell nematic orientations in radians [0, pi).
///
/// Returns None if fewer than 2 cells are provided.
pub fn nematic_order_parameter(cell_thetas: &[Real]) -> Option<NematicOrder> {
    let n = cell_thetas.len();
    if n < 2 {
        return None;
    }

    let nf = n as Real;
    let mut sin_sum = 0.0;
    let mut cos_sum = 0.0;
    for &t in cell_thetas {
        sin_sum += (2.0 * t).sin();
        cos_sum += (2.0 * t).cos();
    }
    let mean_angle = 0.5 * (sin_sum / nf).atan2(cos_sum / nf);

    let mut s_sum = 0.0;
    for &t in cell_thetas {
        s_sum += (2.0 * (t - mean_angle)).cos();
    }
    let s = s_sum / nf;

    Some(NematicOrder {
        s,
        mean_angle: mean_angle.rem_euclid(std::f64::consts::PI),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn perfectly_aligned() {
        let thetas = vec![0.5; 100];
        let order = nematic_order_parameter(&thetas).unwrap();
        assert!((order.s - 1.0).abs() < 1e-10, "S should be 1.0 for aligned cells, got {}", order.s);
    }

    #[test]
    fn isotropic() {
        let n = 1000;
        let thetas: Vec<f64> = (0..n).map(|i| PI * i as f64 / n as f64).collect();
        let order = nematic_order_parameter(&thetas).unwrap();
        assert!(order.s.abs() < 0.05, "S should be near 0 for uniform distribution, got {}", order.s);
    }

    #[test]
    fn too_few_cells() {
        assert!(nematic_order_parameter(&[0.5]).is_none());
    }
}
