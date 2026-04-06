// mermin-theory/src/activity.rs

use mermin_core::Real;

/// Estimate the effective activity parameter zeta_eff from defect density.
///
/// From mean-field active nematic theory:
///   defect_spacing ~ sqrt(K / zeta_eff)
///   rho_defect ~ zeta_eff / K
///   => zeta_eff ~ K * rho_defect
///
/// `n_defects`: number of defects detected.
/// `area`: total image area in physical units (um^2).
/// `k_elastic`: Frank elastic constant from LdG fitting.
pub fn estimate_activity(n_defects: usize, area: Real, k_elastic: Real) -> Real {
    if area < 1e-15 {
        return 0.0;
    }
    let rho = n_defects as Real / area;
    k_elastic * rho
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_defects_zero_activity() {
        let zeta = estimate_activity(0, 1000.0, 0.1);
        assert!((zeta - 0.0).abs() < 1e-15);
    }

    #[test]
    fn activity_scales_with_defects() {
        let z1 = estimate_activity(5, 1000.0, 0.1);
        let z2 = estimate_activity(10, 1000.0, 0.1);
        assert!(z2 > z1, "more defects should give higher activity");
        assert!((z2 / z1 - 2.0).abs() < 1e-10, "should scale linearly");
    }
}
