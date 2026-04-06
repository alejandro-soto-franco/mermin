// mermin-topo/src/poincare_hopf.rs

use crate::defects::Defect;
use mermin_core::Real;

/// Validate the Poincare-Hopf theorem: sum of defect charges must equal
/// the Euler characteristic of the domain.
///
/// For a disk (simply connected planar domain), chi = 1.
/// For a torus, chi = 0. For a sphere, chi = 2.
///
/// Returns (charge_sum, expected_chi, is_valid).
pub fn validate_poincare_hopf(
    defects: &[Defect],
    euler_characteristic: i32,
    tolerance: Real,
) -> (Real, i32, bool) {
    let charge_sum: Real = defects.iter().map(|d| d.charge).sum();
    let expected = euler_characteristic as Real;
    let valid = (charge_sum - expected).abs() < tolerance;
    (charge_sum, euler_characteristic, valid)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn balanced_charges() {
        let defects = vec![
            Defect { position: [1.0, 1.0], charge: 0.5, angle: 3.0 },
            Defect { position: [5.0, 5.0], charge: 0.5, angle: 3.0 },
        ];
        let (sum, chi, valid) = validate_poincare_hopf(&defects, 1, 0.1);
        assert_eq!(chi, 1);
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(valid);
    }
}
