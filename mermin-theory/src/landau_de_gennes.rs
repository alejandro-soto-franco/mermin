// mermin-theory/src/landau_de_gennes.rs

use mermin_core::Real;
use serde::{Deserialize, Serialize};

/// Landau-de Gennes free energy parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LdGParams {
    /// Landau coefficient a (< 0 for ordered phase).
    pub a: Real,
    /// Landau coefficient b (cubic term, 3D only; 0 for 2D).
    pub b: Real,
    /// Landau coefficient c (quartic stabilization, > 0).
    pub c: Real,
    /// Frank elastic constant K (one-constant approximation).
    pub k_elastic: Real,
}

/// Compute the bulk Landau-de Gennes free energy density for a 2D Q-tensor.
///
/// For 2D (traceless symmetric 2x2):
///   f_bulk = (a/2) * |Q|^2 + (c/4) * |Q|^4
///
/// where |Q|^2 = tr(Q^2) = 2 * S^2 for uniaxial Q = S * (n tensor n - I/2).
pub fn bulk_energy_density_2d(s: Real, params: &LdGParams) -> Real {
    let q_sq = 2.0 * s * s;
    (params.a / 2.0) * q_sq + (params.c / 4.0) * q_sq * q_sq
}

/// Estimate Landau-de Gennes parameters from the observed scalar order parameter distribution.
///
/// Uses moment matching:
///   - Equilibrium S_eq = sqrt(-a / (2c)) in 2D
///   - K is estimated from the orientational correlation length: K ~ xi^2 * |a|
///
/// `s_values`: per-cell scalar order parameter |psi_2|.
/// `correlation_length`: xi from G_2(r) fit.
/// `pixel_size`: um per pixel, for converting xi to physical units.
pub fn estimate_ldg_params(
    s_values: &[Real],
    correlation_length: Real,
    pixel_size: Real,
) -> LdGParams {
    let n = s_values.len() as Real;
    if n < 1.0 {
        return LdGParams {
            a: 0.0,
            b: 0.0,
            c: 0.0,
            k_elastic: 0.0,
        };
    }

    let _mean_s: Real = s_values.iter().sum::<Real>() / n;
    let mean_s2: Real = s_values.iter().map(|&s| s * s).sum::<Real>() / n;

    // For equilibrium 2D: S_eq^2 = -a/(2c) and we set c = 1 (normalization freedom)
    // Then a = -2 * S_eq^2
    let c = 1.0;
    let a = -2.0 * mean_s2;

    // K from correlation length: K ~ xi^2 * |a| (mean-field scaling)
    let xi_physical = correlation_length * pixel_size;
    let k_elastic = xi_physical * xi_physical * a.abs();

    LdGParams {
        a,
        b: 0.0, // 2D, no cubic term
        c,
        k_elastic,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equilibrium_s_consistent() {
        // If we input S values that are at equilibrium for known a, c,
        // the estimated a should approximately recover the input.
        let a_true = -0.5;
        let c_true = 1.0;
        let s_eq = (-a_true / (2.0 * c_true) as Real).sqrt();
        let s_values = vec![s_eq; 100];

        let params = estimate_ldg_params(&s_values, 10.0, 0.345);
        assert!(
            (params.a - a_true).abs() < 0.01,
            "estimated a = {}, expected {}",
            params.a,
            a_true
        );
    }
}
