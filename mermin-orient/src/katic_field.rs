// mermin-orient/src/katic_field.rs

use crate::structure_tensor::StructureTensorResult;
use mermin_core::{ImageField, Real};

/// k-atic order parameter field psi_k(x) = C(x) * exp(i*k*theta(x)).
///
/// Returns (|psi_k|, Re(psi_k), Im(psi_k)) fields.
/// |psi_k| is the local k-atic alignment magnitude.
pub struct KAticField {
    /// |psi_k| magnitude at each pixel, in [0, 1].
    pub magnitude: ImageField,
    /// Real part of psi_k = C * cos(k * theta).
    pub real_part: ImageField,
    /// Imaginary part of psi_k = C * sin(k * theta).
    pub imag_part: ImageField,
}

/// Compute the k-atic order parameter field from a structure tensor result.
pub fn katic_order_field(st: &StructureTensorResult, k: u32) -> KAticField {
    let (w, h) = (st.theta.width, st.theta.height);
    let kf = k as Real;
    let n = w * h;

    let mut magnitude = ImageField::zeros(w, h);
    let mut real_part = ImageField::zeros(w, h);
    let mut imag_part = ImageField::zeros(w, h);

    for i in 0..n {
        let c = st.coherence.data[i];
        let theta = st.theta.data[i];
        let re = c * (kf * theta).cos();
        let im = c * (kf * theta).sin();
        real_part.data[i] = re;
        imag_part.data[i] = im;
        magnitude.data[i] = c; // |exp(ik*theta)| = 1, so |psi_k| = C
    }

    KAticField {
        magnitude,
        real_part,
        imag_part,
    }
}

/// Compute mean k-atic order parameter over a masked region.
///
/// Returns |<psi_k>| where the average is over all pixels where mask > 0.
/// This is the alignment magnitude for the region (1 = perfect alignment, 0 = isotropic).
pub fn mean_katic_order(field: &KAticField, mask: &[bool]) -> Real {
    let mut re_sum = 0.0;
    let mut im_sum = 0.0;
    let mut count = 0;

    for (i, &in_mask) in mask.iter().enumerate() {
        if in_mask {
            re_sum += field.real_part.data[i];
            im_sum += field.imag_part.data[i];
            count += 1;
        }
    }

    if count == 0 {
        return 0.0;
    }

    let n = count as Real;
    ((re_sum / n).powi(2) + (im_sum / n).powi(2)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_orientation_perfect_order() {
        // All pixels oriented the same way with coherence 1
        let n = 100;
        let theta = ImageField::new(vec![0.5; n], 10, 10);
        let coherence = ImageField::new(vec![1.0; n], 10, 10);
        let lambda1 = ImageField::new(vec![1.0; n], 10, 10);
        let lambda2 = ImageField::zeros(10, 10);

        let st = StructureTensorResult {
            theta,
            coherence,
            lambda1,
            lambda2,
        };

        let field = katic_order_field(&st, 2);
        let mask = vec![true; n];
        let order = mean_katic_order(&field, &mask);
        assert!(
            (order - 1.0).abs() < 1e-10,
            "uniform orientation should give order = 1, got {order}"
        );
    }

    #[test]
    fn random_orientation_low_order() {
        // Orientations evenly spaced across [0, pi): should average to ~0
        let n = 1000;
        let w = 50;
        let h = 20;
        let theta_data: Vec<Real> = (0..n)
            .map(|i| std::f64::consts::PI * i as Real / n as Real)
            .collect();
        let theta = ImageField::new(theta_data, w, h);
        let coherence = ImageField::new(vec![1.0; n], w, h);
        let lambda1 = ImageField::new(vec![1.0; n], w, h);
        let lambda2 = ImageField::zeros(w, h);

        let st = StructureTensorResult {
            theta,
            coherence,
            lambda1,
            lambda2,
        };

        let field = katic_order_field(&st, 2);
        let mask = vec![true; n];
        let order = mean_katic_order(&field, &mask);
        assert!(
            order < 0.05,
            "uniformly distributed orientations should give near-zero order, got {order}"
        );
    }
}
