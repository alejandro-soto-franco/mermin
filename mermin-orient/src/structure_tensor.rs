// mermin-orient/src/structure_tensor.rs

use crate::gaussian::gaussian_blur_triple;
use crate::scharr_gradient;
use mermin_core::{ImageField, Real};
use rayon::prelude::*;

/// Result of structure tensor analysis at a single scale.
pub struct StructureTensorResult {
    /// Local orientation angle theta(x) in [0, pi), in radians.
    pub theta: ImageField,
    /// Coherence C(x) = (lambda1 - lambda2) / (lambda1 + lambda2) in [0, 1].
    /// 1 = perfectly oriented, 0 = isotropic.
    pub coherence: ImageField,
    /// Larger eigenvalue lambda1(x).
    pub lambda1: ImageField,
    /// Smaller eigenvalue lambda2(x).
    pub lambda2: ImageField,
}

/// Compute the structure tensor at a given smoothing scale sigma.
///
/// J_sigma(x) = G_sigma * (grad I tensor grad I)
///
/// The gradient is computed via Scharr, then the outer product components
/// (Ix*Ix, Ix*Iy, Iy*Iy) are Gaussian-smoothed at scale sigma.
/// All passes are parallelized with rayon.
pub fn structure_tensor(image: &ImageField, sigma: Real) -> StructureTensorResult {
    let (gx, gy) = scharr_gradient(image);
    let (w, h) = (image.width, image.height);
    let n = w * h;

    // Compute outer product components
    let mut jxx_data = vec![0.0_f64; n];
    let mut jxy_data = vec![0.0_f64; n];
    let mut jyy_data = vec![0.0_f64; n];

    // Parallelize the outer product computation
    jxx_data
        .par_iter_mut()
        .zip(jxy_data.par_iter_mut())
        .zip(jyy_data.par_iter_mut())
        .enumerate()
        .for_each(|(i, ((xx, xy), yy))| {
            let ix = gx.data[i];
            let iy = gy.data[i];
            *xx = ix * ix;
            *xy = ix * iy;
            *yy = iy * iy;
        });

    let jxx = ImageField::new(jxx_data, w, h);
    let jxy = ImageField::new(jxy_data, w, h);
    let jyy = ImageField::new(jyy_data, w, h);

    // Smooth all three tensor components in a single fused pass
    let (jxx, jxy, jyy) = gaussian_blur_triple(&jxx, &jxy, &jyy, sigma);

    // Eigendecompose at each pixel (parallel)
    let mut theta_data = vec![0.0_f64; n];
    let mut coherence_data = vec![0.0_f64; n];
    let mut lambda1_data = vec![0.0_f64; n];
    let mut lambda2_data = vec![0.0_f64; n];

    theta_data
        .par_iter_mut()
        .zip(coherence_data.par_iter_mut())
        .zip(lambda1_data.par_iter_mut())
        .zip(lambda2_data.par_iter_mut())
        .enumerate()
        .for_each(|(i, (((th, coh), l1), l2))| {
            let a = jxx.data[i];
            let b = jxy.data[i];
            let d = jyy.data[i];

            let trace = a + d;
            let det_term = ((a - d) * (a - d) + 4.0 * b * b).sqrt();

            let lam1 = (trace + det_term) * 0.5;
            let lam2 = (trace - det_term) * 0.5;

            *l1 = lam1;
            *l2 = lam2;

            let sum = lam1 + lam2;
            *coh = if sum > 1e-15 {
                (lam1 - lam2) / sum
            } else {
                0.0
            };

            let mut angle = 0.5 * (2.0 * b).atan2(a - d) + std::f64::consts::FRAC_PI_2;
            if angle < 0.0 {
                angle += std::f64::consts::PI;
            }
            if angle >= std::f64::consts::PI {
                angle -= std::f64::consts::PI;
            }
            *th = angle;
        });

    StructureTensorResult {
        theta: ImageField::new(theta_data, w, h),
        coherence: ImageField::new(coherence_data, w, h),
        lambda1: ImageField::new(lambda1_data, w, h),
        lambda2: ImageField::new(lambda2_data, w, h),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mermin_core::ImageField;

    #[test]
    fn vertical_stripes_orientation() {
        let w = 100;
        let h = 100;
        let data: Vec<Real> = (0..h)
            .flat_map(|_| (0..w).map(|c| (2.0 * std::f64::consts::PI * c as Real / 10.0).sin()))
            .collect();
        let field = ImageField::new(data, w, h);
        let result = structure_tensor(&field, 3.0);

        let mut sum_theta = 0.0;
        let mut count = 0;
        for row in 20..80 {
            for col in 20..80 {
                if result.coherence.get(row, col) > 0.5 {
                    sum_theta += result.theta.get(row, col);
                    count += 1;
                }
            }
        }
        let mean_theta = sum_theta / count as Real;
        assert!(
            (mean_theta - std::f64::consts::FRAC_PI_2).abs() < 0.3,
            "vertical stripes should give theta ~ pi/2, got {mean_theta:.3}"
        );
    }
}
