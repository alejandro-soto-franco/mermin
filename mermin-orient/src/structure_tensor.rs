// mermin-orient/src/structure_tensor.rs

use crate::{gaussian_blur, scharr_gradient};
use mermin_core::{ImageField, Real};

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
pub fn structure_tensor(image: &ImageField, sigma: Real) -> StructureTensorResult {
    let (gx, gy) = scharr_gradient(image);
    let (w, h) = (image.width, image.height);

    // Compute outer product components
    let mut jxx = ImageField::zeros(w, h);
    let mut jxy = ImageField::zeros(w, h);
    let mut jyy = ImageField::zeros(w, h);

    for i in 0..w * h {
        let ix = gx.data[i];
        let iy = gy.data[i];
        jxx.data[i] = ix * ix;
        jxy.data[i] = ix * iy;
        jyy.data[i] = iy * iy;
    }

    // Smooth the tensor components
    let jxx = gaussian_blur(&jxx, sigma);
    let jxy = gaussian_blur(&jxy, sigma);
    let jyy = gaussian_blur(&jyy, sigma);

    // Eigendecompose at each pixel
    let mut theta = ImageField::zeros(w, h);
    let mut coherence = ImageField::zeros(w, h);
    let mut lambda1 = ImageField::zeros(w, h);
    let mut lambda2 = ImageField::zeros(w, h);

    for i in 0..w * h {
        let a = jxx.data[i];
        let b = jxy.data[i];
        let d = jyy.data[i];

        // 2x2 symmetric eigendecomposition:
        // lambda = ((a+d) +/- sqrt((a-d)^2 + 4b^2)) / 2
        let trace = a + d;
        let det_term = ((a - d) * (a - d) + 4.0 * b * b).sqrt();

        let l1 = (trace + det_term) * 0.5;
        let l2 = (trace - det_term) * 0.5;

        lambda1.data[i] = l1;
        lambda2.data[i] = l2;

        // Coherence
        let sum = l1 + l2;
        coherence.data[i] = if sum > 1e-15 {
            (l1 - l2) / sum
        } else {
            0.0
        };

        // Orientation: angle of the minor eigenvector (perpendicular to gradient direction).
        // For structure tensor, the orientation of the *structure* (fiber direction)
        // is perpendicular to the dominant gradient direction.
        // theta = 0.5 * atan2(2b, a - d) gives the gradient direction;
        // we add pi/2 to get fiber direction.
        let mut angle = 0.5 * (2.0 * b).atan2(a - d) + std::f64::consts::FRAC_PI_2;
        // Normalize to [0, pi)
        if angle < 0.0 {
            angle += std::f64::consts::PI;
        }
        if angle >= std::f64::consts::PI {
            angle -= std::f64::consts::PI;
        }
        theta.data[i] = angle;
    }

    StructureTensorResult {
        theta,
        coherence,
        lambda1,
        lambda2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertical_stripes_orientation() {
        // Vertical stripes: I(r,c) = sin(2*pi*c/10)
        // Structure should be oriented horizontally (theta ~ 0)
        // because gradient is horizontal and fiber is vertical? Actually:
        // Gradient of vertical stripes is horizontal (dI/dx).
        // The structure (the stripes themselves) is vertical (theta ~ pi/2).
        let w = 100;
        let h = 100;
        let data: Vec<Real> = (0..h)
            .flat_map(|_| {
                (0..w).map(|c| {
                    (2.0 * std::f64::consts::PI * c as Real / 10.0).sin()
                })
            })
            .collect();
        let field = ImageField::new(data, w, h);
        let result = structure_tensor(&field, 3.0);

        // Check interior pixels (avoid boundary effects)
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
        // Vertical stripes -> theta ~ pi/2
        assert!(
            (mean_theta - std::f64::consts::FRAC_PI_2).abs() < 0.3,
            "vertical stripes should give theta ~ pi/2, got {mean_theta:.3}"
        );
    }
}
