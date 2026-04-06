// mermin-orient/src/gaussian.rs

use mermin_core::{ImageField, Real};

/// 1D Gaussian kernel, truncated at 4*sigma.
fn gaussian_kernel(sigma: Real) -> Vec<Real> {
    let radius = (4.0 * sigma).ceil() as usize;
    let size = 2 * radius + 1;
    let mut kernel = vec![0.0; size];
    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut sum = 0.0;

    for (i, k) in kernel.iter_mut().enumerate().take(size) {
        let x = i as Real - radius as Real;
        *k = (-x * x / two_sigma_sq).exp();
        sum += *k;
    }

    // Normalize
    for v in &mut kernel {
        *v /= sum;
    }
    kernel
}

/// Apply separable Gaussian blur to an ImageField.
/// Uses zero-padding at boundaries.
pub fn gaussian_blur(field: &ImageField, sigma: Real) -> ImageField {
    if sigma < 0.5 {
        return field.clone();
    }

    let kernel = gaussian_kernel(sigma);
    let radius = kernel.len() / 2;
    let (w, h) = (field.width, field.height);

    // Horizontal pass
    let mut temp = ImageField::zeros(w, h);
    for row in 0..h {
        for col in 0..w {
            let mut val = 0.0;
            for (ki, &kv) in kernel.iter().enumerate() {
                let src_col = col as isize + ki as isize - radius as isize;
                if src_col >= 0 && (src_col as usize) < w {
                    val += field.get(row, src_col as usize) * kv;
                }
            }
            *temp.get_mut(row, col) = val;
        }
    }

    // Vertical pass
    let mut out = ImageField::zeros(w, h);
    for row in 0..h {
        for col in 0..w {
            let mut val = 0.0;
            for (ki, &kv) in kernel.iter().enumerate() {
                let src_row = row as isize + ki as isize - radius as isize;
                if src_row >= 0 && (src_row as usize) < h {
                    val += temp.get(src_row as usize, col) * kv;
                }
            }
            *out.get_mut(row, col) = val;
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blur_preserves_constant() {
        // Use a large enough image so interior pixels are far from zero-padded boundaries.
        // Kernel radius = ceil(4*2) = 8, so we need at least 8px margin from each edge.
        let size = 30;
        let field = ImageField::new(vec![5.0; size * size], size, size);
        let blurred = gaussian_blur(&field, 2.0);
        for row in 10..20 {
            for col in 10..20 {
                assert!(
                    (blurred.get(row, col) - 5.0).abs() < 1e-6,
                    "constant field should be unchanged after blur at ({row},{col}), got {}",
                    blurred.get(row, col)
                );
            }
        }
    }

    #[test]
    fn blur_smooths_delta() {
        let mut field = ImageField::zeros(21, 21);
        *field.get_mut(10, 10) = 1.0;
        let blurred = gaussian_blur(&field, 2.0);
        // Peak should be reduced
        assert!(blurred.get(10, 10) < 0.5, "delta peak should be smoothed");
        // Neighbors should be positive
        assert!(
            blurred.get(10, 11) > 0.0,
            "neighbors should get some signal"
        );
    }
}
