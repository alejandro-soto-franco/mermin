// mermin-orient/src/gradient.rs

use mermin_core::{ImageField, Real};

/// Compute image gradient using the Scharr operator (better rotational symmetry
/// than Sobel for orientation analysis).
///
/// Returns (grad_x, grad_y) as separate ImageFields.
pub fn scharr_gradient(field: &ImageField) -> (ImageField, ImageField) {
    let (w, h) = (field.width, field.height);
    let mut gx = ImageField::zeros(w, h);
    let mut gy = ImageField::zeros(w, h);

    // Scharr kernels:
    // Kx = [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]] / 32
    // Ky = [[-3, -10, -3], [0, 0, 0], [3, 10, 3]] / 32
    let norm: Real = 1.0 / 32.0;

    for row in 1..h - 1 {
        for col in 1..w - 1 {
            let v = |r: usize, c: usize| field.get(r, c);

            let dx = -3.0 * v(row - 1, col - 1)
                + 3.0 * v(row - 1, col + 1)
                - 10.0 * v(row, col - 1)
                + 10.0 * v(row, col + 1)
                - 3.0 * v(row + 1, col - 1)
                + 3.0 * v(row + 1, col + 1);

            let dy = -3.0 * v(row - 1, col - 1)
                - 10.0 * v(row - 1, col)
                - 3.0 * v(row - 1, col + 1)
                + 3.0 * v(row + 1, col - 1)
                + 10.0 * v(row + 1, col)
                + 3.0 * v(row + 1, col + 1);

            *gx.get_mut(row, col) = dx * norm;
            *gy.get_mut(row, col) = dy * norm;
        }
    }

    (gx, gy)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn horizontal_ramp_gradient() {
        // Image with horizontal ramp: I(r,c) = c
        let w = 20;
        let h = 20;
        let data: Vec<Real> = (0..h)
            .flat_map(|_| (0..w).map(|c| c as Real))
            .collect();
        let field = ImageField::new(data, w, h);
        let (gx, gy) = scharr_gradient(&field);

        // Interior pixels should have gx ~ 1.0, gy ~ 0.0
        for row in 2..h - 2 {
            for col in 2..w - 2 {
                assert!(
                    (gx.get(row, col) - 1.0).abs() < 0.1,
                    "gx at ({row},{col}) = {}, expected ~1.0",
                    gx.get(row, col)
                );
                assert!(
                    gy.get(row, col).abs() < 0.1,
                    "gy at ({row},{col}) = {}, expected ~0.0",
                    gy.get(row, col)
                );
            }
        }
    }
}
