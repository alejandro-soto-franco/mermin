// mermin-orient/src/gradient.rs

use mermin_core::{ImageField, Real};
use rayon::prelude::*;

/// Compute image gradient using the Scharr operator (better rotational symmetry
/// than Sobel for orientation analysis).
///
/// Returns (grad_x, grad_y) as separate ImageFields.
/// Parallelized with rayon over rows.
pub fn scharr_gradient(field: &ImageField) -> (ImageField, ImageField) {
    let (w, h) = (field.width, field.height);
    let mut gx_data = vec![0.0_f64; w * h];
    let mut gy_data = vec![0.0_f64; w * h];

    let norm: Real = 1.0 / 32.0;
    let src = &field.data;

    gx_data
        .par_chunks_mut(w)
        .zip(gy_data.par_chunks_mut(w))
        .enumerate()
        .for_each(|(row, (gx_row, gy_row))| {
            if row == 0 || row >= h - 1 {
                return;
            }
            let prev = (row - 1) * w;
            let curr = row * w;
            let next = (row + 1) * w;

            for col in 1..w - 1 {
                let tl = src[prev + col - 1];
                let tr = src[prev + col + 1];
                let ml = src[curr + col - 1];
                let mr = src[curr + col + 1];
                let bl = src[next + col - 1];
                let br = src[next + col + 1];
                let tc = src[prev + col];
                let bc = src[next + col];

                gx_row[col] =
                    (-3.0 * tl + 3.0 * tr - 10.0 * ml + 10.0 * mr - 3.0 * bl + 3.0 * br) * norm;

                gy_row[col] =
                    (-3.0 * tl - 10.0 * tc - 3.0 * tr + 3.0 * bl + 10.0 * bc + 3.0 * br) * norm;
            }
        });

    (
        ImageField::new(gx_data, w, h),
        ImageField::new(gy_data, w, h),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn horizontal_ramp_gradient() {
        let w = 20;
        let h = 20;
        let data: Vec<Real> = (0..h).flat_map(|_| (0..w).map(|c| c as Real)).collect();
        let field = ImageField::new(data, w, h);
        let (gx, gy) = scharr_gradient(&field);

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
