// mermin-orient/src/gaussian.rs

// Performance-critical inner loops use explicit indexing for clarity.
#![allow(clippy::needless_range_loop)]

use mermin_core::{ImageField, Real};
use rayon::prelude::*;

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

    for v in &mut kernel {
        *v /= sum;
    }
    kernel
}

/// Apply separable Gaussian blur to an ImageField.
/// Parallelized with rayon. Uses zero-padding at boundaries.
pub fn gaussian_blur(field: &ImageField, sigma: Real) -> ImageField {
    if sigma < 0.5 {
        return field.clone();
    }

    let kernel = gaussian_kernel(sigma);
    let radius = kernel.len() / 2;
    let (w, h) = (field.width, field.height);

    // Horizontal pass: each row is independent, parallelize over rows
    let mut temp_data = vec![0.0_f64; w * h];
    temp_data
        .par_chunks_mut(w)
        .enumerate()
        .for_each(|(row, out_row)| {
            let src = &field.data[row * w..(row + 1) * w];

            // Interior columns: no bounds check needed
            for col in radius..w.saturating_sub(radius) {
                let mut val = 0.0;
                for (ki, &kv) in kernel.iter().enumerate() {
                    // SAFETY: col + ki - radius is in [0, w) because
                    // col >= radius and col < w - radius and ki < 2*radius+1
                    val += unsafe { *src.get_unchecked(col + ki - radius) } * kv;
                }
                out_row[col] = val;
            }

            // Left boundary
            for col in 0..radius.min(w) {
                let mut val = 0.0;
                for (ki, &kv) in kernel.iter().enumerate() {
                    let src_col = col + ki;
                    if src_col >= radius && src_col - radius < w {
                        val += src[src_col - radius] * kv;
                    }
                }
                out_row[col] = val;
            }

            // Right boundary
            for col in w.saturating_sub(radius)..w {
                let mut val = 0.0;
                for (ki, &kv) in kernel.iter().enumerate() {
                    let src_col = col + ki;
                    if src_col >= radius && src_col - radius < w {
                        val += src[src_col - radius] * kv;
                    }
                }
                out_row[col] = val;
            }
        });

    // Vertical pass: parallelize over columns
    // For cache efficiency, process in column strips
    let mut out_data = vec![0.0_f64; w * h];
    out_data
        .par_chunks_mut(w)
        .enumerate()
        .for_each(|(row, out_row)| {
            // Interior rows: no bounds check
            if row >= radius && row + radius < h {
                for col in 0..w {
                    let mut val = 0.0;
                    for (ki, &kv) in kernel.iter().enumerate() {
                        let src_row = row + ki - radius;
                        val += unsafe { *temp_data.get_unchecked(src_row * w + col) } * kv;
                    }
                    out_row[col] = val;
                }
            } else {
                // Boundary rows
                for col in 0..w {
                    let mut val = 0.0;
                    for (ki, &kv) in kernel.iter().enumerate() {
                        let src_row_offset = row + ki;
                        if src_row_offset >= radius && src_row_offset - radius < h {
                            val += temp_data[(src_row_offset - radius) * w + col] * kv;
                        }
                    }
                    out_row[col] = val;
                }
            }
        });

    ImageField::new(out_data, w, h)
}

/// Apply Gaussian blur to three fields simultaneously, sharing the kernel computation.
/// This is more efficient than calling gaussian_blur three times because
/// the kernel is computed once and cache pressure is reduced.
pub fn gaussian_blur_triple(
    f1: &ImageField,
    f2: &ImageField,
    f3: &ImageField,
    sigma: Real,
) -> (ImageField, ImageField, ImageField) {
    if sigma < 0.5 {
        return (f1.clone(), f2.clone(), f3.clone());
    }

    let kernel = gaussian_kernel(sigma);
    let radius = kernel.len() / 2;
    let (w, h) = (f1.width, f1.height);
    let n = w * h;

    // Interleave the three fields for better cache usage during vertical pass
    // Horizontal pass for all three
    let mut t1 = vec![0.0_f64; n];
    let mut t2 = vec![0.0_f64; n];
    let mut t3 = vec![0.0_f64; n];

    // Horizontal pass: parallel over rows
    let chunk_size = w;
    t1.par_chunks_mut(chunk_size)
        .zip(t2.par_chunks_mut(chunk_size))
        .zip(t3.par_chunks_mut(chunk_size))
        .enumerate()
        .for_each(|(row, ((o1, o2), o3))| {
            let s1 = &f1.data[row * w..(row + 1) * w];
            let s2 = &f2.data[row * w..(row + 1) * w];
            let s3 = &f3.data[row * w..(row + 1) * w];

            for col in radius..w.saturating_sub(radius) {
                let mut v1 = 0.0;
                let mut v2 = 0.0;
                let mut v3 = 0.0;
                for (ki, &kv) in kernel.iter().enumerate() {
                    let idx = col + ki - radius;
                    unsafe {
                        v1 += *s1.get_unchecked(idx) * kv;
                        v2 += *s2.get_unchecked(idx) * kv;
                        v3 += *s3.get_unchecked(idx) * kv;
                    }
                }
                o1[col] = v1;
                o2[col] = v2;
                o3[col] = v3;
            }

            // Boundaries
            for col in (0..radius.min(w)).chain(w.saturating_sub(radius)..w) {
                let mut v1 = 0.0;
                let mut v2 = 0.0;
                let mut v3 = 0.0;
                for (ki, &kv) in kernel.iter().enumerate() {
                    let sc = col + ki;
                    if sc >= radius && sc - radius < w {
                        let idx = sc - radius;
                        v1 += s1[idx] * kv;
                        v2 += s2[idx] * kv;
                        v3 += s3[idx] * kv;
                    }
                }
                o1[col] = v1;
                o2[col] = v2;
                o3[col] = v3;
            }
        });

    // Vertical pass: parallel over rows
    let mut o1 = vec![0.0_f64; n];
    let mut o2 = vec![0.0_f64; n];
    let mut o3 = vec![0.0_f64; n];

    o1.par_chunks_mut(w)
        .zip(o2.par_chunks_mut(w))
        .zip(o3.par_chunks_mut(w))
        .enumerate()
        .for_each(|(row, ((r1, r2), r3))| {
            if row >= radius && row + radius < h {
                for col in 0..w {
                    let mut v1 = 0.0;
                    let mut v2 = 0.0;
                    let mut v3 = 0.0;
                    for (ki, &kv) in kernel.iter().enumerate() {
                        let sr = (row + ki - radius) * w + col;
                        unsafe {
                            v1 += *t1.get_unchecked(sr) * kv;
                            v2 += *t2.get_unchecked(sr) * kv;
                            v3 += *t3.get_unchecked(sr) * kv;
                        }
                    }
                    r1[col] = v1;
                    r2[col] = v2;
                    r3[col] = v3;
                }
            } else {
                for col in 0..w {
                    let mut v1 = 0.0;
                    let mut v2 = 0.0;
                    let mut v3 = 0.0;
                    for (ki, &kv) in kernel.iter().enumerate() {
                        let sro = row + ki;
                        if sro >= radius && sro - radius < h {
                            let sr = (sro - radius) * w + col;
                            v1 += t1[sr] * kv;
                            v2 += t2[sr] * kv;
                            v3 += t3[sr] * kv;
                        }
                    }
                    r1[col] = v1;
                    r2[col] = v2;
                    r3[col] = v3;
                }
            }
        });

    (
        ImageField::new(o1, w, h),
        ImageField::new(o2, w, h),
        ImageField::new(o3, w, h),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blur_preserves_constant() {
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
        assert!(blurred.get(10, 10) < 0.5, "delta peak should be smoothed");
        assert!(
            blurred.get(10, 11) > 0.0,
            "neighbors should get some signal"
        );
    }

    #[test]
    fn triple_matches_individual() {
        let w = 50;
        let h = 50;
        let f1 = ImageField::new((0..w * h).map(|i| (i as f64) * 0.01).collect(), w, h);
        let f2 = ImageField::new((0..w * h).map(|i| (i as f64) * 0.02).collect(), w, h);
        let f3 = ImageField::new((0..w * h).map(|i| (i as f64) * 0.03).collect(), w, h);

        let (t1, t2, t3) = gaussian_blur_triple(&f1, &f2, &f3, 3.0);
        let s1 = gaussian_blur(&f1, 3.0);
        let s2 = gaussian_blur(&f2, 3.0);
        let s3 = gaussian_blur(&f3, 3.0);

        for i in 0..w * h {
            assert!(
                (t1.data[i] - s1.data[i]).abs() < 1e-12,
                "triple f1 mismatch at {i}"
            );
            assert!(
                (t2.data[i] - s2.data[i]).abs() < 1e-12,
                "triple f2 mismatch at {i}"
            );
            assert!(
                (t3.data[i] - s3.data[i]).abs() < 1e-12,
                "triple f3 mismatch at {i}"
            );
        }
    }
}
