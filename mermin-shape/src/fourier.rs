// mermin-shape/src/fourier.rs

use mermin_core::{BoundaryContour, Real};

/// Fourier decomposition of the boundary contour in polar coordinates
/// relative to the centroid.
///
/// Computes a_k = (1/N) * sum_i r_i * exp(-i*k*theta_i)
/// where (r_i, theta_i) are polar coordinates of boundary point i
/// relative to the centroid.
///
/// Returns (|a_k|, phase_k) where |a_k| is the amplitude and phase_k is the
/// argument of the complex coefficient.
pub fn fourier_mode(contour: &BoundaryContour, k: u32) -> (Real, Real) {
    let cen = contour.centroid();
    let n = contour.n_points() as Real;
    let kf = k as Real;
    let mut re = 0.0;
    let mut im = 0.0;

    for p in &contour.points {
        let dx = p.x - cen.x;
        let dy = p.y - cen.y;
        let r = (dx * dx + dy * dy).sqrt();
        let theta = dy.atan2(dx);

        re += r * (kf * theta).cos();
        im -= r * (kf * theta).sin(); // exp(-ik*theta) = cos - i*sin
    }

    re /= n;
    im /= n;

    let amplitude = (re * re + im * im).sqrt();
    let phase = im.atan2(re);
    (amplitude, phase)
}

/// Compute normalized Fourier amplitudes |a_k| / |a_0| for k = [1, 2, 3, 4, 6].
/// a_0 is the mean radius.
pub fn fourier_spectrum(contour: &BoundaryContour) -> [Real; 5] {
    let (a0, _) = fourier_mode(contour, 0);
    if a0 < 1e-15 {
        return [0.0; 5];
    }
    let ks = [1, 2, 3, 4, 6];
    let mut spectrum = [0.0; 5];
    for (i, &k) in ks.iter().enumerate() {
        let (ak, _) = fourier_mode(contour, k);
        spectrum[i] = ak / a0;
    }
    spectrum
}

#[cfg(test)]
mod tests {
    use super::*;
    use mermin_core::Point2;

    #[test]
    fn circle_fourier_isotropic() {
        let n = 200;
        let pts: Vec<Point2> = (0..n)
            .map(|i| {
                let theta = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
                Point2::new(theta.cos(), theta.sin())
            })
            .collect();
        let contour = BoundaryContour::new(pts).unwrap();
        let spec = fourier_spectrum(&contour);
        for (i, &v) in spec.iter().enumerate() {
            assert!(v < 0.02, "circle fourier[{i}] should be ~0, got {v}");
        }
    }

    #[test]
    fn ellipse_k2_dominant() {
        // Ellipse with semi-axes a=3, b=1
        let n = 200;
        let pts: Vec<Point2> = (0..n)
            .map(|i| {
                let theta = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
                Point2::new(3.0 * theta.cos(), 1.0 * theta.sin())
            })
            .collect();
        let contour = BoundaryContour::new(pts).unwrap();
        let spec = fourier_spectrum(&contour);
        // k=2 should dominate
        assert!(spec[1] > spec[0], "ellipse: k=2 ({}) > k=1 ({})", spec[1], spec[0]);
        assert!(spec[1] > spec[2], "ellipse: k=2 ({}) > k=3 ({})", spec[1], spec[2]);
    }
}
