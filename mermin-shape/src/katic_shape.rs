// mermin-shape/src/katic_shape.rs

use mermin_core::{BoundaryContour, Real};

/// Compute the k-atic shape amplitude from the boundary contour.
///
/// Uses the Minkowski tensor approach: for each edge, accumulate
///   q_k = sum_edges |e_i| * exp(i * k * phi_i)
/// where phi_i is the angle of the outward normal of edge i.
///
/// Returns the normalized amplitude |q_k| / W_1 in [0, 1].
/// Value of 1 means perfect k-fold symmetry, 0 means isotropic.
pub fn katic_shape_amplitude(contour: &BoundaryContour, k: u32) -> Real {
    let pts = &contour.points;
    let n = pts.len();
    let mut re = 0.0;
    let mut im = 0.0;
    let mut total_length = 0.0;

    for i in 0..n {
        let j = (i + 1) % n;
        let dx = pts[j].x - pts[i].x;
        let dy = pts[j].y - pts[i].y;
        let edge_len = (dx * dx + dy * dy).sqrt();

        if edge_len < 1e-15 {
            continue;
        }

        // Outward normal angle (edge rotated +90 degrees for CCW)
        let phi = dy.atan2(-dx);
        let kf = k as Real;
        re += edge_len * (kf * phi).cos();
        im += edge_len * (kf * phi).sin();
        total_length += edge_len;
    }

    if total_length < 1e-15 {
        return 0.0;
    }

    (re * re + im * im).sqrt() / total_length
}

/// Compute k-atic shape amplitudes for k = [1, 2, 4, 6].
/// Returns [q_1, q_2, q_4, q_6], each in [0, 1].
pub fn katic_shape_spectrum(contour: &BoundaryContour) -> [Real; 4] {
    [
        katic_shape_amplitude(contour, 1),
        katic_shape_amplitude(contour, 2),
        katic_shape_amplitude(contour, 4),
        katic_shape_amplitude(contour, 6),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use mermin_core::Point2;

    #[test]
    fn regular_hexagon_k6() {
        let pts: Vec<Point2> = (0..6)
            .map(|i| {
                let theta = std::f64::consts::FRAC_PI_3 * i as f64;
                Point2::new(theta.cos(), theta.sin())
            })
            .collect();
        let contour = BoundaryContour::new(pts).unwrap();
        let q6 = katic_shape_amplitude(&contour, 6);
        // Regular hexagon should have strong k=6 mode
        assert!(q6 > 0.9, "regular hexagon q6 should be ~1.0, got {q6}");
    }

    #[test]
    fn circle_isotropic() {
        let n = 200;
        let pts: Vec<Point2> = (0..n)
            .map(|i| {
                let theta = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
                Point2::new(theta.cos(), theta.sin())
            })
            .collect();
        let contour = BoundaryContour::new(pts).unwrap();
        let spectrum = katic_shape_spectrum(&contour);
        for (i, &q) in spectrum.iter().enumerate() {
            assert!(q < 0.05, "circle q[{i}] should be ~0, got {q}");
        }
    }

    #[test]
    fn rectangle_nematic() {
        let contour = BoundaryContour::new(vec![
            Point2::new(0.0, 0.0),
            Point2::new(4.0, 0.0),
            Point2::new(4.0, 1.0),
            Point2::new(0.0, 1.0),
        ])
        .unwrap();
        let q2 = katic_shape_amplitude(&contour, 2);
        // Rectangle has strong 2-fold symmetry
        assert!(q2 > 0.4, "rectangle q2 should be significant, got {q2}");
    }
}
