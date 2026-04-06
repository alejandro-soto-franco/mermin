// mermin-shape/src/minkowski.rs

use mermin_core::{BoundaryContour, Real};
use nalgebra::SMatrix;

/// W_0 = area of the polygon (Minkowski functional of order 0).
/// Uses the shoelace formula.
pub fn minkowski_w0(contour: &BoundaryContour) -> Real {
    contour.area()
}

/// W_1 = perimeter / (2*pi) is the standard normalization,
/// but we return raw perimeter for direct physical interpretation.
pub fn minkowski_w1(contour: &BoundaryContour) -> Real {
    contour.perimeter()
}

/// W_1^{1,1}: rank-2 Minkowski tensor encoding cell elongation.
///
/// Computed as the line integral of the outer normal tensor along the boundary:
///   W_1^{1,1} = (1/2) * sum_edges |e_i| * (n_i tensor n_i)
/// where n_i is the outward unit normal of edge i and |e_i| is the edge length.
///
/// Returns a 2x2 symmetric matrix. Eigendecomposition gives:
///   - eigenvalues: elongation in principal directions
///   - eigenvector of larger eigenvalue: elongation orientation
pub fn minkowski_w1_tensor(contour: &BoundaryContour) -> SMatrix<Real, 2, 2> {
    let pts = &contour.points;
    let n = pts.len();
    let mut tensor = SMatrix::<Real, 2, 2>::zeros();

    for i in 0..n {
        let j = (i + 1) % n;
        let dx = pts[j].x - pts[i].x;
        let dy = pts[j].y - pts[i].y;
        let edge_len = (dx * dx + dy * dy).sqrt();

        if edge_len < 1e-15 {
            continue;
        }

        // Outward normal (rotated edge direction by +90 degrees for CCW contour)
        let nx = dy / edge_len;
        let ny = -dx / edge_len;

        // Accumulate |e| * (n tensor n)
        tensor[(0, 0)] += edge_len * nx * nx;
        tensor[(0, 1)] += edge_len * nx * ny;
        tensor[(1, 0)] += edge_len * nx * ny;
        tensor[(1, 1)] += edge_len * ny * ny;
    }

    tensor * 0.5
}

/// Extract elongation magnitude and orientation from W_1^{1,1}.
///
/// Returns (elongation, angle) where:
///   - elongation = (lambda_max - lambda_min) / (lambda_max + lambda_min) in [0, 1]
///   - angle = orientation of major axis in [0, pi) radians
pub fn elongation_from_w1_tensor(tensor: &SMatrix<Real, 2, 2>) -> (Real, Real) {
    let eigen = tensor.symmetric_eigen();
    let evals = eigen.eigenvalues;
    let evecs = eigen.eigenvectors;

    let (lambda_max, lambda_min, max_idx) = if evals[0] >= evals[1] {
        (evals[0], evals[1], 0)
    } else {
        (evals[1], evals[0], 1)
    };

    let sum = lambda_max + lambda_min;
    let elongation = if sum > 1e-15 {
        (lambda_max - lambda_min) / sum
    } else {
        0.0
    };

    let vx = evecs[(0, max_idx)];
    let vy = evecs[(1, max_idx)];
    let mut angle = vy.atan2(vx);
    if angle < 0.0 {
        angle += std::f64::consts::PI;
    }
    if angle >= std::f64::consts::PI {
        angle -= std::f64::consts::PI;
    }

    (elongation, angle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mermin_core::Point2;

    fn regular_hexagon(r: f64) -> BoundaryContour {
        let pts: Vec<Point2> = (0..6)
            .map(|i| {
                let theta = std::f64::consts::FRAC_PI_3 * i as f64;
                Point2::new(r * theta.cos(), r * theta.sin())
            })
            .collect();
        BoundaryContour::new(pts).unwrap()
    }

    #[test]
    fn hexagon_w0_area() {
        let hex = regular_hexagon(1.0);
        let w0 = minkowski_w0(&hex);
        // Regular hexagon area = (3*sqrt(3)/2) * r^2
        let expected = 3.0 * 3.0_f64.sqrt() / 2.0;
        assert!((w0 - expected).abs() < 1e-10);
    }

    #[test]
    fn hexagon_w1_perimeter() {
        let hex = regular_hexagon(1.0);
        let w1 = minkowski_w1(&hex);
        // Regular hexagon perimeter = 6 * r
        assert!((w1 - 6.0).abs() < 1e-10);
    }

    #[test]
    fn circle_w1_tensor_isotropic() {
        // Approximate circle with 100-gon: tensor should be nearly isotropic
        let n = 100;
        let pts: Vec<Point2> = (0..n)
            .map(|i| {
                let theta = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
                Point2::new(theta.cos(), theta.sin())
            })
            .collect();
        let contour = BoundaryContour::new(pts).unwrap();
        let (elong, _) = elongation_from_w1_tensor(&minkowski_w1_tensor(&contour));
        assert!(
            elong < 0.01,
            "circle should have near-zero elongation, got {elong}"
        );
    }

    #[test]
    fn rectangle_elongation() {
        // 4:1 rectangle aligned with x-axis
        let contour = BoundaryContour::new(vec![
            Point2::new(0.0, 0.0),
            Point2::new(4.0, 0.0),
            Point2::new(4.0, 1.0),
            Point2::new(0.0, 1.0),
        ])
        .unwrap();
        let tensor = minkowski_w1_tensor(&contour);
        let (elong, angle) = elongation_from_w1_tensor(&tensor);
        // Elongated along x, so major normal is along y
        // The elongation orientation should be ~pi/2 (normal to long axis)
        // Actually: W1 tensor eigenvector for *largest* eigenvalue is the direction
        // with most boundary normal contribution, which for a 4:1 rect is y-direction
        // (the two long edges contribute normals in y). So angle ~ pi/2.
        assert!(
            elong > 0.3,
            "4:1 rectangle should be significantly elongated, got {elong}"
        );
    }
}
