// mermin-orient/src/cell_orientation.rs

use mermin_core::Real;

/// Result of fitting an ellipse to a nuclear mask via second central moments.
#[derive(Debug, Clone, Copy)]
pub struct NuclearEllipse {
    /// Aspect ratio = major_axis / minor_axis (>= 1.0, 1.0 = circular).
    pub aspect_ratio: Real,
    /// Orientation of major axis in radians [0, pi).
    pub angle: Real,
    /// Major semi-axis length in pixels.
    pub semi_major: Real,
    /// Minor semi-axis length in pixels.
    pub semi_minor: Real,
    /// Centroid row.
    pub centroid_row: Real,
    /// Centroid col.
    pub centroid_col: Real,
}

/// Fit an ellipse to a binary nuclear mask using second central moments (moments of inertia).
///
/// `pixels` is a list of (row, col) coordinates of all pixels in the nuclear mask.
pub fn fit_nuclear_ellipse(pixels: &[(usize, usize)]) -> Option<NuclearEllipse> {
    let n = pixels.len();
    if n < 3 {
        return None;
    }

    let nf = n as Real;

    // Centroid
    let (mut cr, mut cc) = (0.0, 0.0);
    for &(r, c) in pixels {
        cr += r as Real;
        cc += c as Real;
    }
    cr /= nf;
    cc /= nf;

    // Second central moments
    let (mut mu20, mut mu02, mut mu11) = (0.0, 0.0, 0.0);
    for &(r, c) in pixels {
        let dr = r as Real - cr;
        let dc = c as Real - cc;
        mu20 += dr * dr;
        mu02 += dc * dc;
        mu11 += dr * dc;
    }
    mu20 /= nf;
    mu02 /= nf;
    mu11 /= nf;

    // Eigenvalues of the inertia tensor [[mu20, mu11], [mu11, mu02]]
    let trace = mu20 + mu02;
    let det_term = ((mu20 - mu02).powi(2) + 4.0 * mu11 * mu11).sqrt();

    let l1 = (trace + det_term) * 0.5;
    let l2 = (trace - det_term) * 0.5;

    let (lambda_max, lambda_min) = if l1 >= l2 { (l1, l2) } else { (l2, l1) };

    if lambda_min < 1e-15 {
        return None;
    }

    // Semi-axes: proportional to sqrt of eigenvalues.
    // For a uniform ellipse, mu along axis = a^2/4, so a = 2*sqrt(mu).
    let semi_major = 2.0 * lambda_max.sqrt();
    let semi_minor = 2.0 * lambda_min.sqrt();
    let aspect_ratio = semi_major / semi_minor;

    // Orientation: angle of eigenvector corresponding to lambda_max.
    // For the 2x2 symmetric matrix, eigenvector for larger eigenvalue:
    let mut angle = 0.5 * (2.0 * mu11).atan2(mu20 - mu02);
    if angle < 0.0 {
        angle += std::f64::consts::PI;
    }
    if angle >= std::f64::consts::PI {
        angle -= std::f64::consts::PI;
    }

    Some(NuclearEllipse {
        aspect_ratio,
        angle,
        semi_major,
        semi_minor,
        centroid_row: cr,
        centroid_col: cc,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circular_nucleus() {
        // Circle of radius 10 centered at (50, 50)
        let mut pixels = Vec::new();
        for r in 40..61 {
            for c in 40..61 {
                let dr = r as f64 - 50.0;
                let dc = c as f64 - 50.0;
                if dr * dr + dc * dc <= 100.0 {
                    pixels.push((r, c));
                }
            }
        }
        let e = fit_nuclear_ellipse(&pixels).unwrap();
        assert!(
            (e.aspect_ratio - 1.0).abs() < 0.15,
            "circle aspect ratio should be ~1.0, got {}",
            e.aspect_ratio
        );
    }

    #[test]
    fn elongated_nucleus() {
        // Horizontal ellipse: semi-major ~20 (horizontal), semi-minor ~5 (vertical)
        let mut pixels = Vec::new();
        for r in 0..100 {
            for c in 0..100 {
                let dr = (r as f64 - 50.0) / 5.0;
                let dc = (c as f64 - 50.0) / 20.0;
                if dr * dr + dc * dc <= 1.0 {
                    pixels.push((r, c));
                }
            }
        }
        let e = fit_nuclear_ellipse(&pixels).unwrap();
        assert!(
            e.aspect_ratio > 3.0,
            "4:1 ellipse should have aspect ratio > 3, got {}",
            e.aspect_ratio
        );
    }
}
