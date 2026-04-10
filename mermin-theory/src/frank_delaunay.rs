// mermin-theory/src/frank_delaunay.rs

use mermin_core::Real;
use std::f64::consts::PI;

/// Frank elastic energy computed on a cell-centroid Delaunay mesh.
#[derive(Debug, Clone)]
pub struct DelaunayFrankEnergy {
    /// Total splay energy (area-weighted).
    pub splay: Real,
    /// Total bend energy (area-weighted).
    pub bend: Real,
    /// Ratio splay/bend.
    pub ratio: Real,
    /// Number of valid (non-excluded) triangles.
    pub n_triangles: usize,
}

/// Compute Frank splay and bend energy on a pre-computed Delaunay mesh.
///
/// For each triangle, computes the director gradient via linear FEM shape
/// functions, decomposes into splay (div n)^2 and bend (n x curl n)^2
/// using the mean triangle orientation, and returns area-weighted totals.
///
/// `centroids`: Nx2 positions in physical units.
/// `thetas`: N-length per-cell nematic orientations in [0, pi).
/// `simplices`: Mx3 vertex indices.
/// `max_edge`: maximum allowed edge length.
pub fn frank_energy_delaunay(
    centroids: &[[Real; 2]],
    thetas: &[Real],
    simplices: &[[usize; 3]],
    max_edge: Real,
) -> DelaunayFrankEnergy {
    let half_pi = PI / 2.0;
    let mut total_splay = 0.0;
    let mut total_bend = 0.0;
    let mut n_valid = 0usize;

    for &[i0, i1, i2] in simplices {
        let (x0, y0) = (centroids[i0][0], centroids[i0][1]);
        let (x1, y1) = (centroids[i1][0], centroids[i1][1]);
        let (x2, y2) = (centroids[i2][0], centroids[i2][1]);

        // Check edge lengths
        let e01 = ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt();
        let e12 = ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt();
        let e20 = ((x0 - x2).powi(2) + (y0 - y2).powi(2)).sqrt();
        if e01 > max_edge || e12 > max_edge || e20 > max_edge {
            continue;
        }

        // Triangle area via cross product
        let d1x = x1 - x0;
        let d1y = y1 - y0;
        let d2x = x2 - x0;
        let d2y = y2 - y0;
        let area = 0.5 * (d1x * d2y - d1y * d2x).abs();
        if area < 1e-10 {
            continue;
        }

        // Unwrap angles relative to vertex 0 using nematic symmetry
        let t0 = thetas[i0];
        let dt1 = (thetas[i1] - t0 + half_pi).rem_euclid(PI) - half_pi;
        let dt2 = (thetas[i2] - t0 + half_pi).rem_euclid(PI) - half_pi;
        let t1u = t0 + dt1;
        let t2u = t0 + dt2;

        // Determinant of the coordinate transform
        let det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
        if det.abs() < 1e-10 {
            continue;
        }

        // Gradients of theta within triangle (linear FEM shape functions)
        let dtheta_dx = (t0 * (y1 - y2) + t1u * (y2 - y0) + t2u * (y0 - y1)) / det;
        let dtheta_dy = (t0 * (x2 - x1) + t1u * (x0 - x2) + t2u * (x1 - x0)) / det;

        // Mean director angle in triangle
        let t_mean = (t0 + t1u + t2u) / 3.0;
        let cos_t = t_mean.cos();
        let sin_t = t_mean.sin();

        // Splay = (div n)^2 = (-sin(t) dt/dx + cos(t) dt/dy)^2
        let splay = (-sin_t * dtheta_dx + cos_t * dtheta_dy).powi(2);
        // Bend = (cos(t) dt/dx + sin(t) dt/dy)^2
        let bend = (cos_t * dtheta_dx + sin_t * dtheta_dy).powi(2);

        total_splay += splay * area;
        total_bend += bend * area;
        n_valid += 1;
    }

    let ratio = if total_bend > 1e-20 {
        total_splay / total_bend
    } else {
        1.0
    };

    DelaunayFrankEnergy {
        splay: total_splay,
        bend: total_bend,
        ratio,
        n_triangles: n_valid,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_field_zero_energy() {
        let centroids = vec![[0.0, 0.0], [10.0, 0.0], [5.0, 8.66], [15.0, 8.66]];
        let thetas = vec![0.7; 4];
        let simplices = vec![[0, 1, 2], [1, 3, 2]];
        let energy = frank_energy_delaunay(&centroids, &thetas, &simplices, 100.0);
        assert!(
            energy.splay < 1e-10,
            "uniform field: splay should be zero, got {}",
            energy.splay
        );
        assert!(
            energy.bend < 1e-10,
            "uniform field: bend should be zero, got {}",
            energy.bend
        );
        assert_eq!(energy.n_triangles, 2);
    }

    #[test]
    fn nonzero_gradient_has_energy() {
        let centroids = vec![[0.0, 0.0], [10.0, 0.0], [5.0, 8.66]];
        let thetas = vec![0.0, PI / 4.0, PI / 2.0]; // strong gradient
        let simplices = vec![[0, 1, 2]];
        let energy = frank_energy_delaunay(&centroids, &thetas, &simplices, 100.0);
        assert!(
            energy.splay + energy.bend > 0.0,
            "gradient field should have nonzero energy"
        );
    }
}
