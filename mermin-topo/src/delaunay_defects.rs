// mermin-topo/src/delaunay_defects.rs

use mermin_core::Real;
use std::f64::consts::PI;

/// A defect detected on a cell-centroid Delaunay mesh.
#[derive(Debug, Clone)]
pub struct DelaunayDefect {
    /// Position (x, y) in physical units (centroid of the triangle).
    pub x: Real,
    pub y: Real,
    /// Topological charge: +0.5 or -0.5 for nematic half-disclinations.
    pub charge: Real,
}

/// Detect nematic +/-1/2 defects on a pre-computed Delaunay mesh of cell centroids.
///
/// `centroids`: Nx2 flat array of (x, y) positions in physical units.
/// `thetas`: N-length array of per-cell nematic orientations in [0, pi).
/// `simplices`: Mx3 flat array of vertex indices (from scipy Delaunay).
/// `max_edge`: maximum allowed edge length; triangles with any edge exceeding this are skipped.
///
/// Returns a list of detected defects.
pub fn detect_defects_delaunay(
    centroids: &[[Real; 2]],
    thetas: &[Real],
    simplices: &[[usize; 3]],
    max_edge: Real,
) -> Vec<DelaunayDefect> {
    let half_pi = PI / 2.0;
    let mut defects = Vec::new();

    for &[i0, i1, i2] in simplices {
        let p0 = centroids[i0];
        let p1 = centroids[i1];
        let p2 = centroids[i2];

        // Check edge lengths
        let e01 = ((p1[0] - p0[0]).powi(2) + (p1[1] - p0[1]).powi(2)).sqrt();
        let e12 = ((p2[0] - p1[0]).powi(2) + (p2[1] - p1[1]).powi(2)).sqrt();
        let e20 = ((p0[0] - p2[0]).powi(2) + (p0[1] - p2[1]).powi(2)).sqrt();
        if e01 > max_edge || e12 > max_edge || e20 > max_edge {
            continue;
        }

        // Winding number: sum of nematic angle differences mapped to [-pi/2, pi/2]
        let t = [thetas[i0], thetas[i1], thetas[i2]];
        let pairs = [(0, 1), (1, 2), (2, 0)];
        let mut winding = 0.0;
        for &(a, b) in &pairs {
            let diff = (t[b] - t[a] + half_pi).rem_euclid(PI) - half_pi;
            winding += diff;
        }
        let charge = winding / PI;
        if charge.abs() > 0.3 {
            let cx = (p0[0] + p1[0] + p2[0]) / 3.0;
            let cy = (p0[1] + p1[1] + p2[1]) / 3.0;
            defects.push(DelaunayDefect {
                x: cx,
                y: cy,
                charge: (charge * 2.0).round() / 2.0, // snap to +/-0.5
            });
        }
    }

    defects
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_field_no_defects() {
        // All cells aligned at 0.5 rad: no defects
        let centroids = vec![[0.0, 0.0], [1.0, 0.0], [0.5, 0.87], [1.5, 0.87]];
        let thetas = vec![0.5; 4];
        let simplices = vec![[0, 1, 2], [1, 3, 2]];
        let defects = detect_defects_delaunay(&centroids, &thetas, &simplices, 10.0);
        assert!(defects.is_empty(), "uniform field should have no defects");
    }

    #[test]
    fn long_edge_excluded() {
        let centroids = vec![[0.0, 0.0], [100.0, 0.0], [50.0, 87.0]];
        let thetas = vec![0.0, PI / 4.0, PI / 2.0];
        let simplices = vec![[0, 1, 2]];
        // max_edge = 50 should exclude this triangle (edges ~100)
        let defects = detect_defects_delaunay(&centroids, &thetas, &simplices, 50.0);
        assert!(defects.is_empty());
    }
}
