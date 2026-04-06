// mermin-topo/src/defects.rs

use crate::director_mesh::directors_to_frames;
use cartan_geo::holonomy::scan_disclinations;
use mermin_core::Real;
use serde::{Deserialize, Serialize};

/// A topological defect detected in the orientation field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Defect {
    /// Grid position (row, col) of the plaquette center.
    pub position: [Real; 2],
    /// Topological charge: +0.5, -0.5 for nematic; +1, -1 for polar.
    pub charge: Real,
    /// Holonomy rotation angle in radians.
    pub angle: Real,
}

/// Detect topological defects on a 2D grid of director angles.
///
/// `thetas` is a row-major array of director angles, shaped (ny, nx).
/// `k` is the k-atic symmetry order (k=2 for nematic, k=1 for polar).
/// `threshold` is the minimum holonomy angle to classify as a defect
/// (pi/2 is standard for nematic half-disclinations).
///
/// Returns a list of Defect structs.
pub fn detect_defects(
    thetas: &[Real],
    nx: usize,
    ny: usize,
    k: u32,
    threshold: Real,
) -> Vec<Defect> {
    // Multiply angles by k/2 to map k-atic symmetry to nematic-equivalent
    // for cartan's holonomy (which detects pi rotations = 1/2 disclinations).
    // For k=2 (nematic), the angles are used directly.
    // For k=1 (polar), double the angles so +-1 defects map to +-pi rotations.
    // For k=6 (hexatic), multiply by 3 so +-1/6 defects map to +-pi/2.
    let scaled_thetas: Vec<Real> = thetas
        .iter()
        .map(|&t| t * (k as Real) / 2.0)
        .collect();

    let frames = directors_to_frames(&scaled_thetas);
    let disclinations = scan_disclinations(&frames, nx, ny, threshold);

    disclinations
        .into_iter()
        .map(|d| {
            let (py, px) = d.plaquette;
            // Convert plaquette indices to center coordinates
            let cx = px as Real + 0.5;
            let cy = py as Real + 0.5;

            // Charge: for nematic, angle ~ pi means +/- 1/2.
            // Sign from the holonomy trace: if the off-diagonal is positive,
            // the rotation is CCW (+1/2), otherwise CW (-1/2).
            let sign = if d.holonomy[(1, 0)] >= 0.0 {
                1.0
            } else {
                -1.0
            };
            let charge = sign * 0.5 * 2.0 / (k as Real);

            Defect {
                position: [cy, cx],
                charge,
                angle: d.angle,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_field_no_defects() {
        // All directors pointing the same way: no defects
        let nx = 10;
        let ny = 10;
        let thetas = vec![0.5; nx * ny];
        let defects = detect_defects(&thetas, nx, ny, 2, std::f64::consts::FRAC_PI_2);
        assert_eq!(defects.len(), 0, "uniform field should have no defects");
    }

    #[test]
    fn aster_defect() {
        // Radial aster pattern centered at (5,5): theta = atan2(y-5, x-5)
        // This should produce a +1 defect (or two +1/2 defects depending on resolution)
        let nx = 11;
        let ny = 11;
        let mut thetas = vec![0.0; nx * ny];
        for row in 0..ny {
            for col in 0..nx {
                let dy = row as Real - 5.0;
                let dx = col as Real - 5.0;
                let mut angle = dy.atan2(dx);
                if angle < 0.0 {
                    angle += std::f64::consts::PI;
                }
                thetas[row * nx + col] = angle;
            }
        }
        let defects = detect_defects(&thetas, nx, ny, 2, std::f64::consts::FRAC_PI_4);
        assert!(!defects.is_empty(), "aster should have at least one defect");
    }
}
