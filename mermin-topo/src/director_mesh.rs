// mermin-topo/src/director_mesh.rs

use mermin_core::Real;
use nalgebra::SMatrix;

/// Embed a 2D director angle theta (in [0, pi)) as a 3D SO(3) frame.
///
/// The director n = (cos(theta), sin(theta), 0) is embedded as
/// a rotation about the z-axis by angle theta:
///   R = [[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]]
///
/// This allows using cartan's 3D holonomy machinery for 2D defect detection.
pub fn embed_director_as_frame(theta: Real) -> SMatrix<Real, 3, 3> {
    let c = theta.cos();
    let s = theta.sin();
    SMatrix::<Real, 3, 3>::new(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0)
}

/// Convert an array of per-cell director angles to SO(3) frames.
pub fn directors_to_frames(thetas: &[Real]) -> Vec<SMatrix<Real, 3, 3>> {
    thetas.iter().map(|&t| embed_director_as_frame(t)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_is_rotation() {
        let frame = embed_director_as_frame(0.7);
        // R^T * R should be identity
        let rtr = frame.transpose() * frame;
        let id = SMatrix::<Real, 3, 3>::identity();
        assert!((rtr - id).norm() < 1e-12, "frame should be orthogonal");
        // det should be +1
        assert!(
            (frame.determinant() - 1.0).abs() < 1e-12,
            "det should be +1"
        );
    }
}
