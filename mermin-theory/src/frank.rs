// mermin-theory/src/frank.rs

use mermin_core::Real;
use serde::{Deserialize, Serialize};

/// Frank elastic energy decomposition result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrankEnergy {
    /// Total splay energy: integral of (div n)^2.
    pub splay: Real,
    /// Total bend energy: integral of |n x curl n|^2.
    pub bend: Real,
    /// Ratio splay/bend. >1 means splay-dominated (contractile-like),
    /// <1 means bend-dominated (extensile-like).
    pub ratio: Real,
}

/// Compute Frank elastic energy from a 2D director field on a regular grid.
///
/// For a 2D director n = (cos(theta), sin(theta)):
///   splay = (div n)^2 = (d(cos theta)/dx + d(sin theta)/dy)^2
///   bend  = (curl n . z)^2 = (d(sin theta)/dx - d(cos theta)/dy)^2
///
/// `thetas`: row-major director angles, shape (ny, nx).
/// `dx`: grid spacing in physical units.
///
/// Returns energy densities integrated over the domain.
pub fn frank_energy(thetas: &[Real], nx: usize, ny: usize, dx: Real) -> FrankEnergy {
    let mut splay_total = 0.0;
    let mut bend_total = 0.0;
    let dx2 = 2.0 * dx;

    for row in 1..ny - 1 {
        for col in 1..nx - 1 {
            let idx = |r: usize, c: usize| thetas[r * nx + c];

            // Central differences for cos(theta) and sin(theta)
            let dcos_dx =
                (idx(row, col + 1).cos() - idx(row, col - 1).cos()) / dx2;
            let dsin_dy =
                (idx(row + 1, col).sin() - idx(row - 1, col).sin()) / dx2;
            let dsin_dx =
                (idx(row, col + 1).sin() - idx(row, col - 1).sin()) / dx2;
            let dcos_dy =
                (idx(row + 1, col).cos() - idx(row - 1, col).cos()) / dx2;

            let s = dcos_dx + dsin_dy; // div n
            let b = dsin_dx - dcos_dy; // (curl n) . z

            splay_total += s * s;
            bend_total += b * b;
        }
    }

    // Multiply by cell area dx^2 for integration
    splay_total *= dx * dx;
    bend_total *= dx * dx;

    let ratio = if bend_total > 1e-15 {
        splay_total / bend_total
    } else {
        Real::INFINITY
    };

    FrankEnergy {
        splay: splay_total,
        bend: bend_total,
        ratio,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_field_zero_energy() {
        let nx = 20;
        let ny = 20;
        let thetas = vec![0.7; nx * ny];
        let energy = frank_energy(&thetas, nx, ny, 1.0);
        assert!(energy.splay < 1e-10, "uniform field should have zero splay");
        assert!(energy.bend < 1e-10, "uniform field should have zero bend");
    }

    #[test]
    fn pure_splay_aster() {
        // Radial aster: theta = atan2(y, x). This is a pure splay pattern.
        let nx = 41;
        let ny = 41;
        let cx = 20.0;
        let cy = 20.0;
        let mut thetas = vec![0.0; nx * ny];
        for row in 0..ny {
            for col in 0..nx {
                let dy = row as Real - cy;
                let dx = col as Real - cx;
                thetas[row * nx + col] = dy.atan2(dx);
            }
        }
        let energy = frank_energy(&thetas, nx, ny, 1.0);
        // Aster has splay but also some bend near the core.
        // Away from the singularity, splay should dominate.
        assert!(energy.splay > 0.0, "aster should have nonzero splay");
    }
}
