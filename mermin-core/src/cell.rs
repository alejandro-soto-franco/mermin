// mermin-core/src/cell.rs

use crate::Real;
use serde::{Deserialize, Serialize};

/// Complete per-cell measurement record.
/// Each field is filled progressively by different analysis stages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellRecord {
    /// Unique cell label from segmentation mask.
    pub label: i32,
    /// Centroid position in physical coordinates (um).
    pub centroid: [Real; 2],

    // -- Shape (mermin-shape) --
    /// Cell area in um^2.
    pub area: Real,
    /// Cell perimeter in um.
    pub perimeter: Real,
    /// Shape index p_0 = P / sqrt(A).
    pub shape_index: Real,
    /// Convexity = A / A_convex_hull.
    pub convexity: Real,
    /// Elongation magnitude from Minkowski W1^{1,1} tensor.
    pub elongation: Real,
    /// Elongation orientation angle in radians [0, pi).
    pub elongation_angle: Real,
    /// k-atic shape mode amplitudes |W1^{s,0}| / W0 for k = [1, 2, 4, 6].
    pub shape_katic: [Real; 4],

    // -- Orientation (mermin-orient) --
    /// Per-cell mean internal alignment |psi_k| for k = [1, 2, 4, 6],
    /// at the optimal scale sigma*.
    pub internal_katic: [Real; 4],
    /// Optimal structure tensor scale sigma* (pixels) where |psi_2| is maximized.
    pub optimal_scale: Real,
    /// Nuclear aspect ratio (major_axis / minor_axis from DAPI).
    pub nuclear_aspect_ratio: Real,
    /// Nuclear orientation angle in radians [0, pi).
    pub nuclear_angle: Real,

    // -- Topology (mermin-topo) --
    /// Number of Delaunay neighbors.
    pub n_neighbors: usize,
}

impl CellRecord {
    /// Create a partially filled CellRecord with only label and centroid.
    /// All other fields initialized to zero/default.
    pub fn new(label: i32, centroid: [Real; 2]) -> Self {
        Self {
            label,
            centroid,
            area: 0.0,
            perimeter: 0.0,
            shape_index: 0.0,
            convexity: 0.0,
            elongation: 0.0,
            elongation_angle: 0.0,
            shape_katic: [0.0; 4],
            internal_katic: [0.0; 4],
            optimal_scale: 0.0,
            nuclear_aspect_ratio: 0.0,
            nuclear_angle: 0.0,
            n_neighbors: 0,
        }
    }
}
