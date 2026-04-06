// mermin-topo/src/persistence.rs

use mermin_core::Real;
use serde::{Deserialize, Serialize};

/// A single persistence pair (birth, death) in a persistence diagram.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PersistencePair {
    /// Filtration value at which the feature is born.
    pub birth: Real,
    /// Filtration value at which the feature dies. f64::INFINITY for essential features.
    pub death: Real,
    /// Homological dimension: 0 = connected component, 1 = loop.
    pub dimension: usize,
}

impl PersistencePair {
    pub fn persistence(&self) -> Real {
        self.death - self.birth
    }
}

/// Result of persistent homology computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceDiagram {
    pub pairs: Vec<PersistencePair>,
}

impl PersistenceDiagram {
    /// Filter to only pairs of a given dimension.
    pub fn dimension(&self, dim: usize) -> Vec<PersistencePair> {
        self.pairs
            .iter()
            .filter(|p| p.dimension == dim)
            .copied()
            .collect()
    }
}

/// Compute persistent homology of a simplicial complex filtration.
///
/// Input: a filtration of simplices, each with a birth time and boundary.
/// Uses the standard column reduction algorithm on the boundary matrix.
///
/// `vertices`: (vertex_index, filtration_value) pairs.
/// `edges`: (v0, v1, filtration_value) triples.
/// `triangles`: (v0, v1, v2, filtration_value) quads.
///
/// Filtration values for edges and triangles should be the max of their
/// vertex values (lower-star filtration).
pub fn compute_persistence(
    vertices: &[(usize, Real)],
    edges: &[(usize, usize, Real)],
    triangles: &[(usize, usize, usize, Real)],
) -> PersistenceDiagram {
    // Build the filtration: all simplices sorted by (filtration_value, dimension, index)
    // Simplex representation: (filtration_value, dimension, original_index, boundary_indices)

    let n_vert = vertices.len();
    let n_edge = edges.len();
    let n_tri = triangles.len();
    let n_total = n_vert + n_edge + n_tri;

    // Assign global indices: vertices [0..n_vert), edges [n_vert..n_vert+n_edge),
    // triangles [n_vert+n_edge..)

    struct Simplex {
        filt: Real,
        dim: usize,
        global_idx: usize,
        boundary: Vec<usize>, // global indices of boundary simplices
    }

    let mut simplices: Vec<Simplex> = Vec::with_capacity(n_total);

    // Vertices (dimension 0, empty boundary)
    for (i, &(_, filt)) in vertices.iter().enumerate() {
        simplices.push(Simplex {
            filt,
            dim: 0,
            global_idx: i,
            boundary: vec![],
        });
    }

    // Build vertex_index -> global_index map
    let mut vert_map = std::collections::HashMap::new();
    for (i, &(vi, _)) in vertices.iter().enumerate() {
        vert_map.insert(vi, i);
    }

    // Build edge -> global_index map for triangle boundaries
    let mut edge_map = std::collections::HashMap::new();
    for (i, &(v0, v1, filt)) in edges.iter().enumerate() {
        let gi = n_vert + i;
        let key = if v0 < v1 { (v0, v1) } else { (v1, v0) };
        edge_map.insert(key, gi);
        let bv0 = *vert_map.get(&v0).unwrap_or(&0);
        let bv1 = *vert_map.get(&v1).unwrap_or(&0);
        simplices.push(Simplex {
            filt,
            dim: 1,
            global_idx: gi,
            boundary: vec![bv0, bv1],
        });
    }

    // Triangles (dimension 2, boundary = 3 edges)
    for (i, &(v0, v1, v2, filt)) in triangles.iter().enumerate() {
        let gi = n_vert + n_edge + i;
        let mut bdry = Vec::new();
        for &(a, b) in &[(v0, v1), (v1, v2), (v0, v2)] {
            let key = if a < b { (a, b) } else { (b, a) };
            if let Some(&edge_gi) = edge_map.get(&key) {
                bdry.push(edge_gi);
            }
        }
        simplices.push(Simplex {
            filt,
            dim: 2,
            global_idx: gi,
            boundary: bdry,
        });
    }

    // Sort by (filtration value, dimension) for the filtration order
    simplices.sort_by(|a, b| a.filt.partial_cmp(&b.filt).unwrap().then(a.dim.cmp(&b.dim)));

    // Build permutation: global_idx -> filtration_order
    let mut order_map = vec![0usize; n_total];
    for (order, s) in simplices.iter().enumerate() {
        order_map[s.global_idx] = order;
    }

    // Boundary matrix in filtration order (columns are simplices, entries are boundary indices)
    let mut columns: Vec<Vec<usize>> = Vec::with_capacity(n_total);
    for s in &simplices {
        let mut col: Vec<usize> = s.boundary.iter().map(|&gi| order_map[gi]).collect();
        col.sort();
        columns.push(col);
    }

    // Standard column reduction (left-to-right)
    let mut low: Vec<Option<usize>> = vec![None; n_total]; // low[col] = lowest row index
    let mut pivot_col: Vec<Option<usize>> = vec![None; n_total]; // pivot_col[row] = which col has this as pivot

    for j in 0..n_total {
        loop {
            let lowest = columns[j].last().copied();
            match lowest {
                None => break,
                Some(l) => {
                    match pivot_col[l] {
                        None => {
                            low[j] = Some(l);
                            pivot_col[l] = Some(j);
                            break;
                        }
                        Some(j_prime) => {
                            // XOR (symmetric difference) columns[j] with columns[j_prime]
                            let other = columns[j_prime].clone();
                            let mut merged = Vec::new();
                            let (mut a, mut b) = (0, 0);
                            while a < columns[j].len() && b < other.len() {
                                match columns[j][a].cmp(&other[b]) {
                                    std::cmp::Ordering::Less => {
                                        merged.push(columns[j][a]);
                                        a += 1;
                                    }
                                    std::cmp::Ordering::Greater => {
                                        merged.push(other[b]);
                                        b += 1;
                                    }
                                    std::cmp::Ordering::Equal => {
                                        // Cancel (mod 2)
                                        a += 1;
                                        b += 1;
                                    }
                                }
                            }
                            while a < columns[j].len() {
                                merged.push(columns[j][a]);
                                a += 1;
                            }
                            while b < other.len() {
                                merged.push(other[b]);
                                b += 1;
                            }
                            columns[j] = merged;
                        }
                    }
                }
            }
        }
    }

    // Extract persistence pairs
    let mut pairs = Vec::new();
    let mut paired = vec![false; n_total];

    for j in 0..n_total {
        if let Some(i) = low[j] {
            // (i, j) is a persistence pair: i is born, j kills it
            paired[i] = true;
            paired[j] = true;
            let birth = simplices[i].filt;
            let death = simplices[j].filt;
            let dim = simplices[i].dim;
            if (death - birth).abs() > 1e-15 {
                pairs.push(PersistencePair {
                    birth,
                    death,
                    dimension: dim,
                });
            }
        }
    }

    // Essential features: unpaired simplices with dimension 0 get infinite death
    for j in 0..n_total {
        if !paired[j] && simplices[j].dim == 0 {
            pairs.push(PersistencePair {
                birth: simplices[j].filt,
                death: Real::INFINITY,
                dimension: 0,
            });
        }
    }

    PersistenceDiagram { pairs }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triangle_persistence() {
        // 3 vertices, 3 edges, 1 triangle
        // Filtration: vertices at 0, edges at 1, triangle at 2
        let vertices = vec![(0, 0.0), (1, 0.0), (2, 0.0)];
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)];
        let triangles = vec![(0, 1, 2, 2.0)];

        let pd = compute_persistence(&vertices, &edges, &triangles);

        // H0: 3 components born at 0, two die at 1 (when edges connect them), one essential
        let h0 = pd.dimension(0);
        let essential: Vec<_> = h0.iter().filter(|p| p.death.is_infinite()).collect();
        assert_eq!(essential.len(), 1, "should have 1 essential H0 feature");

        // H1: one loop born at 1 (third edge closes cycle), killed at 2 (triangle fills it)
        let h1 = pd.dimension(1);
        assert_eq!(h1.len(), 1, "should have 1 H1 feature");
        assert!((h1[0].birth - 1.0).abs() < 1e-10);
        assert!((h1[0].death - 2.0).abs() < 1e-10);
    }
}
