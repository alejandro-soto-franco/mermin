//! mermin: k-atic alignment analysis of fluorescence microscopy.
//!
//! Named after N. David Mermin, whose 1979 Reviews of Modern Physics paper
//! "The topological theory of defects in ordered media" provides the
//! mathematical framework this tool implements.

pub use mermin_core as core;
pub use mermin_orient as orient;
pub use mermin_shape as shape;
pub use mermin_stats as stats;
pub use mermin_theory as theory;
pub use mermin_topo as topo;

// Re-export most-used types at crate root
pub use mermin_core::{
    BoundaryContour, CellRecord, ImageField, KValue, MerminError, Point2, Real, Result,
    K_HEXATIC, K_NEMATIC, K_POLAR, K_TETRATIC,
};
