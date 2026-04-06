//! Topological defect detection, Poincare-Hopf validation, and persistent homology.

pub mod defects;
pub mod director_mesh;
pub mod persistence;
pub mod poincare_hopf;

pub use defects::{Defect, detect_defects};
pub use director_mesh::{directors_to_frames, embed_director_as_frame};
pub use persistence::{PersistenceDiagram, PersistencePair, compute_persistence};
pub use poincare_hopf::validate_poincare_hopf;
