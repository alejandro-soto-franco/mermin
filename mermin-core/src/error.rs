// mermin-core/src/error.rs

use thiserror::Error;

/// All fallible operations in mermin return this error type.
#[derive(Debug, Error)]
pub enum MerminError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: String, got: String },

    #[error("contour has {n} points, need at least {min}")]
    ContourTooShort { n: usize, min: usize },

    #[error("empty cell mask: label {label} has zero pixels")]
    EmptyMask { label: i32 },

    #[error("singular matrix in eigendecomposition")]
    SingularMatrix,

    #[error("invalid k-atic order: k={k}, must be positive")]
    InvalidK { k: u32 },

    #[error("no cells found in segmentation")]
    NoCells,

    #[error("Poincare-Hopf violation: charge sum {sum:.3} != Euler characteristic {chi}")]
    PoincareHopfViolation { sum: f64, chi: i32 },

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, MerminError>;
