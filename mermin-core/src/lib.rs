// mermin-core/src/lib.rs

//! Core types, traits, and error handling for mermin.

pub mod cell;
pub mod contour;
pub mod error;
pub mod field;
pub mod types;

pub use cell::CellRecord;
pub use contour::BoundaryContour;
pub use error::{MerminError, Result};
pub use field::ImageField;
pub use types::{K_HEXATIC, K_NEMATIC, K_POLAR, K_TETRATIC, KValue, Point2, Real};
