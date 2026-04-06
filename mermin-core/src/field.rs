// mermin-core/src/field.rs

use crate::Real;

/// A 2D scalar field on a regular grid, stored row-major.
/// `data[row * width + col]` gives the value at pixel (row, col).
#[derive(Debug, Clone)]
pub struct ImageField {
    pub data: Vec<Real>,
    pub width: usize,
    pub height: usize,
}

impl ImageField {
    pub fn new(data: Vec<Real>, width: usize, height: usize) -> Self {
        assert_eq!(
            data.len(),
            width * height,
            "data length must equal width * height"
        );
        Self {
            data,
            width,
            height,
        }
    }

    pub fn zeros(width: usize, height: usize) -> Self {
        Self {
            data: vec![0.0; width * height],
            width,
            height,
        }
    }

    /// Access pixel at (row, col).
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Real {
        self.data[row * self.width + col]
    }

    /// Mutable access to pixel at (row, col).
    #[inline]
    pub fn get_mut(&mut self, row: usize, col: usize) -> &mut Real {
        &mut self.data[row * self.width + col]
    }
}
