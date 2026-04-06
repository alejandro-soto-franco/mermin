// mermin-core/src/types.rs

/// Floating-point type used throughout mermin (matches cartan's Real = f64).
pub type Real = f64;

/// Supported k-atic symmetry orders.
/// k=1 (polar), k=2 (nematic), k=4 (tetratic), k=6 (hexatic).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KValue(u32);

impl KValue {
    pub fn new(k: u32) -> crate::Result<Self> {
        if k == 0 {
            return Err(crate::MerminError::InvalidK { k });
        }
        Ok(Self(k))
    }

    pub fn get(self) -> u32 {
        self.0
    }
}

/// Standard k values for biological tissues.
pub const K_POLAR: KValue = KValue(1);
pub const K_NEMATIC: KValue = KValue(2);
pub const K_TETRATIC: KValue = KValue(4);
pub const K_HEXATIC: KValue = KValue(6);

/// 2D point in image coordinates (row, col) or physical coordinates (x, y).
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct Point2 {
    pub x: Real,
    pub y: Real,
}

impl Point2 {
    pub fn new(x: Real, y: Real) -> Self {
        Self { x, y }
    }

    pub fn distance_to(self, other: Self) -> Real {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kvalue_valid() {
        let k = KValue::new(2).unwrap();
        assert_eq!(k.get(), 2);
    }

    #[test]
    fn kvalue_zero_rejected() {
        assert!(KValue::new(0).is_err());
    }

    #[test]
    fn point2_distance() {
        let a = Point2::new(0.0, 0.0);
        let b = Point2::new(3.0, 4.0);
        assert!((a.distance_to(b) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn standard_k_values() {
        assert_eq!(K_POLAR.get(), 1);
        assert_eq!(K_NEMATIC.get(), 2);
        assert_eq!(K_TETRATIC.get(), 4);
        assert_eq!(K_HEXATIC.get(), 6);
    }
}
