// mermin-core/src/contour.rs

use crate::{MerminError, Point2, Real, Result};

/// Ordered boundary contour of a single cell, in physical coordinates (um).
/// Points are ordered counter-clockwise. The contour is implicitly closed
/// (last point connects back to first).
#[derive(Debug, Clone)]
pub struct BoundaryContour {
    /// Ordered vertices of the boundary polygon.
    pub points: Vec<Point2>,
}

impl BoundaryContour {
    /// Create a contour from ordered points. Requires at least 3 points.
    pub fn new(points: Vec<Point2>) -> Result<Self> {
        if points.len() < 3 {
            return Err(MerminError::ContourTooShort {
                n: points.len(),
                min: 3,
            });
        }
        Ok(Self { points })
    }

    pub fn n_points(&self) -> usize {
        self.points.len()
    }

    /// Centroid of the polygon (average of vertices).
    pub fn centroid(&self) -> Point2 {
        let n = self.points.len() as Real;
        let (sx, sy) = self.points.iter().fold((0.0, 0.0), |(sx, sy), p| {
            (sx + p.x, sy + p.y)
        });
        Point2::new(sx / n, sy / n)
    }

    /// Signed area via the shoelace formula. Positive if CCW.
    pub fn signed_area(&self) -> Real {
        let pts = &self.points;
        let n = pts.len();
        let mut area = 0.0;
        for i in 0..n {
            let j = (i + 1) % n;
            area += pts[i].x * pts[j].y - pts[j].x * pts[i].y;
        }
        area * 0.5
    }

    /// Absolute area of the polygon.
    pub fn area(&self) -> Real {
        self.signed_area().abs()
    }

    /// Perimeter of the polygon.
    pub fn perimeter(&self) -> Real {
        let pts = &self.points;
        let n = pts.len();
        (0..n)
            .map(|i| pts[i].distance_to(pts[(i + 1) % n]))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square_contour() -> BoundaryContour {
        // Unit square CCW: (0,0) -> (1,0) -> (1,1) -> (0,1)
        BoundaryContour::new(vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ])
        .unwrap()
    }

    #[test]
    fn contour_too_short() {
        let r = BoundaryContour::new(vec![Point2::new(0.0, 0.0), Point2::new(1.0, 0.0)]);
        assert!(r.is_err());
    }

    #[test]
    fn square_area() {
        let c = square_contour();
        assert!((c.area() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn square_perimeter() {
        let c = square_contour();
        assert!((c.perimeter() - 4.0).abs() < 1e-12);
    }

    #[test]
    fn square_centroid() {
        let c = square_contour();
        let cen = c.centroid();
        assert!((cen.x - 0.5).abs() < 1e-12);
        assert!((cen.y - 0.5).abs() < 1e-12);
    }
}
