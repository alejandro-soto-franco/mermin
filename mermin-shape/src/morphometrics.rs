// mermin-shape/src/morphometrics.rs

use mermin_core::{BoundaryContour, Point2, Real};

/// Shape index p_0 = P / sqrt(A).
/// For a circle p_0 = 2*sqrt(pi) ~ 3.545.
/// Higher values indicate more irregular/elongated shapes.
pub fn shape_index(contour: &BoundaryContour) -> Real {
    let a = contour.area();
    if a < 1e-15 {
        return 0.0;
    }
    contour.perimeter() / a.sqrt()
}

/// Convexity = A / A_convex_hull.
/// Returns 1.0 for convex shapes, <1.0 for concave shapes.
pub fn convexity(contour: &BoundaryContour) -> Real {
    let a = contour.area();
    let hull = convex_hull(&contour.points);
    let hull_contour = BoundaryContour::new(hull).unwrap();
    let a_hull = hull_contour.area();
    if a_hull < 1e-15 {
        return 0.0;
    }
    a / a_hull
}

/// Graham scan convex hull. Returns CCW-ordered hull vertices.
fn convex_hull(points: &[Point2]) -> Vec<Point2> {
    let mut pts: Vec<Point2> = points.to_vec();
    let n = pts.len();
    if n < 3 {
        return pts;
    }

    // Find lowest-y (then leftmost-x) point as pivot
    let mut pivot_idx = 0;
    for i in 1..n {
        if pts[i].y < pts[pivot_idx].y
            || (pts[i].y == pts[pivot_idx].y && pts[i].x < pts[pivot_idx].x)
        {
            pivot_idx = i;
        }
    }
    pts.swap(0, pivot_idx);
    let pivot = pts[0];

    // Sort by polar angle relative to pivot
    pts[1..].sort_by(|a, b| {
        let angle_a = (a.y - pivot.y).atan2(a.x - pivot.x);
        let angle_b = (b.y - pivot.y).atan2(b.x - pivot.x);
        angle_a.partial_cmp(&angle_b).unwrap()
    });

    // Graham scan
    let mut hull: Vec<Point2> = Vec::with_capacity(n);
    for p in &pts {
        while hull.len() >= 2 {
            let a = hull[hull.len() - 2];
            let b = hull[hull.len() - 1];
            let cross = (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
            if cross <= 0.0 {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(*p);
    }

    hull
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circle_shape_index() {
        let n = 200;
        let pts: Vec<Point2> = (0..n)
            .map(|i| {
                let theta = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
                Point2::new(theta.cos(), theta.sin())
            })
            .collect();
        let contour = BoundaryContour::new(pts).unwrap();
        let p0 = shape_index(&contour);
        let expected = 2.0 * std::f64::consts::PI.sqrt();
        assert!((p0 - expected).abs() < 0.05, "circle shape index ~ 3.545, got {p0}");
    }

    #[test]
    fn convex_shape_convexity_one() {
        let contour = BoundaryContour::new(vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ])
        .unwrap();
        let c = convexity(&contour);
        assert!((c - 1.0).abs() < 1e-10, "square convexity should be 1.0, got {c}");
    }
}
