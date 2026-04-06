// mermin-py/src/py_shape.rs

use mermin_core::{BoundaryContour, Point2};
use mermin_shape::{
    convexity, elongation_from_w1_tensor, fourier_spectrum, katic_shape_spectrum, minkowski_w0,
    minkowski_w1, minkowski_w1_tensor, shape_index,
};
use numpy::{PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Analyze a single cell boundary contour.
/// `contour_xy`: Nx2 numpy array of boundary points (x, y) in physical units.
///
/// Returns a dict with all shape measurements.
#[pyfunction]
pub fn analyze_shape(py: Python<'_>, contour_xy: PyReadonlyArray2<'_, f64>) -> PyResult<PyObject> {
    let arr = contour_xy.as_array();
    let n = arr.nrows();
    let points: Vec<Point2> = (0..n)
        .map(|i| Point2::new(arr[[i, 0]], arr[[i, 1]]))
        .collect();

    let contour = BoundaryContour::new(points)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let w0 = minkowski_w0(&contour);
    let w1 = minkowski_w1(&contour);
    let tensor = minkowski_w1_tensor(&contour);
    let (elong, angle) = elongation_from_w1_tensor(&tensor);
    let p0 = shape_index(&contour);
    let conv = convexity(&contour);
    let katic = katic_shape_spectrum(&contour);
    let fourier = fourier_spectrum(&contour);

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("area", w0)?;
    dict.set_item("perimeter", w1)?;
    dict.set_item("elongation", elong)?;
    dict.set_item("elongation_angle", angle)?;
    dict.set_item("shape_index", p0)?;
    dict.set_item("convexity", conv)?;
    dict.set_item("shape_katic", PyArray1::from_slice(py, &katic))?;
    dict.set_item("fourier_spectrum", PyArray1::from_slice(py, &fourier))?;

    Ok(dict.into())
}

/// Batch analyze multiple cell boundary contours.
/// `contours`: list of Nx2 numpy arrays.
///
/// Returns list of dicts.
#[pyfunction]
pub fn analyze_shapes_batch(
    py: Python<'_>,
    contours: Vec<PyReadonlyArray2<'_, f64>>,
) -> PyResult<Vec<PyObject>> {
    contours.into_iter().map(|c| analyze_shape(py, c)).collect()
}
