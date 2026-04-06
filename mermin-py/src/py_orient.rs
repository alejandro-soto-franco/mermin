// mermin-py/src/py_orient.rs

use mermin_core::ImageField;
use mermin_orient::{
    fit_nuclear_ellipse, multiscale_structure_tensor, optimal_scale_map, structure_tensor,
};
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

/// Compute structure tensor at a single scale.
/// Returns dict with "theta", "coherence" as 2D numpy arrays.
#[pyfunction]
pub fn compute_structure_tensor(
    py: Python<'_>,
    image: PyReadonlyArray2<'_, f64>,
    sigma: f64,
) -> PyResult<PyObject> {
    let arr = image.as_array();
    let (h, w) = (arr.nrows(), arr.ncols());
    let data: Vec<f64> = arr.iter().copied().collect();
    let field = ImageField::new(data, w, h);

    let result = structure_tensor(&field, sigma);

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item(
        "theta",
        PyArray2::from_vec2(py, &to_2d(&result.theta.data, h, w))?,
    )?;
    dict.set_item(
        "coherence",
        PyArray2::from_vec2(py, &to_2d(&result.coherence.data, h, w))?,
    )?;
    Ok(dict.into())
}

/// Compute multiscale structure tensor.
/// Returns dict with "scales" (list of sigma values) and per-scale theta/coherence.
#[pyfunction]
pub fn compute_multiscale_structure_tensor(
    py: Python<'_>,
    image: PyReadonlyArray2<'_, f64>,
    sigmas: Vec<f64>,
) -> PyResult<PyObject> {
    let arr = image.as_array();
    let (h, w) = (arr.nrows(), arr.ncols());
    let data: Vec<f64> = arr.iter().copied().collect();
    let field = ImageField::new(data, w, h);

    let result = multiscale_structure_tensor(&field, &sigmas);
    let (opt_sigma, max_coh) = optimal_scale_map(&result);

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("sigmas", &sigmas)?;
    dict.set_item(
        "optimal_sigma",
        PyArray2::from_vec2(py, &to_2d(&opt_sigma.data, h, w))?,
    )?;
    dict.set_item(
        "max_coherence",
        PyArray2::from_vec2(py, &to_2d(&max_coh.data, h, w))?,
    )?;

    // Include theta and coherence at each scale
    let scale_results = pyo3::types::PyList::empty(py);
    for (sigma, st) in &result.scales {
        let sd = pyo3::types::PyDict::new(py);
        sd.set_item("sigma", *sigma)?;
        sd.set_item(
            "theta",
            PyArray2::from_vec2(py, &to_2d(&st.theta.data, h, w))?,
        )?;
        sd.set_item(
            "coherence",
            PyArray2::from_vec2(py, &to_2d(&st.coherence.data, h, w))?,
        )?;
        scale_results.append(sd)?;
    }
    dict.set_item("scale_results", scale_results)?;

    Ok(dict.into())
}

/// Fit nuclear ellipses from a labeled mask.
/// `nuclear_mask`: 2D array where each pixel is a cell label (0 = background).
/// Returns list of dicts with aspect_ratio, angle, centroid_row, centroid_col.
#[pyfunction]
pub fn fit_nuclear_ellipses(
    py: Python<'_>,
    nuclear_mask: PyReadonlyArray2<'_, i32>,
) -> PyResult<Vec<PyObject>> {
    let arr = nuclear_mask.as_array();
    let (h, w) = (arr.nrows(), arr.ncols());

    // Group pixels by label
    let mut label_pixels: std::collections::HashMap<i32, Vec<(usize, usize)>> =
        std::collections::HashMap::new();
    for row in 0..h {
        for col in 0..w {
            let label = arr[[row, col]];
            if label > 0 {
                label_pixels.entry(label).or_default().push((row, col));
            }
        }
    }

    let mut results = Vec::new();
    let mut labels: Vec<i32> = label_pixels.keys().copied().collect();
    labels.sort();

    for label in labels {
        let pixels = &label_pixels[&label];
        if let Some(ellipse) = fit_nuclear_ellipse(pixels) {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("label", label)?;
            dict.set_item("aspect_ratio", ellipse.aspect_ratio)?;
            dict.set_item("angle", ellipse.angle)?;
            dict.set_item("semi_major", ellipse.semi_major)?;
            dict.set_item("semi_minor", ellipse.semi_minor)?;
            dict.set_item("centroid_row", ellipse.centroid_row)?;
            dict.set_item("centroid_col", ellipse.centroid_col)?;
            results.push(dict.into());
        }
    }

    Ok(results)
}

fn to_2d(data: &[f64], h: usize, w: usize) -> Vec<Vec<f64>> {
    (0..h)
        .map(|r| data[r * w..(r + 1) * w].to_vec())
        .collect()
}
