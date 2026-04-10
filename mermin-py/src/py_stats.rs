// mermin-py/src/py_stats.rs

use mermin_core::{Point2, Real};
use mermin_stats::{
    nematic_order_parameter, orientational_correlation, permutation_test, ripley_k,
};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Compute the orientational correlation function G_k(r).
///
/// `centroids`: Nx2 numpy array of cell centroid positions.
/// `thetas`: 1D array of per-cell director angles.
/// `k`: k-atic symmetry order.
/// `max_r`: maximum distance to compute.
/// `n_bins`: number of distance bins.
///
/// Returns dict with "r_bins", "g_values", "counts", "correlation_length".
#[pyfunction(name = "orientational_correlation")]
pub fn orientational_correlation_py(
    py: Python<'_>,
    centroids: PyReadonlyArray2<'_, f64>,
    thetas: PyReadonlyArray1<'_, f64>,
    k: u32,
    max_r: f64,
    n_bins: usize,
) -> PyResult<PyObject> {
    let c_arr = centroids.as_array();
    let n = c_arr.nrows();
    let points: Vec<Point2> = (0..n)
        .map(|i| Point2::new(c_arr[[i, 0]], c_arr[[i, 1]]))
        .collect();
    let theta_slice = thetas.as_slice()?;

    let result = orientational_correlation(&points, theta_slice, k, max_r, n_bins);

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("r_bins", PyArray1::from_slice(py, &result.r_bins))?;
    dict.set_item("g_values", PyArray1::from_slice(py, &result.g_values))?;
    let counts_f64: Vec<f64> = result.counts.iter().map(|&c| c as f64).collect();
    dict.set_item("counts", PyArray1::from_slice(py, &counts_f64))?;
    dict.set_item("correlation_length", result.correlation_length)?;
    Ok(dict.into())
}

/// Compute Ripley's K-function for a 2D point pattern.
///
/// `points`: Nx2 numpy array of point positions.
/// `bbox`: [x_min, y_min, x_max, y_max] bounding box.
/// `r_values`: 1D array of distances at which to evaluate K.
///
/// Returns dict with "r_values", "k_values", "l_values".
#[pyfunction(name = "ripley_k")]
pub fn ripley_k_py(
    py: Python<'_>,
    points: PyReadonlyArray2<'_, f64>,
    bbox: [f64; 4],
    r_values: PyReadonlyArray1<'_, f64>,
) -> PyResult<PyObject> {
    let p_arr = points.as_array();
    let n = p_arr.nrows();
    let pts: Vec<Point2> = (0..n)
        .map(|i| Point2::new(p_arr[[i, 0]], p_arr[[i, 1]]))
        .collect();
    let r_slice = r_values.as_slice()?;

    let result = ripley_k(&pts, bbox, r_slice);

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("r_values", PyArray1::from_slice(py, &result.r_values))?;
    dict.set_item("k_values", PyArray1::from_slice(py, &result.k_values))?;
    dict.set_item("l_values", PyArray1::from_slice(py, &result.l_values))?;
    Ok(dict.into())
}

/// Compute the global nematic order parameter S = <cos 2(theta - theta_mean)>.
///
/// `cell_thetas`: 1D array of per-cell nematic orientations in [0, pi).
///
/// Returns dict with "s" (order parameter in [0, 1]) and "mean_angle".
#[pyfunction(name = "nematic_order")]
pub fn nematic_order_py(
    py: Python<'_>,
    cell_thetas: PyReadonlyArray1<'_, f64>,
) -> PyResult<PyObject> {
    let thetas = cell_thetas.as_slice()?;
    let result = nematic_order_parameter(thetas)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("need at least 2 cells"))?;
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("s", result.s)?;
    dict.set_item("mean_angle", result.mean_angle)?;
    Ok(dict.into())
}

/// Two-sample permutation test using difference of means.
///
/// `values_a`: 1D array of measurements from condition A.
/// `values_b`: 1D array of measurements from condition B.
/// `n_permutations`: number of random permutations.
/// `seed`: random seed for reproducibility.
///
/// Returns dict with "observed", "p_value", "n_permutations".
#[pyfunction(name = "permutation_test")]
pub fn permutation_test_py(
    py: Python<'_>,
    values_a: PyReadonlyArray1<'_, f64>,
    values_b: PyReadonlyArray1<'_, f64>,
    n_permutations: usize,
    seed: u64,
) -> PyResult<PyObject> {
    let a_slice = values_a.as_slice()?;
    let b_slice = values_b.as_slice()?;

    // Default statistic: difference of means
    let diff_of_means = |a: &[Real], b: &[Real]| -> Real {
        let ma: Real = a.iter().sum::<Real>() / a.len() as Real;
        let mb: Real = b.iter().sum::<Real>() / b.len() as Real;
        ma - mb
    };

    let result = permutation_test(a_slice, b_slice, diff_of_means, n_permutations, seed);

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("observed", result.observed)?;
    dict.set_item("p_value", result.p_value)?;
    dict.set_item("n_permutations", result.n_permutations)?;
    Ok(dict.into())
}
