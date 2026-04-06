// mermin-py/src/py_theory.rs

use mermin_theory::{
    build_volterra_params, estimate_ldg_params, frank_energy, to_json, FrankEnergy, LdGParams,
};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

/// Compute Frank elastic energy from a 2D director field.
///
/// `thetas`: flat row-major array of director angles, shape (ny * nx).
/// `nx`, `ny`: grid dimensions.
/// `dx`: grid spacing in physical units.
///
/// Returns dict with "splay", "bend", "ratio".
#[pyfunction(name = "frank_energy")]
pub fn frank_energy_py(
    py: Python<'_>,
    thetas: PyReadonlyArray1<'_, f64>,
    nx: usize,
    ny: usize,
    dx: f64,
) -> PyResult<PyObject> {
    let thetas_slice = thetas.as_slice()?;
    let energy = frank_energy(thetas_slice, nx, ny, dx);

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("splay", energy.splay)?;
    dict.set_item("bend", energy.bend)?;
    dict.set_item("ratio", energy.ratio)?;
    Ok(dict.into())
}

/// Estimate Landau-de Gennes parameters from the scalar order parameter distribution.
///
/// `s_values`: 1D array of per-cell scalar order parameter |psi_2|.
/// `correlation_length`: xi from G_2(r) exponential decay fit (in pixels).
/// `pixel_size`: um per pixel.
///
/// Returns dict with "a", "b", "c", "k_elastic".
#[pyfunction(name = "estimate_ldg_params")]
pub fn estimate_ldg_params_py(
    py: Python<'_>,
    s_values: PyReadonlyArray1<'_, f64>,
    correlation_length: f64,
    pixel_size: f64,
) -> PyResult<PyObject> {
    let s_slice = s_values.as_slice()?;
    let params = estimate_ldg_params(s_slice, correlation_length, pixel_size);

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("a", params.a)?;
    dict.set_item("b", params.b)?;
    dict.set_item("c", params.c)?;
    dict.set_item("k_elastic", params.k_elastic)?;
    Ok(dict.into())
}

/// Build a volterra-compatible parameter set from analysis results.
///
/// Accepts individual parameters directly:
///   `a`, `b`, `c`, `k_elastic`: LdG parameters.
///   `splay`, `bend`: Frank energy components.
///   `n_defects`: number of detected defects.
///   `image_area_um2`: total image area in um^2.
///
/// Returns dict with all volterra parameters, plus a "json" key with the serialized form.
#[pyfunction(name = "build_volterra_params")]
#[pyo3(signature = (a, b, c, k_elastic, splay, bend, n_defects, image_area_um2))]
pub fn build_volterra_params_py(
    py: Python<'_>,
    a: f64,
    b: f64,
    c: f64,
    k_elastic: f64,
    splay: f64,
    bend: f64,
    n_defects: usize,
    image_area_um2: f64,
) -> PyResult<PyObject> {
    let ldg = LdGParams {
        a,
        b,
        c,
        k_elastic,
    };
    let ratio = if bend > 1e-15 {
        splay / bend
    } else {
        f64::INFINITY
    };
    let frank = FrankEnergy {
        splay,
        bend,
        ratio,
    };

    let params = build_volterra_params(&ldg, &frank, n_defects, image_area_um2);
    let json_str = to_json(&params);

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("a_landau", params.a_landau)?;
    dict.set_item("b_landau", params.b_landau)?;
    dict.set_item("c_landau", params.c_landau)?;
    dict.set_item("k_r", params.k_r)?;
    dict.set_item("zeta_eff", params.zeta_eff)?;
    dict.set_item("frank_splay", params.frank_splay)?;
    dict.set_item("frank_bend", params.frank_bend)?;
    dict.set_item("frank_ratio", params.frank_ratio)?;
    dict.set_item("json", json_str)?;
    Ok(dict.into())
}
