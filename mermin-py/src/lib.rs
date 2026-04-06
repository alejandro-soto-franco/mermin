//! Python bindings for mermin.

use pyo3::prelude::*;

mod py_orient;
mod py_shape;
mod py_stats;
mod py_theory;
mod py_topo;

/// mermin native extension module.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Shape analysis
    m.add_function(wrap_pyfunction!(py_shape::analyze_shape, m)?)?;
    m.add_function(wrap_pyfunction!(py_shape::analyze_shapes_batch, m)?)?;

    // Orientation analysis
    m.add_function(wrap_pyfunction!(py_orient::compute_structure_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(
        py_orient::compute_multiscale_structure_tensor,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(py_orient::fit_nuclear_ellipses, m)?)?;

    // Topology
    m.add_function(wrap_pyfunction!(py_topo::detect_defects_py, m)?)?;
    m.add_function(wrap_pyfunction!(py_topo::validate_poincare_hopf_py, m)?)?;
    m.add_function(wrap_pyfunction!(py_topo::compute_persistence_py, m)?)?;

    // Statistics
    m.add_function(wrap_pyfunction!(py_stats::orientational_correlation_py, m)?)?;
    m.add_function(wrap_pyfunction!(py_stats::ripley_k_py, m)?)?;
    m.add_function(wrap_pyfunction!(py_stats::permutation_test_py, m)?)?;

    // Theory
    m.add_function(wrap_pyfunction!(py_theory::frank_energy_py, m)?)?;
    m.add_function(wrap_pyfunction!(py_theory::estimate_ldg_params_py, m)?)?;
    m.add_function(wrap_pyfunction!(py_theory::build_volterra_params_py, m)?)?;

    Ok(())
}
