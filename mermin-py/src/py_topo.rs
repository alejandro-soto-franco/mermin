// mermin-py/src/py_topo.rs

use mermin_topo::{
    compute_persistence,
    defects::{detect_defects, Defect},
    poincare_hopf::validate_poincare_hopf,
};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

/// Detect topological defects on a 2D grid of director angles.
///
/// `thetas`: flat row-major array of director angles, shape (ny * nx).
/// `nx`, `ny`: grid dimensions.
/// `k`: k-atic symmetry order (2 for nematic, 1 for polar, 6 for hexatic).
/// `threshold`: minimum holonomy angle to classify as a defect.
///
/// Returns list of dicts with "position" ([row, col]), "charge", "angle".
#[pyfunction(name = "detect_defects")]
pub fn detect_defects_py(
    py: Python<'_>,
    thetas: PyReadonlyArray1<'_, f64>,
    nx: usize,
    ny: usize,
    k: u32,
    threshold: f64,
) -> PyResult<Vec<PyObject>> {
    let thetas_slice = thetas.as_slice()?;
    let defects = detect_defects(thetas_slice, nx, ny, k, threshold);
    defects_to_py(py, &defects)
}

/// Validate the Poincare-Hopf theorem: sum of defect charges should equal
/// the Euler characteristic.
///
/// `charges`: list of defect charges.
/// `euler_characteristic`: expected chi (1 for disk, 0 for torus, 2 for sphere).
/// `tolerance`: how close the sum must be.
///
/// Returns dict with "charge_sum", "euler_characteristic", "is_valid".
#[pyfunction(name = "validate_poincare_hopf")]
pub fn validate_poincare_hopf_py(
    py: Python<'_>,
    charges: Vec<f64>,
    euler_characteristic: i32,
    tolerance: f64,
) -> PyResult<PyObject> {
    // Build Defect structs from charges (positions and angles are irrelevant for validation)
    let defects: Vec<Defect> = charges
        .iter()
        .map(|&c| Defect {
            position: [0.0, 0.0],
            charge: c,
            angle: 0.0,
        })
        .collect();

    let (charge_sum, chi, valid) =
        validate_poincare_hopf(&defects, euler_characteristic, tolerance);

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("charge_sum", charge_sum)?;
    dict.set_item("euler_characteristic", chi)?;
    dict.set_item("is_valid", valid)?;
    Ok(dict.into())
}

/// Compute persistent homology of a simplicial complex filtration.
///
/// `vertices`: list of (vertex_index, filtration_value) tuples.
/// `edges`: list of (v0, v1, filtration_value) tuples.
/// `triangles`: list of (v0, v1, v2, filtration_value) tuples.
///
/// Returns list of dicts with "birth", "death", "dimension".
#[pyfunction(name = "compute_persistence")]
pub fn compute_persistence_py(
    py: Python<'_>,
    vertices: Vec<(usize, f64)>,
    edges: Vec<(usize, usize, f64)>,
    triangles: Vec<(usize, usize, usize, f64)>,
) -> PyResult<Vec<PyObject>> {
    let diagram = compute_persistence(&vertices, &edges, &triangles);

    let mut results = Vec::new();
    for pair in &diagram.pairs {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("birth", pair.birth)?;
        dict.set_item("death", pair.death)?;
        dict.set_item("dimension", pair.dimension)?;
        dict.set_item("persistence", pair.persistence())?;
        results.push(dict.into());
    }
    Ok(results)
}

fn defects_to_py(py: Python<'_>, defects: &[Defect]) -> PyResult<Vec<PyObject>> {
    let mut results = Vec::new();
    for d in defects {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("position", d.position.to_vec())?;
        dict.set_item("charge", d.charge)?;
        dict.set_item("angle", d.angle)?;
        results.push(dict.into());
    }
    Ok(results)
}
