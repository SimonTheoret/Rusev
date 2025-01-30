use pyo3::prelude::*;
use rusev;

#[inline(always)]
#[pyfunction]
pub fn classification_report<'a>(
    y_true: Vec<Vec<&'a str>>,
    y_pred: Vec<Vec<&'a str>>,
    sample_weight: Option<Vec<f32>>,
    zero_division: &'a str, // DivByZeroStrat,
    scheme: &'a str,        //SchemeType,
    suffix: bool,
    parallel: bool,
) -> Result<Reporter, ComputationError<String>> {
    todo!()
}

/// A Python module implemented in Rust.
#[pymodule]
fn py_bindings(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(classification_report, m)?)?;
    Ok(())
}
