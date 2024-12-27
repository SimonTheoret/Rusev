use pyo3::prelude::*;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn classification_report() -> PyResult<String> {
    Ok(String::from("Ok"))
}

/// A Python module implemented in Rust.
#[pymodule]
fn py_bindings(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}
