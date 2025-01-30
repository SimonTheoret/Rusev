use pyo3::prelude::*;
use pyo3::types::PyList;
use rusev::{classification_report as clf_report, ClassMetrics, DivByZeroStrat, SchemeType};
use std::collections::{HashMap, HashSet};
use std::str::FromStr;

#[pyfunction]
#[pyo3(signature = (y_true, y_pred, zero_division, suffix, scheme=None, sample_weight=None))]
pub fn classification_report<'a>(
    y_true: Bound<'a, PyList>,
    y_pred: Bound<'a, PyList>,
    zero_division: String,
    suffix: bool,
    scheme: Option<String>,
    sample_weight: Option<Vec<f32>>,
) -> HashMap<String, HashMap<&'static str, f32>> {
    let y_true_ref: Vec<Vec<String>> = y_true
        .extract()
        .expect("Could not convert Python list to Vec<Vec<String>>");
    let mut y_true: Vec<Vec<&str>> = Vec::with_capacity(y_true_ref.len());
    for v_ref in y_true_ref.iter() {
        let mut v: Vec<&str> = Vec::with_capacity(v_ref.len());
        for s in v_ref.iter() {
            v.push(s.as_str())
        }
        y_true.push(v);
    }
    let y_pred_ref: Vec<Vec<String>> = y_pred
        .extract()
        .expect("Could not convert Python list to Vec<Vec<String>>");
    let mut y_pred: Vec<Vec<&str>> = Vec::with_capacity(y_pred_ref.len());
    for v_ref in y_pred_ref.iter() {
        let mut v: Vec<&str> = Vec::with_capacity(v_ref.len());
        for s in v_ref.iter() {
            v.push(s.as_str())
        }
        y_pred.push(v);
    }
    let scheme = scheme
        .map(|s| SchemeType::from_str(s.as_str()).expect("Could not parse the scheme argument"));
    let zero_div = DivByZeroStrat::from_str(zero_division.as_ref())
        .expect("Could not parse the zero_division argument");
    let reporter_dict: HashSet<ClassMetrics> = HashSet::from(
        clf_report(
            y_true,
            y_pred,
            sample_weight,
            zero_div,
            scheme,
            suffix,
            false,
        )
        .unwrap(),
    );
    let mut res = HashMap::new();
    for cm in reporter_dict.into_iter() {
        let mut row = HashMap::new();
        let key = cm.class;
        row.insert("precision", cm.precision);
        row.insert("recall", cm.recall);
        row.insert("fscore", cm.fscore);
        row.insert("support", cm.support as f32);
        res.insert(key, row);
    }
    res
}

/// A Python module implemented in Rust.
#[pymodule]
fn py_bindings(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(classification_report, m)?)
}
