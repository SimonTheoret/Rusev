use crate::metrics::Average;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::{cmp::PartialOrd, ops::Deref};

#[derive(Serialize, Deserialize)]
struct Reporter<'a> {
    classes: BTreeMap<&'a str, ClassMetric<'a>>,
}

impl<'a> Deref for Reporter<'a> {
    type Target = BTreeMap<&'a str, ClassMetric<'a>>;
    fn deref(&self) -> &Self::Target {
        &self.classes
    }
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct ClassMetric<'a> {
    class: &'a str,
    average: Average,
    precision: f32,
    recall: f32,
    fscore: f32,
    support: usize,
}
impl<'a> PartialOrd for ClassMetric<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.class.cmp(other.class))
    }
}
