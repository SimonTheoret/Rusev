use crate::metrics::Average;
use crate::metrics::OverallAverage;
use serde::{Deserialize, Serialize};
use std::cmp::PartialOrd;
use std::collections::BTreeSet;
use std::fmt::Display;
use std::ops::Deref;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Reporter {
    classes: BTreeSet<ClassMetrics>,
}

// #[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq, Default, PartialOrd, Ord)]
// /// This struct is a thing wrapper around a String. It modifies the PartialOrd implementation to
// /// allow the `Overall_x` classes to stay first.
// pub struct Class(String);
//
// impl Deref for Class {
//     type Target = String;
//     fn deref(&self) -> &Self::Target {
//         &self.0
//     }
// }
// impl DerefMut for Class {
//     fn deref_mut(&mut self) -> &mut Self::Target {
//         &mut self.0
//     }
// }

impl Default for Reporter {
    fn default() -> Reporter {
        Reporter {
            classes: BTreeSet::new(),
        }
    }
}

impl Reporter {
    ///TODO: Change the api to NOT allow the user to insert the same class twice
    pub fn insert(&mut self, metrics: ClassMetrics) -> bool {
        let ret = self.classes.insert(metrics);
        ret
    }
}
impl Deref for Reporter {
    type Target = BTreeSet<ClassMetrics>;
    fn deref(&self) -> &Self::Target {
        &self.classes
    }
}

/// The Reporter struct acts as a dataframe when displayed.
impl Display for Reporter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Class, Precision, Recall, Fscore, Support")?;
        for v in self.iter().rev(){ //Must call `.rev()` because the iter is in ascending order
            writeln!(f, "{}", v)?
        }
        Ok(())
    }
}

// NOTE: DerefMut might be a good idea if we plan to expose the
// complete BTreeMap API to the consumer of the reporter. But, it also
// implies that we need to patch many methods using insert.
// impl<'a> DerefMut for Reporter {
//     fn deref_mut(&mut self) -> &mut Self::Target {
//         &mut self.classes
//     }
// }

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ClassMetrics {
    /// The class, such as "PER", "GEO", "MISC", etc.
    pub class: String,
    /// The average used to compute this class' metrics
    pub average: Average,
    /// Precision metric
    pub precision: f32,
    /// Recall metric
    pub recall: f32,
    /// Fscore metric
    pub fscore: f32,
    /// Support metric
    pub support: usize,
}
impl PartialEq for ClassMetrics {
    fn eq(&self, other: &Self) -> bool {
        self.class == other.class && self.average == other.average
    }
}
impl Eq for ClassMetrics {}

impl PartialOrd for ClassMetrics {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match self.average.cmp(&other.average) {
            std::cmp::Ordering::Equal => self.class.partial_cmp(&other.class),
            v => Some(v),
        }
    }
}

impl Ord for ClassMetrics {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl ClassMetrics {
    pub fn new_overall(
        average: OverallAverage,
        precision: f32,
        recall: f32,
        fscore: f32,
        support: usize,
    ) -> Self {
        let class = average.to_string();
        ClassMetrics {
            class,
            average: average.into(),
            precision,
            recall,
            fscore,
            support,
        }
    }
}

// impl PartialOrd for ClassMetrics {
//     fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
//         let cmp_average = self.average.partial_cmp(&other.average);
//         Some(self.class.cmp(&other.class))
//     }
// }
// impl Ord for ClassMetrics{
//     fn cmp(&self, other: &Self) -> std::cmp::Ordering {
//         self.partial_cmp(other).unwrap()
//     }
// }
/// The Classmetrics struct acts as a line in a dataframe when displayed.
impl Display for ClassMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}, {}, {}, {}, {}",
            self.class, self.precision, self.recall, self.fscore, self.support
        )
    }
}
