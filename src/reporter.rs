use crate::metrics::Average;
use serde::{Deserialize, Serialize};
use std::cmp::PartialOrd;
use std::collections::BTreeMap;
use std::fmt::Display;
use std::ops::{ Deref, DerefMut };

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Reporter {
    classes: BTreeMap<Class, ClassMetrics>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq, Default)]
/// This struct is a thing wrapper around a String. It modifies the PartialOrd implementation to
/// allow the `Overall_x` classes to stay first.
pub struct Class(String);

impl Deref for Class{
    type Target = String;
    fn deref(&self) -> &Self::Target {&self.0}
}
impl DerefMut for Class{
    fn deref_mut(&mut self) -> &mut Self::Target {&mut self.0}
}

impl PartialOrd for Class{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if Average::ALL_SPECIAL_ORDERED_CLASS.contains(&self.as_str()) && Average::ALL_SPECIAL_ORDERED_CLASS.contains(&other.as_str()) {
            Some(std::cmp::Ordering::Equal)
        } else if  Average::ALL_SPECIAL_ORDERED_CLASS.contains(&self.as_str()) && !Average::ALL_SPECIAL_ORDERED_CLASS.contains(&other.as_str()) {
            return Some(std::cmp::Ordering::Greater)
        } else if  !Average::ALL_SPECIAL_ORDERED_CLASS.contains(&self.as_str()) && Average::ALL_SPECIAL_ORDERED_CLASS.contains(&other.as_str()) {
            return Some(std::cmp::Ordering::Less)
        } else{
            return Some(self.0.cmp(&other.0))
        }
    }
}

impl Ord for Class{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {self.partial_cmp(other).unwrap()}
}

impl Default for Reporter {
    fn default() -> Reporter {
        Reporter {
            classes: BTreeMap::new(),
        }
    }
}

impl Reporter {
    ///TODO: Change the api to NOT allow the user to insert the same class twice
    pub fn insert(&mut self, metrics: ClassMetrics) -> Option<ClassMetrics> {
        let key: String = metrics.class.clone();
        self.classes.insert(Class(key), metrics)
    }
}
impl Deref for Reporter {
    type Target = BTreeMap<Class, ClassMetrics>;
    fn deref(&self) -> &Self::Target {
        &self.classes
    }
}

/// The Reporter struct acts as a full dataframe when displayed.
impl Display for Reporter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Class, Precision, Recall, Fscore, Support")?;
        for v in self.values() {
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

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
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
        match self.average {
            Average::None => write!(
                f,
                "{}, {}, {}, {}, {}",
                self.class, self.precision, self.recall, self.fscore, self.support
            ),
            _ => write!(
                f,
                "{}_{}, {}, {}, {}, {}",
                self.class, self.average, self.precision, self.recall, self.fscore, self.support
            ),
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_name() {}
}
