/**
This modules gives a few tools to prettyprint the output for all the classes and the overall
metrics.
*/
use serde::{Deserialize, Serialize};
use std::cmp::PartialOrd;
use std::collections::{BTreeSet, HashSet};
use std::fmt::Display;
use std::hash::Hash;
use std::str::FromStr;

/// The reporter holds the metrics of a given class and the overall metrics. It can be used to
/// display the results (i.e. prettyprint them) as if they were collected into a dataframe and can
/// be consumed to obtain a `BTreeSet` containing the metrics. The reporter can be built with the
/// `classification_report` function.
//
/// # Example
///
/// ```rust
/// use rusev::{ classification_report, DivByZeroStrat, SchemeType };
///
///
/// let y_true = vec![vec!["B-TEST", "B-NOTEST", "O", "B-TEST"]];
/// let y_pred = vec![vec!["O", "B-NOTEST", "B-OTHER", "B-TEST"]];
///
///
/// let reporter = classification_report(y_true,
///                                      y_pred,
///                                      None,
///                                      DivByZeroStrat::ReplaceBy0,
///                                      Some(SchemeType::IOB2),
///                                      false,
///                                      false).unwrap();
///
/// let expected_report =
/// "Class, Precision, Recall, Fscore, Support
/// Overall_Weighted, 1, 0.6666667, 0.77777785, 3
/// Overall_Micro, 0.6666667, 0.6666667, 0.6666667, 3
/// Overall_Macro, 0.6666667, 0.5, 0.5555556, 3
/// NOTEST, 1, 1, 1, 1
/// OTHER, 0, 0, 0, 0
/// TEST, 1, 0.5, 0.6666667, 2\n";
///
///
/// assert_eq!(expected_report, reporter.to_string());
/// ```
///
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct Reporter {
    pub(crate) classes: BTreeSet<ClassMetricsInner>,
}

/// By converting the reporter into a `BTreeSet` of `ClassMetrics`, you lose the ordering and the
/// partial equality implemented for the reporter. If you mean to consume the data without
/// prettypriting it, this is not a problem.
impl From<Reporter> for HashSet<ClassMetrics> {
    fn from(value: Reporter) -> Self {
        value.classes.into_iter().map(ClassMetrics::from).collect()
    }
}

impl Reporter {
    pub(crate) fn insert(&mut self, metrics: ClassMetricsInner) -> bool {
        self.classes.insert(metrics)
    }
}
// impl Deref for Reporter {
//     type Target = BTreeSet<ClassMetricsInner>;
//     fn deref(&self) -> &Self::Target {
//         &self.classes
//     }
// }
//
// impl DerefMut for Reporter {
//     fn deref_mut(&mut self) -> &mut Self::Target {
//         &mut self.classes
//     }
// }

/// The Reporter struct acts as a dataframe when displayed.
impl Display for Reporter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Class, Precision, Recall, Fscore, Support")?;
        for v in self.classes.iter().rev() {
            //Must call `.rev()` because the iter is in ascending order
            writeln!(f, "{}", v)?
        }
        Ok(())
    }
}

#[derive(Debug)]
/// Datastructure holding metrics about a given class.
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

impl Hash for ClassMetrics {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.class.hash(state);
        self.average.hash(state)
    }
}

impl PartialEq for ClassMetrics {
    fn eq(&self, other: &Self) -> bool {
        self.class == other.class && self.average == other.average
    }
}
impl Eq for ClassMetrics {}

impl From<ClassMetricsInner> for ClassMetrics {
    fn from(value: ClassMetricsInner) -> Self {
        Self {
            class: value.class,
            average: value.average,
            precision: value.precision,
            recall: value.recall,
            fscore: value.fscore,
            support: value.support,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
/// ClassMetrics hold the metrics for a single class. They can't be constructed explicitely and
/// they implement a special version of the `Display` trait, allowing them to be treated as the
/// line of a dataframe. They implement a different algorithm for PartialOrd/Ord and they
pub(crate) struct ClassMetricsInner {
    /// The class, such as "PER", "GEO", "MISC", etc.
    pub(crate) class: String,
    /// The average used to compute this class' metrics
    pub(crate) average: Average,
    /// Precision metric
    pub(crate) precision: f32,
    /// Recall metric
    pub(crate) recall: f32,
    /// Fscore metric
    pub(crate) fscore: f32,
    /// Support metric
    pub(crate) support: usize,
}
impl PartialEq for ClassMetricsInner {
    fn eq(&self, other: &Self) -> bool {
        self.class == other.class && self.average == other.average
    }
}
impl Eq for ClassMetricsInner {}

#[allow(clippy::non_canonical_partial_ord_impl)]
impl PartialOrd for ClassMetricsInner {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match self.average.cmp(&other.average) {
            std::cmp::Ordering::Equal => self.class.partial_cmp(&other.class),
            v => Some(v),
        }
    }
}

impl Ord for ClassMetricsInner {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl ClassMetricsInner {
    pub(crate) fn new_overall(
        average: OverallAverage,
        precision: f32,
        recall: f32,
        fscore: f32,
        support: usize,
    ) -> Self {
        let class = average.to_string();
        ClassMetricsInner {
            class,
            average: average.into(),
            precision,
            recall,
            fscore,
            support,
        }
    }
}

/// The Classmetrics struct acts as a line in a dataframe when displayed.
impl Display for ClassMetricsInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}, {}, {}, {}, {}",
            self.class, self.precision, self.recall, self.fscore, self.support
        )
    }
}

/// Enumeration of the different types of averaging possible and supported by this crate. &str can
/// be parsed to create an `Average`.
#[derive(Debug, Hash, PartialEq, Eq, Copy, Clone, Serialize, Deserialize)]
pub enum Average {
    None,
    Micro,
    Macro,
    Weighted,
    Samples,
}
impl Display for Average {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}
impl FromStr for Average {
    type Err = AverageParsingError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "none" => Ok(Average::None),
            "micro" => Ok(Average::Micro),
            "macro" => Ok(Average::Macro),
            "weighted" => Ok(Average::Weighted),
            "samples" => Ok(Average::Samples),
            _ => Err(AverageParsingError(String::from(s))),
        }
    }
}

#[derive(Debug, PartialEq, PartialOrd, Eq, Ord, Clone)]
pub struct AverageParsingError(String);
impl Display for AverageParsingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Impossible to parse the string ({}) into an Average",
            self.0
        )
    }
}

/// Average implements partial ordering. This is used during the
/// reporting to represent the ClassMetrics with an `average` other
/// than `None` as `Greater` than those with `None`.
#[allow(clippy::non_canonical_partial_ord_impl)]
impl PartialOrd for Average {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (Self::None, _) => Some(std::cmp::Ordering::Less),
            (_, Self::None) => Some(std::cmp::Ordering::Greater),
            _ => Some(std::cmp::Ordering::Equal),
        }
    }
}
impl Ord for Average {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Copy, Clone, Serialize, Deserialize)]
pub enum OverallAverage {
    Micro,
    Macro,
    Weighted,
}

impl Display for OverallAverage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let str_content = match self {
            Self::Micro => "Overall_Micro",
            Self::Macro => "Overall_Macro",
            Self::Weighted => "Overall_Weighted",
        };
        write!(f, "{}", str_content)
    }
}

impl From<OverallAverage> for Average {
    fn from(value: OverallAverage) -> Self {
        match value {
            OverallAverage::Micro => Average::Micro,
            OverallAverage::Macro => Average::Macro,
            OverallAverage::Weighted => Average::Weighted,
        }
    }
}
