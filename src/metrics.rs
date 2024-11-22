use crate::reporter::{ClassMetrics, Reporter};
use crate::schemes::TryFromVec;
use crate::{ConversionError, Entities, SchemeType};
use core::fmt;
use itertools::multizip;
use ndarray::Data;
use ndarray::{prelude::*, Array, ScalarOperand};
use ndarray_stats::{errors::MultiInputError, SummaryStatisticsExt};
use num::{Float, Num, NumCast};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt::{Debug, Display};
use std::str::FromStr;
use std::sync::OnceLock;

const WARN_FOR: [Metric; 3] = [Metric::Precision, Metric::Recall, Metric::FScore];

#[derive(Debug, Clone)]
pub struct ArrayNotUniqueOrEmpty(usize);

impl Display for ArrayNotUniqueOrEmpty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "This array contains more than one element or is empty. It has length: {} Cannot call `item` on it", self.0
        )
    }
}
impl Error for ArrayNotUniqueOrEmpty {}

trait ItemArrayExt<Output> {
    /// Returns the element out of the Array. Can return an error if the array is empty of if the
    /// array has a length superior to 1.
    fn item(&self) -> Result<Output, ArrayNotUniqueOrEmpty> {
        match self.length() {
            1 => Ok(self.get_first()),
            n => Err(ArrayNotUniqueOrEmpty(n)),
        }
    }
    /// Returns the length of the array;
    fn length(&self) -> usize;
    /// Gets the first element of the array
    fn get_first(&self) -> Output;
}

impl<F: Clone, T: Data<Elem = F>> ItemArrayExt<F> for ArrayBase<T, Dim<[usize; 1]>> {
    fn length(&self) -> usize {
        self.len()
    }
    fn get_first(&self) -> F {
        self.first().unwrap().clone()
    }

    fn item(&self) -> Result<F, ArrayNotUniqueOrEmpty> {
        match self.length() {
            1 => Ok(self.get_first()),
            n => Err(ArrayNotUniqueOrEmpty(n)),
        }
    }
}
#[derive(Debug, PartialEq, Hash, Clone, Copy)]
enum Metric {
    FScore,
    Precision,
    Recall,
}
impl Display for Metric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum DivisionByZeroStrategy {
    /// Replace denominator equal to `0` by `1` for the calculations
    ReplaceBy1,
    /// Returns an error
    ReturnError,
}
impl Default for DivisionByZeroStrategy {
    fn default() -> Self {
        Self::ReplaceBy1
    }
}

#[derive(Debug)]
pub struct ParsingDivisionByZeroStrategyError<S: Debug + Display>(S);

impl<S: Debug + Display> Display for ParsingDivisionByZeroStrategyError<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Could not parse the {} into a a `DivisionByZeroStrategy`",
            self.0
        )
    }
}
impl<S: Debug + Display> Error for ParsingDivisionByZeroStrategyError<S> {}

impl FromStr for DivisionByZeroStrategy {
    type Err = ParsingDivisionByZeroStrategyError<String>;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_ref() {
            "replaceby1" | "replacebyone" => Ok(DivisionByZeroStrategy::ReplaceBy1),
            "returnerror" | "error" => Ok(DivisionByZeroStrategy::ReturnError),
            _ => Err(ParsingDivisionByZeroStrategyError(String::from(s))),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DivisionByZeroError;

impl Display for DivisionByZeroError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Encountered division by zero")
    }
}

impl Error for DivisionByZeroError {}

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

/// Average implements partial ordering. This is used during the
/// reporting to represent the ClassMetrics with an `average` other
/// than `None` as `Greater` than those with `None`.
impl PartialOrd for Average {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (Self::None, _) => Some(std::cmp::Ordering::Less),
            (_, Self::None) => Some(std::cmp::Ordering::Greater),
            _ => Some(std::cmp::Ordering::Equal),
        }
    }
}

impl Average {
    // pub(crate) const ALL_AVERAGES_STRINGS: [&'static str; 5] =
    //     ["None", "Micro", "Macro", "Weighted", "Samples"];
    pub(crate) const OVERALL_PREFIX: &'static str = "Overall";
    pub(crate) const ALL_SPECIAL_ORDERED_CLASS: [&'static str; 4] = [
        "Overall_Micro",
        "Overall_Macro",
        "Overall_Weighted",
        "Overall_Samples",
    ];
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

/// Internal extension trait for Num's Float trait
pub trait FloatExt: Float + Send + Sync + Clone + ScalarOperand + Debug {}

impl<T: Float + Send + Sync + Clone + Copy + ScalarOperand + Debug> FloatExt for T {}

// /// Internal extension trait for Num's Integer trait
// pub trait IntExt: Integer + Send + Sync + Clone + Copy + ScalarOperand + Debug {}
//
// impl<T: Integer + Send + Sync + Clone + Copy + ScalarOperand + Debug> IntExt for T {}

fn prf_divide<I: Num + Clone + Send + Sync + Copy, D: Dimension>(
    numerator: ArcArray<I, D>,
    denominator: ArrayViewMut<I, D>,
    parallel: bool,
    metric: Metric,
    zero_division: DivisionByZeroStrategy,
) -> Result<ArcArray<I, D>, DivisionByZeroError> {
    let (mut result, found_0_in_denom) = if parallel {
        par_prf_divide_results_and_mask(numerator, denominator)
    } else {
        prf_divide_results_and_mask(numerator, denominator)
    };
    if found_0_in_denom {
        match zero_division {
            DivisionByZeroStrategy::ReturnError => Err(DivisionByZeroError),
            DivisionByZeroStrategy::ReplaceBy1 => {
                if parallel {
                    result = par_replace(result, I::zero(), I::one());
                } else {
                    result = replace(result, I::zero(), I::one());
                }
                if WARN_FOR.contains(&metric) {
                    eprintln!(
                        "Warning: Encountered a division by zero while computing {:?}",
                        metric
                    );
                }
                Ok(result)
            }
        }
    } else {
        Ok(result)
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
/// Error type to represent when two lists or arrays are not of the
/// same length (when they should be).
pub struct InconsistentLengthError(usize, usize);

impl Display for InconsistentLengthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Inconsistent length between two lists. `y_true` is length {}, `y_pred` is length {}",
            self.0, self.1
        )
    }
}
impl Error for InconsistentLengthError {}

fn check_consistent_length<T>(
    y_true: &[Vec<T>],
    y_pred: &[Vec<T>],
) -> Result<(), InconsistentLengthError> {
    let y_true_lengths: Vec<_> = y_true.iter().map(|v| v.len()).collect();
    let y_pred_lengths: Vec<_> = y_pred.iter().map(|v| v.len()).collect();
    let y_true_len = y_true_lengths.len();
    let y_pred_len = y_pred_lengths.len();

    if y_true_len != y_pred_len {
        return Err(InconsistentLengthError(y_true_len, y_pred_len));
    }
    let iter = y_true_lengths.into_iter().zip(y_pred_lengths);
    for (t_l, p_l) in iter {
        if t_l != p_l {
            return Err(InconsistentLengthError(t_l, p_l));
        }
    }
    Ok(())
}

/// predicted sum, true positive sum and true sum
type ActualTPCorrect<T> = (Array1<T>, Array1<T>, Array1<T>);

fn extract_tp_actual_correct<'a>(
    y_true: Vec<Vec<&'a str>>,
    y_pred: Vec<Vec<&'a str>>,
    scheme: SchemeType,
    suffix: bool,
    delimiter: char,
    entities_true: Option<&Entities<'a>>,
    entities_pred: Option<&Entities<'a>>,
) -> Result<ActualTPCorrect<usize>, ComputationError<&'a str>> {
    let entities_true_res = match entities_true {
        Some(e) => e,
        None => &Entities::try_from_vecs(y_true, scheme, suffix, delimiter, None)?,
    };
    let entities_pred_res = match entities_pred {
        Some(e) => e,
        None => &Entities::try_from_vecs(y_pred, scheme, suffix, delimiter, None)?,
    };
    let entities_pred_unique_tags = entities_pred_res.unique_tags();
    let entities_true_unique_tags = entities_true_res.unique_tags();

    let target_names =
        BTreeSet::from_iter(entities_pred_unique_tags.union(&entities_true_unique_tags));

    //TODO: This should be a for loop to avoid cloning target_names all over the place
    let pred_sum: Array1<usize> = Array::from_iter(
        target_names
            .clone()
            .into_iter()
            .map(|t| entities_pred_res.filter(*t).len()),
    );
    let tp_sum: Array1<usize> = Array::from_iter(target_names.clone().into_iter().map(|t| {
        entities_true_res
            .filter(*t)
            .intersection(&entities_pred_res.filter(*t))
            .count()
    }));
    let test = target_names
        .into_iter()
        .map(|t| entities_true_res.filter(*t).len());
    let true_sum: Array1<usize> = Array::from_iter(test);

    Ok((pred_sum, tp_sum, true_sum))
}

#[derive(Debug, Clone)]
/// Enum error encompassing many type of failures that could happen when computing the precison,
/// recall, f-score and the support.
pub enum ComputationError<S: AsRef<str> + std::fmt::Debug> {
    BetaNotPositive,
    InconsistentLenght(InconsistentLengthError),
    ConversionError(ConversionError<S>),
    DivisionByZero(DivisionByZeroError),
    NoSampleWeight,
    InputError(MultiInputError),
    EmptyArray(String),
    EmptyOrNotUnique(ArrayNotUniqueOrEmpty),
}
impl<S: AsRef<str> + std::fmt::Debug> Display for ComputationError<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BetaNotPositive => write!(f, "Beta value is not positive"),
            Self::InconsistentLenght(length_err) => std::fmt::Display::fmt(length_err, f),
            Self::ConversionError(conv_err) => std::fmt::Display::fmt(&conv_err, f),
            Self::DivisionByZero(div_err) => std::fmt::Display::fmt(&div_err, f),
            Self::NoSampleWeight => write!(f, "Using sample weighting and no sample weight given"),
            Self::InputError(input_err) => std::fmt::Display::fmt(&input_err, f),
            Self::EmptyArray(empty_err) => write!(f, "Found an empty array in {}", empty_err),
            Self::EmptyOrNotUnique(size_err) => std::fmt::Display::fmt(size_err, f),
        }
    }
}
impl<S: AsRef<str> + std::fmt::Debug> Error for ComputationError<S> {}

impl<S: AsRef<str> + std::fmt::Debug> From<InconsistentLengthError> for ComputationError<S> {
    fn from(value: InconsistentLengthError) -> Self {
        Self::InconsistentLenght(value)
    }
}
impl<S: AsRef<str> + std::fmt::Debug> From<ConversionError<S>> for ComputationError<S> {
    fn from(value: ConversionError<S>) -> Self {
        Self::ConversionError(value)
    }
}
impl<S: AsRef<str> + std::fmt::Debug> From<DivisionByZeroError> for ComputationError<S> {
    fn from(value: DivisionByZeroError) -> Self {
        Self::DivisionByZero(value)
    }
}

impl<S: AsRef<str> + std::fmt::Debug> From<MultiInputError> for ComputationError<S> {
    fn from(value: MultiInputError) -> Self {
        Self::InputError(value)
    }
}

impl<S: AsRef<str> + std::fmt::Debug> From<ArrayNotUniqueOrEmpty> for ComputationError<S> {
    fn from(value: ArrayNotUniqueOrEmpty) -> Self {
        Self::EmptyOrNotUnique(value)
    }
}

type PrecisionRecallFScoreTrueSum = (
    Array<f32, Dim<[usize; 1]>>,
    Array<f32, Dim<[usize; 1]>>,
    Array<f32, Dim<[usize; 1]>>,
    Array<usize, Dim<[usize; 1]>>,
);

/// Main entrypoint of the Rusev library. This function computes the precision, recall, fscore and
/// support of the true and predicted tokens.
///
/// * `y_true`: True tokens
/// * `y_pred`: Predicted tokens
/// * `beta`: Value of the `beta` parameter of the fscore. `beta=1` for F1 and `beta=0.5` for F0.5.
/// * `average`: What type of average to use.
/// * `sample_weight`: Optional weights of the samples.
/// * `zero_division`: What to do in case of division by zero.
/// * `scheme`: What scheme are we using?
/// * `suffix`: What char to use as suffix?
/// * `delimiter`: What delimiter are we using to differentiate the prefix from
///   the rest of the tag.
/// * `parallel`: Can we use parallelism for computations?
/// * `entities_true`: Optional entities used to reduce the computation load.
/// * `entities_pred`: Optional entities used to reduce the computation load.
pub fn precision_recall_fscore_support<'a, F: FloatExt>(
    y_true: Vec<Vec<&'a str>>,
    y_pred: Vec<Vec<&'a str>>,
    beta: F,
    average: Average,
    sample_weight: Option<ArcArray<f32, Dim<[usize; 1]>>>,
    zero_division: DivisionByZeroStrategy,
    scheme: SchemeType,
    suffix: bool,
    delimiter: char,
    parallel: bool,
    entities_true: Option<&Entities<'a>>,
    entities_pred: Option<&Entities<'a>>,
) -> Result<PrecisionRecallFScoreTrueSum, ComputationError<&'a str>> {
    if beta.is_sign_negative() {
        return Err(ComputationError::BetaNotPositive);
    };
    check_consistent_length(&y_true, &y_pred)?;
    let (mut pred_sum, mut tp_sum, mut true_sum) = extract_tp_actual_correct(
        y_true,
        y_pred,
        scheme,
        suffix,
        delimiter,
        entities_true,
        entities_pred,
    )?;
    let beta2 = beta.powi(2);
    if matches!(average, Average::Micro) {
        tp_sum = array![tp_sum.sum()];
        pred_sum = array![pred_sum.sum()];
        true_sum = array![true_sum.sum()];
    };
    let arc_tp_sum = tp_sum.mapv(|x| x as f32).to_shared();

    let precision = prf_divide(
        arc_tp_sum.clone(), // ArcArray are (often) inexpensive to clone. They are in fact `Copy`
        pred_sum.mapv(|x| x as f32).view_mut(),
        true,
        Metric::Precision,
        zero_division,
    )?;
    let recall = prf_divide(
        arc_tp_sum,
        true_sum.mapv(|x| x as f32).view_mut(),
        parallel,
        Metric::Recall,
        zero_division,
    )?;
    let f_score: ArcArray<f32, Dim<[usize; 1]>> = if beta2.is_infinite() && beta2.is_sign_positive()
    {
        recall.clone()
    } else {
        let denom = precision.clone() + recall.view();
        let denom_non_zero = if parallel {
            par_replace(denom, 0.0, 1.0)
        } else {
            replace(denom, 0.0, 1.0)
        };
        let beta2p1 = beta2 + F::one();
        let beta2p1_cast: f32 = <f64 as NumCast>::from(beta2p1)
            .expect("Casting from f64 to f32 should always be possible")
            as f32;
        beta2p1_cast * precision.clone() * recall.view() / denom_non_zero
    };
    match average {
        Average::Weighted => {
            let tmp_weights = true_sum;
            if tmp_weights.sum() == 0 {
                match zero_division {
                    DivisionByZeroStrategy::ReturnError => {
                        return Err(ComputationError::DivisionByZero(DivisionByZeroError))
                    }
                    DivisionByZeroStrategy::ReplaceBy1 => {
                        return Ok((
                            precision.to_owned(),
                            recall.to_owned(),
                            f_score.to_owned(),
                            array![0],
                        ))
                    }
                }
            };
            let final_tmp_weights = tmp_weights.mapv(|x| x as f32).into_shared();
            let final_precision =
                Array::from_vec(vec![precision.weighted_mean(&final_tmp_weights)?]);
            let final_recall = Array::from_vec(vec![recall.weighted_mean(&final_tmp_weights)?]);
            let final_f_score = Array::from_vec(vec![f_score.weighted_mean(&final_tmp_weights)?]);
            // let final_true_sum = Array::from_vec(vec![true_sum.sum()]);
            let final_true_sum = array![tmp_weights.sum()];
            Ok((final_precision, final_recall, final_f_score, final_true_sum))
        }
        Average::Samples => {
            let final_tmp_weights = sample_weight
                .ok_or(ComputationError::NoSampleWeight)?
                .into_shared();
            let final_precision =
                Array::from_vec(vec![precision.weighted_mean(&final_tmp_weights)?]);
            let final_recall = Array::from_vec(vec![recall.weighted_mean(&final_tmp_weights)?]);
            let final_f_score = Array::from_vec(vec![f_score.weighted_mean(&final_tmp_weights)?]);
            let final_true_sum = array![true_sum.sum()];
            Ok((final_precision, final_recall, final_f_score, final_true_sum))
        }
        Average::None => {
            let final_precision = precision.into_owned();
            let final_recall = recall.into_owned();
            let final_f_score = f_score.into_owned();
            Ok((final_precision, final_recall, final_f_score, true_sum))
        }
        _ => {
            let final_precision = Array::from_vec(vec![precision.mean().ok_or_else(|| {
                ComputationError::EmptyArray(String::from("precision array was empty"))
            })?]);
            let final_recall = Array::from_vec(vec![recall.mean().ok_or_else(|| {
                ComputationError::EmptyArray(String::from("precision array was empty"))
            })?]);
            let final_f_score = Array::from_vec(vec![f_score.mean().ok_or_else(|| {
                ComputationError::EmptyArray(String::from("precision array was empty"))
            })?]);
            let final_true_sum = array![true_sum.sum()];
            Ok((final_precision, final_recall, final_f_score, final_true_sum))
        }
    }
}
// # Average the results
// if average == "weighted":
//     weights = true_sum
//     if weights.sum() == 0:
//         zero_division_value = 0.0 if zero_division in ["warn", 0] else 1.0
//         # precision is zero_division if there are no positive predictions
//         # recall is zero_division if there are no positive labels
//         # fscore is zero_division if all labels AND predictions are
//         # negative
//         return (
//             zero_division_value if pred_sum.sum() == 0 else 0.0,
//             zero_division_value,
//             zero_division_value if pred_sum.sum() == 0 else 0.0,
//             sum(true_sum),
//         )

// elif average == "samples":
//     weights = sample_weight
// else:
//     weights = None

// if average is not None:
//     precision = np.average(precision, weights=weights)
//     recall = np.average(recall, weights=weights)
//     f_score = np.average(f_score, weights=weights)
//     true_sum = sum(true_sum)

// return precision, recall, f_score, true_sum

type Found0InDenominator = bool;

/// This function computes the result in parallel. For a synchronous
/// version of this function, see `prf_divide_results`. The second
/// return argument is `true` if it foufnd a zero in the
/// denominator. Else, it is `false`.
///
/// * `numerator`: Numerator of the division
/// * `denominator`: denominator of the division
fn par_prf_divide_results_and_mask<I: Num + Clone + Send + Sync, D: Dimension>(
    numerator: ArcArray<I, D>,
    mut denominator: ArrayViewMut<I, D>,
) -> (ArcArray<I, D>, Found0InDenominator) {
    let found_zero_in_denom_cell = OnceLock::new();
    denominator.par_mapv_inplace(|v| {
        if v == I::zero() {
            found_zero_in_denom_cell.get_or_init(|| false);
            I::one()
        } else {
            v
        }
    });
    let found_zero_in_denom = found_zero_in_denom_cell.into_inner().unwrap_or(false);
    (numerator / denominator, found_zero_in_denom)
}

/// This function computes the result synchronously. For a parallel
/// version of this function, see `par_prf_divide_results`. The second
/// return argument is `true` if it foufnd a zero in the
/// denominator. Else, it is `false`.
///
/// * `numerator`: Numerator of the division
/// * `denominator`: denominator of the division
fn prf_divide_results_and_mask<Data: Num + Clone, Dim: Dimension>(
    numerator: ArcArray<Data, Dim>,
    mut denominator: ArrayViewMut<Data, Dim>,
) -> (ArcArray<Data, Dim>, Found0InDenominator) {
    let mut found_zero_in_num: Found0InDenominator = false;
    denominator.mapv_inplace(|v| {
        if v == Data::zero() {
            found_zero_in_num = true;
            Data::one()
        } else {
            v
        }
    });
    (numerator / denominator, found_zero_in_num)
}

/// Helper function to replace values from an array.
fn replace<Data: PartialEq + Copy, D: Dimension>(
    mut array: ArcArray<Data, D>,
    replaced: Data,
    new_value: Data,
) -> ArcArray<Data, D> {
    array.mapv_inplace(|v| if v == replaced { new_value } else { v });
    array
}

/// Helper function to replace values from an array in parallel.
fn par_replace<Data: PartialEq + Send + Sync + Copy, D: Dimension>(
    mut array: ArcArray<Data, D>,
    replaced: Data,
    new_value: Data,
) -> ArcArray<Data, D> {
    array.par_mapv_inplace(|v| if v == replaced { new_value } else { v });
    array
}

pub fn classification_report<'a>(
    y_true: Vec<Vec<&'a str>>,
    y_pred: Vec<Vec<&'a str>>,
    sample_weight: Option<ArcArray<f32, Dim<[usize; 1]>>>,
    zero_division: DivisionByZeroStrategy,
    scheme: SchemeType,
    suffix: bool,
    delimiter: char,
    parallel: bool,
) -> Result<Reporter, ComputationError<&'a str>> {
    check_consistent_length(y_true.as_ref(), y_pred.as_ref())?;
    let entities_true = Entities::try_from_vecs(y_true, scheme, suffix, delimiter, None)?;
    let entities_pred = Entities::try_from_vecs(y_pred, scheme, suffix, delimiter, None)?;
    let entities_true_unique_tags = &entities_true.unique_tags();
    let tmp_ahash_set = &entities_pred.unique_tags();
    let unsorted_target_names = tmp_ahash_set | entities_true_unique_tags;
    let target_names_sorted_iter = BTreeSet::from_iter(unsorted_target_names); // NOTE: Is it a good idea to convert to BTreeSet? What about a vec? A custom structure?
    let (p, r, f1, s) = precision_recall_fscore_support::<f32>(
        vec![vec![]], // We use the entities_pred/true instead of the vecs of tokens
        vec![vec![]],
        1.0,
        Average::None,
        sample_weight.clone(), //inexpensive to clone!
        zero_division,
        scheme,
        suffix,
        delimiter,
        parallel,
        Some(&entities_true),
        Some(&entities_pred),
    )?;
    let mut reporter = Reporter::default();
    for (name, precision, recall, fscore, support) in multizip((
        target_names_sorted_iter.iter(),
        p.into_iter(),
        r.into_iter(),
        f1.into_iter(),
        s.into_iter(),
    )) {
        let tmp_metrics = ClassMetrics {
            class: String::from(*name),
            precision,
            recall,
            fscore,
            support,
            average: Average::None,
        };
        reporter.insert(tmp_metrics);
    }
    for avg in [
        OverallAverage::Micro,
        OverallAverage::Macro,
        OverallAverage::Weighted,
    ]
    .into_iter()
    {
        let (p, r, f1, s) = precision_recall_fscore_support::<f32>(
            vec![vec![]], // We use the entities_pred/true instead of the vecs of tokens
            vec![vec![]],
            1.0,
            avg.into(),
            sample_weight.clone(),
            zero_division,
            scheme,
            suffix,
            delimiter,
            parallel,
            Some(&entities_true),
            Some(&entities_pred),
        )?;
        let tmp_metrics =
            ClassMetrics::new_overall(avg, p.item()?, r.item()?, f1.item()?, s.item()?);
        reporter.insert(tmp_metrics);
    }
    Ok(reporter)
}

//     average_options = ("micro", "macro", "weighted")
//     for average in average_options:
//         avg_p, avg_r, avg_f1, support = precision_recall_fscore_support(
//             y_true,
//             y_pred,
//             average=average,
//             sample_weight=sample_weight,
//             zero_division=zero_division,
//             scheme=scheme,
//             suffix=suffix,
//             entities_true=entities_true,
//             entities_pred=entities_pred,
//         )
//         reporter.write("{} avg".format(average), avg_p, avg_r, avg_f1, support)
//     reporter.write_blank()

//     return reporter.report()

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classification_report() {
        let y_true = vec![
            vec!["O", "O", "O", "B-MISC", "I-MISC", "I-MISC", "O"],
            vec!["B-PER", "I-PER", "O"],
        ];
        let y_pred = vec![
            vec!["O", "O", "B-MISC", "I-MISC", "I-MISC", "I-MISC", "O"],
            vec!["B-PER", "I-PER", "O"],
        ];
        let reporter = classification_report(
            y_true,
            y_pred,
            None,
            DivisionByZeroStrategy::ReplaceBy1,
            SchemeType::IOB2,
            false,
            '-',
            true,
        );
        let reporter_unwrapped = reporter.unwrap();
        println!("{}", reporter_unwrapped);
        assert!(false)
    }

    #[test]
    fn test_par_divide_results_and_mask() {
        let numerator = array![[1., 2., 4., 5.]].into_shared();
        let mut cloned = numerator.clone();
        let mut same_cloned = numerator.clone();
        let denominator = cloned.view_mut();
        let same_denominator = same_cloned.view_mut();
        let (div_result, has_zero) =
            prf_divide_results_and_mask(numerator.clone(), same_denominator);
        let (par_div_result, par_has_zero) =
            par_prf_divide_results_and_mask(numerator, denominator);
        let has_no_zero = !has_zero;
        let par_has_no_zero = !par_has_zero;
        assert!(has_no_zero);
        assert!(par_has_no_zero);
        assert_eq!(div_result, array![[1., 1., 1., 1.,]]);
        assert_eq!(par_div_result, array![[1., 1., 1., 1.,]]);
    }
    #[test]
    fn test_replace_0s_by_1s() {
        let to_be_replaced =
            array![[[1.0, 0.0, 0.0, -1.0, 100.0], [10., 0.0, 0.0, 5.0, 10.]]].to_shared();
        let synchronous_actual = replace(to_be_replaced.clone(), 0.0, 1.0);
        let parallel_actual = par_replace(to_be_replaced, 0.0, 1.0);
        let expected = array![[[1.0, 1.0, 1.0, -1.0, 100.0], [10., 1.0, 1.0, 5.0, 10.]]];
        assert_eq!(synchronous_actual, expected);
        assert_eq!(parallel_actual, expected);
    }
    #[test]
    fn test_check_lengths() {
        let y_true = vec![vec![1, 2, 3], vec![11, 23, 90], vec![10, 2, 1, 7]];
        let y_pred = vec![vec![1, 20, 30], vec![111, 23, 90], vec![10, 20, 1, 7]];
        let y_pred_not_same_length = vec![vec![1, 2], vec![11, 23, 90], vec![10, 2, 1, 7]];
        let y_pred_not_same_lengths = vec![vec![11, 23, 90], vec![10, 2, 1, 7]];
        assert!(check_consistent_length(&y_true.clone(), &y_pred).is_ok());
        assert!(check_consistent_length(&y_true.clone(), &y_pred_not_same_length).is_err());
        assert!(check_consistent_length(&y_true, &y_pred_not_same_lengths).is_err());
    }

    #[test]
    fn test_extract_tp_actual_correct() {
        let y_true = vec![
            vec!["O", "O", "O", "B-MISC", "I-MISC", "I-MISC", "O"],
            vec!["B-PER", "I-PER", "O"],
        ];
        let y_pred = vec![
            vec!["O", "O", "B-MISC", "I-MISC", "I-MISC", "I-MISC", "O"],
            vec!["B-PER", "I-PER", "O"],
        ];
        // predicted sum, true positive sum and true sum
        let (predicted_sum, true_positive_sum, true_sum) =
            extract_tp_actual_correct(y_true, y_pred, SchemeType::IOB2, false, '-', None, None)
                .unwrap();
        dbg!(true_positive_sum.clone());
        let expected = (vec![1, 1], vec![0, 1], vec![1, 1]);
        assert_eq!(
            (expected),
            (
                predicted_sum.to_vec(),
                true_positive_sum.to_vec(),
                true_sum.to_vec(),
            )
        );
    }
    #[test]
    fn test_precision_recall_fscore_support() {
        let y_true = vec![
            vec!["O", "O", "O", "B-MISC", "I-MISC", "I-MISC", "O"],
            vec!["B-PER", "I-PER", "O"],
        ];
        let y_pred = vec![
            vec!["O", "O", "B-MISC", "I-MISC", "I-MISC", "I-MISC", "O"],
            vec!["B-PER", "I-PER", "O"],
        ];
        let (precision, recall, fscore, support) = precision_recall_fscore_support::<f32>(
            y_true,
            y_pred,
            1.0,
            Average::Macro,
            None,
            DivisionByZeroStrategy::ReplaceBy1,
            SchemeType::IOB2,
            false,
            '-',
            true,
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            (0.5, 0.5, 0.5, 2),
            (
                precision.item().unwrap(),
                recall.item().unwrap(),
                fscore.item().unwrap(),
                support.item().unwrap()
            )
        );
    }
    #[test]
    fn test_len_all_averages_strings() {
        let actual = Average::ALL_SPECIAL_ORDERED_CLASS.into_iter().count();
        let expected = 4;
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_len_special_ordered_classes() {
        let actual = Average::ALL_SPECIAL_ORDERED_CLASS.into_iter().count();
        let expected = 4;
        assert_eq!(actual, expected);
    }
    // >>> from seqeval.metrics.v1 import precision_recall_fscore_support
    // >>> from seqeval.scheme import IOB2
    // >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
    // >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
    // >>> precision_recall_fscore_support(y_true, y_pred, average='macro', scheme=IOB2)
    // (0.5, 0.5, 0.5, 2)
    // >>> precision_recall_fscore_support(y_true, y_pred, average='micro', scheme=IOB2)
    // (0.5, 0.5, 0.5, 2)
    // >>> precision_recall_fscore_support(y_true, y_pred, average='weighted', scheme=IOB2)
    // (0.5, 0.5, 0.5, 2)
    //
    // It is possible to compute per-label precisions, recalls, F1-scores and
    // supports instead of averaging:
    //
    // >>> precision_recall_fscore_support(y_true, y_pred, average=None, scheme=IOB2)
    // (array([0., 1.]), array([0., 1.]), array([0., 1.]), array([1, 1]))
}
