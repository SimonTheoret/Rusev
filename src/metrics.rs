/**
This module computes the metrics (precision, recall, f-score, support) of a ground-truth
sequence and a predicted sequence.
*/
use crate::reporter::{Average, ClassMetricsInner, OverallAverage, Reporter};
use crate::schemes::{
    get_entities_lenient, ConversionError, Entities, InvalidToken, ParsingPrefixError, SchemeType,
    TryFromVecStrict,
};
use ahash::{HashMap as AHashMap, HashSet as AHashSet};
use core::fmt;
use itertools::multizip;
use ndarray::{prelude::*, Array, Data, ScalarOperand, Zip};
use ndarray_stats::{errors::MultiInputError, SummaryStatisticsExt};
use num::{Float, Num, NumCast};
use std::{
    borrow::Borrow,
    cmp,
    collections::BTreeSet,
    error::Error,
    fmt::{Debug, Display},
    str::FromStr,
};

#[derive(Debug, Clone, PartialEq)]
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
/// How do we handle cases with a division by zero? Do we replace the denominator by 1, return an
/// error, or replace the division result with 0? SeqEval uses by default the `ReplaceBy0`
/// strategy. It is not recommended to use the ReturnError; it will stop the computation. It can be
/// useful if you believe there should be no 0 in the denominator.
pub enum DivByZeroStrat {
    /// Replace denominator equal to `0` by `1` for the calculations
    ReplaceBy1,
    /// Returns an error
    ReturnError,
    /// Returns 0 when the denominator is 0
    ReplaceBy0,
}
impl Default for DivByZeroStrat {
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

impl FromStr for DivByZeroStrat {
    type Err = ParsingDivisionByZeroStrategyError<String>;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_ref() {
            "replaceby1" | "replacebyone" => Ok(DivByZeroStrat::ReplaceBy1),
            "returnerror" | "error" => Ok(DivByZeroStrat::ReturnError),
            _ => Err(ParsingDivisionByZeroStrategyError(String::from(s))),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DivisionByZeroError;

impl Display for DivisionByZeroError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Encountered division by zero")
    }
}

impl Error for DivisionByZeroError {}

/// Internal extension trait for Num's Float trait
pub trait FloatExt: Float + Send + Sync + Clone + ScalarOperand + Debug {}

impl<T: Float + Send + Sync + Clone + Copy + ScalarOperand + Debug> FloatExt for T {}

// /// Internal extension trait for Num's Integer trait
// pub trait IntExt: Integer + Send + Sync + Clone + Copy + ScalarOperand + Debug {}
//
// impl<T: Integer + Send + Sync + Clone + Copy + ScalarOperand + Debug> IntExt for T {}

fn prf_divide<I: Debug + Num + Clone + Send + Sync + Copy, D: Dimension>(
    numerator: ArcArray<I, D>,
    denominator: ArrayViewMut<I, D>,
    parallel: bool,
    zero_division: DivByZeroStrat,
) -> Result<ArcArray<I, D>, DivisionByZeroError> {
    let (mut result, zero_mask) = if parallel {
        par_prf_divide_results_and_mask(numerator, denominator)
    } else {
        prf_divide_results_and_mask(numerator, denominator)
    };

    match zero_division {
        DivByZeroStrat::ReturnError => Err(DivisionByZeroError),
        DivByZeroStrat::ReplaceBy1 => {
            if parallel {
                result = par_replace(result, I::zero(), I::one());
            } else {
                result = replace(result, I::zero(), I::one());
            }
            Ok(result)
        }
        DivByZeroStrat::ReplaceBy0 => {
            let final_result = result * zero_mask;
            Ok(final_result)
        }
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

fn extract_tp_actual_correct_strict<'a>(
    y_true: Vec<Vec<&'a str>>,
    y_pred: Vec<Vec<&'a str>>,
    scheme: SchemeType,
    suffix: bool,
    delimiter: char,
    entities_true: Option<&Entities<'a>>,
    entities_pred: Option<&Entities<'a>>,
) -> Result<ActualTPCorrect<usize>, ComputationError<String>> {
    let entities_true_res = match entities_true {
        Some(e) => e,
        None => &Entities::try_from_vecs_strict(y_true, scheme, suffix, delimiter, None)?,
    };
    let entities_pred_res = match entities_pred {
        Some(e) => e,
        None => &Entities::try_from_vecs_strict(y_pred, scheme, suffix, delimiter, None)?,
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

fn extract_tp_actual_correct_lenient<'a>(
    y_true: Vec<Vec<&'a str>>,
    y_pred: Vec<Vec<&'a str>>,
    _scheme: SchemeType,
    suffix: bool,
    delimiter: char,
    entities_true: Option<&Entities<'a>>,
    entities_pred: Option<&Entities<'a>>,
) -> Result<ActualTPCorrect<usize>, ComputationError<String>> {
    let entities_true_tmp = match entities_true {
        Some(v) => v,
        None => &get_entities_lenient(y_true.as_ref(), suffix, delimiter)?,
    };
    let mut entities_true_init: AHashMap<&str, AHashSet<(usize, usize)>> = AHashMap::default();
    for e in entities_true_tmp.iter().flatten() {
        let (start, end) = (e.start.clone(), e.end.clone());
        match entities_true_init.get_mut(e.tag.as_ref()) {
            Some(set) => {
                set.insert((start, end));
            }
            None => {
                let mut tmp_set: AHashSet<(usize, usize)> = AHashSet::default();
                tmp_set.insert((start, end));
                entities_true_init.insert(e.tag.borrow(), tmp_set);
            }
        }
    }
    let entities_pred_tmp = match entities_pred {
        Some(v) => v,
        None => &get_entities_lenient(y_pred.as_ref(), suffix, delimiter)?,
    };
    let mut entities_pred_init: AHashMap<&str, AHashSet<(usize, usize)>> = AHashMap::default();
    for e in entities_pred_tmp.iter().flatten() {
        let (start, end) = (e.start.clone(), e.end.clone());
        match entities_pred_init.get_mut(e.tag.as_ref()) {
            Some(set) => {
                set.insert((start, end));
            }
            None => {
                let mut tmp_set: AHashSet<(usize, usize)> = AHashSet::default();
                tmp_set.insert((start, end));
                entities_pred_init.insert(e.tag.borrow(), tmp_set);
            }
        }
    }
    let y_pred_keys_set = BTreeSet::from_iter(entities_true_init.keys());
    let y_true_keys_set = BTreeSet::from_iter(entities_true_init.keys());
    let target_name = y_pred_keys_set.union(&y_true_keys_set);
    let max_size = cmp::max(y_pred_keys_set.len(), y_true_keys_set.len());
    let mut tp_sum = Vec::with_capacity(max_size);
    let mut pred_sum = Vec::with_capacity(max_size);
    let mut true_sum = Vec::with_capacity(max_size);

    for type_name in target_name {
        let true_sum_len = entities_true_init
            .get(*type_name)
            .map(|s| s.len())
            .unwrap_or(0);
        true_sum.push(true_sum_len);
        let pred_sum_len = entities_pred_init
            .get(**type_name)
            .map(|s| s.len())
            .unwrap_or(0);
        pred_sum.push(pred_sum_len);
        let tmp_pred_init_set = match entities_pred_init.get(**type_name) {
            Some(set) => set,
            None => &AHashSet::default(),
        };
        let tmp_true_init_set = match entities_true_init.get(**type_name) {
            Some(set) => set,
            None => &AHashSet::default(),
        };
        dbg!(tmp_true_init_set.clone());
        dbg!(tmp_pred_init_set.clone());
        let tp_sum_len = tmp_pred_init_set.intersection(tmp_true_init_set);
        dbg!(tp_sum_len.clone().collect::<Vec<_>>());
        tp_sum.push(tp_sum_len.count());
    }
    dbg!(true_sum.clone());
    dbg!(pred_sum.clone());
    return Ok((
        Array::from(true_sum),
        Array::from(tp_sum),
        Array::from(pred_sum),
    ));
}

// entities_true = defaultdict(set)
// entities_pred = defaultdict(set)
// for type_name, start, end in get_entities(y_true, suffix):
//     entities_true[type_name].add((start, end))
// for type_name, start, end in get_entities(y_pred, suffix):
//     entities_pred[type_name].add((start, end))
//
// target_names = sorted(set(entities_true.keys()) | set(entities_pred.keys()))
//
// tp_sum = np.array([], dtype=np.int32)
// pred_sum = np.array([], dtype=np.int32)
// true_sum = np.array([], dtype=np.int32)
// for type_name in target_names:
//     entities_true_type = entities_true.get(type_name, set())
//     entities_pred_type = entities_pred.get(type_name, set())
//     tp_sum = np.append(tp_sum, len(entities_true_type & entities_pred_type))
//     pred_sum = np.append(pred_sum, len(entities_pred_type))
//     true_sum = np.append(true_sum, len(entities_true_type))
//
// return pred_sum, tp_sum, true_sum

#[derive(Debug, Clone, PartialEq)]
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

impl<S: AsRef<str> + std::fmt::Debug> From<ParsingPrefixError<S>> for ComputationError<S> {
    fn from(value: ParsingPrefixError<S>) -> Self {
        let tmp_value = ConversionError::from(value);
        Self::ConversionError(tmp_value)
    }
}

impl<S: AsRef<str> + std::fmt::Debug> From<InvalidToken> for ComputationError<S> {
    fn from(value: InvalidToken) -> Self {
        let tmp_value = ConversionError::from(value);
        Self::ConversionError(tmp_value)
    }
}

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

/// One of the main entrypoints of the Rusev library. This function computes the precision, recall,
/// fscore and support of the true and predicted tokens.
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
/// * `parallel`: Can we use multiple cores for computations?
/// * `entities_true`: Optional entities used to reduce the computation load.
/// * `entities_pred`: Optional entities used to reduce the computation load.
/// * `strict`: Are we using the strict mode.
fn precision_recall_fscore_support<'a, F: FloatExt>(
    y_true: Vec<Vec<&'a str>>,
    y_pred: Vec<Vec<&'a str>>,
    beta: F,
    average: Average,
    sample_weight: Option<ArcArray<f32, Dim<[usize; 1]>>>,
    zero_division: DivByZeroStrat,
    scheme: SchemeType,
    suffix: bool,
    delimiter: char,
    parallel: bool,
    entities_true: Option<&Entities<'a>>,
    entities_pred: Option<&Entities<'a>>,
    strict: bool,
) -> Result<PrecisionRecallFScoreTrueSum, ComputationError<String>> {
    if beta.is_sign_negative() {
        return Err(ComputationError::BetaNotPositive);
    };
    let (mut pred_sum, mut tp_sum, mut true_sum) = if strict {
        extract_tp_actual_correct_strict(
            y_true,
            y_pred,
            scheme,
            suffix,
            delimiter,
            entities_true,
            entities_pred,
        )?
    } else {
        extract_tp_actual_correct_lenient(
            y_true,
            y_pred,
            scheme,
            suffix,
            delimiter,
            entities_true,
            entities_pred,
        )?
    };
    dbg!(&pred_sum, &tp_sum, &true_sum);
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
        parallel,
        zero_division,
    )?;
    let recall = prf_divide(
        arc_tp_sum,
        true_sum.mapv(|x| x as f32).view_mut(),
        parallel,
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
                    DivByZeroStrat::ReturnError => {
                        return Err(ComputationError::DivisionByZero(DivisionByZeroError))
                    }
                    _ => {
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

/// This function computes the result in parallel. For a synchronous
/// version of this function, see `prf_divide_results`.
///
/// * `numerator`: Numerator of the division
/// * `denominator`: Denominator of the division
fn par_prf_divide_results_and_mask<I: Debug + Num + Clone + Send + Sync, D: Dimension>(
    numerator: ArcArray<I, D>,
    mut denominator: ArrayViewMut<I, D>,
) -> (ArcArray<I, D>, Array<I, D>) {
    let zero_at_mask = Zip::from(&mut denominator).par_map_collect(|d| {
        if *d == I::zero() {
            I::zero()
        } else {
            I::one()
        }
    });
    denominator.par_mapv_inplace(|v| if v == I::zero() { I::one() } else { v });
    // denominator.par_mapv_inplace(|v| if v == I::zero() { I::one() } else { v });
    // zero_at_mask.par_mapv_inplace()

    (numerator / denominator, zero_at_mask)
}

/// This function computes the result synchronously. For a parallel
/// version of this function, see `par_prf_divide_results`.
///
/// * `numerator`: Numerator of the division
/// * `denominator`: Denominator of the division
fn prf_divide_results_and_mask<I: Debug + Num + Clone, D: Dimension>(
    numerator: ArcArray<I, D>,
    mut denominator: ArrayViewMut<I, D>,
) -> (ArcArray<I, D>, Array<I, D>) {
    let zero_at_mask =
        Zip::from(&mut denominator)
            .map_collect(|d| if *d == I::zero() { I::zero() } else { I::one() });
    denominator.mapv_inplace(|v| if v == I::zero() { I::one() } else { v });
    (numerator / denominator, zero_at_mask)
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

/// Main entrypoint of the Rusev library. This function computes the precision, recall, fscore and
/// support of the true and predicted tokens. It returns information about the individual classes
/// and different overall averages. The returned structure can be used to prettyprint the results.
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
///   the rest of the tag? The most common delimiter is `-`.
/// * `parallel`: Can we use multiple cores for matrix computations?
pub fn classification_report<'a>(
    y_true: Vec<Vec<&'a str>>,
    y_pred: Vec<Vec<&'a str>>,
    sample_weight: Option<Vec<f32>>,
    zero_division: DivByZeroStrat,
    scheme: SchemeType,
    suffix: bool,
    delimiter: char,
    parallel: bool,
    strict: bool,
) -> Result<Reporter, ComputationError<String>> {
    check_consistent_length(y_true.as_ref(), y_pred.as_ref())?;
    let sample_weight_array = sample_weight.map(|x| ArcArray::from_vec(x));
    let entities_true = if strict {
        Entities::try_from_vecs_strict(y_true, scheme, suffix, delimiter, None)?
    } else {
        get_entities_lenient(y_true.as_ref(), suffix, delimiter)?
    };
    let entities_pred = if strict {
        Entities::try_from_vecs_strict(y_pred, scheme, suffix, delimiter, None)?
    } else {
        get_entities_lenient(y_pred.as_ref(), suffix, delimiter)?
    };
    let entities_true_unique_tags = &entities_true.unique_tags();
    let tmp_ahash_set = &entities_pred.unique_tags();
    let unsorted_target_names = tmp_ahash_set | entities_true_unique_tags;
    let target_names_sorted_iter = BTreeSet::from_iter(unsorted_target_names);
    let (p, r, f1, s) = precision_recall_fscore_support::<f32>(
        vec![vec![]], // We use the entities_pred/true instead of the vecs of tokens
        vec![vec![]],
        1.0,
        Average::None,
        sample_weight_array.clone(), //inexpensive to clone!
        zero_division,
        scheme,
        suffix,
        delimiter,
        parallel,
        Some(&entities_true),
        Some(&entities_pred),
        strict,
    )?;
    let mut reporter = Reporter::default();
    for (name, precision, recall, fscore, support) in multizip((
        target_names_sorted_iter.iter(),
        p.into_iter(),
        r.into_iter(),
        f1.into_iter(),
        s.into_iter(),
    )) {
        let tmp_metrics = ClassMetricsInner {
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
            sample_weight_array.clone(),
            zero_division,
            scheme,
            suffix,
            delimiter,
            parallel,
            Some(&entities_true),
            Some(&entities_pred),
            strict,
        )?;
        let tmp_metrics =
            ClassMetricsInner::new_overall(avg, p.item()?, r.item()?, f1.item()?, s.item()?);
        reporter.insert(tmp_metrics);
    }
    Ok(reporter)
}

#[cfg(test)]
mod tests {
    pub trait CloseEnough {
        fn are_close(&self, other: &Self, eps: f32) -> bool;
    }
    // ClassMetrics does not have the default PartialEq implementation.
    impl CloseEnough for ClassMetricsInner {
        fn are_close(&self, other: &Self, eps: f32) -> bool {
            let are_equal = self == other;
            let precision_is_equal = f32::abs(self.precision - other.precision) < eps;
            let recall_is_equal = f32::abs(self.recall - other.recall) < eps;
            let fscore_is_equal = f32::abs(self.fscore - other.fscore) < eps;
            let support_is_equal = self.support == other.support;
            return are_equal
                && precision_is_equal
                && recall_is_equal
                && fscore_is_equal
                && support_is_equal;
        }
    }
    impl CloseEnough for Reporter {
        fn are_close(&self, other: &Self, eps: f32) -> bool {
            for (c1, c2) in self.classes.iter().zip(other.classes.iter()) {
                let are_close = c1.are_close(c2, eps);
                if !are_close {
                    return false;
                };
            }
            true
        }
    }

    use super::*;
    #[test]
    fn test_reporter_output() {
        let y_true = vec![vec!["B-A", "B-B", "O", "B-A"]];
        let y_pred = vec![vec!["O", "B-B", "B-C", "B-A"]];
        let actual = classification_report(
            y_true,
            y_pred,
            None,
            DivByZeroStrat::ReplaceBy0,
            SchemeType::IOB2,
            false,
            '-',
            false,
            true,
        )
        .unwrap();
        {
            let expected = Reporter {
                classes: BTreeSet::from_iter(vec![
                    ClassMetricsInner {
                        class: String::from("A"),
                        fscore: 0.6666666666666666,
                        precision: 1.0,
                        recall: 0.5,
                        support: 2,
                        average: Average::None,
                    },
                    ClassMetricsInner {
                        class: String::from("B"),
                        fscore: 1.0,
                        precision: 1.0,
                        recall: 1.0,
                        support: 1,
                        average: Average::None,
                    },
                    ClassMetricsInner {
                        class: String::from("C"),
                        fscore: 0.0,
                        precision: 0.0,
                        recall: 0.0,
                        support: 0,
                        average: Average::None,
                    },
                    ClassMetricsInner::new_overall(
                        OverallAverage::Macro,
                        0.66666666666666,
                        0.5,
                        0.55555555555555,
                        3,
                    ),
                    ClassMetricsInner::new_overall(
                        OverallAverage::Micro,
                        0.66666666666666,
                        0.66666666666666,
                        0.66666666666666,
                        3,
                    ),
                    ClassMetricsInner::new_overall(
                        OverallAverage::Weighted,
                        1.0,
                        0.66666666666666,
                        0.77777777777777,
                        3,
                    ),
                ]),
            };
            dbg!(&actual, &expected);
            assert!(actual.are_close(&expected, 1e-6));
        }
    }

    #[test]
    fn test_check_consistent_length() {
        let test_cases = [
            (vec![vec![]], vec![vec![]], Ok(())),
            (vec![vec!['B']], vec![vec!['B']], Ok(())),
            (
                vec![vec![]],
                vec![vec!['B']],
                Err(InconsistentLengthError(0, 1)),
            ),
            (
                vec![vec!['B'], vec![]],
                vec![vec!['B']],
                Err(InconsistentLengthError(2, 1)),
            ),
            (
                vec![vec!['B'], vec![]],
                vec![vec!['B'], vec!['I']],
                Err(InconsistentLengthError(0, 1)),
            ),
        ];
        for (y_true, y_pred, expected) in test_cases.into_iter() {
            dbg!(y_true.clone(), y_pred.clone());
            assert_eq!(check_consistent_length(&y_true, &y_pred), expected)
        }
    }

    #[test]
    fn test_classification_report_inconsistent_length() {
        let test_cases: Vec<(
            Vec<Vec<&str>>,
            Vec<Vec<&str>>,
            Result<Reporter, ComputationError<String>>,
        )> = vec![
            (
                vec![vec!["B-PER"], vec!["I-PER"]],
                vec![vec![]],
                Err(ComputationError::InconsistentLenght(
                    InconsistentLengthError(2, 1),
                )),
            ),
            (
                vec![vec![]],
                vec![],
                Err(ComputationError::InconsistentLenght(
                    InconsistentLengthError(1, 0),
                )),
            ),
        ];
        for (y_true, y_pred, expected) in test_cases {
            dbg!(y_true.clone(), y_pred.clone());
            // let expected_err =
            //     expected.expect_err("This test is only veryfing the right error type");
            let actual = classification_report(
                y_true,
                y_pred,
                None,
                DivByZeroStrat::ReplaceBy1,
                SchemeType::IOB2,
                false,
                '-',
                true,
                true,
            );
            // .expect_err("There was no error here!");
            assert_eq!(actual, expected)
        }
    }

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
            DivByZeroStrat::ReplaceBy0,
            SchemeType::IOB2,
            false,
            '-',
            true,
            true,
        );
        let reporter_unwrapped = reporter.unwrap();
        dbg!("{}", reporter_unwrapped.clone());
        // NOTE: Do not change the indentation
        let expected = "Class, Precision, Recall, Fscore, Support
Overall_Weighted, 0.5, 0.5, 0.5, 2
Overall_Micro, 0.5, 0.5, 0.5, 2
Overall_Macro, 0.5, 0.5, 0.5, 2
MISC, 0, 0, 0, 1
PER, 1, 1, 1, 1\n";
        let actual = reporter_unwrapped.to_string();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_par_divide_results_and_mask() {
        let numerator = array![[1., 2., 4., 5.]].into_shared();
        let mut cloned = numerator.clone();
        let mut same_cloned = numerator.clone();
        let denominator = cloned.view_mut();
        let same_denominator = same_cloned.view_mut();
        let (div_result, zero_mask) =
            prf_divide_results_and_mask(numerator.clone(), same_denominator);
        let (par_div_result, par_zero_mask) =
            par_prf_divide_results_and_mask(numerator, denominator);
        let has_no_zero = zero_mask == ArcArray::ones(div_result.raw_dim());
        let par_has_no_zero = par_zero_mask == ArcArray::ones(par_div_result.raw_dim());
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
        let (predicted_sum, true_positive_sum, true_sum) = extract_tp_actual_correct_strict(
            y_true,
            y_pred,
            SchemeType::IOB2,
            false,
            '-',
            None,
            None,
        )
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
            y_true.clone(),
            y_pred.clone(),
            1.0,
            Average::Macro,
            None,
            DivByZeroStrat::ReplaceBy0,
            SchemeType::IOB2,
            false,
            '-',
            true,
            None,
            None,
            true,
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
        let (precision, recall, fscore, support) = precision_recall_fscore_support::<f32>(
            y_true,
            y_pred,
            1.0,
            Average::Micro,
            None,
            DivByZeroStrat::ReplaceBy0,
            SchemeType::IOB2,
            false,
            '-',
            true,
            None,
            None,
            true,
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
    fn test_f1_score() {
        let y_true = vec![vec!["B-ORG", "I-ORG"]];
        let y_pred = vec![vec!["I-ORG", "I-ORG"]];
        let test_cases = vec![
            (Average::Micro, 0.0),
            (Average::Macro, 0.0),
            (Average::Weighted, 0.0),
        ];
        for (avg, expected) in test_cases {
            let (_, _, f1, _) = precision_recall_fscore_support(
                y_true.clone(),
                y_pred.clone(),
                1.0,
                avg,
                None,
                DivByZeroStrat::ReplaceBy0,
                SchemeType::IOB2,
                false,
                '-',
                true,
                None,
                None,
                true,
            )
            .unwrap();
            let actual = f1.item().unwrap();
            assert_eq!(actual, expected)
        }
    }
    #[test]
    fn test_precision_recall_fscore_support_lenient_macro() {
        let y_true = vec![
            vec!["O", "O", "O", "B-MISC", "I-MISC", "I-MISC", "O"],
            vec!["B-PER", "I-PER", "O"],
        ];
        let y_pred = vec![
            vec!["O", "O", "B-MISC", "I-MISC", "I-MISC", "I-MISC", "O"],
            vec!["B-PER", "I-PER", "O"],
        ];
        let (arr_p, arr_r, arr_f, arr_s) = precision_recall_fscore_support(
            y_true,
            y_pred,
            1.0,
            Average::Macro,
            None,
            DivByZeroStrat::ReplaceBy0,
            SchemeType::IOB2,
            false,
            '-',
            true,
            None,
            None,
            false,
        )
        .unwrap();
        let actual = (
            arr_p.item().unwrap(),
            arr_r.item().unwrap(),
            arr_f.item().unwrap(),
            arr_s.item().unwrap(),
        );
        let expected = (0.5, 0.5, 0.5, 2);
        assert_eq!(actual, expected)
    }
    #[test]
    fn test_precision_recall_fscore_support_lenient_micro() {
        let y_true = vec![
            vec!["O", "O", "O", "B-MISC", "I-MISC", "I-MISC", "O"],
            vec!["B-PER", "I-PER", "O"],
        ];
        let y_pred = vec![
            vec!["O", "O", "B-MISC", "I-MISC", "I-MISC", "I-MISC", "O"],
            vec!["B-PER", "I-PER", "O"],
        ];
        let (arr_p, arr_r, arr_f, arr_s) = precision_recall_fscore_support(
            y_true,
            y_pred,
            1.0,
            Average::Micro,
            None,
            DivByZeroStrat::ReplaceBy0,
            SchemeType::IOB2,
            false,
            '-',
            true,
            None,
            None,
            false,
        )
        .unwrap();
        let actual = (
            arr_p.item().unwrap(),
            arr_r.item().unwrap(),
            arr_f.item().unwrap(),
            arr_s.item().unwrap(),
        );
        let expected = (0.5, 0.5, 0.5, 2);
        assert_eq!(actual, expected)
    }

    #[test]
    fn test_precision_recall_fscore_support_lenient_weighted() {
        let y_true = vec![
            vec!["O", "O", "O", "B-MISC", "I-MISC", "I-MISC", "O"],
            vec!["B-PER", "I-PER", "O"],
        ];
        let y_pred = vec![
            vec!["O", "O", "B-MISC", "I-MISC", "I-MISC", "I-MISC", "O"],
            vec!["B-PER", "I-PER", "O"],
        ];
        let (arr_p, arr_r, arr_f, arr_s) = precision_recall_fscore_support(
            y_true,
            y_pred,
            1.0,
            Average::Weighted,
            None,
            DivByZeroStrat::ReplaceBy0,
            SchemeType::IOB2,
            false,
            '-',
            true,
            None,
            None,
            false,
        )
        .unwrap();
        let actual = (
            arr_p.item().unwrap(),
            arr_r.item().unwrap(),
            arr_f.item().unwrap(),
            arr_s.item().unwrap(),
        );
        let expected = (0.5, 0.5, 0.5, 2);
        assert_eq!(actual, expected)
    }

    #[test]
    fn test_precision_recall_fscore_support_lenient_no_average() {
        let y_true = vec![
            vec!["O", "O", "O", "B-MISC", "I-MISC", "I-MISC", "O"],
            vec!["B-PER", "I-PER", "O"],
        ];
        let y_pred = vec![
            vec!["O", "O", "B-MISC", "I-MISC", "I-MISC", "I-MISC", "O"],
            vec!["B-PER", "I-PER", "O"],
        ];
        let (arr_p, arr_r, arr_f, arr_s) = precision_recall_fscore_support(
            y_true,
            y_pred,
            1.0,
            Average::None,
            None,
            DivByZeroStrat::ReplaceBy0,
            SchemeType::IOB2,
            false,
            '-',
            true,
            None,
            None,
            false,
        )
        .unwrap();
        let actual = (arr_p, arr_r, arr_f, arr_s);
        let expected = (
            array![0. as f32, 1. as f32],
            array![0. as f32, 1. as f32],
            array![0. as f32, 1. as f32],
            array![1, 1],
        );
        assert_eq!(actual, expected)
    }
}
