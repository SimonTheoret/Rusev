/*
 * This modules contains some quality of life structs and alias. Most importantly, it contains the
 * `RusevConfig` struct, which implements the default trait. This config can be passed to the
 * `classification_report` function or the `precision_recall_fscore_support` function to simplify
 * their arguments.
*/
use crate::entity::SchemeType;
use crate::metrics::DivByZeroStrat;
use std::fmt::{Debug, Display};

/// Reasonable default configuration when computation metrics.
pub type DefaultRusevConfig = RusevConfig<Vec<f32>, DivByZeroStrat, SchemeType>;

impl<Samples, ZeroDiv, Scheme> From<(Option<Samples>, ZeroDiv, Option<Scheme>, bool, bool, bool)>
    for RusevConfig<Samples, ZeroDiv, Scheme>
where
    Samples: Into<Option<Vec<f32>>>,
    ZeroDiv: Into<DivByZeroStrat>,
    Scheme: Into<SchemeType>,
{
    fn from(value: (Option<Samples>, ZeroDiv, Option<Scheme>, bool, bool, bool)) -> Self {
        Self {
            sample_weights: value.0,
            zero_division: value.1,
            scheme: value.2,
            suffix: value.3,
            parallel: value.4,
            strict: value.5,
        }
    }
}

impl<Samples, ZeroDiv, Scheme> From<RusevConfig<Samples, ZeroDiv, Scheme>>
    for (
        Option<Vec<f32>>,
        DivByZeroStrat,
        Option<crate::entity::SchemeType>,
        bool,
        bool,
        bool,
    )
where
    Samples: Into<Option<Vec<f32>>>,
    ZeroDiv: Into<DivByZeroStrat>,
    Scheme: Into<SchemeType>,
{
    fn from(value: RusevConfig<Samples, ZeroDiv, Scheme>) -> Self {
        (
            value.sample_weights.map(|v| v.into()).unwrap(),
            value.zero_division.into(),
            value.scheme.map(|v| v.into()),
            value.suffix,
            value.parallel,
            value.strict,
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
/// Config struct used to simplify the imputs of parameters to the main functions of `Rusev`. It
/// implements the default trait.
pub struct RusevConfig<Samples, ZeroDiv, Scheme>
where
    Samples: Into<Option<Vec<f32>>>,
    ZeroDiv: Into<DivByZeroStrat>,
    Scheme: Into<SchemeType>,
{
    // y_true: Input,
    // y_pred: Input,
    sample_weights: Option<Samples>,
    zero_division: ZeroDiv,
    scheme: Option<Scheme>,
    suffix: bool,
    parallel: bool,
    strict: bool,
}

impl Default for DefaultRusevConfig {
    fn default() -> Self {
        Self {
            sample_weights: None,
            zero_division: DivByZeroStrat::ReplaceBy0,
            scheme: None,
            suffix: false,
            parallel: false,
            strict: false,
        }
    }
}

impl<Samples, ZeroDiv, Scheme> Display for RusevConfig<Samples, ZeroDiv, Scheme>
where
    Samples: Into<Option<Vec<f32>>> + Debug,
    ZeroDiv: Into<DivByZeroStrat> + Debug,
    Scheme: Into<SchemeType> + Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let string = format!("Samples weights: {:?}\n Strategy when encountering a division by zero: {:?}\n Optional scheme used:{:?}\n Prefix located in the front of the tokens: {}\n Using parallel computations: {}\n Using strict mode: {}", self.sample_weights, self.zero_division, self.scheme, self.suffix, self.parallel, self.strict);
        write!(f, "{}", string)
    }
}

enum LeftOrRight<L, R> {
    Left(L),
    Right(R),
}

/// This builder can be used to build and customize a `RusevConfig` stucture.
pub struct RusevConfigBuilder<Samples, ZeroDiv, Scheme>
where
    Samples: Into<Option<Vec<f32>>>,
    ZeroDiv: Into<DivByZeroStrat>,
    Scheme: Into<Scheme>,
{
    sample_weights: Option<Samples>,
    zero_division: LeftOrRight<ZeroDiv, DivByZeroStrat>,
    scheme: Option<Scheme>,
    suffix: bool,
    parallel: bool,
    strict: bool,
}

impl<Samples, ZeroDiv, Scheme> RusevConfigBuilder<Samples, ZeroDiv, Scheme>
where
    Samples: Into<Option<Vec<f32>>>,
    ZeroDiv: Into<DivByZeroStrat>,
    Scheme: Into<Scheme>,
{
    pub fn sample_weights(mut self, samples_weights: Samples) -> Self {
        self.sample_weights = Some(samples_weights);
        self
    }
    pub fn division_by_zero(mut self, division_by_zero: ZeroDiv) -> Self {
        self.zero_division = LeftOrRight::Left(division_by_zero);
        self
    }
    pub fn scheme(mut self, scheme: Scheme) -> Self {
        self.scheme = Some(scheme);
        self
    }
    pub fn strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }
    pub fn suffix(mut self, suffix: bool) -> Self {
        self.suffix = suffix;
        self
    }
    pub fn new() -> Self {
        Self {
            sample_weights: None,
            zero_division: LeftOrRight::Right(DivByZeroStrat::ReplaceBy0),
            scheme: None,
            suffix: false,
            parallel: false,
            strict: false,
        }
    }
}
