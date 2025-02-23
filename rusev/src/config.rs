/*
 * This modules contains some quality of life structs and alias. Most importantly, it contains the
 * `RusevConfig` struct, which implements the default trait. This config can be passed to the
 * `classification_report` function or the `precision_recall_fscore_support` function to simplify
 * their arguments.
*/
use crate::metrics::DivByZeroStrat;
use either::Either as LeftOrRight;
use named_entity_parsing::SchemeType;
use std::fmt::{Debug, Display};

/// Reasonable default configuration when computing metrics.
pub type DefaultRusevConfig = RusevConfig<Vec<f32>, DivByZeroStrat, SchemeType>;

impl DefaultRusevConfig {
    pub fn new() -> Self {
        Self {
            sample_weights: None,
            zero_division: DivByZeroStrat::ReplaceBy0,
            scheme: None,
            suffix: false,
            parallel: false,
        }
    }
}

impl<Samples, ZeroDiv, Scheme> From<(Option<Samples>, ZeroDiv, Option<Scheme>, bool, bool)>
    for RusevConfig<Samples, ZeroDiv, Scheme>
where
    Samples: Into<Option<Vec<f32>>>,
    ZeroDiv: Into<DivByZeroStrat>,
    Scheme: Into<SchemeType>,
{
    fn from(value: (Option<Samples>, ZeroDiv, Option<Scheme>, bool, bool)) -> Self {
        Self {
            sample_weights: value.0,
            zero_division: value.1,
            scheme: value.2,
            suffix: value.3,
            parallel: value.4,
        }
    }
}

impl<Samples, ZeroDiv, Scheme> From<RusevConfigBuilder<Samples, ZeroDiv, Scheme>>
    for RusevConfig<Samples, DivByZeroStrat, SchemeType>
where
    Samples: Into<Option<Vec<f32>>>,
    ZeroDiv: Into<DivByZeroStrat>,
    Scheme: Into<SchemeType>,
{
    fn from(value: RusevConfigBuilder<Samples, ZeroDiv, Scheme>) -> Self {
        Self {
            sample_weights: value.sample_weights,
            zero_division: value.zero_division.either_into(),
            scheme: value.scheme.map(|s| s.into()),
            parallel: value.parallel,
            suffix: value.suffix,
        }
    }
}

impl<Samples, ZeroDiv, Scheme> From<RusevConfig<Samples, ZeroDiv, Scheme>>
    for (
        Option<Vec<f32>>,
        DivByZeroStrat,
        Option<named_entity_parsing::SchemeType>,
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
            value.sample_weights.map(|v| v.into().unwrap()),
            value.zero_division.into(),
            value.scheme.map(|v| v.into()),
            value.suffix,
            value.parallel,
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
/// Config struct used to simplify the imputs of parameters to the main functions of `Rusev`. It
/// Implements the `Default` trait.
pub struct RusevConfig<Samples, ZeroDiv, Scheme>
where
    Samples: Into<Option<Vec<f32>>>,
    ZeroDiv: Into<DivByZeroStrat>,
    Scheme: Into<SchemeType>,
{
    /// Optional weights of the samples. We expect this vector to be of length
    /// `number_of_samples`.
    sample_weights: Option<Samples>,
    ///  What to do in case of division by zero. The most common solution is to replace the result by
    ///  0.
    zero_division: ZeroDiv,
    /// What scheme are we using? If no scheme is provided (eg. `None`), we assume a that we are
    /// not in `strict` mode. `strict` mode is equivalent to `strict` mode in `SeqEval`:
    /// <https://github.com/chakki-works/seqeval/blob/master/README.md#usage>
    scheme: Option<Scheme>,
    /// Do we expect to have the prefix (such as "B", "I", "L", "O", "U" or "E") at the end of the
    /// token? If so, suffix should be `true`. If the prefix is located at the start of the token,
    /// suffix should be `false`.
    suffix: bool,
    /// Can we use multiple cores for matrix computations? This is not recommended unless you have a
    /// *very* large number of samples. It is most often better to keep it to `false`.
    parallel: bool,
}

impl Default for DefaultRusevConfig {
    fn default() -> Self {
        Self {
            sample_weights: None,
            zero_division: DivByZeroStrat::ReplaceBy0,
            scheme: None,
            suffix: false,
            parallel: false,
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
        let string = format!("Samples weights: {:?}\n Strategy when encountering a division by zero: {:?}\n Optional scheme used:{:?}\n Prefix located in the front of the tokens: {}\n Using parallel computations: {}", self.sample_weights, self.zero_division, self.scheme, self.suffix, self.parallel);
        write!(f, "{}", string)
    }
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

impl Default for RusevConfigBuilder<Vec<f32>, DivByZeroStrat, SchemeType> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Samples, ZeroDiv, Scheme> RusevConfigBuilder<Samples, ZeroDiv, Scheme>
where
    Samples: Into<Option<Vec<f32>>>,
    ZeroDiv: Into<DivByZeroStrat>,
    Scheme: Into<SchemeType>,
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
    pub fn build(self) -> RusevConfig<Samples, DivByZeroStrat, SchemeType> {
        RusevConfig::from(self)
    }
}
#[cfg(test)]
mod test {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case(DivByZeroStrat::ReplaceBy1)]
    #[case(DivByZeroStrat::ReplaceBy0)]
    #[case(DivByZeroStrat::ReturnError)]
    fn test_builder_setters_division_by_zero(#[case] strat: DivByZeroStrat) {
        let builder = RusevConfigBuilder::default();
        let div_by_zero = strat;
        let config = builder.division_by_zero(div_by_zero).build();
        assert_eq!(config.zero_division, div_by_zero)
    }

    #[test]
    fn test_builder_setters_samples() {
        let builder = RusevConfigBuilder::default();
        let samples = vec![0.3, 0.1, -100.];
        let config = builder.sample_weights(samples.clone()).build();
        assert_eq!(config.sample_weights.unwrap(), samples)
    }

    #[rstest]
    #[case(SchemeType::IOE1)]
    #[case(SchemeType::IOE2)]
    #[case(SchemeType::IOB1)]
    #[case(SchemeType::IOB2)]
    #[case(SchemeType::BILOU)]
    #[case(SchemeType::IOBES)]
    fn test_builder_setters_scheme(#[case] scheme: SchemeType) {
        let builder = RusevConfigBuilder::default();
        let config = builder.scheme(scheme).build();
        assert_eq!(config.scheme, Some(scheme))
    }

    #[rstest]
    #[case(true)]
    #[case(false)]
    fn test_builder_setters_parallel(#[case] parallel: bool) {
        let builder = RusevConfigBuilder::default();
        let config = builder.parallel(parallel).build();
        assert_eq!(config.parallel, parallel)
    }

    #[rstest]
    #[case(true)]
    #[case(false)]
    fn test_builder_setters_suffix(#[case] suffix: bool) {
        let builder = RusevConfigBuilder::default();
        let config = builder.suffix(suffix).build();
        assert_eq!(config.suffix, suffix)
    }
}
