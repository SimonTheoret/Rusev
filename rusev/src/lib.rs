/*!
This library is  a re-implementation of the SeqEval library. It is built with a focus on
performance and soudness.
# SCHEMES
The current schemes are supported:
* IOB1: Here, `I` is a token inside a chunk, `O` is a token outside a chunk and `B` is the
    beginning of the chunk immediately following another chunk of the same named entity.
* IOB2: It is same as IOB1, except that a `B` tag is given for every token, which exists at the
    beginning of the chunk.
* IOE1: An `E` tag used to mark the last token of a chunk immediately preceding another chunk of
    the same named entity.
* IOE2: It is same as IOE1, except that an `E` tag is given for every token, which exists at the
    end of the chunk.
* BILOU/IOBES: 'E' and 'L' denotes `Last` or `Ending` character in a sequence and 'S' denotes a
    single element  and 'U' a unit element.

The BILOU and IOBES schemes are only supported in strict mode.

## More information about schemes
* [Wikipedia](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging))
* [Article](https://cs229.stanford.edu/proj2005/KrishnanGanapathy-NamedEntityRecognition.pdf), chapter 2

# Terminology
This library partially reuses the terminology of the SeqEval library. The concepts might not be
mapped one to one.
* A class is an entity we are interested in, such as 'LOC' for location, 'PER' for person, 'GEO'
    for geography, etc. It can be anything, but must be represented by a string.
* A token is a string containing a class, such a `GEO`, `LOC`, `PER` and a prefix. The prefix
    indicates where we are in the current chunk. For a given scheme, the list of possible prefix are
    the letters of the scheme, such as I-O-B or I-O-E. Prefix are limited to the letters `O`, `I`,
    `B`, `E`, `U` and `L`. It is essential that the tokens use these prefix.
* A chunk is list of at least one token associated with a named entity. A chunk could be `["B-PER",
    "I-PER". "I-PER"]`
* A Scheme gives us enough information to parse a list of tokens into a chunk. The `SchemeType` can
    be used to autodetect the `Scheme` used in a given list of sequences.
*/
//TODO: Add information about the different options, such as `strict`, `parallel`, `zero_division`,
//`suffix`, `sample_weight`.

mod config;
mod datastructure;
mod entity;
mod metrics;
mod reporter;

// The public api starts here
pub use entity::SchemeType;

pub use metrics::{
    classification_report, precision_recall_fscore_support, ComputationError, DivByZeroStrat,
    PrecisionRecallFScoreTrueSum,
};

pub use reporter::{Average, ClassMetrics, Reporter};

pub use config::{DefaultRusevConfig, RusevConfig, RusevConfigBuilder};

/// Main entrypoint of the Rusev library. This function computes the precision, recall, fscore and
/// support of the true and predicted tokens. It returns information about the individual classes
/// and different overall averages. The returned structure can be used to prettyprint the results
/// or be converted into a HashSet. Instead of taking in the raw parameters, this function takes a
/// `RusevConfig` struct and uses sensible defaults.
///
/// * `y_true`: True tokens
/// * `y_pred`: Predicted tokens
/// * `config`: Parameters used to compute the metrics of each classes.
///
/// #Example
/// ```rust
/// use rusev::{SchemeType, RusevConfigBuilder, DefaultRusevConfig, classification_report_conf};
///
/// let y_true = vec![vec!["B-TEST", "B-NOTEST", "O", "B-TEST"]];
/// let y_pred = vec![vec!["O", "B-NOTEST", "B-OTHER", "B-TEST"]];
/// let config: DefaultRusevConfig =
/// RusevConfigBuilder::default().scheme(SchemeType::IOB2).strict(true).build();
///
/// let wrapped_reporter = classification_report_conf(y_true, y_pred, config);
/// let reporter = wrapped_reporter.unwrap();
/// let expected_report = "Class, Precision, Recall, Fscore, Support
/// Overall_Weighted, 1, 0.6666667, 0.77777785, 3
/// Overall_Micro, 0.6666667, 0.6666667, 0.6666667, 3
/// Overall_Macro, 0.6666667, 0.5, 0.5555556, 3
/// NOTEST, 1, 1, 1, 1
/// OTHER, 0, 0, 0, 0
/// TEST, 1, 0.5, 0.6666667, 2\n";
///
/// assert_eq!(expected_report, reporter.to_string());
/// ```
pub fn classification_report_conf<Samples, ZeroDiv, Scheme>(
    y_true: Vec<Vec<&str>>,
    y_pred: Vec<Vec<&str>>,
    config: RusevConfig<Samples, ZeroDiv, Scheme>,
) -> Result<Reporter, ComputationError<String>>
where
    Samples: Into<Option<Vec<f32>>>,
    ZeroDiv: Into<DivByZeroStrat>,
    Scheme: Into<SchemeType>,
{
    let (sample_weights, div_by_zero, scheme, suffix, parallel) = config.into();
    classification_report(
        y_true,
        y_pred,
        sample_weights,
        div_by_zero,
        scheme,
        suffix,
        parallel,
    )
}
