/*!
This library is  a re-implementation of the SeqEval library. SeqEval is built with python and
can be slow when handling a large amount of strings. This library hopes to fulfill the same
niche, but hopefully in a much more performant way.
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
for geography, etc. It can be anything.
* A token is a string containing a class, such a `GEO`, `LOC`, `PER` and a prefix. The prefix
indicates where we are in the current chunk. For a given scheme, the list of possible prefix are
the letters of the scheme, such as I-O-B or I-O-E. Prefix can only be a single ascii character.
* A chunk is list of at least one token associated with a named entity.
* A Scheme gives us enough information to parse a list of tokens into a chunk.
*/
//TODO: Add information about the different options, such as `strict`, `parallel`, `zero_division`,
//`suffix`, `sample_weight`.

mod entity;
mod metrics;
mod reporter;

// The public api starts here
pub use entity::SchemeType;

pub use metrics::{classification_report, ComputationError, DivByZeroStrat, precision_recall_fscore_support};

pub use reporter::{ClassMetrics, Reporter};
