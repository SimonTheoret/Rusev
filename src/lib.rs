/*!
This library is  a re-implementation of the SeqEval library. SeqEval is built with python and
can be slow when handling a large amount of strings. This library hopes to fulfill the same
niche, but hopefully in a much more performant way.
# SCHEMES
The current schemes are supported:
* IOB1: Here, I is a token inside a chunk, O is a token outside a chunk and B is the beginning
  of chunk immediately following another chunk of the same Named Entity.
* IOB2: It is same as IOB1, except that a B tag is given for every token, which exists at the
  beginning of the chunk.
* IOE1: An E tag used to mark the last token of a chunk immediately preceding another chunk of
  the same named entity.
* IOE2: It is same as IOE1, except that an E tag is given for every token, which exists at the
  end of the chunk.
* BILOU/IOBES: 'E' and 'L' denotes Last or Ending character in a sequence and 'S' denotes a single
  element  and 'U' a unit element.
# NOTE ON B-TAG
The B-prefix before a tag indicates that the tag is the beginning of a chunk that immediately
follows another chunk of the same type without O tags between them. It is used only in that
case: when a chunk comes after an O tag, the first token of the chunk takes the I- prefix.
*/

mod metrics;
mod reporter;
mod schemes;

pub use schemes::{
    ConversionError, Entities, Entity, InvalidToken, ParsingPrefixError, Prefix, SchemeType,
};

pub use metrics::{
    classification_report, precision_recall_fscore_support, ComputationError,
    DivisionByZeroStrategy,
};

pub use reporter::{Reporter, ClassMetrics};
