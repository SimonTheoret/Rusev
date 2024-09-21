//! This library is  a re-implementation of the SeqEval library. SeqEval is built with python and
//! is too slow when handling a large amount of strings. This library hopes to fulfill the same
//! niche, but hopefully in a much more performant way.

use std::borrow::Cow;
use std::error::Error;
use std::fmt::Display;

#[derive(Debug, Hash, PartialEq, Clone)]
struct Entity<'a> {
    sent_id: usize,
    start: usize,
    end: usize,
    tag: Cow<'a, str>,
}

impl<'a> Display for Entity<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({}, {}, {}, {})",
            self.sent_id, self.tag, self.start, self.end
        )
    }
}

impl<'a> Entity<'a> {
    fn as_tuple(&'a self) -> (usize, usize, usize, &'a str) {
        (self.sent_id, self.start, self.end, self.tag.as_ref())
    }
}

#[derive(Debug, PartialEq, Hash, Clone)]
pub enum Prefix {
    I,
    O,
    B,
    E,
    S,
    U,
    L,
    ANY,
}
#[derive(Debug, PartialEq, Hash, Clone)]
pub enum Tag {
    SAME,
    DIFF,
    ANY,
}

#[derive(Debug, PartialEq, Hash)]
pub struct Token<'a> {
    token: Cow<'a, str>,
    prefix: Prefix,
    tag: Cow<'a, str>,
    allowed_prefix: Option<Vec<Prefix>>,
    /// Pattern are an enum representing the diffrent positions of tokens.
    patterns: todo!(),
}

impl<'a> Token<'a> {
    /// Check whether the prefix is allowed or not
    fn is_valid(&self) -> Result<bool, InvalidTokenError> {
        match &self.allowed_prefix {
            None => Err(InvalidTokenError::from(self)),
            Some(vec_of_allowed_prefixes) => {
                let prefix_is_allowed = vec_of_allowed_prefixes.contains(&self.prefix);
                if prefix_is_allowed {
                    Ok(true)
                } else {
                    Err(InvalidTokenError::from(self))
                }
            }
        }
    }
    fn get_token_ref(&'a self) -> &'a str {
        &self.token
    }
    fn get_token_owned(&'a self) -> String {
        match &self.token {
            Cow::Owned(owned_string) => owned_string.clone(),
            Cow::Borrowed(borrowed_string) => borrowed_string.to_string(),
        }
    }
    fn get_allowed_prefixes_ref(&'a self) -> &Option<Vec<Prefix>> {
        &self.allowed_prefix
    }
    fn get_allowed_prefixes_owned(&'a self) -> Option<Vec<Prefix>> {
        self.allowed_prefix.clone()
    }
    fn is_start(&self, prev: &Token) -> bool {
        todo!()
    }
    fn is_inside(&self, prev: &Token) -> bool {
        todo!()
    }
    fn is_end(&self, prev: &Token) -> bool {
        todo!()
    }
    fn check_tag(&self, prev: &Token, cond: Tag) -> bool {
        match cond {
            Tag::ANY => true,
            Tag::SAME if prev.tag == self.tag => true,
            Tag::DIFF if prev.tag != self.tag => true,
            _ => false,
        }
    }
    /// """Check whether the prefix patterns are matched."""
    ///
    /// * `prev`: Previous token
    /// * `patterns`: Patterns to match the token against
    fn check_patterns(&self, prev: &Token, patterns: TokenWithPatterns, pattern: Pattern) -> bool {
        todo!()
    }
}

#[derive(Debug, Hash)]
struct InvalidTokenError(String, Option<Vec<Prefix>>);

impl<'a> From<&Token<'a>> for InvalidTokenError {
    fn from(value: &Token<'a>) -> Self {
        InvalidTokenError(value.get_token_owned(), value.get_allowed_prefixes_owned())
    }
}

impl Display for InvalidTokenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "The current token ({}) is not allowed. Only the following tokens are allowd: {:?}",
            self.0, self.1
        )
    }
}

impl Error for InvalidTokenError {}

enum TokenWithPatterns<'a> {
    IOB1 { token: Token<'a> },
    IOE1 { token: Token<'a> },
    IOB2 { token: Token<'a> },
    IOE2 { token: Token<'a> },
    IOBES { token: Token<'a> },
    BILOU { token: Token<'a> },
}

impl<'a> TokenWithPatterns<'a> {
    const IOB1_ALLOWED_PREFIXES: [Prefix; 3] = [Prefix::I, Prefix::O, Prefix::B];
    const IOB1_START_PATTERNS: [(Prefix, Prefix, Tag); 5] = [
        (Prefix::O, Prefix::I, Tag::ANY),
        (Prefix::I, Prefix::I, Tag::DIFF),
        (Prefix::B, Prefix::I, Tag::ANY),
        (Prefix::I, Prefix::B, Tag::SAME),
        (Prefix::B, Prefix::B, Tag::SAME),
    ];
    const IOB1_INSIDE_PATTERNS: [(Prefix, Prefix, Tag); 2] = [
        (Prefix::B, Prefix::I, Tag::SAME),
        (Prefix::I, Prefix::I, Tag::SAME),
    ];
    const IOB1_END_PATTERNS: [(Prefix, Prefix, Tag); 6] = [
        (Prefix::I, Prefix::I, Tag::DIFF),
        (Prefix::I, Prefix::O, Tag::ANY),
        (Prefix::I, Prefix::B, Tag::ANY),
        (Prefix::B, Prefix::O, Tag::ANY),
        (Prefix::B, Prefix::I, Tag::DIFF),
        (Prefix::B, Prefix::B, Tag::SAME),
    ];
    const IOE1_ALLOWED_PREFIXES: [Prefix; 3] = [Prefix::I, Prefix::O, Prefix::E];
    const IOE1_START_PATTERNS: [(Prefix, Prefix, Tag); 4] = [
        (Prefix::O, Prefix::I, Tag::ANY),
        (Prefix::I, Prefix::I, Tag::DIFF),
        (Prefix::E, Prefix::I, Tag::ANY),
        (Prefix::E, Prefix::E, Tag::SAME),
    ];
    const IOE1_INSIDE_PATTERNS: [(Prefix, Prefix, Tag); 2] = [
        (Prefix::I, Prefix::I, Tag::SAME),
        (Prefix::I, Prefix::E, Tag::SAME),
    ];
    const IOE1_END_PATTERNS: [(Prefix, Prefix, Tag); 5] = [
        (Prefix::I, Prefix::I, Tag::DIFF),
        (Prefix::I, Prefix::O, Tag::ANY),
        (Prefix::I, Prefix::E, Tag::DIFF),
        (Prefix::E, Prefix::I, Tag::SAME),
        (Prefix::E, Prefix::E, Tag::SAME),
    ];

    const IOB2_ALLOWED_PREFIXES: [Prefix; 3] = [Prefix::I, Prefix::O, Prefix::B];
    const IOB2_START_PATTERNS: [(Prefix, Prefix, Tag); 1] = [(Prefix::ANY, Prefix::I, Tag::ANY)];
    const IOB2_INSIDE_PATTERNS: [(Prefix, Prefix, Tag); 2] = [
        (Prefix::I, Prefix::I, Tag::SAME),
        (Prefix::I, Prefix::E, Tag::SAME),
    ];
    const IOB2_END_PATTERNS: [(Prefix, Prefix, Tag); 6] = [
        (Prefix::I, Prefix::O, Tag::ANY),
        (Prefix::I, Prefix::I, Tag::DIFF),
        (Prefix::I, Prefix::B, Tag::ANY),
        (Prefix::B, Prefix::O, Tag::ANY),
        (Prefix::B, Prefix::I, Tag::DIFF),
        (Prefix::B, Prefix::B, Tag::ANY),
    ];

    const IOE2_ALLOWED_PREFIXES: [Prefix; 3] = [Prefix::I, Prefix::O, Prefix::E];
    const IOE2_START_PATTERNS: [(Prefix, Prefix, Tag); 6] = [
        (Prefix::O, Prefix::I, Tag::ANY),
        (Prefix::O, Prefix::E, Tag::ANY),
        (Prefix::E, Prefix::I, Tag::ANY),
        (Prefix::E, Prefix::E, Tag::ANY),
        (Prefix::I, Prefix::I, Tag::DIFF),
        (Prefix::I, Prefix::E, Tag::DIFF),
    ];
    const IOE2_INSIDE_PATTERNS: [(Prefix, Prefix, Tag); 2] = [
        (Prefix::I, Prefix::E, Tag::SAME),
        (Prefix::I, Prefix::I, Tag::SAME),
    ];
    const IOE2_END_PATTERNS: [(Prefix, Prefix, Tag); 1] = [(Prefix::E, Prefix::ANY, Tag::ANY)];

    const IOBES_ALLOWED_PREFIXES: [Prefix; 5] =
        [Prefix::I, Prefix::O, Prefix::E, Prefix::B, Prefix::S];
    const IOBES_START_PATTERNS: [(Prefix, Prefix, Tag); 4] = [
        (Prefix::B, Prefix::I, Tag::SAME),
        (Prefix::B, Prefix::E, Tag::SAME),
        (Prefix::I, Prefix::I, Tag::SAME),
        (Prefix::I, Prefix::E, Tag::SAME),
    ];
    const IOBES_INSIDE_PATTERNS: [(Prefix, Prefix, Tag); 2] = [
        (Prefix::S, Prefix::ANY, Tag::ANY),
        (Prefix::E, Prefix::ANY, Tag::ANY),
    ];
    const IOBES_END_PATTERNS: [(Prefix, Prefix, Tag); 2] = [
        (Prefix::S, Prefix::ANY, Tag::ANY),
        (Prefix::E, Prefix::ANY, Tag::ANY),
    ];

    const BILOU_ALLOWED_PREFIXES: [Prefix; 5] =
        [Prefix::I, Prefix::O, Prefix::U, Prefix::B, Prefix::O];
    const BILOU_START_PATTERNS: [(Prefix, Prefix, Tag); 2] = [
        (Prefix::ANY, Prefix::B, Tag::ANY),
        (Prefix::ANY, Prefix::U, Tag::ANY),
    ];
    const BILOU_INSIDE_PATTERNS: [(Prefix, Prefix, Tag); 4] = [
        (Prefix::B, Prefix::I, Tag::SAME),
        (Prefix::B, Prefix::L, Tag::SAME),
        (Prefix::I, Prefix::I, Tag::SAME),
        (Prefix::I, Prefix::L, Tag::SAME),
    ];
    const BILOU_END_PATTERNS: [(Prefix, Prefix, Tag); 2] = [
        (Prefix::U, Prefix::ANY, Tag::ANY),
        (Prefix::L, Prefix::ANY, Tag::ANY),
    ];
    fn allowed_prefixes<'b>(&'a self) -> &'a [Prefix] {
        todo!();
    }
    fn start_patterns<'b>(&'a self) -> &'a [(Prefix, Prefix, Tag)] {
        todo!();
    }
    fn inside_patterns<'b>(&'a self) -> &'a [(Prefix, Prefix, Tag)] {
        todo!();
    }
    fn end_patterns<'b>(&'a self) -> &'a [(Prefix, Prefix, Tag)] {
        todo!();
    }
}
