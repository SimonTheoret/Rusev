use ahash::AHashSet;
use std::borrow::Borrow;
use std::cmp::Ordering;
use std::convert::TryFrom;
use std::error::Error;
use std::fmt::{Debug, Display};
use std::mem::take;
use std::ops::{Deref, DerefMut};
use std::str::FromStr;
use std::{borrow::Cow, cell::RefCell};
use unicode_segmentation::UnicodeSegmentation;

/// An entity represent a named objet in named entity recognition (NER). It contains a start and an
/// end(i.e. at what index of the list does it starts and ends) and a tag, which the associated
/// entity (such as `LOC`, `NAME`, `PER`, etc.)
#[derive(Debug, Hash, Clone, PartialEq, Eq, PartialOrd)]
pub struct Entity<'a> {
    sent_id: Option<usize>,
    start: usize,
    end: usize,
    tag: Cow<'a, str>,
}

impl<'a> Display for Entity<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({:?}, {}, {}, {})",
            self.sent_id, self.tag, self.start, self.end
        )
    }
}

impl<'a> Entity<'a> {
    pub fn as_tuple(&'a self) -> (Option<usize>, usize, usize, &'a str) {
        (self.sent_id, self.start, self.end, self.tag.as_ref())
    }
}

#[derive(Debug, PartialEq, Hash, Clone)]
/// Prefix represent an annotation specifying the place of a token in a chunk. For example, in
/// `IOB1`, the `I` prefix is used to indicate that the token is inside a NER.
pub enum Prefix {
    I,
    O,
    B,
    E,
    S,
    U,
    L,
    Any,
}
impl Prefix {
    /// This functions verifies that this prefix and the other prefix are the same or one of them
    /// is the `PrefixAny` prefix.
    ///
    /// * `other`: The prefix to compare with `self`
    fn are_the_same_or_contains_any(&self, other: &Prefix) -> bool {
        match (self, other) {
            (&Prefix::Any, _) => true,
            (_, &Prefix::Any) => true,
            (s, o) if s == o => true,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Could not parse the string into a `Prefix`
pub struct ParsingPrefixError<S: AsRef<str>>(S);

impl<S: AsRef<str>> Display for ParsingPrefixError<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let content = self.0.as_ref();
        write!(
            f,
            "Could not parse the following string into a Prefix: {}",
            content
        )
    }
}
impl<S: AsRef<str> + Error> Error for ParsingPrefixError<S> {}

// impl<'a> TryFrom<&'a str> for Prefix {
//     type Error = ParsingPrefixError<&'a str>;
//     fn try_from(value: &'a str) -> Result<Self, Self::Error> {
//         match value {
//             "I" => Ok(Prefix::I),
//             "O" => Ok(Prefix::O),
//             "B" => Ok(Prefix::B),
//             "E" => Ok(Prefix::E),
//             "S" => Ok(Prefix::S),
//             "U" => Ok(Prefix::U),
//             "L" => Ok(Prefix::L),
//             "ANY" => Ok(Prefix::Any),
//             _ => Err(ParsingPrefixError(value)),
//         }
//     }
// }

impl FromStr for Prefix {
    type Err = ParsingPrefixError<&'static str>;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::try_from_with_static_error(s)
    }
}

impl<'a> Prefix {
    fn try_from_with_static_error(
        value: &'a str,
    ) -> Result<Self, ParsingPrefixError<&'static str>> {
        match value {
            "I" => Ok(Prefix::I),
            "O" => Ok(Prefix::O),
            "B" => Ok(Prefix::B),
            "E" => Ok(Prefix::E),
            "S" => Ok(Prefix::S),
            "U" => Ok(Prefix::U),
            "L" => Ok(Prefix::L),
            "ANY" => Ok(Prefix::Any),
            _ => Err(ParsingPrefixError(String::from(value).leak())),
        }
    }
}

impl TryFrom<String> for Prefix {
    type Error = ParsingPrefixError<String>;
    fn try_from(value: String) -> Result<Self, Self::Error> {
        let ref_val = value.as_ref();
        match ref_val {
            "I" => Ok(Prefix::I),
            "O" => Ok(Prefix::O),
            "B" => Ok(Prefix::B),
            "E" => Ok(Prefix::E),
            "S" => Ok(Prefix::S),
            "U" => Ok(Prefix::U),
            "L" => Ok(Prefix::L),
            "ANY" => Ok(Prefix::Any),
            _ => Err(ParsingPrefixError(value)),
        }
    }
}

#[derive(Debug, PartialEq, Hash, Clone)]
enum Tag {
    Same,
    Diff,
    Any,
}

#[derive(Debug, PartialEq, Hash, Clone)]
struct InnerToken<'a> {
    token: Cow<'a, str>,
    prefix: Prefix,
    tag: Cow<'a, str>,
}

impl<'a> Default for InnerToken<'a> {
    fn default() -> Self {
        InnerToken {
            token: Cow::Borrowed(""),
            prefix: Prefix::I,
            tag: Cow::Borrowed(""),
        }
    }
}

// TODO: Move this enum into its own module, as to hide its `new` function.
///
/// This enum represents the positon of the Prefix in a token (a Cow<'_, str>).
enum UnicodeIndex {
    /// This variant indicates that the prefix is located at the start of the token
    Start(usize),
    /// This variant indicates that the prefix is located at the end of the token
    End(usize),
}
impl UnicodeIndex {
    fn new<I: Iterator>(suffix: bool, unicode_iterator: I) -> Self {
        if !suffix {
            UnicodeIndex::Start(0)
        } else {
            UnicodeIndex::End(unicode_iterator.count())
        }
    }
    fn to_index(&self) -> usize {
        match self {
            Self::Start(start) => *start,
            Self::End(end) => *end,
        }
    }
}

impl<'a> InnerToken<'a> {
    /// Create InnerToken
    ///
    /// * `token`: str or String to parse the InnerToken from
    /// * `suffix`: Marker indicating if prefix is located at the end (when suffix is true) or the
    ///    end (when suffix is false) of the token
    /// * `delimiter`: Indicates the char used to separate the Prefix from the rest of the tag
    fn new(
        token: Cow<'a, str>,
        suffix: bool,
        delimiter: char,
    ) -> Result<Self, ParsingPrefixError<&'a str>> {
        let ref_iter = token.graphemes(true);
        let unicode_index = UnicodeIndex::new(suffix, ref_iter);
        let (char_index, prefix_char) = token
            .grapheme_indices(true)
            .nth(unicode_index.to_index())
            .ok_or(ParsingPrefixError("None"))?;
        let prefix = Prefix::try_from_with_static_error(prefix_char)?;
        let tag_before_strip = match unicode_index {
            UnicodeIndex::Start(_) => &token[char_index + 1..],
            UnicodeIndex::End(_) => &token[..char_index],
        };
        let tag = Cow::Owned(String::from(tag_before_strip.trim_matches(delimiter)));
        Ok(Self { token, prefix, tag })
    }

    #[inline]
    fn check_tag(&self, prev: &InnerToken, cond: &Tag) -> bool {
        match cond {
            Tag::Any => true,
            Tag::Same if prev.tag == self.tag => true,
            Tag::Diff if prev.tag != self.tag => true,
            _ => false,
        }
    }
    /// Check whether the prefix patterns are matched.
    ///
    /// * `prev`: Previous token
    /// * `patterns`: Patterns to match the token against
    fn check_patterns(
        &self,
        prev: &InnerToken,
        patterns_to_check: &[(Prefix, Prefix, Tag)],
    ) -> bool {
        for (prev_prefix, current_prefix, tag_cond) in patterns_to_check {
            if prev_prefix.are_the_same_or_contains_any(&prev.prefix)
                && current_prefix.are_the_same_or_contains_any(&self.prefix)
                && self.check_tag(prev, tag_cond)
            {
                return true;
            }
        }
        false
    }
}

#[derive(Debug, Clone, Copy)]
pub enum SchemeType {
    IOB1,
    IOE1,
    IOB2,
    IOE2,
    IOBES,
    BILOU,
}

#[derive(Debug, Clone, PartialEq)]
/// Encoutered an invalid token
pub struct InvalidToken(String);

impl Display for InvalidToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Invalid token: {}", self.0)
    }
}

impl Error for InvalidToken {}

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, PartialEq)]
enum Token<'a> {
    IOB1 { token: InnerToken<'a> },
    IOE1 { token: InnerToken<'a> },
    IOB2 { token: InnerToken<'a> },
    IOE2 { token: InnerToken<'a> },
    IOBES { token: InnerToken<'a> },
    BILOU { token: InnerToken<'a> },
}
// impl<'a> Default for &'a mut Token<'a> {
//     fn default() -> Self {
//         // let token: InnerToken = InnerToken::default();
//         Token::BILOU{token: InnerToken::default()}
//     }
// }

impl<'a> Default for Token<'a> {
    fn default() -> Self {
        Token::IOB1 {
            token: InnerToken::default(),
        }
    }
}

impl<'a> Token<'a> {
    const IOB1_ALLOWED_PREFIXES: [Prefix; 3] = [Prefix::I, Prefix::O, Prefix::B];
    const IOB1_START_PATTERNS: [(Prefix, Prefix, Tag); 5] = [
        (Prefix::O, Prefix::I, Tag::Any),
        (Prefix::I, Prefix::I, Tag::Diff),
        (Prefix::B, Prefix::I, Tag::Any),
        (Prefix::I, Prefix::B, Tag::Same),
        (Prefix::B, Prefix::B, Tag::Same),
    ];
    const IOB1_INSIDE_PATTERNS: [(Prefix, Prefix, Tag); 2] = [
        (Prefix::B, Prefix::I, Tag::Same),
        (Prefix::I, Prefix::I, Tag::Same),
    ];
    const IOB1_END_PATTERNS: [(Prefix, Prefix, Tag); 6] = [
        (Prefix::I, Prefix::I, Tag::Diff),
        (Prefix::I, Prefix::O, Tag::Any),
        (Prefix::I, Prefix::B, Tag::Any),
        (Prefix::B, Prefix::O, Tag::Any),
        (Prefix::B, Prefix::I, Tag::Diff),
        (Prefix::B, Prefix::B, Tag::Same),
    ];
    const IOE1_ALLOWED_PREFIXES: [Prefix; 3] = [Prefix::I, Prefix::O, Prefix::E];
    const IOE1_START_PATTERNS: [(Prefix, Prefix, Tag); 4] = [
        (Prefix::O, Prefix::I, Tag::Any),
        (Prefix::I, Prefix::I, Tag::Diff),
        (Prefix::E, Prefix::I, Tag::Any),
        (Prefix::E, Prefix::E, Tag::Same),
    ];
    const IOE1_INSIDE_PATTERNS: [(Prefix, Prefix, Tag); 2] = [
        (Prefix::I, Prefix::I, Tag::Same),
        (Prefix::I, Prefix::E, Tag::Same),
    ];
    const IOE1_END_PATTERNS: [(Prefix, Prefix, Tag); 5] = [
        (Prefix::I, Prefix::I, Tag::Diff),
        (Prefix::I, Prefix::O, Tag::Any),
        (Prefix::I, Prefix::E, Tag::Diff),
        (Prefix::E, Prefix::I, Tag::Same),
        (Prefix::E, Prefix::E, Tag::Same),
    ];

    const IOB2_ALLOWED_PREFIXES: [Prefix; 3] = [Prefix::I, Prefix::O, Prefix::B];
    const IOB2_START_PATTERNS: [(Prefix, Prefix, Tag); 1] = [(Prefix::Any, Prefix::B, Tag::Any)];
    const IOB2_INSIDE_PATTERNS: [(Prefix, Prefix, Tag); 2] = [
        (Prefix::B, Prefix::I, Tag::Same),
        (Prefix::I, Prefix::I, Tag::Same),
    ];
    const IOB2_END_PATTERNS: [(Prefix, Prefix, Tag); 6] = [
        (Prefix::I, Prefix::O, Tag::Any),
        (Prefix::I, Prefix::I, Tag::Diff),
        (Prefix::I, Prefix::B, Tag::Any),
        (Prefix::B, Prefix::O, Tag::Any),
        (Prefix::B, Prefix::I, Tag::Diff),
        (Prefix::B, Prefix::B, Tag::Any),
    ];
    const IOE2_ALLOWED_PREFIXES: [Prefix; 3] = [Prefix::I, Prefix::O, Prefix::E];
    const IOE2_START_PATTERNS: [(Prefix, Prefix, Tag); 6] = [
        (Prefix::O, Prefix::I, Tag::Any),
        (Prefix::O, Prefix::E, Tag::Any),
        (Prefix::E, Prefix::I, Tag::Any),
        (Prefix::E, Prefix::E, Tag::Any),
        (Prefix::I, Prefix::I, Tag::Diff),
        (Prefix::I, Prefix::E, Tag::Diff),
    ];
    const IOE2_INSIDE_PATTERNS: [(Prefix, Prefix, Tag); 2] = [
        (Prefix::I, Prefix::E, Tag::Same),
        (Prefix::I, Prefix::I, Tag::Same),
    ];
    const IOE2_END_PATTERNS: [(Prefix, Prefix, Tag); 1] = [(Prefix::E, Prefix::Any, Tag::Any)];

    const IOBES_ALLOWED_PREFIXES: [Prefix; 5] =
        [Prefix::I, Prefix::O, Prefix::E, Prefix::B, Prefix::S];
    const IOBES_START_PATTERNS: [(Prefix, Prefix, Tag); 4] = [
        (Prefix::B, Prefix::I, Tag::Same),
        (Prefix::B, Prefix::E, Tag::Same),
        (Prefix::I, Prefix::I, Tag::Same),
        (Prefix::I, Prefix::E, Tag::Same),
    ];
    const IOBES_INSIDE_PATTERNS: [(Prefix, Prefix, Tag); 2] = [
        (Prefix::S, Prefix::Any, Tag::Any),
        (Prefix::E, Prefix::Any, Tag::Any),
    ];
    const IOBES_END_PATTERNS: [(Prefix, Prefix, Tag); 2] = [
        (Prefix::S, Prefix::Any, Tag::Any),
        (Prefix::E, Prefix::Any, Tag::Any),
    ];

    const BILOU_ALLOWED_PREFIXES: [Prefix; 5] =
        [Prefix::I, Prefix::O, Prefix::U, Prefix::B, Prefix::O];
    const BILOU_START_PATTERNS: [(Prefix, Prefix, Tag); 2] = [
        (Prefix::Any, Prefix::B, Tag::Any),
        (Prefix::Any, Prefix::U, Tag::Any),
    ];
    const BILOU_INSIDE_PATTERNS: [(Prefix, Prefix, Tag); 4] = [
        (Prefix::B, Prefix::I, Tag::Same),
        (Prefix::B, Prefix::L, Tag::Same),
        (Prefix::I, Prefix::I, Tag::Same),
        (Prefix::I, Prefix::L, Tag::Same),
    ];
    const BILOU_END_PATTERNS: [(Prefix, Prefix, Tag); 2] = [
        (Prefix::U, Prefix::Any, Tag::Any),
        (Prefix::L, Prefix::Any, Tag::Any),
    ];
    fn allowed_prefixes(&'a self) -> &'static [Prefix] {
        match self {
            Self::IOB1 { .. } => &Self::IOB1_ALLOWED_PREFIXES,
            Self::IOE1 { .. } => &Self::IOE1_ALLOWED_PREFIXES,
            Self::IOB2 { .. } => &Self::IOB2_ALLOWED_PREFIXES,
            Self::IOE2 { .. } => &Self::IOE2_ALLOWED_PREFIXES,
            Self::IOBES { .. } => &Self::IOBES_ALLOWED_PREFIXES,
            Self::BILOU { .. } => &Self::BILOU_ALLOWED_PREFIXES,
        }
    }
    fn start_patterns(&'a self) -> &'static [(Prefix, Prefix, Tag)] {
        match self {
            Self::IOB1 { .. } => &Self::IOB1_START_PATTERNS,
            Self::IOE1 { .. } => &Self::IOE1_START_PATTERNS,
            Self::IOB2 { .. } => &Self::IOB2_START_PATTERNS,
            Self::IOE2 { .. } => &Self::IOE2_START_PATTERNS,
            Self::IOBES { .. } => &Self::IOBES_START_PATTERNS,
            Self::BILOU { .. } => &Self::BILOU_START_PATTERNS,
        }
    }
    fn inside_patterns(&'a self) -> &'static [(Prefix, Prefix, Tag)] {
        match self {
            Self::IOB1 { .. } => &Self::IOB1_INSIDE_PATTERNS,
            Self::IOE1 { .. } => &Self::IOE1_INSIDE_PATTERNS,
            Self::IOB2 { .. } => &Self::IOB2_INSIDE_PATTERNS,
            Self::IOE2 { .. } => &Self::IOE2_INSIDE_PATTERNS,
            Self::IOBES { .. } => &Self::IOBES_INSIDE_PATTERNS,
            Self::BILOU { .. } => &Self::BILOU_INSIDE_PATTERNS,
        }
    }
    fn end_patterns(&'a self) -> &'static [(Prefix, Prefix, Tag)] {
        match self {
            Self::IOB1 { .. } => &Self::IOB1_END_PATTERNS,
            Self::IOE1 { .. } => &Self::IOE1_END_PATTERNS,
            Self::IOB2 { .. } => &Self::IOB2_END_PATTERNS,
            Self::IOE2 { .. } => &Self::IOE2_END_PATTERNS,
            Self::IOBES { .. } => &Self::IOBES_END_PATTERNS,
            Self::BILOU { .. } => &Self::BILOU_END_PATTERNS,
        }
    }
    fn new(scheme: SchemeType, token: InnerToken<'a>) -> Self {
        match scheme {
            SchemeType::IOB1 => Token::IOB1 { token },
            SchemeType::IOB2 => Token::IOB2 { token },
            SchemeType::IOE1 => Token::IOE1 { token },
            SchemeType::IOE2 => Token::IOE2 { token },
            SchemeType::IOBES => Token::IOBES { token },
            SchemeType::BILOU => Token::BILOU { token },
        }
    }

    fn inner(&self) -> &InnerToken {
        match self {
            Self::IOE1 { token } => token,
            Self::IOE2 { token } => token,
            Self::IOB1 { token } => token,
            Self::IOB2 { token } => token,
            Self::BILOU { token } => token,
            Self::IOBES { token } => token,
        }
    }

    fn is_valid(&self) -> bool {
        self.allowed_prefixes().contains(&self.inner().prefix)
    }

    /// Check whether the current token is the start of chunk.
    fn is_start(&self, prev: &InnerToken) -> bool {
        match self {
            Self::IOB1 { token } => token.check_patterns(prev, self.start_patterns()),
            Self::IOB2 { token } => token.check_patterns(prev, self.start_patterns()),
            Self::IOE1 { token } => token.check_patterns(prev, self.start_patterns()),
            Self::IOE2 { token } => token.check_patterns(prev, self.start_patterns()),
            Self::IOBES { token } => token.check_patterns(prev, self.start_patterns()),
            Self::BILOU { token } => token.check_patterns(prev, self.start_patterns()),
        }
    }
    /// Check whether the current token is the inside of chunk.
    fn is_inside(&self, prev: &InnerToken) -> bool {
        match self {
            Self::IOB1 { token } => token.check_patterns(prev, self.inside_patterns()),
            Self::IOB2 { token } => token.check_patterns(prev, self.inside_patterns()),
            Self::IOE1 { token } => token.check_patterns(prev, self.inside_patterns()),
            Self::IOE2 { token } => token.check_patterns(prev, self.inside_patterns()),
            Self::IOBES { token } => token.check_patterns(prev, self.inside_patterns()),
            Self::BILOU { token } => token.check_patterns(prev, self.inside_patterns()),
        }
    }
    /// Check whether the *previous* token is the end of chunk.
    fn is_end(&self, prev: &InnerToken) -> bool {
        match self {
            Self::IOB1 { token } => token.check_patterns(prev, self.end_patterns()),
            Self::IOB2 { token } => token.check_patterns(prev, self.end_patterns()),
            Self::IOE1 { token } => token.check_patterns(prev, self.end_patterns()),
            Self::IOE2 { token } => token.check_patterns(prev, self.end_patterns()),
            Self::IOBES { token } => token.check_patterns(prev, self.end_patterns()),
            Self::BILOU { token } => token.check_patterns(prev, self.end_patterns()),
        }
    }
    fn take_tag(&mut self) -> Cow<'a, str> {
        match self {
            Self::IOB1 { token } => take(&mut token.tag),
            Self::IOE1 { token } => take(&mut token.tag),
            Self::IOB2 { token } => take(&mut token.tag),
            Self::IOE2 { token } => take(&mut token.tag),
            Self::IOBES { token } => take(&mut token.tag),
            Self::BILOU { token } => take(&mut token.tag),
        }
    }
}
/// This struct a struct capable of building efficiently the Tokens with a given outside_token.
/// This iterator avoids reallocation and keeps good ergonomic inside the `new` function of
/// `Tokens`.
struct ExtendedTokensIterator<'a> {
    outside_token: Token<'a>,
    tokens: Vec<Cow<'a, str>>,
    scheme: SchemeType,
    suffix: bool,
    delimiter: char,
    index: usize,
    /// Total length to iterate over. This length is equal to token.len()
    total_len: usize,
}
impl<'a> Iterator for ExtendedTokensIterator<'a> {
    type Item = Result<Token<'a>, ParsingPrefixError<&'a str>>;
    fn next(&mut self) -> Option<Self::Item> {
        // let ret: Option<Result<Token, ParsingPrefixError<&'a str>>>;
        let ret = match self.index.cmp(&self.total_len) {
            Ordering::Greater => None,
            Ordering::Equal => Some(Ok(take(&mut self.outside_token))),
            Ordering::Less => {
                let cow_str = unsafe { take(self.tokens.get_unchecked_mut(self.index)) };
                let inner_token = InnerToken::new(cow_str, self.suffix, self.delimiter);
                match inner_token {
                    Err(msg) => Some(Err(msg)),
                    Ok(res) => Some(Ok(Token::new(self.scheme, res))),
                }
            }
        };
        self.index += 1;
        ret
    }
}
impl<'a> ExtendedTokensIterator<'a> {
    fn new(
        outside_token: Token<'a>,
        tokens: Vec<Cow<'a, str>>,
        scheme: SchemeType,
        suffix: bool,
        delimiter: char,
    ) -> Self {
        let total_len = tokens.len();
        Self {
            outside_token,
            tokens,
            scheme,
            suffix,
            delimiter,
            index: 0,
            total_len,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Tokens<'a> {
    extended_tokens: Vec<Token<'a>>,
    sent_id: Option<usize>,
}
impl<'a> Tokens<'a> {
    fn new(
        tokens: Vec<Cow<'a, str>>,
        scheme: SchemeType,
        suffix: bool,
        delimiter: char,
        sent_id: Option<usize>,
    ) -> Result<Self, ParsingPrefixError<&'a str>> {
        // let inner_token_prefix =
        // let outside_token_inner_token = Token{token: Cow::Borrowed("O"), };
        let outside_token_inner = InnerToken::new(Cow::Borrowed("O"), suffix, delimiter)?;
        let outside_token = Token::new(scheme, outside_token_inner);
        let tokens_iter =
            ExtendedTokensIterator::new(outside_token, tokens, scheme, suffix, delimiter);
        let extended_tokens: Result<Vec<Token>, ParsingPrefixError<&str>> = tokens_iter.collect();
        match extended_tokens {
            Err(prefix_error) => Err(prefix_error),
            Ok(tokens) => Ok(Self {
                extended_tokens: tokens,
                sent_id,
            }),
        }
    }

    /// Returns the index + 1 of the last token inside the current chunk when given a `start` index and
    /// the previous token.
    ///
    /// * `start`: Indexing at which we are starting to look for a token not inside.
    /// * `prev`: Previous token. This token is necessary to know if the token at index `start` is
    ///    inside or not.
    fn forward(&self, start: usize, prev: &Token<'a>) -> usize {
        let slice_of_interest = &self.extended_tokens()[start..];
        let mut swap_token = prev;
        for (i, current_token) in slice_of_interest.iter().enumerate() {
            if current_token.is_inside(swap_token.inner()) {
                swap_token = current_token;
            } else {
                return i + start;
            }
        }
        &self.extended_tokens.len() - 2
    }

    /// This method returns a bool if the token at index `i` is *NOT*
    /// part of the same chunk as token at `i-1` or is not part of a
    /// chunk at all. Else, it returns false
    ///
    /// * `i`: Index of the token.
    fn is_end(&self, i: usize) -> bool {
        let token = &self.extended_tokens()[i];
        let prev = &self.extended_tokens()[i - 1];
        token.is_end(prev.inner())
    }

    fn extended_tokens(&'a self) -> &'a Vec<Token<'a>> {
        let res: &Vec<Token> = self.extended_tokens.as_ref();
        res
    }
}

/// Iterator and adaptor for iterating over the entities of a Tokens struct
///
/// * `index`: Index of the current iteration
/// * `current`: Current token
/// * `prev`:  Previous token
/// * `prev_prev`: Previous token of the previous token
struct EntitiesIterAdaptor<'a> {
    index: usize,
    tokens: RefCell<Tokens<'a>>,
    len: usize,
    // current: &'a mut Token<'a>,
    // prev: &'a mut Token<'a>,
}

// i = 0
// entities = []
// prev = self.outside_token
// while i < len(self.extended_tokens):
//     token = self.extended_tokens[i]
//     token.is_valid()
//     if token.is_start(prev):
//         end = self._forward(start=i + 1, prev=token)
//         if self._is_end(end):
//             entity = Entity(sent_id=self.sent_id, start=i, end=end, tag=token.tag)
//             entities.append(entity)
//         i = end
//     else:
//         i += 1
//     prev = self.extended_tokens[i - 1]
// return entities
impl<'a> Iterator for EntitiesIterAdaptor<'a> {
    type Item = Option<Result<Entity<'a>, InvalidToken>>;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let ret: Option<Option<Result<Entity<'a>, InvalidToken>>>;
        if self.index >= self.len - 1 {
            return None;
        }
        let mut_tokens = &self.tokens;
        let mut mut_tokens_ref = mut_tokens.borrow_mut();
        let (current_pre_ref_cell, prev) =
            unsafe { Self::take_out_pair(&mut mut_tokens_ref, self.index) };
        let current = RefCell::new(current_pre_ref_cell);
        let borrowed_current = current.borrow();
        let is_valid = borrowed_current.is_valid();
        if !is_valid {
            ret = Some(Some(Err(InvalidToken(
                borrowed_current.inner().token.to_string(),
            ))))
        } else if borrowed_current.is_start(prev.inner()) {
            drop(mut_tokens_ref);
            let end = mut_tokens
                .borrow()
                .forward(self.index + 1, &borrowed_current);
            if mut_tokens.borrow().is_end(end) {
                drop(borrowed_current);
                let tag = current.into_inner().take_tag();
                let entity = Entity {
                    sent_id: mut_tokens.borrow().sent_id,
                    start: self.index,
                    end,
                    tag,
                };
                self.index = end;
                ret = Some(Some(Ok(entity)));
            } else {
                self.index = end;
                ret = Some(None);
            }
        } else {
            self.index += 1;
            ret = Some(None);
        };
        ret
    }
}
impl<'a, 'b> EntitiesIterAdaptor<'a>
where
    'a: 'b,
{
    /// Takes out the current tokens and gets a reference to the
    /// previous tokens (in that order) when given an index. The index
    /// must be >= 0 and < tokens.len() or this function will result
    /// in UB. Calling this function with an already used index will
    /// result in default tokens returned. This functions behaves
    /// differently, depending on the value of the index to accomodate
    /// the `outside_token`, located at the end of the
    /// `extended_vector` vector. If `index` is 0, the previous token
    /// is the outside token of the extended tokens. Else, it takes
    /// the tokens at index `i` and `i-1`.
    ///
    /// SAFETY: The index must be >= 0 and <= tokens.len()-1, or this
    /// function will result in UB.
    ///
    /// * `tokens`: The tokens. The current and previous tokens are
    ///    extracted from its extended_tokens field.
    /// * `index`: Index specifying the current token. `index-1` is
    ///    used to take the previous token if index!=1.
    unsafe fn take_out_pair(
        tokens: &'b mut Tokens<'a>,
        index: usize,
    ) -> (Token<'a>, &'b Token<'a>) {
        if index == 0 {
            let index_of_outside_token = tokens.extended_tokens.len() - 1;
            let current_token = take(tokens.extended_tokens.get_unchecked_mut(0));
            let previous_token = tokens.extended_tokens.get_unchecked(index_of_outside_token);
            (current_token, previous_token)
        } else {
            let current_token = take(tokens.extended_tokens.get_unchecked_mut(index));
            let previous_token = tokens.extended_tokens.get_unchecked(index - 1);
            (current_token, previous_token)
        }
    }
    fn new(tokens: Tokens<'a>) -> Self {
        let len = tokens.extended_tokens.len();
        Self {
            index: 0,
            tokens: RefCell::new(tokens),
            len,
        }
    }
}

struct EntitiesIter<'a>(EntitiesIterAdaptor<'a>);

impl<'a> Iterator for EntitiesIter<'a> {
    type Item = Result<Entity<'a>, InvalidToken>;
    fn next(&mut self) -> Option<Self::Item> {
        let mut res: Option<Option<Result<Entity<'a>, InvalidToken>>> = self.0.next();
        // Removes the Some(None) cases
        while matches!(&res, Some(None)) {
            res = self.0.next();
        }
        // Could be None or Some(Some(..))
        match res {
            Some(Some(result_value)) => Some(result_value),
            None => None,
            Some(None) => unreachable!(),
        }
    }
}

impl<'a> EntitiesIter<'a> {
    fn new(tokens: Tokens<'a>) -> Self {
        let adaptor = EntitiesIterAdaptor::new(tokens);
        EntitiesIter(adaptor)
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Enum of errors wrapping the actual error structs.
pub enum ConversionError<S: AsRef<str>> {
    /// Invalid token encoutered when
    InvalidToken(InvalidToken),
    /// Could not parse the string into a `Prefix`
    ParsingPrefix(ParsingPrefixError<S>),
}

impl<S: AsRef<str>> From<InvalidToken> for ConversionError<S> {
    fn from(value: InvalidToken) -> Self {
        Self::InvalidToken(value)
    }
}

impl<S: AsRef<str>> From<ParsingPrefixError<S>> for ConversionError<S> {
    fn from(value: ParsingPrefixError<S>) -> Self {
        Self::ParsingPrefix(value)
    }
}

impl<S: AsRef<str>> Display for ConversionError<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidToken(it) => std::fmt::Display::fmt(&it, f),
            Self::ParsingPrefix(pp) => pp.fmt(f),
        }
    }
}

impl<S: AsRef<str> + Debug> Error for ConversionError<S> {}

#[derive(Debug, PartialEq, Clone)]
/// Entites are the unique tokens contained in a sequence. Entities can be built with the
/// TryFromVec trait. This trait allows to collect from a vec
pub struct Entities<'a>(Vec<Vec<Entity<'a>>>);

impl<'a> Deref for Entities<'a> {
    type Target = Vec<Vec<Entity<'a>>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<'a> DerefMut for Entities<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<'a> IntoIterator for Entities<'a> {
    type Item = Entity<'a>;
    type IntoIter = std::iter::Flatten<std::vec::IntoIter<Vec<Entity<'a>>>>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter().flatten()
    }
}

/// This trait mimics the TryFrom trait from the std lib. It is used
/// to *try* to build an Entities structure. It can fail if there is a
/// malformed token in `tokens`.
///
/// * `tokens`: Vector containing the raw tokens.
/// * `scheme`: The scheme type to use (ex: IOB2, BILOU, etc.). The
///    supported scheme are the variant of SchemeType.
/// * `suffix`: Set it to `true` if the Tag is located at the start of
///    the token and set it to `false` if the Tag is located at the
///    end of the token.
/// * `delimiter`: The character used separate the Tag from the Prefix
///    (ex: `I-PER`, where the tag is `PER` and the prefix is `I`)
/// * `sent_id`: An optional id.
pub(crate) trait TryFromVecStrict<'a, T: Into<&'a str>> {
    type Error: Error;
    fn try_from_vecs_strict(
        tokens: Vec<Vec<T>>,
        scheme: SchemeType,
        suffix: bool,
        delimiter: char,
        sent_id: Option<usize>,
    ) -> Result<Entities<'a>, Self::Error>;
    fn try_from_convertible<C: Into<Vec<Vec<T>>>>(
        tokens: C,
        scheme: SchemeType,
        suffix: bool,
        delimiter: char,
        sent_id: Option<usize>,
    ) -> Result<Entities<'a>, Self::Error> {
        let vec_of_vec_of_str: Vec<Vec<T>> = tokens.into();
        Self::try_from_vecs_strict(vec_of_vec_of_str, scheme, suffix, delimiter, sent_id)
    }
}

pub(crate) trait TryFromVecLenient<'a, T: Into<&'a str>> {
    type Error: Error;
    fn try_from_vecs_lenient();
}

impl<'a, T: Into<&'a str>> TryFromVecStrict<'a, T> for Entities<'a> {
    type Error = ConversionError<&'a str>;
    fn try_from_vecs_strict(
        vec_of_tokens_2d: Vec<Vec<T>>,
        scheme: SchemeType,
        suffix: bool,
        delimiter: char,
        sent_id: Option<usize>,
    ) -> Result<Entities<'a>, Self::Error> {
        let vec_of_tokens: Result<Vec<_>, ParsingPrefixError<&str>> = vec_of_tokens_2d
            .into_iter()
            .map(|v| v.into_iter().map(|x| Cow::from(x.into())).collect())
            .map(|v| Tokens::new(v, scheme, suffix, delimiter, sent_id))
            .collect();
        let entities: Result<Vec<Vec<Entity>>, InvalidToken> = match vec_of_tokens {
            Ok(vec_of_toks) => vec_of_toks
                .into_iter()
                .map(|t| EntitiesIter::new(t).collect())
                .collect(),
            Err(msg) => Err(ConversionError::from(msg))?,
        };
        Ok(Entities::new(entities?))
    }
}

impl<'a> Entities<'a> {
    /// Consumes the 2D array of vecs and builds the Entities.
    fn new(entities: Vec<Vec<Entity<'a>>>) -> Self {
        Entities(entities)
    }

    //TODO: Convert the return type to be an iterator. Avoids allocations
    #[inline(always)]
    /// Filters the entities for a given tag name and returns them in a HashSet.
    ///
    /// * `tag_name`: This variable is used to compare the tag of the
    ///   entity with. Only those whose tag is equal to a reference to
    ///   `tag_name` are added into the returned HashSet.
    pub fn filter<S: AsRef<str>>(&self, tag_name: S) -> AHashSet<&Entity> {
        let tag_name_ref = tag_name.as_ref();
        AHashSet::from_iter(
            self.iter()
                .flat_map(|v| v.iter())
                .filter(|e| e.tag == tag_name_ref),
        )
    }

    pub fn unique_tags(&self) -> AHashSet<&str> {
        AHashSet::from_iter(self.iter().flat_map(|v| v.iter()).map(|e| e.tag.borrow()))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_entities_try_from() {
        let vec_of_tokens = vec![
            vec!["B-PER", "I-PER", "O", "B-LOC"],
            vec![
                "B-GEO", "I-GEO", "O", "B-GEO", "O", "B-PER", "I-PER", "I-PER", "B-LOC",
            ],
        ];
        let entities =
            Entities::try_from_vecs_strict(vec_of_tokens, SchemeType::IOB2, false, '-', None)
                .unwrap();
        assert_eq!(
            entities.0,
            vec![
                vec![
                    Entity {
                        sent_id: None,
                        start: 0,
                        end: 2,
                        tag: Cow::from("PER")
                    },
                    Entity {
                        sent_id: None,
                        start: 3,
                        end: 4,
                        tag: Cow::from("LOC")
                    }
                ],
                vec![
                    Entity {
                        sent_id: None,
                        start: 0,
                        end: 2,
                        tag: Cow::from("GEO")
                    },
                    Entity {
                        sent_id: None,
                        start: 3,
                        end: 4,
                        tag: Cow::from("GEO")
                    },
                    Entity {
                        sent_id: None,
                        start: 5,
                        end: 8,
                        tag: Cow::from("PER")
                    },
                    Entity {
                        sent_id: None,
                        start: 8,
                        end: 9,
                        tag: Cow::from("LOC")
                    },
                ]
            ]
        );
    }

    #[test]
    fn test_entities_filter() {
        let tokens = build_tokens();
        println!("{:?}", tokens);
        let entities = build_entities();
        let expected = vec![
            Entity {
                sent_id: None,
                start: 0,
                end: 2,
                tag: Cow::Borrowed("PER"),
            },
            Entity {
                sent_id: None,
                start: 3,
                end: 4,
                tag: Cow::Borrowed("LOC"),
            },
        ];
        assert_eq!(entities, expected);
    }

    fn build_entities() -> Vec<Entity<'static>> {
        let tokens = build_tokens();
        let entities: Result<Vec<_>, InvalidToken> = EntitiesIter::new(tokens).collect();
        entities.unwrap()
    }

    #[test]
    fn test_entity_iter() {
        let tokens = build_tokens();
        println!("tokens: {:?}", tokens);
        let iter = EntitiesIter(EntitiesIterAdaptor::new(tokens.clone()));
        let wrapped_entities: Result<Vec<_>, InvalidToken> = iter.collect();
        let entities = wrapped_entities.unwrap();
        let expected_entities = vec![
            Entity {
                sent_id: None,
                start: 0,
                end: 2,
                tag: Cow::Borrowed("PER"),
            },
            Entity {
                sent_id: None,
                start: 3,
                end: 4,
                tag: Cow::Borrowed("LOC"),
            },
        ];
        assert_eq!(expected_entities, entities)
    }

    #[test]
    fn test_entity_adaptor_iterator() {
        let tokens = build_tokens();
        println!("tokens: {:?}", tokens);
        let mut iter = EntitiesIterAdaptor::new(tokens.clone());
        let first_entity = iter.next().unwrap();
        println!("first entity: {:?}", first_entity);
        assert!(first_entity.is_some());
        let second_entity = iter.next().unwrap();
        println!("second entity: {:?}", second_entity);
        assert!(second_entity.is_none());
        let third_entity = iter.next().unwrap();
        println!("third entity: {:?}", third_entity);
        assert!(third_entity.is_some());
        // let forth_entity = iter.next().unwrap();
        // println!("forth entity: {:?}", forth_entity);
        // assert!(forth_entity.is_none());
        let iteration_has_ended = iter.next().is_none();
        assert!(iteration_has_ended);
    }
    #[test]
    fn test_is_start() {
        let tokens: Tokens = build_tokens();
        dbg!(tokens.clone());
        let first_token = tokens.extended_tokens.first().unwrap();
        let second_token = tokens.extended_tokens.get(1).unwrap();
        assert!(first_token.is_start(second_token.inner()));
        let outside_token = tokens.extended_tokens.last().unwrap();
        assert!(first_token.is_start(outside_token.inner()));
    }
    #[test]
    fn test_tokens_is_end() {
        let tokens: Tokens = build_tokens();
        let is_end_of_chunk = tokens.is_end(2);
        dbg!(tokens.clone());
        // let first_non_outside_token = &tokens.extended_tokens.get(1).unwrap();
        // let second_non_outside_token = &tokens.extended_tokens.get(2).unwrap();
        assert!(is_end_of_chunk);
        let is_end_of_chunk = tokens.is_end(3);
        assert!(!is_end_of_chunk)
    }

    #[test]
    fn test_innertoken_is_end() {
        let tokens: Tokens = build_tokens();
        let first_non_outside_token = tokens.extended_tokens.first().unwrap();
        let second_non_outside_token = tokens.extended_tokens.get(1).unwrap();
        let third_non_outside_token = tokens.extended_tokens.get(2).unwrap();
        let is_end = second_non_outside_token.is_end(first_non_outside_token.inner());
        assert!(!is_end);
        let is_end = third_non_outside_token.is_end(first_non_outside_token.inner());
        assert!(is_end)
    }

    #[test]
    fn test_token_is_start() {
        let tokens = build_tokens();
        println!("{:?}", tokens);
        println!("{:?}", tokens.extended_tokens());
        let prev = tokens.extended_tokens().first().unwrap();
        let is_start = tokens
            .extended_tokens()
            .get(1)
            .unwrap()
            .is_start(prev.inner());
        assert!(!is_start)
    }
    #[test]
    fn test_forward_method() {
        let tokens = build_tokens();
        println!("{:?}", &tokens);
        let end = tokens.forward(1, tokens.extended_tokens.first().unwrap());
        let expected_end = 2;
        assert_eq!(end, expected_end)
    }
    #[test]
    fn test_new_tokens() {
        let tokens = build_tokens();
        println!("{:?}", tokens);
        assert_eq!(tokens.extended_tokens.len(), 5);
    }
    #[test]
    fn test_innertoken_new() {
        let token = Cow::from("B-PER");
        let suffix = false;
        let delimiter = '-';
        let inner_token = InnerToken::new(token, suffix, delimiter).unwrap();
        let expected_inner_token = InnerToken {
            token: Cow::Borrowed("B-PER"),
            prefix: Prefix::B,
            tag: Cow::Owned(String::from("PER")),
        };
        assert_eq!(inner_token, expected_inner_token)
    }
    #[test]
    fn test_unique_tags() {
        let sequences = vec![build_str_vec(), build_str_vec_diff()];
        let entities =
            Entities::try_from_vecs_strict(sequences, SchemeType::IOB2, false, '-', None).unwrap();
        let actual_unique_tags = entities.unique_tags();
        let expected_unique_tags: AHashSet<&str> = AHashSet::from_iter(vec!["PER", "LOC", "GEO"]);
        assert_eq!(actual_unique_tags, expected_unique_tags);
    }
    fn build_tokens() -> Tokens<'static> {
        let tokens = build_tokens_vec();
        let scheme = SchemeType::IOB2;
        let delimiter = '-';
        let suffix = false;
        Tokens::new(tokens, scheme, suffix, delimiter, None).unwrap()
    }
    fn build_tokens_vec() -> Vec<Cow<'static, str>> {
        vec![
            Cow::from("B-PER"),
            Cow::from("I-PER"),
            Cow::from("O"),
            Cow::from("B-LOC"),
        ]
    }
    fn build_str_vec() -> Vec<&'static str> {
        vec!["B-PER", "I-PER", "O", "B-LOC"]
    }
    fn build_str_vec_diff() -> Vec<&'static str> {
        vec![
            "B-GEO", "I-GEO", "O", "B-GEO", "O", "B-PER", "I-PER", "I-PER", "B-LOC",
        ]
    }
}
