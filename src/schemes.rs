/**
This modules gives the tooling necessary to parse a sequence of tokens into a list of entities.
*/
use ahash::AHashSet;
use enum_iterator::{all, Sequence};
use std::borrow::Borrow;
use std::cmp::Ordering;
use std::convert::TryFrom;
use std::error::Error;
use std::fmt::{Debug, Display};
use std::mem::take;
use std::ops::{Deref, DerefMut};
use std::slice::Iter;
use std::str::FromStr;
use std::{borrow::Cow, cell::RefCell};
use unicode_segmentation::UnicodeSegmentation;

/// An entity represent a named objet in named entity recognition (NER). It contains a start and an
/// end(i.e. at what index of the list does it starts and ends) and a tag, which the associated
/// entity (such as `LOC`, `NAME`, `PER`, etc.)
#[derive(Debug, Hash, Clone, PartialEq, Eq, PartialOrd)]
pub struct Entity<'a> {
    sent_id: Option<usize>,
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) tag: Cow<'a, str>,
}

impl<'a> Entity<'a> {
    pub(crate) fn new(sent_id: Option<usize>, start: usize, end: usize, tag: Cow<'a, str>) -> Self {
        Entity {
            sent_id,
            start,
            end,
            tag,
        }
    }
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

#[derive(Debug, PartialEq, Hash, Clone, Sequence, Eq)]
/// The inner prefix are the actual prefix that can be supplied by the user. All user prefix are of
/// length 1 (as a unicode char and regular char).
enum UserPrefix {
    I,
    O,
    B,
    E,
    S,
    U,
    L,
}

impl FromStr for UserPrefix {
    type Err = ParsingError<String>;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "I" => Ok(Self::I),
            "O" => Ok(Self::O),
            "B" => Ok(Self::B),
            "E" => Ok(Self::E),
            "S" => Ok(Self::S),
            "U" => Ok(Self::U),
            "L" => Ok(Self::L),
            _ => Err(ParsingError::PrefixError(String::from(s))),
        }
    }
}

impl TryFrom<char> for UserPrefix {
    type Error = ParsingError<String>;
    fn try_from(value: char) -> Result<Self, Self::Error> {
        match value {
            'I' => Ok(Self::I),
            'O' => Ok(Self::O),
            'B' => Ok(Self::B),
            'E' => Ok(Self::E),
            'S' => Ok(Self::S),
            'U' => Ok(Self::U),
            'L' => Ok(Self::L),
            _ => Err(ParsingError::PrefixError(String::from(value))),
        }
    }
}

impl UserPrefix {
    fn as_str(&self) -> &str {
        match self {
            UserPrefix::I => "I",
            UserPrefix::O => "O",
            UserPrefix::B => "B",
            UserPrefix::E => "E",
            UserPrefix::S => "S",
            UserPrefix::U => "U",
            UserPrefix::L => "L",
        }
    }

    fn len_in_bytes(&self) -> usize {
        self.clone().as_str().len()
    }
}

impl From<UserPrefix> for Prefix {
    fn from(value: UserPrefix) -> Self {
        match value {
            UserPrefix::I => Self::I,
            UserPrefix::O => Self::O,
            UserPrefix::B => Self::B,
            UserPrefix::E => Self::E,
            UserPrefix::S => Self::S,
            UserPrefix::U => Self::U,
            UserPrefix::L => Self::L,
        }
    }
}

#[derive(Debug, PartialEq, Hash, Clone, Sequence, Eq)]
/// Prefix represent an annotation specifying the place of a token in a chunk. For example, in
/// `IOB1`, the `I` prefix is used to indicate that the token is inside a NER. Prefix can only be a
/// single ascii character.
pub(crate) enum Prefix {
    I,
    O,
    B,
    E,
    S,
    U,
    L,
    /// The `ANY` prefix is more of a marker than a real prefix. It is not suppposed to be supplied
    /// by the user.
    Any,
}

impl Display for UserPrefix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Display for Prefix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
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
pub enum ParsingError<S: AsRef<str>> {
    PrefixError(S),
    EmptyToken,
}

impl<S: AsRef<str>> Display for ParsingError<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PrefixError(s) => {
                let content = s.as_ref();
                write!(
                    f,
                    "Could not parse the following string into a Prefix: {}",
                    content
                )
            }
            Self::EmptyToken => {
                write!(f, "Received an empty string/&str")
            }
        }
    }
}
impl<S: AsRef<str> + Error> Error for ParsingError<S> {}

//TODO: Remove this impl and use UserPrefix instead
impl FromStr for Prefix {
    type Err = ParsingError<String>;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::try_from_with_static_error(s)
    }
}

//TODO: Remove this impl and use UserPrefix instead
impl<'a> Prefix {
    fn try_from_with_static_error(value: &'a str) -> Result<Self, ParsingError<String>> {
        match value {
            "I" => Ok(Prefix::I),
            "O" => Ok(Prefix::O),
            "B" => Ok(Prefix::B),
            "E" => Ok(Prefix::E),
            "S" => Ok(Prefix::S),
            "U" => Ok(Prefix::U),
            "L" => Ok(Prefix::L),
            _ => Err(ParsingError::PrefixError(String::from(value))),
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
    /// The full token, such as `"B-PER"`, `"I-LOC"`, etc.
    token: Cow<'a, str>,
    /// The prefix, such as `B`, `I`, `O`, etc.
    prefix: UserPrefix,
    /// The tag, such as '"PER"', '"LOC"'
    tag: Cow<'a, str>,
}

impl<'a> Default for InnerToken<'a> {
    fn default() -> Self {
        InnerToken {
            token: Cow::Borrowed(""),
            prefix: UserPrefix::I,
            tag: Cow::Borrowed(""),
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, PartialOrd, Ord)]
/// This enum represents the positon of the Prefix in a token. It assumes the length (unicode char)
/// of the prefix is 1.
enum PrefixCharIndex {
    /// This variant indicates that the prefix is located at the start of the token
    Start(usize),
    /// This variant indicates that the prefix is located at the end of the token
    End(usize, usize),
}
impl PrefixCharIndex {
    fn try_new<'a>(suffix: bool, token: &'a Cow<'a, str>) -> Result<Self, ParsingError<String>> {
        if suffix {
            let count = token.as_ref().chars().count();
            if count == 0 {
                return Err(ParsingError::EmptyToken);
            }
            Ok(PrefixCharIndex::End(count - 1, count))
        } else {
            Ok(PrefixCharIndex::Start(0))
        }
    }
    fn to_index(&self) -> usize {
        match self {
            Self::Start(start) => *start,
            Self::End(end, _) => *end,
        }
    }
}

impl<'a> InnerToken<'a> {
    /// Create an InnerToken
    ///
    /// * `token`: str or String to parse the InnerToken from * `suffix`: Marker indicating if
    /// prefix is located at the end (when suffix is true) or the end (when suffix is false) of the
    /// token * `delimiter`: Indicates the char used to separate the Prefix from the rest of the
    /// tag
    fn try_new(
        token: Cow<'a, str>,
        suffix: bool,
        delimiter: char,
    ) -> Result<Self, ParsingError<String>> {
        //1. Identify where is the `Prefix` located
        let prefix_index_struct = PrefixCharIndex::try_new(suffix, &token)?;
        let prefix_char_index = prefix_index_struct.to_index();
        let char_res = token.chars().nth(prefix_char_index);
        // let (prefix_index, prefix_char) = token.char_indices().nth(prefix_char_index);
        // .ok_or();
        // NOTE: Might be better as an ok_or_else()
        let prefix = match char_res {
            Some(c) => UserPrefix::try_from(c)?,
            None => return Err(ParsingError::PrefixError(String::from(token))),
        };
        // .grapheme_indices(true)
        // .nth(unicode_index.to_index())
        // .ok_or()?;
        // let prefix = UserPrefix::try_from(prefix_char)?;
        let tag_before_strip = match prefix_index_struct {
            PrefixCharIndex::Start(_) => token.chars().skip(2).collect::<String>(), /* .collect::<&str>(), */
            PrefixCharIndex::End(_, count) if count >= 2 => token
                .chars()
                .enumerate()
                .take_while(|(i, _)| i < &(count - 2))
                .map(|(_, c)| c)
                .collect::<String>(),
            //count > 0 due to PrefixCharIndex::try_new
            PrefixCharIndex::End(_, count) if count < 2 => String::from(token), // PrefixCharIndex::End(_) => token.chars().rev().skip(2).rev().collect::<String>(),
        };
        // tag is mutable only to allow to reasign after

        //TODO: Strip in an intellligent way: probably use index to avoid trimming many matches.
        let mut tag = Cow::Owned(tag_before_strip);
        if tag == "" {
            tag = Cow::Borrowed("_");
        }
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
            if prev_prefix.are_the_same_or_contains_any(&Prefix::from(prev.prefix.clone()))
                && current_prefix.are_the_same_or_contains_any(&Prefix::from(self.prefix.clone()))
                && self.check_tag(prev, tag_cond)
            {
                return true;
            }
        }
        false
    }
}

#[derive(Debug, Clone, Copy, Sequence, Hash, Eq, PartialEq)]
/// Enumeration of the supported Schemes. They are use to indicate how we are supposed to parse and
/// chunk the different tokens.
pub enum SchemeType {
    IOB1,
    IOE1,
    IOB2,
    IOE2,
    IOBES,
    BILOU,
}

/// Leniently retrieves the entities from a sequence.
pub(crate) fn get_entities_lenient<'a>(
    sequence: &'a [Vec<&'a str>],
    suffix: bool,
    delimiter: char,
) -> Result<Entities<'a>, ParsingError<String>> {
    let mut res = vec![];
    for vec_of_chunks in sequence.iter() {
        let vec_of_entities: Result<Vec<_>, _> =
            LenientChunkIter::new(vec_of_chunks, suffix, delimiter).collect();
        res.push(vec_of_entities?)
    }
    Ok(Entities(res))
}

/// This wrapper around the content iterator appends a single `"O"` at the end of its inner
/// iterator.
struct InnerLenientChunkIter<'a> {
    content: Iter<'a, &'a str>,
    is_at_end: bool,
}
impl<'a> InnerLenientChunkIter<'a> {
    fn new(seq: &'a [&'a str]) -> Self {
        InnerLenientChunkIter {
            content: seq.iter(),
            is_at_end: false,
        }
    }
}
impl<'a> Iterator for InnerLenientChunkIter<'a> {
    type Item = &'a str;
    fn next(&mut self) -> Option<Self::Item> {
        let next_value = self.content.next();
        if next_value.is_none() {
            match self.is_at_end {
                true => None,
                false => {
                    self.is_at_end = true;
                    Some("O")
                }
            }
        } else {
            next_value.map(|v| &**v)
        }
    }
}

/// This struct iterates over a *single* sequence and returns the chunks associated with it.
struct LenientChunkIter<'a> {
    /// The content on which we are iterating
    inner: InnerLenientChunkIter<'a>,
    /// The prefix of the previous chunk (e.g. 'I')
    prev_prefix: UserPrefix,
    /// The type of the previous chunk (e.g. `"PER"`)
    prev_type: Option<Cow<'a, str>>,
    begin_offset: usize,
    suffix: bool,
    delimiter: char,
    index: usize,
}

impl<'a> LenientChunkIter<'a> {
    fn new(sequence: &'a [&'a str], suffix: bool, delimiter: char) -> Self {
        LenientChunkIter {
            inner: InnerLenientChunkIter::new(sequence),
            prev_type: None,
            prev_prefix: UserPrefix::O,
            begin_offset: 0,
            suffix,
            delimiter,
            index: 0,
        }
    }
}

impl<'a> Iterator for LenientChunkIter<'a> {
    type Item = Result<Entity<'a>, ParsingError<String>>;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let current_chunk = self.inner.next()?; // no more chunks. We are done
            let mut inner_token =
                match InnerToken::try_new(Cow::from(current_chunk), self.suffix, self.delimiter) {
                    Ok(v) => v,
                    Err(e) => {
                        self.index += 1;
                        return Some(Err(e));
                    }
                };
            let ret: Option<Self::Item>;
            if self.end_of_chunk(&inner_token.prefix, &inner_token.tag) {
                ret = Some(Ok(Entity::new(
                    None,
                    self.begin_offset,
                    self.index - 1,
                    take(&mut self.prev_type).unwrap(),
                )));
                self.prev_prefix = inner_token.prefix;
                self.prev_type = Some(inner_token.tag);
                self.index += 1;
                return ret;
            };
            if self.start_of_chunk(&inner_token.prefix, &inner_token.tag) {
                self.begin_offset = self.index;
            };
            self.prev_prefix = inner_token.prefix;
            self.prev_type = Some(take(&mut inner_token.tag));
            self.index += 1;
        }
    }
}
impl<'a> LenientChunkIter<'a> {
    //     tag -> prefix
    //     type -> classe
    ///     """Checks if a chunk ended between the previous and current word.
    fn end_of_chunk(&self, current_prefix: &UserPrefix, current_type: &Cow<'a, str>) -> bool {
        let wrapped_type = Some(current_type);
        // Cloning a prefix is very inexpensive
        match (self.prev_prefix.clone(), current_prefix) {
            (UserPrefix::E, _) => true,
            (UserPrefix::S, _) => true,
            (UserPrefix::B, UserPrefix::B) => true,
            (UserPrefix::B, UserPrefix::S) => true,
            (UserPrefix::B, UserPrefix::O) => true,
            (UserPrefix::I, UserPrefix::B) => true,
            (UserPrefix::I, UserPrefix::S) => true,
            (UserPrefix::I, UserPrefix::O) => true,
            (self_prefix, _) => {
                if !matches!(self_prefix, UserPrefix::O)
                    && &self.prev_type.as_ref() != &wrapped_type
                {
                    true
                } else {
                    false
                }
            }
        }
    }

    ///     """Checks if a chunk started between the previous and current word.
    fn start_of_chunk(&self, current_prefix: &UserPrefix, current_type: &Cow<'a, str>) -> bool {
        let wrapped_type = Some(current_type);
        match (self.prev_prefix.clone(), current_prefix) {
            // Cloning a prefix is very inexpensive
            (_, UserPrefix::B) => true,
            (_, UserPrefix::S) => true,
            (UserPrefix::E, UserPrefix::E) => true,
            (UserPrefix::E, UserPrefix::I) => true,
            (UserPrefix::S, UserPrefix::E) => true,
            (UserPrefix::S, UserPrefix::I) => true,
            (UserPrefix::O, UserPrefix::E) => true,
            (UserPrefix::O, UserPrefix::I) => true,
            (_, curr_prefix) => {
                if !matches!(curr_prefix, UserPrefix::O)
                    && &self.prev_type.as_ref() != &wrapped_type
                {
                    true
                } else {
                    false
                }
            }
        }
    }
}
//
//     if tag != 'O' and tag != '.' and prev_type != type_:
//         chunk_start = True
//
//     return chunk_start
#[derive(Debug, PartialEq, Eq)]
pub enum AutoDetectError {
    TooManySchemesParsed(AHashSet<SchemeType>),
    NoSchemeParsed,
}
impl Display for AutoDetectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooManySchemesParsed(surviving_schemes) => write!(f, "Too many schemes could be parsed and used from the input. It is therefore ambiguous. Detected schemes: {:?}", surviving_schemes),
            Self::NoSchemeParsed => write!(f, "No scheme could be parsed from the input"),
        }
    }
}

impl Error for AutoDetectError {}

/// Helper structure used to give the necessary information about the structure of the tokens when
/// auto-detecting the scheme.
struct AutoDetectConfig<'a> {
    tokens: &'a [Vec<&'a str>],
    delimiter: char,
    suffix: bool,
    lenient: bool,
}

/// We can try to auto-detect the SchemeType used. This would allow to simplify the interface of
/// the lib and to simplify the life of the user, who might not know what scheme they are using.
impl<'a> TryFrom<AutoDetectConfig<'a>> for SchemeType {
    type Error = AutoDetectError;
    fn try_from(value: AutoDetectConfig) -> Result<Self, Self::Error> {
        Self::try_auto_detect_scheme_by_parsing(&value)
    }
}

/// This impl block contains the logic of the auto-detect feature.
impl SchemeType {
    /// This function incurs a runtime cost but it is called only once.
    fn try_auto_detect_scheme_by_parsing(
        config: &AutoDetectConfig,
    ) -> Result<SchemeType, AutoDetectError> {
        let mut possible_schemes = Self::list_possible_schemes();
        for sequence in config.tokens.iter() {
            let possible_schemes_clone = possible_schemes.clone();
            if possible_schemes_clone.is_empty() {
                return Err(AutoDetectError::NoSchemeParsed);
            } else if possible_schemes_clone.len() == 1 {
                return Ok(possible_schemes_clone.into_iter().nth(0).unwrap());
            } else {
                for scheme in possible_schemes_clone.iter() {
                    let vec_of_tok: Vec<Vec<&str>> = vec![sequence.clone()];
                    let is_parsed = match config.lenient {
                        false => Entities::try_from_vecs_strict(
                            vec_of_tok,
                            *scheme,
                            config.suffix,
                            config.delimiter,
                            None,
                        )
                        .is_ok(),
                        true => get_entities_lenient(&vec_of_tok, config.suffix, config.delimiter)
                            .is_ok(),
                    };
                    if is_parsed {
                        continue;
                    } else {
                        //TODO: remove this runtime check (.unwrap() + then_some(()))
                        //small runtime check, making sure we are effectively removing schemes. Should
                        //probably removed after some testing.
                        possible_schemes.remove(scheme).then_some(()).unwrap();
                    }
                }
            }
        }
        return Err(AutoDetectError::TooManySchemesParsed(possible_schemes));
    }

    fn try_auto_detect_scheme_by_prefix(
        config: &AutoDetectConfig,
    ) -> Result<SchemeType, AutoDetectError> {
        todo!()
    }
    fn list_possible_schemes() -> AHashSet<SchemeType> {
        all::<SchemeType>().collect()
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Encoutered an invalid token when parsing the entities.
pub struct InvalidToken(String);

impl Display for InvalidToken {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
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
        self.allowed_prefixes()
            .contains(&Prefix::from(self.inner().prefix.clone()))
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

/// This struct is capable of building efficiently the Tokens with a given outside_token. This
/// iterator avoids reallocation and keeps good ergonomic inside the `new` function of `Tokens`.
/// The `outside_token` field is the *last* token generated by this struct when calling `.next()`.
/// This struct is used to parse the tokens into an easier to use structs called `Token`s. During
/// iteration, it returns as last token the `'O'` tag.
struct ExtendedTokensIterator<'a> {
    outside_token: Token<'a>,
    tokens: Vec<Cow<'a, str>>,
    scheme: SchemeType,
    suffix: bool,
    delimiter: char,
    index: usize,
    /// Total length to iterate over. *This length is equal to token.len()*
    total_len: usize,
}
impl<'a> Iterator for ExtendedTokensIterator<'a> {
    type Item = Result<Token<'a>, ParsingError<String>>;
    fn next(&mut self) -> Option<Self::Item> {
        let ret = match self.index.cmp(&self.total_len) {
            Ordering::Greater => None,
            Ordering::Equal => Some(Ok(take(&mut self.outside_token))),
            Ordering::Less => {
                let cow_str = unsafe { take(self.tokens.get_unchecked_mut(self.index)) };
                let inner_token = InnerToken::try_new(cow_str, self.suffix, self.delimiter);
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
/// Intermediary type used to build the entities of a Vec of cows. It is a wrapper around a Vec of
/// `Token` and allows us to parse itself.
struct Tokens<'a> {
    /// Extended tokens are the parsed list of token with an `O` token as first token.
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
    ) -> Result<Self, ParsingError<String>> {
        let outside_token_inner = InnerToken::try_new(Cow::Borrowed("O"), suffix, delimiter)?;
        let outside_token = Token::new(scheme, outside_token_inner);
        let tokens_iter =
            ExtendedTokensIterator::new(outside_token, tokens, scheme, suffix, delimiter);
        let extended_tokens: Result<Vec<Token>, ParsingError<String>> = tokens_iter.collect();
        match extended_tokens {
            Err(prefix_error) => Err(prefix_error),
            Ok(tokens) => Ok(Self {
                extended_tokens: tokens,
                sent_id,
            }),
        }
    }

    /// Returns the index + 1 of the last token inside the current chunk when given a `start` index and
    /// the previous token. It allows us to call `next = Tokens[start, self.forward(i, prev)]`>
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

/// Iterator and adaptor for iterating over the `Entities` of a Tokens struct.
///
/// * `index`: Index of the current iteration
/// * `current`: Current token
/// * `prev`:  Previous token
/// * `prev_prev`: Previous token of the previous token
struct EntitiesIterAdaptor<'a> {
    index: usize,
    tokens: RefCell<Tokens<'a>>,
    len: usize,
}

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
    /// must be `>= 0` and `< tokens.len()` or this function will result
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
            // The outside token is actually the last token, but is treated as the first one.
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

/// The EntitiesIter struct parses the `Tokens` into Entities. The heavy lifting is actually done
/// with the EntitiesIterAdaptor struct.
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
    ParsingPrefix(ParsingError<S>),
}

// impl ConversionError<&str> {
//     pub(crate) fn to_string(self) -> ConversionError<String> {
//         match self {
//             Self::InvalidToken(t) => Self::InvalidToken(t),
//             Self::ParsingPrefix(ParsingPrefixError(ref_str)) => {
//                 Self::ParsingPrefix(ParsingPrefixError(ref_str.to_string()))
//             }
//         }
//     }
// }

impl<S: AsRef<str>> From<InvalidToken> for ConversionError<S> {
    fn from(value: InvalidToken) -> Self {
        Self::InvalidToken(value)
    }
}

impl<S: AsRef<str>> From<ParsingError<S>> for ConversionError<S> {
    fn from(value: ParsingError<S>) -> Self {
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

// impl<'a> Entities<'a> {
//     pub(crate) impl
// }

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
}

impl<'a, T: Into<&'a str>> TryFromVecStrict<'a, T> for Entities<'a> {
    type Error = ConversionError<String>;
    fn try_from_vecs_strict(
        vec_of_tokens_2d: Vec<Vec<T>>,
        scheme: SchemeType,
        suffix: bool,
        delimiter: char,
        sent_id: Option<usize>,
    ) -> Result<Entities<'a>, Self::Error> {
        let vec_of_tokens: Result<Vec<_>, ParsingError<String>> = vec_of_tokens_2d
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
    pub(crate) fn new(entities: Vec<Vec<Entity<'a>>>) -> Self {
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

    use quickcheck::{self, TestResult};
    #[macro_use(quickcheck)]
    use quickcheck_macros::quickcheck as quickcheck_test;
    use super::*;

    impl<'a> Entity<'a> {
        pub fn as_tuple(&'a self) -> (Option<usize>, usize, usize, &'a str) {
            (self.sent_id, self.start, self.end, self.tag.borrow())
        }
    }

    impl quickcheck::Arbitrary for Prefix {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
            let mut choice_slice: Vec<Prefix> = all::<Prefix>().collect();
            // Removes the `ALL` prefix
            choice_slice.swap_remove(choice_slice.len() - 1);
            g.choose(choice_slice.as_ref()).unwrap().clone()
        }
    }

    impl quickcheck::Arbitrary for UserPrefix {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
            let mut choice_slice: Vec<UserPrefix> = all::<UserPrefix>().collect();
            // Removes the `ALL` prefix
            choice_slice.swap_remove(choice_slice.len() - 1);
            g.choose(choice_slice.as_ref()).unwrap().clone()
        }
    }

    #[quickcheck_test]
    fn propertie_test_new_token_len(
        chars: Vec<char>,
        prefix: UserPrefix,
        suffix: bool,
        delimiter: char,
    ) -> TestResult {
        let tag: String = chars.into_iter().collect();
        if tag.len() == 0 {
            return TestResult::discard();
        };
        let tag_len = tag.len();
        dbg!(tag_len);
        let token_string = if suffix {
            tag + &String::from(delimiter) + &prefix.to_string()
        } else {
            prefix.to_string() + &String::from(delimiter) + &tag
        };
        let token = Cow::from(token_string);
        dbg!(token.clone());
        let inner_token_res = InnerToken::try_new(token, suffix, delimiter);
        match inner_token_res {
            Err(err) => match err {
                ParsingError::PrefixError(s) => panic!("{}", ParsingError::PrefixError(s)),
                ParsingError::EmptyToken => TestResult::discard(),
            },
            Ok(inner_token) => {
                if inner_token.tag.len() == tag_len {
                    TestResult::passed()
                } else {
                    TestResult::failed()
                }
            }
        }
    }

    #[quickcheck_test]
    fn propertie_test_new_token_prefix(
        chars: Vec<char>,
        prefix: UserPrefix,
        suffix: bool,
        delimiter: char,
    ) -> TestResult {
        let tag: String = chars.into_iter().collect();
        if tag.len() == 0 {
            return TestResult::discard();
        };
        let token_string = if suffix {
            tag + &String::from(delimiter) + &prefix.to_string()
        } else {
            prefix.to_string() + &String::from(delimiter) + &tag
        };
        let token = Cow::from(token_string);
        dbg!(token.clone());
        let inner_token_res = InnerToken::try_new(token, suffix, delimiter);
        match inner_token_res {
            Err(err) => match err {
                ParsingError::PrefixError(s) => panic!("{}", ParsingError::PrefixError(s)),
                ParsingError::EmptyToken => TestResult::discard(),
            },
            Ok(inner_token) => {
                if inner_token.prefix == prefix {
                    TestResult::passed()
                } else {
                    TestResult::failed()
                }
            }
        }
    }

    #[test]
    fn test_empty_token_return_err() {
        let token = Cow::from("");
        let suffix = true;
        let delimiter = '-';
        let res = InnerToken::try_new(token, suffix, delimiter).is_err();
        assert!(res);
    }

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
    fn test_innertoken_new_with_suffix() {
        let tokens = vec![
            (Cow::from("PER-I"), Cow::from("PER"), UserPrefix::I),
            (Cow::from("PER-B"), Cow::from("PER"), UserPrefix::B),
            (Cow::from("LOC-I"), Cow::from("LOC"), UserPrefix::I),
            (Cow::from("O"), Cow::from("_"), UserPrefix::O),
        ];
        let suffix = true;
        let delimiter = '-';
        for (i, (token, tag, prefix)) in tokens.into_iter().enumerate() {
            let inner_token = InnerToken::try_new(token.clone(), suffix, delimiter).unwrap();
            let expected_inner_token = InnerToken { token, prefix, tag };
            dbg!(i);
            assert_eq!(inner_token, expected_inner_token)
        }
    }
    #[test]
    fn test_innertoken_new() {
        let tokens = vec![
            (Cow::from("I-PER"), Cow::from("PER"), UserPrefix::I),
            (Cow::from("B-PER"), Cow::from("PER"), UserPrefix::B),
            (Cow::from("I-LOC"), Cow::from("LOC"), UserPrefix::I),
            (Cow::from("O"), Cow::from("_"), UserPrefix::O),
        ];
        let suffix = false;
        let delimiter = '-';
        for (i, (token, tag, prefix)) in tokens.into_iter().enumerate() {
            let inner_token = InnerToken::try_new(token.clone(), suffix, delimiter).unwrap();
            let expected_inner_token = InnerToken { token, prefix, tag };
            dbg!(i);
            assert_eq!(inner_token, expected_inner_token)
        }
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

    #[test]
    fn test_auto_detect_scheme() {
        let inputs = vec![build_str_vec_diff(), build_str_vec()];
        let config = AutoDetectConfig {
            tokens: &inputs,
            delimiter: '-',
            suffix: false,
            lenient: false,
        };
        let actual = SchemeType::try_auto_detect_scheme_by_parsing(&config).unwrap();
        let expected = SchemeType::IOB2;
        assert_eq!(actual, expected)
    }

    #[test]
    fn test_get_entities_lenient() {
        let seq = vec![vec!["B-PER", "I-PER", "O", "B-LOC"]];
        let actual = get_entities_lenient(seq.as_ref(), false, '-').unwrap();
        let entities = vec![vec![
            Entity::new(None, 0, 1, Cow::from("PER")),
            Entity::new(None, 3, 3, Cow::from("LOC")),
        ]];
        let expected = Entities::new(entities);
        assert_eq!(actual, expected)
    }

    #[allow(non_snake_case)]
    #[test]
    fn test_LenientChunkIterator() {
        let tokens = build_str_vec();
        let iter = LenientChunkIter::new(tokens.as_ref(), false, '-');
        let actual = iter.collect::<Vec<_>>();
        let expected: Vec<Result<Entity, ParsingError<String>>> = vec![
            Ok(Entity::new(None, 0, 1, Cow::Borrowed("PER"))),
            Ok(Entity::new(None, 3, 3, Cow::Borrowed("LOC"))),
        ];
        assert_eq!(actual, expected)
    }

    #[test]
    fn test_get_entities() {
        let seq = vec![vec![
            "O", "O", "O", "B-MISC", "I-MISC", "I-MISC", "O", "B-PER", "I-PER",
        ]];
        let binding = seq.clone();
        let binding2 = get_entities_lenient(binding.as_ref(), false, '-').unwrap();
        let actual = binding2
            .0
            .iter()
            .flat_map(|v| v.iter())
            .map(|e| e.as_tuple())
            .collect::<Vec<_>>();
        let expected: Vec<(Option<usize>, usize, usize, &str)> =
            vec![(None, 3, 5, "MISC"), (None, 7, 8, "PER")];
        assert_eq!(expected, actual)
    }

    #[test]
    fn test_get_entities_with_suffix() {
        let seq = vec![vec![
            "O", "O", "O", "MISC-B", "MISC-I", "MISC-I", "O", "PER-B", "PER-I",
        ]];
        let binding = seq.clone();
        let binding2 = get_entities_lenient(binding.as_ref(), true, '-').unwrap();
        let actual = binding2
            .0
            .iter()
            .flat_map(|v| v.iter())
            .map(|e| e.as_tuple())
            .collect::<Vec<_>>();
        let expected: Vec<(Option<usize>, usize, usize, &str)> =
            vec![(None, 3, 5, "MISC"), (None, 7, 8, "PER")];
        assert_eq!(expected, actual)
    }

    #[test]
    /// This function tests an implicit invariant used when building `Tokens` and `UnicodeIndex`:
    /// The prefix must be composed of a single unicode character.
    fn test_prefix_unicode_length() {
        let all_prefixes: Vec<_> = all::<UserPrefix>().collect();
        for p in all_prefixes {
            let len = p.to_string().graphemes(true).count();
            assert_eq!(1, len);
        }
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
