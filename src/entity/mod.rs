use crate::entity::schemes::{InnerToken, Token, UserPrefix};
use ahash::AHashSet;
use std::{
    borrow::{Borrow, Cow},
    cell::RefCell,
    cmp::Ordering,
    error::Error,
    fmt::{Debug, Display},
    mem::take,
    ops::{Deref, DerefMut},
    slice::Iter,
};

mod autodetect;
mod schemes;

// Re-exporting
pub use schemes::{InvalidToken, ParsingError, SchemeType};

/// An entity represent a named objet in named entity recognition (NER). It contains a start and an
/// end(i.e. at what index of the list does it starts and ends) and a tag, which the associated
/// entity (such as `LOC`, `NAME`, `PER`, etc.)
#[derive(Debug, Hash, Clone, PartialEq, Eq, PartialOrd)]
pub struct Entity<'a> {
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) tag: Cow<'a, str>,
}

impl<'a> Entity<'a> {
    pub(crate) fn new(start: usize, end: usize, tag: Cow<'a, str>) -> Self {
        Entity { start, end, tag }
    }
}

impl<'a> Display for Entity<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {}, {})", self.tag, self.start, self.end)
    }
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
}
impl<'a> Tokens<'a> {
    fn new(
        tokens: Vec<Cow<'a, str>>,
        scheme: SchemeType,
        suffix: bool,
        delimiter: char,
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
pub(crate) trait TryFromVecStrict<'a, T: Into<&'a str>> {
    type Error: Error;
    fn try_from_vecs_strict(
        tokens: Vec<Vec<T>>,
        scheme: SchemeType,
        suffix: bool,
        delimiter: char,
    ) -> Result<Entities<'a>, Self::Error>;
}

impl<'a, T: Into<&'a str>> TryFromVecStrict<'a, T> for Entities<'a> {
    type Error = ConversionError<String>;
    fn try_from_vecs_strict(
        vec_of_tokens_2d: Vec<Vec<T>>,
        scheme: SchemeType,
        suffix: bool,
        delimiter: char,
    ) -> Result<Entities<'a>, Self::Error> {
        let vec_of_tokens: Result<Vec<_>, ParsingError<String>> = vec_of_tokens_2d
            .into_iter()
            .map(|v| v.into_iter().map(|x| Cow::from(x.into())).collect())
            .map(|v| Tokens::new(v, scheme, suffix, delimiter))
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

    /// Filters the entities for a given tag name and return the number of entities..
    ///
    /// * `tag_name`: This variable is used to compare the tag of the
    ///   entity with. Only those whose tag is equal to a reference to
    ///   `tag_name` are added into the returned HashSet.
    pub fn filter_count<S: AsRef<str>>(&self, tag_name: S) -> usize {
        let tag_name_ref = tag_name.as_ref();
        self.iter()
            .flat_map(|v| v.iter())
            .filter(|e| e.tag == tag_name_ref)
            .count()
    }

    pub fn unique_tags(&self) -> AHashSet<&str> {
        AHashSet::from_iter(self.iter().flat_map(|v| v.iter()).map(|e| e.tag.borrow()))
    }
}

#[cfg(test)]
pub(super) mod tests {

    use super::*;
    use enum_iterator::{all, Sequence};
    use quickcheck::{self, TestResult};

    impl<'a> Entity<'a> {
        pub fn as_tuple(&'a self) -> (usize, usize, &'a str) {
            (self.start, self.end, self.tag.borrow())
        }
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
            Entities::try_from_vecs_strict(vec_of_tokens, SchemeType::IOB2, false, '-').unwrap();
        assert_eq!(
            entities.0,
            vec![
                vec![
                    Entity {
                        start: 0,
                        end: 2,
                        tag: Cow::from("PER")
                    },
                    Entity {
                        start: 3,
                        end: 4,
                        tag: Cow::from("LOC")
                    }
                ],
                vec![
                    Entity {
                        start: 0,
                        end: 2,
                        tag: Cow::from("GEO")
                    },
                    Entity {
                        start: 3,
                        end: 4,
                        tag: Cow::from("GEO")
                    },
                    Entity {
                        start: 5,
                        end: 8,
                        tag: Cow::from("PER")
                    },
                    Entity {
                        start: 8,
                        end: 9,
                        tag: Cow::from("LOC")
                    },
                ]
            ]
        );
    }

    #[derive(Debug, PartialEq, Hash, Clone, Sequence, Eq)]
    pub(crate) enum TokensToTest {
        BPER,
        BGEO,
        BLOC,
        O,
    }
    impl From<TokensToTest> for &str {
        fn from(value: TokensToTest) -> Self {
            match value {
                TokensToTest::BPER => "B-PER",
                TokensToTest::BLOC => "B-LOC",
                TokensToTest::BGEO => "B-GEO",
                TokensToTest::O => "O",
            }
        }
    }
    impl quickcheck::Arbitrary for TokensToTest {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
            let mut choice_slice: Vec<TokensToTest> = all::<TokensToTest>().collect();
            // Removes the `ALL` prefix
            choice_slice.swap_remove(choice_slice.len() - 1);
            g.choose(choice_slice.as_ref()).unwrap().clone()
        }
    }

    #[test]
    fn test_propertie_entities_try_from() {
        #[allow(non_snake_case)]
        fn propertie_entities_try_from_vecs_strict_IO_only(
            tokens: Vec<Vec<TokensToTest>>,
        ) -> TestResult {
            let tok = tokens;
            let entities =
                Entities::try_from_vecs_strict(tok, SchemeType::IOB2, false, '-').unwrap();
            for entity in entities {
                let diff = entity.end - entity.start;
                if diff != 1 {
                    return TestResult::failed();
                };
            }
            TestResult::passed()
        }
        let mut qc = quickcheck::QuickCheck::new().tests(2000);
        qc.quickcheck(
            propertie_entities_try_from_vecs_strict_IO_only
                as fn(Vec<Vec<TokensToTest>>) -> TestResult,
        )
    }

    #[test]
    fn test_entities_filter() {
        let tokens = build_tokens();
        println!("{:?}", tokens);
        let entities = build_entities();
        let expected = vec![
            Entity {
                start: 0,
                end: 2,
                tag: Cow::Borrowed("PER"),
            },
            Entity {
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
                start: 0,
                end: 2,
                tag: Cow::Borrowed("PER"),
            },
            Entity {
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
    fn test_unique_tags() {
        let sequences = vec![build_str_vec(), build_str_vec_diff()];
        let entities =
            Entities::try_from_vecs_strict(sequences, SchemeType::IOB2, false, '-').unwrap();
        let actual_unique_tags = entities.unique_tags();
        let expected_unique_tags: AHashSet<&str> = AHashSet::from_iter(vec!["PER", "LOC", "GEO"]);
        assert_eq!(actual_unique_tags, expected_unique_tags);
    }

    #[test]
    fn test_get_entities_lenient() {
        let seq = vec![vec!["B-PER", "I-PER", "O", "B-LOC"]];
        let actual = get_entities_lenient(seq.as_ref(), false, '-').unwrap();
        let entities = vec![vec![
            Entity::new(0, 1, Cow::from("PER")),
            Entity::new(3, 3, Cow::from("LOC")),
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
            Ok(Entity::new(0, 1, Cow::Borrowed("PER"))),
            Ok(Entity::new(3, 3, Cow::Borrowed("LOC"))),
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
        let expected: Vec<(usize, usize, &str)> = vec![(3, 5, "MISC"), (7, 8, "PER")];
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
        let expected: Vec<(usize, usize, &str)> = vec![(3, 5, "MISC"), (7, 8, "PER")];
        assert_eq!(expected, actual)
    }

    fn build_tokens() -> Tokens<'static> {
        let tokens = build_tokens_vec();
        let scheme = SchemeType::IOB2;
        let delimiter = '-';
        let suffix = false;
        Tokens::new(tokens, scheme, suffix, delimiter).unwrap()
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
