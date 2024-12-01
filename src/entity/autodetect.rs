use crate::entity::{
    get_entities_lenient, schemes::SchemeType, Entities, InnerToken, TryFromVecStrict, UserPrefix,
};
use ahash::AHashSet;
use enum_iterator::all;
use std::{borrow::Cow, error::Error, fmt::Display};

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

fn auto_detect(sequences: &[Vec<&str>], suffix: bool, delimiter: char) -> SchemeType {
    let mut prefixes: AHashSet<UserPrefix> = AHashSet::default();
    for tokens in sequences {
        for token in tokens {
            let tok = InnerToken::try_new(Cow::from(*token), suffix, delimiter);
            match tok {
                Ok(p) => prefixes.insert(p.prefix),
                Err(e) => continue,
            };
        }
    }
    if prefixes.
}
//     if prefixes in allowed_iob2_prefixes:
//         return IOB2
//     elif prefixes in allowed_ioe2_prefixes:
//         return IOE2
//     elif prefixes in allowed_iobes_prefixes:
//         return IOBES
//     elif prefixes in allowed_bilou_prefixes:
//         return BILOU
//     else:
//         raise ValueError(error_message.format(prefixes))

#[cfg(test)]
mod test {
    use super::*;
    use crate::SchemeType;

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

    fn build_str_vec() -> Vec<&'static str> {
        vec!["B-PER", "I-PER", "O", "B-LOC"]
    }
    fn build_str_vec_diff() -> Vec<&'static str> {
        vec![
            "B-GEO", "I-GEO", "O", "B-GEO", "O", "B-PER", "I-PER", "I-PER", "B-LOC",
        ]
    }
}
