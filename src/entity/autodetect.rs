use crate::entity::{schemes::SchemeType, InnerToken, UserPrefix};
use ahash::AHashSet;
use std::{ error::Error, fmt::Display, sync::LazyLock};

static ALLOWED_IOB2_PREFIXES: LazyLock<[AHashSet<UserPrefix>; 4]> = LazyLock::new(|| {
    [
        AHashSet::from([UserPrefix::I, UserPrefix::O, UserPrefix::B]),
        AHashSet::from([UserPrefix::I, UserPrefix::B]),
        AHashSet::from([UserPrefix::B, UserPrefix::O]),
        AHashSet::from([UserPrefix::B]),
    ]
});

static ALLOWED_IOE2_PREFIXES: LazyLock<[AHashSet<UserPrefix>; 4]> = LazyLock::new(|| {
    [
        AHashSet::from([UserPrefix::I, UserPrefix::O, UserPrefix::E]),
        AHashSet::from([UserPrefix::I, UserPrefix::E]),
        AHashSet::from([UserPrefix::E, UserPrefix::O]),
        AHashSet::from([UserPrefix::E]),
    ]
});
static ALLOWED_IOBES_PREFIXES: LazyLock<[AHashSet<UserPrefix>; 9]> = LazyLock::new(|| {
    [
        AHashSet::from([
            UserPrefix::I,
            UserPrefix::O,
            UserPrefix::B,
            UserPrefix::E,
            UserPrefix::S,
        ]),
        AHashSet::from([UserPrefix::I, UserPrefix::B, UserPrefix::E, UserPrefix::S]),
        AHashSet::from([UserPrefix::I, UserPrefix::O, UserPrefix::B, UserPrefix::E]),
        AHashSet::from([UserPrefix::O, UserPrefix::B, UserPrefix::E, UserPrefix::S]),
        AHashSet::from([UserPrefix::I, UserPrefix::B, UserPrefix::E]),
        AHashSet::from([UserPrefix::B, UserPrefix::E, UserPrefix::S]),
        AHashSet::from([UserPrefix::O, UserPrefix::B, UserPrefix::E]),
        AHashSet::from([UserPrefix::B, UserPrefix::E]),
        AHashSet::from([UserPrefix::S]),
    ]
});

static ALLOWED_BILOU_PREFIXES: LazyLock<[AHashSet<UserPrefix>; 9]> = LazyLock::new(|| {
    [
        AHashSet::from([
            UserPrefix::I,
            UserPrefix::O,
            UserPrefix::B,
            UserPrefix::L,
            UserPrefix::U,
        ]),
        AHashSet::from([UserPrefix::I, UserPrefix::B, UserPrefix::L, UserPrefix::U]),
        AHashSet::from([UserPrefix::I, UserPrefix::O, UserPrefix::B, UserPrefix::L]),
        AHashSet::from([UserPrefix::O, UserPrefix::B, UserPrefix::L, UserPrefix::U]),
        AHashSet::from([UserPrefix::I, UserPrefix::B, UserPrefix::L]),
        AHashSet::from([UserPrefix::B, UserPrefix::L, UserPrefix::U]),
        AHashSet::from([UserPrefix::O, UserPrefix::B, UserPrefix::L]),
        AHashSet::from([UserPrefix::B, UserPrefix::L]),
        AHashSet::from([UserPrefix::U]),
    ]
});
// allowed_bilou_prefixes = [
//     {Prefix.I, Prefix.O, Prefix.B, Prefix.L, Prefix.U},
//     {Prefix.I, Prefix.B, Prefix.L, Prefix.U},
//     {Prefix.I, Prefix.O, Prefix.B, Prefix.L},
//     {Prefix.O, Prefix.B, Prefix.L, Prefix.U},
//     {Prefix.I, Prefix.B, Prefix.L},
//     {Prefix.B, Prefix.L, Prefix.U},
//     {Prefix.O, Prefix.B, Prefix.L},
//     {Prefix.B, Prefix.L},
//     {Prefix.U}
// ]

#[derive(Debug, PartialEq, Eq)]
pub enum AutoDetectError {
    TooManySchemesParsed(AHashSet<SchemeType>),
    NoSchemeParsed,
    UnsupportedScheme,
}
impl Display for AutoDetectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooManySchemesParsed(surviving_schemes) => write!(f, "Too many schemes could be parsed and used from the input. It is therefore ambiguous. Detected schemes: {:?}", surviving_schemes),
            Self::NoSchemeParsed => write!(f, "No scheme could be parsed from the input"),
            Self::UnsupportedScheme => write!(f, "Could not detect the scheme. Only IOB2, IOE2, BILOU and IOBES can be auto-detected")
        }
    }
}

impl Error for AutoDetectError {}

/// Helper structure used to give the necessary information about the structure of the tokens when
/// auto-detecting the scheme.
struct AutoDetectScheme<'a> {
    tokens: &'a [Vec<&'a str>],
    suffix: bool,
}

/// We can try to auto-detect the SchemeType used. This would allow to simplify the interface of
/// the lib and to simplify the life of the user, who might not know what scheme they are using.
impl TryFrom<AutoDetectScheme<'_>> for SchemeType {
    type Error = AutoDetectError;
    fn try_from(value: AutoDetectScheme) -> Result<Self, Self::Error> {
        Self::try_auto_detect_by_prefix(&value).ok_or(AutoDetectError::UnsupportedScheme)
    }
}

/// This impl block contains the logic of the auto-detect feature.
impl SchemeType {
    /// auto_detect supports the following schemes:
    /// - IOB2
    /// - IOE2
    /// - IOBES
    /// - BILOU
    fn try_auto_detect_by_prefix(config: &AutoDetectScheme) -> Option<SchemeType> {
        let sequences = config.tokens;
        let suffix = config.suffix;
        let mut prefixes: AHashSet<UserPrefix> = AHashSet::default();
        for tokens in sequences {
            for token in tokens {
                let tok = InnerToken::try_new(token, suffix, );
                match tok {
                    Ok(p) => prefixes.insert(p.prefix),
                    Err(_) => continue,
                };
            }
        }
        if ALLOWED_IOB2_PREFIXES.contains(&prefixes) {
            return Some(SchemeType::IOB2);
        } else if ALLOWED_IOE2_PREFIXES.contains(&prefixes) {
            return Some(SchemeType::IOE2);
        } else if ALLOWED_BILOU_PREFIXES.contains(&prefixes) {
            return Some(SchemeType::BILOU);
        } else if ALLOWED_IOBES_PREFIXES.contains(&prefixes) {
            return Some(SchemeType::IOBES);
        };
        None
    }
}
#[cfg(test)]
mod test {
    use super::*;
    use crate::SchemeType;

    #[test]
    fn test_auto_detect_scheme_by_prefix() {
        let inputs = vec![build_str_vec_diff(), build_str_vec()];
        let config = AutoDetectScheme {
            tokens: &inputs,
            suffix: false,
        };
        let actual = SchemeType::try_auto_detect_by_prefix(&config).unwrap();
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
