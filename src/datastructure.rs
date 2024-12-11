use std::borrow::Cow;
use std::slice::{Iter, IterMut};

/// Custom datastructure built for reducing cache misses.
#[derive(Debug, Eq, PartialEq, PartialOrd, Ord, Hash, Clone)]
pub(crate) struct TokenVecs<T> {
    pub(crate) tokens: Box<[T]>,
    pub(crate) indices: Box<[usize]>,
}

impl<'a> TokenVecs<Cow<'a, str>> {
    fn new(vecs: Vec<Vec<&'a str>>) -> Self {
        Self::from(vecs)
    }
    pub(crate) fn len(&self) -> usize {
        self.tokens.len()
    }
}

impl<'a, T> TokenVecs<T> {
    pub(crate) fn from_map<F, Y>(value: Vec<Vec<Y>>, f: F) -> Self
    where
        F: Fn(Y) -> T,
    {
        let length: usize = value.iter().map(|v| v.len()).sum();
        let indices_length = value.len();
        let mut flattened = Vec::with_capacity(length);
        let mut indices = Vec::with_capacity(indices_length);
        let mut current_indice = 0;
        indices.push(0);
        for vec in value.into_iter() {
            let mut count = 0;
            for s in vec {
                current_indice += 1;
                count += 1;
                flattened.push(f(s));
            }
            if count == 0 {
                indices.push(current_indice);
            }
        }
        let tokens = flattened.into_boxed_slice();
        let indices_boxed = indices.into_boxed_slice();
        Self {
            tokens,
            indices: indices_boxed,
        }
    }
}

impl<'a, T> From<Vec<Vec<T>>> for TokenVecs<T> {
    fn from(value: Vec<Vec<T>>) -> Self {
        let length: usize = value.iter().map(|v| v.len()).sum();
        let indices_length = value.len();
        let mut flattened = Vec::with_capacity(length);
        let mut indices = Vec::with_capacity(indices_length);
        let mut current_indice = 0;
        indices.push(0);
        for vec in value.into_iter() {
            let mut count = 0;
            for s in vec {
                current_indice += 1;
                count += 1;
                flattened.push(s);
            }
            if count == 0 {
                indices.push(current_indice);
            }
        }
        let tokens = flattened.into_boxed_slice();
        let indices_boxed = indices.into_boxed_slice();
        Self {
            tokens,
            indices: indices_boxed,
        }
    }
}

impl<'a> From<Vec<Vec<&'a str>>> for TokenVecs<Cow<'a, str>> {
    fn from(value: Vec<Vec<&'a str>>) -> Self {
        let length: usize = value.iter().map(|v| v.len()).sum();
        let indices_length = value.len();
        let mut flattened = Vec::with_capacity(length);
        let mut indices = Vec::with_capacity(indices_length);
        let mut current_indice = 0;
        indices.push(0);
        for vec in value.into_iter() {
            for s in &vec {
                current_indice += 1;
                flattened.push(Cow::from(*s));
            }
            if !vec.is_empty() {
                indices.push(current_indice);
            }
        }
        let tokens = flattened.into_boxed_slice();
        let indices_boxed = indices.into_boxed_slice();
        Self {
            tokens,
            indices: indices_boxed,
        }
    }
}

impl<'a, T> TokenVecs<T> {
    pub(crate) fn iter(&'a self) -> Iter<'a, T> {
        self.tokens.iter()
    }
    pub(crate) fn iter_mut(&'a mut self) -> IterMut<'a, T> {
        self.tokens.iter_mut()
    }
    pub(crate) fn iter_vec(&'a self) -> VecsIter<'a, T> {
        VecsIter::new(&self)
    }
    pub(crate) fn iter_vec_mut(&'a mut self) -> VecsIterMut<'a, T> {
        VecsIterMut::new(self)
    }
}

struct VecsIter<'a, T>
where
    T: 'a,
{
    indice_index: usize,
    token_vecs: &'a TokenVecs<T>,
    counter: usize,
}

impl<'a, T> VecsIter<'a, T> {
    fn new(token_vecs: &'a TokenVecs<T>) -> Self {
        Self {
            indice_index: 0,
            token_vecs,
            counter: 0,
        }
    }
}
impl<'a, T> Iterator for VecsIter<'a, T> {
    type Item = &'a [T];
    fn next(&mut self) -> Option<Self::Item> {
        if self.counter >= self.token_vecs.indices.len() - 1 {
            return None;
        }
        //TODO: Change these unwraps to something better
        let start = *self.token_vecs.indices.get(self.indice_index).unwrap();
        let end = *self.token_vecs.indices.get(self.indice_index + 1).unwrap();
        self.counter += 1;
        self.indice_index += 1;
        self.token_vecs.tokens.get(start..end)
    }
}

struct VecsIterMut<T> {
    indice_index: usize,
    token_vecs: TokenVecs<T>,
    counter: usize,
}

impl<T> VecsIterMut<T> {
    fn new(token_vecs: TokenVecs<T>) -> Self {
        Self {
            indice_index: 0,
            token_vecs,
            counter: 0,
        }
    }
}
impl<T> Iterator for VecsIterMut<T> {
    type Item = &[T];
    fn next(&mut self) -> Option<Self::Item> {
        if self.counter >= self.token_vecs.indices.len() - 1 {
            return None;
        }
        //TODO: Change these unwraps to something better
        let start = *self.token_vecs.indices.get(self.indice_index).unwrap();
        let end = *self.token_vecs.indices.get(self.indice_index + 1).unwrap();
        self.counter += 1;
        self.indice_index += 1;
        self.token_vecs.tokens.get(start..end)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    #[allow(non_snake_case)]
    fn test_new_TokenVec() {
        let vecs = build_vecs();
        let actual = TokenVecs::new(vecs);
        let expected_tokens = Box::new([
            Cow::from("O"),
            Cow::from("O"),
            Cow::from("O"),
            Cow::from("B-MISC"),
            Cow::from("I-MISC"),
            Cow::from("I-MISC"),
            Cow::from("O"),
            Cow::from("B-PER"),
            Cow::from("I-PER"),
            Cow::from("O"),
        ]);

        let expected_indices = Box::new([0, 7, 10]);
        let expected = TokenVecs {
            tokens: expected_tokens,
            indices: expected_indices,
        };
        assert_eq!(expected, actual);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_iter_TokenVec_len() {
        let vecs = build_vecs();
        let token_vecs = TokenVecs::new(vecs);
        dbg!(token_vecs.clone());
        let expected = 2;
        let actual = token_vecs.iter_vec().count();
        assert_eq!(expected, actual);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_iter_TokenVec() {
        let vecs = build_vecs();
        let token_vecs = TokenVecs::new(vecs);
        dbg!(token_vecs.clone());
        for (i, actual) in token_vecs.iter_vec().enumerate() {
            if i == 0 {
                dbg!(i);
                let expected = &[
                    Cow::from("O"),
                    Cow::from("O"),
                    Cow::from("O"),
                    Cow::from("B-MISC"),
                    Cow::from("I-MISC"),
                    Cow::from("I-MISC"),
                    Cow::from("O"),
                ];
                assert_eq!(expected, actual);
            } else if i == 1 {
                dbg!(i);
                let expected = &[Cow::from("B-PER"), Cow::from("I-PER"), Cow::from("O")];
                assert_eq!(expected, actual);
            } else if i > 1 {
                dbg!(i);
                dbg!(actual);
                dbg!(token_vecs.clone());
                panic!("Only two iterations possible")
            }
        }
    }
    fn build_vecs() -> Vec<Vec<&'static str>> {
        vec![
            vec!["O", "O", "O", "B-MISC", "I-MISC", "I-MISC", "O"],
            vec!["B-PER", "I-PER", "O"],
        ]
    }
}
