use std::borrow::Cow;
use std::marker::PhantomData;
use std::slice::Iter;

/// Custom datastructure built for reducing cache misses.
#[derive(Debug, Eq, PartialEq, PartialOrd, Ord, Hash, Clone, Default)]
pub(crate) struct TokenVecs<T> {
    pub(crate) tokens: Box<[T]>,
    pub(crate) indices: Box<[usize]>,
}

impl<T> TokenVecs<T> {
    pub(crate) fn new(vecs: Vec<Vec<T>>) -> Self {
        Self::from(vecs)
    }
}

impl<T> TokenVecs<T> {
    pub(crate) fn len(&self) -> usize {
        self.tokens.len()
    }
}

impl<T> From<Vec<Vec<T>>> for TokenVecs<T> {
    #[inline(always)]
    fn from(value: Vec<Vec<T>>) -> Self {
        let length: usize = value.iter().map(|v| v.len()).sum();
        let indices_length = value.len();
        let mut flattened = Vec::with_capacity(length);
        let mut indices = Vec::with_capacity(indices_length);
        indices.push(0);
        for vec in value.into_iter() {
            indices.push(indices.last().unwrap() + vec.len());
            for s in vec {
                flattened.push(s);
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
    #[inline(always)]
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
    pub(crate) fn iter_vec(&'a self) -> VecsIter<'a, T> {
        VecsIter::new(self)
    }
    pub(crate) fn iter_vec_mut(&'a mut self) -> VecsIterMut<'a, T> {
        VecsIterMut::new(self)
    }
}

pub(crate) struct VecsIter<'a, T>
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
        if self.token_vecs.indices.len() == 0 || self.counter >= self.token_vecs.indices.len() - 1 {
            return None;
        }
        let start = unsafe { *self.token_vecs.indices.get_unchecked(self.indice_index) };
        let end = unsafe { *self.token_vecs.indices.get_unchecked(self.indice_index + 1) };
        self.counter += 1;
        self.indice_index += 1;
        self.token_vecs.tokens.get(start..end)
    }
}

pub(crate) struct VecsIterMut<'a, T> {
    indice_index: usize,
    token_vecs: &'a mut TokenVecs<T>,
    counter: usize,
    phantom_data: PhantomData<&'a T>,
}

impl<'a, T> VecsIterMut<'a, T> {
    fn new(token_vecs: &'a mut TokenVecs<T>) -> Self {
        Self {
            indice_index: 0,
            token_vecs,
            counter: 0,
            phantom_data: PhantomData,
        }
    }
}

impl<'a, T> Iterator for VecsIterMut<'a, T> {
    type Item = &'a mut [T];
    fn next(&mut self) -> Option<Self::Item> {
        if self.counter >= self.token_vecs.indices.len() - 1 {
            return None;
        }
        let start = unsafe { *self.token_vecs.indices.get_unchecked(self.indice_index) };
        let end = unsafe { *self.token_vecs.indices.get_unchecked(self.indice_index + 1) };
        self.counter += 1;
        self.indice_index += 1;
        self.token_vecs
            .tokens
            .get_mut(start..end)
            .map(|r| unsafe { &mut *(r as *mut [T]) })
    }
}

// impl<'a, T> VecsIterMut<'a, T> {
//     /// This function act as the `next` method of an `Iterator`. It does not implements the
//     /// `Iterator` trait due to the difference in the function signature.
//     // #[inline]
//     // TODO: inline this and bench perf diff
//     pub(crate) fn custom_next<'b: 'a>(&'b mut self) -> Option<&'a mut [T]> {
//         if self.counter >= self.token_vecs.indices.len() - 1 {
//             return None;
//         }
//         //TODO: Change these unwraps to something better
//         let start = *self.token_vecs.indices.get(self.indice_index).unwrap();
//         let end = *self.token_vecs.indices.get(self.indice_index + 1).unwrap();
//         self.counter += 1;
//         self.indice_index += 1;
//         self.token_vecs.tokens.get_mut(start..end)
//     }
// }

/// This method allocates. It should only be used in the testing environment.
#[cfg(test)]
impl<T> From<TokenVecs<T>> for Vec<Vec<T>>
where
    T: Clone,
{
    fn from(value: TokenVecs<T>) -> Self {
        value.iter_vec().map(|v| Vec::from(v)).collect()
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
            "O", "O", "O", "B-MISC", "I-MISC", "I-MISC", "O", "B-PER", "I-PER", "O",
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
