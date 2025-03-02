/// Small crate containing a custom datastructure. This datastructure
/// is called a `FlatArray` and is a cache-friendly alternative to a
/// Vec<Vec<T>>. Instead, a `FlatArray` is of fixed size but can be
/// iterated over as if it was a `Vec<Vec<T>>`, with the help of the
/// `iter_arrays` and `iter_arrays_mut` methods.
///
/// This crate is intended to be used by other crates, such as `named_entity` and
/// `rusev`, which explains why its API surface is so small.
use std::borrow::Cow;
use std::{marker::PhantomData, ops::Deref, slice::Iter};

#[derive(Debug)]
/// This struct can be used to dynamically build a `FlatArray` by
/// pushing element into it.
pub struct FlatArrayBuilder<T> {
    content: Vec<T>,
    indices: Vec<usize>,
}

impl<T> FlatArrayBuilder<T> {
    pub fn push_exact_sized<I: IntoIterator<Item = T> + ExactSizeIterator>(&mut self, item: I) {
        unsafe {
            self.indices
                .push(self.indices.last().unwrap_unchecked() + item.len())
        };
        for s in item {
            self.content.push(s);
        }
    }
    pub fn push<I: IntoIterator<Item = T>>(&mut self, item: I) {
        let mut current_indice = unsafe { *self.indices.last().unwrap_unchecked() };
        for s in item {
            self.content.push(s);
            current_indice += 1;
        }
        self.indices.push(current_indice)
    }
    pub fn build(self) -> FlatArray<T> {
        FlatArray {
            content: self.content.into_boxed_slice(),
            indices: self.indices.into_boxed_slice(),
        }
    }
}

impl<T> Default for FlatArrayBuilder<T> {
    fn default() -> Self {
        Self {
            content: vec![],
            indices: vec![0],
        }
    }
}

/// Custom datastructure built for reducing cache misses.
#[derive(Debug, Eq, PartialEq, PartialOrd, Ord, Hash, Clone, Default)]
pub struct FlatArray<T> {
    content: Box<[T]>,
    indices: Box<[usize]>,
}

impl<T> FlatArray<T> {
    pub fn new(vecs: Vec<Vec<T>>) -> Self {
        Self::from(vecs)
    }
    pub fn from_raw(content: impl Into<Box<[T]>>, indices: impl Into<Box<[usize]>>) -> Self {
        Self {
            content: content.into(),
            indices: indices.into(),
        }
    }
}

impl<T> FlatArray<T> {
    /// Borrows the content of the FlatArray as a slice.
    pub fn get_content(&self) -> &[T] {
        self.content.as_ref()
    }
}

impl<T> Deref for FlatArray<T> {
    type Target = [T];
    /// `FlatArray` implements the `Deref` trait. It allows users to
    /// treat a `FlatArray` as if it was a slice (e.g. `&[T]`).
    fn deref(&self) -> &Self::Target {
        self.content.as_ref()
    }
}

impl<E, I> FromIterator<I> for FlatArray<E>
where
    I: Iterator<Item = E> + ExactSizeIterator,
{
    /// Build the `FlatArray` from an iterator over vectors. Prefer to
    /// use the `From(Vec<Vec<T>>)` implementation over `from_iter`:
    /// they pre-allocate a minimal amount of memory before filling
    /// the `FlatArray`.
    fn from_iter<T: IntoIterator<Item = I>>(iter: T) -> Self {
        let mut flattened = Vec::new();
        let mut indices = Vec::new();
        indices.push(0);
        for vec in iter.into_iter() {
            indices.push(indices.last().unwrap() + vec.len());
            for s in vec {
                flattened.push(s);
            }
        }
        let tokens = flattened.into_boxed_slice();
        let indices_boxed = indices.into_boxed_slice();
        Self {
            content: tokens,
            indices: indices_boxed,
        }
    }
}

impl<T> From<Vec<Vec<T>>> for FlatArray<T> {
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
            content: tokens,
            indices: indices_boxed,
        }
    }
}

impl<'a> From<Vec<Vec<&'a str>>> for FlatArray<Cow<'a, str>> {
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
            content: tokens,
            indices: indices_boxed,
        }
    }
}

impl<'a, T> FlatArray<T> {
    /// Returns an iterator over the content. This iterator returns the individual elements.
    pub fn iter(&'a self) -> Iter<'a, T> {
        self.content.iter()
    }
    /// Returns an iterator over the arrays/vectors used to build the
    /// `FlatArray`. The iterator will return a slice of type `&[T]`.
    pub fn iter_arrays(&'a self) -> ArraysIter<'a, T> {
        ArraysIter::new(self)
    }
    /// Returns an iterator over the arrays/vectors used to build the
    /// `FlatArray`. The iterator will return a slice of type `&mut
    /// [T]`.
    pub fn iter_arrays_mut(&'a mut self) -> ArraysIterMut<'a, T> {
        ArraysIterMut::new(self)
    }
}

pub struct ArraysIter<'a, T>
where
    T: 'a,
{
    indice_index: usize,
    token_vecs: &'a FlatArray<T>,
    counter: usize,
}

impl<'a, T> ArraysIter<'a, T> {
    fn new(token_vecs: &'a FlatArray<T>) -> Self {
        Self {
            indice_index: 0,
            token_vecs,
            counter: 0,
        }
    }
}
impl<'a, T> Iterator for ArraysIter<'a, T> {
    type Item = &'a [T];
    // NOTE: Inlining this function seems to _reduce_ the performance
    // #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.token_vecs.indices.len() == 0 || self.counter >= self.token_vecs.indices.len() - 1 {
            return None;
        }
        let start = unsafe { *self.token_vecs.indices.get_unchecked(self.indice_index) };
        let end = unsafe { *self.token_vecs.indices.get_unchecked(self.indice_index + 1) };
        self.counter += 1;
        self.indice_index += 1;
        self.token_vecs.content.get(start..end)
    }
}

pub struct ArraysIterMut<'a, T> {
    indice_index: usize,
    token_vecs: &'a mut FlatArray<T>,
    counter: usize,
    phantom_data: PhantomData<&'a T>,
}

impl<'a, T> ArraysIterMut<'a, T> {
    fn new(token_vecs: &'a mut FlatArray<T>) -> Self {
        Self {
            indice_index: 0,
            token_vecs,
            counter: 0,
            phantom_data: PhantomData,
        }
    }
}

impl<'a, T> Iterator for ArraysIterMut<'a, T> {
    type Item = &'a mut [T];
    // NOTE: Inlining this function seems to reduce the performance
    // #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.counter >= self.token_vecs.indices.len() - 1 {
            return None;
        }
        let start = unsafe { *self.token_vecs.indices.get_unchecked(self.indice_index) };
        let end = unsafe { *self.token_vecs.indices.get_unchecked(self.indice_index + 1) };
        self.counter += 1;
        self.indice_index += 1;
        self.token_vecs
            .content
            .get_mut(start..end)
            .map(|r| unsafe { &mut *(r as *mut [T]) })
    }
}

/// This method allocates. It should only be used in the testing environment.
#[cfg(test)]
impl<T> From<FlatArray<T>> for Vec<Vec<T>>
where
    T: Clone,
{
    fn from(value: FlatArray<T>) -> Self {
        value.iter_arrays().map(Vec::from).collect()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    #[allow(non_snake_case)]
    fn test_new_TokenVec() {
        let vecs = build_vecs();
        let actual = FlatArray::new(vecs);
        let expected_tokens = Box::new([
            "O", "O", "O", "B-MISC", "I-MISC", "I-MISC", "O", "B-PER", "I-PER", "O",
        ]);

        let expected_indices = Box::new([0, 7, 10]);
        let expected = FlatArray {
            content: expected_tokens,
            indices: expected_indices,
        };
        assert_eq!(expected, actual);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_iter_TokenVec_len() {
        let vecs = build_vecs();
        let token_vecs = FlatArray::new(vecs);
        dbg!(token_vecs.clone());
        let expected = 2;
        let actual = token_vecs.iter_arrays().count();
        assert_eq!(expected, actual);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_iter_TokenVec() {
        let vecs = build_vecs();
        let token_vecs = FlatArray::new(vecs);
        dbg!(token_vecs.clone());
        for (i, actual) in token_vecs.iter_arrays().enumerate() {
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
