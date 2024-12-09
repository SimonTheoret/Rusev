/// Custom datastructure built for reducing cache misses.
struct TokenVecs<'a> {
    tokens: Box<[&'a str]>,
    indices: Box<[usize]>,
}

impl<'a> From<Vec<Vec<&'a str>>> for TokenVecs<'a> {
    fn from(value: Vec<Vec<&'a str>>) -> Self {
        let length: usize = value.iter().map(|v| v.len()).sum();
        let indices_length = value.len();
        let mut flattened = Vec::with_capacity(length);
        let mut indices = Vec::with_capacity(indices_length);
        let mut current_indice = 0;
        for vec in value.into_iter() {
            for s in vec {
                current_indice += 1;
                flattened.push(s);
            }
            indices.push(current_indice);
        }
        let tokens = flattened.into_boxed_slice();
        let indices_boxed = indices.into_boxed_slice();
        Self {
            tokens,
            indices: indices_boxed,
        }
    }
}
