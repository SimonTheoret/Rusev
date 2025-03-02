# FlatArray
This crates contains the code for a cache-friendly datastructure named
`FlatArray`. It mimics the behavior of a `Vec<Vec<T>>` by allowing the
users to iterate over the vectors and can be dereferenced into a
`&[T]`.  `FlatArray` is of fixed size but can be iterated over as if
it was a `Vec<Vec<T>>`, with the help of the `iter_arrays` and
`iter_arrays_mut` methods.
