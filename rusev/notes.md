# TODO
- [x] Remove `delimiter` argument. It is not used!
- [x] Start fuzzy testing or property testing
- [x] Finish the calculation implementation
  - [x] Remodel Prefix into UserPrefix AND Prefix
- [x] Write a lot of tests
  - [x] Write property tests
- [x] Start benchmarking
- [x] Test for no tag tokens as input (eg. "B", "I", "I", "O")
- [x] Clean up directory:
    - [x] Remove `Some(...)` in the lenient versions
    - [x] Verify the unwraps during classification
    - [x] Correct bug in `profiling.rs`
- [x] Create new crates out of the entity module: `Named_Entity` and `FlatArray`
- [x] Change name of `py_binding` crate

## Add an interface for `Token`

``` rust
// Token
fn is_valid(&self) -> bool; // Verifies that the current token is valid
fn inner(&self) -> &InnerToken; // returns the inner_token
fn is_start(&self, prev: &InnerToken) -> bool; // Check whether the current token is the start of chunk.
fn forward(&self, start: usize, prev: &Token<'a>) -> usize;
fn is_end(&self, i: usize) -> bool;
fn get_tag(&mut self) -> &'a str;
```
### Notes
- `inner` should be removed
- Modify the `InnerToken` into `Token`.
- 


