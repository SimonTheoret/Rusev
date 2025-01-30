# Rusev: Rust Sequence Evaluation framework
[![codecov](https://codecov.io/gh/SimonTheoret/Rusev/graph/badge.svg?token=GOYLEM2QN7)](https://codecov.io/gh/SimonTheoret/Rusev)

This crates is a port of the `SeqEval` library, focused on performance and
soudness. It presents a simple interface, composed of two functions and a
variation:  `classification_report(_conf)` and
`precision_recall_fscore_support`. One can use these two functions to obtain
the precision, the recall, the fscore and the support of each named entity and
the overall metrics.  Users can obtain these metrics with the `conf` variation
of the `classification_report` function:

 ```rust
 use rusev::{SchemeType, RusevConfigBuilder, DefaultRusevConfig, classification_report_conf};

 let y_true = vec![vec!["B-TEST", "B-NOTEST", "O", "B-TEST"]];
 let y_pred = vec![vec!["O", "B-NOTEST", "B-OTHER", "B-TEST"]];
 let config: DefaultRusevConfig =
 RusevConfigBuilder::default().scheme(SchemeType::IOB2).strict(true).build();

 let wrapped_reporter = classification_report_conf(y_true, y_pred, config);
 let reporter = wrapped_reporter.unwrap();
 let expected_report = "Class, Precision, Recall, Fscore, Support
 Overall_Weighted, 1, 0.6666667, 0.77777785, 3
 Overall_Micro, 0.6666667, 0.6666667, 0.6666667, 3
 Overall_Macro, 0.6666667, 0.5, 0.5555556, 3
 NOTEST, 1, 1, 1, 1
 OTHER, 0, 0, 0, 0
 TEST, 1, 0.5, 0.6666667, 2\n";

 assert_eq!(expected_report, reporter.to_string());
 ```

It is also possible to specify all the arguments manually, like so:
```rust
use rusev::{ classification_report, DivByZeroStrat, SchemeType };


let y_true = vec![vec!["B-TEST", "B-NOTEST", "O", "B-TEST"]];
let y_pred = vec![vec!["O", "B-NOTEST", "B-OTHER", "B-TEST"]];


let reporter = classification_report(y_true, y_pred, None, DivByZeroStrat::ReplaceBy0,
 Some(SchemeType::IOB2), false, false ).unwrap();
let expected_report = "Class, Precision, Recall, Fscore, Support
Overall_Weighted, 1, 0.6666667, 0.77777785, 3
Overall_Micro, 0.6666667, 0.6666667, 0.6666667, 3
Overall_Macro, 0.6666667, 0.5, 0.5555556, 3
NOTEST, 1, 1, 1, 1
OTHER, 0, 0, 0, 0
TEST, 1, 0.5, 0.6666667, 2\n";


assert_eq!(expected_report, reporter.to_string());
 ```

## Why another implementation
This implementation was build for performance. On some benchmarks, it is 14 to
23 times faster than the original library, making it useful to reduce the time
spent evaluating models during.

## Testing and Benchmarks
This library was tested against
[CoNLL-2002](https://www.clips.uantwerpen.be/conll2002/ner/) in
`tests/public_api.rs` and was benchmarked with generated data. It was between
14 and 23 times faster than the original
[SeqEval](https://github.com/chakki-works/seqeval) implementation when using
the pure Rust crate. To reproduce the benchmarks, follow the instructions in
the `data/README.md` file. Note that the results might differ due to the random
shuffling in the generated data.

