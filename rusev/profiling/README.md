
To compare Rusev and SeqEval, install the virtual env for the profiling task
and activate it. Then, use the following commands:
``` sh
python3 -m profiling.seqeval_benchmarks.bench --n_samples 100 --dataset small
cargo run -p profiling -r -- --n-samples 100 --dataset small
```
