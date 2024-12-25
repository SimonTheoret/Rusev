#!/usr/bin/env sh

echo "Compare the performance on small datasets (128 sequences)"

echo "Rusev"
cargo run -p profiling -r -- --n-samples 100 --dataset small

echo "If not done already, build and source the virtual environement first"

echo "SeqEval"
python3 -m profiling.seqeval_benchmarks.bench --n_samples 100 --dataset small
