use std::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion};
use rusev::classification_report;


fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("fib 20", |b| b.iter(|| fibonacci(black_box(20))));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
