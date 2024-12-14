use criterion::{criterion_group, criterion_main, Criterion};
use pprof::criterion::{Output, PProfProfiler};
use rusev::classification_report;
use serde::Deserialize;
use serde_jsonlines::json_lines;
use std::path::Path;

#[derive(Deserialize)]
struct Example {
    true_tags: Vec<String>,
    predicted_tags: Vec<String>,
}

impl Example {
    fn into<P: AsRef<Path>>(path: P) -> (Vec<Vec<String>>, Vec<Vec<String>>) {
        let examples = json_lines::<Example, P>(path)
            .unwrap()
            .map(|r| r.unwrap())
            .collect::<Vec<_>>();
        let mut vec_true_ner = Vec::with_capacity(examples.len());
        let mut vec_predicted_ner = Vec::with_capacity(examples.len());
        for ex in examples {
            vec_true_ner.push(ex.true_tags);
            vec_predicted_ner.push(ex.predicted_tags);
        }
        (vec_true_ner, vec_predicted_ner)
    }
}

fn build_vecs(path: &'static str) -> (Vec<Vec<&'static str>>, Vec<Vec<&'static str>>) {
    let (true_vec_owned, pred_vec_owned) = Example::into(path);
    let true_vec: Vec<Vec<&str>> = true_vec_owned
        .into_iter()
        .map(|v| {
            v.into_iter()
                .map(|s| -> &'static str { s.leak() })
                .collect()
        })
        .collect();
    let pred_vec: Vec<Vec<&str>> = pred_vec_owned
        .into_iter()
        .map(|v| {
            v.into_iter()
                .map(|s| -> &'static str { s.leak() })
                .collect()
        })
        .collect();
    return (true_vec, pred_vec);
}

fn benchmark_small_dataset(c: &mut Criterion) {
    let (true_vec, pred_vec) = build_vecs("./data/datasets/small_dataset.jsonl");
    c.bench_function("small_dataset_report", |b| {
        b.iter(|| {
            classification_report(
                true_vec.clone(),
                pred_vec.clone(),
                None,
                rusev::DivByZeroStrat::ReplaceBy0,
                rusev::SchemeType::IOB2,
                false,
                true,
                true,
            )
            .unwrap()
        })
    });
}
fn benchmark_huge_dataset(c: &mut Criterion) {
    let (true_vec, pred_vec) = build_vecs("./data/datasets/huge_dataset.jsonl");
    c.bench_function("huge_dataset_report", |b| {
        b.iter(|| {
            classification_report(
                true_vec.clone(),
                pred_vec.clone(),
                None,
                rusev::DivByZeroStrat::ReplaceBy0,
                rusev::SchemeType::IOB2,
                false,
                true,
                true,
            )
            .unwrap()
        })
    });
}
fn benchmark_full_dataset(c: &mut Criterion) {
    let (true_vec, pred_vec) = build_vecs("./data/datasets/full_dataset.jsonl");
    c.bench_function("full_dataset_report", |b| {
        b.iter(|| {
            classification_report(
                true_vec.clone(),
                pred_vec.clone(),
                None,
                rusev::DivByZeroStrat::ReplaceBy0,
                rusev::SchemeType::IOB2,
                false,
                true,
                true,
            )
            .unwrap()
        })
    });
}
fn benchmark_big_dataset(c: &mut Criterion) {
    let (true_vec, pred_vec) = build_vecs("./data/datasets/big_dataset.jsonl");
    c.bench_function("big_dataset_report", |b| {
        b.iter(|| {
            classification_report(
                true_vec.clone(),
                pred_vec.clone(),
                None,
                rusev::DivByZeroStrat::ReplaceBy0,
                rusev::SchemeType::IOB2,
                false,
                true,
                true,
            )
            .unwrap()
        })
    });
}

criterion_group!(
    name=long_report_benches;
    config = Criterion::default().sample_size(100).with_profiler(PProfProfiler::new(3000, Output::Flamegraph(None)));
    targets = benchmark_big_dataset,
    benchmark_full_dataset,
    benchmark_small_dataset,
    benchmark_huge_dataset
);
criterion_main!(long_report_benches);
