use criterion::{criterion_group, criterion_main, Criterion};
use pprof::criterion::{Output, PProfProfiler};
use rusev::classification_report;
use serde::Deserialize;
use serde_jsonlines::json_lines;
use std::path::Path;
use std::time::Duration;

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
    (true_vec, pred_vec)
}

fn benchmark_small_dataset_lenient(c: &mut Criterion) {
    let (true_vec, pred_vec) = build_vecs("./data/datasets/small_dataset.jsonl");
    c.bench_function("small_dataset_lenient_report", |b| {
        b.iter(|| {
            classification_report(
                true_vec.clone(),
                pred_vec.clone(),
                None,
                rusev::DivByZeroStrat::ReplaceBy0,
                None,
                false,
                false,
            )
            .unwrap()
        })
    });
}

fn benchmark_huge_dataset_lenient(c: &mut Criterion) {
    let (true_vec, pred_vec) = build_vecs("./data/datasets/huge_dataset.jsonl");
    c.bench_function("huge_dataset_lenient_report", |b| {
        b.iter(|| {
            classification_report(
                true_vec.clone(),
                pred_vec.clone(),
                None,
                rusev::DivByZeroStrat::ReplaceBy0,
                None,
                false,
                false,
            )
            .unwrap()
        })
    });
}
fn benchmark_big_dataset_lenient(c: &mut Criterion) {
    let (true_vec, pred_vec) = build_vecs("./data/datasets/big_dataset.jsonl");
    c.bench_function("big_dataset_lenient_report", |b| {
        b.iter(|| {
            classification_report(
                true_vec.clone(),
                pred_vec.clone(),
                None,
                rusev::DivByZeroStrat::ReplaceBy0,
                None,
                false,
                false,
            )
            .unwrap()
        })
    });
}

fn benchmark_small_dataset_strict(c: &mut Criterion) {
    let (true_vec, pred_vec) = build_vecs("./data/datasets/small_dataset.jsonl");
    c.bench_function("small_dataset_strict_report", |b| {
        b.iter(|| {
            classification_report(
                true_vec.clone(),
                pred_vec.clone(),
                None,
                rusev::DivByZeroStrat::ReplaceBy0,
                Some(rusev::SchemeType::IOB2),
                false,
                false,
            )
            .unwrap()
        })
    });
}

fn benchmark_huge_dataset_strict(c: &mut Criterion) {
    let (true_vec, pred_vec) = build_vecs("./data/datasets/huge_dataset.jsonl");
    c.bench_function("huge_dataset_strict_report", |b| {
        b.iter(|| {
            classification_report(
                true_vec.clone(),
                pred_vec.clone(),
                None,
                rusev::DivByZeroStrat::ReplaceBy0,
                Some(rusev::SchemeType::IOB2),
                false,
                false,
            )
            .unwrap()
        })
    });
}
fn benchmark_big_dataset_strict(c: &mut Criterion) {
    let (true_vec, pred_vec) = build_vecs("./data/datasets/big_dataset.jsonl");
    c.bench_function("big_dataset_strict_report", |b| {
        b.iter(|| {
            classification_report(
                true_vec.clone(),
                pred_vec.clone(),
                None,
                rusev::DivByZeroStrat::ReplaceBy0,
                Some(rusev::SchemeType::IOB2),
                false,
                false,
            )
            .unwrap()
        })
    });
}

criterion_group!(
    name=fast_report_benches;
    config = Criterion::default().sample_size(250).with_profiler(PProfProfiler::new(3000, Output::Flamegraph(None))).measurement_time(Duration::from_secs(10));
    targets =
    benchmark_small_dataset_lenient,
    benchmark_big_dataset_lenient,
    benchmark_huge_dataset_lenient,
    benchmark_small_dataset_strict,
    benchmark_big_dataset_strict,
    benchmark_huge_dataset_strict
);
criterion_main!(fast_report_benches);
