use rusev::classification_report;
use serde::Deserialize;
use serde_jsonlines::json_lines;
use std::ops::Range;
use std::path::Path;
use std::time::{Duration, Instant};

use clap::Parser;

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

fn build_vecs<P: AsRef<Path>>(path: P) -> (Vec<Vec<&'static str>>, Vec<Vec<&'static str>>) {
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

// #[derive(Debug, Clone)]
// enum Datasets {
//     Small,
//     Big,
//     Huge,
//     Full,
// }
// impl Display for Datasets {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         match self {
//             Self::Small => write!(f, "small"),
//             Self::Full => write!(f, "full"),
//             Self::Big => write!(f, "big"),
//             Self::Huge => write!(f, "huge"),
//         }
//     }
// }
//
// impl From<&str> for Datasets {
//     fn from(value: &str) -> Self {
//         match value.to_lowercase().as_str() {
//             "small" => Datasets::Small,
//             "big" => Datasets::Big,
//             "huge" => Datasets::Huge,
//             "full" => Datasets::Full,
//             _ => panic!(),
//         }
//     }
// }

#[derive(Debug, Parser)]
struct Args {
    #[arg(short, long, default_value_t = 1)]
    n_samples: u32,
    #[arg(short, long, default_value_t=String::from("big"))]
    dataset: String,
}

fn main() {
    let args = Args::parse();
    let n_samples = args.n_samples;
    let iter = Range {
        start: 0,
        end: n_samples,
    };
    let mut total_duration = Duration::ZERO;
    let path = format!("./data/datasets/{}_dataset.jsonl", args.dataset);
    for _ in iter {
        let (true_vec, pred_vec) = build_vecs(&path);
        let now = Instant::now();
        {
            classification_report(
                true_vec,
                pred_vec,
                None,
                rusev::DivByZeroStrat::ReplaceBy0,
                rusev::SchemeType::IOB2,
                false,
                false,
                true,
            )
            .unwrap();
        }
        let elapsed = now.elapsed();
        total_duration += elapsed;
    }
    println!(
        "Total duration: {} with {n_samples} samples",
        total_duration.as_secs_f64()
    )
}
