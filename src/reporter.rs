use crate::metrics::Average;

enum AverageOrAll {
    Averaged(Average),
    All,
}

struct Reporter<'a> {
    average: AverageOrAll,
    classes: Vec<ClassMetric<'a>>,
}

// TODO: Impl ord for sorting the classes in lexicogaph order
#[derive(Debug, PartialEq)]
struct ClassMetric<'a> {
    class: &'a str,
    precision: f32,
    recall: f32,
    fscore: f32,
    support: usize,
}
