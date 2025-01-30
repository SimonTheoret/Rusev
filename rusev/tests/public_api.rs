use rusev::{
    classification_report_conf, ClassMetrics, DivByZeroStrat, Reporter, RusevConfigBuilder,
    SchemeType,
};
use std::collections::HashSet;
use std::fs::read_to_string;

pub trait CloseEnough {
    fn are_close(&self, other: &Self, eps: f32) -> bool;
}

// ClassMetrics does not have the default PartialEq implementation.
impl CloseEnough for ClassMetrics {
    fn are_close(&self, other: &Self, eps: f32) -> bool {
        let are_equal = self == other;
        let precision_is_equal = f32::abs(self.precision - other.precision) < eps;
        let recall_is_equal = f32::abs(self.recall - other.recall) < eps;
        let fscore_is_equal = f32::abs(self.fscore - other.fscore) < eps;
        are_equal && precision_is_equal && recall_is_equal && fscore_is_equal
    }
}

#[test]
fn comparison_to_conlleval() {
    let content =
        read_to_string("tests/output.txt").expect("file output.txt not found in test directory");
    let mut true_y: Vec<&str> = vec![];
    let mut pred_y: Vec<&str> = vec![];
    for line in content.lines() {
        if line.is_empty() {
            continue;
        }
        let vect: Vec<_> = line.split(' ').collect();
        true_y.push(vect[2]);
        pred_y.push(vect[3]);
    }
    let config = RusevConfigBuilder::default()
        .division_by_zero(DivByZeroStrat::ReplaceBy0)
        .strict(true)
        .scheme(SchemeType::IOB2)
        .build();
    let actual_reporter = classification_report_conf(vec![true_y], vec![pred_y], config).unwrap();
    let mut expected_reporter: HashSet<ClassMetrics> = Reporter::default().into();
    expected_reporter.insert(ClassMetrics {
        class: String::from("Overall_Micro"),
        average: rusev::Average::Micro,
        precision: 0.6883,
        recall: 0.8083,
        fscore: 0.7435,
        support: 539,
    });
    expected_reporter.insert(ClassMetrics {
        class: String::from("ADJP"),
        average: rusev::Average::None,
        precision: 0.000,
        recall: 0.000,
        fscore: 0.000,
        support: 1,
    });
    expected_reporter.insert(ClassMetrics {
        class: String::from("ADVP"),
        average: rusev::Average::None,
        precision: 0.4545,
        recall: 0.6250,
        fscore: 0.5263,
        support: 11,
    });
    expected_reporter.insert(ClassMetrics {
        class: String::from("NP"),
        average: rusev::Average::None,
        precision: 0.6498,
        recall: 0.7863,
        fscore: 0.7116,
        support: 317,
    });
    expected_reporter.insert(ClassMetrics {
        class: String::from("PP"),
        average: rusev::Average::None,
        precision: 0.8318,
        recall: 0.9889,
        fscore: 0.9036,
        support: 107,
    });
    expected_reporter.insert(ClassMetrics {
        class: String::from("SBAR"),
        average: rusev::Average::None,
        precision: 0.6667,
        recall: 0.3333,
        fscore: 0.4444,
        support: 3,
    });
    expected_reporter.insert(ClassMetrics {
        class: String::from("VP"),
        average: rusev::Average::None,
        precision: 0.6900,
        recall: 0.7931,
        fscore: 0.7380,
        support: 100,
    });
    let actual_reporter: HashSet<_> = actual_reporter.into();
    for expected_class in expected_reporter.into_iter() {
        dbg!(&expected_class);
        let actual_class = actual_reporter.get(&expected_class).unwrap();
        dbg!(actual_class);
        assert!(actual_class.are_close(&expected_class, 0.001));
    }
}
