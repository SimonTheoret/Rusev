from dataclasses import dataclass
from argparse import ArgumentParser
from time import time
from seqeval.metrics.v1 import classification_report
from seqeval.scheme import IOB2
from typing import Self
import jsonlines


@dataclass
class Example:
    true_tags: list[list[str]]
    predicted_tags: list[list[str]]

    def __init__(self, line: dict[str, list[list[str]]]) -> None:
        self.true_tags = line["true_tags"]
        self.predicted_tags = line["predicted_tags"]


def build_lists(path: str) -> tuple[list[list[str]], list[list[str]]]:
    examples = []
    with jsonlines.open(path) as f:
        for line in f:
            examples.append(Example(line))
    true_list: list[list[str]] = []
    pred_list: list[list[str]] = []
    for ex in examples:
        true_list.append(ex.true_tags)
        pred_list.append(ex.predicted_tags)
    return (true_list, pred_list)


def main(n_samples: int, dataset: str):
    total_duration: float = 0.0
    path: str = f"./data/datasets/{dataset}_dataset.jsonl"
    for _ in range(n_samples):
        true_list, pred_list = build_lists(path)
        now = time()
        classification_report(
            y_true=true_list,
            y_pred=pred_list,
            sample_weight=None,
            digits=2,
            output_dict=False,
            zero_division=0,
            suffix=False,
            scheme=IOB2,
        )
        elapsed = time() - now
        total_duration += elapsed
    print(f"Total duration: {total_duration} with {n_samples} samples")


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="Seqeval Benchmark",
        description="Benchmark SeqEval on different size of datasets",
    )
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="big")
    args = parser.parse_args()
    main(args.n_samples, args.dataset)
