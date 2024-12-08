from dataclasses import dataclass
from argparse import ArgumentParser
from seqeval.metrics import classification_report
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


def main():
    print("Hello from seqeval-benchmarks!")


if __name__ == "__main__":
    args = ArgumentParser.parse_args()
    main()
