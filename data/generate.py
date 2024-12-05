from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from typing import Mapping


class Example:
    _NER_TAGS_MAP: dict[int, str] = {
        0: "O",
        1: "B-PER",
        2: "I-PER",
        3: "B-ORG",
        4: "I-ORG",
        5: "B-LOC",
        6: "I-LOC",
        7: "B-MISC",
        8: "I-MISC",
    }

    def __init__(
        self, tokens: list[str], true_tags: list[int], predicted_tags: list[str]
    ) -> None:
        self.tokens = tokens
        self.true_tags = [self._NER_TAGS_MAP[i] for i in true_tags]
        self.predicted_tags = predicted_tags

    def __repr__(self) -> str:
        return f"tokens: {self.tokens}, true tags: {self.true_tags}, predicted tags: {self.predicted_tags}"


def main():
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    data = load_dataset("eriktks/conll2003", trust_remote_code=True)
    lists: list[Example] = []
    sets = ["train", "test", "validation"]
    for set_instance in sets:
        assert isinstance(data, DatasetDict)
        dataset = data[set_instance]
        for example in dataset:
            assert isinstance(example, Mapping)
            res = nlp(example["tokens"])
            # assert isinstance(res, Mapping)
            ex = Example(example["tokens"], example["ner_tags"], res[1])
            print(ex)
            lists.append(ex)


if __name__ == "__main__":
    main()
