from datasets import load_dataset, DatasetDict
from typing import Any, Callable, Generator, Self
from random import choice, random
import jsonlines
import fire
from abc import ABC, abstractmethod


class RandomModel:
    """Class used to build random new tags sequence out of a real tag sequence."""

    def __init__(self, possible_ner_tags: list[str], proba: float = 0.75) -> None:
        self.ner_tags = possible_ner_tags
        self.proba = proba

    def __call__(self, true_ner_tags: list[str]) -> Any:
        return [
            choice(self.ner_tags) if random() > self.proba else true_tag
            for true_tag in true_ner_tags
        ]


class Example:
    def __init__(self, true_tags: list[str], predicted_tags: list[str]) -> None:
        self.true_tags = true_tags
        self.predicted_tags = predicted_tags
        assert len(true_tags) == len(predicted_tags)
        assert len(true_tags) > 0

    def to_dict(self) -> dict[str, list[str]]:
        return {"true_tags": self.true_tags, "predicted_tags": self.predicted_tags}

    def __repr__(self) -> str:
        return f"True tags: {self.true_tags}, Predicted tags: {self.predicted_tags}"

    def __len__(self) -> int:
        return len(self.true_tags)


class JsonlDataset:
    def __init__(
        self, file_destination: str, filter_f: Callable[[Example], bool] | None = None
    ) -> None:
        self.file = file_destination
        self.examples_list: list[Example] = []
        self.converted: list[dict]
        self.filter_f = filter_f

    def append(self, example: Example) -> None:
        if self.filter_f is not None:
            if self.filter_f(example):
                self.examples_list.append(example)
        else:
            self.examples_list.append(example)

    def _convert(self) -> None:
        """Populte the `self.converted` attribute."""
        self.converted = [ex.to_dict() for ex in self.examples_list]

    def write(self):
        self._convert()
        written_bytes = 0
        with jsonlines.open(self.file, "w") as writer:
            for ex_dict in self.converted:
                written_bytes += writer.write(ex_dict)
        print(f"wrote {written_bytes} bytes")

    def __len__(self) -> int:
        return len(self.examples_list)


class WrappedDataset(ABC):
    @abstractmethod
    def __iter__(self) -> Self:
        pass

    @abstractmethod
    def __next__(self) -> list[str]:
        pass

    @abstractmethod
    def all_tags(self) -> list[str]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def shuffle(self) -> None:
        pass


TAGS = [
    "O",
    "B-PERSON",
    "I-PERSON",
    "B-NORP",
    "I-NORP",
    "B-FAC",
    "I-FAC",
    "B-ORG",
    "I-ORG",
    "B-GPE",
    "I-GPE",
    "B-LOC",
    "I-LOC",
    "B-PRODUCT",
    "I-PRODUCT",
    "B-DATE",
    "I-DATE",
    "B-TIME",
    "I-TIME",
    "B-PERCENT",
    "I-PERCENT",
    "B-MONEY",
    "I-MONEY",
    "B-QUANTITY",
    "I-QUANTITY",
    "B-ORDINAL",
    "I-ORDINAL",
    "B-CARDINAL",
    "I-CARDINAL",
    "B-EVENT",
    "I-EVENT",
    "B-WORK_OF_ART",
    "I-WORK_OF_ART",
    "B-LAW",
    "I-LAW",
    "B-LANGUAGE",
    "I-LANGUAGE",
]

ID_TO_TAG = {i: tag for i, tag in enumerate(TAGS)}


class Conll2012OntoNotesv5DatasetEnglish(WrappedDataset):
    def __init__(self) -> None:
        super().__init__()
        self.CONLL2012_ONTONOTESV5_TAGS = TAGS
        self.CONLL2012_ONTONOTESV5_TAGS_ID_TO_TAG = {
            i: tag for i, tag in enumerate(self.CONLL2012_ONTONOTESV5_TAGS)
        }

        self.data_v12 = load_dataset(
            "ontonotes/conll2012_ontonotesv5",
            "english_v12",
            trust_remote_code=True,
        )

        self.data_v4 = load_dataset(
            "ontonotes/conll2012_ontonotesv5",
            "english_v4",
            trust_remote_code=True,
        )

        self.data_chinese = load_dataset(
            "ontonotes/conll2012_ontonotesv5",
            "chinese_v4",
            trust_remote_code=True,
        )
        self.data_arabic = load_dataset(
            "ontonotes/conll2012_ontonotesv5",
            "arabic_v4",
            trust_remote_code=True,
        )
        self.datasets = [
            self.data_v12,
            self.data_v4,
            self.data_chinese,
            self.data_arabic,
        ]
        assert isinstance(self.data_v12, DatasetDict)
        assert isinstance(self.data_v4, DatasetDict)
        assert isinstance(self.data_chinese, DatasetDict)
        assert isinstance(self.data_arabic, DatasetDict)

        self.sets = ["train", "test", "validation"]

    def all_tags(self) -> list[str]:
        return self.CONLL2012_ONTONOTESV5_TAGS

    def _gen(self) -> Generator[list[str], Any, Any]:
        for data in self.datasets:
            assert isinstance(data, DatasetDict)
            for set_instance in self.sets:
                hf_dataset = data[set_instance]
                for document in hf_dataset:
                    assert isinstance(document, dict)
                    for j, sentence in enumerate(document["sentences"]):
                        true_ne: list[str] = [
                            self.CONLL2012_ONTONOTESV5_TAGS_ID_TO_TAG[i]
                            for i in sentence["named_entities"]
                        ]
                        yield true_ne

    def __iter__(self) -> Self:
        self.gen = self._gen()
        return self

    def __next__(self) -> list[str]:
        return next(self.gen)

    def __len__(self) -> int:
        total_len = 0
        for dataset in self.datasets:
            assert isinstance(dataset, DatasetDict)
            total_len += len(dataset)
        return total_len

    def shuffle(self) -> None:
        for dset in self.datasets:
            dset.shuffle()


class WrappedDatasetFactory:
    def __init__(self, name: str, **dataset_kwargs: dict[str, Any]):
        self.name = name
        self.dataset_kwargs = dataset_kwargs

    def init(self) -> WrappedDataset:
        match self.name:
            case "Conll2012OEnglishV12":
                return Conll2012OntoNotesv5DatasetEnglish()
            case _:
                raise ValueError(f"No dataset named {self.name}")


def len_filter(min_len: int, max_len: int) -> Callable[[Example], bool]:
    def f(ex: Example) -> bool:
        length = len(ex)
        return length >= min_len and length < max_len

    return f


def main(
    len_filter_args: tuple[int, int] | None = None,
    n_sentences: int = 5000,
    out_file_name="data.jsonl",
):
    if len_filter_args is not None:
        filter = len_filter(*len_filter_args)
    else:
        filter = None
    dataset = JsonlDataset(out_file_name, filter_f=filter)
    wrapped_dataset = Conll2012OntoNotesv5DatasetEnglish()
    rand_model = RandomModel(wrapped_dataset.all_tags())
    for _, true_ne in enumerate(wrapped_dataset):
        predicted_ne: list[str] = rand_model(true_ne)
        example = Example(true_ne, predicted_ne)
        dataset.append(example)
        if len(dataset) >= n_sentences:
            break
    dataset.write()


if __name__ == "__main__":
    fire.Fire(main)
