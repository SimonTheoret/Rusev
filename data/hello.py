import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


def main():
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    data = load_dataset("eriktks/conll2003")
    lists = []
    sets = ["train", "test", "validation"]
    for set_instance in sets:
        dataset = data[set_instance]
        for example in dataset:
            print(example)
            lists.append(example.tokens, example.ner_tags)


if __name__ == "__main__":
    main()
