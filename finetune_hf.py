import json
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
)


def load_jsonl(path):
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def main():
    train = load_jsonl("data/final_corpus_annotated.jsonl")
    dataset = Dataset.from_list(train).train_test_split(test_size=0.1)
    model_name = "t5-small"
    tok = AutoTokenizer.from_pretrained(model_name)
    max_len = 128

    def preprocess(ex):
        inpt = tok(ex["text"], truncation=True, padding="max_length", max_length=max_len)
        tgt = tok(ex["neutral_rewrite"], truncation=True, padding="max_length", max_length=max_len)
        inpt["labels"] = tgt["input_ids"]
        return inpt

    all_cols = dataset["train"].column_names
    tokenized = dataset.map(preprocess, batched=True, remove_columns=all_cols)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    args = TrainingArguments(
        output_dir="./polite-bot",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        do_eval=True,
        eval_steps=500,
        logging_steps=100,
        save_steps=500,
        save_total_limit=2,
        num_train_epochs=3,
        learning_rate=5e-5,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tok,
    )
    trainer.train()
    trainer.save_model("./polite-bot")
    tok.save_pretrained("./polite-bot")


if __name__ == "__main__":
    main()
