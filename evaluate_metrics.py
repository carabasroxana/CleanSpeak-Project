import json
from pathlib import Path
from typing import List, Dict

import torch
from transformers import pipeline
from evaluate import load as load_metric
rouge = load_metric("rouge")

from model.architecture import load_model_and_tokenizer, format_input

CORPUS_PATH = Path("data/final_corpus.jsonl")
CHECKPOINT_PATH = Path("model/final.pt")
SAMPLE_SIZE = 200
MAX_LEN = 128

ROUGE = load_metric("rouge")
BERTSCORE = load_metric("bertscore")
TOXICITY_CLASSIFIER = pipeline(
    "text-classification", model="unitary/toxic-bert", return_all_scores=False
)


def load_test_samples(path: Path, sample_size: int) -> List[Dict[str, str]]:
    """
    Read the last `sample_size` records from the JSONL corpus file.
    """
    records = []
    for line in path.read_text(encoding="utf-8").splitlines()[-sample_size:]:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def generate_predictions(
    samples: List[Dict[str, str]],
    model: torch.nn.Module,
    tokenizer,
    max_len: int
) -> List[str]:
    """
    Generate neutral rewrites for each sample
    using the fine-tuned model and tokenizer.
    """
    predictions = []
    model.eval()
    for rec in samples:
        inp = format_input(rec.get("text", ""), rec.get("emotion", "neutral"))
        inputs = tokenizer(
            inp,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_len
        )
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_len
            )
        pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        predictions.append(pred)
    return predictions


def compute_and_print_metrics(
    references: List[str],
    predictions: List[str]
) -> None:
    """
    Compute ROUGE, BERTScore, and toxicity rates,
    then print the results.
    """
    rouge_res = ROUGE.compute(predictions=predictions, references=references)
    bert_res = BERTSCORE.compute(
        predictions=predictions, references=references, lang="en"
    )
    tox_results = TOXICITY_CLASSIFIER(predictions)

    print("\n==== Evaluation Results ====")
    print("ROUGE:")
    for key, scores in rouge_res.items():
        print(f"  {key}: {{:.4f}}".format(scores.mid.fmeasure))
    print(f"BERTScore (P/R/F1): {bert_res['precision']:.4f}/{bert_res['recall']:.4f}/{bert_res['f1']:.4f}")

    toxic_count = sum(1 for r in tox_results if r['label'] == 'TOXIC')
    print(f"Toxicity in outputs: {toxic_count}/{len(predictions)} flagged as toxic")


def main():
    model, tokenizer = load_model_and_tokenizer()
    model.load_state_dict(torch.load(CHECKPOINT_PATH))

    samples = load_test_samples(CORPUS_PATH, SAMPLE_SIZE)
    references = [rec.get("neutral_rewrite", "") for rec in samples]

    preds = generate_predictions(samples, model, tokenizer, MAX_LEN)

    compute_and_print_metrics(references, preds)


if __name__ == "__main__":
    main()
