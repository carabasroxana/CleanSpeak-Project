import json
from pathlib import Path
from typing import List, Dict

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from evaluate import load as load_metric

from serve import device
from model.architecture import format_input, EMOTION_TOKENS

CORPUS_PATH     = Path("data/final_corpus.jsonl")
SAMPLE_SIZE     = 200
MAX_LEN         = 128

rouge_metric     = load_metric("rouge")
bertscore_metric = load_metric("bertscore")

tox_tokenizer = AutoTokenizer.from_pretrained(
    "unitary/toxic-bert",
    truncation=True,
    model_max_length=512
)
TOXICITY = pipeline(
    "text-classification",
    model="unitary/toxic-bert",
    tokenizer=tox_tokenizer,
    return_all_scores=False,
    device=0
)

tok   = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("./polite-bot").to(device)

def politeify(text: str) -> str:
    """Take a raw text string, run it through your fine-tuned model, return the decoded output."""
    inputs = tok(text,
                 return_tensors="pt",
                 truncation=True,
                 padding="max_length",
                 max_length=MAX_LEN).to(device)
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_length=MAX_LEN)
    return tok.decode(out_ids[0], skip_special_tokens=True)

def load_test_samples(path: Path, sample_size: int) -> List[Dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()[-sample_size:]
    return [json.loads(l) for l in lines]

def compute_metrics(
    references: List[str],
    predictions: List[str]
) -> None:
    rouge_res = rouge_metric.compute(predictions=predictions, references=references)
    bert_res  = bertscore_metric.compute(
        predictions=predictions,
        references=references,
        lang="en"
    )
    avg_p = sum(bert_res["precision"]) / len(bert_res["precision"])
    avg_r = sum(bert_res["recall"])    / len(bert_res["recall"])
    avg_f = sum(bert_res["f1"])        / len(bert_res["f1"])

    tox_results = [TOXICITY(pred)[0] for pred in predictions]
    toxic_count = sum(1 for r in tox_results if r["label"] == "TOXIC")

    print("\n===== Evaluation =====")
    print("ROUGE scores:")
    for m, score in rouge_res.items():
        print(f"  {m}: {score:.4f}")
    print(f"BERTScore P/R/F: {avg_p:.4f}/{avg_r:.4f}/{avg_f:.4f}")
    print(f"Toxicity: {toxic_count}/{len(predictions)} outputs flagged TOXIC")

def main():
    samples    = load_test_samples(CORPUS_PATH, SAMPLE_SIZE)
    references = [rec.get("neutral_rewrite", "") for rec in samples]

    empty_refs = sum(1 for r in references if not r.strip())
    print(f"→ {empty_refs}/{len(references)} empty reference rewrites")

    predictions = [politeify(rec["text"]) for rec in samples]

    compute_metrics(references, predictions)
    print("\n🎉 Evaluation complete!")

if __name__ == "__main__":
    print("🚀 Starting evaluation…")
    main()
