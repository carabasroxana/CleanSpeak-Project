# evaluate_metrics.py

import json
from pathlib import Path
from typing import List, Dict

import torch
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
from evaluate import load as load_metric

from serve import device
from model.architecture import format_input, EMOTION_TOKENS

# ──────────────────────────────────────────────────────────────────────────────
# Config
CORPUS_PATH     = Path("data/final_corpus.jsonl")
CHECKPOINT_PATH = Path("model/final.pt")
SAMPLE_SIZE     = 200
MAX_LEN         = 128   # max length for generation

# ──────────────────────────────────────────────────────────────────────────────
# Metrics
rouge_metric     = load_metric("rouge")
bertscore_metric = load_metric("bertscore")

# Toxicity pipeline with explicit truncation
tox_tokenizer = AutoTokenizer.from_pretrained(
    "unitary/toxic-bert",
    truncation=True,
    model_max_length=512
)
tox_model = AutoModelForSequenceClassification.from_pretrained("unitary/toxic-bert")
TOXICITY = pipeline(
    "text-classification",
    model=tox_model,
    tokenizer=tox_tokenizer,
    return_all_scores=False,
    device=0  # use -1 for CPU
)

# ──────────────────────────────────────────────────────────────────────────────
def load_test_samples(path: Path, sample_size: int) -> List[Dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()[-sample_size:]
    return [json.loads(l) for l in lines]

def generate_predictions(
    samples: List[Dict[str, str]],
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
) -> List[str]:
    outs = []
    model.eval()
    for rec in samples:
        inp = format_input(rec["text"], rec.get("emotion", "neutral"))
        tok = tokenizer(
            inp,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN
        ).to(device)

        with torch.no_grad():
            out_ids = model.generate(
                input_ids=tok.input_ids,
                attention_mask=tok.attention_mask,
                max_length=MAX_LEN
            )
        decoded = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        outs.append(decoded)
    return outs

def compute_metrics(
    references: List[str],
    predictions: List[str]
) -> None:
    # ROUGE returns floats under the new API
    rouge_res = rouge_metric.compute(predictions=predictions, references=references)

    # BERTScore returns lists of per-sample scores → average them
    bert_res = bertscore_metric.compute(
        predictions=predictions,
        references=references,
        lang="en"
    )
    avg_p = sum(bert_res["precision"]) / len(bert_res["precision"]) if bert_res["precision"] else 0.0
    avg_r = sum(bert_res["recall"])    / len(bert_res["recall"])    if bert_res["recall"]    else 0.0
    avg_f = sum(bert_res["f1"])        / len(bert_res["f1"])        if bert_res["f1"]        else 0.0

    # Toxicity (truncate each call)
    tox_results = [
        TOXICITY(pred, truncation=True, max_length=512)[0]
        for pred in predictions
    ]
    toxic_count = sum(1 for r in tox_results if r.get("label") == "TOXIC")

    # Print everything
    print("\n===== Evaluation =====")
    print("ROUGE scores:")
    for m, score in rouge_res.items():
        print(f"  {m}: {score:.4f}")

    print(f"BERTScore P/R/F: {avg_p:.4f}/{avg_r:.4f}/{avg_f:.4f}")
    print(f"Toxicity: {toxic_count}/{len(predictions)} outputs flagged TOXIC")

def main():
    # 1) Load T5 + tokenizer + checkpoint
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model     = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)

    state = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state)

    tokenizer.add_special_tokens({"additional_special_tokens": EMOTION_TOKENS})
    model.resize_token_embeddings(len(tokenizer))

    # 2) Load samples & references
    samples    = load_test_samples(CORPUS_PATH, SAMPLE_SIZE)
    references = [rec.get("neutral_rewrite", "") for rec in samples]

    empty_refs = sum(1 for r in references if not r.strip())
    print(f"→ {empty_refs}/{len(references)} empty reference rewrites")

    # 3) Generate predictions
    predictions = generate_predictions(samples, model, tokenizer)

    # 4) Compute & display metrics
    compute_metrics(references, predictions)
    print("\n🎉 Evaluation complete!")

if __name__ == "__main__":
    print("🚀 Starting evaluation…")
    main()
