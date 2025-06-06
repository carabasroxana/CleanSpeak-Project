import json
import html
import sys
from pathlib import Path
from typing import List, Dict

import torch
from transformers import pipeline

print(">>> Checking for GPU...")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(">>> Found CUDA device! Using:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print(">>> No GPU detected, falling back to CPU.")

print(f">>> Setting device index for pipelines → {device}\n")

pipeline_device = 0 if device.type == "cuda" else -1

print(">>> Building zero‐shot classification pipeline (offense)…")
offense_clf = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=pipeline_device,
)

print(">>> Building zero‐shot classification pipeline (emotion)…")
emotion_clf = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=pipeline_device,
)

print(">>> Building T5‐based rewriting pipeline…")
rewriter = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    tokenizer="google/flan-t5-base",
    device=pipeline_device,
    max_length=128,
    clean_up_tokenization_spaces=True,
)

OFFENSE_LABELS = ["mild", "strong"]
EMOTION_LABELS = ["anger", "sadness", "sarcasm", "fear", "joy", "neutral"]


def annotate_file(in_path: Path, out_path: Path) -> None:
    """
    Read each JSONL line in `in_path`, classify offense‐level and emotion, then
    generate a “neutral_rewrite” via the rewriter pipeline. Finally, write out
    a new JSONL with additional keys: "offensive_level", "emotion", "neutral_rewrite".
    """
    out_path.parent.mkdir(exist_ok=True, parents=True)

    with in_path.open("r", encoding="utf-8") as f_count:
        total_lines = sum(1 for _ in f_count)

    print(f"\n>>> Annotating {in_path.name} ({total_lines} total lines) …")

    with in_path.open("r", encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout:

        for idx, line in enumerate(fin, start=1):
            rec = json.loads(line)
            raw_text = rec.get("text", "").strip()
            if not raw_text:
                continue

            normalized_text = html.unescape(raw_text)

            off_res = offense_clf(
                sequences=normalized_text,
                candidate_labels=OFFENSE_LABELS,
                multi_label=False
            )
            predicted_offense = off_res["labels"][0]

            emo_res = emotion_clf(
                sequences=normalized_text,
                candidate_labels=EMOTION_LABELS,
                multi_label=False
            )
            predicted_emotion = emo_res["labels"][0]

            prompt = (
                f"Preserve the {predicted_emotion} tone but rewrite the following sentence "
                f"to remove insults, profanity, or slurs:\n\n"
                f"{normalized_text}"
            )

            rewrite_res = rewriter(prompt)
            generated = rewrite_res[0]["generated_text"].strip()

            rec["offensive_level"] = predicted_offense
            rec["emotion"] = predicted_emotion
            rec["neutral_rewrite"] = generated

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

            percent_done = (idx / total_lines) * 100
            print(f"  → {idx}/{total_lines} ({percent_done:6.2f}%)", end="\r")

    print(f"\n✅ Finished annotating {in_path.name} → {out_path.name}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python auto_annotate.py <input_path_or_dir> [<output_dir>]")
        sys.exit(1)

    target = Path(sys.argv[1])
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else target.parent / "auto_annotated"

    if target.is_dir():
        outdir.mkdir(exist_ok=True, parents=True)
        for path in sorted(target.glob("*.jsonl")):
            out_path = outdir / path.name.replace(".jsonl", "_auto.jsonl")
            annotate_file(path, out_path)
    else:
        out_path = outdir / target.name.replace(".jsonl", "_auto.jsonl")
        annotate_file(target, out_path)
