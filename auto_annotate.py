import json
from pathlib import Path
from transformers import pipeline

offense_clf = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0  # or remove if you’re on CPU
)
emotion_clf = offense_clf  # reuse the same model

rewriter = pipeline(
    "text2text-generation",
    model="t5-small",
    tokenizer="t5-small",
    device=0,
    max_length=128,
    clean_up_tokenization_spaces=True
)

OFFENSE_LABELS = ["mild", "strong"]
EMOTION_LABELS = ["anger", "sadness", "sarcasm", "fear", "joy", "neutral"]

def annotate_file(in_path: Path, out_path: Path):
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with in_path.open("r", encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            rec = json.loads(line)
            text = rec["text"]

            off = offense_clf(
                sequences=text,
                candidate_labels=OFFENSE_LABELS
            )["labels"][0]

            emo = emotion_clf(
                sequences=text,
                candidate_labels=EMOTION_LABELS
            )["labels"][0]

            prompt = (
                f"preserve the {emo} tone but make it non-offensive:\n\n"
                f"{text}"
            )
            rewrite = rewriter(prompt)[0]["generated_text"]

            rec["offensive_level"] = off
            rec["emotion"] = emo
            rec["neutral_rewrite"] = rewrite

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    import sys
    from glob import glob

    target = Path(sys.argv[1])
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else target.parent / "auto_annotated"

    if target.is_dir():
        for path in sorted(target.glob("*.jsonl")):
            annotate_file(path, outdir / path.name.replace(".jsonl", "_auto.jsonl"))
            print("✅", path.name)
    else:
        outpath = outdir / target.name.replace(".jsonl", "_auto.jsonl")
        annotate_file(target, outpath)
        print("✅", outpath.name)
