import json
from pathlib import Path

def annotate(input_path, output_path):
    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            rec = json.loads(line)
            print("\n───")
            print("Text:", rec["text"])
            rec["offensive_level"] = input("Offensive level (mild/strong): ").strip()
            rec["emotion"]         = input("Emotion (anger/sadness/sarcasm/fear/joy/neutral): ").strip()
            rec["neutral_rewrite"] = input("Neutral rewrite: ").strip()
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    import sys
    inp = sys.argv[1]
    out = sys.argv[2]
    annotate(inp, out)
