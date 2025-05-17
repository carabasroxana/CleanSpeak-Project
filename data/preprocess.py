import json
import re
from pathlib import Path
from langdetect import detect
from typing import List, Dict

def load_raw_texts(input_path: str) -> List[str]:
    """
    Load raw offensive texts from a JSONL file.
    """
    lines = Path(input_path).read_text().splitlines()
    return [json.loads(line)["text"] for line in lines]

def clean_text(text: str) -> str:
    """
    Remove URLs, mentions, and extra whitespace.
    """
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    return re.sub(r"\s+", " ", text).strip()

def is_valid(text: str, min_words: int = 3) -> bool:
    """
    Check if text has a minimum word count and is English.
    """
    if len(text.split()) < min_words:
        return False
    try:
        return detect(text) == "en"
    except Exception:
        return False

def filter_texts(texts: List[str]) -> List[Dict[str, str]]:
    """
    Clean and filter a list of texts, returning records ready for annotation.
    """
    cleaned = []
    for t in texts:
        t_clean = clean_text(t)
        if is_valid(t_clean):
            cleaned.append({"text": t_clean})
    return cleaned

def save_jsonl(records: List[Dict[str, str]], output_path: str) -> None:
    """
    Save a list of dictionaries to a JSONL file, ensuring directory exists.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(output_path).open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"✔️ Saved {len(records)} records to {output_path!r}")

def main():
    input_path = "data/raw_offensive.jsonl"
    output_path = "data/for_annotation.jsonl"

    texts = load_raw_texts(input_path)
    records = filter_texts(texts)
    save_jsonl(records, output_path)

if __name__ == "__main__":
    main()
