import json
from pathlib import Path
from typing import List, Dict

def load_records(input_path: str) -> List[Dict[str, str]]:
    """
    Load JSONL records from the given file path.
    """
    records = []
    for line in Path(input_path).read_text(encoding='utf-8').splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records

def chunk_records(records: List[Dict[str, str]], chunk_size: int) -> List[List[Dict[str, str]]]:
    """
    Split the list of records into chunks of size `chunk_size`.
    """
    return [records[i:i + chunk_size] for i in range(0, len(records), chunk_size)]

def save_chunks(chunks: List[List[Dict[str, str]]], output_dir: str) -> None:
    """
    Save each chunk as a separate JSONL file under `output_dir`.
    Filenames: batch_1.jsonl, batch_2.jsonl, ...
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    for idx, chunk in enumerate(chunks, start=1):
        batch_file = out_path / f"batch_{idx}.jsonl"
        with batch_file.open('w', encoding='utf-8') as f:
            for rec in chunk:
                        template = {
                                "text": rec.get("text", ""),
                                "offensive_level": "",
                                "emotion": "",
                                "neutral_rewrite": ""
                                                    }
                        f.write(json.dumps(template, ensure_ascii=False) + "\n")
        print(f"✔️  Saved {len(chunk)} records to {batch_file}")

def main():
    input_path = "data/for_annotation.jsonl"
    output_dir = "data/batches"
    chunk_size = 200

    records = load_records(input_path)
    chunks = chunk_records(records, chunk_size)

    # Save each chunk for annotation
    save_chunks(chunks, output_dir)

if __name__ == "__main__":
    main()
