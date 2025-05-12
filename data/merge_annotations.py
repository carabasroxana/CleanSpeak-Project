import json
from pathlib import Path
from typing import List, Dict

def load_batches(batch_dir: str) -> List[Dict[str, str]]:
    """
    Load all JSONL records from batch files in the given directory.
    """
    records: List[Dict[str, str]] = []
    batch_files = sorted(Path(batch_dir).glob('batch_*.jsonl'))
    for file_path in batch_files:
        for line in file_path.read_text(encoding='utf-8').splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records

def save_merged(records: List[Dict[str, str]], output_path: str) -> None:
    """
    Save a list of records to a single JSONL file, ensuring directory exists.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open('w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f"✔️ Saved {len(records)} merged records to {output_path!r}")

def main():
    batch_dir = 'data/batches'
    output_path = 'data/final_corpus.jsonl'

    # Load all annotated batches
    records = load_batches(batch_dir)

    # Save them into one merged file
    save_merged(records, output_path)

if __name__ == '__main__':
    main()