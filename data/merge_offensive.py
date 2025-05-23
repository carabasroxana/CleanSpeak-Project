import pandas as pd
from pathlib import Path


def rename_to_text(df: pd.DataFrame, source_col: str) -> pd.DataFrame:
    """
    Rename a column containing the raw post to 'text'.
    """
    return df.rename(columns={source_col: 'text'})


def merge_and_dedupe(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate a list of DataFrames and drop duplicates based on 'text'.
    """
    combined = pd.concat(dfs, ignore_index=True)
    unique = combined.drop_duplicates(subset='text')
    return unique


def save_jsonl(df: pd.DataFrame, output_path: str) -> None:
    """
    Save a DataFrame to JSONL, creating parent dirs if needed.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_json(output_path, orient='records', lines=True)
    print(f"✔️  Saved {len(df)} unique records to {output_path!r}")


if __name__ == "__main__":
    toxic = pd.read_csv('data/toxic_comments.csv')[['comment_text']].rename(columns={'comment_text': 'text'})
    off = pd.read_csv('data/olid_offensive.tsv', sep='\t')[['tweet']]
    off_d = pd.read_csv('data/davidson/dataset.csv')[['tweet']]

    off = rename_to_text(off, 'tweet')
    off_d = rename_to_text(off_d, 'tweet')

    merged_df = merge_and_dedupe([toxic, off, off_d])

    save_jsonl(merged_df, 'data/raw_offensive.jsonl')
