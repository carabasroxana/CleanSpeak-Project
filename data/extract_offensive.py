import pandas as pd
from pathlib import Path
from typing import List

# def load_jigsaw(path: str) -> pd.DataFrame:
#     """
#     Load Jigsaw Toxic Comment dataset and return toxic comments as 'text'.
#     """
#     df = pd.read_csv(path)
#     toxic = df.loc[df['toxic'] == 1, ['comment_text']]
#     return toxic.rename(columns={'comment_text': 'text'})

def load_olid(path: str) -> pd.DataFrame:
    """Load OLID dataset…"""
    df = pd.read_csv(path, sep="\t")
    offensive = df[df["subtask_a"] == "OFF"][["tweet"]]
    return offensive.rename(columns={"tweet": "text"})

def load_davidson(path: str) -> pd.DataFrame:
    """
    Load the moved Davidson CSV (class 0 = hate, 1 = offensive).
    """
    df = pd.read_csv(path)
    subset = df[df["class"].isin([0, 1])][["tweet"]]
    return subset.rename(columns={"tweet": "text"})


def merge_datasets(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate dataframes and drop duplicates based on 'text'.
    """
    combined = pd.concat(dfs, ignore_index=True)
    return combined.drop_duplicates(subset=['text'])

def save_jsonl(df: pd.DataFrame, output_path: str) -> None:
    """
    Save a DataFrame to JSONL, ensuring the output directory exists.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(out_path, orient='records', lines=True)
    print(f"✔️ Saved {len(df)} records to {output_path!r}")

def main():
    # jigsaw_path   = 'data/train.csv'
    olid_path     = 'data/olid-training-v1.0.tsv'
    davidson_path = 'data/davidson/dataset.csv'
    output_path   = 'data/raw_offensive.jsonl'

    # jigsaw_df   = load_jigsaw(jigsaw_path)
    olid_df = load_olid("data/olid-training-v1.0.tsv")
    davidson_df = load_davidson("data/dataset.csv")

    merged_df = merge_datasets([olid_df, davidson_df])

    save_jsonl(merged_df, output_path)

if __name__ == '__main__':
    main()
