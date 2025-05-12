import pandas as pd
from pathlib import Path


def load_jigsaw(path: str) -> pd.DataFrame:
    """
    Load the Jigsaw Toxic Comment dataset and filter toxic comments.
    """
    df = pd.read_csv(path)
    toxic = df[df["toxic"] == 1][["comment_text"]].rename(columns={"comment_text": "text"})
    return toxic


def load_olid(path: str) -> pd.DataFrame:
    """
    Load the OLID dataset and filter offensive tweets.
    """
    df = pd.read_csv(path, sep="\t")
    offensive = df[df["subtask_a"] == "OFF"][["tweet"]].rename(columns={"tweet": "text"})
    return offensive


def load_davidson(path: str) -> pd.DataFrame:
    """
    Load the Davidson et al. dataset and filter hate or offensive labels.
    """
    df = pd.read_csv(path)
    # label 1 = hate speech, 2 = offensive language
    off_d = df[df["label"].isin([1, 2])][["tweet"]].rename(columns={"tweet": "text"})
    return off_d


def merge_and_save(output_path: str, *dfs: pd.DataFrame):
    """
    Concatenate given DataFrames, dedupe on 'text', and save as JSONL.
    """
    combined = pd.concat(dfs, ignore_index=True)
    unique = combined.drop_duplicates(subset=["text"])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    unique.to_json(output_path, orient="records", lines=True)
    print(f"Saved {len(unique)} unique offensive texts to {output_path!r}")


if __name__ == "__main__":
    jigsaw_path = "data/train.csv"
    olid_path = "data/olid-training-v1.0.tsv"
    davidson_path = "data/davidson/dataset.csv"
    output_path = "data/raw_offensive.jsonl"

    jigsaw_df = load_jigsaw(jigsaw_path)
    olid_df = load_olid(olid_path)
    davidson_df = load_davidson(davidson_path)

    merge_and_save(output_path, jigsaw_df, olid_df, davidson_df)
