import os
from pathlib import Path

def download_jigsaw(dest_dir="data"):
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    cmd = (
        "kaggle competitions download "
        "-c jigsaw-toxic-comment-classification-challenge "
        f"-p {dest_dir}"
    )
    os.system(cmd)

def clone_repo(repo_url, dest_dir):
    Path(dest_dir).parent.mkdir(parents=True, exist_ok=True)
    os.system(f"git clone {repo_url} {dest_dir}")

if __name__ == "__main__":
    download_jigsaw()
    clone_repo(
        "https://github.com/t-davidson/hate-speech-and-offensive-language",
        "data/davidson"
    )
