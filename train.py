import os
# ──────────────────────────────────────────────────────────────────────────────
# Disable MPS memory limits entirely and free cache between batches
# Only relevant for MPS; won't affect CUDA
os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = '0.0'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

import gc
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from transformers import T5TokenizerFast, T5ForConditionalGeneration

from model.architecture import format_input

# ──────────────────────────────────────────────────────────────────────────────
# Config
DATA_PATH       = Path("data/final_corpus.jsonl")
MODEL_SAVE_PATH = Path("model/final.pt")

# Lower batch size to reduce peak MPS memory usage (won't affect CUDA)
BATCH_SIZE      = 8
MAX_LEN         = 128
EPOCHS          = 3
LEARNING_RATE   = 3e-5
WARMUP_STEPS    = 100

# ──────────────────────────────────────────────────────────────────────────────
# Device setup: prefer CUDA, then MPS, then CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"▶️  Training on device: {device}")

# ──────────────────────────────────────────────────────────────────────────────
class CorpusDataset(Dataset):
    """
    PyTorch Dataset for the offensive-to-neutral parallel corpus.
    """
    def __init__(self, file_path: Path, tokenizer: Any, max_len: int = MAX_LEN):
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.records: List[Dict[str, str]] = [
            json.loads(line)
            for line in file_path.read_text(encoding="utf-8").splitlines()
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec      = self.records[idx]
        src_text = format_input(rec["text"], rec.get("emotion", "neutral"))
        tgt_text = rec.get("neutral_rewrite", "")

        enc = self.tokenizer(
            src_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        tgt = self.tokenizer(
            tgt_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels":         tgt["input_ids"].squeeze(),
        }

# ──────────────────────────────────────────────────────────────────────────────
def load_model_and_tokenizer(model_name: str):
    tokenizer = T5TokenizerFast.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return model, tokenizer

def train():
    # 1) Load model & tokenizer
    model, tokenizer = load_model_and_tokenizer("t5-base")
    model.to(device)

    # 2) Prepare data
    dataset    = CorpusDataset(DATA_PATH, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3) Optimizer & scheduler
    optimizer   = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(dataloader) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps
    )

    # 4) Training loop
    model.train()
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0

        for i, batch in enumerate(
            tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}", unit="batch")
        ):
            if i == 0:
                print("🔄 Starting first batch (this may take a bit)...")

            optimizer.zero_grad()

            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            # ─────────────── DISABLE LEGACY CACHE ───────────────
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
                return_dict=True
            )
            # ───────────────────────────────────────────────────────

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            # free up unused MPS allocations
            if device.type == "mps":
                torch.mps.empty_cache()
            gc.collect()

            if i == 0:
                print("✅ First batch done—moving on!")

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"\n→ Epoch {epoch}/{EPOCHS} finished — avg loss: {avg_loss:.4f}\n")

    # 5) Save checkpoint
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✔️  Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
