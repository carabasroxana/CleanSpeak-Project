import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from model.architecture import load_model_and_tokenizer, format_input

DATA_PATH = Path("data/final_corpus.jsonl")
MODEL_SAVE_PATH = Path("model/final.pt")
BATCH_SIZE = 16
MAX_LEN = 128
EPOCHS = 3
LEARNING_RATE = 3e-5
WARMUP_STEPS = 100


class CorpusDataset(Dataset):
    """
    PyTorch Dataset for the offensive-to-neutral parallel corpus.
    Each item is prepared for seq2seq training.
    """

    def __init__(self, file_path: Path, tokenizer: Any, max_len: int = MAX_LEN):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.records: List[Dict[str, str]] = []
        for line in file_path.read_text(encoding="utf-8").splitlines():
            self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]
        src_text = format_input(record["text"], record.get("emotion", "neutral"))
        tgt_text = record.get("neutral_rewrite", "")

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
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": tgt["input_ids"].squeeze(),
        }


def train():
    """
    Fine-tune the T5 model on the parallel corpus.
    """
    model, tokenizer = load_model_and_tokenizer()
    dataset = CorpusDataset(DATA_PATH, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps
    )

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}/{EPOCHS} — Loss: {avg_loss:.4f}")

    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✔️ Model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()
