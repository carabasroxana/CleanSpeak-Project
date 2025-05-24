import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from model.architecture import format_input  # ← your helper from architecture.py

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def rewrite_text(tokenizer, model, text: str, emotion: str, max_len: int = 128) -> str:
    inp = format_input(text, emotion)
    print(">> model sees:", inp)
    batch = tokenizer(
        inp,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_len
    ).to(device)
    with torch.no_grad():
        out_ids = model.generate(**batch, max_length=max_len)
    return tokenizer.decode(out_ids[0], skip_special_tokens=True)

def main():
    p = argparse.ArgumentParser("CLI for your polite-rewriter T5 model")
    p.add_argument(
        "text",
        nargs="+",
        help="(Possibly offensive) sentence to rewrite"
    )
    p.add_argument(
        "--model-dir",
        type=str,
        default="./polite-bot",
        help="Where your fine-tuned model lives"
    )
    p.add_argument(
        "--emotion",
        type=str,
        default="neutral",
        choices=["anger","sadness","joy","fear","sarcasm","neutral"],
        help="Which emotional tone to preserve"
    )
    args = p.parse_args()

    raw_text = " ".join(args.text)
    print(f"\n→ Loading model from `{args.model_dir}` on {device}\n")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model     = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
    model.to(device).eval()

    polite = rewrite_text(tokenizer, model, raw_text, args.emotion)
    print(f"\n✍️  Rewritten:\n{polite}\n")

if __name__ == "__main__":
    main()
