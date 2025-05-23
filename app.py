from pathlib import Path
from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from model.architecture import format_input, EMOTION_TOKENS

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

MODEL_DIR  = Path("model")
BOT_DIR    = Path("polite-bot")
HOST, PORT = "0.0.0.0", 5000


tokenizer = AutoTokenizer.from_pretrained("t5-small")
model     = AutoModelForSeq2SeqLM.from_pretrained(str(BOT_DIR))
tokenizer.add_special_tokens({"additional_special_tokens": EMOTION_TOKENS})
model.resize_token_embeddings(len(tokenizer))

model.to(device).eval()

def create_app():
    app = Flask(__name__)

    @app.route("/rewrite", methods=["POST"])
    def rewrite():
        data    = request.get_json(force=True)
        text    = data.get("text", "")
        emotion = data.get("emotion", "neutral")

        inp = format_input(text, emotion)
        tokens = tokenizer(
            inp,
            return_tensors="pt",
            truncation=True,
            padding="longest",
            max_length=128
        ).to(device)

        with torch.no_grad():
            out_ids = model.generate(
                **tokens,
                max_length=128
            )

        rewrite = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        return jsonify({"rewrite": rewrite})

    return app

if __name__ == "__main__":
    if not BOT_DIR.exists():
        raise FileNotFoundError(f"Folder not found: {BOT_DIR}")
    create_app().run(host=HOST, port=PORT)
