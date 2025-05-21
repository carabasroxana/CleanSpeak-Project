from pathlib import Path
from flask import Flask, request, jsonify
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from model.architecture import load_model_and_tokenizer, format_input, EMOTION_TOKENS

# ──────────────────────────────────────────────────────────────────────────────
# Device setup (so you can move your model/tensors to GPU/MPS if available)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

MODEL_DIR  = Path("model")
MODEL_FILE = MODEL_DIR / "final.pt"
HOST       = "0.0.0.0"
PORT       = 5000


def load_model():
    # 1) Base T5
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model     = T5ForConditionalGeneration.from_pretrained("t5-base")

    # 2) Load checkpoint
    state = torch.load(MODEL_FILE, map_location="cpu")
    model.load_state_dict(state)

    # 3) Add emotion tokens + resize
    tokenizer.add_special_tokens({"additional_special_tokens": EMOTION_TOKENS})
    model.resize_token_embeddings(len(tokenizer))

    # 4) Move to CPU/GPU/MPS, set eval
    model.to(device)
    model.eval()
    return model, tokenizer



def create_app(model, tokenizer):
    """
    Create and configure the Flask application.

    Args:
        model: The sequence-to-sequence model.
        tokenizer: Tokenizer for encoding inputs and decoding outputs.

    Returns:
        app (Flask): The Flask server instance.
    """
    app = Flask(__name__)

    @app.route("/rewrite", methods=["POST"])
    def rewrite():
        """
        Endpoint: Accepts JSON {"text": str, "emotion": str}
        Returns: {"rewrite": str}
        """
        data = request.get_json(force=True)
        text = data.get("text", "")
        emotion = data.get("emotion", "neutral")

        # Prepare input and generate output
        inp = format_input(text, emotion)
        tokens = tokenizer(
            inp,
            return_tensors="pt",
            truncation=True,
            padding="longest"
        )
        with torch.no_grad():
            out_ids = model.generate(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
                max_length=128
            )
        result = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        return jsonify({"rewrite": result})

    return app


def main():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {MODEL_FILE}")

    model, tokenizer = load_model()
    app = create_app(model, tokenizer)
    app.run(host=HOST, port=PORT)


if __name__ == "__main__":
    main()
