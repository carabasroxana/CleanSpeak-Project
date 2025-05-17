from pathlib import Path
from flask import Flask, request, jsonify
import torch

from model.architecture import load_model_and_tokenizer, format_input

MODEL_DIR = Path("model")
MODEL_FILE = MODEL_DIR / "final.pt"
HOST = "0.0.0.0"
PORT = 5000


def load_model():
    """
    Load the fine-tuned model and tokenizer from disk.

    Returns:
        model (torch.nn.Module): The loaded T5 model.
        tokenizer: The corresponding tokenizer.
    """
    model, tokenizer = load_model_and_tokenizer()
    state_dict = torch.load(MODEL_FILE, map_location="cpu")
    model.load_state_dict(state_dict)
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
