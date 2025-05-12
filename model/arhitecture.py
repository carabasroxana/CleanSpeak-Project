from transformers import T5ForConditionalGeneration, T5Tokenizer
# Define the base model name and emotion tokens
MODEL_NAME = "t5-base"
EMOTION_TOKENS = [
    "<anger>",
    "<sadness>",
    "<sarcasm>",
    "<fear>",
    "<joy>",
    "<neutral>"
]


def load_model_and_tokenizer() -> tuple[T5ForConditionalGeneration, T5Tokenizer]:
    """
    Load the T5 model and tokenizer, add emotion tokens,
    and resize the model's embeddings to include them.

    Returns:
        model (T5ForConditionalGeneration): The sequence-to-sequence model.
        tokenizer (T5Tokenizer): The tokenizer with additional tokens.
    """
    # Load pretrained tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    # Add custom emotion tokens to the tokenizer
    tokenizer.add_special_tokens({"additional_special_tokens": EMOTION_TOKENS})

    # Resize model embeddings to match the updated tokenizer
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def format_input(text: str, emotion: str) -> str:
    """
    Prepend the emotion token to the input text.

    Args:
        text (str): The original offensive text.
        emotion (str): One of the defined emotions (no brackets).

    Returns:
        formatted (str): A string like '<emotion>: text'.
    """
    token = f"<{emotion}>"
    return f"{token}: {text}"
