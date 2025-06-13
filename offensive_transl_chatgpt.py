import torch
from sklearn.pipeline            import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes         import MultinomialNB
from sklearn.metrics             import classification_report, accuracy_score
from sklearn.model_selection     import train_test_split

from datasets      import Dataset
from transformers  import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
)

corpus = [
    {"text": "Shut the hell up, you idiot!",
     "offensive_level": "strong",
     "neutral_rewrite": "Please be quiet, I find that upsetting."},
    {"text": "I hate this stupid interface.",
     "offensive_level": "mild",
     "neutral_rewrite": "I'm not a fan of this interface."},
    {"text": "What the fuck is going on here?",
     "offensive_level": "strong",
     "neutral_rewrite": "Could you explain what's happening?"},
    {"text": "Damn, you nailed that solution!",
     "offensive_level": "mild",
     "neutral_rewrite": "Great job on that solution!"},
    {"text": "You're ridiculous, get a grip!",
     "offensive_level": "strong",
     "neutral_rewrite": "Let's stay calm and work this out."},
    {"text": "Stop crying like a baby.",
     "offensive_level": "mild",
     "neutral_rewrite": "Let's remain composed, please."},
    {"text": "Holy shit, that was amazing!",
     "offensive_level": "mild",
     "neutral_rewrite": "That was truly impressive!"},
    {"text": "I can't stand these morons around me.",
     "offensive_level": "strong",
     "neutral_rewrite": "I'm finding the situation frustrating."},
]

texts    = [ex["text"]            for ex in corpus]
levels   = [ex["offensive_level"] for ex in corpus]
rewrites = [ex["neutral_rewrite"] for ex in corpus]

X_train, X_test, y_train, y_test, rw_train, rw_test = train_test_split(
    texts, levels, rewrites,
    test_size=0.25, random_state=42, stratify=levels
)

nb_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=2000)),
    ("nb",    MultinomialNB()),
])
nb_pipe.fit(X_train, y_train)

print("\n=== NB BASELINE on ORIGINAL test sentences ===")
y_pred_orig = nb_pipe.predict(X_test)
print(classification_report(y_test, y_pred_orig, digits=4))
print("→ Accuracy:", accuracy_score(y_test, y_pred_orig))


print("\n=== Fine-tuning T5-small on the tiny corpus ===")
model_id = "t5-small"
tok      = AutoTokenizer.from_pretrained(model_id)
model    = AutoModelForSeq2SeqLM.from_pretrained(model_id)

hf_ds = Dataset.from_list(corpus).train_test_split(test_size=0.25, seed=42)
max_len = 64

def preprocess(batch):
    enc = tok(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=max_len
    )
    tgt = tok(
        batch["neutral_rewrite"],
        truncation=True,
        padding="max_length",
        max_length=max_len
    )
    enc["labels"] = tgt["input_ids"]
    return enc

tokenized = hf_ds.map(
    preprocess,
    batched=True,
    remove_columns=["text","offensive_level","neutral_rewrite"]
)

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./polite-bot",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        logging_steps=10,
        save_steps=50,
        save_total_limit=1,
        learning_rate=5e-5,
    ),
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tok,
)
trainer.train()

print("\n=== Scoring NB on HELD-OUT rewrites ===")
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t5_tok   = tok
t5_model = model.to(device).eval()

def rewrite_fn(text: str) -> str:
    prompt = (
        "Preserve the neutral tone but rewrite the following sentence "
        "to remove insults, profanity, or slurs:\n\n" + text
    )
    inp = t5_tok(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_len
    ).to(device)
    with torch.no_grad():
        out = t5_model.generate(
            **inp,
            max_length=max_len,
            num_beams=4,
            no_repeat_ngram_size=2,
            length_penalty=1.1
        )
    return t5_tok.decode(out[0], skip_special_tokens=True)

rewritten = [rewrite_fn(s) for s in X_test]

print("\noriginal → rewritten")
for o, r in zip(X_test, rewritten):
    print(f" • {o!r}\n   ↪ {r!r}")

print("\n=== NB on REWRITTEN test sentences ===")
y_pred_rew = nb_pipe.predict(rewritten)
print(classification_report(y_test, y_pred_rew, digits=4))
print("→ Accuracy:", accuracy_score(y_test, y_pred_rew))


print("\n=== Your turn: type an offensive sentence (ENTER to quit) ===")
while True:
    usr = input("Offensive sentence → ").strip()
    if not usr:
        print("👋 bye!")
        break
    print("↪ Non-offensive:", rewrite_fn(usr), "\n")
