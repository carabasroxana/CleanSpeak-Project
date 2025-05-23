from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained("t5-small")
model     = AutoModelForSeq2SeqLM.from_pretrained("./polite-bot")
model.to(device).eval()

app = FastAPI()

class RewriteRequest(BaseModel):
    text: str
    emotion: str = "neutral"    # optional, defaults to “neutral”

@app.post("/rewrite")
def rewrite(req: RewriteRequest):
    # you can also call format_input(req.text, req.emotion) if you need emotion-prefixing
    inputs = tokenizer(
        req.text,
        return_tensors="pt",
        truncation=True,
        padding="longest",
        max_length=128
    ).to(device)

    with torch.no_grad():
        out_ids = model.generate(**inputs, max_length=128)

    polite = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    return {"rewrite": polite}
