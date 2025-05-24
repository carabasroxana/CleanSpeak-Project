import requests

payload = {
    "text": "Why are you so late, damn!",
    "emotion": "anger"
}

resp = requests.post("http://localhost:8000/rewrite", json=payload)
print(resp.json())
