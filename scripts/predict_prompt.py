import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "models/tinybert_poison_classifier_v3"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

print("Prompt Injection Detector")
print("Type a prompt and press enter (Ctrl+C to exit)\n")

while True:
    text = input("Prompt > ")

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    pred = torch.argmax(logits, dim=1).item()
    confidence = torch.softmax(logits, dim=1)[0][pred].item()

    if pred == 1:
        verdict = "POISONED / PROMPT INJECTION"
    else:
        verdict = "BENIGN"

    print(f"Verdict: {verdict}")
    print(f"Confidence: {confidence:.3f}\n")
