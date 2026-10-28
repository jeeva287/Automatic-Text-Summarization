from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

print("Loading summarization model...")

model_name = "sshleifer/distilbart-cnn-12-6"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

print("Model loaded successfully!")
print("\nPaste your article below.")
print("When finished, press ENTER, then type END and press ENTER.\n")

lines = []

while True:
    line = input()

    if line.strip() == "END":
        break

    lines.append(line)

article = "\n".join(lines)

inputs = tokenizer(
    article,
    return_tensors="pt",
    max_length=1024,
    truncation=True
)

summary_ids = model.generate(
    inputs["input_ids"],
    max_length=80,
    min_length=20,
    num_beams=4,
    early_stopping=True
)

summary = tokenizer.decode(
    summary_ids[0],
    skip_special_tokens=True
)

print("\n--- GENERATED SUMMARY ---")
print(summary)