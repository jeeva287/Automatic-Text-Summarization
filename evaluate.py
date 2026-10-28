from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from rouge_score import rouge_scorer

print("Loading model...")

model_name = "sshleifer/distilbart-cnn-12-6"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

print("Loading dataset...")

dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")

# Get one article and its predefined reference summary
article = dataset["test"][0]["article"]
reference = dataset["test"][0]["highlights"]

# Generate summary
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

generated = tokenizer.decode(
    summary_ids[0],
    skip_special_tokens=True
)

print("\n--- GENERATED SUMMARY ---")
print(generated)

print("\n--- REFERENCE SUMMARY ---")
print(reference)

# Calculate ROUGE
scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=True
)

scores = scorer.score(reference, generated)

print("\n--- ROUGE SCORES ---")

for name, score in scores.items():
    print(f"{name.upper()}: {score.fmeasure:.2f}")