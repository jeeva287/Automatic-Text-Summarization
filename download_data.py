from datasets import load_dataset

print("Loading dataset...")

dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")

article = dataset["train"][0]["article"]
summary = dataset["train"][0]["highlights"]

print("\n--- ARTICLE ---")
print(article)

print("\n--- SUMMARY ---")
print(summary)