from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time

app = Flask(__name__)

print("Loading AI model...")

model_name = "sshleifer/distilbart-cnn-12-6"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

print("Model loaded successfully!")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/summarize", methods=["POST"])
def summarize():

    data = request.get_json()

    article = data.get("article", "").strip()

    if not article:
        return jsonify({
            "error": "Please enter an article."
        })

    start_time = time.time()

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

    end_time = time.time()

    original_words = len(article.split())
    summary_words = len(summary.split())

    if original_words > 0:
        reduction = (
            1 - summary_words / original_words
        ) * 100
    else:
        reduction = 0

    return jsonify({
        "summary": summary,
        "original_words": original_words,
        "summary_words": summary_words,
        "reduction": round(reduction, 1),
        "time": round(end_time - start_time, 2)
    })


if __name__ == "__main__":
    app.run(debug=True)