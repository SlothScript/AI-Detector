#!/usr/bin/env python3
"""
Interactive Flask web app for AI vs Human text detection.
- Uses spaCy for sentence segmentation.
- Highlights text inline with color intensity based on model confidence.
- Supports auto-analysis after idle typing.
- Keeps existing highlights for unchanged text.
"""

from flask import Flask, render_template, request, jsonify
import joblib
from pathlib import Path
import spacy
import hashlib

app = Flask(__name__)

# Load the model and spaCy
ROOT_DIR   = Path(__file__).parent.parent
MODEL_PATH = ROOT_DIR / "models" / "ai_detector.pkl"
pipeline   = joblib.load(MODEL_PATH)
nlp        = spacy.load("en_core_web_sm")

# ------------------ Utilities ------------------

def hash_text(s: str) -> str:
    """Create a short hash for a string to track sentence identity."""
    return hashlib.md5(s.strip().encode("utf-8")).hexdigest()[:10]

def split_sentences(text: str):
    """Use spaCy to split text into sentences."""
    doc = nlp(text)
    return [sent.text for sent in doc.sents if sent.text.strip()]

def color_intensity(prob: float) -> str:
    """Map probability (0=Human, 1=AI) to rgba highlight."""
    base_opacity = abs(prob - 0.5) * 1.8
    opacity = min(max(base_opacity, 0.15), 0.9)
    if prob >= 0.5:
        return f"background-color: rgba(255, 80, 80, {opacity:.2f});"
    else:
        return f"background-color: rgba(80, 255, 120, {opacity:.2f});"

def analyze_text(text, old_results=None):
    """Analyze text, preserving highlights for unchanged sentences."""
    sentences = split_sentences(text)
    results = []

    for s in sentences:
        sid = hash_text(s)
        # Reuse cached probability if same sentence as before
        if old_results and sid in old_results:
            prob = old_results[sid]
        else:
            prob = float(pipeline.predict_proba([s])[0][1])
        results.append((s, prob, sid))
    return results

def highlight(results):
    """Return text with inline highlighting spans."""
    highlighted = ""
    for s, prob, sid in results:
        style = color_intensity(prob)
        highlighted += f"<span data-id='{sid}' style='{style}'>{s}</span> "
    return highlighted.strip()

# ------------------ Routes ------------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text", "")
    old_results = data.get("oldResults", {})

    results = analyze_text(text, old_results)
    html = highlight(results)

    # Return both highlighted HTML and cached result dict
    result_dict = {sid: prob for _, prob, sid in results}
    return jsonify({"html": html, "results": result_dict})

# ------------------ Run ------------------

if __name__ == "__main__":
    app.run(debug=True)
