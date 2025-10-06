#!/usr/bin/env python
"""
Predict AI vs Human for a list of sentences.
Usage:
    python predict.py "Your sentence here." "Another sentence."
"""

import sys
import joblib
from pathlib import Path

ROOT_DIR   = Path(__file__).parent.parent
MODEL_PATH = ROOT_DIR / "models" / "ai_detector.pkl"

# Load the pipeline
pipeline = joblib.load(MODEL_PATH)

def predict(text: str) -> str:
    prob = pipeline.predict_proba([text])[0][1]  # probability that it is AI
    label = "AI" if prob >= 0.5 else "Human"
    return f"✅ {label}  (p= {prob:.2f})"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Provide at least one sentence to classify.")
        sys.exit(1)

    for sentence in sys.argv[1:]:
        print(predict(sentence))
