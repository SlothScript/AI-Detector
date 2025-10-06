#!/usr/bin/env python
"""
Train an AI‑vs‑Human sentence detector.

Usage:
    python train.py  # runs the whole pipeline
"""

import os
import re
import string
from pathlib import Path
import joblib

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline

# ------------------------------------------------------------------
# 1️⃣  Load data
# ------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"

def read_lines(path: Path) -> list[str]:
    """Read a file, strip whitespace and filter empty lines."""
    with path.open(encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

human_sentences = read_lines(DATA_DIR / "humanData.txt")
ai_sentences     = read_lines(DATA_DIR / "aiData.txt")

# ------------------------------------------------------------------
# 2️⃣  Create DataFrame
# ------------------------------------------------------------------
df = pd.DataFrame(
    {
        "sentence": human_sentences + ai_sentences,
        "label":    [0]*len(human_sentences) + [1]*len(ai_sentences)
    }
)
print(f"Dataset shape: {df.shape}")
print(df["label"].value_counts())

# ------------------------------------------------------------------
# 3️⃣  (Optional) Basic cleaning – remove URLs, emails, punctuation
# ------------------------------------------------------------------
def basic_clean(text: str) -> str:
    # Remove URLs
    text = re.sub(r"http\S+|www.\S+", "", text)
    # Remove emails
    text = re.sub(r"\S+@\S+", "", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Lowercase
    return text.lower()

df["cleaned"] = df["sentence"].apply(basic_clean)

# ------------------------------------------------------------------
# 4️⃣  Train / test split (stratified)
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["cleaned"], df["label"],
    test_size=0.20,   # 80% train, 20% test
    random_state=42,
    stratify=df["label"]
)

# ------------------------------------------------------------------
# 5️⃣  Build a pipeline: TF‑IDF → LogisticRegression
# ------------------------------------------------------------------
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),   # unigrams + bigrams
    max_features=50000,  # keep top 50k terms
    stop_words="english",
    lowercase=False      # already lowercased
)

clf = LogisticRegression(
    max_iter=2000,
    n_jobs=-1,
    class_weight="balanced",
    penalty="l2",
    C=1.0,
    solver="lbfgs"
)

pipeline = Pipeline([
    ("tfidf", tfidf),
    ("clf", clf)
])

# ------------------------------------------------------------------
# 6️⃣  Train
# ------------------------------------------------------------------
print("Training …")
pipeline.fit(X_train, y_train)

# ------------------------------------------------------------------
# 7️⃣  Evaluate
# ------------------------------------------------------------------
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=["Human", "AI"]))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== Metrics ===")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1‑Score : {f1_score(y_test, y_pred):.4f}")
print(f"ROC‑AUC  : {roc_auc_score(y_test, y_proba):.4f}")

# ------------------------------------------------------------------
# Save the model (vectoriser + classifier)
# ------------------------------------------------------------------
OUTPUT_DIR = ROOT_DIR / "models"
OUTPUT_DIR.mkdir(exist_ok=True)

joblib.dump(pipeline, OUTPUT_DIR / "ai_detector.pkl")
print(f"\nModel saved to {OUTPUT_DIR / 'ai_detector.pkl'}")
