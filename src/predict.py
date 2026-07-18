"""Predict sentiment for one text value with the saved model artifacts."""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import clean_text

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}


def load_artifacts(model_path: str, vectorizer_path: str) -> tuple[Any, Any]:
    with Path(model_path).open("rb") as handle:
        model = pickle.load(handle)
    with Path(vectorizer_path).open("rb") as handle:
        vectorizer = pickle.load(handle)
    return model, vectorizer


def predict_text(text: str, model: Any, vectorizer: Any) -> dict[str, Any]:
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Text is empty after preprocessing.")

    vector = vectorizer.transform([cleaned])
    raw_label = model.predict(vector)[0]
    label = LABEL_MAP.get(int(raw_label), "neutral")
    result = {"text": text, "cleaned_text": cleaned, "sentiment": label}

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vector)[0]
        result["confidence"] = float(probabilities[int(raw_label)])

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict sentiment for one social post.")
    parser.add_argument("text", help="Social post text to classify.")
    parser.add_argument("--model", default="models/sentiment_model.pkl")
    parser.add_argument("--vectorizer", default="models/tfidf_vectorizer.pkl")
    args = parser.parse_args()

    model, vectorizer = load_artifacts(args.model, args.vectorizer)
    print(json.dumps(predict_text(args.text, model, vectorizer), indent=2))


if __name__ == "__main__":
    main()
