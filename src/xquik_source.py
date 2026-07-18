"""
Load live X posts from Xquik and normalize them for the dashboard.
"""

import os
from typing import Any

import pandas as pd

from src.preprocess import clean_text

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
RESULT_COLUMNS = [
    "text",
    "cleaned_text",
    "sentiment",
    "brand",
    "platform",
    "location",
    "likes",
    "retweets",
    "timestamp",
]


def _read_field(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _count(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _predict_label(text: str, model: Any, vectorizer: Any) -> str:
    cleaned_text = clean_text(text)
    vector = vectorizer.transform([cleaned_text])
    label = model.predict(vector)[0]
    return LABEL_MAP.get(label, "neutral")


def load_xquik_posts(
    query: str,
    brand: str,
    limit: int,
    model: Any,
    vectorizer: Any,
    client: Any = None,
) -> pd.DataFrame:
    if model is None or vectorizer is None:
        raise RuntimeError("Train the sentiment model before loading live X posts.")

    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("Enter an X search query first.")
    normalized_brand = brand.strip()
    if not normalized_brand:
        raise ValueError("Enter the brand being monitored.")
    if not 1 <= limit <= 200:
        raise ValueError("Limit must be between 1 and 200 posts.")

    if client is None:
        try:
            from x_twitter_scraper import XTwitterScraper
        except ImportError as exc:
            raise RuntimeError(
                "Install x_twitter_scraper to load live X posts from Xquik."
            ) from exc

        api_key = os.environ.get("XQUIK_API_KEY") or os.environ.get(
            "X_TWITTER_SCRAPER_API_KEY"
        )
        if not api_key:
            raise RuntimeError("Set XQUIK_API_KEY before loading live X posts.")
        client = XTwitterScraper(api_key=api_key)
    page = client.x.tweets.search(q=normalized_query, limit=limit)

    rows = []
    for tweet in _read_field(page, "tweets", default=[]):
        text = str(_read_field(tweet, "text", default="")).strip()
        if not text:
            continue

        cleaned_text = clean_text(text)
        rows.append(
            {
                "text": text,
                "cleaned_text": cleaned_text,
                "sentiment": _predict_label(text, model, vectorizer),
                "brand": normalized_brand,
                "platform": "X",
                "location": "Live Search",
                "likes": _count(_read_field(tweet, "like_count", "likeCount")),
                "retweets": _count(
                    _read_field(tweet, "retweet_count", "retweetCount")
                ),
                "timestamp": _read_field(
                    tweet, "created_at", "createdAt", default=pd.NaT
                ),
            }
        )

    return pd.DataFrame(rows, columns=RESULT_COLUMNS)
