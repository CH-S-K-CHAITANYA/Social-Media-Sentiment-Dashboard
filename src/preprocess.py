"""
preprocess.py
-------------
Handles all text cleaning and NLP preprocessing steps:
  - Lowercase conversion
  - URL / mention / hashtag removal
  - Punctuation & number stripping
  - Stopword removal
  - Lemmatization
"""

import re
import string
import pandas as pd

try:
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    STOP_WORDS = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
except (ImportError, LookupError):
    lemmatizer = None
    STOP_WORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "he", "in", "is", "it", "its", "of", "on", "or", "that",
        "the", "this", "to", "was", "were", "will", "with", "you", "your",
        "i", "me", "my", "we", "our", "they", "their", "them", "so",
        "very", "once", "all", "any", "but", "if", "then", "than",
    }
# Keep negation words — they flip sentiment!
NEGATION_WORDS = {"not", "no", "never", "neither", "nor", "none", "cannot",
                  "couldn't", "didn't", "doesn't", "don't", "hadn't",
                  "hasn't", "haven't", "isn't", "mightn't", "mustn't",
                  "needn't", "shouldn't", "wasn't", "weren't", "won't",
                  "wouldn't"}
STOP_WORDS -= NEGATION_WORDS


def lemmatize_word(word: str) -> str:
    if lemmatizer is not None:
        return lemmatizer.lemmatize(word)

    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s"):
        return word[:-1]
    return word


# ──────────────────────────────────────────────
# CORE CLEANING FUNCTION
# ──────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Full pipeline:
    raw text → lowercase → no URLs → no mentions → no hashtags
             → no punctuation → no numbers → no extra spaces → lemmatized
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove @mentions
    text = re.sub(r"@\w+", "", text)

    # Remove #hashtags (keep the word, drop the #)
    text = re.sub(r"#(\w+)", r"\1", text)

    # Remove emojis (basic unicode range)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize → remove stopwords → lemmatize
    tokens = text.split()
    tokens = [lemmatize_word(w) for w in tokens if w not in STOP_WORDS and len(w) > 1]

    return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Apply clean_text to every row.
    Adds a 'cleaned_text' column and drops rows that are empty after cleaning.
    """
    print("🔄  Preprocessing text...")
    df = df.copy()
    df["cleaned_text"] = df[text_col].apply(clean_text)

    # Drop rows where cleaned text is empty
    before = len(df)
    df = df[df["cleaned_text"].str.strip() != ""].reset_index(drop=True)
    after = len(df)
    print(f"   Removed {before - after} empty rows after cleaning.")
    print(f"✅  Preprocessing complete. {after} rows ready.")
    return df


# ──────────────────────────────────────────────
# QUICK TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    samples = [
        "Absolutely LOVE this product!!! 🔥🔥 #trending @Zomato",
        "Worst app ever. Crashes every time http://example.com",
        "Just received my order. Will update later. 😊",
        "The food was NOT good at all. Very disappointed.",
    ]
    print("\n── Raw vs Cleaned ──")
    for s in samples:
        print(f"  IN : {s}")
        print(f"  OUT: {clean_text(s)}")
        print()
