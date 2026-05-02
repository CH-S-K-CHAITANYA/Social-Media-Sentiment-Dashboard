"""
train_model.py
--------------
Full ML pipeline:
  1. Load & preprocess data
  2. TF-IDF vectorization
  3. Train Logistic Regression, Naive Bayes, and Random Forest
  4. Evaluate models (accuracy, classification report, confusion matrix)
  5. Save best model + vectorizer to /models/

Run: python src/train_model.py
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for servers
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)

# Add project root to path so we can import preprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import preprocess_dataframe

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
DATA_PATH    = "data/social_media_posts.csv"
MODEL_DIR    = "models"
OUTPUT_DIR   = "outputs/charts"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABEL_MAP    = {"positive": 2, "neutral": 1, "negative": 0}
LABEL_NAMES  = ["negative", "neutral", "positive"]


# ──────────────────────────────────────────────
# 1.  LOAD DATA
# ──────────────────────────────────────────────

def load_data():
    print("\n📂  Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"   Shape: {df.shape}")
    print(df["sentiment"].value_counts())
    return df


# ──────────────────────────────────────────────
# 2.  PREPARE FEATURES
# ──────────────────────────────────────────────

def prepare_features(df):
    df = preprocess_dataframe(df)
    df["label"] = df["sentiment"].map(LABEL_MAP)

    X = df["cleaned_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n🔀  Train: {len(X_train)} | Test: {len(X_test)}")

    # TF-IDF  (bigrams, max 5000 features, sub-linear TF scaling)
    print("\n🔢  Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    return X_train_vec, X_test_vec, y_train, y_test, vectorizer, df


# ──────────────────────────────────────────────
# 3.  TRAIN & EVALUATE ALL MODELS
# ──────────────────────────────────────────────

def train_evaluate(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs", multi_class="multinomial"
        ),
        "Naive Bayes": MultinomialNB(alpha=0.5),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=20, random_state=42, n_jobs=1
        ),
    }

    results = {}
    print("\n" + "="*55)
    print("  MODEL TRAINING & EVALUATION")
    print("="*55)

    for name, model in models.items():
        print(f"\n▶  Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc   = accuracy_score(y_test, preds)
        print(f"   Accuracy: {acc:.4f}  ({acc*100:.2f}%)")
        print(classification_report(y_test, preds,
                                    target_names=LABEL_NAMES,
                                    zero_division=0))
        results[name] = {"model": model, "preds": preds, "acc": acc}

    return results


# ──────────────────────────────────────────────
# 4.  CONFUSION MATRIX PLOT
# ──────────────────────────────────────────────

def plot_confusion_matrix(y_test, preds, model_name):
    cm   = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual",    fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fname = f"{OUTPUT_DIR}/confusion_matrix_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"   📊  Saved confusion matrix → {fname}")


# ──────────────────────────────────────────────
# 5.  ACCURACY COMPARISON BAR CHART
# ──────────────────────────────────────────────

def plot_accuracy_comparison(results):
    names  = list(results.keys())
    accs   = [results[n]["acc"] for n in names]
    colors = ["#4361ee", "#7209b7", "#f72585"]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(names, [a * 100 for a in accs], color=colors,
                  width=0.5, edgecolor="white", linewidth=1.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f"{acc*100:.1f}%",
                ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Model Accuracy Comparison", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fname = f"{OUTPUT_DIR}/model_accuracy_comparison.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"   📊  Saved accuracy chart → {fname}")


# ──────────────────────────────────────────────
# 6.  SENTIMENT DISTRIBUTION
# ──────────────────────────────────────────────

def plot_sentiment_distribution(df):
    counts = df["sentiment"].value_counts()
    colors = {"positive": "#06d6a0", "neutral": "#ffd166", "negative": "#ef476f"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    bars = axes[0].bar(counts.index, counts.values,
                       color=[colors[s] for s in counts.index],
                       edgecolor="white", linewidth=1.5)
    for bar, v in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 5, str(v),
                     ha="center", fontweight="bold", fontsize=11)
    axes[0].set_title("Sentiment Distribution (Count)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Count")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Pie chart
    axes[1].pie(counts.values,
                labels=[s.capitalize() for s in counts.index],
                colors=[colors[s] for s in counts.index],
                autopct="%1.1f%%",
                startangle=140,
                wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[1].set_title("Sentiment Distribution (%)", fontsize=13, fontweight="bold")

    plt.suptitle("Social Media Sentiment Overview", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fname = f"{OUTPUT_DIR}/sentiment_distribution.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   📊  Saved sentiment distribution → {fname}")


# ──────────────────────────────────────────────
# 7.  SAVE BEST MODEL
# ──────────────────────────────────────────────

def save_best_model(results, vectorizer):
    best_name = max(results, key=lambda k: results[k]["acc"])
    best_model = results[best_name]["model"]

    print(f"\n🏆  Best model: {best_name}  ({results[best_name]['acc']*100:.2f}%)")

    with open(f"{MODEL_DIR}/sentiment_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open(f"{MODEL_DIR}/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(f"{MODEL_DIR}/model_info.txt", "w") as f:
        f.write(f"Best Model  : {best_name}\n")
        f.write(f"Accuracy    : {results[best_name]['acc']*100:.2f}%\n")
        f.write(f"Label Map   : {LABEL_MAP}\n")

    print(f"✅  Model saved → {MODEL_DIR}/sentiment_model.pkl")
    print(f"✅  Vectorizer saved → {MODEL_DIR}/tfidf_vectorizer.pkl")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data()
    X_tr, X_te, y_tr, y_te, vectorizer, df_clean = prepare_features(df)

    results = train_evaluate(X_tr, X_te, y_tr, y_te)

    print("\n📈  Generating charts...")
    for name, res in results.items():
        plot_confusion_matrix(y_te, res["preds"], name)
    plot_accuracy_comparison(results)
    plot_sentiment_distribution(df_clean)

    save_best_model(results, vectorizer)

    print("\n" + "="*55)
    print("  TRAINING COMPLETE!")
    print("="*55)
