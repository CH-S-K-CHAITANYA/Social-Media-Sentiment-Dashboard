# 📡 Social Media Sentiment Dashboard

> **ML-powered sentiment intelligence system to analyze social media posts, classify public opinion, and visualize brand sentiment trends through an interactive Streamlit dashboard.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Plotly](https://img.shields.io/badge/Plotly-6.7-3F4F75?logo=plotly&logoColor=white)](https://plotly.com)

---

## 📖 Overview

Social Media Sentiment Dashboard is a **supervised machine learning** project that classifies social media posts as **Positive**, **Neutral**, or **Negative**. It uses a synthetic social media dataset, text preprocessing, TF-IDF vectorization, multiple ML classifiers, and an interactive dashboard for real-time sentiment prediction and analytics.

The system simulates posts from brands such as **Zomato, Swiggy, Netflix, Flipkart, Amazon, HDFC Bank, Jio, Airtel, Ola, Uber, Nykaa**, and more.

### Why It Matters

| Use Case | Description |
|---|---|
| 🚨 **Brand Risk Detection** | Identify negative customer feedback before it escalates |
| 📊 **Social Listening** | Track public opinion across platforms and brands |
| 🎯 **Customer Experience Insights** | Understand what users feel about products and services |
| 📈 **Trend Monitoring** | Analyze sentiment changes across time, location, and platform |
| 🧠 **Live Prediction** | Predict sentiment instantly for any custom social media text |

### Workflow

```text
Synthetic Posts → Text Cleaning → TF-IDF Vectorization → Model Training → Evaluation → Streamlit Dashboard
```

---

## 🏗️ Architecture

```text
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Synthetic   │───▶│  Preprocess  │───▶│  TF-IDF      │───▶│  Train ML    │
│  Posts       │    │  Text Data   │    │  Features    │    │  Models      │
└──────────────┘    └──────────────┘    └──────────────┘    └──────┬───────┘
                                                                   │
                                                                   ▼
                    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
                    │  Streamlit   │◀───│  Prediction  │◀───│  Best Model  │
                    │  Dashboard   │    │  Engine      │    │  + Vectorizer│
                    └──────────────┘    └──────────────┘    └──────────────┘
```

**Models Trained:**

- Logistic Regression
- Multinomial Naive Bayes
- Random Forest Classifier

**Best Model Saved:**

- `models/sentiment_model.pkl`
- `models/tfidf_vectorizer.pkl`
- `models/model_info.txt`

---

## 📁 Project Structure

```text
Social-Media-Sentiment-Dashboard1/
├── app/
│   ├── __init__.py
│   └── dashboard.py                  # Main Streamlit dashboard
├── data/
│   └── social_media_posts.csv        # Generated synthetic social media dataset
├── docs/                             # Documentation assets
├── images/                           # README screenshots or UI images
├── models/
│   ├── sentiment_model.pkl           # Trained sentiment classifier
│   ├── tfidf_vectorizer.pkl          # Saved TF-IDF vectorizer
│   └── model_info.txt                # Best model metadata
├── notebooks/                        # Jupyter notebooks
├── outputs/
│   └── charts/
│       ├── confusion_matrix_logistic_regression.png
│       ├── confusion_matrix_naive_bayes.png
│       ├── confusion_matrix_random_forest.png
│       ├── model_accuracy_comparison.png
│       └── sentiment_distribution.png
├── src/
│   ├── __init__.py
│   ├── create_dataset.py             # Synthetic dataset generator
│   ├── preprocess.py                 # Text cleaning and preprocessing
│   ├── train_model.py                # Model training and evaluation pipeline
│   ├── predict.py                    # Prediction/model utility script
│   └── dashboard.py                  # Compatibility launcher
├── main.py                           # Root Streamlit entry point
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- pip package manager
- A terminal or IDE such as VS Code

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/Social-Media-Sentiment-Dashboard1.git
cd Social-Media-Sentiment-Dashboard1

# 2. Create virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Generate Dataset

```bash
python src/create_dataset.py
```

This creates:

1. ✅ 1500 synthetic social media posts
2. ✅ Balanced sentiment classes
3. ✅ Brand, platform, location, engagement, and timestamp fields
4. ✅ CSV file at `data/social_media_posts.csv`

### Train Models

```bash
python src/train_model.py
```

This will:

1. ✅ Load the generated dataset
2. ✅ Clean and preprocess text
3. ✅ Convert text into TF-IDF features
4. ✅ Train 3 machine learning models
5. ✅ Evaluate model accuracy and classification metrics
6. ✅ Save the best model and vectorizer
7. ✅ Generate evaluation charts in `outputs/charts/`

### Launch Dashboard

```bash
streamlit run main.py
```

Alternative commands:

```bash
streamlit run app/dashboard.py
streamlit run src/dashboard.py
```

Open `http://localhost:8501` in your browser.

---

## 📊 Features

### Dataset Features

| Category | Features |
|---|---|
| **Post Content** | Text, cleaned text, sentiment label |
| **Brand Data** | Brand name such as Zomato, Netflix, Amazon, Flipkart |
| **Platform Data** | Twitter, Facebook, Instagram, Reddit, App Store, Play Store |
| **Location Data** | Mumbai, Delhi, Bangalore, Hyderabad, Chennai, Pune, and more |
| **Engagement Metrics** | Likes and retweets |
| **Time Data** | Timestamp, date, and hour-based analysis |
| **Target** | Sentiment: Positive, Neutral, Negative |

### Text Preprocessing

- Lowercase conversion
- URL removal
- Mention removal
- Hashtag cleanup
- Emoji and non-ASCII cleanup
- Punctuation removal
- Number removal
- Stopword removal
- Lemmatization when NLTK data is available
- Built-in fallback preprocessing when NLTK resources are unavailable

### Dashboard Pages

| Page | Description |
|---|---|
| 🏠 **Overview** | KPI cards, sentiment distribution, brand sentiment, platform breakdown |
| 🔍 **Live Predictor** | Enter any post and get instant sentiment prediction with confidence scores |
| 📊 **Deep Analytics** | Location analysis, engagement scatter plots, text length analysis, sample data |
| 🏷️ **Brand Monitor** | Brand-specific sentiment health, platform mix, and negative feedback examples |
| 📅 **Time Analysis** | Daily sentiment trend and hourly posting activity |

---

## 📈 Sample Outputs

### Sentiment Distribution

![Sentiment Distribution](outputs/charts/sentiment_distribution.png)

### Model Accuracy Comparison

![Model Accuracy Comparison](outputs/charts/model_accuracy_comparison.png)

### Logistic Regression Confusion Matrix

![Logistic Regression Confusion Matrix](outputs/charts/confusion_matrix_logistic_regression.png)

### Naive Bayes Confusion Matrix

![Naive Bayes Confusion Matrix](outputs/charts/confusion_matrix_naive_bayes.png)

### Random Forest Confusion Matrix

![Random Forest Confusion Matrix](outputs/charts/confusion_matrix_random_forest.png)

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-Learn |
| NLP Features | TF-IDF Vectorizer, NLTK-compatible preprocessing |
| Visualization | Matplotlib, Seaborn, Plotly |
| Dashboard | Streamlit |
| Model Persistence | Pickle |

---

## 🎯 Virtual Simulation

This project uses **synthetic social media data** to simulate real-world brand monitoring.

- **Balanced classes**: 500 positive, 500 neutral, and 500 negative posts
- **Realistic brands**: Indian and global consumer-facing brands
- **Multiple platforms**: Social networks, review platforms, and app stores
- **Engagement signals**: Likes and retweets simulate post popularity
- **Time variation**: Posts are generated across recent dates and times
- **Noise injection**: Hashtags, informal language, punctuation, and social-media style text are added

This approach is useful because:

1. Real social media data often requires API access and privacy handling
2. Synthetic data keeps the project reproducible
3. The ML pipeline is similar to real sentiment analysis workflows
4. It allows quick experimentation with preprocessing, modeling, and dashboards

---

## 🧪 Model Performance

The training pipeline compares three classifiers using the generated dataset:

| Model | Purpose |
|---|---|
| Logistic Regression | Strong baseline for TF-IDF text classification |
| Multinomial Naive Bayes | Fast probabilistic classifier for text data |
| Random Forest | Tree-based comparison model |

The best-performing model is saved automatically in the `models/` folder and used by the Streamlit dashboard for live predictions.

---

## 🔮 Example Prediction Flow

```text
"The app crashes every time I open it. Fix your bugs!"
        ↓
Text Cleaning
        ↓
TF-IDF Vectorization
        ↓
Saved ML Model
        ↓
Negative Sentiment + Confidence Score
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -m "Add new feature"`)
4. Push to branch (`git push origin feature/new-feature`)
5. Open a Pull Request

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👤 Author

**CH S K CHAITANYA**

- GitHub: [@CH-S-K-CHAITANYA](https://github.com/CH-S-K-CHAITANYA)
- LinkedIn: [CH S K CHAITANYA](https://linkedin.com/in/chskchaitanya)

---

> ⭐ Star this repository if you found it helpful!
