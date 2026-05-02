"""
app/dashboard.py
----------------
Full Streamlit dashboard for the Social Media Sentiment Analysis project.

Run: streamlit run app/dashboard.py
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import clean_text

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="SentiScope | Social Media Analytics",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CUSTOM CSS  (futuristic dark theme)
# ──────────────────────────────────────────────

st.markdown("""
<style>
  /* Import fonts */
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  /* Root overrides */
  html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
  }

  /* Background */
  .stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1421 50%, #0a0e1a 100%);
    color: #e2e8f0;
  }

  /* Sidebar */
  .css-1d391kg, [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1421 0%, #111827 100%);
    border-right: 1px solid rgba(99,102,241,0.2);
  }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 12px;
    padding: 16px;
    backdrop-filter: blur(10px);
  }

  /* Titles */
  h1, h2, h3 {
    color: #e2e8f0;
    font-family: 'Space Grotesk', sans-serif;
  }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s;
  }
  .stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 25px rgba(99,102,241,0.4);
  }

  /* Text input */
  .stTextArea textarea, .stTextInput input {
    background: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
  }

  /* Select box */
  .stSelectbox [data-baseweb="select"] {
    background: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    background: rgba(15,23,42,0.6);
    border-radius: 10px;
    padding: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    color: #94a3b8;
    border-radius: 8px;
    font-weight: 500;
  }
  .stTabs [aria-selected="true"] {
    background: rgba(99,102,241,0.2) !important;
    color: #818cf8 !important;
  }

  /* Divider */
  hr { border-color: rgba(99,102,241,0.2); }

  /* Dataframe */
  .dataframe { background: rgba(15,23,42,0.8); }

  /* Custom cards */
  .stat-card {
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    backdrop-filter: blur(10px);
  }
  .positive-card { border-color: rgba(6,214,160,0.4); }
  .negative-card { border-color: rgba(239,71,111,0.4); }
  .neutral-card  { border-color: rgba(255,209,102,0.4); }

  .sentiment-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 50px;
    font-weight: 600;
    font-size: 1.1rem;
  }
  .badge-positive { background: rgba(6,214,160,0.15); color: #06d6a0; border: 1px solid rgba(6,214,160,0.4); }
  .badge-negative { background: rgba(239,71,111,0.15); color: #ef476f; border: 1px solid rgba(239,71,111,0.4); }
  .badge-neutral  { background: rgba(255,209,102,0.15); color: #ffd166; border: 1px solid rgba(255,209,102,0.4); }

  /* Header banner */
  .header-banner {
    background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(139,92,246,0.1) 100%);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
  }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# LOAD MODEL
# ──────────────────────────────────────────────

@st.cache_resource
def load_model():
    try:
        with open("models/sentiment_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("models/tfidf_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        return None, None


@st.cache_data
def load_dataset():
    try:
        return pd.read_csv("data/social_media_posts.csv")
    except FileNotFoundError:
        return None


LABEL_MAP_INV = {0: "Negative", 1: "Neutral", 2: "Positive"}
SENTIMENT_COLORS = {
    "Positive": "#06d6a0",
    "Negative": "#ef476f",
    "Neutral":  "#ffd166",
}


def predict_sentiment(text, model, vectorizer):
    cleaned = clean_text(text)
    vec     = vectorizer.transform([cleaned])
    label   = model.predict(vec)[0]
    proba   = model.predict_proba(vec)[0]
    return {
        "sentiment":  LABEL_MAP_INV[label],
        "confidence": proba[label] * 100,
        "neg": proba[0] * 100,
        "neu": proba[1] * 100,
        "pos": proba[2] * 100,
    }


# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────

def render_header():
    st.markdown("""
    <div class="header-banner">
        <h1 style="margin:0; font-size:2rem; color:#e2e8f0;">
            📡 SentiScope
        </h1>
        <p style="margin:4px 0 0; color:#94a3b8; font-size:1rem;">
            Social Media Sentiment Intelligence Platform
        </p>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────

def render_sidebar(df):
    with st.sidebar:
        st.markdown("### 🎛️ Dashboard Controls")
        st.divider()

        page = st.radio(
            "Navigate",
            ["🏠 Overview", "🔍 Live Predictor", "📊 Deep Analytics",
             "🏷️ Brand Monitor", "📅 Time Analysis"],
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("### 🔧 Filters")
        selected_brand = None
        selected_platform = None

        if df is not None:
            brands = ["All Brands"] + sorted(df["brand"].unique().tolist())
            selected_brand = st.selectbox("Brand", brands)

            platforms = ["All Platforms"] + sorted(df["platform"].unique().tolist())
            selected_platform = st.selectbox("Platform", platforms)

        st.divider()
        st.markdown("""
        <div style="color:#64748b; font-size:0.8rem; text-align:center;">
            Built using<br>
            Python · Scikit-learn · Streamlit
        </div>
        """, unsafe_allow_html=True)

    return page, selected_brand, selected_platform


# ──────────────────────────────────────────────
# OVERVIEW PAGE
# ──────────────────────────────────────────────

def page_overview(df):
    if df is None:
        st.warning("⚠️ Dataset not found. Run `python src/create_dataset.py` first.")
        return

    # KPI Row
    total   = len(df)
    pos_pct = (df["sentiment"] == "positive").sum() / total * 100
    neg_pct = (df["sentiment"] == "negative").sum() / total * 100
    neu_pct = (df["sentiment"] == "neutral").sum()  / total * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📨 Total Posts",    f"{total:,}")
    col2.metric("🟢 Positive",       f"{pos_pct:.1f}%",  delta=f"+{pos_pct:.1f}%")
    col3.metric("🔴 Negative",       f"{neg_pct:.1f}%",  delta=f"-{neg_pct:.1f}%", delta_color="inverse")
    col4.metric("🟡 Neutral",        f"{neu_pct:.1f}%")

    st.divider()
    c1, c2 = st.columns([1, 1])

    # Donut chart
    with c1:
        st.markdown("#### Sentiment Distribution")
        counts = df["sentiment"].value_counts()
        fig = go.Figure(go.Pie(
            labels=[s.capitalize() for s in counts.index],
            values=counts.values,
            hole=0.65,
            marker_colors=[SENTIMENT_COLORS.get(s.capitalize(), "#888") for s in counts.index],
            textinfo="label+percent",
            hovertemplate="%{label}: %{value} posts<extra></extra>",
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
            height=320,
        )
        fig.add_annotation(text=f"{total:,}<br>Posts",
                           x=0.5, y=0.5, font_size=18, showarrow=False,
                           font_color="#e2e8f0")
        st.plotly_chart(fig, use_container_width=True)

    # Brand sentiment heatmap
    with c2:
        st.markdown("#### Top Brands by Sentiment")
        top_brands = df["brand"].value_counts().head(8).index.tolist()
        brand_df   = df[df["brand"].isin(top_brands)]
        pivot      = (brand_df.groupby(["brand", "sentiment"])
                               .size().unstack(fill_value=0)
                               .apply(lambda r: r / r.sum() * 100, axis=1))
        pivot = pivot.rename(columns=str.capitalize)

        fig2 = px.bar(
            pivot.reset_index().melt(id_vars="brand",
                                     value_vars=["Positive","Negative","Neutral"]),
            x="brand", y="value", color="sentiment",
            color_discrete_map=SENTIMENT_COLORS,
            labels={"value": "Percentage (%)", "brand": ""},
            height=320,
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            legend_title_text="",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=30, b=10, l=10, r=10),
            xaxis=dict(tickangle=-30),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Platform breakdown
    st.markdown("#### Platform Sentiment Breakdown")
    plat_pivot = (df.groupby(["platform", "sentiment"])
                    .size().unstack(fill_value=0)
                    .rename(columns=str.capitalize))

    fig3 = px.bar(
        plat_pivot.reset_index().melt(id_vars="platform"),
        x="platform", y="value", color="sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        labels={"value": "Posts", "platform": ""},
        height=350,
        barmode="group",
    )
    fig3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        legend_title_text="",
        margin=dict(t=10, b=10, l=10, r=10),
        xaxis=dict(tickangle=-30),
    )
    st.plotly_chart(fig3, use_container_width=True)


# ──────────────────────────────────────────────
# LIVE PREDICTOR PAGE
# ──────────────────────────────────────────────

def page_live_predictor(model, vectorizer):
    st.markdown("### 🔍 Live Sentiment Predictor")
    st.markdown("Type any social media post to get instant sentiment analysis.")

    if model is None:
        st.error("❌ Model not loaded. Run `python src/train_model.py` first.")
        return

    col_input, col_output = st.columns([1, 1])

    with col_input:
        text_input = st.text_area(
            "Enter Post / Comment / Review",
            height=180,
            placeholder="e.g. 'Zomato delivered hot food in 15 min! Absolutely amazing experience!'",
        )

        example_posts = [
            "Absolutely love this product! Best purchase of the year!",
            "Horrible service, my order never arrived and no refund given.",
            "Just received my order. Will update once I try the product.",
            "Netflix's new series is mind-blowing. Can't stop watching!",
            "The app crashes every 5 minutes. Fix your bugs!",
        ]
        st.markdown("**Quick examples:**")
        for ex in example_posts:
            if st.button(f"▸ {ex[:55]}...", key=ex):
                text_input = ex

        predict_btn = st.button("🔮 Analyze Sentiment", type="primary")

    with col_output:
        if predict_btn and text_input.strip():
            result = predict_sentiment(text_input, model, vectorizer)
            s = result["sentiment"]

            badge_class = f"badge-{s.lower()}"
            st.markdown(f"""
            <div style="padding: 20px; background: rgba(15,23,42,0.8);
                        border-radius: 14px; border: 1px solid rgba(99,102,241,0.3);">
                <div style="margin-bottom: 16px;">
                    <span class="sentiment-badge {badge_class}">{s.upper()}</span>
                    <span style="color: #94a3b8; margin-left: 12px; font-size:0.9rem;">
                        {result['confidence']:.1f}% confidence
                    </span>
                </div>
            """, unsafe_allow_html=True)

            # Probability gauge bars
            for label, score, color in [
                ("Positive", result["pos"], "#06d6a0"),
                ("Neutral",  result["neu"], "#ffd166"),
                ("Negative", result["neg"], "#ef476f"),
            ]:
                st.markdown(f"""
                <div style="margin: 8px 0;">
                    <div style="display:flex; justify-content:space-between;
                                font-size:0.85rem; color:#94a3b8; margin-bottom:4px;">
                        <span>{label}</span><span>{score:.1f}%</span>
                    </div>
                    <div style="background:rgba(255,255,255,0.05); border-radius:50px; height:8px;">
                        <div style="width:{score}%; background:{color};
                                    border-radius:50px; height:8px;
                                    transition: width 0.5s ease;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

            # Radar / polar chart
            fig = go.Figure(go.Scatterpolar(
                r=[result["pos"], result["neu"], result["neg"]],
                theta=["Positive", "Neutral", "Negative"],
                fill="toself",
                fillcolor="rgba(99,102,241,0.2)",
                line_color="#6366f1",
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                showlegend=False,
                height=250,
                margin=dict(t=20, b=20, l=20, r=20),
            )
            st.plotly_chart(fig, use_container_width=True)

        elif predict_btn:
            st.warning("Please enter some text first.")


# ──────────────────────────────────────────────
# DEEP ANALYTICS PAGE
# ──────────────────────────────────────────────

def page_deep_analytics(df):
    if df is None:
        st.warning("Dataset not found.")
        return

    st.markdown("### 📊 Deep Analytics")

    col1, col2 = st.columns(2)

    with col1:
        # Top locations
        st.markdown("#### Posts by Location")
        loc_sentiment = (df.groupby(["location", "sentiment"])
                           .size().unstack(fill_value=0)
                           .rename(columns=str.capitalize))
        fig = px.bar(loc_sentiment.reset_index().melt(id_vars="location"),
                     x="location", y="value", color="sentiment",
                     color_discrete_map=SENTIMENT_COLORS, height=300)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#e2e8f0",
                          margin=dict(t=10,b=30,l=10,r=10),
                          xaxis_tickangle=-30,
                          legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Engagement scatter
        st.markdown("#### Likes vs Sentiment")
        sample_df = df.sample(min(300, len(df)), random_state=42).copy()
        sample_df["sentiment_cap"] = sample_df["sentiment"].str.capitalize()
        fig2 = px.scatter(sample_df, x="likes", y="retweets",
                          color="sentiment_cap",
                          color_discrete_map=SENTIMENT_COLORS,
                          opacity=0.7, height=300,
                          labels={"sentiment_cap": "Sentiment"})
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#e2e8f0",
                           legend_title_text="",
                           margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig2, use_container_width=True)

    # Post length analysis
    st.markdown("#### Post Length Distribution by Sentiment")
    df["text_length"] = df["text"].str.len()
    fig3 = px.box(df, x="sentiment", y="text_length",
                  color="sentiment",
                  color_discrete_map={k.lower(): v for k, v in SENTIMENT_COLORS.items()},
                  height=300,
                  labels={"text_length": "Characters", "sentiment": ""})
    fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)",
                       font_color="#e2e8f0",
                       showlegend=False,
                       margin=dict(t=10,b=10,l=10,r=10))
    st.plotly_chart(fig3, use_container_width=True)

    # Raw data table
    st.markdown("#### Sample Data Preview")
    show_cols = ["text", "sentiment", "brand", "platform", "location", "likes"]
    st.dataframe(df[show_cols].head(20), use_container_width=True)


# ──────────────────────────────────────────────
# BRAND MONITOR PAGE
# ──────────────────────────────────────────────

def page_brand_monitor(df):
    if df is None:
        st.warning("Dataset not found.")
        return

    st.markdown("### 🏷️ Brand Health Monitor")

    brand = st.selectbox("Select Brand to Monitor",
                         sorted(df["brand"].unique().tolist()))
    brand_df = df[df["brand"] == brand]

    counts = brand_df["sentiment"].value_counts()
    total  = len(brand_df)
    pos    = counts.get("positive", 0)
    neg    = counts.get("negative", 0)
    neu    = counts.get("neutral", 0)

    # Brand score (simple NPS-style)
    brand_score = int(((pos - neg) / total) * 100)
    score_color = "#06d6a0" if brand_score > 20 else "#ef476f" if brand_score < -10 else "#ffd166"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📨 Total Posts",    f"{total}")
    col2.metric("🟢 Positive",       f"{pos/total*100:.1f}%")
    col3.metric("🔴 Negative",       f"{neg/total*100:.1f}%")
    col4.metric("🏆 Brand Score",    f"{brand_score:+d}",
                delta="Good" if brand_score > 0 else "At Risk")

    col_a, col_b = st.columns(2)
    with col_a:
        # Sentiment donut for this brand
        fig = go.Figure(go.Pie(
            labels=["Positive","Negative","Neutral"],
            values=[pos, neg, neu],
            hole=0.6,
            marker_colors=["#06d6a0","#ef476f","#ffd166"],
        ))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#e2e8f0", height=300,
                          title=f"{brand} Sentiment Mix",
                          margin=dict(t=40,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        # Platform breakdown for this brand
        plat = (brand_df.groupby(["platform","sentiment"])
                         .size().unstack(fill_value=0)
                         .rename(columns=str.capitalize))
        fig2 = px.bar(plat.reset_index().melt(id_vars="platform"),
                      x="platform", y="value", color="sentiment",
                      color_discrete_map=SENTIMENT_COLORS, height=300,
                      title=f"{brand} — Platform Breakdown")
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#e2e8f0",
                           legend_title_text="",
                           margin=dict(t=40,b=30,l=10,r=10),
                           xaxis_tickangle=-30)
        st.plotly_chart(fig2, use_container_width=True)

    # Sample negative posts
    neg_posts = brand_df[brand_df["sentiment"] == "negative"]["text"].head(5).tolist()
    if neg_posts:
        st.markdown("#### ⚠️ Recent Negative Feedback")
        for p in neg_posts:
            st.markdown(f"""
            <div style="background:rgba(239,71,111,0.08); border-left:3px solid #ef476f;
                        padding:10px 16px; border-radius:0 8px 8px 0; margin:6px 0;
                        color:#e2e8f0; font-size:0.9rem;">
                {p}
            </div>
            """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# TIME ANALYSIS PAGE
# ──────────────────────────────────────────────

def page_time_analysis(df):
    if df is None:
        st.warning("Dataset not found.")
        return

    st.markdown("### 📅 Time Series Analysis")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"]      = df["timestamp"].dt.date
    df["hour"]      = df["timestamp"].dt.hour

    # Daily trend
    daily = (df.groupby(["date","sentiment"])
               .size().reset_index(name="count")
               .rename(columns={"sentiment":"Sentiment"}))
    daily["Sentiment"] = daily["Sentiment"].str.capitalize()

    fig = px.line(daily, x="date", y="count", color="Sentiment",
                  color_discrete_map=SENTIMENT_COLORS,
                  labels={"count":"Posts","date":"Date"},
                  title="Daily Post Volume by Sentiment", height=350)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#e2e8f0",
                      legend_title_text="",
                      margin=dict(t=40,b=10,l=10,r=10))
    st.plotly_chart(fig, use_container_width=True)

    # Hourly heatmap
    hourly = df.groupby(["hour","sentiment"]).size().unstack(fill_value=0)
    hourly.columns = [c.capitalize() for c in hourly.columns]

    fig2 = px.bar(hourly.reset_index().melt(id_vars="hour"),
                  x="hour", y="value", color="variable",
                  color_discrete_map=SENTIMENT_COLORS,
                  labels={"hour":"Hour of Day","value":"Posts","variable":""},
                  title="Posting Activity by Hour", height=320)
    fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)",
                       font_color="#e2e8f0",
                       margin=dict(t=40,b=10,l=10,r=10))
    st.plotly_chart(fig2, use_container_width=True)


# ──────────────────────────────────────────────
# MAIN APP
# ──────────────────────────────────────────────

def main():
    render_header()
    model, vectorizer = load_model()
    df = load_dataset()

    page, selected_brand, selected_platform = render_sidebar(df)

    # Apply filters
    filtered_df = df.copy() if df is not None else None
    if filtered_df is not None and selected_brand and selected_brand != "All Brands":
        filtered_df = filtered_df[filtered_df["brand"] == selected_brand]
    if filtered_df is not None and selected_platform and selected_platform != "All Platforms":
        filtered_df = filtered_df[filtered_df["platform"] == selected_platform]

    if   "Overview"       in page: page_overview(filtered_df)
    elif "Predictor"      in page: page_live_predictor(model, vectorizer)
    elif "Deep Analytics" in page: page_deep_analytics(filtered_df)
    elif "Brand Monitor"  in page: page_brand_monitor(df)   # always use full df for brand monitor
    elif "Time"           in page: page_time_analysis(filtered_df)


if __name__ == "__main__":
    main()