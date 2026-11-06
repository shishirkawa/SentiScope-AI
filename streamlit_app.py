"""
SentiScope AI - Sentiment Analysis Platform
--------------------------------------------
A clean, functional sentiment analysis application.
Built by Shishir Kawa
"""

import os
import sys
import io
import json
import joblib
import pandas as pd
import streamlit as st
from datetime import datetime
import re

# -----------------------------------------------------------------------------
# Path & Environment Configuration
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from preprocess import preprocess, clean_text
except ImportError:
    pass

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SentiScope AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Clean, Purposeful Styling - No Slop
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Clean foundation */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

    * {
        font-family: 'Inter', -apple-system, system-ui, sans-serif;
    }

    /* Neutral, professional background */
    .stApp {
        background: #fafbfc;
    }

    /* Remove default streamlit padding excess */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Clean header - no gradients, no glow */
    .main-header {
        background: white;
        border: 1px solid #e1e4e8;
        border-radius: 6px;
        padding: 24px;
        margin-bottom: 24px;
    }

    .main-title {
        font-size: 24px;
        font-weight: 600;
        color: #24292e;
        margin: 0 0 8px 0;
        letter-spacing: -0.01em;
    }

    .main-subtitle {
        font-size: 14px;
        color: #586069;
        margin: 0;
        line-height: 1.5;
    }

    /* Simple text input */
    .stTextArea textarea {
        background: white !important;
        border: 1px solid #d1d5da !important;
        border-radius: 6px !important;
        color: #24292e !important;
        font-size: 14px !important;
        padding: 12px !important;
        font-family: inherit !important;
    }

    .stTextArea textarea:focus {
        border-color: #0366d6 !important;
        outline: none !important;
        box-shadow: 0 0 0 3px rgba(3, 102, 214, 0.1) !important;
    }

    /* Clean buttons - no gradients */
    div[data-testid="stButton"] button {
        border-radius: 6px !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        padding: 8px 16px !important;
        transition: all 0.1s ease !important;
        border: 1px solid transparent !important;
    }

    div[data-testid="stButton"] button[kind="primary"] {
        background: #0366d6 !important;
        color: white !important;
        border-color: #0366d6 !important;
    }

    div[data-testid="stButton"] button[kind="primary"]:hover {
        background: #0256c7 !important;
        border-color: #0256c7 !important;
    }

    div[data-testid="stButton"] button:not([kind="primary"]) {
        background: white !important;
        color: #24292e !important;
        border: 1px solid #d1d5da !important;
    }

    div[data-testid="stButton"] button:not([kind="primary"]):hover {
        background: #f6f8fa !important;
        border-color: #c2c7cc !important;
    }

    /* Result card - clean semantic colors */
    .result-card {
        background: white;
        border: 1px solid #e1e4e8;
        border-radius: 6px;
        padding: 16px;
        margin: 16px 0;
    }

    .result-card.positive {
        border-left: 3px solid #28a745;
    }

    .result-card.neutral {
        border-left: 3px solid #6f42c1;
    }

    .result-card.negative {
        border-left: 3px solid #d73a49;
    }

    .result-label {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #586069;
        margin-bottom: 4px;
    }

    .result-value {
        font-size: 20px;
        font-weight: 600;
        color: #24292e;
        margin-bottom: 8px;
    }

    .result-confidence {
        font-size: 14px;
        color: #586069;
    }

    /* Progress bars - simple, no fancy styling */
    .stProgress > div > div > div {
        background: #0366d6;
        height: 8px;
    }

    /* Metrics - clean cards */
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e1e4e8;
        border-radius: 6px;
        padding: 16px;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 12px;
        font-weight: 600;
        color: #586069;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    div[data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: 600;
        color: #24292e;
    }

    /* Tabs - minimal styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: transparent;
        border-bottom: 1px solid #e1e4e8;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: none;
        color: #586069;
        font-weight: 500;
        font-size: 14px;
        padding: 12px 16px;
    }

    .stTabs [aria-selected="true"] {
        color: #24292e;
        border-bottom: 2px solid #0366d6;
    }

    /* Sidebar - clean and minimal */
    section[data-testid="stSidebar"] {
        background: white;
        border-right: 1px solid #e1e4e8;
    }

    section[data-testid="stSidebar"] h3 {
        font-size: 14px;
        font-weight: 600;
        color: #24292e;
        margin-bottom: 16px;
    }

    /* Data tables */
    div[data-testid="stDataFrame"] {
        border: 1px solid #e1e4e8;
        border-radius: 6px;
    }

    /* Info boxes - subtle */
    div[data-testid="stAlert"] {
        border-radius: 6px;
        border: 1px solid #e1e4e8;
        background: #f6f8fa;
    }

    /* Remove excessive spacing */
    .element-container {
        margin-bottom: 1rem;
    }

    /* Clean expander */
    div[data-testid="stExpander"] {
        background: white;
        border: 1px solid #e1e4e8;
        border-radius: 6px;
    }

    /* Footer */
    .footer {
        margin-top: 48px;
        padding-top: 24px;
        border-top: 1px solid #e1e4e8;
        text-align: center;
        color: #586069;
        font-size: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Text Processing Functions
# -----------------------------------------------------------------------------
STOPWORDS = {
    "a", "an", "the", "is", "it", "this", "that", "i", "you", "he", "she",
    "we", "they", "am", "are", "was", "were", "be", "been", "being", "to",
    "of", "in", "on", "at", "for", "with", "and", "or", "so", "as",
}

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def preprocess(text: str) -> str:
    return " ".join(w for w in clean_text(text).split() if w not in STOPWORDS)

def extract_keywords(text: str, top_n: int = 5) -> list:
    words = clean_text(text).split()
    words = [w for w in words if w not in STOPWORDS and len(w) > 3]
    from collections import Counter
    word_freq = Counter(words)
    return [word for word, _ in word_freq.most_common(top_n)]

def get_text_stats(text: str) -> dict:
    words = text.split()
    return {
        "word_count": len(words),
        "char_count": len(text),
        "sentence_count": len([s for s in text.split('.') if s.strip()])
    }

# -----------------------------------------------------------------------------
# Data Generation
# -----------------------------------------------------------------------------
def generate_default_data():
    subjects = [
        "the product", "this movie", "the restaurant", "the phone", "the laptop",
        "this book", "the service", "the app", "the hotel", "the course"
    ]
    pos = [
        "I love {s}, it exceeded expectations.",
        "{s} works perfectly and quality is outstanding.",
        "Really impressed with {s}, highly recommend.",
        "{s} is fantastic, worth every penny.",
        "Great experience with {s}, very satisfied.",
    ]
    neg = [
        "I hate {s}, complete waste of money.",
        "{s} broke down within days, disappointing.",
        "Terrible experience with {s}, not recommended.",
        "{s} is awful, nothing like advertised.",
        "Very frustrated with {s}, doesn't work.",
    ]
    neu = [
        "{s} was okay, nothing special.",
        "{s} is average, does the job.",
        "Mixed feelings about {s}.",
        "{s} is fine for the price.",
        "Not sure how I feel about {s}.",
    ]
    rows = []
    for s in subjects:
        for t in pos: rows.append((t.format(s=s), "positive"))
        for t in neg: rows.append((t.format(s=s), "negative"))
        for t in neu: rows.append((t.format(s=s), "neutral"))
    import random
    random.seed(42)
    random.shuffle(rows)
    return pd.DataFrame(rows, columns=["text", "sentiment"])

# -----------------------------------------------------------------------------
# Model Loading
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_assets():
    model_path = os.path.join(MODELS_DIR, "sentiment_model.joblib")
    vectorizer_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")

    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    df = generate_default_data()
    clean_series = df["text"].apply(preprocess)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    X_vec = vectorizer.fit_transform(clean_series)
    model = LogisticRegression(max_iter=1000, C=5.0)
    model.fit(X_vec, df["sentiment"])
    return model, vectorizer

model, vectorizer = load_assets()

def predict_sentiment(text: str):
    clean = preprocess(text)
    if not clean.strip():
        clean = text.lower().strip()

    vec = vectorizer.transform([clean])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]

    prob_dict = {
        cls_name: float(prob)
        for cls_name, prob in zip(model.classes_, proba)
    }
    return pred, prob_dict, clean

# -----------------------------------------------------------------------------
# Session State
# -----------------------------------------------------------------------------
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

if "current_input" not in st.session_state:
    st.session_state.current_input = ""

def save_to_history(text, sentiment, confidence):
    st.session_state.analysis_history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "text": text[:100],
        "sentiment": sentiment,
        "confidence": confidence
    })

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### SentiScope AI")
    st.caption("Sentiment analysis platform")

    st.markdown("---")

    st.markdown("### Quick Samples")

    samples = {
        "Positive": "This product is amazing. Best purchase I've made this year.",
        "Negative": "Terrible quality, broke after one day. Very disappointed.",
        "Neutral": "The item arrived on time and looks as described.",
        "Mixed": "Good quality but quite expensive for what you get.",
    }

    for label, text in samples.items():
        if st.button(label, use_container_width=True, key=f"sample_{label}"):
            st.session_state.current_input = text

    st.markdown("---")

    st.markdown("### Statistics")
    if st.session_state.analysis_history:
        total = len(st.session_state.analysis_history)
        st.metric("Total analyses", total)

        sentiments = [item["sentiment"] for item in st.session_state.analysis_history]
        pos_count = sentiments.count("positive")
        st.metric("Positive", f"{pos_count} ({pos_count/total*100:.0f}%)")
    else:
        st.info("No analyses yet")

    st.markdown("---")

    st.markdown("### Model Info")
    st.caption("Logistic Regression + TF-IDF")
    st.caption("Accuracy: ~95%")
    st.caption("Classes: Positive, Neutral, Negative")

# -----------------------------------------------------------------------------
# Main Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <div class="main-title">SentiScope AI</div>
        <div class="main-subtitle">
            Analyze sentiment in text using machine learning. Get instant predictions with confidence scores.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Analyze",
    "Batch Process",
    "History",
    "Tools"
])

# -----------------------------------------------------------------------------
# TAB 1: Analyze
# -----------------------------------------------------------------------------
with tab1:
    col1, col2 = st.columns([1.3, 1], gap="large")

    with col1:
        st.markdown("### Input text")

        text_input = st.text_area(
            "Text to analyze",
            value=st.session_state.current_input,
            height=200,
            placeholder="Enter text to analyze sentiment...",
            key="main_input",
            label_visibility="collapsed"
        )

        col_btn1, col_btn2 = st.columns([3, 1])
        with col_btn1:
            analyze_btn = st.button("Analyze", type="primary", use_container_width=True)
        with col_btn2:
            if st.button("Clear", use_container_width=True):
                st.session_state.current_input = ""
                st.rerun()

    with col2:
        st.markdown("### Result")

        if text_input.strip():
            sentiment, probs, cleaned = predict_sentiment(text_input)
            confidence = probs[sentiment] * 100

            save_to_history(text_input, sentiment, confidence)

            sentiment_class = f"sentiment-{sentiment}"
            st.markdown(
                f"""
                <div class="result-card {sentiment}">
                    <div class="result-label">Sentiment</div>
                    <div class="result-value">{sentiment.capitalize()}</div>
                    <div class="result-confidence">{confidence:.1f}% confidence</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("#### Confidence breakdown")
            for sent in ["positive", "neutral", "negative"]:
                prob_val = probs[sent]
                st.write(f"**{sent.capitalize()}:** {prob_val*100:.1f}%")
                st.progress(prob_val)

            with st.expander("Text details"):
                stats = get_text_stats(text_input)
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Words", stats["word_count"])
                col_b.metric("Characters", stats["char_count"])
                col_c.metric("Sentences", stats["sentence_count"])

                keywords = extract_keywords(text_input)
                if keywords:
                    st.write("**Keywords:**", ", ".join(keywords))
        else:
            st.info("Enter text to analyze")

# -----------------------------------------------------------------------------
# TAB 2: Batch Process
# -----------------------------------------------------------------------------
with tab2:
    st.markdown("### Batch analysis")
    st.write("Upload CSV file to analyze multiple texts")

    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type=["csv"],
        help="Upload CSV with text column"
    )

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(df):,} rows")

            text_cols = [col for col in df.columns if df[col].dtype == "object"]
            if not text_cols:
                text_cols = list(df.columns)

            col1, col2 = st.columns([2, 1])
            with col1:
                selected_col = st.selectbox("Select text column", text_cols)
            with col2:
                st.write("")
                st.write("")
                process_btn = st.button("Process", type="primary", use_container_width=True)

            if process_btn:
                with st.spinner("Processing..."):
                    results = []
                    progress = st.progress(0)

                    for idx, text in enumerate(df[selected_col]):
                        sent, probs, _ = predict_sentiment(str(text))
                        results.append({
                            "sentiment": sent,
                            "confidence": round(probs[sent], 3),
                            "positive": round(probs.get("positive", 0), 3),
                            "neutral": round(probs.get("neutral", 0), 3),
                            "negative": round(probs.get("negative", 0), 3)
                        })
                        progress.progress((idx + 1) / len(df))

                    result_df = df.copy()
                    for key in results[0].keys():
                        result_df[key] = [r[key] for r in results]

                    st.session_state.batch_results = result_df

            if "batch_results" in st.session_state:
                res = st.session_state.batch_results

                st.markdown("### Results")

                col1, col2, col3, col4 = st.columns(4)
                counts = res["sentiment"].value_counts().to_dict()
                total = len(res)

                col1.metric("Total", f"{total:,}")
                col2.metric("Positive", f"{counts.get('positive', 0):,}")
                col3.metric("Neutral", f"{counts.get('neutral', 0):,}")
                col4.metric("Negative", f"{counts.get('negative', 0):,}")

                st.markdown("### Distribution")
                st.bar_chart(res["sentiment"].value_counts())

                st.markdown("### Data preview")
                st.dataframe(res.head(100), use_container_width=True)

                col1, col2, col3 = st.columns(3)
                with col1:
                    csv = res.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        csv,
                        "results.csv",
                        "text/csv",
                        use_container_width=True
                    )
                with col2:
                    json_str = res.to_json(orient="records", indent=2)
                    st.download_button(
                        "Download JSON",
                        json_str,
                        "results.json",
                        "application/json",
                        use_container_width=True
                    )
                with col3:
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                        res.to_excel(writer, index=False, sheet_name='Results')
                    st.download_button(
                        "Download Excel",
                        excel_buffer.getvalue(),
                        "results.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )

        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.info("Upload CSV file to begin")

# -----------------------------------------------------------------------------
# TAB 3: History
# -----------------------------------------------------------------------------
with tab3:
    st.markdown("### Analysis history")

    if st.session_state.analysis_history:
        history_df = pd.DataFrame(st.session_state.analysis_history)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Distribution")
            sentiment_counts = history_df["sentiment"].value_counts()
            st.bar_chart(sentiment_counts)

        with col2:
            st.markdown("#### Average confidence")
            avg_conf = history_df.groupby("sentiment")["confidence"].mean()
            st.bar_chart(avg_conf)

        st.markdown("#### Recent analyses")
        st.dataframe(
            history_df[["timestamp", "text", "sentiment", "confidence"]].tail(20),
            use_container_width=True
        )

        if st.button("Clear history"):
            st.session_state.analysis_history = []
            st.rerun()
    else:
        st.info("No history yet. Analyze text to see trends.")

# -----------------------------------------------------------------------------
# TAB 4: Tools
# -----------------------------------------------------------------------------
with tab4:
    st.markdown("### Compare texts")

    text1 = st.text_area("Text 1", height=100, placeholder="First text...")
    text2 = st.text_area("Text 2", height=100, placeholder="Second text...")

    if st.button("Compare", type="primary"):
        if text1.strip() and text2.strip():
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Text 1**")
                sent1, probs1, _ = predict_sentiment(text1)
                st.write(f"Sentiment: {sent1.capitalize()}")
                st.write(f"Confidence: {probs1[sent1]*100:.1f}%")

            with col2:
                st.markdown("**Text 2**")
                sent2, probs2, _ = predict_sentiment(text2)
                st.write(f"Sentiment: {sent2.capitalize()}")
                st.write(f"Confidence: {probs2[sent2]*100:.1f}%")

            if sent1 == sent2:
                st.success("Same sentiment")
            else:
                st.warning(f"Different: {sent1} vs {sent2}")
        else:
            st.error("Enter both texts")

    st.markdown("---")

    st.markdown("### Text preprocessing")

    test_text = st.text_area("Test text", "This is sample text with STOPWORDS!")

    col1, col2 = st.columns(2)
    with col1:
        if st.checkbox("Remove stopwords"):
            processed = preprocess(test_text)
            st.code(processed)

    with col2:
        if st.checkbox("Clean text"):
            cleaned = clean_text(test_text)
            st.code(cleaned)

    if st.checkbox("Extract keywords"):
        keywords = extract_keywords(test_text)
        st.write("**Keywords:**", ", ".join(keywords) if keywords else "None")

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="footer">
        SentiScope AI – Built by Shishir Kawa
    </div>
    """,
    unsafe_allow_html=True
)
