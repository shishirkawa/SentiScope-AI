
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "reviews.csv"


def generate_default_data():
    """Create a small deterministic dataset when the optional CSV is absent."""
    subjects = ["product", "movie", "restaurant", "phone", "service", "app"]
    templates = {
        "positive": [
            "I love the {subject}; it is excellent.",
            "The {subject} is fantastic and works perfectly.",
            "I am very happy with this {subject}.",
        ],
        "negative": [
            "I hate the {subject}; it is terrible.",
            "The {subject} is awful and does not work.",
            "I am very disappointed with this {subject}.",
        ],
        "neutral": [
            "The {subject} is okay and works as expected.",
            "This {subject} is average, nothing special.",
            "The {subject} is acceptable for the price.",
        ],
    }
    rows = [
        (template.format(subject=subject), sentiment)
        for subject in subjects
        for sentiment, sentiment_templates in templates.items()
        for template in sentiment_templates
    ]
    return pd.DataFrame(rows, columns=["text", "sentiment"])


if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)
else:
    print(f"Warning: {DATA_PATH} was not found; using generated example data.")
    df = generate_default_data()

df = df.dropna(subset=["text", "sentiment"])
df["text"] = df["text"].astype(str)
df["sentiment"] = df["sentiment"].astype(str).str.lower()

X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["sentiment"],
    test_size=0.2,
    random_state=42,
    stratify=df["sentiment"],
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Multinomial Naive Bayes": MultinomialNB(),
}

results = []

for name, classifier in models.items():
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=1),
        classifier,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Macro Precision": precision,
        "Macro Recall": recall,
        "Macro F1": f1,
    })

    print(f"\n--- {name} ---")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))

results_df = pd.DataFrame(results).sort_values(
    "Macro F1", ascending=False
)

print("\nMODEL COMPARISON")
print(results_df.round(4).to_string(index=False))

output_path = ROOT / "outputs" / "model_comparison.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(output_path, index=False)

print(f"\nResults saved to: {output_path}")
