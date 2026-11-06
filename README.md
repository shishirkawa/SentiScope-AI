# ReviewLens

An offline sentiment-analysis application built with Streamlit. It classifies
text as **positive**, **neutral**, or **negative** using TF-IDF features and
Logistic Regression.

## Project structure

```text
streamlit_app.py          # Streamlit UI and inference pipeline
src/
  compare_models.py       # Compare Logistic Regression and Naive Bayes
requirements.txt
README.md
```

The application generates a small deterministic training dataset automatically
when model files are not present, so it can run without external downloads.

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

The app provides:

- Live sentiment prediction with confidence scores
- Quick example reviews and negation handling
- Batch CSV classification with downloadable results
- Model performance and dataset inspection

## Compare models

```bash
python src/compare_models.py
```

If `data/reviews.csv` exists, it is used as the input dataset. Otherwise,
`compare_models.py` uses deterministic example data and writes the comparison
to `outputs/model_comparison.csv`.

For a custom dataset, provide a CSV with these columns:

```text
text,sentiment
The service was excellent,positive
The delivery was late,negative
```

## Deployment

The app can be deployed to Streamlit Community Cloud by selecting
`streamlit_app.py` as the main file. Streamlit Cloud installs the packages
listed in `requirements.txt` automatically.
