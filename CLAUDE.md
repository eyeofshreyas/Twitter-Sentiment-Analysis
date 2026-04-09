# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Twitter Sentiment Analysis — a Jupyter notebook that classifies tweets as positive (1) or negative (0) using Logistic Regression with TF-IDF features.

## Running the Notebook

Open and run [Twitter.ipynb](Twitter.ipynb) top-to-bottom in Jupyter. The notebook must be run from this directory so the relative path `twitter_data.csv` resolves correctly.

```bash
jupyter notebook Twitter.ipynb
# or
jupyter lab Twitter.ipynb
```

## Dependencies

```bash
pip install numpy pandas scikit-learn nltk scipy requests
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"
```

## Pipeline Architecture

The notebook implements a sequential ML pipeline in a single file with no modular separation:

1. **Data loading** — `twitter_data.csv` is read with `csv.QUOTE_NONE` and `encoding='ISO-8859-1'` because the raw CSV contains unescaped quotes. Columns: `target, id, date, flag, user, text`.

2. **Label normalization** — target values are 0 (negative) and 4 (positive); the notebook replaces 4 → 1.

3. **Text preprocessing (`stemming` function)** — strips non-alpha characters via regex, lowercases, removes NLTK English stopwords, applies Porter stemmer. NaN/float values return `''`.

4. **Feature extraction** — `TfidfVectorizer` fitted on `X_train`, then applied to both splits. Result is a sparse CSR matrix (shape ~66k × 3k).

5. **Model** — `LogisticRegression(max_iter=1000)` trained on the TF-IDF matrix.

## Known Data Issues

- The CSV parsing produces malformed rows due to nested quotes in the raw data — most rows end up with NaN in `user` and `text` columns (65k and 79k missing respectively out of 83k rows). The `stemming` function handles NaN by returning `''`, so most training examples are empty strings.
- The `target` column parsing fails due to the same quote issue — `target.value_counts()` shows dates instead of 0/1 labels, meaning the label normalization step (`replace {4: 1}`) has no effect on the corrupted rows.

## Imports Present but Unused

`requests`, `pickle`, `scipy.sparse` are imported but not used in existing cells — likely placeholders for model saving and API inference steps that were not yet implemented.
