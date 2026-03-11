"""
build_dataset.py

Builds the cleaned dataset from raw parquet files and saves it to data/.
Output: data/shopping_queries_dataset_final.csv with columns:
    query_id, product_id, clean_query, clean_product_document, relevance_score, split
"""

import re
import html
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from pathlib import Path

nltk.download('wordnet', quiet=True)

DATA_DIR = Path(__file__).parent.parent / "data"

STOPWORDS = {
    'i', 'me', 'my', 'a', 'an', 'the', 'and', 'or', 'but', 'are',
    'is', 'was', 'to', 'of', 'in', 'it', 'its', 'this', 'that',
    'for', 'on', 'at', 'be', 'by', 'as', 'up', 'do', 'so', 'if', 's'
}

lemmatizer = WordNetLemmatizer()

RELEVANCE_MAP = {
    'e': 1.0,
    's': 0.6,
    'i': 0.1,
    'c': 0.0,
}


def clean_text(text):
    text = html.unescape(str(text))
    text = re.sub(r'<.*?>|&nbsp;', ' ', text)
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [lemmatizer.lemmatize(t) for t in text.split() if t not in STOPWORDS]
    return ' '.join(tokens)


def build():
    print("Loading raw data...")
    data_examples = pd.read_parquet(DATA_DIR / "shopping_queries_dataset_examples.parquet")
    data_products = pd.read_parquet(DATA_DIR / "shopping_queries_dataset_products.parquet")

    print("Merging examples and products...")
    df = pd.merge(
        data_examples,
        data_products,
        how='left',
        on=['product_locale', 'product_id']
    )

    # Filter to small version, US locale
    df = df[df['small_version'] == 1]
    df = df[df['product_locale'] == 'us'].copy()

    print("Building product documents...")
    for col in ['product_title', 'product_description', 'product_bullet_point', 'product_brand', 'product_color']:
        df[col] = df[col].fillna('')

    df['product_document'] = (
        df['product_title'] + " " +
        df['product_description'] + " " +
        df['product_bullet_point'] + " " +
        df['product_brand'] + " " +
        df['product_color']
    )

    print("Cleaning text (this may take a while)...")
    df['clean_product_document'] = df['product_document'].apply(clean_text)
    df['clean_query'] = df['query'].apply(clean_text)

    df['relevance_score'] = df['esci_label'].str.lower().map(RELEVANCE_MAP)

    df_final = df[[
        'query_id',
        'product_id',
        'clean_query',
        'clean_product_document',
        'relevance_score',
        'split',
    ]].copy()

    output_path = DATA_DIR / "shopping_queries_dataset_final.csv"
    df_final.to_csv(output_path, index=False)
    print(f"Saved {len(df_final)} rows to {output_path}")


if __name__ == "__main__":
    build()