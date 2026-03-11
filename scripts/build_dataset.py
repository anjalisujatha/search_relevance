"""
build_dataset.py

Builds the cleaned dataset from raw parquet files and saves it to data/processed/.
Input:  data/raw/shopping_queries_dataset_examples.parquet
        data/raw/shopping_queries_dataset_products.parquet
Output: data/processed/shopping_queries_dataset_final.csv with columns:
    query_id, product_id, clean_query, clean_product_document, relevance_score, split
"""

import re
import html
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from pathlib import Path

from src.utils import normalize
nltk.download('wordnet', quiet=True)

ROOT_DIR = Path(__file__).parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

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
    tokens = [lemmatizer.lemmatize(t) for t in text.split() if t not in normalize.STOPWORDS]
    return ' '.join(tokens)


def build():
    print("Loading raw data...")
    data_examples = pd.read_parquet(RAW_DIR / "shopping_queries_dataset_examples.parquet")
    data_products = pd.read_parquet(RAW_DIR / "shopping_queries_dataset_products.parquet")

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

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / "shopping_queries_dataset_final.csv"
    df_final.to_csv(output_path, index=False)
    print(f"Saved {len(df_final)} rows to {output_path}")


if __name__ == "__main__":
    build()