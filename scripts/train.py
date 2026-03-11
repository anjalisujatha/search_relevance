"""
Four-stage pipeline:
  1. fit      – train TF-IDF and BM25 on the training split, pickle model objects
  2. load     – load pickled models from disk
  3. predict  – score every test query's candidate docs
  4. evaluate – compute NDCG@k from predicted scores vs. ground-truth relevance

Usage:
    python -m scripts.train            # runs all stages
    python -m scripts.train --fit      # fit + save only
    python -m scripts.train --predict  # load + predict + evaluate only
"""

import argparse
import pickle
import time
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from src.rankers.base import BaseRanker
from src.rankers.bm25 import BM25Index
from src.rankers.tf_idf import TFIDFRanker

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
CONFIG_PATH = ROOT_DIR / "configs" / "train.yaml"

TFIDF_PATH = MODELS_DIR / "tfidf.pkl"
BM25_PATH = MODELS_DIR / "bm25.pkl"

SPLIT_TRAIN = "train"
SPLIT_TEST = "test"
REQUIRED_COLUMNS = {"split", "clean_query", "clean_product_document", "product_id", "relevance_score", "query_id"}


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Stage 1 – Fit
# ---------------------------------------------------------------------------

def fit(train_df: pd.DataFrame, overwrite: bool = False) -> tuple[TFIDFRanker, BM25Index]:
    """Fit both rankers on the training corpus and pickle them to MODELS_DIR."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not overwrite and TFIDF_PATH.exists() and BM25_PATH.exists():
        print("  Models already exist. Use --overwrite to retrain.")
        return load_models()

    corpus = train_df["clean_product_document"].dropna().unique().tolist()
    print(f"  Training corpus: {len(corpus):,} unique documents")

    print("  Fitting TF-IDF ...")
    t0 = time.time()
    tfidf = TFIDFRanker(corpus)
    print(f"  TF-IDF trained in {time.time() - t0:.2f}s")

    print("  Fitting BM25 ...")
    t0 = time.time()
    bm25 = BM25Index(corpus)
    print(f"  BM25 trained in {time.time() - t0:.2f}s")

    with open(TFIDF_PATH, "wb") as f:
        pickle.dump(tfidf, f)
    print(f"  Saved TF-IDF  → {TFIDF_PATH}")

    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)
    print(f"  Saved BM25    → {BM25_PATH}")

    return tfidf, bm25


# ---------------------------------------------------------------------------
# Stage 2 – Predict
# ---------------------------------------------------------------------------

def load_models() -> tuple[TFIDFRanker, BM25Index]:
    """Load pickled rankers from disk."""
    if not TFIDF_PATH.exists() or not BM25_PATH.exists():
        raise FileNotFoundError(
            f"Model files not found in {MODELS_DIR}. Run fit first."
        )
    with open(TFIDF_PATH, "rb") as f:
        tfidf = pickle.load(f)
    with open(BM25_PATH, "rb") as f:
        bm25 = pickle.load(f)
    print(f"  Loaded TF-IDF from {TFIDF_PATH}")
    print(f"  Loaded BM25   from {BM25_PATH}")
    return tfidf, bm25


def predict(ranker: BaseRanker, test_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """
    Score every test query's candidate documents.

    Returns a DataFrame with columns:
        query_id, product_id, predicted_score, relevance_score
    """
    records = []
    for query_id, group in test_df.groupby("query_id"):
        query = group["clean_query"].iloc[0]
        docs = group["clean_product_document"].tolist()
        scores = ranker.score_docs(query, docs)

        for product_id, relevance_score, score in zip(group["product_id"], group["relevance_score"], scores):
            records.append({
                "model": model_name,
                "query_id": query_id,
                "product_id": product_id,
                "predicted_score": score,
                "relevance_score": relevance_score,
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Stage 3 – Evaluate
# ---------------------------------------------------------------------------

def _ndcg_at_k(predicted_scores: np.ndarray, true_relevances: np.ndarray, k: int) -> float:
    order = np.argsort(predicted_scores)[::-1][:k]
    gains = true_relevances[order]
    discounts = np.log2(np.arange(2, len(gains) + 2))
    dcg = np.sum(gains / discounts)

    ideal_order = np.argsort(true_relevances)[::-1][:k]
    ideal_gains = true_relevances[ideal_order]
    idcg = np.sum(ideal_gains / np.log2(np.arange(2, len(ideal_gains) + 2)))

    return dcg / idcg if idcg > 0 else 0.0


def evaluate(predictions_df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """
    Compute mean NDCG@k per model.

    Returns a summary DataFrame with columns: model, ndcg@k
    """
    results = []
    for model_name, model_preds in predictions_df.groupby("model"):
        ndcg_scores = []
        for query_id, group in model_preds.groupby("query_id"):
            ndcg_scores.append(
                _ndcg_at_k(
                    group["predicted_score"].values,
                    group["relevance_score"].values,
                    k=k,
                )
            )
        results.append({"model": model_name, f"ndcg@{k}": round(float(np.mean(ndcg_scores)), 4)})

    return pd.DataFrame(results).sort_values(f"ndcg@{k}", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(do_fit: bool = True, do_predict: bool = True, overwrite: bool = False):
    cfg = load_config()
    dataset_path = ROOT_DIR / cfg["data"]["dataset_path"]
    train_size = cfg["sampling"]["train_size"]
    test_size = cfg["sampling"]["test_size"]
    k = cfg["evaluation"]["k"]

    print(f"Loading dataset from {dataset_path} ...")
    df = pd.read_csv(dataset_path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    train_pool = df[df["split"] == SPLIT_TRAIN]
    test_pool = df[df["split"] == SPLIT_TEST]
    train_df = train_pool.sample(n=min(train_size, len(train_pool)), random_state=42).reset_index(drop=True)
    test_df = test_pool.sample(n=min(test_size, len(test_pool)), random_state=42).reset_index(drop=True)
    print(f"  Train rows: {len(train_df):,}  |  Test rows: {len(test_df):,}")

    if do_fit:
        print("\n[Stage 1] Fitting models ...")
        fit(train_df, overwrite=overwrite)

    if do_predict:
        print("\n[Stage 2] Loading models ...")
        tfidf, bm25 = load_models()

        print("\n[Stage 3] Running predictions on test split ...")
        preds_tfidf = predict(tfidf, test_df, model_name="tfidf")
        preds_bm25 = predict(bm25, test_df, model_name="bm25")
        all_preds = pd.concat([preds_tfidf, preds_bm25], ignore_index=True)

        print(f"\n[Stage 4] Evaluating (NDCG@{k}) ...")
        summary = evaluate(all_preds, k=k)
        print(summary.to_string(index=False))
        return summary

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", action="store_true", help="Fit and save models only")
    parser.add_argument("--predict", action="store_true", help="Load models, predict and evaluate only")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing model files")
    args = parser.parse_args()

    if args.fit:
        run(do_fit=True, do_predict=False, overwrite=args.overwrite)
    elif args.predict:
        run(do_fit=False, do_predict=True, overwrite=args.overwrite)
    else:
        run(do_fit=True, do_predict=True, overwrite=args.overwrite)