"""
pipeline.py

Three-stage pipeline:
  1. fit   – train TF-IDF and BM25 on the training split, pickle model objects
  2. predict – load pickled models, score every test query's candidate docs
  3. evaluate – compute NDCG@k from predicted scores vs. ground-truth relevance

Usage:
    python -m src.pipeline            # runs all three stages
    python -m src.pipeline --fit      # fit + save only
    python -m src.pipeline --predict  # load + predict + evaluate only
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from src.rankers.bm25 import BM25Index
from src.rankers.tf_idf import TFIDFRanker

DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = DATA_DIR / "models"
DATASET_PATH = DATA_DIR / "shopping_queries_dataset_final.csv"

TFIDF_PATH = MODELS_DIR / "tfidf.pkl"
BM25_PATH = MODELS_DIR / "bm25.pkl"


# ---------------------------------------------------------------------------
# Stage 1 – Fit
# ---------------------------------------------------------------------------

def fit(train_df: pd.DataFrame) -> tuple[TFIDFRanker, BM25Index]:
    """Fit both rankers on the training corpus and pickle them to MODELS_DIR."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    corpus = train_df["clean_product_document"].dropna().unique().tolist()
    print(f"  Training corpus: {len(corpus):,} unique documents")

    print("  Fitting TF-IDF ...")
    tfidf = TFIDFRanker(corpus)

    print("  Fitting BM25 ...")
    bm25 = BM25Index(corpus)

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


def predict(ranker, test_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
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

        for (_, row), score in zip(group.iterrows(), scores):
            records.append({
                "model": model_name,
                "query_id": query_id,
                "product_id": row["product_id"],
                "predicted_score": score,
                "relevance_score": row["relevance_score"],
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

def run(do_fit: bool = True, do_predict: bool = True, k: int = 10):
    print(f"Loading dataset from {DATASET_PATH} ...")
    df = pd.read_csv(DATASET_PATH)

    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]
    print(f"  Train rows: {len(train_df):,}  |  Test rows: {len(test_df):,}")

    if do_fit:
        print("\n[Stage 1] Fitting models ...")
        fit(train_df)

    if do_predict:
        print("\n[Stage 2] Loading models ...")
        tfidf, bm25 = load_models()

        print("\n[Stage 2] Running predictions on test split ...")
        preds_tfidf = predict(tfidf, test_df, model_name="tfidf")
        preds_bm25 = predict(bm25, test_df, model_name="bm25")
        all_preds = pd.concat([preds_tfidf, preds_bm25], ignore_index=True)

        print(f"\n[Stage 3] Evaluating (NDCG@{k}) ...")
        summary = evaluate(all_preds, k=k)
        print(summary.to_string(index=False))
        return summary

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", action="store_true", help="Fit and save models only")
    parser.add_argument("--predict", action="store_true", help="Load models, predict and evaluate only")
    args = parser.parse_args()

    if args.fit:
        run(do_fit=True, do_predict=False)
    elif args.predict:
        run(do_fit=False, do_predict=True)
    else:
        run(do_fit=True, do_predict=True)
