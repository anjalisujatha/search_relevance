# Search Relevance

Implements and evaluates TF-IDF and BM25+ ranking models on the Amazon ESCI e-commerce search dataset. Models are scored using NDCG@k to determine which produces more relevant search rankings.

## Models

### TF-IDF
Sparse matrix-based implementation supporting multiple TF variants (normalized, boolean, summation, log-scaled) and IDF variants (standard, smoothed). Uses `scipy.sparse.csr_matrix` for memory efficiency.

### BM25+
Probabilistic ranking with parameters `k=1.2` (term saturation), `b=0.75` (length normalization), and `delta=1.0`. Precomputes IDF and document lengths for fast inference.


## Evaluation

Models are evaluated using **NDCG@10** (Normalized Discounted Cumulative Gain at k=10), averaged across all test queries.

| Model  | NDCG@10 |
|--------|---------|
| BM25+  | 0.9538  |
| TF-IDF | 0.9534  |

Relevance scores are derived from ESCI labels:

| Label | Meaning      | Score |
|-------|--------------|-------|
| e     | Exact match  | 1.0   |
| s     | Substitute   | 0.6   |
| i     | Irrelevant   | 0.1   |
| c     | Complement   | 0.0   |


## Setup

Requires Python 3.9+.

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

### 1. Build the dataset

Download the [Amazon ESCI dataset](https://github.com/amazon-science/esci-data) parquet files into `data/raw/`, then run:

```bash
python -m scripts.build_dataset
```

This merges product metadata with query-product pairs, applies text normalization, maps ESCI labels to relevance scores, and outputs `data/processed/shopping_queries_dataset_final.csv`.

### 2. Train and evaluate

```bash
# Full pipeline: fit models, score test queries, evaluate
python -m scripts.train

# Fit only (saves models to data/models/)
python -m scripts.train --fit

# Evaluate only (loads saved models)
python -m scripts.train --predict

# Overwrite existing saved models
python -m scripts.train --overwrite
```

### Configuration

Edit `configs/train.yaml` to adjust dataset paths, sample sizes, and evaluation settings:

```yaml
data:
  dataset_path: data/processed/shopping_queries_dataset_final.csv
sampling:
  train_size: 10000
  test_size: 1000
evaluation:
  k: 10
```



