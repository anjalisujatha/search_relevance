import numpy as np
import scipy.sparse as sp
from enum import Enum
from collections import Counter
from ..utils.normalize import normalize as preprocess
from .base import BaseRanker


class TFMethod(Enum):
    NORMALIZED = "normalized"
    BOOLEAN = "boolean"
    SUMMATION = "summation"
    LOG_SCALED = "log_scaled"

    def compute_vector(self, counts, doc_len):
        """Computes TF values for a dictionary of {word_idx: count}."""
        if self == TFMethod.NORMALIZED:
            return {idx: c / doc_len for idx, c in counts.items()}
        elif self == TFMethod.BOOLEAN:
            return {idx: 1.0 for idx in counts}
        elif self == TFMethod.SUMMATION:
            return {idx: float(c) for idx, c in counts.items()}
        elif self == TFMethod.LOG_SCALED:
            return {idx: np.log10(1 + c) for idx, c in counts.items()}
        return {}


class IDFMethod(Enum):
    STANDARD = "standard"
    SMOOTHED = "smoothed"

    def compute_all(self, df_array, n_docs):
        """Vectorized computation of all IDF values at once."""
        if self == IDFMethod.STANDARD:
            # Avoid division by zero with np.where
            return np.log10(n_docs / np.where(df_array > 0, df_array, 1))
        elif self == IDFMethod.SMOOTHED:
            return np.log10((n_docs + 1) / (df_array + 1)) + 1
        return np.zeros_like(df_array)


class TFIDFRanker(BaseRanker):
    def __init__(self, corpus, tf_method=TFMethod.NORMALIZED, idf_method=IDFMethod.SMOOTHED):
        super().__init__(corpus)
        self._tf_method = tf_method
        self._idf_method = idf_method

        # 1. Tokenize once
        self.tokenized_docs = [preprocess(doc) for doc in corpus]

        # 2. Build Vocabulary mapping
        unique_words = sorted(list(set(w for doc in self.tokenized_docs for w in doc)))
        self.word_to_index = {word: i for i, word in enumerate(unique_words)}
        self.vocab_size = len(unique_words)

        # 3. Build Weight Matrix
        self.matrix = self._build_matrix()

    def _build_matrix(self):
        n_docs = len(self.tokenized_docs)
        if n_docs == 0:
            return sp.csr_matrix((0, 0))

        # Calculate Document Frequency (DF) for the whole corpus
        df_counts = np.zeros(self.vocab_size)
        doc_counters = []

        for doc in self.tokenized_docs:
            counts = Counter(self.word_to_index[w] for w in doc if w in self.word_to_index)
            doc_counters.append(counts)
            for word_idx in counts:
                df_counts[word_idx] += 1

        # Precompute the IDF vector (one value per word in vocab)
        idf_vector = self._idf_method.compute_all(df_counts, n_docs)
        self.idf_vector = idf_vector  # stored for use in score_docs

        # Build sparse TF-IDF matrix (N_docs x M_vocab)
        rows, cols, vals = [], [], []
        for i, counts in enumerate(doc_counters):
            doc_len = len(self.tokenized_docs[i])
            if doc_len == 0:
                continue
            tf_map = self._tf_method.compute_vector(counts, doc_len)
            for idx, tf_val in tf_map.items():
                rows.append(i)
                cols.append(idx)
                vals.append(tf_val * idf_vector[idx])

        return sp.csr_matrix((vals, (rows, cols)), shape=(n_docs, self.vocab_size))

    def score(self, query):
        """Calculates scores using dot product for maximum speed."""
        query_words = preprocess(query)
        query_indices = [self.word_to_index[w] for w in query_words if w in self.word_to_index]

        if not query_indices:
            return np.zeros(len(self.corpus))

        # Create a binary query vector
        query_vec = np.zeros(self.vocab_size)
        query_vec[query_indices] = 1

        # sparse matrix (N x M) dot query_vec (M,) -> scores (N,)
        return np.asarray(self.matrix.dot(query_vec)).flatten()

    def rank(self, query, top_n=5):
        scores = self.score(query)
        n = len(scores)
        if n <= top_n:
            top_indices = np.argsort(scores)[::-1]
        else:
            part = np.argpartition(scores, -top_n)[-top_n:]
            top_indices = part[np.argsort(scores[part])[::-1]]

        return [
            (round(float(scores[idx]), 4), self.corpus[idx])
            for idx in top_indices
        ]

    def score_docs(self, query, docs):
        """Score an arbitrary list of docs reusing the trained vocabulary and IDF values."""
        query_words = preprocess(query)
        query_indices = [self.word_to_index[w] for w in query_words if w in self.word_to_index]
        if not query_indices:
            return [0.0] * len(docs)

        scores = []
        for doc in docs:
            tokens = preprocess(doc)
            counts = Counter(self.word_to_index[w] for w in tokens if w in self.word_to_index)
            doc_len = len(tokens)
            if doc_len == 0:
                scores.append(0.0)
                continue
            tf_map = self._tf_method.compute_vector(counts, doc_len)
            score = sum(tf_map.get(idx, 0.0) * self.idf_vector[idx] for idx in query_indices)
            scores.append(score)
        return scores