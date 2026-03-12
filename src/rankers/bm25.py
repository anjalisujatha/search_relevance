"""BM25+ ranking model."""

from collections import Counter

import numpy as np

from ..utils.normalize import normalize
from .base import BaseRanker


class BM25Index(BaseRanker):  # pylint: disable=too-many-instance-attributes
    """BM25+ ranker that precomputes IDF and term counters at index time."""

    def __init__(self, corpus, k=1.2, b=0.75, delta=1.0):
        super().__init__(corpus)
        self.k = k
        self.b = b
        self.delta = delta
        self.n_docs = len(corpus)

        # Precompute tokenized docs once at index-build time
        self.tokenized_docs = [normalize(doc) for doc in corpus]

        # Precompute doc lengths, avgdl, and per-doc term counters
        self.doc_lengths = [len(doc) for doc in self.tokenized_docs]
        self.avgdl = sum(self.doc_lengths) / self.n_docs if self.n_docs > 0 else 0
        self.doc_counters = [Counter(doc) for doc in self.tokenized_docs]

        # Build vocabulary and precompute IDF for every term
        vocab = set(w for doc in self.tokenized_docs for w in doc)
        self.idf_cache = {word: self._compute_idf(word) for word in vocab}

    def _compute_idf(self, word):
        doc_freq = sum(1 for doc in self.tokenized_docs if word in doc)
        return float(np.log((self.n_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1))

    def _score_term(self, word, doc_idx):
        tf = self.doc_counters[doc_idx].get(word, 0)
        if tf == 0 or self.avgdl == 0:
            return 0.0

        dl = self.doc_lengths[doc_idx]
        tf_norm = (
            tf * (self.k + 1)
            / (tf + self.k * (1 - self.b + self.b * (dl / self.avgdl)))
        )
        idf = self.idf_cache.get(word, 0.0)
        return idf * (self.delta + tf_norm)

    def score(self, query):
        """Return BM25+ scores for all documents."""
        query_tokens = normalize(query)
        return [
            sum(self._score_term(word, doc_idx) for word in query_tokens)
            for doc_idx in range(self.n_docs)
        ]

    def rank(self, query, top_n=5):
        """Return top_n documents ranked by BM25+ score."""
        scores = self.score(query)
        ranked = sorted(zip(scores, range(self.n_docs)), reverse=True)
        return [
            (round(score, 4), self.corpus[doc_idx])
            for score, doc_idx in ranked[:top_n]
        ]

    def score_docs(self, query, docs):
        """Score an arbitrary list of docs reusing the trained IDF cache and avgdl."""
        query_tokens = normalize(query)
        tokenized = [normalize(doc) for doc in docs]

        scores = []
        for tokens in tokenized:
            dl = len(tokens)
            tf_map = Counter(tokens)
            score = 0.0

            for word in query_tokens:
                tf = tf_map.get(word, 0)
                if tf == 0 or self.avgdl == 0:
                    continue
                idf = self.idf_cache.get(word, 0.0)
                tf_norm = (
                    tf * (self.k + 1)
                    / (tf + self.k * (1 - self.b + self.b * (dl / self.avgdl)))
                )
                score += idf * (self.delta + tf_norm)
            scores.append(score)
        return scores
