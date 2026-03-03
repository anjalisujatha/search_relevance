import numpy as np
from enum import Enum
from ..utils.preprocessing import normalize as preprocess
from .base import BaseRanker


class TFMethod(Enum):
    NORMALIZED = "normalized"
    BOOLEAN    = "boolean"
    SUMMATION  = "summation"
    LOG_SCALED = "log_scaled"

    def compute(self, word, document):
        if self == TFMethod.NORMALIZED:
            return document.count(word) / len(document)
        elif self == TFMethod.BOOLEAN:
            return int(word in document)
        elif self == TFMethod.SUMMATION:
            return document.count(word)
        elif self == TFMethod.LOG_SCALED:
            return np.log10(1 + document.count(word))
        return None


class IDFMethod(Enum):
    STANDARD = "standard"
    SMOOTHED = "smoothed"   # default — avoids division by zero, always positive

    def compute(self, word, corpus):
        if self == IDFMethod.STANDARD:
            n_docs = len(corpus)
            n_docs_with_word = sum(1 for doc in corpus if word in doc)
            if n_docs_with_word == 0:
                return 0
            return np.log10(n_docs / n_docs_with_word)
        elif self == IDFMethod.SMOOTHED:
            n_docs = len(corpus) + 1
            n_docs_with_word = sum(1 for doc in corpus if word in doc) + 1
            return np.log10(n_docs / n_docs_with_word) + 1


class TFIDFRanker(BaseRanker):

    def __init__(self, corpus, tf_method=TFMethod.NORMALIZED, idf_method=IDFMethod.SMOOTHED):
        super().__init__(corpus)
        self.corpus      = corpus
        self._tf_method  = tf_method
        self._idf_method = idf_method
        self.tokenized_docs = [preprocess(doc) for doc in corpus]
        self.matrix, self.word_list, self.word_to_index = self._build_matrix()

    def _tf_idf(self, word, document, tokenized_corpus):
        """Compute tf-idf using the selected TF and IDF variants."""
        return self._tf_method.compute(word, document) * self._idf_method.compute(word, tokenized_corpus)

    def _build_matrix(self):
        """Precompute tf-idf vectors for all documents at index-build time."""
        word_list     = list(dict.fromkeys(word for doc in self.tokenized_docs for word in doc))
        word_to_index = {word: i for i, word in enumerate(word_list)}

        matrix = []
        for doc in self.tokenized_docs:
            vector = [0.0] * len(word_list)
            for word in doc:
                vector[word_to_index[word]] = self._tf_idf(word, doc, self.tokenized_docs)
            matrix.append(vector)

        return matrix, word_list, word_to_index

    def score(self, query):
        """Return tf-idf scores for all documents."""
        query_words = preprocess(query)
        return [
            sum(
                doc_vector[self.word_to_index[word]]
                for word in query_words
                if word in self.word_to_index
            )
            for doc_vector in self.matrix
        ]

    def rank(self, query, top_n=5):
        """Return top_n documents ranked by tf-idf score."""
        scores = self.score(query)
        ranked = sorted(zip(scores, range(len(self.corpus))), reverse=True)
        return [
            (round(score, 4), self.corpus[doc_idx])
            for score, doc_idx in ranked[:top_n]
            if score > 0
        ]