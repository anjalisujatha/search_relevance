from abc import ABC, abstractmethod


class BaseRanker(ABC):
    def __init__(self, corpus):
        self.corpus = corpus

    @abstractmethod
    def score(self, query):
        """Return a relevance score for every document in the corpus.

        Args:
            query: raw query string

        Returns:
            List of floats, one score per document.
        """

    @abstractmethod
    def rank(self, query, top_n=5):
        """Return the top_n most relevant documents for the query.

        Args:
            query : raw query string
            top_n : number of results to return

        Returns:
            List of (score, document) tuples sorted highest to lowest.
        """