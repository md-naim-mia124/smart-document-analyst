#modules\bm25.py

from typing import List
import math
from collections import Counter, defaultdict

class BM25:
    """
    Minimal BM25 (Okapi) implementation over a list of documents (strings).
    Tokenization: simple whitespace/lowercase split.
    """
    def __init__(self, docs: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b
        self.docs = [self._tok(d) for d in docs]
        self.N = len(self.docs)
        self.df = defaultdict(int)
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = sum(self.doc_len)/max(self.N,1)
        for d in self.docs:
            for t in set(d):
                self.df[t] += 1
        self.idf = {t: math.log(1 + (self.N - df + 0.5)/(df + 0.5)) for t, df in self.df.items()}

    def _tok(self, s: str):
        return s.lower().split()

    def get_scores(self, query: str) -> List[float]:
        q = self._tok(query)
        scores = [0.0]*self.N
        for i, d in enumerate(self.docs):
            dl = self.doc_len[i]
            score = 0.0
            f = Counter(d)
            for term in q:
                if term not in self.idf:
                    continue
                idf = self.idf[term]
                freq = f.get(term, 0)
                denom = freq + self.k1*(1 - self.b + self.b*dl/self.avgdl)
                if denom > 0:
                    score += idf * (freq * (self.k1 + 1)) / denom
            scores[i] = score
        return scores

    def topn(self, query: str, n: int):
        scores = self.get_scores(query)
        idxs = sorted(range(self.N), key=lambda i: scores[i], reverse=True)[:n]
        return [(i, scores[i]) for i in idxs]
