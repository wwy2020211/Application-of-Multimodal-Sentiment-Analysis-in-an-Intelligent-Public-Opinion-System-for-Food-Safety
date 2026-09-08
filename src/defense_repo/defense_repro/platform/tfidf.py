from __future__ import annotations
import math
from collections import Counter


def tfidf(tokenized_docs: list[list[str]]):
    """
    Pure-Python TF-IDF matching the platform slide's keyword-weighting step.
    Returns list[dict[token, score]].
    """
    n = len(tokenized_docs)
    df = Counter()
    for doc in tokenized_docs:
        df.update(set(doc))

    out = []
    for doc in tokenized_docs:
        counts = Counter(doc)
        L = max(len(doc), 1)
        row = {}
        for token, c in counts.items():
            tf = c / L
            idf = math.log((1 + n) / (1 + df[token])) + 1.0
            row[token] = tf * idf
        out.append(row)
    return out
