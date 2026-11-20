"""Extractive baselines: Lead and the bag-of-words TextRank of the original thesis system."""

from __future__ import annotations

import re
from collections import Counter
from typing import List, Optional

import numpy as np


def lead_summary(text: str, n_words: int) -> str:
    return " ".join(text.split()[:n_words])


def sparse_pagerank(bags: List[Counter], alpha: float = 0.85, max_iter: int = 100, tol: float = 1.0e-6) -> Optional[np.ndarray]:
    """Weighted PageRank over the sentence cosine-similarity graph without forming the n x n matrix.

    Matches nx.pagerank(nx.from_numpy_array(S)) for S[i, j] = cos(bag_i, bag_j), S[i, i] = 0,
    including uniform redistribution from dangling nodes and the L1 < n * tol stopping rule.
    Returns None when the iteration does not converge.
    """
    from scipy.sparse import csr_matrix, diags

    n = len(bags)
    vocab, rows, cols, vals = {}, [], [], []
    for i, bag in enumerate(bags):
        for word, count in bag.items():
            rows.append(i)
            cols.append(vocab.setdefault(word, len(vocab)))
            vals.append(float(count))
    X = csr_matrix((vals, (rows, cols)), shape=(n, max(len(vocab), 1)))
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    Xn = diags(np.divide(1.0, norms, out=np.zeros(n), where=norms > 0)) @ X
    self_similarity = np.asarray(Xn.multiply(Xn).sum(axis=1)).ravel()

    def times_similarity(v):
        return Xn @ (Xn.T @ v) - self_similarity * v

    degree = times_similarity(np.ones(n))
    dangling = degree <= 1e-12
    inv_degree = np.divide(1.0, degree, out=np.zeros(n), where=~dangling)
    x = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        last = x
        x = alpha * times_similarity(last * inv_degree) + (alpha * last[dangling].sum() + 1 - alpha) / n
        if np.abs(x - last).sum() < n * tol:
            return x
    return None


def textrank_sentences(text: str) -> List[str]:
    from nltk.tokenize import sent_tokenize

    return sent_tokenize(text)


def textrank_summary(text: str, n_words: int) -> str:
    """Top-ranked sentences up to n_words, in document order (stopwords removed before ranking)."""
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    sentences = textrank_sentences(text)
    if not sentences:
        return ""
    stop = set(stopwords.words("english"))
    bags = [Counter(w for w in word_tokenize(re.sub(r"[^\w\s]", "", s.lower())) if w not in stop) for s in sentences]
    scores = sparse_pagerank(bags)
    if scores is None:
        scores = np.ones(len(sentences))

    chosen, total = [], 0
    for i in sorted(range(len(sentences)), key=lambda k: scores[k], reverse=True):
        if total >= n_words:
            break
        chosen.append(i)
        total += len(sentences[i].split())
    return " ".join(sentences[i] for i in sorted(chosen))
