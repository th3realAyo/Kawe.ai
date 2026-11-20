"""Passage splitting, centrality ranking and redundancy-aware selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np

from kawe.text import split_sentences

MAX_WORDS_PER_CHUNK = 180
MIN_TAIL_WORDS = 40  # a shorter final piece of a segment is merged into the piece before it
DEFAULT_TOP_N = 6
DIVERSITY = 0.5  # weight of the redundancy penalty in MMR selection; 0 = pure centrality

EmbedFn = Callable[[List[str]], List[List[float]]]


@dataclass
class Chunk:
    text: str
    location_label: str
    order: int  # order of the parent segment
    chunk_index: int  # position among all chunks of the document


def _split_segment(text: str, max_words: int = MAX_WORDS_PER_CHUNK, min_tail: int = MIN_TAIL_WORDS) -> List[str]:
    """Packs whole sentences into pieces of at most max_words; only over-long sentences are cut."""
    if len(text.split()) <= max_words:
        return [text]
    pieces: List[List[str]] = []
    current: List[str] = []
    for sentence in split_sentences(text):
        words = sentence.split()
        while len(words) > max_words:  # a "sentence" with no punctuation, e.g. a table dump
            if current:
                pieces.append(current)
                current = []
            pieces.append(words[:max_words])
            words = words[max_words:]
        if current and len(current) + len(words) > max_words:
            pieces.append(current)
            current = []
        current = current + words
    if current:
        if pieces and len(current) < min_tail:
            pieces[-1] = pieces[-1] + current
        else:
            pieces.append(current)
    return [" ".join(p) for p in pieces]


def split_into_chunks(segments: Sequence) -> List[Chunk]:
    """Splits each extracted segment into passages of whole sentences, about MAX_WORDS_PER_CHUNK words each."""
    chunks: List[Chunk] = []
    for seg in segments:
        for piece in _split_segment(seg.text):
            chunks.append(Chunk(piece, seg.location_label, seg.order, len(chunks)))
    return chunks


def _similarity(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=float)
    similarity = np.clip(matrix @ matrix.T, 0.0, None)  # PageRank needs non-negative edge weights
    np.fill_diagonal(similarity, 0.0)
    return similarity


def pagerank(weights: np.ndarray, alpha: float = 0.85, max_iter: int = 100, tol: float = 1.0e-6) -> Optional[np.ndarray]:
    """Weighted PageRank by power iteration; same result as nx.pagerank(nx.from_numpy_array(weights)).

    Dangling nodes spread their rank uniformly. Returns None if it does not converge.
    """
    n = weights.shape[0]
    degree = weights.sum(axis=1)
    dangling = degree <= 1e-12
    transition = np.divide(weights, degree[:, None], out=np.zeros_like(weights), where=~dangling[:, None])
    x = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        last = x
        x = alpha * (last @ transition) + (alpha * last[dangling].sum() + 1 - alpha) / n
        if np.abs(x - last).sum() < n * tol:
            return x
    return None


def centrality_scores(vectors: Sequence[Sequence[float]]) -> List[float]:
    """PageRank over the cosine-similarity graph of L2-normalized embeddings."""
    n = len(vectors)
    if n == 0:
        return []
    scores = pagerank(_similarity(vectors))
    if scores is None:
        return [1.0 / n] * n
    return scores.tolist()


def select_top(
    chunks: List[Chunk],
    scores: Sequence[float],
    top_n: int,
    vectors: Optional[Sequence[Sequence[float]]] = None,
    diversity: float = DIVERSITY,
) -> List[Chunk]:
    """Picks top_n chunks, returned in document order.

    Without vectors this is plain highest-score selection. With vectors it is maximal marginal
    relevance: each pick maximizes (1 - diversity) * centrality - diversity * (similarity to the
    closest passage already picked), so near-duplicate central passages are not all chosen.
    """
    top_n = min(top_n, len(chunks))
    if vectors is None or diversity <= 0:
        ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
        picked = ranked[:top_n]
    else:
        values = np.asarray(scores, dtype=float)
        spread = values.max() - values.min()
        relevance = (values - values.min()) / spread if spread > 0 else np.ones_like(values)
        matrix = np.asarray(vectors, dtype=float)
        similarity = matrix @ matrix.T
        picked = [int(np.argmax(relevance))]
        closest = similarity[picked[0]].copy()
        available = np.ones(len(chunks), dtype=bool)
        available[picked[0]] = False
        while len(picked) < top_n:
            gain = np.where(available, (1 - diversity) * relevance - diversity * closest, -np.inf)
            best = int(np.argmax(gain))
            picked.append(best)
            available[best] = False
            closest = np.maximum(closest, similarity[best])
    return sorted((chunks[i] for i in picked), key=lambda c: (c.order, c.chunk_index))


def rank_chunks(chunks: List[Chunk], top_n: int = DEFAULT_TOP_N, embed_fn: Optional[EmbedFn] = None) -> List[Chunk]:
    if not chunks:
        return []
    if len(chunks) <= top_n:
        return sorted(chunks, key=lambda c: c.order)
    if embed_fn is None:
        from kawe.modal_client import embed as embed_fn
    vectors = embed_fn([c.text for c in chunks])
    return select_top(chunks, centrality_scores(vectors), top_n, vectors)
