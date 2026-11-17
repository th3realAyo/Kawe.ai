"""
chunk_ranker.py

Takes the ExtractedSegment list produced by doc_loader.py, splits it into
smaller passages ("chunks"), embeds them via the Modal embed_chunks
endpoint, and ranks them by semantic importance so only the most
representative passages get sent to the (expensive) LLM rewriting step.

Ranking approach: this is the embedding-based counterpart to what
textrank_summarizer.py did with bag-of-words cosine similarity. Instead of
scoring sentences by word overlap, we build a similarity graph from
semantic embeddings and run PageRank over it — a chunk that is
semantically "central" (similar to many other chunks) is treated as more
representative of the document's core content.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import numpy as np
import requests


class ChunkRankerError(Exception):
    """Raised when embedding via Modal fails, or ranking can't be computed."""


@dataclass
class Chunk:
    text: str
    location_label: str  # inherited from the parent ExtractedSegment
    order: int  # inherited from the parent ExtractedSegment (document order)
    chunk_index: int  # position among all chunks, for stable tie-breaking


# Passages larger than this get split further before embedding/ranking —
# keeps each unit small enough for a focused, grounded per-chunk summary
# later, and avoids diluting a single embedding vector with too much content.
MAX_WORDS_PER_CHUNK = 180

# How many of the highest-scoring chunks get passed on to the LLM rewriting
# step. Tunable: higher = more thorough/expensive summary, lower = terser.
DEFAULT_TOP_N = 6


def split_into_chunks(segments: List["ExtractedSegment"]) -> List[Chunk]:
    """Splits each ExtractedSegment into one or more Chunks, preserving the
    segment's location_label and order so citations still resolve correctly
    even when a page/section gets split into multiple chunks."""
    chunks: List[Chunk] = []
    idx = 0
    for seg in segments:
        words = seg.text.split()
        if len(words) <= MAX_WORDS_PER_CHUNK:
            chunks.append(
                Chunk(
                    text=seg.text,
                    location_label=seg.location_label,
                    order=seg.order,
                    chunk_index=idx,
                )
            )
            idx += 1
        else:
            for start in range(0, len(words), MAX_WORDS_PER_CHUNK):
                sub_text = " ".join(words[start : start + MAX_WORDS_PER_CHUNK])
                chunks.append(
                    Chunk(
                        text=sub_text,
                        location_label=seg.location_label,
                        order=seg.order,
                        chunk_index=idx,
                    )
                )
                idx += 1
    return chunks


def _embed_via_modal(texts: List[str]) -> List[List[float]]:
    """Calls the Modal embed_chunks endpoint. Expects MODAL_EMBED_URL and
    RAG_AUTH_TOKEN to be set as environment variables (HF Space secrets)."""
    endpoint = os.environ["MODAL_EMBED_URL"]
    token = os.environ["RAG_AUTH_TOKEN"]

    try:
        response = requests.post(
            endpoint,
            json={"texts": texts},
            headers={"Authorization": f"Bearer {token}"},
            timeout=300,
        )
    except requests.RequestException as e:
        raise ChunkRankerError(f"Could not reach embedding endpoint: {e}") from e

    if response.status_code != 200:
        raise ChunkRankerError(
            f"Embedding request failed ({response.status_code}): {response.text}"
        )

    return response.json()["embeddings"]


def _semantic_centrality_scores(vectors: List[List[float]]) -> List[float]:
    """PageRank over a cosine-similarity graph of chunk embeddings. Vectors
    from the Modal endpoint are already L2-normalized, so a dot product is
    equivalent to cosine similarity."""
    import networkx as nx

    matrix = np.array(vectors)
    similarity_matrix = matrix @ matrix.T
    np.fill_diagonal(similarity_matrix, 0.0)  # no self-loops

    graph = nx.from_numpy_array(similarity_matrix)

    try:
        scores = nx.pagerank(graph, weight="weight")
    except nx.PowerIterationFailedConvergence:
        # Fall back to equal weighting rather than failing the whole request
        # over a rare convergence edge case on unusual similarity matrices.
        scores = {i: 1.0 for i in range(len(vectors))}

    return [scores[i] for i in range(len(vectors))]


def rank_chunks(chunks: List[Chunk], top_n: int = DEFAULT_TOP_N) -> List[Chunk]:
    """Returns the top_n most semantically important chunks, restored to
    original document order (so the assembled summary reads coherently
    top-to-bottom rather than jumbled by importance rank)."""
    if not chunks:
        return []

    if len(chunks) <= top_n:
        return sorted(chunks, key=lambda c: c.order)

    vectors = _embed_via_modal([c.text for c in chunks])
    scores = _semantic_centrality_scores(vectors)

    scored_chunks = list(zip(chunks, scores))
    scored_chunks.sort(key=lambda pair: pair[1], reverse=True)
    top_chunks = [chunk for chunk, _ in scored_chunks[:top_n]]

    top_chunks.sort(key=lambda c: (c.order, c.chunk_index))
    return top_chunks
