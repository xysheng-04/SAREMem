"""Textualized SAREMem retrieval: multi-view dense/sparse fusion and round expansion."""
from __future__ import annotations

import re
import math
from collections import Counter, defaultdict

import numpy as np

_STOP = {"the", "a", "an", "is", "was", "are", "were", "what", "when", "where",
         "who", "how", "did", "does", "do", "of", "to", "in", "on", "and", "or",
         "for", "with", "that", "this", "it", "as", "at", "by", "be"}


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[\w]+", (text or "").lower(), flags=re.UNICODE)
    return [token for token in tokens if token not in _STOP and len(token) >= 2]


def _collapse_view_ranking(indices: np.ndarray, views: list[dict], limit: int) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for index in indices:
        memory_id = views[int(index)]["memory_id"]
        if memory_id not in seen:
            seen.add(memory_id)
            output.append(memory_id)
            if len(output) >= limit:
                break
    return output


def lexical_rank(
    query_text: str,
    views: list[dict],
    pool: int,
    *,
    use_bm25: bool = False,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
) -> list[str]:
    """Rank memory IDs through one lexical lane, collapsing multiple views by max score.

    The compatibility path uses the original query-token coverage score.  BM25 treats each stored
    view as a document, computes corpus-local IDF/length normalization, then keeps the strongest
    view score for each original Add memory before RRF.
    """
    query_tokens = tokenize(query_text)
    if not query_tokens or not views:
        return []
    tokenized_views = [tokenize(row["text"]) for row in views]
    lexical: dict[str, float] = defaultdict(float)
    if not use_bm25:
        query_terms = set(query_tokens)
        for row, tokens in zip(views, tokenized_views):
            overlap = len(query_terms & set(tokens))
            if overlap:
                lexical[row["memory_id"]] = max(
                    lexical[row["memory_id"]], overlap / len(query_terms)
                )
    else:
        document_count = len(tokenized_views)
        average_length = sum(map(len, tokenized_views)) / max(document_count, 1)
        document_frequency: Counter[str] = Counter()
        for tokens in tokenized_views:
            document_frequency.update(set(tokens))
        for row, tokens in zip(views, tokenized_views):
            if not tokens:
                continue
            frequencies = Counter(tokens)
            length_normalizer = 1.0 - bm25_b + bm25_b * len(tokens) / max(average_length, 1e-12)
            score = 0.0
            for term in set(query_tokens):
                term_frequency = frequencies.get(term, 0)
                if not term_frequency:
                    continue
                frequency = document_frequency[term]
                inverse_document_frequency = math.log(
                    1.0 + (document_count - frequency + 0.5) / (frequency + 0.5)
                )
                score += inverse_document_frequency * (
                    term_frequency * (bm25_k1 + 1.0)
                    / (term_frequency + bm25_k1 * length_normalizer)
                )
            if score:
                lexical[row["memory_id"]] = max(lexical[row["memory_id"]], score)
    return [
        memory_id
        for memory_id, _ in sorted(lexical.items(), key=lambda item: (-item[1], item[0]))[:pool]
    ]


def hybrid_rank(
    query_vectors: np.ndarray,
    query_text: str,
    views: list[dict],
    pool: int,
    *,
    use_bm25: bool = False,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
) -> list[tuple[str, float]]:
    if not views:
        return []
    dimensions = {int(row["embedding"].shape[0]) for row in views}
    if len(dimensions) != 1 or query_vectors.shape[1] not in dimensions:
        raise ValueError("stored and query embedding dimensions do not match")
    matrix = np.stack([row["embedding"] for row in views])
    lanes: list[list[str]] = []
    for query_vector in query_vectors:
        scores = matrix @ query_vector.astype(np.float32)
        order = np.argsort(-scores)
        lanes.append(_collapse_view_ranking(order, views, pool))

    lexical_lane = lexical_rank(
        query_text,
        views,
        pool,
        use_bm25=use_bm25,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
    )
    if lexical_lane:
        lanes.append(lexical_lane)

    fused: dict[str, float] = defaultdict(float)
    for lane in lanes:
        for rank, memory_id in enumerate(lane):
            fused[memory_id] += 1.0 / (61 + rank)
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:pool]
