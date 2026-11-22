# chatbot/rag_retriever.py
"""
Finds the policy sections that answer a customer's question.

Three stages, the last two optional:
1. Embeddings (always): a multilingual model turns the question and every section into
   vectors, and sections are ranked by cosine similarity. Works across English, Hindi
   and Hinglish, because the model maps the same meaning in any language to similar vectors.
2. Hybrid (optional): BM25 keyword search runs as well, and the two rankings are merged
   with reciprocal rank fusion (RRF).
3. Rerank (optional): a cross-encoder reads the question together with each of the top
   candidates and re-orders them. More accurate per pair, but about 15x slower on a CPU.

Which stages run is set in chatbot/config.py. The Phase 2 benchmark (evals/run_retrieval.py)
found embeddings alone best for our 43 short policy sections; see evals/results/v0-summary.md.
"""

import re
from functools import lru_cache

import numpy as np

from chatbot import config
from chatbot.knowledge_base import Section, load_sections

RRF_K = 60       # standard constant for reciprocal rank fusion
CANDIDATES = 10  # how many sections the hybrid and reranker stages work with

# Some embedding models were trained with a label in front of each text, so the model
# knows which side is the short question and which is the passage. They score noticeably
# worse without it.
PREFIXES = {
    "intfloat/multilingual-e5-small": ("query: ", "passage: "),
    "intfloat/multilingual-e5-base": ("query: ", "passage: "),
}


def tokenize(text: str) -> list[str]:
    # Words in Latin script or Devanagari (\w alone would split Hindi words at vowel signs).
    return re.findall(r"[\wऀ-ॿ]+", text.lower())


# Models are cached so building several retrievers (as the benchmark does) loads each once.
@lru_cache(maxsize=None)
def _embedder(name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name, device="cpu")


@lru_cache(maxsize=None)
def _cross_encoder(name: str):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(name, device="cpu")


def rrf(*rankings: list[int]) -> list[int]:
    """Reciprocal rank fusion: each ranking adds 1/(RRF_K + rank) to a section's score."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (RRF_K + rank)
    return sorted(scores, key=scores.get, reverse=True)


class PolicyRetriever:
    def __init__(
        self,
        sections: list[Section] | None = None,
        embed_model: str | None = config.RETRIEVAL_EMBED_MODEL,
        hybrid: bool = config.RETRIEVAL_HYBRID,
        rerank: bool = config.RETRIEVAL_RERANK,
        rerank_model: str = config.RETRIEVAL_RERANK_MODEL,
    ):
        """embed_model=None gives BM25 only (used by the benchmark as a baseline)."""
        if embed_model is None and not hybrid:
            hybrid = True  # no embeddings: BM25 is the only ranking
        self.sections = sections if sections is not None else load_sections()
        self.embed_model, self.hybrid, self.rerank, self.rerank_model = embed_model, hybrid, rerank, rerank_model

        if embed_model:
            self.query_prefix, passage_prefix = PREFIXES.get(embed_model, ("", ""))
            self.doc_vecs = _embedder(embed_model).encode(
                [passage_prefix + s.text for s in self.sections], normalize_embeddings=True)
        if hybrid:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi([tokenize(s.text) for s in self.sections])

    # -- stages: each returns section indices, best first --------------------

    def _dense(self, query: str) -> list[int]:
        q_vec = _embedder(self.embed_model).encode([self.query_prefix + query], normalize_embeddings=True)[0]
        return list(np.argsort(-(self.doc_vecs @ q_vec)))  # cosine similarity: vectors are normalised

    def _bm25(self, query: str) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        # Only sections sharing at least one word with the question. A Hindi question matches
        # nothing in the English documents; without this filter BM25 would return arbitrary
        # sections and RRF would count that noise as a real vote.
        return [int(i) for i in np.argsort(-scores) if scores[i] > 0]

    def rank(self, query: str) -> list[int]:
        """All stages switched on; returns section indices, best first."""
        dense = self._dense(query) if self.embed_model else []
        if self.hybrid:
            bm25 = self._bm25(query)[:CANDIDATES]
            ranking = rrf(bm25, dense[:CANDIDATES]) if dense else bm25
        else:
            ranking = dense

        if self.rerank:
            # Top up from the dense ranking when the hybrid has fewer than CANDIDATES.
            cands = (ranking + [i for i in dense if i not in ranking])[:CANDIDATES]
            scores = _cross_encoder(self.rerank_model).predict([(query, self.sections[i].text) for i in cands])
            ranking = [cands[i] for i in np.argsort(-scores)]
        return ranking

    def search(self, query: str, k: int = config.RETRIEVAL_TOP_K) -> list[Section]:
        return [self.sections[i] for i in self.rank(query)[:k]]

    def describe(self) -> str:
        parts = [self.embed_model or "no embeddings"]
        if self.hybrid:
            parts.append(f"BM25 (RRF k={RRF_K})")
        if self.rerank:
            parts.append(f"top {CANDIDATES} reranked by {self.rerank_model}")
        return " + ".join(parts)


def build_retriever() -> PolicyRetriever:
    """The retriever as configured in chatbot/config.py (environment variables)."""
    return PolicyRetriever()
