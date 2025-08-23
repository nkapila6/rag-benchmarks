#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025-08-20 13:56:01 Wednesday

@author: Nikhil Kapila
"""

from __future__ import annotations
from typing import Dict, Callable, List, Optional
from ..retrieval import (
    TFIDFRetriever,
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    BaseRetriever,
)
from ..rerankers import (
    NoReranker,
    BiEncoderReranker,
    CrossEncoderReranker,
    BaseReranker,
)


class RetrievalPipeline:
    def __init__(self, retriever: BaseRetriever, reranker: BaseReranker):
        self.retriever = retriever
        self.reranker = reranker

    def run(self, query: str, top_k: int):
        print(f"STEP[pipeline]: Retrieving top-{top_k} candidates...")
        initial = self.retriever.retrieve(query, top_k)
        print(f"STEP[pipeline]: Reranking {len(initial)} candidates...")
        return self.reranker.rerank(query, initial, top_k)


# Generic builders

def make_retriever(
    kind: str,
    documents: List[Document],
    dense_model: str,
    rrf_k: int=60,
    device: Optional[str] = None,
) -> BaseRetriever:
    """Factory function to create a retriever."""
    print(
        f"STEP[factory]: Building retriever kind='{kind}' "
        f"(dense_model={dense_model}, rrf_k={rrf_k}, "
    )
    
    if kind == "tfidf":
        return TFIDFRetriever(documents)
    if kind == "bm25":
        return BM25Retriever(documents)
    if kind == "dense":
        return DenseRetriever(documents, model_name=dense_model or "sentence-transformers/all-MiniLM-L6-v2", device=device)
    if kind == "hybrid":
        sparse = BM25Retriever(documents)
        dense = DenseRetriever(documents, model_name=dense_model or "sentence-transformers/all-MiniLM-L6-v2", device=device)
        return HybridRetriever(
            sparse_retriever=sparse,
            dense_retriever=dense,
            rrf_k=rrf_k,
        )
    raise ValueError(f"Unknown retriever kind: {kind}")


def make_reranker(
    kind: str,
    bi_model: str,
    cross_model: str,
    batch_size: int,
    device: Optional[str] = None,
) -> BaseReranker:
    """Factory function to create a reranker."""
    print(
        f"STEP[factory]: Building reranker kind='{kind}' (bi_model={bi_model}, cross_model={cross_model}, batch_size={batch_size})"
    )
    if kind == "none":
        return NoReranker()
    if kind == "bi":
        return BiEncoderReranker(model_name=bi_model or "sentence-transformers/msmarco-distilbert-base-tas-b", batch_size=batch_size, device=device)
    if kind == "cross":
        return CrossEncoderReranker(model_name=cross_model or "cross-encoder/ms-marco-MiniLM-L-12-v2", batch_size=batch_size, device=device)
    raise ValueError(f"Unknown reranker kind: {kind}")

# Named combos for reproducibility
class TFIDF_NoRerank(RetrievalPipeline):
    def __init__(self, documents):
        super().__init__(TFIDFRetriever(documents), NoReranker())


class TFIDF_Bi(RetrievalPipeline):
    def __init__(self, documents, bi_model: str = "sentence-transformers/msmarco-distilbert-base-tas-b", batch_size: int = 32):
        super().__init__(TFIDFRetriever(documents), BiEncoderReranker(model_name=bi_model, batch_size=batch_size))


class TFIDF_Cross(RetrievalPipeline):
    def __init__(self, documents, cross_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2", batch_size: int = 32):
        super().__init__(TFIDFRetriever(documents), CrossEncoderReranker(model_name=cross_model, batch_size=batch_size))


class BM25_NoRerank(RetrievalPipeline):
    def __init__(self, documents):
        super().__init__(BM25Retriever(documents), NoReranker())


class BM25_Bi(RetrievalPipeline):
    def __init__(self, documents, bi_model: str = "sentence-transformers/msmarco-distilbert-base-tas-b", batch_size: int = 32):
        super().__init__(BM25Retriever(documents), BiEncoderReranker(model_name=bi_model, batch_size=batch_size))


class BM25_Cross(RetrievalPipeline):
    def __init__(self, documents, cross_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2", batch_size: int = 32):
        super().__init__(BM25Retriever(documents), CrossEncoderReranker(model_name=cross_model, batch_size=batch_size))


class Dense_NoRerank(RetrievalPipeline):
    def __init__(self, documents, dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__(DenseRetriever(documents, model_name=dense_model), NoReranker())


class Dense_Bi(RetrievalPipeline):
    def __init__(
        self,
        documents,
        dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        bi_model: str = "sentence-transformers/msmarco-distilbert-base-tas-b",
        batch_size: int = 32,
    ):
        super().__init__(DenseRetriever(documents, model_name=dense_model), BiEncoderReranker(model_name=bi_model, batch_size=batch_size))


class Dense_Cross(RetrievalPipeline):
    def __init__(
        self,
        documents,
        dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        cross_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        batch_size: int = 32,
    ):
        super().__init__(DenseRetriever(documents, model_name=dense_model), CrossEncoderReranker(model_name=cross_model, batch_size=batch_size))


class Hybrid_NoRerank(RetrievalPipeline):
    def __init__(
        self,
        documents,
        dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        rrf_k: int = 60,
    ):
        sparse = BM25Retriever(documents)
        dense = DenseRetriever(documents, model_name=dense_model)
        super().__init__(HybridRetriever(sparse, dense, rrf_k=rrf_k), NoReranker())


class Hybrid_Bi(RetrievalPipeline):
    def __init__(
        self,
        documents,
        dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        bi_model: str = "sentence-transformers/msmarco-distilbert-base-tas-b",
        batch_size: int = 32,
        rrf_k: int = 60,
    ):
        sparse = BM25Retriever(documents)
        dense = DenseRetriever(documents, model_name=dense_model)
        super().__init__(
            HybridRetriever(sparse, dense, rrf_k=rrf_k),
            BiEncoderReranker(model_name=bi_model, batch_size=batch_size),
        )


class Hybrid_Cross(RetrievalPipeline):
    def __init__(
        self,
        documents,
        dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        cross_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        batch_size: int = 32,
        rrf_k: int = 60,
    ):
        sparse = BM25Retriever(documents)
        dense = DenseRetriever(documents, model_name=dense_model)
        super().__init__(
            HybridRetriever(sparse, dense, rrf_k=rrf_k),
            CrossEncoderReranker(model_name=cross_model, batch_size=batch_size),
        )


# Simple registry
COMBOS: Dict[str, Callable[..., RetrievalPipeline]] = {
    "tfidf+none": TFIDF_NoRerank,
    "tfidf+bi": TFIDF_Bi,
    "tfidf+cross": TFIDF_Cross,
    "bm25+none": BM25_NoRerank,
    "bm25+bi": BM25_Bi,
    "bm25+cross": BM25_Cross,
    "dense+none": Dense_NoRerank,
    "dense+bi": Dense_Bi,
    "dense+cross": Dense_Cross,
    "hybrid+none": Hybrid_NoRerank,
    "hybrid+bi": Hybrid_Bi,
    "hybrid+cross": Hybrid_Cross,
}