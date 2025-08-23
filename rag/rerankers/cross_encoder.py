#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025-08-20 13:37:15 Wednesday

@author: Nikhil Kapila
"""

from __future__ import annotations
from typing import List, Optional
import numpy as np
from sentence_transformers.cross_encoder import CrossEncoder
from .base import BaseReranker
from ..retrieval.base import RetrievedDocument

class CrossEncoderReranker(BaseReranker):
    def __init__(self, 
                 # source: https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2
                 # we choose ms-marco-miniLM-L6-v2 as it has the highest score and can process 1.8k docs per sec
                 model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2", 
                 batch_size: int = 32,
                 # if None, checks GPU to use
                 # https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html
                 # https://sbert.net/docs/package_reference/cross_encoder/cross_encoder.html
                 device=None
                 ):
        self.model = CrossEncoder(model_name, device=device)
        self.batch_size = batch_size
        print(
            f"STEP[cross-reranker]: Initialized cross-encoder model='{model_name}', batch_size={batch_size}"
        )

    def rerank(self, query: str, candidates: List[RetrievedDocument], top_k: int) -> List[RetrievedDocument]:
        if not candidates:
            return []
        print(
            f"STEP[cross-reranker]: Reranking {len(candidates)} candidates for query (top_k={top_k})"
        )
        pairs = [(query, c.document.page_content) for c in candidates]
        print(
            f"STEP[cross-reranker]: Prepared {len(pairs)} (query, passage) pairs"
        )
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        scores = np.asarray(scores).flatten()
        if scores.size:
            print(
                f"STEP[cross-reranker]: Score stats -> min={float(scores.min()):.4f}, max={float(scores.max()):.4f}, mean={float(scores.mean()):.4f}"
            )
        order = np.argsort(-scores)
        reranked: List[RetrievedDocument] = []
        for idx in order[:top_k]:
            item = candidates[idx]
            reranked.append(RetrievedDocument(doc_id=item.doc_id, document=item.document, score=float(scores[idx])))
        top_ids = [r.doc_id for r in reranked]
        top_scores = [f"{r.score:.4f}" for r in reranked]
        print(
            f"STEP[cross-reranker]: Top-{len(reranked)} doc_ids={top_ids} scores={top_scores}"
        )
        return reranked