#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025-08-13 14:33:56 Wednesday

@author: Nikhil Kapila
"""

from __future__ import annotations
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer, util
from .base import BaseReranker
from ..retrieval.base import RetrievedDocument

class BiEncoderReranker(BaseReranker):
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2",
                 # source: https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
                 # for reranking, we choose all-mpnet because: 
                 # The all-* models were trained on all available training data (more than 1 billion training pairs) and are designed as general purpose models. 
                 # The all-mpnet-base-v2 model provides the best quality, while all-MiniLM-L6-v2 is 5 times faster and still offers good quality.
                 batch_size: int = 32):
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        print(
            f"STEP[bi-reranker]: Initialized bi-encoder model='{model_name}', batch_size={batch_size}"
        )

    def rerank(self, query: str, candidates: List[RetrievedDocument], top_k: int) -> List[RetrievedDocument]:
        if not candidates:
            return []
        print(
            f"STEP[bi-reranker]: Reranking {len(candidates)} candidates for query (top_k={top_k})"
        )
        passages = [c.document.page_content for c in candidates]
        print(
            f"STEP[bi-reranker]: Prepared passages (num_docs={len(passages)})"
        )
        query_emb = self.model.encode([query], batch_size=1, convert_to_tensor=True, normalize_embeddings=True)
        doc_embs = self.model.encode(passages, batch_size=self.batch_size, convert_to_tensor=True, normalize_embeddings=True)
        try:
            print(
                f"STEP[bi-reranker]: Encoded embeddings -> query_emb.shape={tuple(query_emb.shape)}, doc_embs.shape={tuple(doc_embs.shape)}"
            )
        except Exception:
            # Shape introspection best-effort for different tensor backends
            print("STEP[bi-reranker]: Encoded embeddings (shapes unavailable)")
        sims = util.cos_sim(query_emb, doc_embs).cpu().numpy().flatten()
        if sims.size:
            print(
                f"STEP[bi-reranker]: Similarity stats -> min={float(sims.min()):.4f}, max={float(sims.max()):.4f}, mean={float(sims.mean()):.4f}"
            )
        order = np.argsort(-sims)
        reranked: List[RetrievedDocument] = []
        for idx in order[:top_k]:
            item = candidates[idx]
            reranked.append(RetrievedDocument(doc_id=item.doc_id, document=item.document, score=float(sims[idx])))
        top_ids = [r.doc_id for r in reranked]
        top_scores = [f"{r.score:.4f}" for r in reranked]
        print(
            f"STEP[bi-reranker]: Top-{len(reranked)} doc_ids={top_ids} scores={top_scores}"
        )
        return reranked 