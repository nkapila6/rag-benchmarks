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
    def __init__(self, model_name: str = "sentence-transformers/msmarco-distilbert-base-tas-b", batch_size: int = 32):
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size

    def rerank(self, query: str, candidates: List[RetrievedDocument], top_k: int) -> List[RetrievedDocument]:
        if not candidates:
            return []
        passages = [c.document.page_content for c in candidates]
        query_emb = self.model.encode([query], batch_size=1, convert_to_tensor=True, normalize_embeddings=True)
        doc_embs = self.model.encode(passages, batch_size=self.batch_size, convert_to_tensor=True, normalize_embeddings=True)
        sims = util.cos_sim(query_emb, doc_embs).cpu().numpy().flatten()
        order = np.argsort(-sims)
        reranked: List[RetrievedDocument] = []
        for idx in order[:top_k]:
            item = candidates[idx]
            reranked.append(RetrievedDocument(doc_id=item.doc_id, document=item.document, score=float(sims[idx])))
        return reranked 