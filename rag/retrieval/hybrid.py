#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025-08-20 13:31:01 Wednesday

@author: Nikhil Kapila
"""

from __future__ import annotations
from typing import List
from .base import BaseRetriever, RetrievedDocument
from .rrf import rrf

class HybridRetriever(BaseRetriever):
    def __init__(
        self,
        sparse_retriever: BaseRetriever,
        dense_retriever: BaseRetriever,
        # mode: str = "rrf",
        rrf_k: int = 60,
        # weight_sparse: float = 0.5,
        # weight_dense: float = 0.5,
    ) -> None:
        self.sparse = sparse_retriever
        self.dense = dense_retriever
        # self.mode = mode
        self.rrf_k = rrf_k
        # self.weight_sparse = weight_sparse
        # self.weight_dense = weight_dense

    def retrieve(self, query: str, top_k: int) -> List[RetrievedDocument]:
        sparse_list = self.sparse.retrieve(query, top_k)
        dense_list = self.dense.retrieve(query, top_k)
        # if self.mode == "rrf":
        merged = rrf([sparse_list, dense_list], rrf_k=self.rrf_k)
        # else:
            # merged = weighted_linear_fusion([sparse_list, dense_list], [self.weight_sparse, self.weight_dense])
        return merged[:top_k] 