#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025-08-13 13:52:15 Wednesday

@author: Nikhil Kapila
"""


from __future__ import annotations
from typing import List
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from .base import BaseRetriever, RetrievedDocument

class BM25Retriever(BaseRetriever):
    def __init__(self, docs:List[Document]):
        self.docs = docs
        self.doc_ids = [str(doc.metadata.get("doc_id", idx)) for idx, doc in enumerate(docs)]
        tokenized = [self._tokenize(doc.page_content) for doc in docs] # https://pypi.org/project/rank-bm25/
        self.bm25 = BM25Okapi(tokenized)

    @staticmethod
    def _tokenize(text:str)->List[str]:
        return text.lower().split()
        
    def retrieve(self, query: str, top_k: int) -> List[RetrievedDocument]:
        if len(self.docs) == 0: 
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

        results = []

        for i,score in ranked:
            results.append(
                RetrievedDocument(
                    doc_id=self.doc_ids[i],
                    document=self.docs[i],
                    score=float(score)
                    )
                )

        return results