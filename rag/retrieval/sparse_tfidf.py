#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025-08-13 12:58:29 Wednesday

@author: Nikhil Kapila
"""

from __future__ import annotations
from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.documents import Document

from .base import BaseRetriever, RetrievedDocument

class TFIDFRetriever(BaseRetriever):
    def __init__(self, docs:List[Document]):
        self.docs = docs
        self.doc_ids = [str(doc.metadata.get("doc_id", idx)) for idx, doc in enumerate(docs)]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform([doc.page_content for doc in docs])

    def retrieve(self, query: str, top_k: int) -> List[RetrievedDocument]:
        if len(self.docs) == 0: 
            return []

        query_vec = self.vectorizer.transform([query])
        # 1xN matrix of similarities
        sims = cosine_similarity(query_vec, self.matrix).flatten()
        
        if np.isnan(sims).any(): # replace any NaN to 0.0
            sims = np.nan_to_num(sims)

        #top k sort
        idx = np.argsort(-sims)[:top_k]
        results = []

        for i in idx:
            results.append(
                RetrievedDocument(
                    doc_id = self.doc_ids[i],
                    document=self.docs[i],
                    score=float(sims[i])
                    )
                )

        return results