#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025-08-20 12:19:08 Wednesday

@author: Nikhil Kapila
"""

from __future__ import annotations
from .base import BaseRetriever, RetrievedDocument

import warnings
import numpy as np
from typing import List, Tuple
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_community.embeddings")

class DenseRetriever(BaseRetriever):
    def __init__(self, documents: List[Document], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        print(f"STEP[dense]: Initializing DenseRetriever with model='{model_name}' ...")
        self.documents = documents
        self.doc_ids: List[str] = [str(doc.metadata.get("doc_id", idx)) for idx, doc in enumerate(documents)]
        print(f"STEP[dense]: Creating embedding model and building FAISS index for {len(self.doc_ids)} documents ...")
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        print("STEP[dense]: FAISS index built.")

    def retrieve(self, query: str, top_k: int) -> List[RetrievedDocument]:
        """
        Retrieves documents based on semantic similarity.
        """
        safe_query_preview = query[:60].replace("\n", " ")
        print(f"STEP[dense]: Retrieving for query='{safe_query_preview}...' top_k={top_k}")

        if not self.documents:
            return []

        print("STEP[dense]: Running similarity search in FAISS ...")
        docs_and_distances: List[Tuple[Document, float]] = self.vectorstore.similarity_search_with_score(query, k=top_k)

        results: List[RetrievedDocument] = []
        for doc, dist in docs_and_distances:
            score = np.exp(-dist) # non linear mapping from dist to score, 0 is 1
            
            results.append(
                RetrievedDocument(
                    doc_id=str(doc.metadata.get("doc_id", "")),
                    document=doc,
                    score=float(score),
                )
            )

        print(f"STEP[dense]: Retrieved {len(results)} results with scores based on absolute distance.")
        return results