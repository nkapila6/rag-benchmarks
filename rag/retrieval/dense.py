#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025-08-20 12:19:08 Wednesday

@author: Nikhil Kapila
"""

from __future__ import annotations
from .base import BaseRetriever, RetrievedDocument

import warnings
from typing import List, Tuple
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_community.embeddings")

class DenseRetriever(BaseRetriever):
    def __init__(self, documents: List[Document], 
                 # source: https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
                 # for retrieval, we choose all-minilm-l6-v2 because: 
                 # The all-* models were trained on all available training data (more than 1 billion training pairs) and are designed as general purpose models. 
                 # The all-mpnet-base-v2 model provides the best quality, while all-MiniLM-L6-v2 is 5 times faster and still offers good quality.
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
                 ):
        print(f"STEP[dense]: Initializing DenseRetriever with model='{model_name}' ...")
        self.documents = documents
        self.doc_ids: List[str] = [str(doc.metadata.get("doc_id", idx)) for idx, doc in enumerate(documents)]
        print(f"STEP[dense]: Creating embedding model and building FAISS index for {len(self.doc_ids)} documents ...")
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        # https://github.com/langchain-ai/langchain/discussions/9819 -- 22.08 - added to use max inner product instead
        # https://python.langchain.com/api_reference/community/vectorstores/langchain_community.vectorstores.utils.DistanceStrategy.html#langchain_community.vectorstores.utils.DistanceStrategy
        self.vectorstore = FAISS.from_documents(documents, self.embeddings, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT) 
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
            score = dist # score = 1.0 - dist, not required since range is [0,1]
            results.append(
                RetrievedDocument(
                    doc_id=str(doc.metadata.get("doc_id", "")),
                    document=doc,
                    score=float(score),
                )
            )

        print(f"STEP[dense]: Retrieved {len(results)} results with scores based on absolute distance.")
        return results