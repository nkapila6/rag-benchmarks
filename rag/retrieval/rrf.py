#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025-08-20 13:17:42 Wednesday

@author: Nikhil Kapila
"""

from __future__ import annotations
from typing import Dict, List
from .base import RetrievedDocument

# reciprocal rank fusion for aggregation from multiple retrievers
def rrf(rankings: List[List[RetrievedDocument]], rrf_k:int=60)->List[RetrievedDocument]:
    scores = {}
    doc_map = {}
    for r in rankings:
        for rank_idx, item in enumerate(r):
            scores[item.doc_id] = scores.get(item.doc_id, 0.0) + 1.0 / (rrf_k+rank_idx+1)
            if item.doc_id not in doc_map:
                doc_map[item.doc_id] = item

    merged = [RetrievedDocument(doc_id=doc_id, document=doc_map[doc_id].document, score=s) for doc_id, s in scores.items()]
    merged.sort(key=lambda x:x.score,reverse=True)
    return merged
            
    