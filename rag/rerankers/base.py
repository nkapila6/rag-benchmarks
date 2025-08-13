#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025-08-13 14:26:49 Wednesday

@author: Nikhil Kapila
"""

from __future__ import annotations
from typing import List, Protocol
from ..retrieval.base import RetrievedDocument


class BaseReranker(Protocol):
    def rerank(self, query: str, candidates: List[RetrievedDocument], top_k: int) -> List[RetrievedDocument]:
        ... 