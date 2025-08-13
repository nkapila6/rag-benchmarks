#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025-08-13 14:28:29 Wednesday

@author: Nikhil Kapila
"""

from __future__ import annotations
from typing import List
from .base import BaseReranker
from ..retrieval.base import RetrievedDocument


class NoReranker(BaseReranker):
    def rerank(self, query: str, candidates: List[RetrievedDocument], top_k: int) -> List[RetrievedDocument]:
        return candidates[:top_k] 