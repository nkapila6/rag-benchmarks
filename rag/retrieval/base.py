#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025-08-13 12:40:01 Wednesday

@author: Nikhil Kapila
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Protocol
from langchain_core.documents import Document

@dataclass
class RetrievedDocument:
    doc_id: str
    document: Document
    score: float

# abstract class for retrievers
class BaseRetriever(Protocol):
    def retrieve(self, query:str, top_k:int) -> List[RetrievedDocument]:
        ...