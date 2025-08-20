#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025-08-13 18:20:48 Wednesday

@author: Nikhil Kapila
"""

from __future__ import annotations

from typing import Dict, Any, List

from ..retrieval.base import BaseRetriever, RetrievedDocument
from ..rerankers.base import BaseReranker
from .metrics import compute_beir_metrics

def run_pipeline_over_queries(
    retriever: BaseRetriever,
    reranker: BaseReranker,
    queries: Dict[str, str],
    top_k: int,
) -> Dict[str, List[RetrievedDocument]]:
    results: Dict[str, List[RetrievedDocument]] = {}
    total = len(queries)
    print(f"STEP[eval]: Running pipeline over {total} queries with top_k={top_k}")
    for qid, query in queries.items():
        print(f"STEP[eval]: Query id={qid}: retrieving candidates ...")
        initial = retriever.retrieve(query, top_k)
        print(f"STEP[eval]: Query id={qid}: reranking {len(initial)} candidates ...")
        reranked = reranker.rerank(query, initial, top_k)
        print(f"STEP[eval]: Query id={qid}: got {len(reranked)} results after rerank")
        results[qid] = reranked
    return results

def to_beir_results_format(results: Dict[str, List[RetrievedDocument]]) -> Dict[str, Dict[str, float]]:
    formatted: Dict[str, Dict[str, float]] = {}
    for qid, items in results.items():
        formatted[qid] = {item.doc_id: float(item.score) for item in items}
    return formatted


def evaluate_on_beir(
    retriever: BaseRetriever,
    reranker: BaseReranker,
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    top_k: int,
) -> Dict[str, Any]:
    print("STEP[eval]: Starting evaluation...")
    res = run_pipeline_over_queries(retriever, reranker, queries, top_k)
    beir_res = to_beir_results_format(res)
    print("STEP[eval]: Computing BEIR metrics ...")
    metrics = compute_beir_metrics(qrels, beir_res, k_values=[1, 3, 5, 10, 100])
    print("STEP[eval]: Metrics computed.")
    return metrics