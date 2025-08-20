#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025-08-13 17:32:11 Wednesday

@author: Nikhil Kapila
"""

from __future__ import annotations
from typing import Dict, List
from beir.retrieval.evaluation import EvaluateRetrieval

def compute_beir_metrics(qrels, results, k_values: List[int] = [1, 3, 5, 10, 100]) -> Dict[str, Dict[int, float]]:
    print(f"STEP[metrics]: Initializing BEIR evaluator with k_values={k_values}")
    evaluator = EvaluateRetrieval()
    print("STEP[metrics]: Running evaluator.evaluate(...) ...")
    ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, k_values)
    print("STEP[metrics]: Evaluation done. Returning metrics.")
    return {"ndcg": ndcg, "map": _map, "recall": recall, "precision": precision} 