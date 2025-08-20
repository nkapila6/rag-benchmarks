#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025-08-13 18:22:12 Wednesday

@author: Nikhil Kapila
"""

import argparse
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag.utils.downloader import load_beir, corpus_to_documents
from rag.eval.eval_pipeline import evaluate_on_beir
from rag.utils.pipeline import make_retriever, make_reranker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--retriever", type=str, choices=["tfidf", "bm25", "dense", "hybrid"], required=True)
    parser.add_argument("--top_k", type=int, default=10)

    parser.add_argument("--dense_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--hybrid_mode", type=str, choices=["rrf", "weighted"], default="rrf")
    parser.add_argument("--rrf_k", type=int, default=60)
    parser.add_argument("--w_sparse", type=float, default=0.5)
    parser.add_argument("--w_dense", type=float, default=0.5)

    parser.add_argument("--reranker", type=str, choices=["none", "bi", "cross"], required=True)
    parser.add_argument("--bi_model", type=str, default="sentence-transformers/msmarco-distilbert-base-tas-b")
    parser.add_argument("--cross_model", type=str, default="cross-encoder/ms-marco-MiniLM-L-12-v2")
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--output", type=str, default="./outputs/results.json")

    args = parser.parse_args()

    print("STEP: Parsed arguments:")
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "data_dir": args.data_dir,
                "retriever": args.retriever,
                "top_k": args.top_k,
                "dense_model": args.dense_model,
                "hybrid_mode": args.hybrid_mode,
                "rrf_k": args.rrf_k,
                "w_sparse": args.w_sparse,
                "w_dense": args.w_dense,
                "reranker": args.reranker,
                "bi_model": args.bi_model,
                "cross_model": args.cross_model,
                "batch_size": args.batch_size,
                "output": args.output,
            },
            indent=2,
        )
    )

    print("STEP: Loading BEIR dataset...")
    corpus, queries, qrels = load_beir(args.dataset, args.data_dir)
    documents = corpus_to_documents(corpus)
    print(
        f"STEP: Loaded dataset '{args.dataset}'. Corpus size: {len(corpus)}, "
        f"Queries: {len(queries)}, Qrels: {len(qrels)}, Documents converted: {len(documents)}"
    )

    print("STEP: Building retriever...")
    retriever = make_retriever(
        kind=args.retriever,
        documents=documents,
        dense_model=args.dense_model,
        rrf_k=args.rrf_k,
    )
    print("STEP: Retriever built.")
    print("STEP: Building reranker...")
    reranker = make_reranker(kind=args.reranker, bi_model=args.bi_model, cross_model=args.cross_model, batch_size=args.batch_size)
    print("STEP: Reranker built.")

    print("STEP: Running evaluation over queries...")
    metrics = evaluate_on_beir(retriever, reranker, queries, qrels, top_k=args.top_k)
    print("STEP: Evaluation complete.")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"STEP: Saving metrics to {args.output} ...")
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)
    print("STEP: Metrics saved.")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main() 