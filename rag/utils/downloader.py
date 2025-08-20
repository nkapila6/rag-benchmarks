#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025-08-13 14:38:47 Wednesday

@author: Nikhil Kapila
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple, Any
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from langchain_core.documents import Document

def download_beir_dataset(dataset: str, out_dir: str) -> str:
    data_path = os.path.join(out_dir, dataset)
    if os.path.isdir(data_path):
        return data_path
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    out_dir = util.download_and_unzip(url, out_dir)
    return os.path.join(out_dir, dataset)

def load_beir(dataset: str, data_dir: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Dict[str, int]]]:
    print(f"STEP[data]: Preparing to load BEIR dataset '{dataset}' from base dir {data_dir}")
    dataset_path = download_beir_dataset(dataset, data_dir)
    print(f"STEP[data]: Loading BEIR data from {dataset_path} (split='test')")
    corpus, queries, qrels = GenericDataLoader(dataset_path).load(split="test")
    print(
        f"STEP[data]: Loaded corpus={len(corpus)} documents, queries={len(queries)}, qrels={len(qrels)}"
    )
    return corpus, queries, qrels

def corpus_to_documents(corpus: Dict[str, Dict[str, Any]]) -> List[Document]:
    docs: List[Document] = []
    for doc_id, entry in corpus.items():
        text_parts = [entry.get("title", ""), entry.get("text", "")]
        content = "\n\n".join([p for p in text_parts if p])
        metadata = {"doc_id": doc_id, **{k: v for k, v in entry.items() if k not in {"text", "title"}}}
        docs.append(Document(page_content=content, metadata=metadata))
    print(f"STEP[data]: Converted corpus to {len(docs)} LangChain Documents")
    return docs 