#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025-08-13 15:00:10 Wednesday

@author: Nikhil Kapila
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rag.utils.downloader import download_beir_dataset

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--out", type=str, default="./data")
    args = parser.parse_args()
    path = download_beir_dataset(args.dataset, args.out)
    print(f"Downloaded/ready at: {path}")


if __name__ == "__main__":
    main() 