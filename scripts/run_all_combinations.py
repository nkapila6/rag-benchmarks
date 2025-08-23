"""
Script to run all retrieval and reranking combinations for each dataset in the data folder.
"""
import os
import subprocess
import sys
import json
import pandas as pd

# Define datasets
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
DATASETS = [
    d for d in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, d)) and os.path.exists(os.path.join(DATA_DIR, d, 'corpus.jsonl'))
]

# Define retrieval and rerank combos
COMBINATIONS = [
    # Sparse retrieval
    {"retrieval": "tfidf", "rerank": None, "desc": "TFIDF + no reranking (BASELINE)"},
    {"retrieval": "tfidf", "rerank": "bi_encoder", "desc": "TFIDF + bi encoder rerank"},
    {"retrieval": "tfidf", "rerank": "cross_encoder", "desc": "TFIDF + cross encoder rerank"},
    {"retrieval": "bm25", "rerank": None, "desc": "BM25 + No reranking"},
    {"retrieval": "bm25", "rerank": "bi_encoder", "desc": "BM25 + bi encoder rerank"},
    {"retrieval": "bm25", "rerank": "cross_encoder", "desc": "BM25 + cross encoder rerank"},
    # Dense retrieval
    {"retrieval": "dense", "rerank": None, "desc": "Dense + no rerank"},
    {"retrieval": "dense", "rerank": "bi_encoder", "desc": "Dense + bi encoder"},
    {"retrieval": "dense", "rerank": "cross_encoder", "desc": "Dense + cross encoder"},
    # Hybrid retrieval (RRF)
    {"retrieval": "hybrid", "rerank": None, "desc": "Hybrid + no rerank"},
    {"retrieval": "hybrid", "rerank": "bi_encoder", "desc": "Hybrid + bi encoder"},
    {"retrieval": "hybrid", "rerank": "cross_encoder", "desc": "Hybrid + cross encoder"},
]


def main():
    script_dir = os.path.dirname(__file__)
    run_experiment_script = os.path.join(script_dir, "run_experiment.py")
    output_dir = os.path.abspath(os.path.join(script_dir, '..', 'outputs'))
    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    for dataset in DATASETS:
        print(f"\n=== Running experiments for dataset: {dataset} ===")
        for combo in COMBINATIONS:
            print(f"\n--- {combo['desc']} ---")
            try:
                reranker_name = combo["rerank"].split('_')[0] if combo["rerank"] else "none"
                output_filename = f"{dataset}_{combo['retrieval']}_{reranker_name}.json"
                output_path = os.path.join(output_dir, output_filename)

                command = [
                    sys.executable,
                    run_experiment_script,
                    "--dataset",
                    dataset,
                    "--retriever",
                    combo["retrieval"],
                    "--reranker",
                    reranker_name,
                    "--output",
                    output_path,
                    "--top_k",
                    "100",
                ]
                subprocess.run(command, check=True)

                with open(output_path, 'r') as f:
                    results = json.load(f)
                
                results['dataset'] = dataset
                results['retriever'] = combo['retrieval']
                results['reranker'] = combo['rerank'] if combo['rerank'] else "none"
                all_results.append(results)

            except subprocess.CalledProcessError as e:
                print(f"Error running {combo['desc']} for {dataset}: {e}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

    results_df = pd.DataFrame(all_results)
    results_csv_path = os.path.abspath(os.path.join(script_dir, '..', 'all_experiments_results.csv'))
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nAll experiments complete. Results saved to {results_csv_path}")


if __name__ == "__main__":
    main()
