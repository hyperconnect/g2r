"""
Automatic evaluation on generation results
"""
import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from nltk.tokenize.casual import casual_tokenize
from nltk.util import ngrams
from tabulate import tabulate
from tqdm import tqdm


def calculate_dist_metric(responses: List[str], n: int) -> float:
    tokenized_responses = [casual_tokenize(resp) for resp in responses]
    num_all_ngrams = 0
    all_ngram_set = set()

    for tokens in tokenized_responses:
        token_ngrams = list(ngrams(tokens, n))
        num_all_ngrams += len(token_ngrams)
        all_ngram_set.update(token_ngrams)

    return len(all_ngram_set) / num_all_ngrams


def calculate_average_length(responses: List[str]) -> float:
    tokenized_responses = [casual_tokenize(resp) for resp in responses]
    return np.mean([len(tokens) for tokens in tokenized_responses])


def evaluate_single_result(results_path,
                           dist_n_list):
    with open(results_path) as f:
        examples = [json.loads(line.strip()) for line in f]
        responses = [ex["response"] for ex in examples]

    metrics = {
        "avg_length": calculate_average_length(responses),
    }

    for n in dist_n_list:
        metrics[f"dist_{n}"] = calculate_dist_metric(responses, n)
    return metrics


def main(args):
    all_metrics = {}
    for result_path in tqdm(args.result_paths):
        model_name = Path(result_path).stem
        all_metrics[model_name] = evaluate_single_result(
            result_path,
            args.dist_n_list,
        )

    all_metrics_df = pd.DataFrame(all_metrics).transpose()
    print(tabulate(all_metrics_df, headers="keys"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-paths", type=str, nargs="+")
    parser.add_argument("--dist-n-list", type=int, nargs="*", default=[2, 3])

    args = parser.parse_args()
    main(args)
