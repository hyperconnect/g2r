"""
Generate examples using generative models
"""
import argparse
import math
from pathlib import Path

import faiss
import numpy as np
import torch


def _get_index_key_based_on_embedding_num(num_embeddings: int) -> str:
    """https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index"""
    if num_embeddings < 5e6:
        k = 8 * math.sqrt(num_embeddings)
        k_power_2 = 2 ** math.floor(math.log2(k))
        return f"IVF{k_power_2},Flat"
    elif num_embeddings < 1e7:
        return "IVF65536_HNSW32,Flat"
    elif num_embeddings < 1e8:
        return "IVF262144_HNSW32,Flat"
    else:
        return "IVF1048576_HNSW32,Flat"


def build_faiss_index(embedding_encs):
    num_candidates, embedding_dim = embedding_encs.shape
    index = faiss.index_factory(
        embedding_dim,
        "HNSW32,Flat",
        faiss.METRIC_INNER_PRODUCT,
    )

    embedding_encs = embedding_encs.numpy().astype(np.float32)
    index.add(embedding_encs)
    return index


def main(args):
    if Path(args.index_save_path).exists():
        print("Index already exists. Abort")
        return

    print("Loading torch vector")
    encs = torch.load(args.input_encs_path, map_location="cpu")
    print("Start building faiss index")
    index = build_faiss_index(encs)
    print("Building index done.")

    print("Sanity check:")
    print(index.search(encs[0:1].numpy(), 5))

    faiss.write_index(index, args.index_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-encs-path", type=str)
    parser.add_argument("--index-save-path", type=str)

    args = parser.parse_args()
    main(args)
