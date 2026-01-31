import argparse
import os
from typing import Tuple, Union

import numpy as np
import torch

from toolbox.filesystem import make_dataset, read_text_file, write_text_file


def make_parser():
    parser = argparse.ArgumentParser(description="Retrieve features")
    parser.add_argument("-l", "--lib", type=str, help="Directory to the library files")
    parser.add_argument("-q", "--query", type=str, help="Directory to the query files")
    parser.add_argument("-o", "--output", type=str, help="Path to the output file")
    return parser


def load_embeddings(path_or_dir: Union[str, list]) -> Tuple[list, torch.Tensor]:
    """
    Load_embeddings from file(s) of directory.
    Args:
        path_or_dir (str | list): Path to the file or directory.
    """
    file_list = []
    if isinstance(path_or_dir, list):
        file_list = path_or_dir
    elif os.path.isdir(path_or_dir):
        file_list = make_dataset(path_or_dir)
    else:
        file_list = [path_or_dir]

    embeddings = [np.load(path) for path in file_list]
    keys = [os.path.basename(path) for path in file_list]
    embeddings = torch.from_numpy(np.stack(embeddings, axis=0)).float()
    return keys, embeddings


if __name__ == "__main__":
    args = make_parser().parse_args()

    lib_keys, lib_embeddings = load_embeddings(args.lib)
    query_keys, query_embeddings = load_embeddings(args.query)
    if torch.cuda.is_available():
        lib_embeddings = lib_embeddings.cuda()
        query_embeddings = query_embeddings.cuda()

    # Compute the similarity
    values = lib_embeddings / lib_embeddings.norm(dim=1, keepdim=True)  # [M, C]
    quires = query_embeddings / query_embeddings.norm(dim=1, keepdim=True)  # [N, C]
    similarity = quires @ values.t()  # [N, M]
    # get the max similarity for each query
    max_similarity, indices = similarity.max(dim=1)  # [N]
    max_similarity = max_similarity.cpu().tolist()

    results = [
        "\t".join([key, str(score)]) for key, score in zip(query_keys, max_similarity)
    ]
    write_text_file(args.output, results)
