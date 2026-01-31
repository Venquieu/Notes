import logging
import sys

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances


def get_medoid(embeddings, labels, cluster_id):
    """Select the point with the smallest average distance to other points within the cluster"""
    mask = labels == cluster_id
    cluster_embs = embeddings[mask]

    dists = pairwise_distances(cluster_embs, metric="cosine")
    medoid_idx = np.argmin(dists.mean(axis=1))

    ori_idx = np.where(mask)[0]
    return ori_idx[medoid_idx]


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d | %(levelname)s | %(module)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    embed_file = sys.argv[1]
    output_file = sys.argv[2]
    eps = float(sys.argv[3])

    embeddings = np.load(embed_file)  # (N, D)
    # (OPTIONAL) PCA dimensionality reduction
    pca = PCA(n_components=0.95)
    embeddings_new = pca.fit_transform(embeddings)
    # DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=5, metric="cosine", n_jobs=-1)
    labels = dbscan.fit_predict(embeddings_new)

    unique_labels = set(labels) - {-1}  # remove noise label
    rep_idx = []
    for label in unique_labels:
        idx = get_medoid(embeddings_new, labels, label)
        rep_idx.extend(idx.tolist())

    logging.info(
        "#samples: %d\t#clusters: %d\t#noise: %d"
        % (len(embeddings), len(unique_labels), np.sum(labels == -1))
    )
    np.save(output_file, embeddings[rep_idx])
    logging.info("Saved representative embeddings to %s" % output_file)
