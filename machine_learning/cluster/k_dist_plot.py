import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

MIN_SAMPLES = 20  # usually 5~100

embeddings = np.load(sys.argv[1])  # (N, D)

# normalize + euclidean -> cosine
embeddings = normalize(embeddings, norm='l2')
nbrs = NearestNeighbors(n_neighbors=MIN_SAMPLES, metric='euclidean', n_jobs=-1)
nbrs.fit(embeddings)
distances, _ = nbrs.kneighbors(embeddings)
k_distances = distances[:, -1]
k_distances_sorted = np.sort(k_distances)

# plot
plt.figure(figsize=(10, 6))
plt.plot(k_distances_sorted, marker='.', linestyle='none')
plt.xlabel('Points (sorted by k-distance)')
plt.ylabel(f'{MIN_SAMPLES}-distance')
plt.title(f'K-distance Graph for DBSCAN (k = {MIN_SAMPLES})')
plt.grid(True)
plt.show()
