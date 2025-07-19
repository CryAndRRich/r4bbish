import numpy as np
from collections import Counter
from sklearn.cluster import KMeans, Birch, AgglomerativeClustering

def align_labels(reference_labels: np.ndarray,
                 target_labels: np.ndarray,
                 n_clusters: int) -> np.ndarray:
    """
    Align target cluster labels to match the reference clusters using majority vote.

    For each cluster in the target label set, it finds the most common corresponding label
    in the reference label set and remaps accordingly

    Parameters:
        reference_labels: 1D array of reference labels
        target_labels: 1D array of target labels to align
        n_clusters: Number of unique clusters in the labels

    Returns:
        np.ndarray: Aligned version of target_labels
    """
    mapped = np.zeros_like(target_labels)
    for cluster_id in range(n_clusters):
        # Indices of samples belonging to this cluster in the target
        indices = np.where(target_labels == cluster_id)[0]
        if len(indices) == 0:
            continue
        # Find the most common corresponding reference label
        ref_sub = reference_labels[indices]
        most_common = Counter(ref_sub).most_common(1)[0][0]
        mapped[indices] = most_common
    return mapped


def cluster_and_vote(features: np.ndarray,
                     n_clusters: int = 50,
                     reject_threshold: float = 0.6) -> np.ndarray:
    """
    Perform clustering using multiple algorithms and combine results via majority voting

    This function runs KMeans, Birch, and Agglomerative Clustering on the same feature set,
    aligns the label spaces using KMeans as reference, then uses a voting mechanism to decide
    on a final label for each point. If no consensus is reached (below threshold), label is -1

    Parameters:
        features: 2D array of input features with shape (N, D)
        n_clusters: Number of clusters for each algorithm
        reject_threshold (float): Minimum agreement fraction to accept a cluster label;
                                  otherwise, the point is labeled as -1 (rejected)

    Returns:
        np.ndarray: 1D array of final cluster labels, with -1 indicating rejection
    """
    # Initialize clustering algorithms
    clusterers = {
        'kmeans': KMeans(n_clusters=n_clusters, random_state=42),
        'birch': Birch(n_clusters=n_clusters),
        'agg': AgglomerativeClustering(n_clusters=n_clusters)
    }

    # Fit each clustering algorithm and get predictions
    preds = {
        name: clf.fit_predict(features)
        for name, clf in clusterers.items()
    }

    # Use KMeans as the reference for label alignment
    ref = preds['kmeans']
    aligned_preds = {'kmeans': ref}
    for name in ['birch', 'agg']:
        aligned_preds[name] = align_labels(ref, preds[name], n_clusters)

    # Final label assignment by majority vote
    final_labels = np.full(features.shape[0], -1, dtype=int)
    for i in range(features.shape[0]):
        votes = [aligned_preds[name][i] for name in ['kmeans', 'birch', 'agg']]
        vote_count = Counter(votes)
        label, count = vote_count.most_common(1)[0]
        if count / len(votes) >= reject_threshold:
            final_labels[i] = label

    return final_labels
