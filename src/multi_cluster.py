import numpy as np
from collections import Counter
from sklearn.cluster import KMeans, Birch, AgglomerativeClustering

def align_labels(reference_labels, target_labels, n_clusters):
    """
    Map target_labels to reference_labels clusters via majority vote.
    Args:
        reference_labels (np.ndarray): 1D array of reference cluster labels (len N).
        target_labels (np.ndarray): 1D array of target cluster labels (len N).
        n_clusters (int): number of clusters in both label sets.
    Returns:
        mapped_labels (np.ndarray): target_labels remapped to reference label space.
    """
    mapped = np.zeros_like(target_labels)
    for cluster_id in range(n_clusters):
        # find indices where target_labels == cluster_id
        indices = np.where(target_labels == cluster_id)[0]
        if len(indices) == 0:
            continue
        # get corresponding reference labels and choose most common
        ref_sub = reference_labels[indices]
        most_common = Counter(ref_sub).most_common(1)[0][0]
        mapped[indices] = most_common
    return mapped

def cluster_and_vote(features, n_clusters=50, reject_threshold=0.6):
    """
    Perform multi-algorithm clustering and majority vote.
    Args:
        features (np.ndarray): shape (N, D) array of feature vectors.
        n_clusters (int): number of clusters for each algorithm.
        reject_threshold (float): minimal vote fraction to accept label; otherwise -1.
    Returns:
        final_labels (np.ndarray): length-N array of cluster labels or -1 for rejected.
    """
    # Initialize clusterers
    clusterers = {
        'kmeans': KMeans(n_clusters=n_clusters, random_state=42),
        'birch': Birch(n_clusters=n_clusters),
        'agg': AgglomerativeClustering(n_clusters=n_clusters)
    }
    # Fit and predict
    preds = {name: clf.fit_predict(features) for name, clf in clusterers.items()}

    # Align labels: use kmeans as reference
    ref = preds['kmeans']
    aligned_preds = {'kmeans': ref}
    for name in ['birch', 'agg']:
        aligned_preds[name] = align_labels(ref, preds[name], n_clusters)

    # Voting
    final_labels = np.full(features.shape[0], -1, dtype=int)
    for i in range(features.shape[0]):
        votes = [aligned_preds[name][i] for name in ['kmeans', 'birch', 'agg']]
        vote_count = Counter(votes)
        label, count = vote_count.most_common(1)[0]
        if count / len(votes) >= reject_threshold:
            final_labels[i] = label
    return final_labels