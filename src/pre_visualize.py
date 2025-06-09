import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE   

def visualize(before_path=None, after_path="results/cluster_labels.csv", feature_path="results/features.npy"):
    features = np.load(feature_path)
    tsne = TSNE(n_components=2, random_state=42)
    
    if before_path:
        plt.figure(figsize=(10, 8))
        embedded = tsne.fit_transform(features)
        plt.scatter(embedded[:, 0], embedded[:, 1], c='gray', alpha=0.6)
        plt.title("t-SNE Before Contrastive Learning")
        plt.savefig("results/before_contrastive_plot.png")
        plt.show()

    df = pd.read_csv(after_path)
    labels = df["label"].values
    embedded = tsne.fit_transform(features)
    plt.figure(figsize=(10, 8))
    for lbl in np.unique(labels):
        idx = labels == lbl
        plt.scatter(embedded[idx, 0], embedded[idx, 1], label=f"Cluster {lbl}" if lbl != -1 else "Reject", alpha=0.6)
    plt.legend()
    plt.title("t-SNE After Clustering")
    plt.savefig("results/after_clustering_plot.png")
    plt.show()
