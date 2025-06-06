import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np

def visualize(path="results/cluster_labels.csv", feature_path="results/features.npy"):
    df = pd.read_csv(path)
    labels = df["label"].values
    features = np.load(feature_path)

    print("📈 Đang chạy t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    embedded = tsne.fit_transform(features)

    print("🖼 Vẽ biểu đồ...")
    plt.figure(figsize=(10, 8))
    for lbl in np.unique(labels):
        idx = labels == lbl
        plt.scatter(embedded[idx, 0], embedded[idx, 1], label=f"Cluster {lbl}" if lbl != -1 else "Reject", alpha=0.6)

    plt.legend()
    plt.title("t-SNE Clustering Visualization")
    plt.savefig("results/cluster_plot.png")
    plt.show()

if __name__ == "__main__":
    visualize()
