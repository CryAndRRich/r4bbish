import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def t_sne(array_path: str, 
          perplexity: int =30, 
          random_state: int =42):
    """
    Load an array from a .npy file, reduce it to 2D using t-SNE, and plot a scatter chart

    Parameters:
        array_path: Path to the .npy file containing the input array
        perplexity: Perplexity parameter for t-SNE
        random_state: Seed for random number generation
    """
    data = np.load(array_path)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    reduced = tsne.fit_transform(data)

    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], s=10, alpha=0.7)
    plt.title("t-SNE Visualization")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
