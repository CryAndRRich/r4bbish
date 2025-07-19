import numpy as np
import matplotlib.pyplot as plt

def plot_heatmap(array_path: str, 
                 cmap: str = "viridis") -> None:
    """
    Load a 2D array from a .npy file and display it as a heatmap using matplotlib

    Parameters:
        array_path: Path to the .npy file containing a 2D array
        cmap: Matplotlib colormap, e.g., "viridis", "hot", "coolwarm"
    """
    data = np.load(array_path)

    if data.ndim != 2:
        raise ValueError("Input data must be a 2D array to plot a heatmap.")

    plt.figure(figsize=(8, 6))
    plt.imshow(data, cmap=cmap, aspect="auto")
    plt.colorbar()
    plt.title("Heatmap")
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.tight_layout()
    plt.show()
