import matplotlib.pyplot as plt
import numpy as np

def plot_column_chart(datasets: list|str,
                      scores_dict: dict,
                      title: str = "Comparison of Clustering Methods",
                      ylabel: str = "Accuracy (%)",
                      ylim: tuple[int, int] = (70, 102)) -> None:
    """
    Plot a column chart comparing multiple methods across datasets

    Parameters:
        datasets: List of dataset names (X-axis)
        scores_dict: Dictionary with method names as keys and score lists as values
        title: Chart title
        ylabel: Y-axis label
        ylim: Y-axis limits
    """
    x = np.arange(len(datasets))
    width = 0.18
    keys = list(scores_dict.keys())
    colors = ["darkblue", "darkorange", "gray", "gold", "green", "red", "purple"]

    _, ax = plt.subplots(figsize=(10, 7))

    for i, key in enumerate(keys):
        scores = scores_dict[key]
        offset = (i - (len(keys) - 1) / 2) * width * 1.2  # Evenly space columns
        ax.bar(x + offset, scores, width, label=key, color=colors[i % len(colors)])
        for j in range(len(x)):
            ax.text(x[j] + offset, scores[j] + 0.5, f"{scores[j]}%", ha="center", fontsize=9)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(*ylim)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend()
    plt.tight_layout()
    plt.show()
