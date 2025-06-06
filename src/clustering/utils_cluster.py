import torch
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans, Birch, AgglomerativeClustering

def extract_features(model, dataloader, device):
    model.eval()
    all_feats = []
    with torch.no_grad():
        for x_cv, x_vit in dataloader:
            x_cv = x_cv.to(device)
            x_vit = x_vit.to(device)
            h_cv, h_vit = model.encode(x_cv, x_vit)
            combined = torch.cat([h_cv, h_vit], dim=1)
            all_feats.append(combined.cpu())
    return torch.cat(all_feats, dim=0).numpy()

def cluster_and_vote(features, n_clusters=4, reject_threshold=0.6):
    clusterers = {{
        'kmeans': KMeans(n_clusters=n_clusters, random_state=42),
        'birch': Birch(n_clusters=n_clusters),
        'agg': AgglomerativeClustering(n_clusters=n_clusters)
    }}
    preds = {{k: m.fit_predict(features) for k, m in clusterers.items()}}

    final_labels = []
    for i in range(len(features)):
        votes = [preds[k][i] for k in preds]
        c = Counter(votes)
        val, freq = c.most_common(1)[0]
        final_labels.append(val if freq / len(votes) >= reject_threshold else -1)
    return final_labels
