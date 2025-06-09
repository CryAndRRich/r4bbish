import os
import numpy as np
from PIL import Image

def export_cluster_images(dataset, labels, output_dir="results/clusters"):
    os.makedirs(output_dir, exist_ok=True)
    for cluster_id in np.unique(labels):
        if cluster_id == -1:
            continue
        indices = np.where(labels == cluster_id)[0]
        for i, idx in enumerate(indices[:5]):  # Lấy 5 hình ảnh đầu tiên mỗi cụm
            img, _, path = dataset[idx]
            img = Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
            img.save(os.path.join(output_dir, f"cluster_{cluster_id}_img_{i}.png"))
