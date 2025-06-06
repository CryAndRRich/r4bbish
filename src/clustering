from .utils_cluster import extract_features, cluster_and_vote
from config import Config
from contrastive_model import DualEncoderContrastive
from datasets import build_dataloader
from utils import load_checkpoint
import torch
import os
import pandas as pd
import numpy as np

def main():
    os.makedirs("results", exist_ok=True)
    device = Config.DEVICE

    model = DualEncoderContrastive().to(device)
    ckpt = os.path.join(Config.CHECKPOINT_DIR, "checkpoint_epoch_200.pth")
    load_checkpoint(model, torch.optim.AdamW(model.parameters()), ckpt, device)

    dataloader = build_dataloader(Config.VAL_DIR, Config.BATCH_SIZE, Config.NUM_WORKERS, shuffle=False)
    features = extract_features(model, dataloader, device)
    np.save("results/features.npy", features)

    labels = cluster_and_vote(features, n_clusters=4, reject_threshold=0.6)
    pd.DataFrame({{"sample": range(len(labels)), "label": labels}}).to_csv("results/cluster_labels.csv", index=False)

if __name__ == "__main__":
    main()
