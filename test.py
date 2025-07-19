import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import CONFIG
from src.dataset import WasteDataset
from src.encoders import DualEncoderContrastive
from src.utils import extract_features, save_checkpoint, load_checkpoint
from src.multi_cluster import cluster_and_vote

# ============================================================================================
# Step 1: Load dataset from root folder, gather images into DATA_FOLDER, and create Dataset
dataset = WasteDataset(
    root_dir=CONFIG.ROOT_FOLDER,
    output_folder=CONFIG.DATA_FOLDER
)

# Create DataLoader
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Preview one batch
for x_res, x_vit in loader:
    print(x_res.shape, x_vit.shape)
    break

# ============================================================================================
# Step 2: Initialize model and optimizer
model = DualEncoderContrastive(
    encoder_name_res=CONFIG.RES_ENCODER, 
    encoder_name_vit=CONFIG.VIT_ENCODER,
    output_dim=CONFIG.OUTPUT_DIM,
    projection_dim=CONFIG.PROJECTION_DIM,
    temperature=CONFIG.TEMPERATURE
).to(CONFIG.DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CONFIG.ENCODER_LEARN_RATE,
    weight_decay=CONFIG.ENCODER_WEIGHT_DECAY
)

# ============================================================================================
# Step 3: Load checkpoint if available
start_epoch = 0

if os.path.exists(CONFIG.CHECKPOINT_ENCODER):
    start_epoch = load_checkpoint(model, optimizer, CONFIG.CHECKPOINT_ENCODER, CONFIG.DEVICE)
else:
    print("Training from scratch...")

# ============================================================================================
# Step 4: Train the dual encoder model
model.train()
for epoch in range(start_epoch, CONFIG.ENCODER_EPOCHS):
    total_loss = 0

    for step, (x_res, x_vit) in enumerate(loader):
        x_res = x_res.to(CONFIG.DEVICE, non_blocking=True)
        x_vit = x_vit.to(CONFIG.DEVICE, non_blocking=True)

        optimizer.zero_grad()
        loss = model(x_res, x_vit)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch + 1}/{CONFIG.ENCODER_EPOCHS} - Average Loss: {avg_loss:.4f}")

    # Save model after each epoch
    save_checkpoint(model, optimizer, epoch, path="checkpoint_rubbish4.pth")

# ============================================================================================
# Step 5: Extract features from trained model
extract_features(model, loader, CONFIG.DEVICE, CONFIG.FEATURE_OUT_RES , CONFIG.FEATURE_OUT_VIT)

# ============================================================================================
# Step 6: Clustering
features_res = np.load(CONFIG.FEATURE_OUT_RES , allow_pickle=True)
features_vit = np.load(CONFIG.FEATURE_OUT_VIT, allow_pickle=True)

# Concatenate along the feature dimension → (N, 2D)
features = np.concatenate([features_res, features_vit], axis=1)

# Perform clustering with ensemble voting
labels = cluster_and_vote(
    features,
    n_clusters=CONFIG.NUMBER_OF_CLUSTERS,
    reject_threshold=CONFIG.REJECTION_THRESHOLD
)
