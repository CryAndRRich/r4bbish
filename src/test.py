import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import googlenet, GoogLeNet_Weights
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from config import CONFIG

from dataset import WasteDataset
from dual_encoder import DualEncoderContrastive
from utils import extract_features, save_checkpoint, load_checkpoint
from multi_cluster import cluster_and_vote
from googlenet import LabeledImageDataset, train

# ============================================================================================

# Tự động gom ảnh từ root_dir, lưu vào data/, tạo dataset
dataset = WasteDataset(root_dir=CONFIG.ROOT_FOLDER, output_folder=CONFIG.DATA_FOLDER)

# Dùng với DataLoader
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Lấy một batch ảnh
for x_cv, x_vit in loader:
    print(x_cv.shape, x_vit.shape)
    break

# ============================================================================================

device = CONFIG.DEVICE

model = DualEncoderContrastive(
    encoder_name_cv=CONFIG.CV_ENCODER,           
    encoder_name_vit=CONFIG.VIT_ENCODER,  
    output_dim=CONFIG.OUTPUT_DIM,
    projection_dim=CONFIG.PROJECTION_DIM,
    temperature=CONFIG.TEMPERATURE
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG.ENCODER_LEARN_RATE, weight_decay=CONFIG.ENCODER_WEIGHT_DECAY)

# ============================================================================================

encoder_epochs = CONFIG.ENCODER_EPOCHS
feature_out_cv = CONFIG.FEATURE_OUT_CV
feature_out_vit = CONFIG.FEATURE_OUT_VIT

checkpoint_encoder_path = CONFIG.CHECKPOINT_ENCODER
start_epoch = 0

# Nếu đã có file checkpoint thì load lại
if os.path.exists(checkpoint_encoder_path):
    start_epoch = load_checkpoint(model, optimizer, checkpoint_encoder_path, device)
else:
    print("Training from scratch")

# Huấn luyện từ epoch hiện tại
model.train()
for epoch in range(start_epoch, encoder_epochs):
    total_loss = 0
    for step, (x_cv, x_vit) in enumerate(loader):
        x_cv = x_cv.to(device, non_blocking=True)
        x_vit = x_vit.to(device, non_blocking=True)

        optimizer.zero_grad()
        loss = model(x_cv, x_vit)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{encoder_epochs} - Avg Loss: {avg_loss:.4f}")

    # Lưu checkpoint sau mỗi epoch
    save_checkpoint(model, optimizer, epoch, path="checkpoint_rubbish4.pth")

# Sau training, extract feature
extract_features(model, loader, device, feature_out_cv, feature_out_vit)

# ============================================================================================

feat_cv = np.load(feature_out_cv, allow_pickle=True)
feat_vit = np.load(feature_out_vit, allow_pickle=True)
# Concatenate along feature dimension
features = np.concatenate([feat_cv, feat_vit], axis=1)


labels = cluster_and_vote(features, n_clusters=CONFIG.NUMBER_OF_CLUSTERS, reject_threshold=CONFIG.REJECTION_THRESHOLD)
output_labels_path = CONFIG.LABELS_FILE
np.save(output_labels_path, labels)
print(f"Saved final cluster labels to {output_labels_path}")

# Sinh danh sách filename tương ứng với ảnh (đã được đổi tên 1.jpg, 2.jpg, ...)
filenames = [f"{i+1}.jpg" for i in range(len(labels))]

# Tạo DataFrame
df = pd.DataFrame({
    "filename": filenames,
    "label": labels
})

# Lưu ra file CSV
csv_path = CONFIG.LABELS_CSV_FILE
df.to_csv(csv_path, index=False)

print(f"Saved CSV: {csv_path}")
print(df.head())

# ============================================================================================

df = pd.read_csv(CONFIG.LABELS_CSV_FILE)
df = df[df['label'] != -1]  # chỉ giữ label 0..49

# 3. Chia train/test 8:2
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# 4. Transform ảnh
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
])

# 5. Dataset & Dataloader
train_dataset = LabeledImageDataset(train_df, CONFIG.DATA_FOLDER, transform=transform)
test_dataset = LabeledImageDataset(test_df, CONFIG.DATA_FOLDER, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=True, num_workers=CONFIG.NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False, num_workers=CONFIG.NUM_WORKERS)

# 6. Model GoogLeNet (với 50 lớp)
weights = GoogLeNet_Weights.DEFAULT
model = googlenet(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 50)
model = model.to(CONFIG.DEVICE)

# 7. Huấn luyện
train(model, train_loader, test_loader, CONFIG.DEVICE, CONFIG.NUMBER_OF_CLUSTERS, 
      CONFIG.GOOG_EPOCHS, CONFIG.GOOG_LEARN_RATE, CONFIG.CHECKPOINT_GOOG)