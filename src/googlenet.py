from torchvision.models import googlenet
from torch.utils.data import Subset
from sklearn.metrics import classification_report
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config import Config
def train_supervised_model(model, train_loader, test_loader, device, epochs=Config['NUM_EPOCHS'], lr=Config['LEARNING_RATE']):
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 1.0, 5.0]).to(device))  # Xử lý không cân bằng
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels, _ in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        print(f"Epoch {epoch+1}, Loss: {train_loss/len(train_loader):.4f}")
        print(classification_report(all_labels, all_preds, target_names=["recyclable", "residual", "kitchen", "hazardous"]))
    
    torch.save(model.state_dict(), os.path.join(Config.CHECKPOINT_DIR, "googlenet.pth"))

def main():
    os.makedirs("results", exist_ok=True)
    device = Config.DEVICE

    model = DualEncoderContrastive().to(device)
    ckpt = os.path.join(Config.CHECKPOINT_DIR, "checkpoint_epoch_200.pth")
    load_checkpoint(model, torch.optim.AdamW(model.parameters()), ckpt, device)

    dataset = WasteImageDataset(img_dir=Config.VAL_DIR)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)

    # Trích xuất đặc trưng
    features = extract_features(model, dataloader, device)
    np.save("results/features.npy", features)

    # Phân cụm với 50 cụm
    labels = cluster_and_vote(features, n_clusters=50, reject_threshold=0.6)
    pd.DataFrame({"sample": range(len(labels)), "label": labels}).to_csv("results/cluster_labels.csv", index=False)

    
    manual_labels = np.random.randint(0, 4, size=50)  
    final_labels = np.array([manual_labels[label] if label != -1 else -1 for label in labels])

    # Tạo tập huấn luyện và kiểm tra
    train_indices = [i for i in range(len(labels)) if labels[i] != -1]
    test_indices = [i for i in range(len(labels)) if labels[i] == -1]
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
    googlenet_model = googlenet(pretrained=True, num_classes=4).to(device)
    train_supervised_model(googlenet_model, train_loader, test_loader, device)
int __name__ == "__main__":
    main()
