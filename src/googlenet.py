import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.metrics import classification_report

# Dataset đọc ảnh và label
class LabeledImageDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.filenames = dataframe['filename'].tolist()
        self.labels = dataframe['label'].tolist()
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.filenames[idx])
        image = Image.open(img_path).convert('RGB')
        label = int(self.labels[idx])
        if self.transform:
            image = self.transform(image)
        return image, label

# ✅ Huấn luyện GoogLeNet
def train(model, train_loader, test_loader, device, num_classes, epochs, lr, checkpoint_path):
    class_weights = torch.ones(num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # ✅ Đánh giá
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        print(f"\n📉 Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}")
        print(classification_report(
            all_labels,
            all_preds,
            labels=list(range(num_classes)),
            zero_division=0
        ))

    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved model to {checkpoint_path}")