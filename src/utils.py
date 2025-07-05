from tqdm import tqdm
import torch
import numpy as np

def extract_features(model, dataloader, device, save_path_cv, save_path_vit):
    """
    Trích xuất feature từ model và lưu vào file .npy chuẩn bằng np.save().
    Sử dụng torch.cat để gom các batch, đảm bảo không lỗi khi load lại.
    """
    model.eval()
    features_cv = []
    features_vit = []

    with torch.no_grad():
        for x_cv, x_vit in tqdm(dataloader, desc="Extracting features"):
            x_cv = x_cv.to(device)
            x_vit = x_vit.to(device)

            h_cv, h_vit = model.encode(x_cv, x_vit)  # [B, 512]

            features_cv.append(h_cv.cpu())
            features_vit.append(h_vit.cpu())

    # Gộp toàn bộ feature lại thành tensor lớn
    features_cv = torch.cat(features_cv, dim=0).numpy()
    features_vit = torch.cat(features_vit, dim=0).numpy()

    # Lưu ra file .npy chuẩn
    np.save(save_path_cv, features_cv)
    np.save(save_path_vit, features_vit)

    print(f"✅ Saved features to: {save_path_cv}, {save_path_vit}")

def save_checkpoint(model, optimizer, epoch, path):
    save_path = path.format(epoch)
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }, save_path)
    print(f"Saved checkpoint to {save_path}")

def load_checkpoint(model, optimizer, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resumed from checkpoint: {path}, starting at epoch {start_epoch}")
    return start_epoch