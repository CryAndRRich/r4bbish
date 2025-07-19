from tqdm import tqdm
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader

def extract_features(model: nn.Module,
                     dataloader: DataLoader,
                     device: torch.device,
                     save_path_res: str,
                     save_path_vit: str) -> None:
    """
    Extract features from the model and save them to .npy files using np.save()
    
    This function runs inference in evaluation mode using the model"s "encode" method,
    and collects the features from two encoders (e.g., ResNet and ViT). The features
    from all batches are concatenated using torch.cat for efficient saving

    Args:
        model: The model with ".encode(x_res, x_vit)" method returning features
        dataloader: PyTorch DataLoader yielding pairs (x_res, x_vit)
        device: Device to run the model on
        save_path_res: Path to save ResNet (or encoder A) features
        save_path_vit: Path to save ViT (or encoder B) features

    Returns:
        None
    """
    model.eval()
    features_res = []  # List of feature tensors from encoder A
    features_vit = []  # List of feature tensors from encoder B

    with torch.no_grad():
        for x_res, x_vit in tqdm(dataloader, desc="Extracting features"):
            x_res = x_res.to(device)  # (B, 3, 224, 224)
            x_vit = x_vit.to(device)  # (B, 3, 224, 224)

            h_res, h_vit = model.encode(x_res, x_vit)  # Tuple of (B, D) features

            features_res.append(h_res.cpu())
            features_vit.append(h_vit.cpu())

    # Concatenate all batches into full arrays
    features_res_np = torch.cat(features_res, dim=0).numpy()  # (N, D)
    features_vit_np = torch.cat(features_vit, dim=0).numpy()  # (N, D)

    # Save to disk as .npy files
    np.save(save_path_res, features_res_np)
    np.save(save_path_vit, features_vit_np)

    print(f"Saved features to: {save_path_res}, {save_path_vit}")


def save_checkpoint(model: nn.Module,
                    optimizer: optim.Optimizer,
                    epoch: int,
                    path: str) -> None:
    """
    Save model and optimizer state to a checkpoint file

    Args:
        model: The model to save
        optimizer: The optimizer used during training
        epoch: Current epoch number
        path: Path template with placeholder for epoch (e.g., "ckpt_epoch_{}.pt")

    Returns:
        None
    """
    save_path = path.format(epoch)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, save_path)
    print(f" Saved checkpoint to {save_path}")


def load_checkpoint(model: nn.Module,
                    optimizer: optim.Optimizer,
                    path: str,
                    device: torch.device) -> int:
    """
    Load model and optimizer state from a checkpoint file

    Args:
        model: The model to load weights into
        optimizer: The optimizer to resume from
        path: Path to the checkpoint file
        device: Device to load the checkpoint onto

    Returns:
        int: The epoch to resume from (one past the loaded epoch).
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"Resumed from checkpoint: {path}, starting at epoch {start_epoch}")
    return start_epoch
