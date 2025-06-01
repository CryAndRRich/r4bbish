import os
import torch
import logging
from datetime import datetime

def setup_logging(log_dir: str):
    """
    Thiết lập logging để ghi vào file và in ra console.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

def save_checkpoint(state: dict, checkpoint_dir: str, epoch: int):
    """
    Lưu checkpoint (model state + optimizer state) ở epoch nhất định.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filename = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save(state, filename)
    return filename

def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                    checkpoint_path: str, device: str):
    """
    Load model + optimizer state từ file checkpoint_path. 
    Trả về epoch bắt đầu (nếu ghi trong checkpoint), hoặc 0 nếu không có.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    start_epoch = checkpoint.get('epoch', 0) + 1
    return start_epoch

def adjust_learning_rate(optimizer: torch.optim.Optimizer, epoch: int, 
                         init_lr: float, schedule: dict):
    """
    Giảm learning rate theo schedule (nếu cần). Ví dụ: schedule = {100:0.1, 150:0.01}
    Nếu epoch >= milestone, lr = init_lr * scale.
    """
    lr = init_lr
    for milestone, scale in schedule.items():
        if epoch >= milestone:
            lr = init_lr * scale
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
