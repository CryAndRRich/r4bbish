import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from config import Config
from datasets import build_dataloader
from contrastive_model import DualEncoderContrastive
from utils import setup_logging, save_checkpoint, adjust_learning_rate

def main():
    # 1. Ensure các thư mục cần thiết
    Config.ensure_dirs()
    logger = setup_logging(Config.LOG_DIR)
    device = Config.DEVICE

    # 2. Tạo DataLoader
    train_loader = build_dataloader(
        img_dir=Config.TRAIN_DIR,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        shuffle=True
    )

    # 3. Khởi tạo mô hình Dual-Encoder Contrastive
    model = DualEncoderContrastive(
        encoder_name_cv="convnext_xlarge_in22k",
        encoder_name_vit="vit_base_patch16_384",
        output_dim=Config.EMBEDDING_DIM,
        projection_dim=Config.PROJECTION_DIM,
        temperature=Config.TEMPERATURE
    ).to(device)

    # 4. Thiết lập optimizer (có weight decay)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)

    # 5. (Nếu muốn dùng scheduler) Ví dụ: step LR xuống 0.1 lần ở epoch 100 và 150
    # scheduler_config = {100: 0.1, 150: 0.01}
    # start_epoch = 0

    # 6. Khởi tạo biến để load checkpoint nếu cần
    # Nếu muốn resume training, uncomment và chỉ định path file .pth
    # checkpoint_path = "path/to/checkpoint.pth"
    # if os.path.isfile(checkpoint_path):
    #     start_epoch = load_checkpoint(model, optimizer, checkpoint_path, device)
    #     logger.info(f"Resumed training from epoch {start_epoch}")
    # else:
    #     start_epoch = 0

    start_epoch = 0  # Nếu không resume

    # 7. Training loop
    total_steps = len(train_loader)
    logger.info("Start training contrastive model...")
    for epoch in range(start_epoch, Config.NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0

        # (Nếu dùng scheduler thủ công)
        # current_lr = adjust_learning_rate(optimizer, epoch, Config.LEARNING_RATE, scheduler_config)

        for step, (x_cv, x_vit) in enumerate(train_loader):
            # Move data lên GPU/CPU
            x_cv = x_cv.to(device, non_blocking=True)   # [B, 3, 224, 224]
            x_vit = x_vit.to(device, non_blocking=True) # [B, 3, 384, 384]

            # Zero gradients
            optimizer.zero_grad()
            # Forward + compute loss
            loss = model(x_cv, x_vit)
            # Backward
            loss.backward()
            # Update weights
            optimizer.step()

            epoch_loss += loss.item()

            # In log định kỳ
            if (step + 1) % Config.LOG_FREQ == 0:
                avg_loss = epoch_loss / (step + 1)
                logger.info(f"Epoch [{epoch+1}/{Config.NUM_EPOCHS}] Step [{step+1}/{total_steps}] "
                            f"Loss: {avg_loss:.4f}")

        avg_epoch_loss = epoch_loss / total_steps
        logger.info(f"Epoch [{epoch+1}/{Config.NUM_EPOCHS}] Completed. Avg Loss: {avg_epoch_loss:.4f}")

        # Lưu checkpoint theo tần suất SAVE_FREQ
        if (epoch + 1) % Config.SAVE_FREQ == 0 or (epoch + 1) == Config.NUM_EPOCHS:
            checkpoint_state = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict()
            }
            ckpt_path = save_checkpoint(checkpoint_state, Config.CHECKPOINT_DIR, epoch + 1)
            logger.info(f"Saved checkpoint: {ckpt_path}")

    logger.info("Finish training contrastive model.")

if __name__ == "__main__":
    main()
