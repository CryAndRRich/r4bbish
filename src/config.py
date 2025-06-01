import os

class Config:
    # ===== Paths =====
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(ROOT_DIR, 'data', 'images')
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    VAL_DIR = os.path.join(DATA_DIR, 'val')
    CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'experiments', 'checkpoints')
    LOG_DIR = os.path.join(ROOT_DIR, 'experiments', 'logs')
    
    # ===== Hyperparameters =====
    # Kích thước đầu vào cho encoder ConvNeXt và ViT
    IMG_SIZE_CV = 224      # Kích thước ảnh cho ConvNeXt
    IMG_SIZE_VIT = 384     # Kích thước ảnh cho ViT

    EMBEDDING_DIM = 2048   # Kích thước output của mỗi encoder (sau mapping)
    PROJECTION_DIM = 128   # Kích thước embedding cuối cùng dùng để contrastive
    TEMPERATURE = 0.1      # Nhiệt độ cho NT-Xent loss

    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 200

    # Tần suất lưu checkpoint và log
    SAVE_FREQ = 10      # Lưu checkpoint mỗi n epoch
    LOG_FREQ = 100      # In log mỗi n bước train

    # Số workers cho DataLoader
    NUM_WORKERS = 4

    # Thiết lập device
    DEVICE = 'cuda' if (os.environ.get('CUDA_VISIBLE_DEVICES', None) is not None) else 'cpu'

    @staticmethod
    def ensure_dirs():
        """Tạo các thư mục cần thiết nếu chưa có."""
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(Config.LOG_DIR, exist_ok=True)
