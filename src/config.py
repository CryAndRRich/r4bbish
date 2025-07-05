import torch 

class CONFIG:
    ROOT_FOLDER = "/kaggle/input/trashnet/dataset-resized/"
    DATA_FOLDER = "/data"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CV_ENCODER = "convnext_tiny"
    VIT_ENCODER = "vit_tiny_patch16_224"

    OUTPUT_DIM = 512
    PROJECTION_DIM = 64
    TEMPERATURE = 0.2
    ENCODER_LEARN_RATE = 0.001
    ENCODER_WEIGHT_DECAY = 0.00001
    ENCODER_EPOCHS = 200

    FEATURE_OUT_CV = "features_cv.npy"
    FEATURE_OUT_VIT = "features_vit.npy"
    CHECKPOINT_ENCODER = "/kaggle/input/checkpoint/checkpoint_rubbish4.pth"

    NUMBER_OF_CLUSTERS = 50
    REJECTION_THRESHOLD = 0.6

    LABELS_FILE = "final_labels.npy"
    LABELS_CSV_FILE = "pseudo_labels.csv"

    BATCH_SIZE = 32
    NUM_WORKERS = 2
    GOOG_EPOCHS = 50
    GOOG_LEARN_RATE = 0.001
    CHECKPOINT_GOOG = "googlenet.pth"