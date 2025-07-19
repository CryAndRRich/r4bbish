import torch 

class CONFIG:
    ROOT_FOLDER = "data/trashnet"
    DATA_FOLDER = "data/data"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RESNET_ENCODER = "resnet50"
    VIT_ENCODER = "vit_tiny_patch16_224"

    OUTPUT_DIM = 512
    PROJECTION_DIM = 64
    TEMPERATURE = 0.2
    ENCODER_LEARN_RATE = 0.001
    ENCODER_WEIGHT_DECAY = 0.00001
    ENCODER_EPOCHS = 200

    FEATURE_OUT_RES = "checkpoints/features_resnet.npy"
    FEATURE_OUT_VIT = "checkpoints/features_vit.npy"
    CHECKPOINT_ENCODER = "checkpoints/checkpoint_resnet_vit_100.pth"

    NUMBER_OF_CLUSTERS = 50
    REJECTION_THRESHOLD = 0.6