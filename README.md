rubbish4/
├── data/                      
│   └── trashnet/              # Directory containing raw waste images
│
├── src/                       # Main source code
│   ├── visualize/             # Functions for visualizing embeddings, clusters, etc.
│   │
│   ├── encoders.py            # Defines the Dual-Encoder contrastive model and loss functions
│   ├── dataset.py             # PyTorch Dataset and DataLoader for loading waste images
│   ├── utils.py               # Utility functions (feature extraction, checkpoint saving, etc.)
│   ├── config.py              # Configuration for hyperparameters, paths, and global settings
│   └── multi_cluster.py       # Scripts for running clustering algorithms (KMeans, BIRCH, Agglomerative)
│
├── checkpoints/               # Stores model checkpoints (.pth) during training
│
├── test.py                    # Script to run the full unsupervised classification pipeline
├── requirements.txt           # Required Python packages (timm, torch, torchvision, etc.)
├── README.md                  # Project overview and usage instructions
└── .gitignore
