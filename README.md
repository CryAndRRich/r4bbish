# rubbish4

```
rubbish4/
├── data/                      
│   ├── images/                # Thư mục chứa ảnh gốc (raw images) hoặc link đến bộ dữ liệu
│   │   ├── train/             # Ảnh dùng để train contrastive (chưa có nhãn)
│   │   └── val/               # Ảnh để validation (có thể dùng để giám sát loss)
│   └── README.md              # Hướng dẫn ngắn cách chuẩn bị dữ liệu
│
├── src/                       # Mã nguồn chính
│   ├── __init__.py
│   ├── encoders.py            # Định nghĩa EncoderWrapper (đã có sẵn, chỉ cần bổ sung nếu cần)
│   ├── datasets.py            # DataLoader/PyTorch Dataset để load ảnh và áp dụng Augmentations
│   ├── contrastive_model.py   # Định nghĩa mô hình Dual-Encoder Contrastive + Loss
│   ├── train_contrastive.py   # Script chạy training contrastive learning
│   ├── utils.py               # Các hàm tiện ích chung (logging, checkpoint, v.v.)
│   └── config.py              # Cấu hình hyperparameters, đường dẫn, thiết lập chung
│
├── experiments/               # Thư mục chứa logs, checkpoint sau khi train
│   ├── logs/
│   │   ├── train.log
│   │   └── tensorboard/       # Nếu dùng TensorBoard
│   └── checkpoints/           # Lưu model checkpoints (.pth) trong quá trình train
│
├── requirements.txt           # Các thư viện Python cần cài (timm, torch, torchvision, ...)
├── README.md                  # Giới thiệu tổng quan về dự án, cách chạy
└── .gitignore
DATASET
https://www.kaggle.com/code/dhiazoghlami/waste-classification-model
https://www.kaggle.com/datasets/alveddian/waste-dataset
```
