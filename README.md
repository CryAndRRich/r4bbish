# rubbish4
```
rubbish4/
├── data/                      
│   └── trashnet/              # Thư mục chứa ảnh gốc (raw images)
│
├── src/                       # Mã nguồn chính
│   ├── __init__.py
│   ├── encoders.py            # Định nghĩa mô hình Dual-Encoder Contrastive + Loss
│   ├── dataset.py             # DataLoader/PyTorch Dataset để load ảnh
│   ├── utils.py               # Các hàm tiện ích chung (extract, checkpoint, v.v.)
│   ├── config.py              # Cấu hình hyperparameters, đường dẫn, thiết lập chung
|   ├── multi_cluster.py       # Chạy các mô hình phân cụm (Kmeans, BIRCH, Agg)
|   |
|   └── visualize/             # Các hàm vẽ các biểu đồ minh hoạ
|
├── checkpoints/               # Lưu model checkpoints (.pth) trong quá trình train
│
├── test.py                    # Script chạy toàn bộ pipeline
├── requirements.txt           # Các thư viện Python cần cài (timm, torch, torchvision, ...)
├── README.md                  # Giới thiệu tổng quan về dự án, cách chạy
└── .gitignore
```
