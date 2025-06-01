import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class WasteImageDataset(Dataset):
    """
    PyTorch Dataset để load tập ảnh (không nhãn) cho unsupervised contrastive learning.
    Mỗi sample gồm 1 ảnh, được crop/resize về 2 size:
      - X_cv: 224x224 để đưa vào ConvNeXt
      - X_vit: 384x384 để đưa vào ViT
    """

    def __init__(self, img_dir: str, 
                 transform_cv=None, transform_vit=None):
        """
        img_dir: thư mục chứa tất cả ảnh (không phân chia train/val ở đây, có thể chia ngoài).
        transform_cv: torchvision transforms dùng cho ConvNeXt branch
        transform_vit: torchvision transforms dùng cho ViT branch
        """

        super(WasteImageDataset, self).__init__()
        self.img_dir = img_dir
        # Lấy danh sách tất cả đường dẫn ảnh trong thư mục con
        self.paths = []
        for root, _, files in os.walk(img_dir):
            for fname in files:
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.paths.append(os.path.join(root, fname))
        self.paths.sort()
        
        # Nếu người dùng không truyền transform, đặt default
        # ConvNeXt: resize ảnh về side=256 rồi center crop 224
        if transform_cv is None:
            self.transform_cv = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform_cv = transform_cv

        # ViT: resize ảnh về side=384 rồi center crop 384 (có thể chỉ resize)
        if transform_vit is None:
            self.transform_vit = T.Compose([
                T.Resize(384),
                T.CenterCrop(384),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform_vit = transform_vit

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        """
        Trả về 2 ảnh (đã transform) dùng cho 2 encoder:
        X_cv: cho ConvNeXt (224x224)
        X_vit: cho ViT (384x384)
        """
        img_path = self.paths[idx]
        img = Image.open(img_path).convert('RGB')
        
        x_cv = self.transform_cv(img)
        x_vit = self.transform_vit(img)

        return x_cv, x_vit  # Không có nhãn

def build_dataloader(img_dir: str, batch_size: int, num_workers: int, shuffle=True):
    """
    Tạo dataloader cho tập ảnh không nhãn.
    """
    dataset = WasteImageDataset(img_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, pin_memory=True)
    return loader
