import os
import shutil
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class WasteDataset(Dataset):
    """
    Dataset không nhãn, tự động thu thập ảnh từ thư mục gốc, 
    chuẩn hóa và cung cấp 2 transform: ConvNeXt (224x224) và ViT (224x224).
    """

    def __init__(self, root_dir: str, output_folder: str = 'data',
                 transform=None):
        """
        Args:
            root_dir (str): Thư mục gốc chứa ảnh (có thể có thư mục con).
            output_folder (str): Nơi lưu ảnh đã đổi tên theo thứ tự.
            transform (callable): Transform ảnh 224x224.
        """
        self.root_dir = root_dir
        self.output_folder = output_folder

        # Bước 1: Duyệt và thu thập ảnh
        self._collect_images()

        # Bước 2: Danh sách ảnh sau khi thu thập
        self.paths = [os.path.join(output_folder, f) 
                      for f in sorted(os.listdir(output_folder)) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        # Bước 3: Thiết lập transform
        self.transform = transform or T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def _collect_images(self):
        """
        Duyệt root_dir, thu thập tất cả ảnh vào output_folder, 
        đổi tên theo thứ tự: 1.jpg, 2.jpg, ...
        """
        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')

        if os.path.exists(self.output_folder):
            shutil.rmtree(self.output_folder)
        os.makedirs(self.output_folder)

        count = 1
        for dirpath, _, filenames in os.walk(self.root_dir):
            for fname in filenames:
                if fname.lower().endswith(valid_ext):
                    src_path = os.path.join(dirpath, fname)
                    try:
                        with Image.open(src_path) as img:
                            img.verify()
                        new_name = f"{count}.jpg"
                        dst_path = os.path.join(self.output_folder, new_name)
                        shutil.copy(src_path, dst_path)
                        count += 1
                    except Exception as e:
                        print(f"Bỏ qua ảnh lỗi: {src_path} ({e})")

        print(f"Đã thu thập {count - 1} ảnh vào '{self.output_folder}'")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img = Image.open(img_path).convert('RGB')
        img_transformed = self.transform(img)

        return img_transformed, img_transformed
