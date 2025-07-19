import os
import shutil
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T

class WasteDataset(Dataset):
    """
    Unlabeled dataset for image preprocessing and standardized loading
    
    This dataset automatically scans a given root directory (recursively), collects all valid image files,
    renames them sequentially, and stores them in a clean output folder. Each image is preprocessed with
    transforms suitable for modern vision models like ResNet and ViT (224x224 input size)
    
    All collected images are transformed using a default torchvision pipeline unless a custom transform is provided
    """

    def __init__(self, 
                 root_dir: str, 
                 output_folder: str = 'data', 
                 transform=None) -> None:
        """
        Parameters:
            root_dir: Root directory containing raw image files (possibly inside subdirectories)
            output_folder: Directory where collected and renamed images will be saved
            transform: A torchvision transform function to apply to each image.
                       Defaults to a standard resize + center crop + normalization
                       (224x224) suitable for ViT/ConvNeXt
        """
        self.root_dir = root_dir
        self.output_folder = output_folder

        # Collect and rename images
        self._collect_images()

        # Store full paths of collected images
        self.paths = [os.path.join(output_folder, f)
                      for f in sorted(os.listdir(output_folder))
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        # Set image transformation
        self.transform = transform or T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def _collect_images(self) -> None:
        """
        Recursively scan the root directory, validate and copy all image files to the output folder.
        Images are renamed sequentially (e.g., 1.jpg, 2.jpg, ...). Invalid or corrupted images are skipped
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
                            img.verify()  # Check for corrupt image
                        new_name = f"{count}.jpg"
                        dst_path = os.path.join(self.output_folder, new_name)
                        shutil.copy(src_path, dst_path)
                        count += 1
                    except Exception as e:
                        print(f"Skipped corrupted image: {src_path} ({e})")

        print(f"Collected {count - 1} images into '{self.output_folder}'.")

    def __len__(self) -> int:
        """
        Return the total number of collected images
        """
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Load and return the transformed image at the given index

        Parameters:
            idx: Index of the image to retrieve

        Returns:
            tuple: The image is returned twice (e.g., for contrastive/self-supervised tasks)
        """
        img_path = self.paths[idx]
        img = Image.open(img_path).convert('RGB')
        img_transformed = self.transform(img)

        return img_transformed, img_transformed
