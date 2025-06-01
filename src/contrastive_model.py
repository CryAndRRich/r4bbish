import torch
import torch.nn as nn
import torch.nn.functional as F
from encoders import EncoderWrapper
from config import Config

class ProjectionHead(nn.Module):
    """
    MLP projection head dùng để biến đổi feature embedding (2048-d) 
    thành không gian thấp hơn (projection_dim), chuẩn hóa để tính contrastive loss.
    """
    def __init__(self, input_dim: int = Config.EMBEDDING_DIM, 
                 projection_dim: int = Config.PROJECTION_DIM):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, projection_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape [B, input_dim]
        Trả về: tensor shape [B, projection_dim], normalized
        """
        x = self.net(x)
        x = F.normalize(x, p=2, dim=1)  # Chuẩn hóa L2 để tính cosine similarity
        return x

class DualEncoderContrastive(nn.Module):
    """
    Mô hình dual-encoder contrastive learning:
      - encoder_cv: ConvNeXt → output 2048-d → projection → 128-d
      - encoder_vit: ViT → output 2048-d → projection → 128-d
    Tính NT-Xent loss giữa 2 embedding của cùng 1 sample (positive),
    và denoted các pairs khác trong batch làm negative.
    """

    def __init__(self, 
                 encoder_name_cv: str = "convnext_xlarge_in22k",
                 encoder_name_vit: str = "vit_base_patch16_384",
                 output_dim: int = Config.EMBEDDING_DIM,
                 projection_dim: int = Config.PROJECTION_DIM,
                 temperature: float = Config.TEMPERATURE):
        super(DualEncoderContrastive, self).__init__()
        self.temperature = temperature

        # Khởi tạo 2 encoder
        self.encoder_cv = EncoderWrapper(encoder_name_cv, output_dim=output_dim)
        self.encoder_vit = EncoderWrapper(encoder_name_vit, output_dim=output_dim)

        # Projection heads
        self.proj_cv = ProjectionHead(input_dim=output_dim, projection_dim=projection_dim)
        self.proj_vit = ProjectionHead(input_dim=output_dim, projection_dim=projection_dim)

    def forward(self, x_cv: torch.Tensor, x_vit: torch.Tensor):
        """
        x_cv: ảnh resized 224x224 → đi vào encoder_cv
        x_vit: ảnh resized 384x384 → đi vào encoder_vit

        Trả về loss contrastive.
        """
        # 1. Lấy feature embedding 2048-d
        h_cv = self.encoder_cv(x_cv)    # [B, 2048]
        h_vit = self.encoder_vit(x_vit) # [B, 2048]

        # 2. Projection về không gian nhỏ hơn và normalize
        z_cv = self.proj_cv(h_cv)       # [B, 128]
        z_vit = self.proj_vit(h_vit)    # [B, 128]

        # 3. Tính NT-Xent loss
        loss = self.nt_xent_loss(z_cv, z_vit)
        return loss

    def nt_xent_loss(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Tính Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).
        Cho batch size B, ta có 2B vectors: [z_i_1, ..., z_i_B, z_j_1, ..., z_j_B]
        Các cặp (z_i_k, z_j_k) là positive pairs. Các cặp khác đều negative.
        """
        batch_size = z_i.shape[0]
        z = torch.cat([z_i, z_j], dim=0)  # [2B, D]
        # Ma trận similarity (cosine) giữa tất cả 2B vectors
        sim = torch.matmul(z, z.T)        # [2B, 2B], vì đã normalize nên là cosine similarity

        # Tạo mask để loại bỏ similarity giữa cùng chính nó
        mask = (~torch.eye(2 * batch_size, dtype=torch.bool, device=sim.device)).float()

        # Scale by temperature
        sim = sim / self.temperature

        # Mỗi cặp positive nằm ở vị trí (i, i+B) và (i+B, i)
        positives = torch.cat([torch.diag(sim, batch_size), torch.diag(sim, -batch_size)], dim=0)  # [2B]
        # Collect tất cả negative similarity (loại bỏ diagonal và cặp positive)
        nominator = torch.exp(positives)

        # denominator: với mỗi index k, sum exp(sim[k, all except k])
        exp_sim = torch.exp(sim) * mask   # đặt exp_sim[k,k] = 0
        denominator = exp_sim.sum(dim=1)  # [2B]

        loss = -torch.log(nominator / denominator)
        loss = loss.mean()
        return loss

    def encode(self, x_cv: torch.Tensor, x_vit: torch.Tensor):
        """
        Hàm để lấy feature embedding (2048-d) từ hai encoder mà không cần projection.
        Dùng khi muốn extract features sau training contrastive.
        """
        with torch.no_grad():
            h_cv = self.encoder_cv(x_cv)    # [B, 2048]
            h_vit = self.encoder_vit(x_vit) # [B, 2048]
        return h_cv, h_vit
