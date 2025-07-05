import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class EncoderWrapper(nn.Module):
    """
    Bọc model từ timm (ConvNeXt, ViT...), loại bỏ head phân loại, thêm Linear nếu cần để match output_dim.
    """
    def __init__(self, encoder_name: str, output_dim: int = 2048):
        super(EncoderWrapper, self).__init__()
        self.encoder_name = encoder_name
        self.encoder = timm.create_model(encoder_name, pretrained=True, num_classes=0)

        # Xác định kích thước đầu ra từ dummy input
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = self.encoder(dummy_input)
            encoder_out_dim = out.shape[1]

        # Nếu cần mapping để match output_dim
        if encoder_out_dim != output_dim:
            self.output_mapping = nn.Linear(encoder_out_dim, output_dim)
        else:
            self.output_mapping = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.output_mapping(x)
        return x


class ProjectionHead(nn.Module):
    """
    MLP projection head: Biến embedding 2048 → projection_dim (default: 128)
    """
    def __init__(self, input_dim: int = 2048, projection_dim: int = 128):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, projection_dim)
        )

    def forward(self, x):
        x = self.net(x)
        return F.normalize(x, dim=1)  # L2-normalize để dùng cosine similarity


class DualEncoderContrastive(nn.Module):
    """
    Mô hình Contrastive Learning với 2 encoder (ConvNeXt & ViT).
    Trả về NT-Xent loss giữa các cặp ảnh augment.
    """

    def __init__(self,
                 encoder_name_cv: str,
                 encoder_name_vit: str,
                 output_dim: int = 2048,
                 projection_dim: int = 128,
                 temperature: float = 0.2):
        super(DualEncoderContrastive, self).__init__()
        self.temperature = temperature

        self.encoder_cv = EncoderWrapper(encoder_name_cv, output_dim=output_dim)
        self.encoder_vit = EncoderWrapper(encoder_name_vit, output_dim=output_dim)

        self.proj_cv = ProjectionHead(input_dim=output_dim, projection_dim=projection_dim)
        self.proj_vit = ProjectionHead(input_dim=output_dim, projection_dim=projection_dim)

    def forward(self, x_cv, x_vit):
        h_cv = self.encoder_cv(x_cv)    # [B, 2048]
        h_vit = self.encoder_vit(x_vit) # [B, 2048]

        z_cv = self.proj_cv(h_cv)       # [B, 128]
        z_vit = self.proj_vit(h_vit)    # [B, 128]

        return self.nt_xent_loss(z_cv, z_vit)

    def nt_xent_loss(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        batch_size = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)                     # [2B, D]
        sim = torch.matmul(z, z.T) / self.temperature        # [2B, 2B]

        # loại bỏ chính nó
        mask = (~torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)).float()
        sim = sim * mask

        # Positive pairs: (i, i+B) và (i+B, i)
        pos_sim = torch.cat([torch.diag(sim, batch_size), torch.diag(sim, -batch_size)], dim=0)  # [2B]
        numerator = torch.exp(pos_sim)

        exp_sim = torch.exp(sim)
        denominator = exp_sim.sum(dim=1)

        loss = -torch.log(numerator / denominator)
        return loss.mean()

    def encode(self, x_cv, x_vit):
        """
        Lấy feature 2048-d từ 2 encoder, dùng cho downstream tasks.
        """
        with torch.no_grad():
            h_cv = self.encoder_cv(x_cv)
            h_vit = self.encoder_vit(x_vit)
        return h_cv, h_vit