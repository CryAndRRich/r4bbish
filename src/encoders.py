import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class EncoderWrapper(nn.Module):
    """
    Wrapper for vision encoders from timm (e.g., ResNet, ViT).
    
    This class removes the classification head and optionally adds a linear layer
    to match the desired output dimension
    """
    def __init__(self, 
                 encoder_name: str, 
                 output_dim: int = 2048) -> None:
        """
        Parameters:
            encoder_name: Name of the encoder model
            output_dim: Desired output feature dimension
        """
        super(EncoderWrapper, self).__init__()
        self.encoder_name = encoder_name
        self.encoder = timm.create_model(encoder_name, pretrained=True, num_classes=0)

        # Infer encoder output dimension using a dummy input
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = self.encoder(dummy_input)
            encoder_out_dim = out.shape[1]

        # Add linear projection if encoder output doesn't match target output_dim
        if encoder_out_dim != output_dim:
            self.output_mapping = nn.Linear(encoder_out_dim, output_dim)
        else:
            self.output_mapping = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            x: Input image tensor of shape (B, 3, 224, 224)

        Returns:
            torch.Tensor: Encoded feature vector of shape (B, output_dim)
        """
        x = self.encoder(x)
        x = self.output_mapping(x)
        return x


class ProjectionHead(nn.Module):
    """
    MLP projection head: Maps embeddings from input_dim to projection_dim.
    
    Typically used in contrastive learning to map representations to a space
    where similarity is computed (e.g., for NT-Xent loss)
    """
    def __init__(self, 
                 input_dim: int = 2048, 
                 projection_dim: int = 128) -> None:
        """
        Parameters:
            input_dim: Input feature dimension.
            projection_dim: Output projection dimension.
        """
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, projection_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            x: Input feature tensor of shape (B, input_dim)

        Returns:
            torch.Tensor: L2-normalized projected features (B, projection_dim)
        """
        x = self.net(x)
        return F.normalize(x, dim=1)  # L2 normalization for cosine similarity


class DualEncoderContrastive(nn.Module):
    """
    Dual-encoder contrastive learning model using ResNet and ViT encoders.
    
    Takes augmented views of the same image, encodes them using different encoders,
    projects them using MLPs, and computes NT-Xent loss between paired features
    """

    def __init__(self,
                 encoder_name_res: str,
                 encoder_name_vit: str,
                 output_dim: int = 2048,
                 projection_dim: int = 128,
                 temperature: float = 0.2) -> None:
        """
        Parameters:
            encoder_name_res: Name of CNN-based encoder (e.g., ResNet)
            encoder_name_vit: Name of ViT-based encoder
            output_dim: Output feature dimension of each encoder
            projection_dim: Output dimension of the projection head
            temperature: Temperature parameter for NT-Xent loss
        """
        super(DualEncoderContrastive, self).__init__()
        self.temperature = temperature

        self.encoder_res = EncoderWrapper(encoder_name_res, output_dim=output_dim)
        self.encoder_vit = EncoderWrapper(encoder_name_vit, output_dim=output_dim)

        self.proj_res = ProjectionHead(input_dim=output_dim, projection_dim=projection_dim)
        self.proj_vit = ProjectionHead(input_dim=output_dim, projection_dim=projection_dim)

    def forward(self, 
                x_res: torch.Tensor, 
                x_vit: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for contrastive training

        Parameters:
            x_res: Input batch for CNN encoder (B, 3, 224, 224)
            x_vit: Input batch for ViT encoder (B, 3, 224, 224)

        Returns:
            torch.Tensor: Scalar NT-Xent loss between encoded representations
        """
        h_res = self.encoder_res(x_res)    # (B, output_dim)
        h_vit = self.encoder_vit(x_vit)    # (B, output_dim)

        z_res = self.proj_res(h_res)       # (B, projection_dim)
        z_vit = self.proj_vit(h_vit)       # (B, projection_dim)

        return self.nt_xent_loss(z_res, z_vit)

    def nt_xent_loss(self, 
                     z_i: torch.Tensor, 
                     z_j: torch.Tensor) -> torch.Tensor:
        """
        Computes the normalized temperature-scaled cross entropy loss (NT-Xent)

        Parameters:
            z_i: Projected features from encoder A, shape (B, D)
            z_j: Projected features from encoder B, shape (B, D)

        Returns:
            torch.Tensor: Scalar contrastive loss
        """
        batch_size = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)                  # (2B, D)
        sim = torch.matmul(z, z.T) / self.temperature     # (2B, 2B)

        # Mask self-similarity
        mask = (~torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)).float()
        sim = sim * mask

        # Positive pairs: (i, i+B) and (i+B, i)
        pos_sim = torch.cat([
            torch.diag(sim, batch_size),
            torch.diag(sim, -batch_size)
        ], dim=0)  # (2B,)

        numerator = torch.exp(pos_sim)
        denominator = torch.exp(sim).sum(dim=1)

        loss = -torch.log(numerator / denominator)
        return loss.mean()

    def encode(self, 
               x_res: torch.Tensor, 
               x_vit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts feature embeddings (without projection) from both encoders

        Parameters:
            x_res: Batch of images for CNN encoder (B, 3, 224, 224)
            x_vit: Batch of images for ViT encoder (B, 3, 224, 224)

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple of feature embeddings (B, output_dim) from both encoders
        """
        with torch.no_grad():
            h_res = self.encoder_res(x_res)
            h_vit = self.encoder_vit(x_vit)
        return h_res, h_vit
