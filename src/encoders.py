import timm
import torch
import torch.nn as nn

class EncoderWrapper(nn.Module):
    """
    Wrapper cho các model encoder từ thư viện timm (ví dụ: convnext, vit, ...)
    Đầu ra của mỗi encoder sẽ được đưa qua một Linear layer mapping sang output_dim.
    Nếu kích thước đầu ra của encoder trùng với output_dim thì dùng Identity.
    """

    def __init__(self, 
                 encoder_name: str, 
                 output_dim: int = 2048) -> None:
        """
        encoder_name: tên model trong timm (ví dụ "convnext_xlarge_in22k", "vit_base_patch16_384", ...)
        output_dim: kích thước embedding mong muốn (mặc định 2048)
        """
        super(EncoderWrapper, self).__init__()
        self.encoder_name = encoder_name
        # Tạo model không có head classification (num_classes=0) để chỉ lấy feature
        self.encoder = timm.create_model(encoder_name, pretrained=True, num_classes=0)

        # Tự động xác định kích thước đầu ra từ encoder
        # Dummy input: nếu "convnext" thì 224×224, else (Ví dụ ViT) 384×384
        dummy_input = torch.randn(1, 3, 224, 224) if "convnext" in encoder_name else torch.randn(1, 3, 384, 384)
        with torch.no_grad():
            out = self.encoder(dummy_input)
            encoder_out_dim = out.shape[1]  # Kích thước feature do encoder xuất ra

        # Nếu output_dim khác encoder_out_dim thì thêm 1 lớp Linear để map về đúng kích thước
        if encoder_out_dim != output_dim:
            self.output_mapping = nn.Linear(encoder_out_dim, output_dim)
        else:
            self.output_mapping = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: tensor shape [B, 3, H, W]
        Trả về: tensor shape [B, output_dim]
        """
        features = self.encoder(x)            # [B, encoder_out_dim]
        output = self.output_mapping(features) # [B, output_dim]
        return output


if __name__ == "__main__":
    # Kiểm tra quick sanity check
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ConvNeXt
    convnext = EncoderWrapper("convnext_xlarge_in22k").to(device)
    dummy_img = torch.randn(2, 3, 224, 224).to(device)
    out_c = convnext(dummy_img)
    print("ConvNeXt output shape:", out_c.shape) # [2, 2048]

    # ViT
    vit = EncoderWrapper("vit_base_patch16_384").to(device)
    dummy_img_vit = torch.randn(2, 3, 384, 384).to(device)
    out_v = vit(dummy_img_vit)
    print("ViT output shape:", out_v.shape) # [2, 2048]
