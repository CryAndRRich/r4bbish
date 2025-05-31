import timm
import torch
import torch.nn as nn

class EncoderWrapper(nn.Module):
    def __init__(self, 
                 encoder_name: str, 
                 output_dim: int = 2048) -> None:
        super(EncoderWrapper, self).__init__()
        self.encoder_name = encoder_name
        self.encoder = timm.create_model(encoder_name, pretrained=True, num_classes=0)  

        dummy_input = torch.randn(1, 3, 224, 224) if "convnext" in encoder_name else torch.randn(1, 3, 384, 384)
        with torch.no_grad():
            out = self.encoder(dummy_input)
            encoder_out_dim = out.shape[1]  

        if encoder_out_dim != output_dim:
            self.output_mapping = nn.Linear(encoder_out_dim, output_dim)
        else:
            self.output_mapping = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)          
        output = self.output_mapping(features) 
        return output

if __name__ == "__main__":
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