import torch
import torch.nn as nn

# ── ENCODER BLOCK ──────────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

# ── UNET-STYLE LANE SEGMENTATION CNN ──────────────────────────────────────────
class LaneNet(nn.Module):
    """
    Lightweight U-Net that takes a (B, 3, H, W) image
    and outputs a (B, 1, H, W) binary lane mask.
    """
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(3,   32)
        self.enc2 = ConvBlock(32,  64)
        self.enc3 = ConvBlock(64,  128)

        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout2d(0.3)

        # Bottleneck
        self.bottleneck = ConvBlock(128, 256)

        # Decoder
        self.up3    = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3   = ConvBlock(256, 128)   # 128 skip + 128 up

        self.up2    = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2   = ConvBlock(128, 64)    # 64 skip + 64 up

        self.up1    = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1   = ConvBlock(64, 32)     # 32 skip + 32 up

        # Output: single-channel binary mask
        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        b  = self.bottleneck(self.drop(self.pool(e3)))

        # Decoder (with skip connections)
        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out(d1)   # raw logits → apply sigmoid for mask

# ── QUICK SANITY CHECK ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = LaneNet()
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")

    dummy = torch.randn(1, 3, 256, 512)
    out   = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")
    print("LaneNet architecture OK!")