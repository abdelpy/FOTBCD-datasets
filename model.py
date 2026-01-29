"""
Change Detection Benchmark Model.

Siamese DINOv3 ViT encoder with CNN decoder for binary change detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class SiameseEncoder(nn.Module):
    """
    Siamese DINOv3 ViT encoder for binary change detection.
    """

    def __init__(
        self,
        model_name: str = "vit_large_patch16_dinov3.sat493m",
        pretrained: bool = True,
        out_dim: int = 256,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.out_dim = out_dim

        # DINOv3 ViT backbone
        self.backbone = timm.create_model(
            model_name=model_name,
            num_classes=0,
            global_pool="",
            pretrained=pretrained,
        )
        self.embed_dim = self.backbone.embed_dim  # 1024 for ViT-L

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Project to output dimension
        self.proj = nn.Sequential(
            nn.Linear(self.embed_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        """
        Args:
            before: (B, 3, H, W)
            after: (B, 3, H, W)

        Returns:
            diff_features: (B, N, out_dim)
        """
        # Encode both images
        feat_before = self.backbone(before)
        feat_after = self.backbone(after)

        # Remove CLS and register tokens
        num_prefix = getattr(self.backbone, "num_prefix_tokens", 1)
        feat_before = feat_before[:, num_prefix:]
        feat_after = feat_after[:, num_prefix:]

        # Project
        feat_before = self.proj(feat_before)
        feat_after = self.proj(feat_after)

        # Absolute difference - binary change detection
        diff = torch.abs(feat_before - feat_after)

        return diff


class ResNetSpatialBranch(nn.Module):
    """
    Pretrained ResNet34 as spatial branch for strong boundary features.
    Processes before/after independently then computes difference features.
    """

    def __init__(self, pretrained: bool = True, shallow_dim: int = 32):
        super().__init__()
        self.shallow_dim = shallow_dim

        # Load pretrained ResNet34
        resnet = timm.create_model("resnet34", pretrained=pretrained, features_only=True)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.act1)  # 64, /2
        self.pool0 = resnet.maxpool  # /2
        self.layer1 = resnet.layer1  # 64, /4
        self.layer2 = resnet.layer2  # 128, /8
        self.layer3 = resnet.layer3  # 256, /16
        self.layer4 = resnet.layer4  # 512, /32

        # Shallow branch at full resolution (before pooling)
        self.shallow = nn.Sequential(
            nn.Conv2d(6, shallow_dim, 3, padding=1),
            nn.BatchNorm2d(shallow_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(shallow_dim, shallow_dim, 3, padding=1),
            nn.BatchNorm2d(shallow_dim),
            nn.ReLU(inplace=True),
        )

        # Project difference features to consistent dimensions
        # ResNet: 64->64, 64->64, 128->128, 256->256, 512->256
        self.proj1 = nn.Conv2d(64 * 2, 64, 1)   # /2 -> match 256
        self.proj2 = nn.Conv2d(64 * 2, 64, 1)   # /4 -> match 128
        self.proj3 = nn.Conv2d(128 * 2, 128, 1)  # /8 -> match 64
        self.proj4 = nn.Conv2d(256 * 2, 256, 1)  # /16 -> match 32
        self.proj5 = nn.Conv2d(512 * 2, 256, 1)  # /32

    def _extract_features(self, x: torch.Tensor) -> list:
        """Extract multi-scale features from single image."""
        f0 = self.layer0(x)      # B, 64, H/2, W/2
        f0p = self.pool0(f0)     # B, 64, H/4, W/4
        f1 = self.layer1(f0p)    # B, 64, H/4, W/4
        f2 = self.layer2(f1)     # B, 128, H/8, W/8
        f3 = self.layer3(f2)     # B, 256, H/16, W/16
        f4 = self.layer4(f3)     # B, 512, H/32, W/32
        return [f0, f1, f2, f3, f4]

    def forward(self, before: torch.Tensor, after: torch.Tensor) -> list:
        """
        Returns multi-scale difference features.
        [s0, s1, s2, s3, s4, s5] at resolutions: full, /2, /4, /8, /16, /32
        """
        # Shallow features at full resolution
        s0 = self.shallow(torch.cat([before, after], dim=1))

        # Extract ResNet features for both images
        feats_before = self._extract_features(before)
        feats_after = self._extract_features(after)

        # Compute difference features (concatenate to preserve info)
        diff1 = self.proj1(torch.cat([feats_before[0], feats_after[0]], dim=1))  # /2
        diff2 = self.proj2(torch.cat([feats_before[1], feats_after[1]], dim=1))  # /4
        diff3 = self.proj3(torch.cat([feats_before[2], feats_after[2]], dim=1))  # /8
        diff4 = self.proj4(torch.cat([feats_before[3], feats_after[3]], dim=1))  # /16
        diff5 = self.proj5(torch.cat([feats_before[4], feats_after[4]], dim=1))  # /32

        return [s0, diff1, diff2, diff3, diff4, diff5]


class PixelShuffleUp(nn.Module):
    """PixelShuffle upsampling"""

    def __init__(self, in_ch: int, out_ch: int, scale: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * (scale ** 2), 3, padding=1)
        self.shuffle = nn.PixelShuffle(scale)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.shuffle(self.conv(x))))


class HybridDecoder(nn.Module):
    """
    Decoder with PixelShuffle upsampling.

    Scale alignment (512x512 input):
    - ViT (patch=16): 32x32 feature map
    - ResNet: s0(512), s1(256), s2(128), s3(64), s4(32), s5(16)

    Decoder: 32 -> 64 -> 128 -> 256 -> 512 (4 up blocks)
    """

    def __init__(self, vit_dim: int = 256, shallow_dim: int = 32):
        super().__init__()
        self.shallow_dim = shallow_dim

        # Bottleneck: fuse ViT (32x32) + s5 (16->32) + s4 (32x32)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(vit_dim + 256 + 256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # PixelShuffle upsampling
        self.up1 = PixelShuffleUp(256, 128)  # 32->64
        self.fuse1 = self._fuse_block(128 + 128, 128)  # +s3

        self.up2 = PixelShuffleUp(128, 64)   # 64->128
        self.fuse2 = self._fuse_block(64 + 64, 64)   # +s2

        self.up3 = PixelShuffleUp(64, 64)    # 128->256
        self.fuse3 = self._fuse_block(64 + 64, 64)   # +s1

        self.up4 = PixelShuffleUp(64, 32)    # 256->512
        self.fuse4 = self._fuse_block(32 + shallow_dim, 32)  # +s0

        # Final head
        self.head = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )

    def _fuse_block(self, in_ch: int, out_ch: int) -> nn.Module:
        """Conv block after skip concat."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        vit_features: torch.Tensor,
        resnet_features: list,
        output_size: tuple,
    ) -> torch.Tensor:
        """
        Args:
            vit_features: (B, N, D) from ViT encoder, N=32*32=1024 for 512 input
            resnet_features: [s0, s1, s2, s3, s4, s5] at 512/256/128/64/32/16
            output_size: target output size
        """
        B, N, D = vit_features.shape
        h = w = int(N ** 0.5)  # 32 for 512 input

        # Reshape ViT features to spatial
        vit = vit_features.permute(0, 2, 1).view(B, D, h, w)  # B, D, 32, 32

        # Unpack ResNet features
        s0, s1, s2, s3, s4, s5 = resnet_features

        # Bottleneck at 32x32: fuse ViT + s5_up + s4
        s5_up = F.interpolate(s5, size=(h, w), mode="nearest")  # nearest for s5, it's small
        x = self.bottleneck(torch.cat([vit, s5_up, s4], dim=1))

        # up1: 32->64, fuse with s3
        x = self.up1(x)
        x = self.fuse1(torch.cat([x, s3], dim=1))

        # up2: 64->128, fuse with s2
        x = self.up2(x)
        x = self.fuse2(torch.cat([x, s2], dim=1))

        # up3: 128->256, fuse with s1
        x = self.up3(x)
        x = self.fuse3(torch.cat([x, s1], dim=1))

        # up4: 256->512, fuse with s0
        x = self.up4(x)
        x = self.fuse4(torch.cat([x, s0], dim=1))

        # Final head
        x = self.head(x)

        if x.shape[2:] != output_size:
            x = F.interpolate(x, size=output_size, mode="nearest")  # nearest, not bilinear

        return x


class ProgressiveRefinement(nn.Module):
    """
    Multi-scale progressive boundary refinement.
    Refines predictions from coarse to fine using edge information.
    """

    def __init__(self, shallow_dim: int = 32, hidden_dim: int = 32):
        super().__init__()
        # Coarse refinement at /2 resolution
        self.refine_coarse = nn.Sequential(
            nn.Conv2d(1 + 64 + 3, hidden_dim, 3, padding=1),  # logits + s1 + rgb_diff
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1),
        )

        # Fine refinement at full resolution
        self.refine_fine = nn.Sequential(
            nn.Conv2d(1 + shallow_dim + 3, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1),
        )

    def forward(
        self,
        logits: torch.Tensor,
        shallow_feats: torch.Tensor,
        s1_feats: torch.Tensor,
        rgb_diff: torch.Tensor,
    ) -> torch.Tensor:
        """
        Progressive refinement from coarse to fine.
        """
        output_size = logits.shape[2:]

        # Coarse refinement at /2
        logits_half = F.interpolate(logits, scale_factor=0.5, mode="bilinear", align_corners=False)
        rgb_half = F.interpolate(rgb_diff, size=logits_half.shape[2:], mode="bilinear", align_corners=False)
        s1_sized = F.interpolate(s1_feats, size=logits_half.shape[2:], mode="bilinear", align_corners=False)

        coarse_residual = self.refine_coarse(torch.cat([logits_half, s1_sized, rgb_half], dim=1))
        logits_half = logits_half + coarse_residual

        # Upsample to full resolution
        logits = F.interpolate(logits_half, size=output_size, mode="bilinear", align_corners=False)

        # Fine refinement at full resolution
        fine_residual = self.refine_fine(torch.cat([logits, shallow_feats, rgb_diff], dim=1))
        return logits + fine_residual


class HybridChangeDetector(nn.Module):
    """
    Hybrid DINOv3 + ResNet Change Detector.

    Combines:
    - DINOv3 ViT: Strong semantic understanding of what changed
    - ResNet34: Strong boundary/geometry features with pretrained weights
    - Edge-guided fusion: Explicit edge attention at every scale
    - Progressive refinement: Coarse-to-fine boundary sharpening

    This architecture keeps DINOv3's semantic power while fixing the geometry issues
    through strong CNN boundary features and explicit edge guidance.
    """

    def __init__(
        self,
        encoder_name: str = "vit_large_patch16_dinov3.sat493m",
        pretrained: bool = True,
        encoder_dim: int = 256,
        freeze_vit: bool = False,
        shallow_dim: int = 32,
        use_refinement: bool = True,
    ):
        super().__init__()
        self.use_refinement = use_refinement
        self.shallow_dim = shallow_dim

        # DINOv3 for semantic features
        self.encoder = SiameseEncoder(
            model_name=encoder_name,
            pretrained=pretrained,
            out_dim=encoder_dim,
            freeze_backbone=freeze_vit,
        )

        # ResNet34 for strong boundary features
        self.spatial_branch = ResNetSpatialBranch(
            pretrained=pretrained,
            shallow_dim=shallow_dim,
        )

        # Hybrid decoder with edge-guided fusion
        self.decoder = HybridDecoder(
            vit_dim=encoder_dim,
            shallow_dim=shallow_dim,
        )

        # Progressive refinement
        if use_refinement:
            self.refinement = ProgressiveRefinement(
                shallow_dim=shallow_dim,
                hidden_dim=32,
            )

    def forward(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        output_size = (before.shape[2], before.shape[3])

        # Semantic features from DINOv3
        semantic_feats = self.encoder(before, after)

        # Spatial/boundary features from ResNet
        spatial_feats = self.spatial_branch(before, after)

        # Decode with U-Net style fusion
        logits = self.decoder(semantic_feats, spatial_feats, output_size)

        if self.use_refinement:
            # s0 = shallow (full res), s1 = /2 resolution
            rgb_diff = torch.abs(after - before)
            s0, s1 = spatial_feats[0], spatial_feats[1]
            logits = self.refinement(logits, s0, s1, rgb_diff)

        return logits

    def predict(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        logits = self.forward(before, after)
        return (torch.sigmoid(logits) > 0.5).float()


def build_model(cfg) -> nn.Module:
    """Build model from config."""
    return HybridChangeDetector(
        encoder_name=getattr(cfg, "MODEL_NAME", "vit_large_patch16_dinov3.sat493m"),
        pretrained=getattr(cfg, "PRETRAINED", True),
        encoder_dim=getattr(cfg, "ENCODER_DIM", 256),
        freeze_vit=getattr(cfg, "FREEZE_ENCODER", False),
        shallow_dim=getattr(cfg, "SHALLOW_DIM", 32),
        use_refinement=getattr(cfg, "USE_REFINEMENT", True),
    )


if __name__ == "__main__":
    print("Testing HybridChangeDetector...")
    model = HybridChangeDetector(pretrained=False)
    x = torch.randn(2, 3, 512, 512)
    out = model(x, x)
    print(f"Input: {x.shape}, Output: {out.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
