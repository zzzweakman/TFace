"""Frozen official DECA parameter extraction and channel-36 residual fusion."""
import hashlib
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

DECA_GROUPS = (("shape", 100), ("tex", 50), ("exp", 50),
               ("pose", 6), ("cam", 3), ("light", 27))
DECA_DIM = sum(size for _, size in DECA_GROUPS)
DEFAULT_ASSETS = Path(__file__).resolve().parent / "deca_assets_20261010"
OFFICIAL_CHECKPOINT_SHA256 = "e714ed293054cba5eea9c96bd3b6b57880074cd84b3fd00d606cbaf0bee7c5c2"


class FrozenDECAEncoder(nn.Module):
    """Exactly the official E_flame; no FLAME mesh or renderer is needed.

    Input is the same augmented RGB crop as the recognition student, normalized
    to [-1, 1]. Upsampling to 224 uses the whole already-aligned crop, without
    another face detection or recrop. Output order follows DECA param_list.
    """
    def __init__(self, checkpoint=None, vendor=None, microbatch=64):
        super().__init__()
        checkpoint = Path(checkpoint or DEFAULT_ASSETS / "deca_model.tar")
        vendor = Path(vendor or DEFAULT_ASSETS / "vendor")
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if digest != OFFICIAL_CHECKPOINT_SHA256:
            raise ValueError("DECA checkpoint checksum does not match verified official weights")
        sys.path.insert(0, str(vendor.resolve()))
        from decalib.models.encoders import ResnetEncoder
        self.encoder = ResnetEncoder(outsize=DECA_DIM)
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        self.encoder.load_state_dict(state["E_flame"], strict=True)
        if any(not torch.isfinite(tensor).all() for tensor in self.encoder.state_dict().values()):
            raise ValueError("Nonfinite DECA encoder weights")
        self.microbatch = int(microbatch)
        if self.microbatch < 1:
            raise ValueError("DECA microbatch must be positive")
        self.checkpoint_sha256 = digest
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # DECA's BN statistics must never change during recognizer training.
        return super().train(False)

    @torch.no_grad()
    def forward(self, rgb_images):
        if rgb_images.ndim != 4 or rgb_images.shape[1] != 3:
            raise ValueError("Expected [B, 3, H, W] RGB crops")
        chunks = []
        for batch in rgb_images.split(self.microbatch):
            pixels = (batch.float() + 1.0) * 0.5
            pixels = F.interpolate(pixels, size=(224, 224), mode="bilinear", align_corners=False)
            codes = self.encoder(pixels).float()
            if codes.shape[1] != DECA_DIM or not torch.isfinite(codes).all():
                raise FloatingPointError("Invalid DECA parameter extraction")
            chunks.append(codes)
        return torch.cat(chunks, dim=0)


class DECAFusionBackbone(nn.Module):
    """Fuse train-standardized 236-d parameters into the student's embedding."""
    def __init__(self, image_backbone, mean=None, std=None, condition_keep=0.9):
        super().__init__()
        self.image_backbone = image_backbone
        self.register_buffer("deca_mean", torch.zeros(DECA_DIM) if mean is None else torch.as_tensor(mean).float())
        self.register_buffer("deca_std", torch.ones(DECA_DIM) if std is None else torch.as_tensor(std).float())
        if self.deca_mean.shape != (DECA_DIM,) or self.deca_std.shape != (DECA_DIM,):
            raise ValueError("DECA normalization must contain 236 values")
        if not torch.isfinite(self.deca_mean).all() or not torch.isfinite(self.deca_std).all() or (self.deca_std <= 0).any():
            raise ValueError("Invalid DECA training normalization")
        self.condition_keep = float(condition_keep)
        if not 0 < self.condition_keep <= 1:
            raise ValueError("DECA condition keep probability must be in (0, 1]")
        # Same hidden size and residual-gate initialization as landmark-vector fusion.
        self.geometry = nn.Sequential(nn.Linear(DECA_DIM, 128), nn.ReLU(inplace=True),
                                      nn.Linear(128, 512), nn.LayerNorm(512))
        self.gate_logit = nn.Parameter(torch.tensor(-2.944439))

    def forward(self, dct_input, parameters, valid=None):
        if parameters.shape != (dct_input.size(0), DECA_DIM):
            raise ValueError("Expected one 236-d DECA vector per student image")
        image_feature = self.image_backbone(dct_input)
        parameters = (parameters.float() - self.deca_mean) / self.deca_std
        deca_feature = self.geometry(parameters)
        keep = torch.ones((parameters.size(0), 1), device=parameters.device, dtype=deca_feature.dtype)
        if valid is not None:
            keep = keep * valid.reshape(-1, 1).to(keep)
        if self.training and self.condition_keep < 1:
            keep = keep * (torch.rand_like(keep) < self.condition_keep)
        fused = image_feature + self.gate_logit.sigmoid() * keep * deca_feature
        return image_feature, fused
