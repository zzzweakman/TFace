"""Spatial landmark conditioning for the channel-36 IR-50 student.

The FaceLocator pattern (convolutional landmark-image encoder with a zero-initialized
output projection) is adapted to 2D recognition from EchoMimic:
https://github.com/antgroup/echomimic/blob/main/src/models/face_locator.py
EchoMimic is Apache-2.0 licensed; this module is a task-specific adaptation rather
than a copy of its video-generation architecture.
"""

import torch
import torch.nn.functional as F
from torch import nn


class LandmarkHeatmap(nn.Module):
    """Rasterize normalized 106-point coordinates into a single spatial map."""

    def __init__(self, size=112, point_keep_probability=0.85):
        super().__init__()
        self.size = size
        self.point_keep_probability = point_keep_probability
        kernel = torch.arange(-2, 3, dtype=torch.float32)
        kernel = torch.exp(-0.5 * (kernel / 1.25).square())
        kernel = torch.outer(kernel, kernel)
        self.register_buffer("blur_kernel", (kernel / kernel.sum()).view(1, 1, 5, 5))

    def forward(self, points, valid):
        if points.ndim != 3 or points.shape[1:] != (106, 2):
            raise ValueError("Expected landmark coordinates of shape [batch, 106, 2]")
        if valid.numel() != points.size(0):
            raise ValueError("Expected one landmark-valid flag per image")
        batch = points.size(0)
        inside = ((points >= 0) & (points < 1)).all(dim=-1)
        inside = inside & torch.isfinite(points).all(dim=-1)
        if self.training and self.point_keep_probability < 1:
            inside = inside & (torch.rand_like(inside, dtype=torch.float32)
                               < self.point_keep_probability)
        inside = inside & valid.reshape(-1, 1).bool()
        pixels = (torch.nan_to_num(points).clamp(0, 1) * self.size).long().clamp(0, self.size - 1)
        flat_index = pixels[..., 1] * self.size + pixels[..., 0]
        heatmap = points.new_zeros((batch, self.size * self.size))
        heatmap.scatter_add_(1, flat_index, inside.to(heatmap.dtype))
        heatmap = heatmap.clamp_(max=1).view(batch, 1, self.size, self.size)
        return F.conv2d(heatmap, self.blur_kernel.to(heatmap.dtype), padding=2) * 16


class FaceLocator2D(nn.Module):
    """EchoMimic-style spatial encoder adapted to the first IR-50 stage."""

    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(1, 16, 3, padding=1)
        self.blocks = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.SiLU(),
        )
        self.conv_out = nn.Conv2d(32, 64, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, heatmap):
        return self.conv_out(self.blocks(F.silu(self.conv_in(heatmap))))


class SpatialLandmarkBackbone(nn.Module):
    """Add landmark maps after IR-50 stage one, before identity embedding."""

    def __init__(self, image_backbone, point_keep_probability=0.85,
                 condition_keep_probability=0.9):
        super().__init__()
        if len(image_backbone.body) != 24:
            raise ValueError("Spatial conditioning currently supports IR-50 only")
        self.image_backbone = image_backbone
        self.heatmap = LandmarkHeatmap(point_keep_probability=point_keep_probability)
        self.face_locator = FaceLocator2D()
        self.condition_keep_probability = condition_keep_probability

    def forward(self, dct_input, points, valid):
        heatmap = self.heatmap(points, valid)
        condition = self.face_locator(heatmap)
        keep = valid.reshape(-1, 1, 1, 1).to(condition.dtype)
        if self.training and self.condition_keep_probability < 1:
            keep = keep * (torch.rand_like(keep) < self.condition_keep_probability)
        x = self.image_backbone.input_layer(dct_input)
        for index, block in enumerate(self.image_backbone.body):
            x = block(x)
            if index == 2:
                if x.shape != condition.shape:
                    raise ValueError("Landmark feature shape does not match IR-50 stage one")
                x = x + condition * keep
        return self.image_backbone.output_layer(x)
