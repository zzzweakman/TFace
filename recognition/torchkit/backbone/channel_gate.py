import torch
import torch.nn as nn


class FrequencyChannelGate(nn.Module):
    """A lightweight, static gate for ranking frequency input channels."""

    def __init__(self, backbone, num_channels, init_value=1.0, use_sigmoid=True):
        super(FrequencyChannelGate, self).__init__()
        self.backbone = backbone
        self.use_sigmoid = use_sigmoid
        if use_sigmoid:
            init_value = min(max(float(init_value), 1e-4), 1.0 - 1e-4)
            init_value = torch.logit(torch.tensor(init_value)).item()
        self.gate = nn.Parameter(torch.full((num_channels,), float(init_value)))

    def gate_values(self):
        if self.use_sigmoid:
            return torch.sigmoid(self.gate)
        return self.gate

    def gate_l1(self):
        return self.gate_values().abs().mean()

    def forward(self, x):
        scale = self.gate_values().view(1, -1, 1, 1).to(dtype=x.dtype, device=x.device)
        return self.backbone(x * scale)
