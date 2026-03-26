from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F


DEFAULT_LOW_FREQ_CHANNELS = [0, 1, 2, 3, 4, 5, 8, 9, 16, 24]


def _get_dct_module():
    from torchjpeg import dct

    return dct


def parse_channel_spec(channel_spec, low_freq_channels=None):
    if low_freq_channels is None:
        low_freq_channels = DEFAULT_LOW_FREQ_CHANNELS
    if channel_spec is None:
        return None
    if isinstance(channel_spec, (list, tuple)):
        channels = [int(item) for item in channel_spec]
    else:
        spec = str(channel_spec).strip().lower()
        if spec in ("", "none"):
            return None
        if spec == "low":
            channels = list(low_freq_channels)
        elif spec == "high":
            channels = [idx for idx in range(64) if idx not in set(low_freq_channels)]
        elif spec == "all":
            channels = list(range(64))
        else:
            channels = [int(item.strip()) for item in spec.split(",") if item.strip()]
    channels = sorted(set(channels))
    invalid = [idx for idx in channels if idx < 0 or idx >= 64]
    if invalid:
        raise RuntimeError("Invalid frequency channels: {}".format(invalid))
    return channels


def get_keep_channels(mode="rgb", keep_channels=None, low_freq_channels=None):
    if low_freq_channels is None:
        low_freq_channels = DEFAULT_LOW_FREQ_CHANNELS
    mode = str(mode).lower()
    if keep_channels is not None:
        return parse_channel_spec(keep_channels, low_freq_channels=low_freq_channels)
    if mode == "rgb":
        return None
    if mode == "high_freq":
        return parse_channel_spec("high", low_freq_channels=low_freq_channels)
    if mode == "low_freq":
        return parse_channel_spec("low", low_freq_channels=low_freq_channels)
    raise RuntimeError("Unsupported preprocess mode: {}".format(mode))


def expected_input_channels(mode="rgb", keep_channels=None, low_freq_channels=None):
    channels = get_keep_channels(mode=mode, keep_channels=keep_channels, low_freq_channels=low_freq_channels)
    if channels is None:
        return 3
    return len(channels) * 3


def images_to_dct_blocks(x, size=8, stride=8, pad=0, dilation=1, ratio=8):
    dct = _get_dct_module()
    x = x * 0.5 + 0.5
    x = F.interpolate(x, scale_factor=ratio, mode="bilinear", align_corners=True)
    x = x * 255.0
    if x.shape[1] == 3:
        x = dct.to_ycbcr(x)
    x = x - 128.0

    batch_size, channels, height, width = x.shape
    block_num = height // stride
    x = x.view(batch_size * channels, 1, height, width)
    x = F.unfold(
        x,
        kernel_size=(size, size),
        dilation=dilation,
        padding=pad,
        stride=(stride, stride),
    )
    x = x.transpose(1, 2)
    x = x.view(batch_size, channels, -1, size, size)
    dct_block = dct.block_dct(x)
    dct_block = dct_block.view(batch_size, channels, block_num, block_num, size * size)
    dct_block = dct_block.permute(0, 1, 4, 2, 3).contiguous()
    return dct_block


def dct_blocks_to_images(dct_block, size=8, stride=8, pad=0, dilation=1, ratio=8):
    dct = _get_dct_module()
    batch_size, channels, _, block_h, block_w = dct_block.shape
    dct_block = dct_block.permute(0, 1, 3, 4, 2).contiguous()
    dct_block = dct_block.view(batch_size, channels, block_h * block_w, size, size)
    x = dct.block_idct(dct_block)
    x = x.view(batch_size * channels, block_h * block_w, size * size)
    x = x.transpose(1, 2)
    x = F.fold(
        x,
        output_size=(block_h * stride, block_w * stride),
        kernel_size=(size, size),
        dilation=dilation,
        padding=pad,
        stride=(stride, stride),
    )
    x = x.view(batch_size, channels, block_h * stride, block_w * stride)
    x = x + 128.0
    if channels == 3:
        x = dct.to_rgb(x)
    x = x / 255.0
    x = F.interpolate(x, scale_factor=1 / ratio, mode="bilinear", align_corners=True)
    return x.clamp(min=0.0, max=1.0)


def select_channels(dct_block, keep_channels, pad_missing=False):
    keep_channels = parse_channel_spec(keep_channels)
    if keep_channels is None:
        raise RuntimeError("keep_channels must not be None when selecting frequency channels")
    batch_size, channels, _, block_h, block_w = dct_block.shape
    selected = dct_block[:, :, keep_channels, :, :]
    if pad_missing:
        padded = dct_block.new_zeros((batch_size, channels, 64, block_h, block_w))
        padded[:, :, keep_channels, :, :] = selected
        return padded
    return selected


def frequency_tensor_from_images(x, mode="rgb", keep_channels=None, low_freq_channels=None, ratio=8):
    keep_channels = get_keep_channels(mode=mode, keep_channels=keep_channels, low_freq_channels=low_freq_channels)
    if keep_channels is None:
        return x
    dct_block = images_to_dct_blocks(x, ratio=ratio)
    selected = select_channels(dct_block, keep_channels, pad_missing=False)
    batch_size = x.shape[0]
    return selected.reshape(batch_size, -1, selected.shape[-2], selected.shape[-1])


def rebuild_spatial_from_selected(x, keep_channels, ratio=8):
    keep_channels = parse_channel_spec(keep_channels)
    if x.ndim != 4:
        raise RuntimeError("Expected x to be a 4D tensor, got shape {}".format(tuple(x.shape)))
    batch_size, all_channels, block_h, block_w = x.shape
    per_color = len(keep_channels)
    if all_channels != per_color * 3:
        raise RuntimeError(
            "Selected tensor channel count {} does not match keep_channels {} x 3".format(all_channels, per_color)
        )
    x = x.view(batch_size, 3, per_color, block_h, block_w)
    padded = x.new_zeros((batch_size, 3, 64, block_h, block_w))
    padded[:, :, keep_channels, :, :] = x
    return dct_blocks_to_images(padded, ratio=ratio)


def build_frequency_views(x, low_freq_channels=None, ratio=8):
    if low_freq_channels is None:
        low_freq_channels = DEFAULT_LOW_FREQ_CHANNELS
    low_tensor = frequency_tensor_from_images(
        x,
        mode="low_freq",
        low_freq_channels=low_freq_channels,
        ratio=ratio,
    )
    high_tensor = frequency_tensor_from_images(
        x,
        mode="high_freq",
        low_freq_channels=low_freq_channels,
        ratio=ratio,
    )
    low_image = rebuild_spatial_from_selected(low_tensor, low_freq_channels, ratio=ratio)
    high_channels = get_keep_channels(mode="high_freq", low_freq_channels=low_freq_channels)
    high_image = rebuild_spatial_from_selected(high_tensor, high_channels, ratio=ratio)
    return low_tensor, high_tensor, low_image, high_image


def load_image_as_tensor(image_path):
    import cv2
    import numpy as np

    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError("Failed to read image: {}".format(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (112, 112))
    image = image.astype(np.float32) / 255.0
    image = (image - 0.5) / 0.5
    image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
    return image


def tensor_to_uint8_image(tensor):
    if tensor.ndim == 4:
        tensor = tensor[0]
    image = tensor.detach().cpu().clamp(0.0, 1.0).numpy()
    image = (image.transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
    return image
