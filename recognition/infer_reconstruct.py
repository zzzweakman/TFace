import argparse
import glob
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from frequency_utils import (
    DEFAULT_LOW_FREQ_CHANNELS,
    frequency_tensor_from_images,
    get_keep_channels,
    rebuild_spatial_from_selected,
)
from torchkit.backbone import get_model
from torchkit.util import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Reconstruct RGB face images from high-frequency DCT inputs.")
    parser.add_argument("--config", default="train_casia_highfreq_reconstruct_50ep.yaml", help="reconstruction config")
    parser.add_argument("--checkpoint", default="", help="model checkpoint; defaults to latest checkpoint in MODEL_ROOT")
    parser.add_argument("--input", default="", help="input image, directory, or text file with image paths")
    parser.add_argument("--output-dir", default="reconstruct_outputs", help="directory to save reconstructed images")
    parser.add_argument("--device", default="cuda:0", help="device, e.g. cuda:0 or cpu")
    parser.add_argument("--max-images", type=int, default=10, help="maximum images to process")
    parser.add_argument("--save-original", action="store_true", help="also save resized original images")
    parser.add_argument("--save-separate", action="store_true", help="save separate original/high/reconstruction images")
    parser.add_argument("--default-data-root", default="/private/codes/exp/face_workspace/zz/dataset/CASIA")
    parser.add_argument("--default-list", default="CASIA_namelist.txt")
    return parser.parse_args()


def find_latest_checkpoint(model_root):
    pattern = os.path.join(model_root, "Backbone_Epoch_*_checkpoint.pth")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        raise RuntimeError("No checkpoint found under {}".format(model_root))

    def epoch_number(path):
        name = os.path.basename(path)
        try:
            return int(name.split("Backbone_Epoch_", 1)[1].split("_checkpoint.pth", 1)[0])
        except (IndexError, ValueError):
            return -1

    return max(checkpoints, key=epoch_number)


def collect_inputs(input_path, extensions, default_data_root=None, default_list=None):
    if not input_path:
        if default_data_root is None or default_list is None:
            raise RuntimeError("--input is required when default dataset settings are not provided")
        list_path = Path(default_data_root) / default_list
        paths = []
        with list_path.open("r") as f:
            for line in f:
                item = line.strip().split()[0] if line.strip() else ""
                if item:
                    paths.append(Path(default_data_root) / item)
        return paths

    input_path = Path(input_path)
    if input_path.is_dir():
        paths = []
        for ext in extensions:
            paths.extend(input_path.rglob("*{}".format(ext)))
            paths.extend(input_path.rglob("*{}".format(ext.upper())))
        return sorted(set(paths))

    if input_path.suffix.lower() == ".txt":
        paths = []
        with input_path.open("r") as f:
            for line in f:
                item = line.strip().split()[0] if line.strip() else ""
                if item:
                    paths.append(Path(item))
        return paths

    return [input_path]


def load_image(image_path, input_size, mean, std):
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError("Failed to read image: {}".format(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, tuple(input_size), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0
    normalized = (image - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0)
    return tensor, image


def tensor_to_rgb_uint8(tensor, mean, std):
    if tensor.ndim == 4:
        tensor = tensor[0]
    image = tensor.detach().float().cpu().numpy().transpose(1, 2, 0)
    image = image * np.array(std, dtype=np.float32) + np.array(mean, dtype=np.float32)
    image = np.clip(image, 0.0, 1.0)
    return (image * 255.0).round().astype(np.uint8)


def tensor_01_to_rgb_uint8(tensor):
    if tensor.ndim == 4:
        tensor = tensor[0]
    image = tensor.detach().float().cpu().clamp(0.0, 1.0).numpy().transpose(1, 2, 0)
    return (image * 255.0).round().astype(np.uint8)


def add_caption(image, caption):
    height, width = image.shape[:2]
    canvas = np.full((height + 36, width, 3), 255, dtype=np.uint8)
    canvas[36:, :, :] = image
    cv2.putText(
        canvas,
        caption,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    return canvas


def make_triplet(original, high_image, recon):
    panels = [
        add_caption(original, "Original"),
        add_caption(high_image, "High-Frequency"),
        add_caption(recon, "UNet Reconstruction"),
    ]
    return np.concatenate(panels, axis=1)


def build_overview(triplets, gap=10):
    if not triplets:
        raise RuntimeError("No images to visualize")
    rows = [triplets[0]]
    for triplet in triplets[1:]:
        spacer = np.full((gap, triplet.shape[1], 3), 255, dtype=np.uint8)
        rows.extend([spacer, triplet])
    return np.concatenate(rows, axis=0)


def make_model(cfg, checkpoint, device):
    model_name = cfg.get("MODEL_NAME", "UNetReconstruction")
    model = get_model(model_name)(
        input_channel=cfg["INPUT_CHANNELS"],
        output_channel=cfg.get("OUTPUT_CHANNELS", 3),
        base_channels=cfg.get("UNET_BASE_CHANNELS", 72),
    )
    state_dict = torch.load(checkpoint, map_location=device)
    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    checkpoint = args.checkpoint or find_latest_checkpoint(cfg["MODEL_ROOT"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = make_model(cfg, checkpoint, device)
    extensions = cfg.get("IMAGE_EXTENSIONS", [".jpg", ".jpeg", ".png"])
    image_paths = collect_inputs(
        args.input,
        extensions,
        default_data_root=args.default_data_root,
        default_list=args.default_list,
    )
    if args.max_images > 0:
        image_paths = image_paths[:args.max_images]

    print("Loaded checkpoint: {}".format(checkpoint))
    print("Processing {} image(s)".format(len(image_paths)))

    mean = cfg.get("RGB_MEAN", [0.5, 0.5, 0.5])
    std = cfg.get("RGB_STD", [0.5, 0.5, 0.5])
    low_freq_channels = cfg.get("FREQ_LOW_CHANNELS", DEFAULT_LOW_FREQ_CHANNELS)
    high_channels = get_keep_channels(mode="high_freq", low_freq_channels=low_freq_channels)
    triplets = []

    with torch.no_grad():
        for idx, image_path in enumerate(image_paths):
            image_tensor, original = load_image(image_path, cfg["INPUT_SIZE"], mean, std)
            image_tensor = image_tensor.to(device)
            freq_tensor = frequency_tensor_from_images(
                image_tensor,
                mode=cfg.get("PREPROCESS_MODE", "high_freq"),
                keep_channels=cfg.get("FREQ_KEEP_CHANNELS", None),
                low_freq_channels=low_freq_channels,
                ratio=cfg.get("DCT_SAMPLING_RATIO", 8),
            )
            output = model(freq_tensor)
            high_spatial = rebuild_spatial_from_selected(
                freq_tensor.detach().cpu(),
                high_channels,
                ratio=cfg.get("DCT_SAMPLING_RATIO", 8),
            )
            high_rgb = tensor_01_to_rgb_uint8(high_spatial)
            recon = tensor_to_rgb_uint8(output, mean, std)
            original_rgb = (original * 255.0).round().astype(np.uint8)

            stem = Path(image_path).stem
            triplet = make_triplet(original_rgb, high_rgb, recon)
            triplet_path = output_dir / "{:02d}_{}_compare.png".format(idx + 1, stem)
            cv2.imwrite(str(triplet_path), cv2.cvtColor(triplet, cv2.COLOR_RGB2BGR))
            triplets.append(triplet)

            if args.save_separate:
                cv2.imwrite(str(output_dir / "{}_high.png".format(stem)), cv2.cvtColor(high_rgb, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(output_dir / "{}_recon.png".format(stem)), cv2.cvtColor(recon, cv2.COLOR_RGB2BGR))
            if args.save_original or args.save_separate:
                original_path = output_dir / "{}_original.png".format(stem)
                cv2.imwrite(str(original_path), cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR))

            print("[{}/{}] {}".format(idx + 1, len(image_paths), triplet_path))

    overview = build_overview(triplets)
    overview_path = output_dir / "overview_original_high_recon.png"
    cv2.imwrite(str(overview_path), cv2.cvtColor(overview, cv2.COLOR_RGB2BGR))
    print("Saved overview: {}".format(overview_path))


if __name__ == "__main__":
    main()
