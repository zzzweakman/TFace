import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))
from frequency_utils import (
    DEFAULT_LOW_FREQ_CHANNELS,
    build_frequency_views,
    get_keep_channels,
    load_image_as_tensor,
    rebuild_spatial_from_selected,
    tensor_to_uint8_image,
)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Visualize low/high-frequency channel splits and spatial reconstructions.",
    )
    parser.add_argument("--input", required=True, help="Input image path.")
    parser.add_argument("--output-dir", required=True, help="Directory for visualization outputs.")
    parser.add_argument(
        "--low-freq-channels",
        default=",".join(str(idx) for idx in DEFAULT_LOW_FREQ_CHANNELS),
        help="Comma-separated low-frequency channel indices in the 8x8 DCT block.",
    )
    parser.add_argument(
        "--save-individual-high",
        action="store_true",
        help="Also reconstruct and save every selected high-frequency channel individually.",
    )
    return parser.parse_args()


def save_rgb(path, image):
    import cv2

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), image_bgr)


def make_channel_map(low_channels):
    import cv2
    import numpy as np

    high_channels = set(get_keep_channels(mode="high_freq", low_freq_channels=low_channels))
    vis = np.zeros((8 * 40, 8 * 40, 3), dtype=np.uint8)
    for idx in range(64):
        row = idx // 8
        col = idx % 8
        color = np.array([76, 175, 80], dtype=np.uint8) if idx in high_channels else np.array([244, 67, 54], dtype=np.uint8)
        vis[row * 40 : (row + 1) * 40, col * 40 : (col + 1) * 40] = color
        cv2.putText(
            vis,
            str(idx),
            (col * 40 + 7, row * 40 + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return vis


def enhance_high_frequency_image(high_image):
    import numpy as np

    high_np = high_image.detach().cpu()
    if high_np.ndim == 4:
        high_np = high_np[0]

    # Center around the neutral gray level used by inverse reconstruction.
    deviation = (high_np - 0.5).abs()
    deviation = deviation.mean(dim=0).numpy()

    max_value = float(deviation.max())
    if max_value > 0:
        deviation = deviation / max_value

    deviation = np.clip(deviation, 0.0, 1.0)
    deviation = (deviation * 255.0).round().astype(np.uint8)
    deviation_rgb = np.repeat(deviation[:, :, None], 3, axis=2)
    return deviation_rgb


def add_caption(image, caption, target_size=None):
    import cv2
    import numpy as np

    if target_size is not None:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_NEAREST)

    height, width = image.shape[:2]
    canvas = np.full((height + 44, width, 3), 255, dtype=np.uint8)
    canvas[44:, :, :] = image
    cv2.putText(
        canvas,
        caption,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    return canvas


def build_overview(images_with_titles, columns=2, gap=12):
    import numpy as np

    widths = [image.shape[1] for image, _ in images_with_titles]
    heights = [image.shape[0] for image, _ in images_with_titles]
    target_width = max(widths)
    target_height = max(heights)

    captioned = []
    for image, title in images_with_titles:
        captioned.append(add_caption(image, title, target_size=(target_width, target_height)))

    rows = []
    for start in range(0, len(captioned), columns):
        row_images = captioned[start : start + columns]
        if len(row_images) < columns:
            blank = np.full_like(captioned[0], 255)
            while len(row_images) < columns:
                row_images.append(blank.copy())
        row = np.concatenate(row_images, axis=1)
        rows.append(row)

    overview = rows[0]
    for row in rows[1:]:
        spacer = np.full((gap, overview.shape[1], 3), 255, dtype=np.uint8)
        overview = np.concatenate([overview, spacer, row], axis=0)
    return overview


def main():
    import cv2

    args = parse_args()
    low_channels = [int(item.strip()) for item in args.low_freq_channels.split(",") if item.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_tensor = load_image_as_tensor(args.input)
    low_tensor, high_tensor, low_image, high_image = build_frequency_views(input_tensor, low_freq_channels=low_channels)

    original = tensor_to_uint8_image(input_tensor * 0.5 + 0.5)
    low_rgb = tensor_to_uint8_image(low_image)
    high_rgb = tensor_to_uint8_image(high_image)
    high_enhanced = enhance_high_frequency_image(high_image)

    save_rgb(output_dir / "original.png", original)
    save_rgb(output_dir / "low_freq_reconstruction.png", low_rgb)
    save_rgb(output_dir / "high_freq_reconstruction.png", high_rgb)
    save_rgb(output_dir / "high_freq_enhanced.png", high_enhanced)
    channel_map = make_channel_map(low_channels)
    cv2.imwrite(str(output_dir / "channel_map.png"), channel_map)

    overview = build_overview(
        [
            (original, "Original Image"),
            (low_rgb, "Low-Frequency Reconstruction"),
            (high_rgb, "High-Frequency Reconstruction"),
            (high_enhanced, "High-Frequency Enhanced"),
            (channel_map, "DCT Channel Assignment"),
        ],
        columns=2,
    )
    save_rgb(output_dir / "overview.png", overview)

    if args.save_individual_high:
        high_channels = get_keep_channels(mode="high_freq", low_freq_channels=low_channels)
        per_color = len(high_channels)
        for idx, channel in enumerate(high_channels):
            select_idx = [idx, per_color + idx, per_color * 2 + idx]
            channel_tensor = high_tensor[:, select_idx, :, :]
            channel_image = rebuild_spatial_from_selected(channel_tensor, [channel])
            save_rgb(output_dir / "high_channel_{:02d}.png".format(channel), tensor_to_uint8_image(channel_image))

    print("Saved visualization outputs to {}".format(output_dir))


if __name__ == "__main__":
    main()
