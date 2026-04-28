import argparse
import csv
import os
import sys

import torch

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))
from frequency_utils import DEFAULT_LOW_FREQ_CHANNELS, get_keep_channels
from torchkit.backbone import get_model
from torchkit.backbone.channel_gate import FrequencyChannelGate
from torchkit.util import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Export learned frequency channel gate scores.")
    parser.add_argument("--config", required=True, help="Config used to train the gated recognition model.")
    parser.add_argument("--checkpoint", required=True, help="Backbone checkpoint path.")
    parser.add_argument("--output", default="channel_gate_scores.csv", help="CSV output path.")
    return parser.parse_args()


def strip_module_prefix(state_dict):
    if any(key.startswith("module.") for key in state_dict.keys()):
        return {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    return state_dict


def build_gated_model(cfg):
    backbone = get_model(cfg["BACKBONE_NAME"])(cfg["INPUT_SIZE"], input_channel=cfg["INPUT_CHANNELS"])
    return FrequencyChannelGate(
        backbone,
        num_channels=cfg["INPUT_CHANNELS"],
        init_value=cfg.get("FREQ_CHANNEL_GATE_INIT", 0.99),
        use_sigmoid=cfg.get("FREQ_CHANNEL_GATE_SIGMOID", True),
    )


def main():
    args = parse_args()
    cfg = load_config(args.config)
    model = build_gated_model(cfg)
    state_dict = strip_module_prefix(torch.load(args.checkpoint, map_location="cpu"))
    model.load_state_dict(state_dict)
    gate_values = model.gate_values().detach().cpu()

    low_channels = cfg.get("FREQ_LOW_CHANNELS", DEFAULT_LOW_FREQ_CHANNELS)
    keep_channels = get_keep_channels(
        mode=cfg.get("PREPROCESS_MODE", "high_freq"),
        keep_channels=cfg.get("FREQ_KEEP_CHANNELS", None),
        low_freq_channels=low_channels,
    )
    per_color = len(keep_channels)

    rows = []
    for pos, dct_index in enumerate(keep_channels):
        rgb_scores = [float(gate_values[pos + color_idx * per_color]) for color_idx in range(3)]
        rows.append({
            "dct_index": dct_index,
            "mean_score": sum(rgb_scores) / len(rgb_scores),
            "score_c0": rgb_scores[0],
            "score_c1": rgb_scores[1],
            "score_c2": rgb_scores[2],
        })
    rows.sort(key=lambda row: row["mean_score"], reverse=True)

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "dct_index", "mean_score", "score_c0", "score_c1", "score_c2"])
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow({"rank": rank, **row})

    print("Saved gate scores to {}".format(args.output))
    print("Top-10 DCT channels:", [row["dct_index"] for row in rows[:10]])


if __name__ == "__main__":
    main()
