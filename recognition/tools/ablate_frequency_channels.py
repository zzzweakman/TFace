import argparse
import csv
import glob
import os
import sys

import torch
import torch.distributed as dist

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))
from frequency_utils import DEFAULT_LOW_FREQ_CHANNELS, get_keep_channels
from torchkit.util import AverageMeter, CkptLoader, accuracy_dist
from train import TrainTask


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DCT channel importance by masking one frequency index at a time.")
    parser.add_argument("--config", required=True, help="Recognition config.")
    parser.add_argument("--model-root", default="", help="Directory containing Backbone/HEAD checkpoints.")
    parser.add_argument("--epoch", type=int, default=-1, help="Checkpoint epoch; defaults to latest.")
    parser.add_argument("--backbone", default="", help="Explicit backbone checkpoint path.")
    parser.add_argument("--head-prefix", default="", help="Explicit head checkpoint prefix before _Split_0_checkpoint.pth.")
    parser.add_argument("--output", default="frequency_ablation_scores.csv", help="CSV output path.")
    parser.add_argument("--max-batches", type=int, default=20, help="Number of batches to evaluate; -1 for full loader.")
    return parser.parse_args()


def ensure_single_process_env():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29599")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")


def latest_epoch(model_root):
    paths = glob.glob(os.path.join(model_root, "Backbone_Epoch_*_checkpoint.pth"))
    if not paths:
        raise RuntimeError("No Backbone_Epoch_* checkpoint found under {}".format(model_root))

    def parse_epoch(path):
        name = os.path.basename(path)
        return int(name.split("Backbone_Epoch_", 1)[1].split("_checkpoint.pth", 1)[0])

    return max(parse_epoch(path) for path in paths)


def checkpoint_paths(cfg, args):
    model_root = args.model_root or cfg["MODEL_ROOT"]
    epoch = args.epoch if args.epoch > 0 else latest_epoch(model_root)
    backbone = args.backbone or os.path.join(model_root, "Backbone_Epoch_{}_checkpoint.pth".format(epoch))
    head_prefix = args.head_prefix or os.path.join(model_root, "HEAD_Epoch_{}".format(epoch))
    return backbone, head_prefix, epoch


def evaluate(task, masks, max_batches):
    meters = {name: {"top1": AverageMeter(), "top5": AverageMeter()} for name in masks}
    head = list(task.heads.values())[0]
    batch_size = task.batch_sizes[0]
    class_shard = task.class_shards[0]

    task.backbone.eval()
    head.eval()
    with torch.no_grad():
        for step, samples in enumerate(task.train_loader):
            if max_batches > 0 and step >= max_batches:
                break
            inputs = samples[0].cuda(non_blocking=True)
            labels = samples[1].cuda(non_blocking=True)
            freq_inputs = task.preprocess_inputs(inputs)

            for name, mask in masks.items():
                masked_inputs = freq_inputs if mask is None else freq_inputs * mask
                features = task.backbone(masked_inputs)
                outputs, _, original_outputs = task.general_head_forward(head, features, labels)
                del outputs
                prec1, prec5 = accuracy_dist(task.cfg, original_outputs.data, labels, class_shard, topk=(1, 5))
                current_batch = min(batch_size, labels.size(0))
                meters[name]["top1"].update(prec1.data.item(), current_batch)
                meters[name]["top5"].update(prec5.data.item(), current_batch)
    return meters


def reduce_meter(meter):
    values = torch.tensor([meter.sum, meter.count], dtype=torch.float64, device="cuda")
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    total_sum = values[0].item()
    total_count = max(values[1].item(), 1.0)
    return total_sum / total_count


def main():
    args = parse_args()
    ensure_single_process_env()
    task = TrainTask(args.config)
    task.init_env()
    task.prepare()

    backbone_ckpt, head_prefix, epoch = checkpoint_paths(task.cfg, args)
    CkptLoader.load_backbone(task.backbone, backbone_ckpt, task.local_rank)
    CkptLoader.load_head(task.heads, head_prefix, task.dist_fc, task.rank)

    keep_channels = get_keep_channels(
        mode=task.cfg.get("PREPROCESS_MODE", "high_freq"),
        keep_channels=task.cfg.get("FREQ_KEEP_CHANNELS", None),
        low_freq_channels=task.cfg.get("FREQ_LOW_CHANNELS", DEFAULT_LOW_FREQ_CHANNELS),
    )
    per_color = len(keep_channels)
    device = torch.device("cuda")

    masks = {"baseline": None}
    for pos, dct_index in enumerate(keep_channels):
        mask = torch.ones(1, task.cfg["INPUT_CHANNELS"], 1, 1, device=device)
        for color_idx in range(3):
            mask[:, pos + color_idx * per_color, :, :] = 0
        masks["dct_{}".format(dct_index)] = mask

    meters = evaluate(task, masks, args.max_batches)
    baseline_top1 = reduce_meter(meters["baseline"]["top1"])
    baseline_top5 = reduce_meter(meters["baseline"]["top5"])

    rows = []
    for dct_index in keep_channels:
        name = "dct_{}".format(dct_index)
        top1 = reduce_meter(meters[name]["top1"])
        top5 = reduce_meter(meters[name]["top5"])
        rows.append({
            "dct_index": dct_index,
            "baseline_top1": baseline_top1,
            "ablated_top1": top1,
            "top1_drop": baseline_top1 - top1,
            "baseline_top5": baseline_top5,
            "ablated_top5": top5,
            "top5_drop": baseline_top5 - top5,
        })
    rows.sort(key=lambda row: row["top1_drop"], reverse=True)

    if task.rank == 0:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "rank",
                    "dct_index",
                    "baseline_top1",
                    "ablated_top1",
                    "top1_drop",
                    "baseline_top5",
                    "ablated_top5",
                    "top5_drop",
                ],
            )
            writer.writeheader()
            for rank, row in enumerate(rows, start=1):
                writer.writerow({"rank": rank, **row})

        print("Evaluated epoch {} from {}".format(epoch, task.cfg["MODEL_ROOT"]))
        print("Saved ablation scores to {}".format(args.output))
        print("Top-10 important DCT channels:", [row["dct_index"] for row in rows[:10]])

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
