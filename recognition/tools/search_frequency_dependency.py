import argparse
import csv
import json
import os
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RECOGNITION_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..")
TEST_DIR = os.path.join(RECOGNITION_DIR, "test")
sys.path.append(RECOGNITION_DIR)
sys.path.append(TEST_DIR)

from frequency_utils import DEFAULT_LOW_FREQ_CHANNELS, frequency_tensor_from_images, get_keep_channels
from utils import evaluate, get_val_pair_from_bin
from torchkit.backbone import get_model
from torchkit.backbone.channel_gate import FrequencyChannelGate
from torchkit.util.utils import load_config


DATASET_NAMES = ["lfw", "cfp_fp", "agedb_30", "calfw", "cplfw"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Search which high-frequency DCT channels a recognition model depends on, and compare against gate ranks."
    )
    parser.add_argument("--target-config", required=True, help="Config for the model being analyzed.")
    parser.add_argument("--target-ckpt", required=True, help="Backbone checkpoint for the model being analyzed.")
    parser.add_argument("--data-root", required=True, help="Directory containing lfw.bin/cfp_fp.bin/... validation bins.")
    parser.add_argument("--output-dir", required=True, help="Directory to save CSV/JSON reports.")
    parser.add_argument("--gpu-ids", default="0", help="Logical GPU ids relative to current CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Verification batch size.")
    parser.add_argument(
        "--candidate-batch-size",
        type=int,
        default=1,
        help="How many channel masks to evaluate together in one forward. Increase only if memory allows.",
    )
    parser.add_argument("--backbone", default="", help="Override backbone name. Defaults to BACKBONE_NAME in config.")
    parser.add_argument(
        "--embedding-size",
        type=int,
        default=0,
        help="Override embedding size. Defaults to EMBEDDING_SIZE in config.",
    )
    parser.add_argument(
        "--datasets",
        default=",".join(DATASET_NAMES),
        help="Comma-separated validation datasets chosen from: {}.".format(",".join(DATASET_NAMES)),
    )
    parser.add_argument(
        "--topk-list",
        default="",
        help="Optional comma-separated k list for keep-topk searches, for example 4,8,12,16,24,32,40,54.",
    )
    parser.add_argument(
        "--gate-config",
        default="",
        help="Optional gated-model config used to load gate ranks from checkpoint when --gate-scores is not provided.",
    )
    parser.add_argument(
        "--gate-ckpt",
        default="",
        help="Optional gated-model backbone checkpoint used to load gate ranks when --gate-scores is not provided.",
    )
    parser.add_argument(
        "--gate-scores",
        default="",
        help="Optional pre-exported gate score CSV with columns rank,dct_index,mean_score,...",
    )
    parser.add_argument(
        "--disable-tta",
        action="store_true",
        help="Disable horizontal-flip test-time augmentation.",
    )
    return parser.parse_args()


def strip_module_prefix(state_dict):
    if any(key.startswith("module.") for key in state_dict.keys()):
        return {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    return state_dict


def load_state_dict(path):
    return strip_module_prefix(torch.load(path, map_location="cpu"))


def is_gated_checkpoint(state_dict):
    return "gate" in state_dict and any(key.startswith("backbone.") for key in state_dict.keys())


def build_model(cfg, ckpt_path, backbone_override="", embedding_size_override=0):
    input_size = cfg["INPUT_SIZE"]
    input_channels = cfg["INPUT_CHANNELS"]
    backbone_name = backbone_override or cfg["BACKBONE_NAME"]
    backbone = get_model(backbone_name)(input_size, input_channel=input_channels)
    state_dict = load_state_dict(ckpt_path)
    if is_gated_checkpoint(state_dict):
        model = FrequencyChannelGate(
            backbone,
            num_channels=input_channels,
            init_value=cfg.get("FREQ_CHANNEL_GATE_INIT", 0.99),
            use_sigmoid=cfg.get("FREQ_CHANNEL_GATE_SIGMOID", True),
        )
    else:
        model = backbone
    model.load_state_dict(state_dict)
    embedding_size = embedding_size_override or cfg.get("EMBEDDING_SIZE", 512)
    return model, backbone_name, embedding_size


def prepare_model_for_gpus(model, gpu_ids):
    gpu_ids = [int(item) for item in gpu_ids.split(",") if item.strip()]
    if not gpu_ids:
        raise RuntimeError("No GPU ids provided")
    visible_gpu_count = torch.cuda.device_count()
    print("CUDA_VISIBLE_DEVICES={}".format(os.environ.get("CUDA_VISIBLE_DEVICES", "")))
    print("Visible GPU count: {}".format(visible_gpu_count))
    if max(gpu_ids) >= visible_gpu_count:
        raise RuntimeError(
            "Requested gpu id {} exceeds visible gpu count {}. Check CUDA_VISIBLE_DEVICES and --gpu-ids.".format(
                max(gpu_ids), visible_gpu_count
            )
        )
    if len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        print("Running with DataParallel on {} GPUs".format(len(gpu_ids)))
    else:
        print("Running on single GPU")
    return model.cuda(), gpu_ids


def parse_csv_int_list(spec):
    if spec is None:
        return []
    spec = str(spec).strip()
    if spec == "":
        return []
    return [int(item.strip()) for item in spec.split(",") if item.strip()]


def load_validation_sets(data_root, dataset_names):
    datasets = OrderedDict()
    for name in dataset_names:
        images, issame = get_val_pair_from_bin(data_root, "{}.bin".format(name))
        datasets[name] = (images, issame)
    return datasets


def get_high_freq_channels_from_cfg(cfg):
    keep_channels = get_keep_channels(
        mode=cfg.get("PREPROCESS_MODE", "high_freq"),
        keep_channels=cfg.get("FREQ_KEEP_CHANNELS", None),
        low_freq_channels=cfg.get("FREQ_LOW_CHANNELS", DEFAULT_LOW_FREQ_CHANNELS),
    )
    if keep_channels is None:
        raise RuntimeError("Config does not use frequency channels: {}".format(cfg.get("PREPROCESS_MODE", "rgb")))
    return keep_channels


def build_preprocess_fn(cfg):
    preprocess_mode = cfg.get("PREPROCESS_MODE", "rgb")
    keep_channels = cfg.get("FREQ_KEEP_CHANNELS", None)
    low_freq_channels = cfg.get("FREQ_LOW_CHANNELS", DEFAULT_LOW_FREQ_CHANNELS)
    ratio = cfg.get("DCT_SAMPLING_RATIO", 8)
    if str(preprocess_mode).lower() == "rgb":
        return None
    return lambda inputs: frequency_tensor_from_images(
        inputs,
        mode=preprocess_mode,
        keep_channels=keep_channels,
        low_freq_channels=low_freq_channels,
        ratio=ratio,
    )


class PreprocessBackboneWrapper(nn.Module):
    def __init__(self, backbone, preprocess_fn, ordered_dct_indices):
        super().__init__()
        self.backbone = backbone
        self.preprocess_fn = preprocess_fn
        self.ordered_dct_indices = [int(item) for item in ordered_dct_indices]
        self.pos_by_dct = {int(dct_index): pos for pos, dct_index in enumerate(self.ordered_dct_indices)}

    def build_mask(self, inputs, keep_dct_indices=None, drop_dct_index=None):
        channels = len(self.ordered_dct_indices)
        mask = torch.ones((1, channels * 3, 1, 1), dtype=inputs.dtype, device=inputs.device)
        if keep_dct_indices is not None:
            keep_set = {int(item) for item in keep_dct_indices}
            mask.zero_()
            for dct_index in keep_set:
                pos = self.pos_by_dct[int(dct_index)]
                for color_idx in range(3):
                    mask[:, pos + color_idx * channels, :, :] = 1.0
            return mask
        if drop_dct_index is not None:
            pos = self.pos_by_dct[int(drop_dct_index)]
            for color_idx in range(3):
                mask[:, pos + color_idx * channels, :, :] = 0.0
        return mask

    def forward(self, inputs, keep_dct_indices=None, drop_dct_index=None):
        if self.preprocess_fn is not None:
            inputs = self.preprocess_fn(inputs)
        if keep_dct_indices is not None or drop_dct_index is not None:
            inputs = inputs * self.build_mask(
                inputs,
                keep_dct_indices=keep_dct_indices,
                drop_dct_index=drop_dct_index,
            )
        return self.backbone(inputs)


def batched(iterable, batch_size):
    for start in range(0, len(iterable), batch_size):
        yield iterable[start:start + batch_size]


def accuracy_and_threshold(embeddings, issame):
    _, _, accuracy, best_thresholds, _ = evaluate(embeddings, issame, nrof_folds=10, pca=0)
    return float(accuracy.mean()), float(best_thresholds.mean())


def evaluate_candidates_on_dataset(
    model,
    images,
    issame,
    candidate_specs,
    embedding_size,
    batch_size,
    candidate_batch_size,
    use_tta,
):
    if candidate_batch_size != 1:
        raise RuntimeError(
            "candidate_batch_size > 1 is not supported in the current multi-GPU path. "
            "Please use --candidate-batch-size 1."
        )
    model.eval()
    results = {}
    total_images = len(images)
    with torch.no_grad():
        for candidate_group in batched(candidate_specs, candidate_batch_size):
            candidate = candidate_group[0]
            group_embeddings = np.zeros((len(candidate_group), total_images, embedding_size), dtype=np.float32)
            for start_idx in range(0, total_images, batch_size):
                end_idx = min(start_idx + batch_size, total_images)
                batch = torch.from_numpy(np.asarray(images[start_idx:end_idx], dtype=np.float32)).cuda(non_blocking=True)
                if use_tta:
                    flip_batch = torch.flip(batch, dims=[3])
                    group_orig = model(
                        batch,
                        keep_dct_indices=candidate.get("keep_dct_indices"),
                        drop_dct_index=candidate.get("drop_dct_index"),
                    ).unsqueeze(0)
                    group_flip = model(
                        flip_batch,
                        keep_dct_indices=candidate.get("keep_dct_indices"),
                        drop_dct_index=candidate.get("drop_dct_index"),
                    ).unsqueeze(0)
                    group_output = F.normalize(group_orig + group_flip, dim=-1)
                else:
                    group_output = F.normalize(
                        model(
                            batch,
                            keep_dct_indices=candidate.get("keep_dct_indices"),
                            drop_dct_index=candidate.get("drop_dct_index"),
                        ).unsqueeze(0),
                        dim=-1,
                    )
                group_embeddings[:, start_idx:end_idx, :] = group_output.detach().cpu().numpy()
            for idx, item in enumerate(candidate_group):
                acc, threshold = accuracy_and_threshold(group_embeddings[idx], issame)
                results[item["name"]] = {
                    "acc": acc,
                    "threshold": threshold,
                }
    return results


def evaluate_candidates_across_datasets(
    model,
    datasets,
    candidate_specs,
    embedding_size,
    batch_size,
    candidate_batch_size,
    use_tta,
):
    all_results = {}
    for dataset_name, (images, issame) in datasets.items():
        print("Evaluating {} candidates on {} ({} images)".format(len(candidate_specs), dataset_name, len(images)))
        dataset_results = evaluate_candidates_on_dataset(
            model=model,
            images=images,
            issame=issame,
            candidate_specs=candidate_specs,
            embedding_size=embedding_size,
            batch_size=batch_size,
            candidate_batch_size=candidate_batch_size,
            use_tta=use_tta,
        )
        for candidate_name, metrics in dataset_results.items():
            all_results.setdefault(candidate_name, {})[dataset_name] = metrics
    return all_results


def summarize_candidate_results(all_results, dataset_names, baseline_name="baseline"):
    summary = {}
    baseline_metrics = all_results.get(baseline_name) if baseline_name is not None else None
    for candidate_name, dataset_metrics in all_results.items():
        per_dataset = {}
        acc_values = []
        drop_values = []
        threshold_values = []
        for dataset_name in dataset_names:
            acc = float(dataset_metrics[dataset_name]["acc"])
            threshold = float(dataset_metrics[dataset_name]["threshold"])
            if baseline_metrics is None:
                drop = 0.0
            else:
                baseline_acc = float(baseline_metrics[dataset_name]["acc"])
                drop = baseline_acc - acc
            per_dataset[dataset_name] = {
                "acc": acc,
                "threshold": threshold,
                "drop": drop,
            }
            acc_values.append(acc)
            drop_values.append(drop)
            threshold_values.append(threshold)
        summary[candidate_name] = {
            "mean_acc": float(np.mean(acc_values)),
            "mean_drop": float(np.mean(drop_values)),
            "mean_threshold": float(np.mean(threshold_values)),
            "per_dataset": per_dataset,
        }
    return summary


def read_gate_scores_from_csv(path):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "dct_index": int(row["dct_index"]),
                    "mean_score": float(row["mean_score"]),
                    "rank": int(row["rank"]),
                    "score_c0": float(row.get("score_c0", 0.0)),
                    "score_c1": float(row.get("score_c1", 0.0)),
                    "score_c2": float(row.get("score_c2", 0.0)),
                }
            )
    rows.sort(key=lambda item: item["mean_score"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def extract_gate_scores_from_checkpoint(cfg, ckpt_path):
    state_dict = load_state_dict(ckpt_path)
    if not is_gated_checkpoint(state_dict):
        raise RuntimeError("Checkpoint does not look like a gated checkpoint: {}".format(ckpt_path))
    backbone = get_model(cfg["BACKBONE_NAME"])(cfg["INPUT_SIZE"], input_channel=cfg["INPUT_CHANNELS"])
    model = FrequencyChannelGate(
        backbone,
        num_channels=cfg["INPUT_CHANNELS"],
        init_value=cfg.get("FREQ_CHANNEL_GATE_INIT", 0.99),
        use_sigmoid=cfg.get("FREQ_CHANNEL_GATE_SIGMOID", True),
    )
    model.load_state_dict(state_dict)
    gate_values = model.gate_values().detach().cpu().numpy()

    keep_channels = get_high_freq_channels_from_cfg(cfg)
    per_color = len(keep_channels)
    rows = []
    for pos, dct_index in enumerate(keep_channels):
        rgb_scores = [float(gate_values[pos + color_idx * per_color]) for color_idx in range(3)]
        rows.append(
            {
                "dct_index": int(dct_index),
                "mean_score": float(sum(rgb_scores) / len(rgb_scores)),
                "score_c0": rgb_scores[0],
                "score_c1": rgb_scores[1],
                "score_c2": rgb_scores[2],
            }
        )
    rows.sort(key=lambda item: item["mean_score"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def load_gate_rows(args):
    if args.gate_scores:
        return read_gate_scores_from_csv(args.gate_scores)
    if args.gate_config and args.gate_ckpt:
        gate_cfg = load_config(args.gate_config)
        return extract_gate_scores_from_checkpoint(gate_cfg, args.gate_ckpt)
    return []


def rankdata_desc(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        avg_rank = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def spearman_corr_desc(values_a, values_b):
    if len(values_a) < 2:
        return None
    ranks_a = rankdata_desc(values_a)
    ranks_b = rankdata_desc(values_b)
    if np.std(ranks_a) == 0 or np.std(ranks_b) == 0:
        return None
    corr = np.corrcoef(ranks_a, ranks_b)[0, 1]
    return float(corr)


def build_ablation_rows(summary, ordered_dct_indices):
    rows = []
    for dct_index in ordered_dct_indices:
        candidate_name = "drop_{}".format(dct_index)
        item = summary[candidate_name]
        row = {
            "dct_index": int(dct_index),
            "mean_acc": item["mean_acc"],
            "mean_drop": item["mean_drop"],
            "mean_threshold": item["mean_threshold"],
        }
        for dataset_name, metrics in item["per_dataset"].items():
            row["{}_acc".format(dataset_name)] = metrics["acc"]
            row["{}_drop".format(dataset_name)] = metrics["drop"]
        rows.append(row)
    rows.sort(key=lambda item: item["mean_drop"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def build_ranking_comparison_rows(ablation_rows, gate_rows):
    gate_by_dct = {row["dct_index"]: row for row in gate_rows}
    rows = []
    for row in ablation_rows:
        gate_row = gate_by_dct.get(row["dct_index"])
        merged = dict(row)
        if gate_row is not None:
            merged["gate_rank"] = gate_row["rank"]
            merged["gate_mean_score"] = gate_row["mean_score"]
            merged["gate_score_c0"] = gate_row.get("score_c0", 0.0)
            merged["gate_score_c1"] = gate_row.get("score_c1", 0.0)
            merged["gate_score_c2"] = gate_row.get("score_c2", 0.0)
        else:
            merged["gate_rank"] = ""
            merged["gate_mean_score"] = ""
            merged["gate_score_c0"] = ""
            merged["gate_score_c1"] = ""
            merged["gate_score_c2"] = ""
        rows.append(merged)
    return rows


def overlap_at_k(rank_rows_a, rank_rows_b, k):
    top_a = {row["dct_index"] for row in rank_rows_a[:k]}
    top_b = {row["dct_index"] for row in rank_rows_b[:k]}
    return sorted(top_a & top_b)


def build_keep_topk_candidates(prefix, ranked_dct_indices, topk_list):
    candidates = []
    total_channels = len(ranked_dct_indices)
    for k in topk_list:
        if k <= 0 or k > total_channels:
            continue
        keep_indices = ranked_dct_indices[:k]
        candidates.append(
            {
                "name": "{}_top{}".format(prefix, k),
                "type": "keep_topk",
                "ranking_source": prefix,
                "topk": int(k),
                "keep_dct_indices": [int(item) for item in keep_indices],
            }
        )
    return candidates


def save_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dataset_names = [item.strip() for item in args.datasets.split(",") if item.strip()]
    invalid = [item for item in dataset_names if item not in DATASET_NAMES]
    if invalid:
        raise RuntimeError("Unsupported dataset names: {}".format(invalid))

    target_cfg = load_config(args.target_config)
    ordered_dct_indices = get_high_freq_channels_from_cfg(target_cfg)

    model, backbone_name, embedding_size = build_model(
        cfg=target_cfg,
        ckpt_path=args.target_ckpt,
        backbone_override=args.backbone,
        embedding_size_override=args.embedding_size,
    )
    preprocess_fn = build_preprocess_fn(target_cfg)
    model = PreprocessBackboneWrapper(model, preprocess_fn, ordered_dct_indices)
    model, gpu_ids = prepare_model_for_gpus(model, args.gpu_ids)
    model.eval()
    print("Using GPUs: {}".format(gpu_ids))
    print("Backbone: {}, embedding size: {}".format(backbone_name, embedding_size))
    print("Target checkpoint: {}".format(args.target_ckpt))

    datasets = load_validation_sets(args.data_root, dataset_names)
    ablation_candidates = [{"name": "baseline", "type": "baseline"}]
    for dct_index in ordered_dct_indices:
        ablation_candidates.append(
            {
                "name": "drop_{}".format(dct_index),
                "type": "drop_one",
                "drop_dct_index": int(dct_index),
            }
        )

    ablation_results = evaluate_candidates_across_datasets(
        model=model,
        datasets=datasets,
        candidate_specs=ablation_candidates,
        embedding_size=embedding_size,
        batch_size=args.batch_size,
        candidate_batch_size=args.candidate_batch_size,
        use_tta=not args.disable_tta,
    )
    ablation_summary = summarize_candidate_results(ablation_results, dataset_names)
    baseline_summary = ablation_summary["baseline"]
    ablation_rows = build_ablation_rows(ablation_summary, ordered_dct_indices)

    ablation_csv = os.path.join(args.output_dir, "ablation_leave_one_out.csv")
    ablation_fieldnames = [
        "rank",
        "dct_index",
        "mean_acc",
        "mean_drop",
        "mean_threshold",
    ]
    for dataset_name in dataset_names:
        ablation_fieldnames.extend(["{}_acc".format(dataset_name), "{}_drop".format(dataset_name)])
    save_csv(ablation_csv, ablation_rows, ablation_fieldnames)

    gate_rows = load_gate_rows(args)
    ranking_comparison_rows = build_ranking_comparison_rows(ablation_rows, gate_rows)
    comparison_csv = os.path.join(args.output_dir, "ranking_comparison.csv")
    comparison_fieldnames = list(ablation_fieldnames) + [
        "gate_rank",
        "gate_mean_score",
        "gate_score_c0",
        "gate_score_c1",
        "gate_score_c2",
    ]
    save_csv(comparison_csv, ranking_comparison_rows, comparison_fieldnames)

    topk_results_rows = []
    topk_list = sorted(set(parse_csv_int_list(args.topk_list)))
    if topk_list:
        keep_topk_candidates = []
        ablation_ranked_indices = [row["dct_index"] for row in ablation_rows]
        keep_topk_candidates.extend(
            build_keep_topk_candidates("ablation", ablation_ranked_indices, topk_list)
        )
        if gate_rows:
            gate_ranked_indices = [row["dct_index"] for row in gate_rows]
            keep_topk_candidates.extend(
                build_keep_topk_candidates("gate", gate_ranked_indices, topk_list)
            )

        if keep_topk_candidates:
            keep_topk_results = evaluate_candidates_across_datasets(
                model=model,
                datasets=datasets,
                candidate_specs=keep_topk_candidates,
                embedding_size=embedding_size,
                batch_size=args.batch_size,
                candidate_batch_size=args.candidate_batch_size,
                use_tta=not args.disable_tta,
            )
            keep_topk_summary = summarize_candidate_results(keep_topk_results, dataset_names, baseline_name=None)
            for candidate in keep_topk_candidates:
                item = keep_topk_summary[candidate["name"]]
                row = {
                    "candidate": candidate["name"],
                    "ranking_source": candidate["ranking_source"],
                    "topk": candidate["topk"],
                    "mean_acc": item["mean_acc"],
                    "mean_drop_vs_full": baseline_summary["mean_acc"] - item["mean_acc"],
                    "keep_dct_indices": ",".join(str(idx) for idx in candidate["keep_dct_indices"]),
                }
                for dataset_name, metrics in item["per_dataset"].items():
                    row["{}_acc".format(dataset_name)] = metrics["acc"]
                    row["{}_drop_vs_full".format(dataset_name)] = (
                        baseline_summary["per_dataset"][dataset_name]["acc"] - metrics["acc"]
                    )
                topk_results_rows.append(row)
            topk_results_rows.sort(key=lambda item: (item["topk"], item["ranking_source"]))
            topk_csv = os.path.join(args.output_dir, "keep_topk_search.csv")
            topk_fieldnames = [
                "candidate",
                "ranking_source",
                "topk",
                "mean_acc",
                "mean_drop_vs_full",
                "keep_dct_indices",
            ]
            for dataset_name in dataset_names:
                topk_fieldnames.extend(["{}_acc".format(dataset_name), "{}_drop_vs_full".format(dataset_name)])
            save_csv(topk_csv, topk_results_rows, topk_fieldnames)

    spearman = None
    overlap = {}
    if gate_rows:
        gate_score_by_dct = {row["dct_index"]: row["mean_score"] for row in gate_rows}
        common_rows = [row for row in ablation_rows if row["dct_index"] in gate_score_by_dct]
        spearman = spearman_corr_desc(
            [row["mean_drop"] for row in common_rows],
            [gate_score_by_dct[row["dct_index"]] for row in common_rows],
        )
        for k in (5, 10, 20):
            overlap[str(k)] = overlap_at_k(ablation_rows, gate_rows, min(k, len(ablation_rows)))

    summary = {
        "target": {
            "config": args.target_config,
            "checkpoint": args.target_ckpt,
            "backbone": backbone_name,
            "embedding_size": embedding_size,
            "gpu_ids": gpu_ids,
            "batch_size": args.batch_size,
            "candidate_batch_size": args.candidate_batch_size,
            "datasets": dataset_names,
            "tta": not args.disable_tta,
        },
        "baseline": {
            "mean_acc": baseline_summary["mean_acc"],
            "per_dataset": {
                dataset_name: baseline_summary["per_dataset"][dataset_name]["acc"] for dataset_name in dataset_names
            },
        },
        "ablation_top10": ablation_rows[:10],
        "gate_top10": gate_rows[:10],
        "comparison": {
            "spearman_drop_vs_gate_score": spearman,
            "topk_overlap": overlap,
        },
        "topk_search": topk_results_rows,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Baseline mean accuracy: {:.6f}".format(baseline_summary["mean_acc"]))
    print("Top-10 leave-one-out important DCT channels: {}".format([row["dct_index"] for row in ablation_rows[:10]]))
    if gate_rows:
        print("Top-10 gate-ranked DCT channels: {}".format([row["dct_index"] for row in gate_rows[:10]]))
        print("Spearman(drop, gate_score): {}".format("None" if spearman is None else "{:.6f}".format(spearman)))
    print("Saved reports to {}".format(args.output_dir))


if __name__ == "__main__":
    main()
