import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Plot reports for search_frequency_dependency outputs.")
    parser.add_argument("--result-dir", required=True, help="Directory containing summary.json and CSV outputs.")
    parser.add_argument("--topn", type=int, default=15, help="How many most important channels to emphasize.")
    return parser.parse_args()


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_csv(path):
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def to_float(value):
    if value in ("", None):
        return None
    return float(value)


def to_int(value):
    if value in ("", None):
        return None
    return int(value)


def parse_ranking_rows(rows):
    parsed = []
    for row in rows:
        item = dict(row)
        item["rank"] = int(row["rank"])
        item["dct_index"] = int(row["dct_index"])
        item["mean_acc"] = float(row["mean_acc"])
        item["mean_drop"] = float(row["mean_drop"])
        item["mean_threshold"] = float(row["mean_threshold"])
        item["gate_rank"] = to_int(row.get("gate_rank"))
        item["gate_mean_score"] = to_float(row.get("gate_mean_score"))
        for key, value in row.items():
            if key.endswith("_acc") or key.endswith("_drop"):
                item[key] = float(value)
        parsed.append(item)
    return parsed


def parse_topk_rows(rows):
    parsed = []
    for row in rows:
        item = dict(row)
        item["topk"] = int(row["topk"])
        item["mean_acc"] = float(row["mean_acc"])
        item["mean_drop_vs_full"] = float(row["mean_drop_vs_full"])
        for key, value in row.items():
            if key.endswith("_acc") or key.endswith("_drop_vs_full"):
                item[key] = float(value)
        parsed.append(item)
    return parsed


def infer_dataset_names(summary):
    return list(summary["baseline"]["per_dataset"].keys())


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def set_plot_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.dpi"] = 160
    plt.rcParams["savefig.dpi"] = 200
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 11


def plot_ablation_bar(rows, output_path, topn):
    top_rows = rows[:topn]
    labels = [f"DCT {row['dct_index']}" for row in top_rows][::-1]
    values = [row["mean_drop"] for row in top_rows][::-1]
    gate_ranks = [row["gate_rank"] for row in top_rows][::-1]

    colors = []
    for gate_rank in gate_ranks:
        if gate_rank is None:
            colors.append("#9aa0a6")
        elif gate_rank <= 10:
            colors.append("#d1495b")
        elif gate_rank <= 20:
            colors.append("#edae49")
        else:
            colors.append("#4e79a7")

    fig, ax = plt.subplots(figsize=(10, max(5, 0.42 * len(top_rows) + 1.5)))
    bars = ax.barh(labels, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_title("Leave-One-Out Importance")
    ax.set_xlabel("Mean accuracy drop after removing the channel")

    for bar, row in zip(bars, top_rows[::-1]):
        text = f"{row['mean_drop']:.4f}"
        if row["gate_rank"] is not None:
            text += f" | gate#{row['gate_rank']}"
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2, text, va="center")

    ax.text(
        0.99,
        0.02,
        "red: gate top-10 | yellow: gate 11-20 | blue: gate >20",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_drop_vs_gate_score(rows, spearman_value, output_path, topn):
    xs = [row["gate_mean_score"] for row in rows if row["gate_mean_score"] is not None]
    ys = [row["mean_drop"] for row in rows if row["gate_mean_score"] is not None]
    filtered = [row for row in rows if row["gate_mean_score"] is not None]

    fig, ax = plt.subplots(figsize=(8.8, 6.8))
    ax.scatter(xs, ys, s=40, color="#4e79a7", alpha=0.75, edgecolor="white", linewidth=0.5)

    if len(xs) >= 2:
        coeff = np.polyfit(xs, ys, deg=1)
        line_x = np.linspace(min(xs), max(xs), 200)
        line_y = coeff[0] * line_x + coeff[1]
        ax.plot(line_x, line_y, color="#d1495b", linewidth=1.5, linestyle="--")

    label_candidates = sorted(filtered, key=lambda item: item["mean_drop"], reverse=True)[:topn]
    gate_top = sorted(filtered, key=lambda item: item["gate_rank"])[: min(8, len(filtered))]
    label_ids = {row["dct_index"] for row in label_candidates} | {row["dct_index"] for row in gate_top}
    for row in filtered:
        if row["dct_index"] not in label_ids:
            continue
        ax.annotate(
            str(row["dct_index"]),
            (row["gate_mean_score"], row["mean_drop"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_title("Channel Dependency vs Gate Score")
    ax.set_xlabel("Gate mean score")
    ax.set_ylabel("Mean accuracy drop after ablation")
    ax.text(
        0.98,
        0.96,
        f"Spearman = {spearman_value:.3f}" if spearman_value is not None else "Spearman = N/A",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "boxstyle": "round,pad=0.25"},
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_dataset_heatmap(rows, dataset_names, output_path, topn):
    top_rows = rows[:topn]
    matrix = np.array([[row[f"{dataset}_drop"] for dataset in dataset_names] for row in top_rows], dtype=np.float64)

    fig_w = 1.2 * len(dataset_names) + 4.0
    fig_h = 0.45 * len(top_rows) + 2.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    vmax = max(np.max(np.abs(matrix)), 1e-8)
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=min(0.0, matrix.min()), vmax=vmax)

    ax.set_xticks(np.arange(len(dataset_names)))
    ax.set_xticklabels(dataset_names, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(top_rows)))
    ax.set_yticklabels([f"DCT {row['dct_index']}" for row in top_rows])
    ax.set_title("Per-Dataset Sensitivity for Top Ablated Channels")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "black" if matrix[i, j] < vmax * 0.55 else "white"
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=8, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cbar.set_label("Accuracy drop")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_topk_curves(rows, baseline_mean_acc, output_path):
    ablation_rows = sorted([row for row in rows if row["ranking_source"] == "ablation"], key=lambda item: item["topk"])
    gate_rows = sorted([row for row in rows if row["ranking_source"] == "gate"], key=lambda item: item["topk"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    for label, items, color in [
        ("Ablation rank", ablation_rows, "#1b9e77"),
        ("Gate rank", gate_rows, "#d95f02"),
    ]:
        if not items:
            continue
        ks = [item["topk"] for item in items]
        mean_acc = [item["mean_acc"] for item in items]
        mean_drop = [item["mean_drop_vs_full"] for item in items]
        axes[0].plot(ks, mean_acc, marker="o", linewidth=2, color=color, label=label)
        axes[1].plot(ks, mean_drop, marker="o", linewidth=2, color=color, label=label)

    axes[0].axhline(baseline_mean_acc, color="#444444", linestyle="--", linewidth=1, label="Full model")
    axes[0].set_title("Keep-TopK Mean Accuracy")
    axes[0].set_xlabel("Top-K channels kept")
    axes[0].set_ylabel("Mean verification accuracy")
    axes[0].legend()

    axes[1].axhline(0.0, color="#444444", linestyle="--", linewidth=1)
    axes[1].set_title("Keep-TopK Accuracy Drop vs Full Model")
    axes[1].set_xlabel("Top-K channels kept")
    axes[1].set_ylabel("Mean accuracy drop")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_rank_scatter(rows, output_path, topn):
    filtered = [row for row in rows if row["gate_rank"] is not None]
    x = [row["gate_rank"] for row in filtered]
    y = [row["rank"] for row in filtered]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x, y, s=36, color="#4e79a7", alpha=0.75, edgecolor="white", linewidth=0.5)
    max_rank = max(max(x), max(y))
    ax.plot([1, max_rank], [1, max_rank], linestyle="--", color="#d1495b", linewidth=1.2)

    label_candidates = filtered[:topn] + sorted(filtered, key=lambda item: item["gate_rank"])[: min(8, len(filtered))]
    seen = set()
    for row in label_candidates:
        if row["dct_index"] in seen:
            continue
        seen.add(row["dct_index"])
        ax.annotate(
            str(row["dct_index"]),
            (row["gate_rank"], row["rank"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xlim(0, max_rank + 2)
    ax.set_ylim(max_rank + 2, 0)
    ax.set_title("Ablation Rank vs Gate Rank")
    ax.set_xlabel("Gate rank (smaller is stronger)")
    ax.set_ylabel("Ablation rank (smaller is more important)")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def build_report(summary, ranking_rows, topk_rows, output_path):
    baseline = summary["baseline"]
    comparison = summary["comparison"]
    top3 = ranking_rows[:3]
    ablation_top8 = next((row for row in topk_rows if row["candidate"] == "ablation_top8"), None)
    gate_top8 = next((row for row in topk_rows if row["candidate"] == "gate_top8"), None)
    ablation_top16 = next((row for row in topk_rows if row["candidate"] == "ablation_top16"), None)

    lines = []
    lines.append("# Frequency Dependency Analysis")
    lines.append("")
    lines.append("## Baseline")
    lines.append("")
    lines.append(f"- Mean verification accuracy: {baseline['mean_acc']:.6f}")
    for dataset, acc in baseline["per_dataset"].items():
        lines.append(f"- {dataset}: {acc:.6f}")
    lines.append("")
    lines.append("## Main Findings")
    lines.append("")
    if len(top3) >= 3:
        lines.append(
            "- The dominant channel is DCT {} with a mean accuracy drop of {:.4f}; "
            "the next strongest channels are DCT {} ({:.4f}) and DCT {} ({:.4f}).".format(
                top3[0]["dct_index"],
                top3[0]["mean_drop"],
                top3[1]["dct_index"],
                top3[1]["mean_drop"],
                top3[2]["dct_index"],
                top3[2]["mean_drop"],
            )
        )
    lines.append(
        "- Gate alignment is weak overall: Spearman(drop, gate_score) = {:.3f}.".format(
            comparison["spearman_drop_vs_gate_score"]
        )
    )
    top5_overlap = comparison["topk_overlap"].get("5", [])
    top10_overlap = comparison["topk_overlap"].get("10", [])
    lines.append(f"- Top-5 overlap between ablation ranking and gate ranking: {top5_overlap}")
    lines.append(f"- Top-10 overlap between ablation ranking and gate ranking: {top10_overlap}")
    if ablation_top8 is not None:
        lines.append(
            "- Keeping the top-8 ablation-ranked channels still reaches {:.6f} mean accuracy "
            "(drop {:.4f} vs full model).".format(
                ablation_top8["mean_acc"],
                ablation_top8["mean_drop_vs_full"],
            )
        )
    if ablation_top16 is not None:
        lines.append(
            "- Keeping the top-16 ablation-ranked channels nearly recovers the full model: "
            "{:.6f} mean accuracy (drop {:.4f}).".format(
                ablation_top16["mean_acc"],
                ablation_top16["mean_drop_vs_full"],
            )
        )
    if gate_top8 is not None:
        lines.append(
            "- In contrast, keeping only the top-8 gate-ranked channels collapses performance to "
            "{:.6f} mean accuracy (drop {:.4f}).".format(
                gate_top8["mean_acc"],
                gate_top8["mean_drop_vs_full"],
            )
        )
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    lines.append("- `figures/01_ablation_top_channels.png`")
    lines.append("- `figures/02_drop_vs_gate_score.png`")
    lines.append("- `figures/03_dataset_heatmap.png`")
    lines.append("- `figures/04_topk_curves.png`")
    lines.append("- `figures/05_rank_scatter.png`")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    result_dir = Path(args.result_dir)
    summary = load_json(result_dir / "summary.json")
    ranking_rows = parse_ranking_rows(load_csv(result_dir / "ranking_comparison.csv"))
    topk_rows = parse_topk_rows(load_csv(result_dir / "keep_topk_search.csv")) if (result_dir / "keep_topk_search.csv").exists() else []

    figures_dir = result_dir / "figures"
    ensure_dir(figures_dir)
    set_plot_style()

    dataset_names = infer_dataset_names(summary)
    plot_ablation_bar(ranking_rows, figures_dir / "01_ablation_top_channels.png", args.topn)
    plot_drop_vs_gate_score(
        ranking_rows,
        summary["comparison"].get("spearman_drop_vs_gate_score"),
        figures_dir / "02_drop_vs_gate_score.png",
        args.topn,
    )
    plot_dataset_heatmap(ranking_rows, dataset_names, figures_dir / "03_dataset_heatmap.png", min(args.topn, 12))
    if topk_rows:
        plot_topk_curves(topk_rows, summary["baseline"]["mean_acc"], figures_dir / "04_topk_curves.png")
    plot_rank_scatter(ranking_rows, figures_dir / "05_rank_scatter.png", args.topn)

    build_report(summary, ranking_rows, topk_rows, result_dir / "analysis_report.md")
    print("Saved plots to {}".format(figures_dir))
    print("Saved report to {}".format(result_dir / "analysis_report.md"))


if __name__ == "__main__":
    main()
