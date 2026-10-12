"""Build figures for the channel-36 privacy/recognition experiment report."""
from pathlib import Path
import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REC = ROOT / "recognition"
OUT = ROOT / "exp/report_assets/privacy_channel36_20261012"
RECON = REC / "reconstruct_outputs/ms1mv2_sweep64_epoch20"
RESULT_PATHS = {
    "teacher": "logs/channel_selection/teacher_ms1mv2_highfreq_epoch20_verification_20260930.json",
    "baseline": "logs/channel_selection/ms1mv2_channel36_distill_54_20ep/verification_recheck_20260930.json",
    "landmark": "logs/channel_selection/ms1mv2_channel36_spatial_landmark_distill_54_20ep_run2/verification_check_20261010/results.json",
    "deca": "logs/channel_selection/ms1mv2_channel36_deca236_distill_54_20ep_bs512_20261010/verification_recheck_20261012_063552/results.json",
}
NAMES = ["lfw", "cfp_fp", "agedb_30", "calfw", "cplfw"]
LABELS = ["LFW", "CFP-FP", "AgeDB-30", "CALFW", "CPLFW"]
COLORS = {"teacher": "#253b52", "baseline": "#84919f", "landmark": "#08846b", "deca": "#386ad0"}
ORANGE = "#d56731"
EVIDENCE = {
    "reconstruction_summary.json": "reconstruct_outputs/ms1mv2_sweep64_epoch20/summary.json",
    "reconstruction_ssim_summary.json": "reconstruct_outputs/ms1mv2_sweep64_epoch20/ssim/summary.json",
    "reconstruction_report.md": "reconstruct_outputs/ms1mv2_sweep64_epoch20/report.md",
    "recognition_teacher.json": RESULT_PATHS["teacher"],
    "recognition_channel36.json": RESULT_PATHS["baseline"],
    "recognition_landmark_spatial.json": RESULT_PATHS["landmark"],
    "recognition_landmark_vector.json": "logs/channel_selection/ms1mv2_channel36_landmark_distill_54_20ep_run2/verification_20261009/results.json",
    "recognition_deca.json": RESULT_PATHS["deca"],
    "baseline_config.yaml": "logs/channel_selection/ms1mv2_channel36_distill_54_20ep/config.yaml",
    "landmark_config.yaml": "logs/channel_selection/ms1mv2_channel36_spatial_landmark_distill_54_20ep_run2/config.yaml",
    "deca_config.yaml": "logs/channel_selection/ms1mv2_channel36_deca236_distill_54_20ep_bs512_20261010/config.yaml",
    "spatial_landmark_fusion.py": "spatial_landmark_fusion.py",
    "deca_fusion.py": "deca_fusion.py",
}


def copy_evidence():
    evidence = OUT / "evidence"
    evidence.mkdir(exist_ok=False)
    for name, source in EVIDENCE.items():
        with (evidence / name).open("xb") as stream:
            stream.write((REC / source).read_bytes())


def save(fig, name):
    fig.savefig(OUT / (name + ".png"), dpi=180, bbox_inches="tight")
    fig.savefig(OUT / (name + ".svg"), bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=False)
    copy_evidence()
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "figure.facecolor": "white",
    })
    recon = json.loads((RECON / "ssim/summary.json").read_text())
    reference = json.loads((RECON / "summary.json").read_text())
    assert len(recon["models"]) == 64 and not recon["failures"]
    assert recon["manifest_sha256"] == reference["manifest_sha256"]
    rows = sorted(recon["models"], key=lambda x: x["channel"])
    spec = [
        ("mae_rgb_01", "MAE [0,1]", False),
        ("mean_psnr_db", "PSNR [dB]", True),
        ("id_cosine", "Identity cosine", True),
        ("ssim", "SSIM", True),
    ]
    values = {
        key: np.array([r["ssim_rgb"]["mean"] if key == "ssim" else r[key] for r in rows])
        for key, _, _ in spec
    }
    rankings = {
        key: sorted(range(64), key=lambda ch: values[key][ch], reverse=not higher)
        for key, _, higher in spec
    }
    assert all(rankings[k][0] == 36 for k in ("mae_rgb_01", "mean_psnr_db", "id_cosine"))
    assert rankings["ssim"][:2] == [54, 36]
    result = {k: json.loads((REC / v).read_text()) for k, v in RESULT_PATHS.items()}
    accuracies = {k: np.array([d["metrics"][n]["acc"] * 100 for n in NAMES]) for k, d in result.items()}

    with (OUT / "reconstruction_metrics.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["channel", "mae_rgb_01", "psnr_db", "identity_cosine", "ssim"])
        for ch in range(64):
            writer.writerow([ch] + [values[key][ch] for key, _, _ in spec])
    with (OUT / "recognition_metrics.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model"] + LABELS + ["macro_mean_percent"])
        for key, row in accuracies.items():
            writer.writerow([key] + list(row) + [row.mean()])
    with (OUT / "sources.json").open("x") as stream:
        json.dump({"reconstruction": str(RECON.relative_to(ROOT)), "recognition": RESULT_PATHS,
                   "manifest_sha256": recon["manifest_sha256"]}, stream, indent=2)

    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.5), layout="constrained")
    for ax, (key, title, higher) in zip(axes, spec):
        image = ax.imshow(values[key].reshape(8, 8), cmap="viridis" if higher else "viridis_r")
        for ch in range(64):
            y, x = divmod(ch, 8)
            ax.text(x, y, str(ch), ha="center", va="center", fontsize=8, color="white")
        ax.add_patch(Rectangle((3.5, 3.5), 1, 1, fill=False, edgecolor=ORANGE, linewidth=3))
        ax.set_title(title + (" (higher = better)" if higher else " (lower = better)"), fontsize=10)
        ax.set_xticks(range(8)); ax.set_yticks(range(8))
        fig.colorbar(image, ax=ax, shrink=.68, pad=.02)
    fig.suptitle("64 DCT positions · Same 1,000 MS1Mv2 images · Orange: channel 36", fontsize=15, weight="bold")
    save(fig, "01_reconstruction_heatmaps")

    fig, axes = plt.subplots(2, 2, figsize=(11.7, 7.0), layout="constrained")
    for ax, (key, title, _) in zip(axes.flat, spec):
        order = rankings[key][:8]
        y = np.arange(8)
        ax.scatter(values[key][order], y, s=75, c=[ORANGE if ch == 36 else "#75899d" for ch in order])
        ax.set_yticks(y, ["ch %02d" % ch for ch in order]); ax.invert_yaxis()
        ax.set_title(title + ": worst 8", fontsize=11)
        ax.grid(axis="x", alpha=.16)
        span = np.ptp(values[key][order])
        ax.set_xlim(values[key][order].min() - span*.15, values[key][order].max() + span*.32)
        for pos, ch in enumerate(order):
            ax.annotate("%.4f" % values[key][ch], (values[key][ch], pos), xytext=(7, 0),
                        textcoords="offset points", va="center", fontsize=9)
    fig.suptitle("Channel 36: worst on three metrics, second worst on SSIM", fontsize=15, weight="bold")
    save(fig, "02_reconstruction_rankings")

    # The source visualizations use the first eight images from the fixed manifest.
    # Select the first four rows and leave the source pixels unchanged.
    channels = [0, 36, 54, 38, 52]
    pictures = {ch: plt.imread(RECON / "visualizations" / ("freq_%02d.png" % ch)) for ch in channels}
    fig, axes = plt.subplots(4, 6, figsize=(12, 8), layout="constrained")
    for row in range(4):
        for col, ax in enumerate(axes[row]):
            ch = channels[max(0, col-1)]
            ax.imshow(pictures[ch], interpolation="nearest")
            left = 0 if col == 0 else 224
            ax.set_xlim(left-.5, left+223.5)
            ax.set_ylim(row*260+259.5, row*260+35.5)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(["Original", "ch 0", "ch 36", "ch 54", "ch 38", "ch 52"][col],
                             color=ORANGE if col == 2 else "#24354a")
            if col == 0:
                ax.set_ylabel("Sample %d" % (row+1))
    fig.suptitle("The same fixed samples under five single-channel reconstruction models", fontsize=15, weight="bold")
    save(fig, "03_reconstruction_examples")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [3.3, 1]}, layout="constrained")
    x = np.arange(5)
    labels = {"baseline": "Channel 36", "landmark": "+ Spatial landmarks", "deca": "+ DECA / 3DMM", "teacher": "54-channel teacher"}
    for i, key in enumerate(("baseline", "landmark", "deca", "teacher")):
        bars = axes[0].bar(x + (i-1.5)*.19, accuracies[key], .19,
                           color=COLORS[key], label=labels[key])
        for bar, val in zip(bars, accuracies[key]):
            axes[0].text(bar.get_x()+bar.get_width()/2, val+.16, "%.2f" % val,
                         ha="center", va="bottom", rotation=90, fontsize=8)
        axes[1].bar(i, accuracies[key].mean(), color=COLORS[key], width=.65)
        axes[1].text(i, accuracies[key].mean()+.1, "%.2f" % accuracies[key].mean(), ha="center")
    axes[0].set_xticks(x, LABELS); axes[0].set_ylim(82, 102.5)
    axes[0].set_ylabel("10-fold verification accuracy (%)")
    axes[0].legend(ncol=2, fontsize=9)
    axes[0].set_title("Final epoch 20 · Original + horizontal flip")
    axes[1].set_xticks(range(4), ["ch36", "+ LM", "+ DECA", "Teacher"])
    axes[1].set_ylim(90, 99); axes[1].set_title("Five-set macro mean")
    for ax in axes:
        ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
    fig.suptitle("Recognition accuracy across five verification benchmarks", fontsize=15, weight="bold")
    save(fig, "04_recognition_accuracy")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    for ax, same_checkpoint in zip(axes, (False, True)):
        for i, key in enumerate(("landmark", "deca")):
            reference_acc = (np.array([result[key]["metrics"][n]["image_branch_only_acc"]*100 for n in NAMES])
                             if same_checkpoint else accuracies["baseline"])
            gain = accuracies[key] - reference_acc
            bars = ax.bar(x + (i-.5)*.34, gain, .34, color=COLORS[key], label=labels[key])
            for bar, val in zip(bars, gain):
                ax.text(bar.get_x()+bar.get_width()/2, val+.025, "+%.2f" % val,
                        ha="center", fontsize=8.5)
        ax.set_xticks(x, LABELS); ax.set_ylim(0, 2.16)
        ax.set_ylabel("Accuracy gain (percentage points)")
        ax.set_title("Versus separate channel-36 run" if not same_checkpoint else "Condition on vs off in same checkpoint", fontsize=10)
        ax.legend(fontsize=9); ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
    fig.suptitle("Condition gains under two distinct comparisons", fontsize=15, weight="bold")
    save(fig, "05_condition_gains")
    src = REC / "landmark_preview/casia_5_demo_contact_sheet.png"
    with (OUT / "06_landmark_examples.png").open("xb") as stream:
        stream.write(src.read_bytes())
    print("Figures and data saved:", OUT)


if __name__ == "__main__":
    main()
