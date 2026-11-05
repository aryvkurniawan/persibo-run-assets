"""Thesis figures for the pitch-calibration model (PnLCalib / HRNet).

Parses the training logs (keypoints + lines heatmap heads) and the final
system evaluation, and renders publication-quality figures:

  - keypoints training curves (loss + heatmap accuracy/precision)
  - lines training curves
  - final evaluation summary (projection error, IOU, completeness)

    python tools/make_calibration_figures.py
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "outputs" / "_logs" / "logs"
OUT = ROOT / "thesis_figures" / "calibration_model"

_LOSS = re.compile(
    r"LOSS train ([0-9.eE+-]+) valid ([0-9.eE+-]+) "
    r"Accuracy ([0-9.eE+-]+) Precision ([0-9.eE+-]+)")


def parse_epochs(log_path: Path):
    """Return per-epoch (train_loss, valid_loss, accuracy, precision)."""
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    rows = [tuple(float(g) for g in m.groups()) for m in _LOSS.finditer(text)]
    return rows


def training_figure(rows, title, out_name, loss_unit="1e-6"):
    epochs = list(range(1, len(rows) + 1))
    tr = [r[0] for r in rows]
    va = [r[1] for r in rows]
    acc = [r[2] for r in rows]
    prec = [r[3] for r in rows]
    scale = 1e-6 if loss_unit == "1e-6" else 1e-5

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.4), dpi=160)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    axL.plot(epochs, [v / scale for v in tr], "-o", ms=3, color="#2b7bba",
             label="train loss")
    axL.plot(epochs, [v / scale for v in va], "-o", ms=3, color="#e0662b",
             label="valid loss")
    axL.set_xlabel("Epoch")
    axL.set_ylabel(f"Heatmap loss (x{loss_unit})")
    axL.set_title("Training / validation loss")
    axL.grid(alpha=0.3)
    axL.legend()

    axR.plot(epochs, acc, "-o", ms=3, color="#3cb44b", label="accuracy")
    axR.plot(epochs, prec, "-o", ms=3, color="#9b30b0", label="precision")
    best = max(range(len(acc)), key=lambda i: acc[i])
    axR.annotate(f"best acc {acc[best]:.3f}", (epochs[best], acc[best]),
                 textcoords="offset points", xytext=(0, -16), fontsize=9,
                 ha="center", color="#3cb44b")
    axR.set_xlabel("Epoch")
    axR.set_ylabel("Score")
    axR.set_ylim(0.6, 1.0)
    axR.set_title("Heatmap accuracy & precision")
    axR.grid(alpha=0.3)
    axR.legend(loc="lower right")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / out_name)
    plt.close(fig)
    print(f"  + {out_name}  ({len(rows)} epochs)")


def evaluation_figure(out_name):
    """Final system accuracy on the held-out evaluation set (887 images)."""
    # Parsed from evaluation_5828.log.
    metrics = {
        "IOU\n(part / lines)": (0.985, 0.988),
        "IOU\n(whole pitch)": (0.956, 0.966),
        "Completeness": (1.000, 1.000),
    }
    proj = {"mean": 0.264, "median": 0.218}     # metres
    reproj = {"mean": 0.0069, "median": 0.0056}  # normalized

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4), dpi=160,
                                   gridspec_kw={"width_ratios": [1.4, 1]})
    fig.suptitle("Pitch-Calibration Model - final evaluation (887 images)",
                 fontsize=13, fontweight="bold")

    labels = list(metrics)
    means = [metrics[k][0] for k in labels]
    meds = [metrics[k][1] for k in labels]
    x = range(len(labels))
    ax1.bar([i - 0.18 for i in x], means, width=0.36, color="#2b7bba",
            label="mean")
    ax1.bar([i + 0.18 for i in x], meds, width=0.36, color="#7fb8e0",
            label="median")
    for i in x:
        ax1.text(i - 0.18, means[i] + 0.005, f"{means[i]:.3f}", ha="center",
                 fontsize=8)
        ax1.text(i + 0.18, meds[i] + 0.005, f"{meds[i]:.3f}", ha="center",
                 fontsize=8)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0.9, 1.01)
    ax1.set_ylabel("Score")
    ax1.set_title("Overlap & completeness (higher = better)")
    ax1.legend(loc="lower left")
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar([0 - 0.18, 0 + 0.18], [proj["mean"], proj["median"]], width=0.36,
            color=["#e0662b", "#f0a070"])
    ax2.text(-0.18, proj["mean"] + 0.005, f'{proj["mean"]:.2f} m', ha="center",
             fontsize=9, fontweight="bold")
    ax2.text(0.18, proj["median"] + 0.005, f'{proj["median"]:.2f} m',
             ha="center", fontsize=9, fontweight="bold")
    ax2.set_xticks([-0.18, 0.18])
    ax2.set_xticklabels(["mean", "median"])
    ax2.set_ylim(0, 0.4)
    ax2.set_ylabel("Projection error (metres)")
    ax2.set_title("Projection error on a 105x68 m pitch\n(lower = better)")
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / out_name)
    plt.close(fig)
    print(f"  + {out_name}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Writing calibration-model figures -> {OUT}")

    kp = parse_epochs(LOGS / "keypoints_5724.log")
    if kp:
        training_figure(
            kp, "Pitch-Calibration Model - Keypoints head (HRNet) training",
            "C1_keypoints_training.png", loss_unit="1e-6")
    ln = parse_epochs(LOGS / "lines_5745.log")
    if ln:
        training_figure(
            ln, "Pitch-Calibration Model - Lines head (HRNet) training",
            "C2_lines_training.png", loss_unit="1e-5")
    evaluation_figure("C3_evaluation_summary.png")
    print("Done.")


if __name__ == "__main__":
    main()
