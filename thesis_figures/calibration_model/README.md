# Pitch-Calibration Model — Figures

Figures for the homography / pitch-calibration model (**PnLCalib**, an HRNet
that predicts pitch **keypoint** and **line** heatmaps and solves the
homography from them). Generated from the training/evaluation logs with:

```bash
python tools/make_calibration_figures.py
```

| File | Shows |
|------|-------|
| `C1_keypoints_training.png` | Keypoints-head training: loss + heatmap accuracy/precision over 26 epochs (best accuracy **0.962**). |
| `C2_lines_training.png` | Lines-head training: loss + accuracy/precision over 32 epochs (best accuracy **0.899**). |
| `C3_evaluation_summary.png` | Final system evaluation on **887 images**: IOU, completeness and projection error. |

## Headline numbers (for the text)

| Metric | Mean | Median |
|--------|------|--------|
| **Projection error (metres)** | **0.264 m** | **0.218 m** |
| Reprojection error (normalized) | 0.0069 | 0.0056 |
| IOU — whole pitch | 0.956 | 0.966 |
| IOU — part (lines) | 0.985 | 0.988 |
| Completeness | 1.000 | — |

- **Keypoints head:** accuracy 0.74 → **0.96**, precision 0.64 → **0.89**.
- **Lines head:** accuracy ≈ **0.90**, precision ≈ **0.83**.

> The headline result: a mean projection error of **~26 cm** (median ~22 cm) on
> a 105×68 m pitch — sub-third-of-a-metre calibration accuracy.

Architecture: **HRNet (HighResolutionNet)**, trained on a SLURM cluster and
tracked with Weights & Biases.
