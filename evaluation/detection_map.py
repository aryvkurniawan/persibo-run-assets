"""Detection accuracy as mean Average Precision (mAP).

Runs the trained YOLO weights through Ultralytics' built-in validator on a
held-out split and reports the COCO-standard detection metrics:

* **mAP@0.5**      — AP averaged over classes at a single IoU of 0.50.
* **mAP@0.5:0.95** — AP averaged over IoU 0.50..0.95 (step 0.05); the headline
  COCO number, much stricter on localisation.
* Per-class AP, plus mean precision / recall at the operating point.

The validation set comes from the same Roboflow project the model was trained
on (``shootaai/soccer_detection`` v3). Pass an API key (or set
``ROBOFLOW_API_KEY``) to auto-download it, or point ``--data`` at a local
Ultralytics ``data.yaml`` if you already have the split on disk.

Usage
-----
    python -m evaluation.detection_map --api-key XXXX            # download + val
    python -m evaluation.detection_map --data path/to/data.yaml  # local split
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from utils.logger import get_logger

logger = get_logger("evaluation.detection_map")

# The Roboflow project this model was trained on (see "Model detection.ipynb").
ROBOFLOW_WORKSPACE = "shootaai"
ROBOFLOW_PROJECT = "soccer_detection"
ROBOFLOW_VERSION = 3
ROBOFLOW_FORMAT = "yolov11"


def download_dataset(
    api_key: str,
    dest: Path,
    workspace: str = ROBOFLOW_WORKSPACE,
    project: str = ROBOFLOW_PROJECT,
    version: int = ROBOFLOW_VERSION,
) -> Path:
    """Download the Roboflow dataset and return the path to its ``data.yaml``."""
    from roboflow import Roboflow

    dest.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s/%s v%d from Roboflow ...", workspace, project, version)
    rf = Roboflow(api_key=api_key)
    ds = rf.workspace(workspace).project(project).version(version).download(
        ROBOFLOW_FORMAT, location=str(dest)
    )
    data_yaml = Path(ds.location) / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found under {ds.location}")
    logger.info("Dataset ready at %s", ds.location)
    return data_yaml


def evaluate(
    weights: str | Path,
    data_yaml: str | Path,
    split: str = "valid",
    imgsz: int = 1280,
    conf: float = 0.001,
    iou: float = 0.6,
    device: str = "cpu",
) -> Dict:
    """Run Ultralytics validation and extract the detection metrics.

    ``conf`` is deliberately tiny: mAP integrates the full precision-recall
    curve, so the validator must see every candidate box, not a thresholded
    subset. ``iou`` is the NMS IoU, not the matching IoU.
    """
    from ultralytics import YOLO

    model = YOLO(str(weights))
    # Roboflow's data.yaml uses "valid" for the validation split; Ultralytics
    # maps split="val" onto that key, so normalise here.
    ul_split = "val" if split in ("val", "valid") else split
    results = model.val(
        data=str(data_yaml),
        split=ul_split,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        verbose=False,
    )
    box = results.box
    names = results.names  # {class_index: name}

    per_class = {}
    for i, class_idx in enumerate(box.ap_class_index):
        per_class[names[int(class_idx)]] = {
            "AP50": round(float(box.ap50[i]), 4),
            "AP50_95": round(float(box.ap[i]), 4),
            "precision": round(float(box.p[i]), 4),
            "recall": round(float(box.r[i]), 4),
        }

    return {
        "metadata": {
            "weights": str(weights),
            "data": str(data_yaml),
            "split": split,
            "imgsz": imgsz,
            "conf": conf,
            "iou_nms": iou,
            "device": device,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
        "overall": {
            "mAP50": round(float(box.map50), 4),
            "mAP50_95": round(float(box.map), 4),
            "mAP75": round(float(box.map75), 4),
            "mean_precision": round(float(box.mp), 4),
            "mean_recall": round(float(box.mr), 4),
        },
        "per_class": per_class,
    }


def write_reports(metrics: Dict, out_dir: Path) -> Path:
    """Write detection metrics as JSON + Markdown; return the JSON path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "detection_metrics.json"
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    o = metrics["overall"]
    lines = [
        "# Detection accuracy — mean Average Precision (mAP)",
        "",
        f"_Model:_ `{metrics['metadata']['weights']}`  ",
        f"_Split:_ {metrics['metadata']['split']}  ",
        f"_Generated:_ {metrics['metadata']['created_at']}",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| **mAP@0.5** | {o['mAP50']:.3f} |",
        f"| **mAP@0.5:0.95** | {o['mAP50_95']:.3f} |",
        f"| mAP@0.75 | {o['mAP75']:.3f} |",
        f"| Mean precision | {o['mean_precision']:.3f} |",
        f"| Mean recall | {o['mean_recall']:.3f} |",
        "",
        "## Per class",
        "",
        "| Class | AP@0.5 | AP@0.5:0.95 | Precision | Recall |",
        "|---|---|---|---|---|",
    ]
    for name, m in metrics["per_class"].items():
        lines.append(
            f"| {name} | {m['AP50']:.3f} | {m['AP50_95']:.3f} | "
            f"{m['precision']:.3f} | {m['recall']:.3f} |"
        )
    md_path = out_dir / "detection_metrics.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote %s and %s", json_path.name, md_path.name)
    return json_path


def _print_summary(metrics: Dict) -> None:
    o = metrics["overall"]
    print("\n================  DETECTION  mAP  ================")
    print(f"  mAP@0.5      : {o['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95 : {o['mAP50_95']:.4f}")
    print(f"  precision    : {o['mean_precision']:.4f}")
    print(f"  recall       : {o['mean_recall']:.4f}")
    print("  --- per class ---")
    for name, m in metrics["per_class"].items():
        print(f"    {name:<12} AP50={m['AP50']:.3f}  AP50-95={m['AP50_95']:.3f}")
    print("=================================================\n")


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Detection mAP evaluation")
    parser.add_argument("--weights", default="weights/best.pt")
    parser.add_argument(
        "--data",
        default=None,
        help="Path to an existing Ultralytics data.yaml (skips download)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ROBOFLOW_API_KEY"),
        help="Roboflow API key (or set ROBOFLOW_API_KEY) to download the split",
    )
    parser.add_argument("--split", default="valid")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dataset-dir", default="evaluation/_dataset")
    parser.add_argument("--out-dir", default="outputs/evaluation")
    args = parser.parse_args(argv)

    if args.data:
        data_yaml = Path(args.data)
    elif args.api_key:
        data_yaml = download_dataset(args.api_key, Path(args.dataset_dir))
    else:
        parser.error("provide --data <data.yaml> or --api-key (ROBOFLOW_API_KEY)")

    metrics = evaluate(
        args.weights, data_yaml, split=args.split,
        imgsz=args.imgsz, device=args.device,
    )
    write_reports(metrics, Path(args.out_dir))
    _print_summary(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
