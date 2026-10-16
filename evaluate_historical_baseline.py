#!/usr/bin/env python3
"""
Evaluate the historical LumVBCanSeg-only nnU-Net predictions
against expert-corrected multiple-myeloma trabecular GT masks.

Expected Windows layout:
E:\Znaceni_dat\Data\Myel_001\Spine_labels\Trabecular_masks\
    Myel_001_trabecular_spine_segmentation_nnUNet_v_1_0.nii.gz
    rev_Myel_001_trabecular_spine_segmentation_nnUNet_v_1_0_cor.nii.gz

The script automatically discovers all Myel_* folders and evaluates
all paired prediction/GT files it finds.

Outputs:
  baseline_metrics_long.csv
  baseline_summary.csv
  baseline_missing_pairs.csv

Metric policy:
- GT present + prediction present: Dice + HD95
- GT present + prediction empty: Dice = 0, HD95 = NaN
- GT empty: not included in the main Dice/HD95 summary
  (absence analysis should be joined later with the dataset QC metadata)

HD95 is calculated in mm using an optimized bounding box around GT ∪ prediction.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure


DEFAULT_BASE = Path(r"E:\Znaceni_dat\Data")

LABEL_NAMES = {
    1: "L1",
    2: "L2",
    3: "L3",
    4: "L4",
    5: "L5",
    6: "L6",
}


def load_label_image(path: Path):
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)

    if not np.issubdtype(data.dtype, np.integer):
        data = np.rint(data).astype(np.int16)

    spacing = tuple(float(x) for x in img.header.get_zooms()[:3])
    voxel_volume_ml = float(np.prod(spacing)) / 1000.0
    return img, data, spacing, voxel_volume_ml


def dice_score(gt: np.ndarray, pred: np.ndarray) -> float:
    gt_sum = int(gt.sum())
    pred_sum = int(pred.sum())

    if gt_sum == 0:
        return np.nan
    if pred_sum == 0:
        return 0.0

    inter = int(np.logical_and(gt, pred).sum())
    return 2.0 * inter / (gt_sum + pred_sum)


def surface(mask: np.ndarray) -> np.ndarray:
    structure = generate_binary_structure(mask.ndim, 1)
    eroded = binary_erosion(mask, structure=structure, border_value=0)
    return np.logical_xor(mask, eroded)


def joint_bbox(gt: np.ndarray, pred: np.ndarray, margin_vox: int = 2):
    union = np.logical_or(gt, pred)
    coords = np.argwhere(union)

    if coords.size == 0:
        return tuple(slice(0, s) for s in gt.shape)

    lo = coords.min(axis=0)
    hi = coords.max(axis=0) + 1

    slices = []
    for ax, size in enumerate(gt.shape):
        start = max(0, int(lo[ax]) - margin_vox)
        stop = min(size, int(hi[ax]) + margin_vox)
        slices.append(slice(start, stop))

    return tuple(slices)


def hd95_mm(gt: np.ndarray, pred: np.ndarray, spacing) -> float:
    if not gt.any() or not pred.any():
        return np.nan

    bbox = joint_bbox(gt, pred, margin_vox=2)
    gt = gt[bbox]
    pred = pred[bbox]

    gt_surface = surface(gt)
    pred_surface = surface(pred)

    dt_to_pred = distance_transform_edt(~pred_surface, sampling=spacing)
    dt_to_gt = distance_transform_edt(~gt_surface, sampling=spacing)

    d_gt_to_pred = dt_to_pred[gt_surface]
    d_pred_to_gt = dt_to_gt[pred_surface]

    distances = np.concatenate([d_gt_to_pred, d_pred_to_gt])

    if distances.size == 0:
        return 0.0

    return float(np.percentile(distances, 95))


def compare_geometry(gt_img, pred_img, case_id: str):
    if gt_img.shape != pred_img.shape:
        raise ValueError(
            f"{case_id}: shape mismatch GT={gt_img.shape}, pred={pred_img.shape}"
        )

    if not np.allclose(gt_img.affine, pred_img.affine, atol=1e-3, rtol=1e-5):
        print(f"WARNING: affine mismatch for {case_id}")


def case_number(case_id: str):
    m = re.search(r"(\d+)", case_id)
    return int(m.group(1)) if m else 999999


def evaluate_case(case_id: str, pred_path: Path, gt_path: Path):
    gt_img, gt_arr, spacing, voxel_volume_ml = load_label_image(gt_path)
    pred_img, pred_arr, _, _ = load_label_image(pred_path)

    compare_geometry(gt_img, pred_img, case_id)

    unexpected_gt = sorted(set(np.unique(gt_arr).tolist()) - set(range(0, 7)))
    unexpected_pred = sorted(set(np.unique(pred_arr).tolist()) - set(range(0, 7)))

    if unexpected_gt:
        raise ValueError(f"{case_id}: unexpected GT labels {unexpected_gt}")
    if unexpected_pred:
        raise ValueError(f"{case_id}: unexpected prediction labels {unexpected_pred}")

    rows = []

    for label, vertebra in LABEL_NAMES.items():
        gt = gt_arr == label
        pred = pred_arr == label

        gt_present = bool(gt.any())
        pred_present = bool(pred.any())

        if gt_present and pred_present:
            status = "segmented"
            dice = dice_score(gt, pred)
            hd95 = hd95_mm(gt, pred, spacing)

        elif gt_present and not pred_present:
            status = "false_negative"
            dice = 0.0
            hd95 = np.nan

        elif not gt_present and pred_present:
            status = "prediction_with_empty_gt"
            dice = np.nan
            hd95 = np.nan

        else:
            status = "both_empty"
            dice = np.nan
            hd95 = np.nan

        rows.append({
            "case_id": case_id,
            "source": "MM",
            "split": "historical_baseline",
            "architecture": "nnunet_original",
            "training_data": "LumVBCanSeg_only",
            "vertebra": vertebra,
            "label": label,
            "dice": dice,
            "hd95_mm": hd95,
            "gt_volume_ml": float(gt.sum()) * voxel_volume_ml,
            "pred_volume_ml": float(pred.sum()) * voxel_volume_ml,
            "gt_present": gt_present,
            "pred_present": pred_present,
            "status": status,
            "prediction_path": str(pred_path),
            "gt_path": str(gt_path),
        })

    return rows


def q25(s):
    return s.quantile(0.25)


def q75(s):
    return s.quantile(0.75)


def make_summary(df: pd.DataFrame):
    x = df[df["gt_present"]].copy()

    return (
        x.groupby(["source", "architecture", "training_data"], dropna=False)
        .agg(
            n_cases=("case_id", "nunique"),
            n_vertebrae=("dice", "size"),
            n_false_negative=("status", lambda s: int((s == "false_negative").sum())),
            dice_mean=("dice", "mean"),
            dice_std=("dice", "std"),
            dice_median=("dice", "median"),
            dice_q25=("dice", q25),
            dice_q75=("dice", q75),
            hd95_n=("hd95_mm", lambda s: int(s.notna().sum())),
            hd95_mean_mm=("hd95_mm", "mean"),
            hd95_std_mm=("hd95_mm", "std"),
            hd95_median_mm=("hd95_mm", "median"),
            hd95_q25_mm=("hd95_mm", q25),
            hd95_q75_mm=("hd95_mm", q75),
        )
        .reset_index()
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base",
        type=Path,
        default=DEFAULT_BASE,
        help=f"Base directory containing Myel_* folders (default: {DEFAULT_BASE})",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <base>/baseline_evaluation)",
    )

    args = parser.parse_args()

    base = args.base
    out_dir = args.out_dir or base / "baseline_evaluation"

    if not base.is_dir():
        raise FileNotFoundError(f"Base directory does not exist: {base}")

    case_dirs = sorted(
        [p for p in base.glob("Myel_*") if p.is_dir()],
        key=lambda p: case_number(p.name),
    )

    print(f"Base: {base}")
    print(f"Myel_* directories found: {len(case_dirs)}")

    all_rows = []
    missing = []

    start = time.perf_counter()
    n_pairs = 0

    for case_dir in case_dirs:
        case_id = case_dir.name
        if case_id == "Myel_017":
            print("Myel_017: SKIP (no trabecular segmentation)")
            continue
        mask_dir = case_dir / "Spine_labels" / "Trabecular_masks"

        pred = mask_dir / (
            f"{case_id}_trabecular_spine_segmentation_nnUNet_v_1_0.nii.gz"
        )
        gt = mask_dir / (
            f"rev_{case_id}_trabecular_spine_segmentation_nnUNet_v_1_0_cor.nii.gz"
        )

        if not pred.is_file() or not gt.is_file():
            missing.append({
                "case_id": case_id,
                "prediction_exists": pred.is_file(),
                "gt_exists": gt.is_file(),
                "prediction_path": str(pred),
                "gt_path": str(gt),
            })
            continue

        t0 = time.perf_counter()
        rows = evaluate_case(case_id, pred, gt)
        all_rows.extend(rows)
        n_pairs += 1

        print(
            f"{case_id}: OK "
            f"({time.perf_counter() - t0:.2f} s)"
        )

    if not all_rows:
        raise RuntimeError("No prediction/GT pairs were found.")

    df = pd.DataFrame(all_rows).sort_values(
        ["case_id", "label"]
    ).reset_index(drop=True)

    summary = make_summary(df)
    missing_df = pd.DataFrame(missing)

    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "baseline_metrics_long.csv"
    summary_path = out_dir / "baseline_summary.csv"
    missing_path = out_dir / "baseline_missing_pairs.csv"

    df.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    missing_df.to_csv(missing_path, index=False)

    print("\nDone.")
    print(f"Paired cases evaluated: {n_pairs}")
    print(f"Missing/incomplete cases: {len(missing)}")
    print(f"Elapsed: {time.perf_counter() - start:.1f} s")
    print(f"\nMetrics: {metrics_path}")
    print(f"Summary: {summary_path}")
    print(f"Missing: {missing_path}")

    print("\nOverall baseline summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
