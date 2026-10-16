#!/usr/bin/env python3
"""
Evaluate BVM 2027 lumbar trabecular segmentation predictions.

Expected current layout
-----------------------
/home/nohel/DATA/nnUNet_results/predikt_bvm/
    pred_910_nnunet_cv5_fullInput/
    pred_910_resencL_cv5_fullInput/
    pred_910_nnunet_cv5_cropInput/
    pred_910_resencL_cv5_cropInput/
    pred_911_nnunet_cv5_cropInput/
    pred_911_resencL_cv5_cropInput/
    pred_911_nnunet_cv5_fullInput/
    pred_911_resencL_cv5_fullInput/
    pred_911_nnunet_cv5_oofValidation/
    pred_911_resencL_cv5_oofValidation/
    pred_910_nnunet_foldall_fullInput/
    pred_910_resencL_foldall_fullInput/
    pred_911_nnunet_foldall_cropInput/
    pred_911_resencL_foldall_cropInput/

Ground truth
------------
Dataset910 ... labelsTs : FullFOV independent test GT
Dataset911 ... labelsTs : cropped independent test GT
Dataset911 ... labelsTr : cropped training GT used for OOF validation

Outputs
-------
metrics_long.csv
summary_metrics.csv
absence_summary.csv
anatomical_absence_summary.csv

Empty-mask policy
-----------------
GT present, prediction present:
    Dice and HD95 are calculated.
GT present, prediction empty:
    Dice = 0, HD95 = NaN, status = false_negative.
GT empty, prediction empty:
    Dice/HD95 = NaN, status = correct_absence.
GT empty, prediction present:
    Dice/HD95 = NaN, status = false_positive_absence_case.

HD95 is calculated in millimetres from bidirectional surface distances.
Volumes are reported in mL.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure


DEFAULT_PRED_ROOT = Path("/home/nohel/DATA/nnUNet_results/predikt_bvm")
DEFAULT_RAW_ROOT = Path("/home/nohel/DATA/nnUNet_raw")
DEFAULT_METADATA_DATASET = "Dataset911_LumbarTrabecularSeg_LumbarCrop"

DATASET_910 = "Dataset910_LumbarTrabecularSeg_FullFOV"
DATASET_911 = "Dataset911_LumbarTrabecularSeg_LumbarCrop"

LABEL_NAMES = {
    1: "L1",
    2: "L2",
    3: "L3",
    4: "L4",
    5: "L5",
    6: "L6",
}

TEST_RE = re.compile(
    r"^pred_(?P<train_ds>910|911)_(?P<arch>nnunet|resencL)_"
    r"(?P<mode>cv5|foldall)_(?P<input_fov>crop|full)Input$"
)
OOF_RE = re.compile(
    r"^pred_(?P<train_ds>910|911)_(?P<arch>nnunet|resencL)_cv5_oofValidation$"
)



def _as_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_case_metadata(dataset_cases_path: Path, qc_missing_path: Path):
    """Load case-level anatomy metadata and pathologically missing labels."""
    if not dataset_cases_path.is_file():
        raise FileNotFoundError(f"Missing metadata file: {dataset_cases_path}")
    if not qc_missing_path.is_file():
        raise FileNotFoundError(f"Missing QC file: {qc_missing_path}")

    cases = pd.read_csv(dataset_cases_path)
    missing = pd.read_csv(qc_missing_path)

    required_cases = {"case_id", "patient_id", "source", "split", "has_L6"}
    required_missing = {"case_id", "missing_labels"}
    if not required_cases.issubset(cases.columns):
        raise ValueError(
            f"{dataset_cases_path} is missing columns: "
            f"{sorted(required_cases - set(cases.columns))}"
        )
    if not required_missing.issubset(missing.columns):
        raise ValueError(
            f"{qc_missing_path} is missing columns: "
            f"{sorted(required_missing - set(missing.columns))}"
        )

    case_meta = {}
    for _, r in cases.iterrows():
        cid = str(r["case_id"])
        source_raw = str(r["source"])
        source = "MM" if source_raw.lower() in {"myeloma", "mm"} else "LumVBCanSeg"
        case_meta[cid] = {
            "patient_id": str(r["patient_id"]),
            "source": source,
            "dataset_split": str(r["split"]).lower(),
            "has_L6": _as_bool(r["has_L6"]),
        }

    pathological_missing = set()
    for _, r in missing.iterrows():
        cid = str(r["case_id"])
        labels = str(r["missing_labels"]) if not pd.isna(r["missing_labels"]) else ""
        for vertebra in [x.strip() for x in labels.split(";") if x.strip()]:
            pathological_missing.add((cid, vertebra))

    return case_meta, pathological_missing


def source_from_case(case_id: str) -> str:
    """Map case ID to source cohort.

    Current MM cases are expected to begin with 'Myel_'.
    Everything else is treated as LumVBCanSeg.
    Edit this function if your LumVBCanSeg naming scheme needs a stricter rule.
    """
    return "MM" if case_id.lower().startswith("myel_") else "LumVBCanSeg"


def patient_from_case(case_id: str, source: str) -> str:
    """Collapse MM follow-up suffixes such as Myel_012_a / Myel_012_b."""
    if source == "MM":
        m = re.match(r"^(Myel_\d+)(?:_[A-Za-z0-9]+)?$", case_id, flags=re.IGNORECASE)
        if m:
            return m.group(1)
    return case_id


def load_label_image(path: Path):
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    # nnU-Net segmentations are integer labels. Avoid unnecessary float arrays.
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


def _joint_bbox(gt: np.ndarray, pred: np.ndarray, margin_vox: int = 2):
    """Return a tight joint bounding box around GT and prediction.

    Cropping to the union bounding box does not change nearest-surface
    distances between GT and prediction, while greatly reducing the size
    of the arrays used by the Euclidean distance transform.
    """
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

    # Major optimization: run surface extraction and distance transforms only
    # in a tight bounding box around GT ∪ prediction, not over the full CT.
    bbox = _joint_bbox(gt, pred, margin_vox=2)
    gt = gt[bbox]
    pred = pred[bbox]

    gt_surface = surface(gt)
    pred_surface = surface(pred)

    # Distance to nearest surface voxel in physical units (mm).
    dt_to_pred = distance_transform_edt(~pred_surface, sampling=spacing)
    dt_to_gt = distance_transform_edt(~gt_surface, sampling=spacing)

    d_gt_to_pred = dt_to_pred[gt_surface]
    d_pred_to_gt = dt_to_gt[pred_surface]
    all_distances = np.concatenate([d_gt_to_pred, d_pred_to_gt])

    if all_distances.size == 0:
        return 0.0
    return float(np.percentile(all_distances, 95))


def parse_experiment(folder: Path):
    name = folder.name

    m = TEST_RE.match(name)
    if m:
        d = m.groupdict()
        training_fov = "full" if d["train_ds"] == "910" else "crop"
        return {
            "experiment": name,
            "split": "test",
            "architecture": d["arch"],
            "training_fov": training_fov,
            "inference_fov": d["input_fov"],
            "model_mode": d["mode"],
            "train_dataset_id": int(d["train_ds"]),
        }

    m = OOF_RE.match(name)
    if m:
        d = m.groupdict()
        training_fov = "full" if d["train_ds"] == "910" else "crop"
        return {
            "experiment": name,
            "split": "validation_oof",
            "architecture": d["arch"],
            "training_fov": training_fov,
            # OOF prediction uses the corresponding training dataset input.
            "inference_fov": training_fov,
            "model_mode": "oof_cv5",
            "train_dataset_id": int(d["train_ds"]),
        }

    return None


def gt_dir_for(exp, raw_root: Path) -> Path:
    if exp["split"] == "validation_oof":
        dataset = DATASET_910 if exp["train_dataset_id"] == 910 else DATASET_911
        return raw_root / dataset / "labelsTr"

    # For test predictions, use GT matching the inference input geometry.
    if exp["inference_fov"] == "full":
        return raw_root / DATASET_910 / "labelsTs"
    return raw_root / DATASET_911 / "labelsTs"


def compare_geometry(gt_img, pred_img, case_id: str, experiment: str):
    if gt_img.shape != pred_img.shape:
        raise ValueError(
            f"{experiment} / {case_id}: shape mismatch "
            f"GT={gt_img.shape}, prediction={pred_img.shape}"
        )

    # Small floating-point differences are harmless; large differences indicate a geometry problem.
    if not np.allclose(gt_img.affine, pred_img.affine, atol=1e-3, rtol=1e-5):
        print(
            f"WARNING: affine mismatch in {experiment} / {case_id}",
            file=sys.stderr,
        )


def evaluate_folder(folder: Path, raw_root: Path, case_meta, pathological_missing):
    exp = parse_experiment(folder)
    if exp is None:
        return []

    gt_dir = gt_dir_for(exp, raw_root)
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"GT directory does not exist: {gt_dir}")

    pred_files = sorted(folder.glob("*.nii.gz"))
    if not pred_files:
        raise RuntimeError(f"No NIfTI predictions found in {folder}")

    rows = []

    for pred_path in pred_files:
        case_id = pred_path.name[:-7]  # strip .nii.gz
        gt_path = gt_dir / f"{case_id}.nii.gz"
        if not gt_path.is_file():
            raise FileNotFoundError(
                f"Missing GT for {exp['experiment']} / {case_id}: {gt_path}"
            )

        gt_img, gt_arr, spacing, voxel_volume_ml = load_label_image(gt_path)
        pred_img, pred_arr, _, _ = load_label_image(pred_path)
        compare_geometry(gt_img, pred_img, case_id, exp["experiment"])

        if case_id not in case_meta:
            raise KeyError(f"{case_id}: case not found in dataset_cases.csv")

        meta = case_meta[case_id]
        source = meta["source"]
        patient_id = meta["patient_id"]

        expected_split = "train" if exp["split"] == "validation_oof" else "test"
        if meta["dataset_split"] != expected_split:
            raise ValueError(
                f"{case_id}: metadata split={meta['dataset_split']}, "
                f"but experiment requires {expected_split}"
            )

        unexpected_gt = sorted(set(np.unique(gt_arr).tolist()) - set(range(0, 7)))
        unexpected_pred = sorted(set(np.unique(pred_arr).tolist()) - set(range(0, 7)))
        if unexpected_gt:
            raise ValueError(f"{case_id}: unexpected GT labels {unexpected_gt}")
        if unexpected_pred:
            raise ValueError(
                f"{exp['experiment']} / {case_id}: unexpected prediction labels {unexpected_pred}"
            )

        for label, vertebra in LABEL_NAMES.items():
            gt = gt_arr == label
            pred = pred_arr == label
            gt_present = bool(gt.any())
            pred_present = bool(pred.any())

            anatomically_expected = (label <= 5) or (label == 6 and meta["has_L6"])
            pathologically_absent = (case_id, vertebra) in pathological_missing
            anatomically_absent = not anatomically_expected

            # QC consistency checks.
            if pathologically_absent and not anatomically_expected:
                raise ValueError(
                    f"{case_id} {vertebra}: marked pathologically absent but "
                    "not anatomically expected"
                )
            if pathologically_absent and gt_present:
                raise ValueError(
                    f"{case_id} {vertebra}: marked pathologically absent but GT is present"
                )
            if anatomically_absent and gt_present:
                raise ValueError(
                    f"{case_id} {vertebra}: GT is present although metadata marks "
                    "this level anatomically absent"
                )

            if anatomically_absent:
                status = (
                    "anatomically_absent_predicted"
                    if pred_present
                    else "anatomically_absent_correct"
                )
                dice = np.nan
                hd95 = np.nan
            elif pathologically_absent:
                status = (
                    "false_positive_pathological_absence"
                    if pred_present
                    else "correct_pathological_absence"
                )
                dice = np.nan
                hd95 = np.nan
            elif gt_present and pred_present:
                status = "segmented"
                dice = dice_score(gt, pred)
                hd95 = hd95_mm(gt, pred, spacing)
            elif gt_present and not pred_present:
                status = "false_negative"
                dice = 0.0
                hd95 = np.nan
            else:
                # Any remaining expected-but-empty GT is not explained by the QC file.
                raise ValueError(
                    f"{case_id} {vertebra}: expected vertebral level has empty GT "
                    "but is not listed in qc_missing_labels.csv"
                )

            gt_volume_ml = float(gt.sum()) * voxel_volume_ml
            pred_volume_ml = float(pred.sum()) * voxel_volume_ml

            row = {
                "case_id": case_id,
                "patient_id": patient_id,
                "source": source,
                "split": exp["split"],
                "architecture": exp["architecture"],
                "training_fov": exp["training_fov"],
                "inference_fov": exp["inference_fov"],
                "model_mode": exp["model_mode"],
                "experiment": exp["experiment"],
                "vertebra": vertebra,
                "label": label,
                "dice": dice,
                "hd95_mm": hd95,
                "gt_volume_ml": gt_volume_ml,
                "pred_volume_ml": pred_volume_ml,
                "gt_present": gt_present,
                "pred_present": pred_present,
                "anatomically_expected": anatomically_expected,
                "anatomically_absent": anatomically_absent,
                "pathologically_absent": pathologically_absent,
                "has_L6": meta["has_L6"],
                "status": status,
            }
            rows.append(row)

    return rows


def q25(s):
    return s.quantile(0.25)


def q75(s):
    return s.quantile(0.75)


def build_metric_summary(df: pd.DataFrame) -> pd.DataFrame:
    # Main metric summary only where GT exists.
    x = df[df["gt_present"]].copy()

    group_cols = [
        "experiment",
        "split",
        "architecture",
        "training_fov",
        "inference_fov",
        "model_mode",
        "source",
    ]

    summary = (
        x.groupby(group_cols, dropna=False)
        .agg(
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
    return summary


def build_absence_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize only pathologically absent trabecular compartments."""
    x = df[df["pathologically_absent"]].copy()

    group_cols = [
        "experiment",
        "split",
        "architecture",
        "training_fov",
        "inference_fov",
        "model_mode",
        "source",
    ]

    if x.empty:
        return pd.DataFrame(
            columns=group_cols
            + [
                "n_pathologically_absent",
                "n_correct_absence",
                "n_false_positive",
                "false_positive_rate",
                "false_positive_volume_ml_mean",
                "false_positive_volume_ml_median",
            ]
        )

    out = (
        x.groupby(group_cols, dropna=False)
        .agg(
            n_pathologically_absent=("status", "size"),
            n_correct_absence=(
                "status",
                lambda s: int((s == "correct_pathological_absence").sum()),
            ),
            n_false_positive=(
                "status",
                lambda s: int((s == "false_positive_pathological_absence").sum()),
            ),
            false_positive_volume_ml_mean=(
                "pred_volume_ml",
                lambda s: float(s[s > 0].mean()) if (s > 0).any() else 0.0,
            ),
            false_positive_volume_ml_median=(
                "pred_volume_ml",
                lambda s: float(s[s > 0].median()) if (s > 0).any() else 0.0,
            ),
        )
        .reset_index()
    )
    out["false_positive_rate"] = (
        out["n_false_positive"] / out["n_pathologically_absent"]
    )
    return out


def build_anatomical_absence_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Optional QC summary for anatomically absent levels (mainly non-L6 anatomy)."""
    x = df[df["anatomically_absent"]].copy()

    group_cols = [
        "experiment",
        "split",
        "architecture",
        "training_fov",
        "inference_fov",
        "model_mode",
        "source",
    ]

    if x.empty:
        return pd.DataFrame()

    out = (
        x.groupby(group_cols, dropna=False)
        .agg(
            n_anatomically_absent=("status", "size"),
            n_correct=(
                "status",
                lambda s: int((s == "anatomically_absent_correct").sum()),
            ),
            n_predicted=(
                "status",
                lambda s: int((s == "anatomically_absent_predicted").sum()),
            ),
            predicted_volume_ml_mean=(
                "pred_volume_ml",
                lambda s: float(s[s > 0].mean()) if (s > 0).any() else 0.0,
            ),
        )
        .reset_index()
    )
    out["prediction_rate"] = out["n_predicted"] / out["n_anatomically_absent"]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred-root",
        type=Path,
        default=DEFAULT_PRED_ROOT,
        help=f"Prediction root (default: {DEFAULT_PRED_ROOT})",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help=f"nnUNet_raw root (default: {DEFAULT_RAW_ROOT})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <pred-root>/evaluation)",
    )
    parser.add_argument(
        "--dataset-cases",
        type=Path,
        default=None,
        help=(
            "Path to dataset_cases.csv "
            "(default: <raw-root>/Dataset911.../dataset_cases.csv)"
        ),
    )
    parser.add_argument(
        "--qc-missing",
        type=Path,
        default=None,
        help=(
            "Path to qc_missing_labels.csv "
            "(default: <raw-root>/Dataset911.../qc_missing_labels.csv)"
        ),
    )
    args = parser.parse_args()

    pred_root = args.pred_root
    raw_root = args.raw_root
    out_dir = args.out_dir or pred_root / "evaluation"
    metadata_dir = raw_root / DEFAULT_METADATA_DATASET
    dataset_cases_path = args.dataset_cases or metadata_dir / "dataset_cases.csv"
    qc_missing_path = args.qc_missing or metadata_dir / "qc_missing_labels.csv"

    case_meta, pathological_missing = load_case_metadata(
        dataset_cases_path, qc_missing_path
    )

    print(f"Metadata cases:        {dataset_cases_path}")
    print(f"Pathological-missing:  {qc_missing_path}")
    print(f"Cases in metadata:     {len(case_meta)}")
    print(f"Pathologically absent levels: {len(pathological_missing)}")

    if not pred_root.is_dir():
        raise FileNotFoundError(f"Prediction root does not exist: {pred_root}")

    folders = sorted(p for p in pred_root.iterdir() if p.is_dir())
    recognized = [(p, parse_experiment(p)) for p in folders]
    recognized = [(p, e) for p, e in recognized if e is not None]

    if not recognized:
        raise RuntimeError(f"No recognized experiment folders found in {pred_root}")

    print("Recognized experiments:")
    for p, e in recognized:
        n = len(list(p.glob("*.nii.gz")))
        print(
            f"  {p.name:48s}  n={n:3d}  "
            f"{e['split']:14s} train={e['training_fov']:4s} input={e['inference_fov']:4s}"
        )

    all_rows = []
    total_start = time.perf_counter()
    for folder, _ in recognized:
        print(f"\nEvaluating {folder.name} ...")
        t0 = time.perf_counter()
        rows = evaluate_folder(folder, raw_root, case_meta, pathological_missing)
        elapsed = time.perf_counter() - t0
        all_rows.extend(rows)
        print(f"  rows added: {len(rows)}")
        print(f"  elapsed: {elapsed:.1f} s")

    df = pd.DataFrame(all_rows)

    # Stable, readable ordering.
    df = df.sort_values(
        [
            "split",
            "model_mode",
            "training_fov",
            "inference_fov",
            "architecture",
            "source",
            "case_id",
            "label",
        ]
    ).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "metrics_long.csv"
    summary_path = out_dir / "summary_metrics.csv"
    absence_path = out_dir / "absence_summary.csv"
    anatomical_absence_path = out_dir / "anatomical_absence_summary.csv"

    df.to_csv(metrics_path, index=False)
    build_metric_summary(df).to_csv(summary_path, index=False)
    build_absence_summary(df).to_csv(absence_path, index=False)
    build_anatomical_absence_summary(df).to_csv(
        anatomical_absence_path, index=False
    )

    print("\nDone.")
    print(f"  Long metrics:     {metrics_path}")
    print(f"  Metric summary:   {summary_path}")
    print(f"  Pathol. absence:  {absence_path}")
    print(f"  Anatom. absence:  {anatomical_absence_path}")
    print(f"  Total rows:       {len(df)}")
    print(f"  Experiments:      {df['experiment'].nunique()}")
    print(f"  Unique cases:     {df['case_id'].nunique()}")
    print(f"  Total elapsed:    {time.perf_counter() - total_start:.1f} s")


if __name__ == "__main__":
    main()
