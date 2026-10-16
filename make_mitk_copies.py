#!/usr/bin/env python3

import sys
from pathlib import Path
import nibabel as nib

BASE_RAW = Path("/home/nohel/DATA/nnUNet_raw/Dataset911_LumbarTrabecularSeg_LumbarCrop")
BASE_PRED = Path("/home/nohel/DATA/nnUNet_results/predikt_bvm")

PRED_DIRS = {
    "nnunet": BASE_PRED / "pred_911_nnunet_cv5_cropInput",
    "resencL": BASE_PRED / "pred_911_resencL_cv5_cropInput",
}


def make_mitk_copy(src: Path, dst: Path):
    if not src.is_file():
        print(f"SKIP - missing: {src}")
        return

    img = nib.load(str(src))
    aff = img.affine.copy()

    out = nib.Nifti1Image(
        img.dataobj,
        aff,
        header=img.header.copy()
    )

    out.set_qform(aff, code=1)
    out.set_sform(aff, code=1)

    nib.save(out, str(dst))

    check = nib.load(str(dst))

    print(f"OK: {src.name}")
    print(f" -> {dst}")
    print(
        f"    qform_code={int(check.header['qform_code'])}, "
        f"sform_code={int(check.header['sform_code'])}"
    )


def main():
    if len(sys.argv) != 2:
        print("Usage:")
        print("  uv run python make_mitk_copies.py Myel_021")
        sys.exit(1)

    case_id = sys.argv[1]

    files = [
        (
            BASE_RAW / "imagesTs" / f"{case_id}_0000.nii.gz",
            BASE_RAW / "imagesTs" / f"{case_id}_0000_MITK.nii.gz",
        ),
        (
            BASE_RAW / "labelsTs" / f"{case_id}.nii.gz",
            BASE_RAW / "labelsTs" / f"{case_id}_MITK.nii.gz",
        ),
        (
            PRED_DIRS["nnunet"] / f"{case_id}.nii.gz",
            PRED_DIRS["nnunet"] / f"{case_id}_MITK.nii.gz",
        ),
        (
            PRED_DIRS["resencL"] / f"{case_id}.nii.gz",
            PRED_DIRS["resencL"] / f"{case_id}_MITK.nii.gz",
        ),
    ]

    print(f"Creating MITK-compatible copies for {case_id}\n")

    for src, dst in files:
        make_mitk_copy(src, dst)

    print("\nDone.")


if __name__ == "__main__":
    main()