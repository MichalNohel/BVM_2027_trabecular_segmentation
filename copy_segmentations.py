#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Copy Myeloma segmentations and align their voxel orientation to the
corresponding CT NIfTI.

The original TCIA-derived segmentation NIfTI files can have a different
voxel-axis direction than the converted CT NIfTI files.

Typical example:

CT:
    z spacing/direction = -0.600537
    z origin            = -555.603

SEG:
    z spacing/direction = +0.600537
    z origin            = -1212.59

These can describe exactly the same physical volume if the segmentation
voxel array is reversed along Z.

IMPORTANT:
    This script DOES NOT resample segmentations.
    It only performs exact voxel-axis flips.

That means:
    - no interpolation
    - no changed label values
    - no altered segmentation boundaries

If the CT/SEG geometry cannot be reconciled using axis flips only,
the script stops for that case instead of guessing.
"""

import logging
from pathlib import Path
from itertools import combinations

import nibabel as nib
import numpy as np


# =====================================================================
# SETTINGS
# =====================================================================

SOURCE_DIR = Path(
    r"F:\TCIA\manifest-1774389300184\MM_NIfTI_Segmentation_release_2"
)

OUTPUT_DIR = Path(
    r"F:\BVM_2027_DATA\Spinal-Multiple-Myeloma-SEG"
)


# CTs must already exist here, for example:
#
# Myel_001_konv.nii.gz
# Myel_012_a_konv.nii.gz
# Myel_012_b_konv.nii.gz


# We mainly need the trabecular segmentation for nnU-Net.
COPY_SPINE_SEGMENTATION = True
COPY_TRABECULAR_SEGMENTATION = True


# Tolerance for comparison of affine matrices
AFFINE_ATOL = 1e-3
AFFINE_RTOL = 1e-5


# =====================================================================
# LOGGING
# =====================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# =====================================================================
# FLIP TRANSFORMATION
# =====================================================================

def get_flip_matrix(shape, axes):
    """
    Create voxel-coordinate transformation matrix for flipping axes.

    Example for Z flip:

        old_z = (Nz - 1) - new_z

    The returned matrix maps NEW voxel indices to OLD voxel indices.

    Therefore:

        new_affine = old_affine @ flip_matrix
    """

    transform = np.eye(4, dtype=float)

    for axis in axes:
        transform[axis, axis] = -1
        transform[axis, 3] = shape[axis] - 1

    return transform


def get_all_flip_combinations():
    """
    Returns:
        ()
        (0,)
        (1,)
        (2,)
        (0,1)
        (0,2)
        (1,2)
        (0,1,2)
    """

    axes = [0, 1, 2]

    combinations_out = [()]

    for n_axes in range(1, 4):
        combinations_out.extend(
            combinations(
                axes,
                n_axes
            )
        )

    return combinations_out


# =====================================================================
# AFFINE MATCH
# =====================================================================

def find_required_flip(
        segmentation_affine,
        ct_affine,
        shape
):
    """
    Determine whether segmentation geometry can be transformed into
    CT geometry using ONLY voxel-axis flips.

    Returns
    -------
    tuple of axes
        Example:
            (2,)      -> flip Z
            (0, 2)   -> flip X and Z
            ()        -> no flip needed

    None
        No exact flip solution found.
    """

    matches = []

    for axes in get_all_flip_combinations():

        flip_matrix = get_flip_matrix(
            shape,
            axes
        )

        transformed_affine = (
            segmentation_affine
            @ flip_matrix
        )

        if np.allclose(
            transformed_affine,
            ct_affine,
            atol=AFFINE_ATOL,
            rtol=AFFINE_RTOL
        ):
            matches.append(
                axes
            )

    if len(matches) == 0:
        return None

    if len(matches) > 1:
        raise RuntimeError(
            "More than one valid flip solution was found: "
            f"{matches}"
        )

    return matches[0]


# =====================================================================
# SEGMENTATION ALIGNMENT
# =====================================================================

def align_segmentation_to_ct(
        segmentation_file,
        ct_file,
        output_file
):
    """
    Align segmentation voxel orientation to CT using exact axis flips.

    NO interpolation is performed.
    """

    # -----------------------------------------------------------------
    # Load
    # -----------------------------------------------------------------

    ct_img = nib.load(
        str(ct_file)
    )

    seg_img = nib.load(
        str(segmentation_file)
    )

    # -----------------------------------------------------------------
    # Shape check
    # -----------------------------------------------------------------

    if ct_img.shape != seg_img.shape:

        raise RuntimeError(
            "\nCT/SEG shape mismatch:\n"
            f"CT:  {ct_file}\n"
            f"     {ct_img.shape}\n"
            f"SEG: {segmentation_file}\n"
            f"     {seg_img.shape}\n\n"
            "A simple axis flip cannot fix different array shapes."
        )

    # -----------------------------------------------------------------
    # Find required flip
    # -----------------------------------------------------------------

    flip_axes = find_required_flip(
        segmentation_affine=seg_img.affine,
        ct_affine=ct_img.affine,
        shape=seg_img.shape
    )

    if flip_axes is None:

        logger.error("")
        logger.error("Affine mismatch cannot be solved by axis flips.")
        logger.error(f"CT:  {ct_file}")
        logger.error(f"SEG: {segmentation_file}")

        logger.error("")
        logger.error("CT affine:")
        logger.error("\n%s", ct_img.affine)

        logger.error("")
        logger.error("SEG affine:")
        logger.error("\n%s", seg_img.affine)

        raise RuntimeError(
            f"Geometry of {segmentation_file.name} cannot be "
            "reconciled with CT using voxel-axis flips only."
        )

    # -----------------------------------------------------------------
    # Read labels WITHOUT converting to floating-point
    # -----------------------------------------------------------------

    segmentation_data = np.asanyarray(
        seg_img.dataobj
    )

    unique_before = np.unique(
        segmentation_data
    )

    # -----------------------------------------------------------------
    # Apply flips
    # -----------------------------------------------------------------

    if len(flip_axes) > 0:

        logger.info(
            "  Required voxel flip: %s",
            ", ".join(
                ["XYZ"[axis] for axis in flip_axes]
            )
        )

        segmentation_data = np.flip(
            segmentation_data,
            axis=flip_axes
        )

    else:

        logger.info(
            "  Geometry already matches CT - no flip required"
        )

    # -----------------------------------------------------------------
    # Label integrity check
    # -----------------------------------------------------------------

    unique_after = np.unique(
        segmentation_data
    )

    if not np.array_equal(
        unique_before,
        unique_after
    ):

        raise RuntimeError(
            "Segmentation label values changed during flip."
        )

    # -----------------------------------------------------------------
    # New NIfTI
    #
    # Use the segmentation header as basis because it contains the
    # correct datatype etc., but assign CT geometry.
    # -----------------------------------------------------------------

    new_header = seg_img.header.copy()

    aligned_img = nib.Nifti1Image(
        segmentation_data,
        ct_img.affine,
        header=new_header
    )

    # -----------------------------------------------------------------
    # Explicitly make qform/sform agree with CT
    # -----------------------------------------------------------------

    ct_qform, ct_qform_code = (
        ct_img.get_qform(
            coded=True
        )
    )

    ct_sform, ct_sform_code = (
        ct_img.get_sform(
            coded=True
        )
    )

    # If CT qform is unavailable, use CT affine
    if ct_qform is None:
        aligned_img.set_qform(
            ct_img.affine,
            code=1
        )
    else:
        aligned_img.set_qform(
            ct_qform,
            code=int(ct_qform_code)
        )

    # If CT sform is unavailable, use CT affine
    if ct_sform is None:
        aligned_img.set_sform(
            ct_img.affine,
            code=1
        )
    else:
        aligned_img.set_sform(
            ct_sform,
            code=int(ct_sform_code)
        )

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------

    nib.save(
        aligned_img,
        str(output_file)
    )

    # -----------------------------------------------------------------
    # Reload and final verification
    # -----------------------------------------------------------------

    saved_seg = nib.load(
        str(output_file)
    )

    if saved_seg.shape != ct_img.shape:

        raise RuntimeError(
            "Saved segmentation shape does not match CT."
        )

    if not np.allclose(
        saved_seg.affine,
        ct_img.affine,
        atol=AFFINE_ATOL,
        rtol=AFFINE_RTOL
    ):

        raise RuntimeError(
            "\nFinal affine verification failed:\n"
            f"CT:\n{ct_img.affine}\n\n"
            f"SEG:\n{saved_seg.affine}"
        )

    saved_values = np.unique(
        np.asanyarray(
            saved_seg.dataobj
        )
    )

    if not np.array_equal(
        unique_before,
        saved_values
    ):

        raise RuntimeError(
            "Label values differ after saving/reloading."
        )

    logger.info(
        "  ✓ Geometry verified against CT"
    )

    logger.info(
        "  ✓ Labels: %s",
        saved_values.tolist()
    )

    return flip_axes


# =====================================================================
# PROCESS ONE SEGMENTATION
# =====================================================================

def process_segmentation(
        patient_folder,
        patient_name,
        segmentation_suffix
):
    """
    Find source segmentation, corresponding output CT and create
    correctly oriented segmentation.
    """

    segmentation_name = (
        patient_name
        + segmentation_suffix
    )

    source_seg = (
        patient_folder
        / segmentation_name
    )

    output_seg = (
        OUTPUT_DIR
        / segmentation_name
    )

    ct_file = (
        OUTPUT_DIR
        / (
            patient_name
            + "_konv.nii.gz"
        )
    )

    # -----------------------------------------------------------------
    # Files
    # -----------------------------------------------------------------

    if not source_seg.exists():

        logger.warning(
            "  ✗ Segmentation not found: %s",
            segmentation_name
        )

        return False

    if not ct_file.exists():

        logger.error(
            "  ✗ Corresponding CT not found: %s",
            ct_file
        )

        return False

    # -----------------------------------------------------------------
    # Align and save
    # -----------------------------------------------------------------

    align_segmentation_to_ct(
        segmentation_file=source_seg,
        ct_file=ct_file,
        output_file=output_seg
    )

    logger.info(
        "  ✓ Saved: %s",
        output_seg.name
    )

    return True


# =====================================================================
# MAIN
# =====================================================================

def copy_and_align_segmentations():

    logger.info(
        "Source: %s",
        SOURCE_DIR
    )

    logger.info(
        "Output: %s",
        OUTPUT_DIR
    )

    logger.info("")

    # -----------------------------------------------------------------
    # Check source
    # -----------------------------------------------------------------

    if not SOURCE_DIR.exists():

        logger.error(
            "Source directory not found: %s",
            SOURCE_DIR
        )

        return

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------------------------------
    # Patient folders
    # -----------------------------------------------------------------

    patient_folders = sorted([
        folder
        for folder in SOURCE_DIR.iterdir()
        if (
            folder.is_dir()
            and folder.name.startswith("Myel_")
        )
    ])

    logger.info(
        "Found %d patient/examination folders",
        len(patient_folders)
    )

    logger.info("")

    success_count = 0
    skipped_count = 0
    error_count = 0

    # -----------------------------------------------------------------
    # Process
    # -----------------------------------------------------------------

    for patient_folder in patient_folders:

        patient_name = (
            patient_folder.name
        )

        # -------------------------------------------------------------
        # No segmentation
        # -------------------------------------------------------------

        if patient_name == "Myel_017":

            logger.info(
                "Skipping %s "
                "(no trabecular segmentation)",
                patient_name
            )

            skipped_count += 1
            continue

        logger.info(
            "=" * 70
        )

        logger.info(
            "Processing: %s",
            patient_name
        )

        try:

            # ---------------------------------------------------------
            # Whole spine segmentation
            # ---------------------------------------------------------

            if COPY_SPINE_SEGMENTATION:

                ok = process_segmentation(
                    patient_folder=
                        patient_folder,

                    patient_name=
                        patient_name,

                    segmentation_suffix=
                        "_spine_segmentation.nii.gz"
                )

                if ok:
                    success_count += 1
                else:
                    error_count += 1

            # ---------------------------------------------------------
            # Trabecular segmentation
            # ---------------------------------------------------------

            if COPY_TRABECULAR_SEGMENTATION:

                ok = process_segmentation(
                    patient_folder=
                        patient_folder,

                    patient_name=
                        patient_name,

                    segmentation_suffix=
                        "_trabecular_spine_segmentation.nii.gz"
                )

                if ok:
                    success_count += 1
                else:
                    error_count += 1

        except Exception as e:

            logger.exception(
                "Error processing %s: %s",
                patient_name,
                e
            )

            error_count += 1

    # =================================================================
    # SUMMARY
    # =================================================================

    logger.info("")
    logger.info(
        "=" * 70
    )

    logger.info(
        "SUMMARY"
    )

    logger.info(
        "=" * 70
    )

    logger.info(
        "Successfully processed: %d",
        success_count
    )

    logger.info(
        "Skipped: %d",
        skipped_count
    )

    logger.info(
        "Errors: %d",
        error_count
    )

    logger.info(
        "Output directory: %s",
        OUTPUT_DIR
    )

    logger.info(
        "=" * 70
    )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":

    copy_and_align_segmentations()