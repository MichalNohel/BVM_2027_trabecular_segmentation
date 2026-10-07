# -*- coding: utf-8 -*-

"""
Create nnU-Net v2 Dataset911 from Dataset910.

Dataset910:
    Dataset910_LumbarTrabecularSeg_FullFOV

Dataset911:
    Dataset911_LumbarTrabecularSeg_LumbarCrop


LOGIC
-----
- Dataset911 uses EXACTLY the same train/test split as Dataset910.
- LumVBCanSeg cases are copied unchanged.
- Myeloma cases are cropped only in the Z axis.
- X and Y dimensions are NOT cropped.
- Crop boundary is defined using the full-spine segmentation.
- T10 = label 17.
- A 20 mm cranial margin above T10 is retained.
- Everything caudal to that boundary is retained until the end of the scan.

IMPORTANT
---------
After conversion to canonical RAS:

    voxel Z index increases toward Superior.

Therefore:

    low Z index  = inferior / caudal
    high Z index = superior / cranial

We therefore keep:

    [:, :, :z_stop]

and NOT:

    [:, :, z_start:]


Full-spine labels
-----------------
1  = C1
...
7  = C7
8  = T1
...
17 = T10
18 = T11
19 = T12
20 = L1
21 = L2
22 = L3
23 = L4
24 = L5
25 = L6, if present
"""

import os
import csv
import json
import shutil
import math

import nibabel as nib
import numpy as np


# =====================================================================
# SETTINGS
# =====================================================================

BASE_DIR = r"F:\BVM_2027_DATA"

MYEL_DIR = os.path.join(
    BASE_DIR,
    "Spinal-Multiple-Myeloma-SEG"
)


# =====================================================================
# nnU-Net
# =====================================================================

NNUNET_RAW = r"E:\nnUNet_v2_MAIN_FILE\nnUNet_raw"

DATASET910_DIR = os.path.join(
    NNUNET_RAW,
    "Dataset910_LumbarTrabecularSeg_FullFOV"
)

TASK_ID = 911
TASK_NAME = "LumbarTrabecularSeg_LumbarCrop"

DATASET_NAME = f"Dataset{TASK_ID:03d}_{TASK_NAME}"

OUTPUT_DIR = os.path.join(
    NNUNET_RAW,
    DATASET_NAME
)


# =====================================================================
# OUTPUT SETTINGS
# =====================================================================

# True = delete existing Dataset911 and recreate it.
# Dataset910 will NOT be modified.
OVERWRITE_DATASET = True


# =====================================================================
# CROP SETTINGS
# =====================================================================

# T10 in the full spine segmentation
T10_LABEL = 17

# Keep additional cranial context above T10.
CROP_MARGIN_MM = 0.0

# If T10 is not present:
#
# True  -> stop with an error
# False -> use the first available more caudal vertebra
#
REQUIRE_T10 = True


# =====================================================================
# OUTPUT FOLDERS
# =====================================================================

IMAGES_TR = os.path.join(
    OUTPUT_DIR,
    "imagesTr"
)

LABELS_TR = os.path.join(
    OUTPUT_DIR,
    "labelsTr"
)

IMAGES_TS = os.path.join(
    OUTPUT_DIR,
    "imagesTs"
)

LABELS_TS = os.path.join(
    OUTPUT_DIR,
    "labelsTs"
)


# =====================================================================
# BASIC HELPERS
# =====================================================================

def maybe_mkdir_p(directory):
    os.makedirs(
        directory,
        exist_ok=True
    )


# =====================================================================
# LOAD EXACT SPLIT FROM DATASET910
# =====================================================================

def load_dataset910_split():

    split_file = os.path.join(
        DATASET910_DIR,
        "train_test_split.json"
    )

    if not os.path.isfile(split_file):

        raise FileNotFoundError(
            "\nDataset910 split file not found:\n"
            f"{split_file}"
        )

    with open(
        split_file,
        "r",
        encoding="utf-8"
    ) as f:

        records = json.load(f)

    if len(records) == 0:

        raise RuntimeError(
            "Dataset910 train_test_split.json is empty."
        )

    # Check unique case IDs
    case_ids = [
        r["case_id"]
        for r in records
    ]

    if len(case_ids) != len(set(case_ids)):

        raise RuntimeError(
            "Duplicate case IDs found in Dataset910 split."
        )

    return records


# =====================================================================
# GET DATASET910 FILES
# =====================================================================

def get_dataset910_files(
        case_id,
        split
):

    if split == "train":

        image_file = os.path.join(
            DATASET910_DIR,
            "imagesTr",
            case_id + "_0000.nii.gz"
        )

        label_file = os.path.join(
            DATASET910_DIR,
            "labelsTr",
            case_id + ".nii.gz"
        )

    elif split == "test":

        image_file = os.path.join(
            DATASET910_DIR,
            "imagesTs",
            case_id + "_0000.nii.gz"
        )

        label_file = os.path.join(
            DATASET910_DIR,
            "labelsTs",
            case_id + ".nii.gz"
        )

    else:

        raise ValueError(
            f"Unknown split: {split}"
        )

    if not os.path.isfile(image_file):

        raise FileNotFoundError(
            f"Dataset910 image not found:\n{image_file}"
        )

    if not os.path.isfile(label_file):

        raise FileNotFoundError(
            f"Dataset910 label not found:\n{label_file}"
        )

    return (
        image_file,
        label_file
    )


# =====================================================================
# GET OUTPUT FILES
# =====================================================================

def get_output_files(
        case_id,
        split
):

    if split == "train":

        image_file = os.path.join(
            IMAGES_TR,
            case_id + "_0000.nii.gz"
        )

        label_file = os.path.join(
            LABELS_TR,
            case_id + ".nii.gz"
        )

    elif split == "test":

        image_file = os.path.join(
            IMAGES_TS,
            case_id + "_0000.nii.gz"
        )

        label_file = os.path.join(
            LABELS_TS,
            case_id + ".nii.gz"
        )

    else:

        raise ValueError(
            f"Unknown split: {split}"
        )

    return (
        image_file,
        label_file
    )


# =====================================================================
# GEOMETRY CHECK
# =====================================================================

def check_same_geometry(
        image1,
        image2,
        name1="Image 1",
        name2="Image 2",
        atol=1e-3
):
    """
    Checks shape and affine.
    """

    if image1.shape != image2.shape:

        raise RuntimeError(
            "\n"
            "====================================================\n"
            "SHAPE MISMATCH\n"
            "====================================================\n"
            f"{name1}: {image1.shape}\n"
            f"{name2}: {image2.shape}"
        )

    if not np.allclose(
        image1.affine,
        image2.affine,
        atol=atol,
        rtol=1e-5
    ):

        raise RuntimeError(
            "\n"
            "====================================================\n"
            "AFFINE MISMATCH\n"
            "====================================================\n"
            f"{name1}:\n"
            f"{image1.affine}\n\n"
            f"{name2}:\n"
            f"{image2.affine}"
        )


# =====================================================================
# COPY LUMVBCANSEG WITHOUT CROP
# =====================================================================

def copy_lum_case(record):

    case_id = record["case_id"]
    split = record["split"]

    source_image, source_label = (
        get_dataset910_files(
            case_id,
            split
        )
    )

    output_image, output_label = (
        get_output_files(
            case_id,
            split
        )
    )

    shutil.copy2(
        source_image,
        output_image
    )

    shutil.copy2(
        source_label,
        output_label
    )

    return {

        "case_id":
            case_id,

        "patient_id":
            record["patient_id"],

        "source":
            "LumVBCanSeg",

        "split":
            split,

        "crop_applied":
            False,

        "t10_found":
            "",

        "crop_start_label":
            "",

        "z_stop":
            "",

        "original_z_slices":
            "",

        "cropped_z_slices":
            "",

        "spacing_z_mm":
            "",

        "margin_voxels":
            "",

        "reduction_percent":
            "",
    }


# =====================================================================
# FIND FULL-SPINE SEGMENTATION
# =====================================================================

def get_myeloma_spine_segmentation(
        original_case_id
):

    spine_file = os.path.join(
        MYEL_DIR,
        original_case_id +
        "_spine_segmentation.nii.gz"
    )

    if not os.path.isfile(
        spine_file
    ):

        raise FileNotFoundError(
            "\nFull spine segmentation not found:\n"
            f"{spine_file}"
        )

    return spine_file


# =====================================================================
# DETERMINE CROP BOUNDARY
# =====================================================================

def determine_z_stop_from_t10(
        spine_data,
        spacing_z
):
    """
    spine_data MUST be in RAS orientation.

    RAS means:

        increasing Z index = Superior / cranial

    We want:

        inferior end -> T10 + cranial margin

    therefore:

        [:, :, :z_stop]
    """

    unique_values = np.unique(
        spine_data
    )

    unique_values = np.round(
        unique_values
    ).astype(int)

    labels_present = sorted([
        int(x)
        for x in unique_values.tolist()
        if x != 0
    ])

    labels_set = set(
        labels_present
    )

    # -------------------------------------------------------------
    # T10
    # -------------------------------------------------------------

    if T10_LABEL in labels_set:

        crop_start_label = T10_LABEL
        t10_found = True

    else:

        t10_found = False

        if REQUIRE_T10:

            raise RuntimeError(
                "\n"
                "====================================================\n"
                "T10 NOT FOUND\n"
                "====================================================\n"
                "T10 = label 17 is missing.\n\n"
                f"Labels present:\n"
                f"{labels_present}"
            )

        # More caudal vertebrae have larger label IDs
        candidates = sorted([
            x
            for x in labels_present
            if 17 < x <= 25
        ])

        if len(candidates) == 0:

            raise RuntimeError(
                "\nNo suitable vertebra below T10 found.\n"
                f"Labels present: {labels_present}"
            )

        crop_start_label = (
            candidates[0]
        )

        print(
            f"    WARNING: T10 missing -> "
            f"using label {crop_start_label}"
        )

    # -------------------------------------------------------------
    # Find T10 voxel extent
    # -------------------------------------------------------------

    mask = (
        spine_data ==
        crop_start_label
    )

    coords = np.where(
        mask
    )

    if len(coords[2]) == 0:

        raise RuntimeError(
            f"Mask for label "
            f"{crop_start_label} is empty."
        )

    # In RAS:
    # max Z = superior-most point of T10
    z_max = int(
        np.max(
            coords[2]
        )
    )

    # -------------------------------------------------------------
    # Margin
    # -------------------------------------------------------------

    margin_voxels = int(
        math.ceil(
            CROP_MARGIN_MM /
            float(spacing_z)
        )
    )

    # Python end index is exclusive.
    z_stop = min(
        spine_data.shape[2],
        z_max +
        margin_voxels +
        1
    )

    return {

        "z_stop":
            z_stop,

        "crop_start_label":
            crop_start_label,

        "t10_found":
            t10_found,

        "labels_present":
            labels_present,

        "margin_voxels":
            margin_voxels,
    }


# =====================================================================
# MYELOMA CROP
# =====================================================================

def crop_myeloma_case(
        record
):

    case_id = record["case_id"]

    original_case_id = (
        record["original_case_id"]
    )

    patient_id = (
        record["patient_id"]
    )

    split = (
        record["split"]
    )

    # -------------------------------------------------------------
    # Dataset910 CT and target GT
    # -------------------------------------------------------------

    image_file, target_file = (
        get_dataset910_files(
            case_id,
            split
        )
    )

    # -------------------------------------------------------------
    # Full-spine segmentation
    # -------------------------------------------------------------

    spine_file = (
        get_myeloma_spine_segmentation(
            original_case_id
        )
    )

    # -------------------------------------------------------------
    # Load
    # -------------------------------------------------------------

    image = nib.load(
        image_file
    )

    target = nib.load(
        target_file
    )

    spine = nib.load(
        spine_file
    )

    # -------------------------------------------------------------
    # Check geometry BEFORE RAS
    # -------------------------------------------------------------

    check_same_geometry(
        image,
        target,
        "CT",
        "Trabecular segmentation"
    )

    check_same_geometry(
        image,
        spine,
        "CT",
        "Full-spine segmentation"
    )

    # -------------------------------------------------------------
    # Reorient all images consistently to RAS
    # -------------------------------------------------------------

    image_ras = (
        nib.as_closest_canonical(
            image
        )
    )

    target_ras = (
        nib.as_closest_canonical(
            target
        )
    )

    spine_ras = (
        nib.as_closest_canonical(
            spine
        )
    )

    # -------------------------------------------------------------
    # Verify true RAS
    # -------------------------------------------------------------

    axcodes = nib.aff2axcodes(
        image_ras.affine
    )

    if axcodes != (
        "R",
        "A",
        "S"
    ):

        raise RuntimeError(
            "\nExpected RAS orientation but got:\n"
            f"{axcodes}"
        )

    # -------------------------------------------------------------
    # Geometry AFTER RAS
    # -------------------------------------------------------------

    check_same_geometry(
        image_ras,
        target_ras,
        "RAS CT",
        "RAS trabecular segmentation"
    )

    check_same_geometry(
        image_ras,
        spine_ras,
        "RAS CT",
        "RAS full-spine segmentation"
    )

    # -------------------------------------------------------------
    # Full-spine segmentation data
    # -------------------------------------------------------------

    spine_data = np.asanyarray(
        spine_ras.dataobj
    )

    # -------------------------------------------------------------
    # Original target labels, for safety check
    # -------------------------------------------------------------

    target_data_before = np.asanyarray(
        target_ras.dataobj
    )

    target_labels_before = set(
        np.unique(
            target_data_before
        ).astype(int).tolist()
    )

    target_labels_before.discard(
        0
    )

    # -------------------------------------------------------------
    # Spacing
    # -------------------------------------------------------------

    spacing = (
        image_ras.header.get_zooms()[:3]
    )

    spacing_z = float(
        spacing[2]
    )

    # -------------------------------------------------------------
    # Determine superior crop limit
    # -------------------------------------------------------------

    crop_info = (
        determine_z_stop_from_t10(
            spine_data,
            spacing_z
        )
    )

    z_stop = (
        crop_info["z_stop"]
    )

    # =============================================================
    # CORRECT CROP
    # =============================================================
    #
    # RAS:
    #
    # low z                                        high z
    #
    # inferior / caudal          superior / cranial
    #
    # pelvis-L5-L4-L3-L2-L1-T12-T11-T10-...-C1
    #
    # We want to keep:
    #
    # |------------|
    # pelvis ... T10 + margin
    #
    # Therefore:
    #
    # [:, :, :z_stop]
    #
    # =============================================================

    cropped_image = (
        image_ras.slicer[
            :,
            :,
            :z_stop
        ]
    )

    cropped_target = (
        target_ras.slicer[
            :,
            :,
            :z_stop
        ]
    )

    cropped_spine = (
        spine_ras.slicer[
            :,
            :,
            :z_stop
        ]
    )

    # -------------------------------------------------------------
    # Geometry after crop
    # -------------------------------------------------------------

    check_same_geometry(
        cropped_image,
        cropped_target,
        "Cropped CT",
        "Cropped trabecular segmentation"
    )

    check_same_geometry(
        cropped_image,
        cropped_spine,
        "Cropped CT",
        "Cropped full-spine segmentation"
    )

    # -------------------------------------------------------------
    # Inspect labels after crop
    # -------------------------------------------------------------

    cropped_spine_data = np.asanyarray(
        cropped_spine.dataobj
    )

    cropped_spine_labels = sorted([
        int(x)
        for x in np.unique(
            cropped_spine_data
        ).tolist()
        if x != 0
    ])

    print(
        f"    spine labels before crop: "
        f"{crop_info['labels_present']}"
    )

    print(
        f"    spine labels after crop:  "
        f"{cropped_spine_labels}"
    )

    # -------------------------------------------------------------
    # Check target segmentation labels were not lost
    # -------------------------------------------------------------

    target_data_after = np.asanyarray(
        cropped_target.dataobj
    )

    target_labels_after = set(
        np.unique(
            target_data_after
        ).astype(int).tolist()
    )

    target_labels_after.discard(
        0
    )

    lost_target_labels = (
        target_labels_before -
        target_labels_after
    )

    if lost_target_labels:

        raise RuntimeError(
            "\n"
            "====================================================\n"
            "TARGET LABEL LOST DURING CROP\n"
            "====================================================\n"
            f"Case: {case_id}\n"
            f"Labels before: {sorted(target_labels_before)}\n"
            f"Labels after:  {sorted(target_labels_after)}\n"
            f"Lost labels:   {sorted(lost_target_labels)}\n\n"
            "The crop would remove part of the target segmentation."
        )

    # -------------------------------------------------------------
    # Output files
    # -------------------------------------------------------------

    output_image, output_target = (
        get_output_files(
            case_id,
            split
        )
    )

    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------

    nib.save(
        cropped_image,
        output_image
    )

    nib.save(
        cropped_target,
        output_target
    )

    # -------------------------------------------------------------
    # Reload for verification
    # -------------------------------------------------------------

    saved_image = nib.load(
        output_image
    )

    saved_target = nib.load(
        output_target
    )

    check_same_geometry(
        saved_image,
        saved_target,
        "Saved CT",
        "Saved target segmentation"
    )

    # -------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------

    original_z = (
        image_ras.shape[2]
    )

    cropped_z = (
        saved_image.shape[2]
    )

    reduction_percent = (
        100.0 *
        (
            1.0 -
            cropped_z /
            original_z
        )
    )

    print(
        f"    T10 found: "
        f"{crop_info['t10_found']}"
    )

    print(
        f"    crop start label: "
        f"{crop_info['crop_start_label']}"
    )

    print(
        f"    spacing Z: "
        f"{spacing_z:.3f} mm"
    )

    print(
        f"    margin: "
        f"{crop_info['margin_voxels']} voxels "
        f"({CROP_MARGIN_MM:.1f} mm)"
    )

    print(
        f"    z_stop: "
        f"{z_stop}"
    )

    print(
        f"    slices: "
        f"{original_z} -> {cropped_z}"
    )

    print(
        f"    reduction: "
        f"{reduction_percent:.1f}%"
    )

    return {

        "case_id":
            case_id,

        "patient_id":
            patient_id,

        "source":
            "Myeloma",

        "split":
            split,

        "crop_applied":
            True,

        "t10_found":
            crop_info["t10_found"],

        "crop_start_label":
            crop_info["crop_start_label"],

        "z_stop":
            z_stop,

        "original_z_slices":
            original_z,

        "cropped_z_slices":
            cropped_z,

        "spacing_z_mm":
            spacing_z,

        "margin_voxels":
            crop_info["margin_voxels"],

        "reduction_percent":
            round(
                reduction_percent,
                2
            ),
    }


# =====================================================================
# DATASET.JSON
# =====================================================================

def create_dataset_json(
        num_training_cases
):

    dataset_json = {

        "name":
            TASK_NAME,

        "description":
            "Trabecular lumbar vertebra segmentation. "
            "Dataset911 uses the exact same train/test split as Dataset910. "
            "Myeloma CT volumes are cropped in the cranio-caudal direction "
            "from the inferior end of the scan through T10.",

        "channel_names": {
            "0": "CT"
        },

        "labels": {
            "background": 0,
            "L1": 1,
            "L2": 2,
            "L3": 3,
            "L4": 4,
            "L5": 5,
            "L6": 6
        },

        "numTraining":
            num_training_cases,

        "file_ending":
            ".nii.gz",

        "overwrite_image_reader_writer":
            "NibabelIOWithReorient"
    }

    output_file = os.path.join(
        OUTPUT_DIR,
        "dataset.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            dataset_json,
            f,
            indent=4,
            ensure_ascii=False
        )


# =====================================================================
# CROP REPORT
# =====================================================================

def save_crop_report(
        rows
):

    filename = os.path.join(
        OUTPUT_DIR,
        "crop_parameters.csv"
    )

    fields = [

        "case_id",
        "patient_id",
        "source",
        "split",
        "crop_applied",
        "t10_found",
        "crop_start_label",
        "z_stop",
        "original_z_slices",
        "cropped_z_slices",
        "spacing_z_mm",
        "margin_voxels",
        "reduction_percent",
    ]

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()

        for row in rows:

            writer.writerow(
                row
            )


# =====================================================================
# COPY METADATA FROM DATASET910
# =====================================================================

def copy_dataset910_metadata():

    files_to_copy = [

        "train_test_split.json",
        "dataset_cases.csv",
        "qc_labels.csv",
        "qc_missing_labels.csv",
    ]

    for filename in files_to_copy:

        source = os.path.join(
            DATASET910_DIR,
            filename
        )

        destination = os.path.join(
            OUTPUT_DIR,
            filename
        )

        if os.path.isfile(
            source
        ):

            shutil.copy2(
                source,
                destination
            )


# =====================================================================
# VERIFY OUTPUT
# =====================================================================

def verify_output(
        records
):

    expected_train = [
        x
        for x in records
        if x["split"] == "train"
    ]

    expected_test = [
        x
        for x in records
        if x["split"] == "test"
    ]

    actual_train_images = [
        x
        for x in os.listdir(
            IMAGES_TR
        )
        if x.endswith(
            "_0000.nii.gz"
        )
    ]

    actual_train_labels = [
        x
        for x in os.listdir(
            LABELS_TR
        )
        if x.endswith(
            ".nii.gz"
        )
    ]

    actual_test_images = [
        x
        for x in os.listdir(
            IMAGES_TS
        )
        if x.endswith(
            "_0000.nii.gz"
        )
    ]

    actual_test_labels = [
        x
        for x in os.listdir(
            LABELS_TS
        )
        if x.endswith(
            ".nii.gz"
        )
    ]

    if len(actual_train_images) != len(expected_train):

        raise RuntimeError(
            "\nTraining image count mismatch:\n"
            f"Expected: {len(expected_train)}\n"
            f"Found:    {len(actual_train_images)}"
        )

    if len(actual_train_labels) != len(expected_train):

        raise RuntimeError(
            "\nTraining label count mismatch:\n"
            f"Expected: {len(expected_train)}\n"
            f"Found:    {len(actual_train_labels)}"
        )

    if len(actual_test_images) != len(expected_test):

        raise RuntimeError(
            "\nTest image count mismatch:\n"
            f"Expected: {len(expected_test)}\n"
            f"Found:    {len(actual_test_images)}"
        )

    if len(actual_test_labels) != len(expected_test):

        raise RuntimeError(
            "\nTest label count mismatch:\n"
            f"Expected: {len(expected_test)}\n"
            f"Found:    {len(actual_test_labels)}"
        )

    print()
    print("=" * 80)
    print("OUTPUT VERIFICATION")
    print("=" * 80)

    print(
        f"Training cases: "
        f"{len(expected_train)}"
    )

    print(
        f"Testing cases:  "
        f"{len(expected_test)}"
    )

    print(
        "Case counts: OK"
    )


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":

    print()
    print("=" * 80)
    print(DATASET_NAME)
    print("=" * 80)

    print()
    print(
        "Source Dataset910:"
    )

    print(
        DATASET910_DIR
    )

    print()
    print(
        "Myeloma spine segmentations:"
    )

    print(
        MYEL_DIR
    )

    print()
    print(
        "Output Dataset911:"
    )

    print(
        OUTPUT_DIR
    )

    # =================================================================
    # Existing Dataset911
    # =================================================================

    if os.path.exists(
        OUTPUT_DIR
    ):

        if OVERWRITE_DATASET:

            print()
            print(
                "Removing existing Dataset911..."
            )

            shutil.rmtree(
                OUTPUT_DIR
            )

        else:

            raise RuntimeError(
                "\n"
                "====================================================\n"
                "DATASET911 ALREADY EXISTS\n"
                "====================================================\n"
                f"{OUTPUT_DIR}\n\n"
                "Set:\n\n"
                "    OVERWRITE_DATASET = True\n\n"
                "to recreate Dataset911."
            )

    # =================================================================
    # Dataset910 must exist
    # =================================================================

    if not os.path.isdir(
        DATASET910_DIR
    ):

        raise FileNotFoundError(
            "\nDataset910 not found:\n"
            f"{DATASET910_DIR}"
        )

    # =================================================================
    # Load exact train/test split
    # =================================================================

    records = (
        load_dataset910_split()
    )

    print()
    print(
        f"Cases loaded from Dataset910: "
        f"{len(records)}"
    )

    # =================================================================
    # Create folders
    # =================================================================

    maybe_mkdir_p(
        IMAGES_TR
    )

    maybe_mkdir_p(
        LABELS_TR
    )

    maybe_mkdir_p(
        IMAGES_TS
    )

    maybe_mkdir_p(
        LABELS_TS
    )

    # =================================================================
    # Process cases
    # =================================================================

    crop_report = []

    for index, record in enumerate(
        records,
        start=1
    ):

        print()
        print("=" * 80)

        print(
            f"{index:4d}/{len(records):4d}  "
            f"{record['case_id']}  "
            f"[{record['source']}, "
            f"{record['split']}]"
        )

        # -------------------------------------------------------------
        # LumVBCanSeg
        # -------------------------------------------------------------

        if (
            record["source"]
            == "LumVBCanSeg"
        ):

            print(
                "    LumVBCanSeg -> "
                "copy without crop"
            )

            report = (
                copy_lum_case(
                    record
                )
            )

        # -------------------------------------------------------------
        # Myeloma
        # -------------------------------------------------------------

        elif (
            record["source"]
            == "Myeloma"
        ):

            print(
                "    Myeloma -> "
                "crop inferior scan through T10 + margin"
            )

            report = (
                crop_myeloma_case(
                    record
                )
            )

        else:

            raise RuntimeError(
                "\nUnknown source:\n"
                f"{record['source']}"
            )

        crop_report.append(
            report
        )

    # =================================================================
    # Dataset JSON
    # =================================================================

    num_training_cases = sum(
        record["split"] == "train"
        for record in records
    )

    create_dataset_json(
        num_training_cases=
            num_training_cases
    )

    # =================================================================
    # Copy original split/QC metadata
    # =================================================================

    copy_dataset910_metadata()

    # =================================================================
    # Save crop report
    # =================================================================

    save_crop_report(
        crop_report
    )

    # =================================================================
    # Verify
    # =================================================================

    verify_output(
        records
    )

    # =================================================================
    # DONE
    # =================================================================

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)

    print()
    print(
        "Dataset911 created:"
    )

    print(
        OUTPUT_DIR
    )

    print()
    print(
        "Crop report:"
    )

    print(
        os.path.join(
            OUTPUT_DIR,
            "crop_parameters.csv"
        )
    )

    print()
    print(
        "After visual inspection run:"
    )

    print()
    print(
        "nnUNetv2_plan_and_preprocess "
        "-d 911 "
        "--verify_dataset_integrity "
        "-np 2"
    )