# -*- coding: utf-8 -*-

"""
Create nnU-Net v2 dataset:

    Dataset910_LumbarTrabecularSeg_FullFOV

Sources
-------
1) LumVBCanSeg
   F:/BVM_2027_DATA/Task908

   Example:
       101002964255.nii.gz
       101002964255_seg.nii.gz


2) Spinal-Multiple-Myeloma-SEG
   F:/BVM_2027_DATA/Spinal-Multiple-Myeloma-SEG

   Example:
       Myel_001_konv.nii.gz
       Myel_001_trabecular_spine_segmentation.nii.gz

   Follow-up example:
       Myel_012_a_konv.nii.gz
       Myel_012_a_trabecular_spine_segmentation.nii.gz

       Myel_012_b_konv.nii.gz
       Myel_012_b_trabecular_spine_segmentation.nii.gz

   Myel_012_a and Myel_012_b belong to the same patient:
       Myel_012


Train/test split
----------------
- approximately 80 % training / 20 % testing
- split is performed at PATIENT level
- LumVBCanSeg and Myeloma datasets are split independently
- follow-ups of the same patient always stay together
- L6 patients:
      Myel_021
      Myel_055
      Myel_060

  are explicitly split:
      2 patients -> training
      1 patient  -> testing


Labels
------
0 = background
1 = L1
2 = L2
3 = L3
4 = L4
5 = L5
6 = L6


Output
------
Dataset910_LumbarTrabecularSeg_FullFOV/
    imagesTr/
    labelsTr/
    imagesTs/
    labelsTs/

    dataset.json
    train_test_split.json
    dataset_cases.csv
    qc_labels.csv
    qc_missing_labels.csv


Important
---------
Missing L1-L5 labels are NOT automatically considered an error.
They are reported in qc_missing_labels.csv for manual review.

The NIfTI files are NOT physically reoriented.
nnU-Net is instructed to use:

    NibabelIOWithReorient
"""

import os
import re
import csv
import json
import shutil
import random

from collections import defaultdict

import nibabel as nib
import numpy as np


# =====================================================================
# SETTINGS
# =====================================================================

BASE_DIR = r"F:\BVM_2027_DATA"

LUM_DIR = os.path.join(
    BASE_DIR,
    "LumVBCanSeg/Task908"
)

MYEL_DIR = os.path.join(
    BASE_DIR,
    "Spinal-Multiple-Myeloma-SEG"
)


# =====================================================================
# nnU-Net output
# =====================================================================

NNUNET_RAW = r"E:\nnUNet_v2_MAIN_FILE\nnUNet_raw"

TASK_ID = 910

TASK_NAME = "LumbarTrabecularSeg_FullFOV"

DATASET_NAME = (
    f"Dataset{TASK_ID:03d}_{TASK_NAME}"
)

OUTPUT_DIR = os.path.join(
    NNUNET_RAW,
    DATASET_NAME
)


# =====================================================================
# SPLIT SETTINGS
# =====================================================================

TRAIN_FRACTION = 0.80

# Reproducible random split
RANDOM_SEED = 42


# =====================================================================
# EXISTING DATASET
# =====================================================================

# False:
#     stop if Dataset910 already exists
#
# True:
#     completely delete Dataset910 and recreate it
#
OVERWRITE_DATASET = False


# =====================================================================
# L6 PATIENTS
# =====================================================================

L6_PATIENTS = {
    "Myel_021",
    "Myel_055",
    "Myel_060",
}

# Exactly one of the three L6 patients goes to test
N_L6_TEST_PATIENTS = 1


# =====================================================================
# LABEL DEFINITIONS
# =====================================================================

LABEL_NAMES = {
    0: "background",
    1: "L1",
    2: "L2",
    3: "L3",
    4: "L4",
    5: "L5",
    6: "L6",
}

ALLOWED_LABELS = set(
    LABEL_NAMES.keys()
)

# Normally expected lumbar vertebrae.
#
# Their absence is reported, but dataset creation does not stop.
#
USUAL_LUMBAR_LABELS = {
    1,
    2,
    3,
    4,
    5,
}


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

# nnU-Net does not require labelsTs for prediction,
# but we keep them for independent hold-out evaluation.
LABELS_TS = os.path.join(
    OUTPUT_DIR,
    "labelsTs"
)


# =====================================================================
# BASIC FUNCTIONS
# =====================================================================

def maybe_mkdir_p(directory):
    """
    Create directory if it does not exist.
    """

    os.makedirs(
        directory,
        exist_ok=True
    )


# =====================================================================
# SEGMENTATION LABELS
# =====================================================================

def get_unique_segmentation_values(filename):
    """
    Return all unique integer labels occurring in segmentation.
    """

    img = nib.load(
        filename
    )

    data = np.asanyarray(
        img.dataobj
    )

    values = np.unique(
        data
    )

    rounded = np.round(
        values
    )

    # Segmentation should contain integer values only
    if not np.allclose(
        values,
        rounded
    ):
        raise RuntimeError(
            "\nSegmentation contains non-integer values.\n"
            f"File:\n{filename}\n\n"
            f"Values:\n{values}"
        )

    return set(
        rounded.astype(int).tolist()
    )


# =====================================================================
# GEOMETRY CHECK
# =====================================================================

def check_same_geometry(
        image_filename,
        label_filename
):
    """
    Basic CT/GT geometry check.

    Checks:
        1) same shape
        2) same affine

    Nothing is modified.
    """

    image = nib.load(
        image_filename
    )

    label = nib.load(
        label_filename
    )

    # -------------------------------------------------------------
    # Shape
    # -------------------------------------------------------------

    if image.shape != label.shape:

        raise RuntimeError(
            "\n"
            "====================================================\n"
            "IMAGE / LABEL SHAPE MISMATCH\n"
            "====================================================\n"
            f"Image:\n{image_filename}\n"
            f"shape = {image.shape}\n\n"
            f"Label:\n{label_filename}\n"
            f"shape = {label.shape}\n"
        )

    # -------------------------------------------------------------
    # Affine
    # -------------------------------------------------------------

    if not np.allclose(
        image.affine,
        label.affine,
        atol=1e-4,
        rtol=1e-5
    ):

        raise RuntimeError(
            "\n"
            "====================================================\n"
            "IMAGE / LABEL AFFINE MISMATCH\n"
            "====================================================\n"
            f"Image:\n{image_filename}\n\n"
            f"Label:\n{label_filename}\n"
        )


# =====================================================================
# MYELOMA PATIENT IDENTIFICATION
# =====================================================================

def get_myeloma_patient_id(case_id):
    """
    Convert a Myeloma examination ID to patient ID.

    Examples
    --------
    Myel_001
        -> Myel_001

    Myel_012_a
        -> Myel_012

    Myel_012_b
        -> Myel_012

    Myel_055_a
        -> Myel_055
    """

    match = re.match(
        r"^(Myel_\d+)",
        case_id,
        flags=re.IGNORECASE
    )

    if match is None:

        raise ValueError(
            "\nCannot determine Myeloma patient ID from:\n"
            f"{case_id}"
        )

    patient_id = match.group(1)

    number = patient_id.split("_")[1]

    # Standardized capitalization
    patient_id = (
        "Myel_" + number
    )

    return patient_id


# =====================================================================
# FIND LUMVBCANSEG CASES
# =====================================================================

def find_lum_cases():
    """
    Find LumVBCanSeg cases.

    Expected:
        101002964255.nii.gz
        101002964255_seg.nii.gz
    """

    if not os.path.isdir(
        LUM_DIR
    ):

        raise FileNotFoundError(
            "\nLumVBCanSeg directory not found:\n"
            f"{LUM_DIR}"
        )

    cases = []

    segmentation_suffix = (
        "_seg.nii.gz"
    )

    for filename in sorted(
        os.listdir(LUM_DIR)
    ):

        if not filename.endswith(
            segmentation_suffix
        ):
            continue

        original_case_id = (
            filename[
                :-len(segmentation_suffix)
            ]
        )

        image_file = os.path.join(
            LUM_DIR,
            original_case_id + ".nii.gz"
        )

        label_file = os.path.join(
            LUM_DIR,
            original_case_id + "_seg.nii.gz"
        )

        # ---------------------------------------------------------
        # Check files
        # ---------------------------------------------------------

        if not os.path.isfile(
            image_file
        ):

            raise FileNotFoundError(
                "\nMissing LumVBCanSeg CT:\n"
                f"{image_file}"
            )

        if not os.path.isfile(
            label_file
        ):

            raise FileNotFoundError(
                "\nMissing LumVBCanSeg segmentation:\n"
                f"{label_file}"
            )

        # ---------------------------------------------------------
        # Prefix avoids possible name collision
        # ---------------------------------------------------------

        nnunet_case_id = (
            "Lum_" +
            original_case_id
        )

        cases.append({

            "source":
                "LumVBCanSeg",

            "original_case_id":
                original_case_id,

            "case_id":
                nnunet_case_id,

            # One examination = one patient here
            "patient_id":
                original_case_id,

            "image":
                image_file,

            "label":
                label_file,
        })

    if len(cases) == 0:

        raise RuntimeError(
            "\nNo LumVBCanSeg cases were found in:\n"
            f"{LUM_DIR}"
        )

    return cases


# =====================================================================
# FIND MYELOMA CASES
# =====================================================================

def find_myeloma_cases():
    """
    Find Myeloma cases.

    Expected CT:
        Myel_001_konv.nii.gz

    Expected GT:
        Myel_001_trabecular_spine_segmentation.nii.gz

    Follow-up:
        Myel_012_a_konv.nii.gz
        Myel_012_b_konv.nii.gz
    """

    if not os.path.isdir(
        MYEL_DIR
    ):

        raise FileNotFoundError(
            "\nMyeloma directory not found:\n"
            f"{MYEL_DIR}"
        )

    cases = []

    image_suffix = (
        "_konv.nii.gz"
    )

    for filename in sorted(
        os.listdir(MYEL_DIR)
    ):

        if not filename.endswith(
            image_suffix
        ):
            continue

        original_case_id = (
            filename[
                :-len(image_suffix)
            ]
        )

        image_file = os.path.join(
            MYEL_DIR,
            original_case_id +
            "_konv.nii.gz"
        )

        label_file = os.path.join(
            MYEL_DIR,
            original_case_id +
            "_trabecular_spine_segmentation.nii.gz"
        )

        # ---------------------------------------------------------
        # Check GT
        # ---------------------------------------------------------

        if not os.path.isfile(
            label_file
        ):

            raise FileNotFoundError(
                "\nMissing Myeloma segmentation:\n\n"
                f"Case:\n"
                f"{original_case_id}\n\n"
                f"Expected:\n"
                f"{label_file}"
            )

        # ---------------------------------------------------------
        # Patient ID
        # ---------------------------------------------------------

        patient_id = (
            get_myeloma_patient_id(
                original_case_id
            )
        )

        # No additional prefix needed because these already
        # start with Myel_ and cannot collide with Lum_ cases.
        nnunet_case_id = (
            original_case_id
        )

        cases.append({

            "source":
                "Myeloma",

            "original_case_id":
                original_case_id,

            "case_id":
                nnunet_case_id,

            "patient_id":
                patient_id,

            "image":
                image_file,

            "label":
                label_file,
        })

    if len(cases) == 0:

        raise RuntimeError(
            "\nNo Myeloma cases were found in:\n"
            f"{MYEL_DIR}"
        )

    return cases


# =====================================================================
# GENERIC PATIENT-LEVEL SPLIT
# =====================================================================

def split_patients(
        patient_ids,
        train_fraction=0.8,
        random_seed=42
):
    """
    Reproducible patient-level train/test split.
    """

    patient_ids = sorted(
        set(patient_ids)
    )

    if len(patient_ids) < 2:

        raise RuntimeError(
            "At least two patients are required "
            "for train/test split."
        )

    rng = random.Random(
        random_seed
    )

    rng.shuffle(
        patient_ids
    )

    n_train = round(
        len(patient_ids) *
        train_fraction
    )

    # Prevent empty train or test sets
    n_train = max(
        1,
        min(
            n_train,
            len(patient_ids) - 1
        )
    )

    train_patients = set(
        patient_ids[:n_train]
    )

    test_patients = set(
        patient_ids[n_train:]
    )

    return (
        train_patients,
        test_patients
    )


# =====================================================================
# MYELOMA SPLIT WITH L6 STRATIFICATION
# =====================================================================

def split_myeloma_patients(
        myel_patients
):
    """
    Myeloma split with explicit handling of rare L6 patients.

    L6:
        Myel_021
        Myel_055
        Myel_060

    Desired:
        2 -> train
        1 -> test
    """

    myel_patients = set(
        myel_patients
    )

    # -------------------------------------------------------------
    # Verify L6 patients
    # -------------------------------------------------------------

    missing_l6 = (
        L6_PATIENTS -
        myel_patients
    )

    if missing_l6:

        raise RuntimeError(
            "\n"
            "====================================================\n"
            "EXPECTED L6 PATIENTS WERE NOT FOUND\n"
            "====================================================\n"
            f"{sorted(missing_l6)}\n\n"
            "Check patient names or update L6_PATIENTS."
        )

    # -------------------------------------------------------------
    # Explicit L6 split
    # -------------------------------------------------------------

    l6_patients = sorted(
        L6_PATIENTS
    )

    rng_l6 = random.Random(
        RANDOM_SEED + 100
    )

    rng_l6.shuffle(
        l6_patients
    )

    l6_test_patients = set(
        l6_patients[
            :N_L6_TEST_PATIENTS
        ]
    )

    l6_train_patients = set(
        l6_patients[
            N_L6_TEST_PATIENTS:
        ]
    )

    # -------------------------------------------------------------
    # Remaining Myeloma patients
    # -------------------------------------------------------------

    regular_patients = (
        myel_patients -
        L6_PATIENTS
    )

    (
        regular_train,
        regular_test
    ) = split_patients(

        regular_patients,

        train_fraction=
            TRAIN_FRACTION,

        random_seed=
            RANDOM_SEED + 1
    )

    # -------------------------------------------------------------
    # Combine
    # -------------------------------------------------------------

    train_patients = (
        regular_train |
        l6_train_patients
    )

    test_patients = (
        regular_test |
        l6_test_patients
    )

    return (
        train_patients,
        test_patients,
        l6_train_patients,
        l6_test_patients
    )


# =====================================================================
# QC
# =====================================================================

def perform_qc(cases):
    """
    Check all source data before copying.

    Checks:
        - CT / segmentation geometry
        - integer segmentation values
        - unexpected labels

    Records:
        - labels present
        - missing L1-L5
        - L6 presence
    """

    print()
    print("=" * 80)
    print("QC: IMAGE GEOMETRY AND SEGMENTATION LABELS")
    print("=" * 80)

    for index, case in enumerate(
        cases,
        start=1
    ):

        print(
            f"{index:4d}/{len(cases):4d}  "
            f"{case['case_id']:30s}",
            end=""
        )

        # ---------------------------------------------------------
        # Geometry
        # ---------------------------------------------------------

        check_same_geometry(
            case["image"],
            case["label"]
        )

        # ---------------------------------------------------------
        # Labels
        # ---------------------------------------------------------

        values = (
            get_unique_segmentation_values(
                case["label"]
            )
        )

        unexpected = (
            values -
            ALLOWED_LABELS
        )

        if unexpected:

            raise RuntimeError(
                "\n\n"
                "====================================================\n"
                "UNEXPECTED SEGMENTATION LABELS\n"
                "====================================================\n"
                f"Case:\n{case['case_id']}\n\n"
                f"File:\n{case['label']}\n\n"
                f"All labels:\n{sorted(values)}\n\n"
                f"Unexpected:\n{sorted(unexpected)}"
            )

        foreground = (
            values -
            {0}
        )

        missing_usual = (
            USUAL_LUMBAR_LABELS -
            foreground
        )

        case["labels_present"] = sorted(
            foreground
        )

        case["usual_labels_missing"] = sorted(
            missing_usual
        )

        case["has_L6"] = (
            6 in foreground
        )

        present_names = [
            LABEL_NAMES[x]
            for x in sorted(foreground)
        ]

        print(
            f"  labels = {present_names}"
        )

    print()
    print("QC completed successfully.")


# =====================================================================
# COPY CASE
# =====================================================================

def copy_case(case):
    """
    Copy case to nnU-Net structure.
    """

    case_id = case["case_id"]

    if case["split"] == "train":

        image_destination = os.path.join(
            IMAGES_TR,
            case_id + "_0000.nii.gz"
        )

        label_destination = os.path.join(
            LABELS_TR,
            case_id + ".nii.gz"
        )

    elif case["split"] == "test":

        image_destination = os.path.join(
            IMAGES_TS,
            case_id + "_0000.nii.gz"
        )

        label_destination = os.path.join(
            LABELS_TS,
            case_id + ".nii.gz"
        )

    else:

        raise RuntimeError(
            f"Unknown split: "
            f"{case['split']}"
        )

    shutil.copy2(
        case["image"],
        image_destination
    )

    shutil.copy2(
        case["label"],
        label_destination
    )


# =====================================================================
# DATASET.JSON
# =====================================================================

def create_dataset_json(
        num_training_cases
):
    """
    Create nnU-Net v2 dataset.json.

    NibabelIOWithReorient is used so input NIfTI files do not need
    to be physically reoriented before preprocessing.
    """

    dataset_json = {

        "name":
            TASK_NAME,

        "description":
            "Trabecular lumbar vertebra segmentation combining "
            "LumVBCanSeg and Spinal-Multiple-Myeloma-SEG. "
            "Full field-of-view CT input.",

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

    output_filename = os.path.join(
        OUTPUT_DIR,
        "dataset.json"
    )

    with open(
        output_filename,
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
# SAVE TRAIN / TEST SPLIT
# =====================================================================

def save_split_json(cases):
    """
    Save complete reproducible train/test assignment.
    """

    records = []

    for case in sorted(
        cases,
        key=lambda x: x["case_id"]
    ):

        records.append({

            "case_id":
                case["case_id"],

            "original_case_id":
                case["original_case_id"],

            "patient_id":
                case["patient_id"],

            "source":
                case["source"],

            "split":
                case["split"],

            "labels_present":
                case["labels_present"],

            "usual_labels_missing":
                case["usual_labels_missing"],

            "has_L6":
                case["has_L6"],
        })

    output_filename = os.path.join(
        OUTPUT_DIR,
        "train_test_split.json"
    )

    with open(
        output_filename,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            records,
            f,
            indent=4,
            ensure_ascii=False
        )


# =====================================================================
# SAVE COMPLETE CASE CSV
# =====================================================================

def save_case_csv(cases):
    """
    Save complete overview of all included scans.
    """

    output_filename = os.path.join(
        OUTPUT_DIR,
        "dataset_cases.csv"
    )

    fieldnames = [
        "case_id",
        "original_case_id",
        "patient_id",
        "source",
        "split",
        "has_L6",
        "labels_present",
        "usual_labels_missing",
        "source_image",
        "source_label",
    ]

    with open(
        output_filename,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for case in sorted(
            cases,
            key=lambda x: x["case_id"]
        ):

            writer.writerow({

                "case_id":
                    case["case_id"],

                "original_case_id":
                    case["original_case_id"],

                "patient_id":
                    case["patient_id"],

                "source":
                    case["source"],

                "split":
                    case["split"],

                "has_L6":
                    case["has_L6"],

                "labels_present":
                    ";".join(
                        str(x)
                        for x
                        in case["labels_present"]
                    ),

                "usual_labels_missing":
                    ";".join(
                        str(x)
                        for x
                        in case["usual_labels_missing"]
                    ),

                "source_image":
                    case["image"],

                "source_label":
                    case["label"],
            })


# =====================================================================
# SAVE LABEL PRESENCE TABLE
# =====================================================================

def save_qc_csv(cases):
    """
    One row per case.
    1 = label exists in GT.
    0 = label does not exist in GT.
    """

    output_filename = os.path.join(
        OUTPUT_DIR,
        "qc_labels.csv"
    )

    fields = [
        "case_id",
        "patient_id",
        "source",
        "split",
        "L1",
        "L2",
        "L3",
        "L4",
        "L5",
        "L6",
    ]

    with open(
        output_filename,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()

        for case in sorted(
            cases,
            key=lambda x: x["case_id"]
        ):

            present = set(
                case["labels_present"]
            )

            writer.writerow({

                "case_id":
                    case["case_id"],

                "patient_id":
                    case["patient_id"],

                "source":
                    case["source"],

                "split":
                    case["split"],

                "L1":
                    int(1 in present),

                "L2":
                    int(2 in present),

                "L3":
                    int(3 in present),

                "L4":
                    int(4 in present),

                "L5":
                    int(5 in present),

                "L6":
                    int(6 in present),
            })


# =====================================================================
# SAVE ONLY CASES WITH MISSING L1-L5
# =====================================================================

def save_missing_label_qc(cases):
    """
    Save shortlist for manual inspection.

    review_status can later be manually filled with:

        OK_TRUE_ABSENCE
        OUTSIDE_FOV
        NON_EVALUABLE
        GT_INCOMPLETE

    Nothing is automatically excluded.
    """

    output_filename = os.path.join(
        OUTPUT_DIR,
        "qc_missing_labels.csv"
    )

    fields = [
        "case_id",
        "patient_id",
        "source",
        "split",
        "missing_labels",
        "labels_present",
        "review_status",
        "review_comment",
    ]

    with open(
        output_filename,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()

        for case in sorted(
            cases,
            key=lambda x: x["case_id"]
        ):

            missing = (
                case["usual_labels_missing"]
            )

            if not missing:
                continue

            writer.writerow({

                "case_id":
                    case["case_id"],

                "patient_id":
                    case["patient_id"],

                "source":
                    case["source"],

                "split":
                    case["split"],

                "missing_labels":
                    ";".join(
                        LABEL_NAMES[x]
                        for x in missing
                    ),

                "labels_present":
                    ";".join(
                        LABEL_NAMES[x]
                        for x
                        in case["labels_present"]
                    ),

                "review_status":
                    "",

                "review_comment":
                    "",
            })


# =====================================================================
# PATIENT LEAKAGE CHECK
# =====================================================================

def check_patient_leakage(cases):
    """
    Verify that one patient never occurs in both train and test.
    """

    grouping = defaultdict(
        set
    )

    for case in cases:

        key = (
            case["source"],
            case["patient_id"]
        )

        grouping[key].add(
            case["split"]
        )

    problems = []

    for patient, splits in grouping.items():

        if len(splits) > 1:

            problems.append(
                (
                    patient,
                    splits
                )
            )

    if problems:

        raise RuntimeError(
            "\n"
            "====================================================\n"
            "PATIENT DATA LEAKAGE DETECTED\n"
            "====================================================\n"
            f"{problems}"
        )

    print()
    print(
        "Patient leakage check: OK"
    )


# =====================================================================
# STATISTICS
# =====================================================================

def print_statistics(cases):
    """
    Print final split and label statistics.
    """

    training = [
        x
        for x in cases
        if x["split"] == "train"
    ]

    testing = [
        x
        for x in cases
        if x["split"] == "test"
    ]

    print()
    print("=" * 80)
    print("FINAL DATASET STATISTICS")
    print("=" * 80)

    print()
    print(
        f"All scans:      {len(cases)}"
    )

    print(
        f"Training scans: {len(training)}"
    )

    print(
        f"Testing scans:  {len(testing)}"
    )

    # -------------------------------------------------------------
    # Per source
    # -------------------------------------------------------------

    for source in [
        "LumVBCanSeg",
        "Myeloma"
    ]:

        source_cases = [
            x
            for x in cases
            if x["source"] == source
        ]

        source_train = [
            x
            for x in source_cases
            if x["split"] == "train"
        ]

        source_test = [
            x
            for x in source_cases
            if x["split"] == "test"
        ]

        source_patients = set(
            x["patient_id"]
            for x in source_cases
        )

        train_patients = set(
            x["patient_id"]
            for x in source_train
        )

        test_patients = set(
            x["patient_id"]
            for x in source_test
        )

        print()
        print(source)

        print(
            f"  patients total: {len(source_patients)}"
        )

        print(
            f"  patients train: {len(train_patients)}"
        )

        print(
            f"  patients test:  {len(test_patients)}"
        )

        print(
            f"  scans train:    {len(source_train)}"
        )

        print(
            f"  scans test:     {len(source_test)}"
        )

    # -------------------------------------------------------------
    # Label occurrences
    # -------------------------------------------------------------

    print()
    print("GT label occurrence:")

    for label_id in range(
        1,
        7
    ):

        train_count = sum(
            label_id in x["labels_present"]
            for x in training
        )

        test_count = sum(
            label_id in x["labels_present"]
            for x in testing
        )

        print(
            f"  {LABEL_NAMES[label_id]}: "
            f"train = {train_count:3d}, "
            f"test = {test_count:3d}"
        )

    # -------------------------------------------------------------
    # Missing labels
    # -------------------------------------------------------------

    missing_cases = [
        x
        for x in cases
        if x["usual_labels_missing"]
    ]

    print()
    print(
        "Cases with at least one missing "
        f"L1-L5 label: {len(missing_cases)}"
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
    print("LumVBCanSeg:")
    print(LUM_DIR)

    print()
    print("Myeloma:")
    print(MYEL_DIR)

    print()
    print("Output:")
    print(OUTPUT_DIR)

    # =================================================================
    # Protect existing output
    # =================================================================

    if os.path.exists(
        OUTPUT_DIR
    ):

        if OVERWRITE_DATASET:

            print()
            print(
                "Removing existing dataset:"
            )

            print(
                OUTPUT_DIR
            )

            shutil.rmtree(
                OUTPUT_DIR
            )

        else:

            raise RuntimeError(
                "\n"
                "====================================================\n"
                "OUTPUT DATASET ALREADY EXISTS\n"
                "====================================================\n"
                f"{OUTPUT_DIR}\n\n"
                "Nothing was changed.\n\n"
                "If you really want to recreate the dataset, set:\n\n"
                "    OVERWRITE_DATASET = True\n"
            )

    # =================================================================
    # Find source data
    # =================================================================

    lum_cases = (
        find_lum_cases()
    )

    myel_cases = (
        find_myeloma_cases()
    )

    all_cases = (
        lum_cases +
        myel_cases
    )

    print()
    print("=" * 80)
    print("SOURCE DATA")
    print("=" * 80)

    print()
    print(
        f"LumVBCanSeg scans found: "
        f"{len(lum_cases)}"
    )

    print(
        f"Myeloma scans found:     "
        f"{len(myel_cases)}"
    )

    print(
        f"Total scans found:        "
        f"{len(all_cases)}"
    )

    # =================================================================
    # Check for duplicate nnU-Net case IDs
    # =================================================================

    case_ids = [
        case["case_id"]
        for case in all_cases
    ]

    if len(case_ids) != len(
        set(case_ids)
    ):

        duplicates = sorted(
            {
                x
                for x in case_ids
                if case_ids.count(x) > 1
            }
        )

        raise RuntimeError(
            "\nDuplicate nnU-Net case IDs detected:\n"
            f"{duplicates}"
        )

    # =================================================================
    # QC BEFORE COPY
    # =================================================================

    perform_qc(
        all_cases
    )

    # =================================================================
    # Patient IDs
    # =================================================================

    lum_patients = sorted(
        set(
            case["patient_id"]
            for case in lum_cases
        )
    )

    myel_patients = sorted(
        set(
            case["patient_id"]
            for case in myel_cases
        )
    )

    print()
    print("=" * 80)
    print("PATIENT COUNTS")
    print("=" * 80)

    print()
    print(
        f"LumVBCanSeg patients: "
        f"{len(lum_patients)}"
    )

    print(
        f"Myeloma patients:     "
        f"{len(myel_patients)}"
    )

    # =================================================================
    # Follow-up report
    # =================================================================

    myel_scans_per_patient = defaultdict(
        list
    )

    for case in myel_cases:

        myel_scans_per_patient[
            case["patient_id"]
        ].append(
            case["original_case_id"]
        )

    print()
    print("=" * 80)
    print("MYELOMA FOLLOW-UP PATIENTS")
    print("=" * 80)

    n_followup_patients = 0

    for patient_id in sorted(
        myel_scans_per_patient
    ):

        scans = (
            myel_scans_per_patient[
                patient_id
            ]
        )

        if len(scans) > 1:

            n_followup_patients += 1

            print(
                f"{patient_id}: "
                f"{', '.join(scans)}"
            )

    print()
    print(
        "Patients with more than one scan: "
        f"{n_followup_patients}"
    )

    # =================================================================
    # LumVBCanSeg patient-level split
    # =================================================================

    (
        lum_train_patients,
        lum_test_patients
    ) = split_patients(

        lum_patients,

        train_fraction=
            TRAIN_FRACTION,

        random_seed=
            RANDOM_SEED
    )

    # =================================================================
    # Myeloma patient-level split + L6 stratification
    # =================================================================

    (
        myel_train_patients,
        myel_test_patients,
        l6_train_patients,
        l6_test_patients
    ) = split_myeloma_patients(
        myel_patients
    )

    print()
    print("=" * 80)
    print("L6 SPLIT")
    print("=" * 80)

    print()
    print(
        "L6 TRAIN:"
    )

    for p in sorted(
        l6_train_patients
    ):
        print(
            f"  {p}"
        )

    print()
    print(
        "L6 TEST:"
    )

    for p in sorted(
        l6_test_patients
    ):
        print(
            f"  {p}"
        )

    # =================================================================
    # Assign splits
    # =================================================================

    for case in lum_cases:

        patient_id = (
            case["patient_id"]
        )

        if patient_id in lum_train_patients:

            case["split"] = (
                "train"
            )

        elif patient_id in lum_test_patients:

            case["split"] = (
                "test"
            )

        else:

            raise RuntimeError(
                "Lum patient was not assigned:\n"
                f"{patient_id}"
            )

    for case in myel_cases:

        patient_id = (
            case["patient_id"]
        )

        if patient_id in myel_train_patients:

            case["split"] = (
                "train"
            )

        elif patient_id in myel_test_patients:

            case["split"] = (
                "test"
            )

        else:

            raise RuntimeError(
                "Myeloma patient was not assigned:\n"
                f"{patient_id}"
            )

    # =================================================================
    # CRITICAL CHECK
    # =================================================================

    check_patient_leakage(
        all_cases
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
    # Copy data
    # =================================================================

    print()
    print("=" * 80)
    print("COPYING DATA")
    print("=" * 80)

    for index, case in enumerate(
        all_cases,
        start=1
    ):

        print(
            f"{index:4d}/{len(all_cases):4d}  "
            f"{case['case_id']:30s}  "
            f"{case['split']:5s}  "
            f"{case['source']}"
        )

        copy_case(
            case
        )

    # =================================================================
    # Number of training CASES, not patients
    # =================================================================

    training_cases = [
        case
        for case in all_cases
        if case["split"] == "train"
    ]

    # =================================================================
    # Create dataset.json
    # =================================================================

    create_dataset_json(
        num_training_cases=
            len(training_cases)
    )

    # =================================================================
    # Additional metadata
    # =================================================================

    save_split_json(
        all_cases
    )

    save_case_csv(
        all_cases
    )

    save_qc_csv(
        all_cases
    )

    save_missing_label_qc(
        all_cases
    )

    # =================================================================
    # Statistics
    # =================================================================

    print_statistics(
        all_cases
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
        "Dataset created at:"
    )

    print(
        OUTPUT_DIR
    )

    print()
    print(
        "Important QC file:"
    )

    print(
        os.path.join(
            OUTPUT_DIR,
            "qc_missing_labels.csv"
        )
    )

    print()
    print(
        "Review qc_missing_labels.csv before final training."
    )

    print()
    print(
        "After that you can run:"
    )

    print()
    print(
        "nnUNetv2_plan_and_preprocess "
        "-d 910 "
        "--verify_dataset_integrity"
    )