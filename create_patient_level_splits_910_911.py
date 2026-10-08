# -*- coding: utf-8 -*-

"""
Create patient-level 5-fold splits_final.json
for Dataset910 and Dataset911.

Important:
    - Uses ONLY training cases from Dataset910 train_test_split.json
    - Follow-up scans from the same Myeloma patient stay in the same fold
    - Dataset910 and Dataset911 receive EXACTLY the same folds
    - L6 patients in the training set are distributed across folds if possible
"""

import os
import json
import random
from collections import defaultdict


# ============================================================
# SETTINGS
# ============================================================

NNUNET_RAW = r"E:\nnUNet_v2_MAIN_FILE\nnUNet_raw"

NNUNET_PREPROCESSED = (
    r"E:\nnUNet_v2_MAIN_FILE\nnUNet_preprocessed"
)


DATASET910_NAME = (
    "Dataset910_LumbarTrabecularSeg_FullFOV"
)

DATASET911_NAME = (
    "Dataset911_LumbarTrabecularSeg_LumbarCrop"
)


DATASET910_RAW = os.path.join(
    NNUNET_RAW,
    DATASET910_NAME
)


DATASET910_PREPROCESSED = os.path.join(
    NNUNET_PREPROCESSED,
    DATASET910_NAME
)

DATASET911_PREPROCESSED = os.path.join(
    NNUNET_PREPROCESSED,
    DATASET911_NAME
)


N_FOLDS = 5

RANDOM_SEED = 42


# Patients known to contain L6
L6_PATIENTS = {
    "Myel_021",
    "Myel_055",
    "Myel_060",
}


# ============================================================
# LOAD DATASET910 SPLIT METADATA
# ============================================================

def load_cases():

    filename = os.path.join(
        DATASET910_RAW,
        "train_test_split.json"
    )

    if not os.path.isfile(filename):

        raise FileNotFoundError(
            f"Cannot find:\n{filename}"
        )

    with open(
        filename,
        "r",
        encoding="utf-8"
    ) as f:

        records = json.load(f)

    # We ONLY use cases belonging to imagesTr
    training_records = [
        record
        for record in records
        if record["split"] == "train"
    ]

    if not training_records:

        raise RuntimeError(
            "No training cases found."
        )

    return training_records


# ============================================================
# GROUP CASES BY PATIENT
# ============================================================

def group_cases_by_patient(records):

    patient_cases = defaultdict(list)

    patient_source = {}

    for record in records:

        patient_id = record["patient_id"]

        case_id = record["case_id"]

        patient_cases[
            patient_id
        ].append(
            case_id
        )

        patient_source[
            patient_id
        ] = record["source"]

    # Sort scans within each patient
    for patient_id in patient_cases:

        patient_cases[
            patient_id
        ].sort()

    return (
        patient_cases,
        patient_source
    )


# ============================================================
# ASSIGN PATIENTS TO FOLDS
# ============================================================

def create_patient_folds(
        patient_cases,
        patient_source
):
    """
    Balanced assignment of PATIENTS to validation folds.

    Strategy:
        1) separate LumVBCanSeg and Myeloma patients
        2) shuffle reproducibly
        3) distribute round-robin across 5 folds
        4) distribute L6 training patients explicitly first
    """

    rng = random.Random(
        RANDOM_SEED
    )

    folds = [
        []
        for _ in range(N_FOLDS)
    ]

    # --------------------------------------------------------
    # L6 patients that are actually in training
    # --------------------------------------------------------

    training_l6_patients = sorted([
        patient_id
        for patient_id in patient_cases
        if patient_id in L6_PATIENTS
    ])

    rng.shuffle(
        training_l6_patients
    )

    print()
    print(
        "L6 patients present in training:"
    )

    print(
        training_l6_patients
    )

    # Put L6 patients in different folds
    for fold_index, patient_id in enumerate(
        training_l6_patients
    ):

        folds[
            fold_index % N_FOLDS
        ].append(
            patient_id
        )

    # --------------------------------------------------------
    # Remaining patients by source
    # --------------------------------------------------------

    lum_patients = []

    myel_patients = []

    for patient_id in patient_cases:

        if patient_id in training_l6_patients:
            continue

        source = patient_source[
            patient_id
        ]

        if source == "LumVBCanSeg":

            lum_patients.append(
                patient_id
            )

        elif source == "Myeloma":

            myel_patients.append(
                patient_id
            )

        else:

            raise RuntimeError(
                f"Unknown source: {source}"
            )

    lum_patients.sort()

    myel_patients.sort()

    rng.shuffle(
        lum_patients
    )

    rng.shuffle(
        myel_patients
    )

    # --------------------------------------------------------
    # Balanced round-robin distribution
    # --------------------------------------------------------

    def distribute(
            patients,
            start_offset=0
    ):

        for i, patient_id in enumerate(
            patients
        ):

            fold_index = (
                i + start_offset
            ) % N_FOLDS

            folds[
                fold_index
            ].append(
                patient_id
            )

    distribute(
        lum_patients,
        start_offset=0
    )

    distribute(
        myel_patients,
        start_offset=2
    )

    return folds


# ============================================================
# CONVERT PATIENT FOLDS TO nnUNet CASE SPLITS
# ============================================================

def build_nnunet_splits(
        patient_folds,
        patient_cases
):

    all_patients = set(
        patient_cases.keys()
    )

    splits = []

    for fold_index in range(
        N_FOLDS
    ):

        val_patients = set(
            patient_folds[
                fold_index
            ]
        )

        train_patients = (
            all_patients -
            val_patients
        )

        val_cases = []

        train_cases = []

        # Patient -> all scans
        for patient_id in sorted(
            val_patients
        ):

            val_cases.extend(
                patient_cases[
                    patient_id
                ]
            )

        for patient_id in sorted(
            train_patients
        ):

            train_cases.extend(
                patient_cases[
                    patient_id
                ]
            )

        val_cases.sort()
        train_cases.sort()

        splits.append({

            "train":
                train_cases,

            "val":
                val_cases
        })

    return splits


# ============================================================
# SAFETY CHECKS
# ============================================================

def verify_splits(
        splits,
        records
):

    expected_cases = set(
        record["case_id"]
        for record in records
    )

    print()
    print("=" * 80)
    print("VERIFYING SPLITS")
    print("=" * 80)

    validation_counts = defaultdict(
        int
    )

    for fold_index, split in enumerate(
        splits
    ):

        train_cases = set(
            split["train"]
        )

        val_cases = set(
            split["val"]
        )

        # No overlap
        overlap = (
            train_cases &
            val_cases
        )

        if overlap:

            raise RuntimeError(
                f"Fold {fold_index}: "
                f"train/val overlap:\n"
                f"{sorted(overlap)}"
            )

        # Together they must equal all training cases
        combined = (
            train_cases |
            val_cases
        )

        if combined != expected_cases:

            missing = (
                expected_cases -
                combined
            )

            extra = (
                combined -
                expected_cases
            )

            raise RuntimeError(
                f"Fold {fold_index} mismatch.\n"
                f"Missing: {sorted(missing)}\n"
                f"Extra: {sorted(extra)}"
            )

        # Count validation occurrence
        for case_id in val_cases:

            validation_counts[
                case_id
            ] += 1

        print()
        print(
            f"Fold {fold_index}:"
        )

        print(
            f"  train cases: {len(train_cases)}"
        )

        print(
            f"  val cases:   {len(val_cases)}"
        )

    # Every case must appear exactly once as validation
    incorrect = {
        case_id: count
        for case_id, count
        in validation_counts.items()
        if count != 1
    }

    if incorrect:

        raise RuntimeError(
            "Some cases do not occur exactly once "
            "in validation:\n"
            f"{incorrect}"
        )

    print()
    print(
        "Every training case appears "
        "exactly once in validation: OK"
    )


# ============================================================
# CHECK FOLLOW-UP LEAKAGE
# ============================================================

def verify_patient_grouping(
        splits,
        records
):

    case_to_patient = {
        record["case_id"]:
            record["patient_id"]
        for record in records
    }

    for fold_index, split in enumerate(
        splits
    ):

        train_patients = {
            case_to_patient[case_id]
            for case_id in split["train"]
        }

        val_patients = {
            case_to_patient[case_id]
            for case_id in split["val"]
        }

        overlap = (
            train_patients &
            val_patients
        )

        if overlap:

            raise RuntimeError(
                f"\nPATIENT LEAKAGE IN FOLD "
                f"{fold_index}:\n"
                f"{sorted(overlap)}"
            )

    print()
    print(
        "Patient-level leakage check: OK"
    )


# ============================================================
# PRINT FOLD CONTENTS
# ============================================================

def print_fold_summary(
        patient_folds,
        patient_cases,
        patient_source
):

    print()
    print("=" * 80)
    print("PATIENT-LEVEL FOLD SUMMARY")
    print("=" * 80)

    for fold_index, patients in enumerate(
        patient_folds
    ):

        print()
        print(
            f"FOLD {fold_index}"
        )

        print(
            "-" * 50
        )

        lum = [
            p
            for p in patients
            if patient_source[p]
            == "LumVBCanSeg"
        ]

        myel = [
            p
            for p in patients
            if patient_source[p]
            == "Myeloma"
        ]

        l6 = [
            p
            for p in patients
            if p in L6_PATIENTS
        ]

        number_of_cases = sum(
            len(
                patient_cases[p]
            )
            for p in patients
        )

        print(
            f"Validation patients: "
            f"{len(patients)}"
        )

        print(
            f"Validation cases:    "
            f"{number_of_cases}"
        )

        print(
            f"LumVBCanSeg patients: "
            f"{len(lum)}"
        )

        print(
            f"Myeloma patients:     "
            f"{len(myel)}"
        )

        print(
            f"L6 patients:          "
            f"{l6}"
        )

        # Print follow-up patients
        followups = [
            p
            for p in patients
            if len(
                patient_cases[p]
            ) > 1
        ]

        if followups:

            print(
                "Follow-up patients:"
            )

            for p in followups:

                print(
                    f"  {p}: "
                    f"{patient_cases[p]}"
                )


# ============================================================
# SAVE
# ============================================================

def save_splits(
        splits,
        output_folder
):

    if not os.path.isdir(
        output_folder
    ):

        raise FileNotFoundError(
            f"Preprocessed dataset folder "
            f"does not exist:\n"
            f"{output_folder}"
        )

    filename = os.path.join(
        output_folder,
        "splits_final.json"
    )

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            splits,
            f,
            indent=4
        )

    print()
    print(
        f"Saved:\n{filename}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 80)
    print(
        "Creating patient-level 5-fold "
        "nnU-Net splits"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # Load training cases
    # --------------------------------------------------------

    records = load_cases()

    print()
    print(
        f"Training scans: "
        f"{len(records)}"
    )

    # --------------------------------------------------------
    # Group by patient
    # --------------------------------------------------------

    (
        patient_cases,
        patient_source
    ) = group_cases_by_patient(
        records
    )

    print(
        f"Training patients: "
        f"{len(patient_cases)}"
    )

    # --------------------------------------------------------
    # Create patient folds
    # --------------------------------------------------------

    patient_folds = (
        create_patient_folds(
            patient_cases,
            patient_source
        )
    )

    # --------------------------------------------------------
    # Convert to nnU-Net splits
    # --------------------------------------------------------

    splits = build_nnunet_splits(
        patient_folds,
        patient_cases
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    verify_splits(
        splits,
        records
    )

    verify_patient_grouping(
        splits,
        records
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print_fold_summary(
        patient_folds,
        patient_cases,
        patient_source
    )

    # --------------------------------------------------------
    # Save SAME splits to 910 and 911
    # --------------------------------------------------------

    save_splits(
        splits,
        DATASET910_PREPROCESSED
    )

    save_splits(
        splits,
        DATASET911_PREPROCESSED
    )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)

    print()
    print(
        "Dataset910 and Dataset911 now use "
        "identical patient-level folds."
    )