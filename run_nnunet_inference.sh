#!/usr/bin/env bash
# =============================================================================
# nnU-Net v2 local inference: Dataset910 (FullFOV) and Dataset911 (LumbarCrop)
#
# Experiment groups (each can be switched on/off below or via env variable):
#   MAIN        matched test predictions, ensemble folds 0-4
#   CROSS_FOV   2x2 FOV cross-experiment, ensemble folds 0-4
#   FOLD_ALL    matched test predictions with fold "all" (separate folders)
#   OOF         out-of-fold validation predictions (fold k model -> val cases of fold k)
#
# Usage:
#   ./run_nnunet_inference.sh              # run everything that is enabled
#   DRY_RUN=1 ./run_nnunet_inference.sh    # only checks + print commands
#   RUN_FOLD_ALL=1 RUN_CROSS_FOV=0 ./run_nnunet_inference.sh
# =============================================================================
set -uo pipefail

# ----------------------------------------------------------------------------
# PATHS  (edit these)
# ----------------------------------------------------------------------------
NNUNET_BASE="${NNUNET_BASE:-/home/nohel/DATA}"
export nnUNet_raw="${nnUNet_raw:-$NNUNET_BASE/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$NNUNET_BASE/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$NNUNET_BASE/nnUNet_results}"

# All predictions are written below this folder
OUTPUT_ROOT="${OUTPUT_ROOT:-$nnUNet_results/predikt_bvm}"

# ----------------------------------------------------------------------------
# DATASETS
# ----------------------------------------------------------------------------
ID_FULL=910
ID_CROP=911
NAME_FULL="Dataset${ID_FULL}_LumbarTrabecularSeg_FullFOV"
NAME_CROP="Dataset${ID_CROP}_LumbarTrabecularSeg_LumbarCrop"

# Input folders (test images; *_0000.nii.gz)
TEST_IMAGES_FULL="${TEST_IMAGES_FULL:-$nnUNet_raw/$NAME_FULL/imagesTs}"
TEST_IMAGES_CROP="${TEST_IMAGES_CROP:-$nnUNet_raw/$NAME_CROP/imagesTs}"

# Input folders for validation cases (all training images; subset selected by splits_final.json)
TRAIN_IMAGES_FULL="${TRAIN_IMAGES_FULL:-$nnUNet_raw/$NAME_FULL/imagesTr}"
TRAIN_IMAGES_CROP="${TRAIN_IMAGES_CROP:-$nnUNet_raw/$NAME_CROP/imagesTr}"

# ----------------------------------------------------------------------------
# MODELS
# ----------------------------------------------------------------------------
CONFIG="3d_fullres"
TRAINER="nnUNetTrainer"
PLANS_STANDARD="nnUNetPlans"
PLANS_RESENC="nnUNetResEncUNetLPlans"
CHECKPOINT="checkpoint_final.pth"
CV_FOLDS=(0 1 2 3 4)

# ----------------------------------------------------------------------------
# SWITCHES  (1 = run, 0 = skip)
# ----------------------------------------------------------------------------
RUN_MAIN="${RUN_MAIN:-1}"
RUN_CROSS_FOV="${RUN_CROSS_FOV:-1}"
RUN_FOLD_ALL="${RUN_FOLD_ALL:-0}"
RUN_OOF="${RUN_OOF:-1}"

# Model types
RUN_NNUNET="${RUN_NNUNET:-1}"
RUN_RESENC="${RUN_RESENC:-1}"

# Datasets used for out-of-fold validation predictions (911 primary, 910 optional)
OOF_ON_911="${OOF_ON_911:-1}"
OOF_ON_910="${OOF_ON_910:-0}"

# ----------------------------------------------------------------------------
# OTHER SETTINGS
# ----------------------------------------------------------------------------
DEVICE="${DEVICE:-cuda}"
EXTRA_PREDICT_ARGS="${EXTRA_PREDICT_ARGS:-}"   # e.g. "--disable_tta"
OVERWRITE="${OVERWRITE:-0}"                    # 1 = delete non-empty output folder first
DRY_RUN="${DRY_RUN:-0}"                        # 1 = do not run nnUNetv2_predict
PYTHON="${PYTHON:-python3}"

# =============================================================================
# Implementation
# =============================================================================
SUMMARY=()
FAILED=0

log()  { printf '%s\n' "$*"; }
err()  { printf 'ERROR: %s\n' "$*" >&2; }
rule() { printf '%0.s=' {1..78}; printf '\n'; }

dataset_name() { [[ "$1" == "$ID_FULL" ]] && echo "$NAME_FULL" || echo "$NAME_CROP"; }
plans_for()    { [[ "$1" == "resencL" ]] && echo "$PLANS_RESENC" || echo "$PLANS_STANDARD"; }
fov_label()    { [[ "$1" == "$ID_FULL" ]] && echo "full" || echo "crop"; }

count_inputs()  { find "$1" -maxdepth 1 \( -type f -o -type l \) -name '*_0000.nii.gz' 2>/dev/null | wc -l; }
count_outputs() { find "$1" -maxdepth 1 -type f -name '*.nii.gz' 2>/dev/null | wc -l; }

# model_exists <dataset_id> <model: nnunet|resencL> <fold>
model_exists() {
    local ds_name plans dir
    ds_name=$(dataset_name "$1"); plans=$(plans_for "$2")
    dir="$nnUNet_results/$ds_name/${TRAINER}__${plans}__${CONFIG}"
    if [[ ! -f "$dir/fold_$3/$CHECKPOINT" ]]; then
        err "missing model: $dir/fold_$3/$CHECKPOINT"
        return 1
    fi
    return 0
}

# check_input_dir <dir>
check_input_dir() {
    if [[ ! -d "$1" ]]; then err "input folder does not exist: $1"; return 1; fi
    if [[ $(count_inputs "$1") -eq 0 ]]; then err "no *_0000.nii.gz in: $1"; return 1; fi
    return 0
}

# prepare_output_dir <dir>
prepare_output_dir() {
    if [[ -d "$1" && -n "$(ls -A "$1" 2>/dev/null)" ]]; then
        if [[ "$OVERWRITE" == "1" ]]; then
            log "OVERWRITE=1 -> removing $1"
            [[ "$DRY_RUN" == "1" ]] || rm -rf "$1"
        else
            err "output folder is not empty (set OVERWRITE=1 to replace): $1"
            return 1
        fi
    fi
    [[ "$DRY_RUN" == "1" ]] || mkdir -p "$1"
    return 0
}

# run_predict <in_dir> <out_dir> <model_dataset_id> <model> <folds...>
run_predict() {
    local in_dir="$1" out_dir="$2" ds_id="$3" model="$4"; shift 4
    local cmd=(nnUNetv2_predict -i "$in_dir" -o "$out_dir" -d "$ds_id" -c "$CONFIG" -f "$@" -device "$DEVICE")
    [[ "$model" == "resencL" ]] && cmd+=(-p "$PLANS_RESENC")
    # shellcheck disable=SC2206
    [[ -n "$EXTRA_PREDICT_ARGS" ]] && cmd+=($EXTRA_PREDICT_ARGS)
    log "COMMAND: ${cmd[*]}"
    [[ "$DRY_RUN" == "1" ]] && return 0
    "${cmd[@]}"
}

record() { SUMMARY+=("$1|$2|$3|$4"); [[ "$2" == "OK" || "$2" == "DRY-RUN" ]] || FAILED=1; }

# run_experiment <name> <in_dir> <model_ds_id> <model> <folds...>
run_experiment() {
    local name="$1" in_dir="$2" ds_id="$3" model="$4"; shift 4
    local folds=("$@") out_dir="$OUTPUT_ROOT/$name" f n_in n_out
    rule; log "EXPERIMENT: $name"

    check_input_dir "$in_dir" || { record "$name" "FAIL(input)" "-" "-"; return; }
    for f in "${folds[@]}"; do
        model_exists "$ds_id" "$model" "$f" || { record "$name" "FAIL(model)" "-" "-"; return; }
    done
    n_in=$(count_inputs "$in_dir")
    prepare_output_dir "$out_dir" || { record "$name" "FAIL(output)" "$n_in" "-"; return; }

    if ! run_predict "$in_dir" "$out_dir" "$ds_id" "$model" "${folds[@]}"; then
        record "$name" "FAIL(predict)" "$n_in" "-"; return
    fi
    if [[ "$DRY_RUN" == "1" ]]; then record "$name" "DRY-RUN" "$n_in" "-"; return; fi

    n_out=$(count_outputs "$out_dir")
    log "inputs: $n_in   outputs: $n_out"
    if [[ "$n_in" -eq "$n_out" ]]; then
        log "DONE: $name"; record "$name" "OK" "$n_in" "$n_out"
    else
        err "count mismatch for $name"; record "$name" "FAIL(count)" "$n_in" "$n_out"
    fi
}

# model_types -> list of enabled model types
enabled_models() {
    [[ "$RUN_NNUNET" == "1" ]] && echo nnunet
    [[ "$RUN_RESENC" == "1" ]] && echo resencL
    return 0
}

# exp_name <train_id> <model> <cv5|foldall> <input_id>
exp_name() { echo "pred_$1_$2_$3_$(fov_label "$4")Input"; }

test_dir_for() { [[ "$1" == "$ID_FULL" ]] && echo "$TEST_IMAGES_FULL" || echo "$TEST_IMAGES_CROP"; }
train_dir_for() { [[ "$1" == "$ID_FULL" ]] && echo "$TRAIN_IMAGES_FULL" || echo "$TRAIN_IMAGES_CROP"; }

# ---- group: matched ensemble ------------------------------------------------
group_main() {
    local m ds
    for m in $(enabled_models); do
        for ds in "$ID_CROP" "$ID_FULL"; do
            run_experiment "$(exp_name "$ds" "$m" cv5 "$ds")" "$(test_dir_for "$ds")" "$ds" "$m" "${CV_FOLDS[@]}"
        done
    done
}

# ---- group: FOV cross experiment --------------------------------------------
group_cross_fov() {
    local m train inp
    for m in $(enabled_models); do
        for train in "$ID_CROP" "$ID_FULL"; do
            inp=$([[ "$train" == "$ID_CROP" ]] && echo "$ID_FULL" || echo "$ID_CROP")
            run_experiment "$(exp_name "$train" "$m" cv5 "$inp")" "$(test_dir_for "$inp")" "$train" "$m" "${CV_FOLDS[@]}"
        done
    done
}

# ---- group: fold all (separate folders) -------------------------------------
group_fold_all() {
    local m ds
    for m in $(enabled_models); do
        for ds in "$ID_CROP" "$ID_FULL"; do
            run_experiment "$(exp_name "$ds" "$m" foldall "$ds")" "$(test_dir_for "$ds")" "$ds" "$m" all
        done
    done
}

# ---- group: out-of-fold validation ------------------------------------------
# val_cases <splits_json> <fold>  -> prints validation case ids
val_cases() {
    "$PYTHON" - "$1" "$2" <<'EOF'
import json, sys
splits = json.load(open(sys.argv[1], encoding="utf-8"))
print("\n".join(splits[int(sys.argv[2])]["val"]))
EOF
}

oof_experiment() {
    local ds="$1" model="$2"
    local ds_name name out_dir splits train_dir f case_id tmp_in
    local n_expected=0 n_out missing=0 status="OK"
    ds_name=$(dataset_name "$ds")
    name="pred_${ds}_${model}_cv5_oofValidation"
    out_dir="$OUTPUT_ROOT/$name"
    splits="$nnUNet_preprocessed/$ds_name/splits_final.json"
    train_dir=$(train_dir_for "$ds")

    rule; log "EXPERIMENT: $name (fold k model -> validation cases of fold k only)"
    if [[ ! -f "$splits" ]]; then err "missing $splits"; record "$name" "FAIL(splits)" "-" "-"; return; fi
    if [[ ! -d "$train_dir" ]]; then err "input folder does not exist: $train_dir"; record "$name" "FAIL(input)" "-" "-"; return; fi
    for f in "${CV_FOLDS[@]}"; do
        model_exists "$ds" "$model" "$f" || { record "$name" "FAIL(model)" "-" "-"; return; }
    done
    prepare_output_dir "$out_dir" || { record "$name" "FAIL(output)" "-" "-"; return; }

    local -A seen=()
    for f in "${CV_FOLDS[@]}"; do
        tmp_in="$OUTPUT_ROOT/_tmp_oof_input_${ds}_${model}_fold${f}"
        [[ "$DRY_RUN" == "1" ]] || { rm -rf "$tmp_in"; mkdir -p "$tmp_in"; }

        local ids=() n_fold=0
        mapfile -t ids < <(val_cases "$splits" "$f")
        for case_id in "${ids[@]}"; do
            [[ -z "$case_id" ]] && continue
            if [[ -n "${seen[$case_id]:-}" ]]; then err "case $case_id is in validation of multiple folds"; status="FAIL(splits)"; fi
            seen[$case_id]=$f
            if [[ ! -f "$train_dir/${case_id}_0000.nii.gz" ]]; then
                err "missing input image: $train_dir/${case_id}_0000.nii.gz"; status="FAIL(input)"; continue
            fi
            [[ "$DRY_RUN" == "1" ]] || ln -s "$train_dir/${case_id}_0000.nii.gz" "$tmp_in/${case_id}_0000.nii.gz"
            n_fold=$((n_fold + 1))
        done
        n_expected=$((n_expected + n_fold))
        log "fold $f: $n_fold validation cases"
        [[ "$status" == "OK" ]] || break

        if ! run_predict "$tmp_in" "$out_dir" "$ds" "$model" "$f"; then status="FAIL(predict)"; break; fi
        if [[ "$DRY_RUN" != "1" ]]; then
            for case_id in "${ids[@]}"; do
                [[ -z "$case_id" ]] && continue
                [[ -f "$out_dir/$case_id.nii.gz" ]] || { err "missing prediction: $case_id (fold $f)"; missing=$((missing + 1)); }
            done
            rm -rf "$tmp_in"
        fi
    done

    if [[ "$DRY_RUN" == "1" ]]; then
        [[ "$status" == "OK" ]] && status="DRY-RUN"
        record "$name" "$status" "$n_expected" "-"; return
    fi
    n_out=$(count_outputs "$out_dir")
    log "validation cases: $n_expected   merged OOF outputs: $n_out"
    [[ "$status" == "OK" && ( "$missing" -ne 0 || "$n_expected" -ne "$n_out" ) ]] && status="FAIL(count)"
    [[ "$status" == "OK" ]] && log "DONE: $name"
    record "$name" "$status" "$n_expected" "$n_out"
}

group_oof() {
    local m
    for m in $(enabled_models); do
        [[ "$OOF_ON_911" == "1" ]] && oof_experiment "$ID_CROP" "$m"
        [[ "$OOF_ON_910" == "1" ]] && oof_experiment "$ID_FULL" "$m"
    done
    return 0
}

# ---- main -------------------------------------------------------------------
rule
log "nnUNet_raw          = $nnUNet_raw"
log "nnUNet_preprocessed = $nnUNet_preprocessed"
log "nnUNet_results      = $nnUNet_results"
log "OUTPUT_ROOT         = $OUTPUT_ROOT"
log "DRY_RUN=$DRY_RUN OVERWRITE=$OVERWRITE DEVICE=$DEVICE"
log "groups: MAIN=$RUN_MAIN CROSS_FOV=$RUN_CROSS_FOV FOLD_ALL=$RUN_FOLD_ALL OOF=$RUN_OOF"

if [[ "$DRY_RUN" != "1" ]] && ! command -v nnUNetv2_predict >/dev/null 2>&1; then
    err "nnUNetv2_predict not found - activate the nnU-Net environment first"
    exit 1
fi

[[ "$RUN_MAIN" == "1" ]]      && group_main
[[ "$RUN_CROSS_FOV" == "1" ]] && group_cross_fov
[[ "$RUN_FOLD_ALL" == "1" ]]  && group_fold_all
[[ "$RUN_OOF" == "1" ]]       && group_oof

rule; log "SUMMARY"; rule
printf '%-48s %-14s %8s %8s\n' EXPERIMENT STATUS INPUTS OUTPUTS
for row in "${SUMMARY[@]}"; do
    IFS='|' read -r a b c d <<<"$row"
    printf '%-48s %-14s %8s %8s\n' "$a" "$b" "$c" "$d"
done
exit "$FAILED"
