#!/usr/bin/env python3
"""
Create paper-ready Results outputs for the BVM lumbar trabecular segmentation study.

Inputs
------
1) metrics_long.csv
   New nnU-Net / ResEncL results from evaluate_bvm_predictions_v2.py

2) baseline_metrics_long.csv
   Historical LumVBCanSeg-only nnU-Net results

Outputs
-------
results_outputs/
  table_main_performance.csv
  table_main_performance.tex
  table_failure_analysis.csv
  table_failure_analysis.tex
  table_fov.csv
  table_fov.tex
  boxplots_main.png
  boxplots_main.pdf
  results_numbers.txt

Recommended paper usage
-----------------------
Use:
  - boxplots_main.pdf
  - table_main_performance.tex
  - table_failure_analysis.tex

Keep table_fov.tex as optional/secondary if space allows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def fmt_mean_sd(mean, sd, decimals=3):
    return f"{mean:.{decimals}f} $\\pm$ {sd:.{decimals}f}"


def latex_escape(s: str) -> str:
    return str(s).replace("_", r"\_")


def select_primary(df: pd.DataFrame) -> pd.DataFrame:
    """Primary cropped experiment: 911 crop-trained models on crop input."""
    is_test = (
        (df["split"] == "test")
        & (df["training_fov"] == "crop")
        & (df["inference_fov"] == "crop")
        & (df["model_mode"] == "cv5")
    )
    is_val = (
        (df["split"] == "validation_oof")
        & (df["training_fov"] == "crop")
        & (df["inference_fov"] == "crop")
        & (df["model_mode"] == "oof_cv5")
    )
    return df[is_test | is_val].copy()


def main_performance_table(primary: pd.DataFrame) -> pd.DataFrame:
    x = primary[primary["gt_present"]].copy()

    rows = []
    split_order = [("validation_oof", "Validation"), ("test", "Test")]
    source_order = ["LumVBCanSeg", "MM"]
    arch_order = [("nnunet", "nnU-Net"), ("resencL", "ResEncL")]

    for split_key, split_label in split_order:
        for source in source_order:
            for arch_key, arch_label in arch_order:
                g = x[
                    (x["split"] == split_key)
                    & (x["source"] == source)
                    & (x["architecture"] == arch_key)
                ]
                if g.empty:
                    continue

                rows.append(
                    {
                        "Split": split_label,
                        "Cohort": source,
                        "Model": arch_label,
                        "N vertebrae": len(g),
                        "Dice mean": g["dice"].mean(),
                        "Dice SD": g["dice"].std(),
                        "Dice median": g["dice"].median(),
                        "HD95 mean [mm]": g["hd95_mm"].mean(),
                        "HD95 SD [mm]": g["hd95_mm"].std(),
                        "HD95 median [mm]": g["hd95_mm"].median(),
                    }
                )

    return pd.DataFrame(rows)


def combined_mm_new(primary: pd.DataFrame, arch: str) -> pd.DataFrame:
    """All 71 MM cases: OOF for train cases + CV5 ensemble for independent test."""
    x = primary[
        (primary["source"] == "MM")
        & (primary["architecture"] == arch)
        & (primary["gt_present"])
    ].copy()

    # Each case should occur in exactly one of validation_oof/test for this architecture.
    return x


def failure_analysis(primary: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    rows = []

    datasets = [
        (
            "Historical nnU-Net",
            baseline[(baseline["source"] == "MM") & baseline["gt_present"]].copy(),
        ),
        ("nnU-Net", combined_mm_new(primary, "nnunet")),
        ("ResEncL", combined_mm_new(primary, "resencL")),
    ]

    for model, g in datasets:
        n = len(g)
        n_cases = g["case_id"].nunique()

        lt08 = g["dice"] < 0.8
        lt05 = g["dice"] < 0.5
        fn = ~g["pred_present"]

        rows.append(
            {
                "Model": model,
                "N vertebrae": n,
                "N cases": n_cases,
                "Dice < 0.8": int(lt08.sum()),
                "Dice < 0.8 [%]": 100 * lt08.mean(),
                "Dice < 0.5": int(lt05.sum()),
                "Dice < 0.5 [%]": 100 * lt05.mean(),
                "Cases with Dice < 0.8": int(g.loc[lt08, "case_id"].nunique()),
                "False negatives": int(fn.sum()),
            }
        )

    return pd.DataFrame(rows)


def fov_table(df: pd.DataFrame) -> pd.DataFrame:
    """Secondary FOV analysis on independent MM test set only."""
    x = df[
        (df["split"] == "test")
        & (df["model_mode"] == "cv5")
        & (df["source"] == "MM")
        & (df["gt_present"])
    ].copy()

    rows = []
    for arch_key, arch_label in [("nnunet", "nnU-Net"), ("resencL", "ResEncL")]:
        for train_fov in ["crop", "full"]:
            for input_fov in ["crop", "full"]:
                g = x[
                    (x["architecture"] == arch_key)
                    & (x["training_fov"] == train_fov)
                    & (x["inference_fov"] == input_fov)
                ]
                if g.empty:
                    continue
                rows.append(
                    {
                        "Model": arch_label,
                        "Training FOV": train_fov,
                        "Inference FOV": input_fov,
                        "Dice mean": g["dice"].mean(),
                        "Dice SD": g["dice"].std(),
                        "HD95 mean [mm]": g["hd95_mm"].mean(),
                        "HD95 SD [mm]": g["hd95_mm"].std(),
                    }
                )
    return pd.DataFrame(rows)


def write_main_latex(df: pd.DataFrame, path: Path):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Segmentation performance for the primary lumbar-cropped experiment. Values are mean $\pm$ standard deviation. Validation results are based on out-of-fold predictions.}",
        r"\label{0000-tab-main-results}",
        r"\begin{tabular}{lllll}",
        r"\hline",
        r"Split & Cohort & Model & Dice & HD95 [mm] \\",
        r"\hline",
    ]
    for _, r in df.iterrows():
        dice = fmt_mean_sd(r["Dice mean"], r["Dice SD"], 3)
        hd = fmt_mean_sd(r["HD95 mean [mm]"], r["HD95 SD [mm]"], 2)
        lines.append(
            f"{r['Split']} & {r['Cohort']} & {r['Model']} & ${dice}$ & ${hd}$ \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_failure_latex(df: pd.DataFrame, path: Path):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Descriptive failure analysis across the 71 multiple-myeloma examinations. The historical model is reported as an annotation-provenance baseline rather than an independent benchmark.}",
        r"\label{0000-tab-failures}",
        r"\begin{tabular}{lrrr}",
        r"\hline",
        r"Model & Dice $<0.8$ & Dice $<0.5$ & Cases $<0.8$ \\",
        r"\hline",
    ]
    for _, r in df.iterrows():
        c08 = f"{int(r['Dice < 0.8'])} ({r['Dice < 0.8 [%]']:.1f}\\%)"
        c05 = f"{int(r['Dice < 0.5'])} ({r['Dice < 0.5 [%]']:.1f}\\%)"
        cases = f"{int(r['Cases with Dice < 0.8'])}/71"
        lines.append(f"{r['Model']} & {c08} & {c05} & {cases} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_fov_latex(df: pd.DataFrame, path: Path):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Secondary field-of-view analysis on the independent multiple-myeloma test set.}",
        r"\label{0000-tab-fov}",
        r"\begin{tabular}{lllll}",
        r"\hline",
        r"Model & Train FOV & Input FOV & Dice & HD95 [mm] \\",
        r"\hline",
    ]
    for _, r in df.iterrows():
        dice = fmt_mean_sd(r["Dice mean"], r["Dice SD"], 3)
        hd = fmt_mean_sd(r["HD95 mean [mm]"], r["HD95 SD [mm]"], 2)
        lines.append(
            f"{r['Model']} & {r['Training FOV']} & {r['Inference FOV']} & "
            f"${dice}$ & ${hd}$ \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines), encoding="utf-8")


def _patient_level_new(primary: pd.DataFrame) -> pd.DataFrame:
    """Patient-level means for the new MM models."""
    x = primary[(primary["source"] == "MM") & (primary["gt_present"])].copy()
    return (
        x.groupby(["split", "architecture", "patient_id"], as_index=False)
        .agg(
            dice_mean=("dice", "mean"),
            hd95_mean=("hd95_mm", "mean"),
        )
    )


def _patient_level_historical(primary: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Historical baseline split according to the same MM validation/test assignment."""
    new_mm = primary[(primary["source"] == "MM") & (primary["gt_present"])].copy()

    # Map each historical case to the validation/test split used by the new models.
    case_split = new_mm[["case_id", "split"]].drop_duplicates()
    if case_split["case_id"].duplicated().any():
        dup = sorted(case_split.loc[case_split["case_id"].duplicated(), "case_id"].unique())
        raise ValueError(f"Duplicate split assignment for MM case(s): {dup}")

    hist = baseline[(baseline["source"] == "MM") & (baseline["gt_present"])].copy()
    hist = hist.merge(case_split, on="case_id", how="left", suffixes=("", "_mapped"))

    if hist["split_mapped"].isna().any():
        missing = sorted(hist.loc[hist["split_mapped"].isna(), "case_id"].unique())
        raise ValueError(
            "Historical baseline contains cases that cannot be mapped to the new split: "
            f"{missing}"
        )

    hist["split"] = hist["split_mapped"]

    # Historical CSV has case_id but no reliable patient_id. For the figure we use
    # examination-level means. New data use patient_id, so repeated follow-ups remain
    # grouped at patient level there.
    return (
        hist.groupby(["split", "case_id"], as_index=False)
        .agg(
            dice_mean=("dice", "mean"),
            hd95_mean=("hd95_mm", "mean"),
        )
        .rename(columns={"case_id": "patient_id"})
        .assign(architecture="historical")
    )


def _draw_mm_boxplot(patient_level: pd.DataFrame, groups, colors, png_path: Path, pdf_path: Path, dice_ylim=(0.85, 1.00)):
    """Draw one two-panel MM patient-level Dice/HD95 figure."""
    dice_data = []
    hd_data = []
    labels = []

    for split, arch, label in groups:
        g = patient_level[
            (patient_level["split"] == split)
            & (patient_level["architecture"] == arch)
        ]
        dice_data.append(g["dice_mean"].dropna().to_numpy())
        hd_data.append(g["hd95_mean"].dropna().to_numpy())
        labels.append(label)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))

    # Dice
    bp1 = axes[0].boxplot(
        dice_data,
        patch_artist=True,
        showfliers=False,
        widths=0.60,
    )
    for patch, color in zip(bp1["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    axes[0].set_title("(a) Dice")
    axes[0].set_ylabel("Patient-level mean Dice")
    axes[0].set_xticks(range(1, len(labels) + 1))
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(*dice_ylim)
    axes[0].grid(axis="y", alpha=0.25)

    # HD95
    bp2 = axes[1].boxplot(
        hd_data,
        patch_artist=True,
        showfliers=False,
        widths=0.60,
    )
    for patch, color in zip(bp2["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    axes[1].set_title("(b) HD95")
    axes[1].set_ylabel("Patient-level mean HD95 [mm]")
    axes[1].set_xticks(range(1, len(labels) + 1))
    axes[1].set_xticklabels(labels)
    axes[1].grid(axis="y", alpha=0.25)

    # Show the individual patient/examination means as small jittered points.
    rng = np.random.default_rng(0)
    for ax, data in ((axes[0], dice_data), (axes[1], hd_data)):
        for i, (vals, color) in enumerate(zip(data, colors), start=1):
            jitter = rng.uniform(-0.14, 0.14, size=len(vals))
            ax.scatter(
                i + jitter,
                vals,
                s=10,
                color=color,
                edgecolors="black",
                linewidths=0.25,
                alpha=0.75,
                zorder=3,
            )

    fig.suptitle("Multiple myeloma cohort", fontsize=13)
    fig.tight_layout()
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def make_boxplots(primary: pd.DataFrame, baseline: pd.DataFrame, out_dir: Path):
    """Create both figure variants: without and with historical baseline."""
    new_patient = _patient_level_new(primary)

    # Variant 1: new models only
    groups_no_hist = [
        ("validation_oof", "nnunet", "Validation\nnnU-Net"),
        ("validation_oof", "resencL", "Validation\nResEncL"),
        ("test", "nnunet", "Test\nnnU-Net"),
        ("test", "resencL", "Test\nResEncL"),
    ]
    colors_no_hist = ["tab:blue", "tab:green", "tab:blue", "tab:green"]
    _draw_mm_boxplot(
        new_patient,
        groups_no_hist,
        colors_no_hist,
        out_dir / "boxplots_main_no_historical.png",
        out_dir / "boxplots_main_no_historical.pdf",
        dice_ylim=(0.85, 1.00),
    )

    # Variant 2: historical + new models
    hist_patient = _patient_level_historical(primary, baseline)
    all_patient = pd.concat(
        [
            hist_patient[["split", "architecture", "patient_id", "dice_mean", "hd95_mean"]],
            new_patient[["split", "architecture", "patient_id", "dice_mean", "hd95_mean"]],
        ],
        ignore_index=True,
    )

    groups_hist = [
        ("validation_oof", "historical", "Validation\nHistorical"),
        ("validation_oof", "nnunet", "Validation\nnnU-Net"),
        ("validation_oof", "resencL", "Validation\nResEncL"),
        ("test", "historical", "Test\nHistorical"),
        ("test", "nnunet", "Test\nnnU-Net"),
        ("test", "resencL", "Test\nResEncL"),
    ]
    colors_hist = [
        "lightgray", "tab:blue", "tab:green",
        "lightgray", "tab:blue", "tab:green",
    ]
    _draw_mm_boxplot(
        all_patient,
        groups_hist,
        colors_hist,
        out_dir / "boxplots_main_with_historical.png",
        out_dir / "boxplots_main_with_historical.pdf",
        dice_ylim=(0.74, 1.00),
    )

def write_numbers(
    main_df: pd.DataFrame,
    failure_df: pd.DataFrame,
    fov_df: pd.DataFrame,
    path: Path,
):
    lines = []
    lines.append("MAIN PERFORMANCE")
    lines.append(main_df.to_string(index=False))
    lines.append("\nFAILURE ANALYSIS")
    lines.append(failure_df.to_string(index=False))
    lines.append("\nFOV ANALYSIS")
    lines.append(fov_df.to_string(index=False))
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path(r"E:\nnUNet_v2_MAIN_FILE\nnUNet_results\predikt_bvm\evaluation\metrics_long.csv"),
        help="metrics_long.csv from evaluate_bvm_predictions_v2.py",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path(r"E:\nnUNet_v2_MAIN_FILE\nnUNet_results\predikt_bvm\baseline_evaluation\baseline_metrics_long.csv"),
        help="baseline_metrics_long.csv from historical baseline evaluation",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(r"E:\nnUNet_v2_MAIN_FILE\nnUNet_results\predikt_bvm\results_outputs"),
    )
    args = parser.parse_args()

    print("Using:")
    print(f"  metrics:  {args.metrics}")
    print(f"  baseline: {args.baseline}")
    print(f"  output:   {args.out_dir}")
    print()

    if not args.metrics.is_file():
        raise FileNotFoundError(f"Metrics file not found: {args.metrics}")
    if not args.baseline.is_file():
        raise FileNotFoundError(f"Baseline file not found: {args.baseline}")

    new = pd.read_csv(args.metrics)
    baseline = pd.read_csv(args.baseline)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    primary = select_primary(new)
    main_df = main_performance_table(primary)
    failures = failure_analysis(primary, baseline)
    fov = fov_table(new)

    main_df.to_csv(out / "table_main_performance.csv", index=False)
    failures.to_csv(out / "table_failure_analysis.csv", index=False)
    fov.to_csv(out / "table_fov.csv", index=False)

    write_main_latex(main_df, out / "table_main_performance.tex")
    write_failure_latex(failures, out / "table_failure_analysis.tex")
    write_fov_latex(fov, out / "table_fov.tex")

    make_boxplots(
        primary,
        baseline,
        out,
    )

    write_numbers(
        main_df,
        failures,
        fov,
        out / "results_numbers.txt",
    )

    print("Created:")
    for p in sorted(out.iterdir()):
        print(" ", p)


if __name__ == "__main__":
    main()
