# DSConv fair re-evaluation

Reproducible code, numeric results and audit references for **Small Gains, Large Noise: An Implementation Defect, Evaluation Choices, and a Fair Re-evaluation of Plug-in Modules for Retinal Vessel Segmentation** (Heng Zhang; manuscript under submission).

The study characterizes a cross-sample indexing defect, measures evaluation-choice effects and compares plug-in modules under a fixed image-paired protocol. The original DSCNet batch1 paper configuration is unaffected by this specific indexing error. Repairing the operator did **not** improve the published shared-LR batch8 comparison; negative results and exploratory controls are retained.

## 1. Environment

Training used Linux, Python **3.12**, **torch 2.5.1+cu124**, torchvision 0.20.1+cu124, **MONAI 1.4.0**, **NumPy 1.26.4**, FP32 and disabled TF32. Use an NVIDIA GPU with 24 GB memory for unchanged batch8 training. Smaller devices must not silently change batch size or precision when claiming the same experiment.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation -r requirements-mamba.txt
```

The final command is needed only for `+mamba`. It installs the official independent forward/backward Mamba2 implementation (mamba-ssm 2.2.4, causal-conv1d 1.5.0.post8), not a substitute scan. Use release wheels matching Python 3.12, torch 2.5 and CUDA 12, or a Linux CUDA 12 toolchain if compilation is required. CPU-only readers can install the CPU torch 2.5.1 wheel from `https://download.pytorch.org/whl/cpu`, then `requirements.txt`; omit Mamba. Synthetic E1 was additionally measured with CPU torch 2.8.0, and the release's data-free checks are verified on that local environment. The exact publication check environment is recorded in `PUBLIC_RELEASE_CHECKS.json`.

## 2. Obtain the baseline and data separately

No weights, dataset images, NIfTI volumes or complete third-party source snapshots are distributed here. Run these commands **from the repository root**:

```bash
git clone https://github.com/YaoleiQi/DSCNet.git DSCNet
git -C DSCNet checkout 0e35b2b6ea748a5e8ce7557ea3e580b9951471d6
git clone https://github.com/SHI-Labs/SGL-Retinal-Vessel-Segmentation.git SGL
python prepare_chase.py
```

DRIVE uses the pinned baseline's `DSCNet_2D_opensource/Data/DRIVE` NIfTI files. DRIVE has 20 supplied training and20 test images. Keep16 for training and hold out training IDs 21,26,31,36 for validation (equivalently sorted positions 1,6,11,16). The script records the actual filenames in the generated split file; it never moves test images into training.

CHASE_DB1 uses SGL's public redistribution: `dataset/CHASE/HR` has 20 training images; `dataset/CHASET/HR` has 8 test images. Labels are in corresponding `VE` directories and use the first observer (`*_1stHO.png`). Hold out sorted training positions 1,6,11,16 for validation, yielding16/4/8. The immutable SGL commit used by the study is `e74eaeac27ce2a483da7717645eb0f6e8e708acd`; for exact provenance also run `git -C SGL checkout e74eaeac27ce2a483da7717645eb0f6e8e708acd` before preprocessing.

`prepare_chase.py` fits15 preprocessing candidates **only on the20 DRIVE training image/NIfTI pairs**, then applies the winning transform unchanged to CHASE at original resolution. The recorded winner is green-channel OpenCV CLAHE, clipLimit2, tile8x8, DRIVE training MAE0.0. The output is generated under `runs/chase_data`; split/provenance files are generated under `runs`. To use existing external checkouts instead of the default directories, set environment variables `DSCNET_DIR` and `SGL_DIR` to your own checkout locations. `DSCNET_DIR` names the repository root, not its nested2D folder.

Metrics use whole images without FOV masking; patch topology uses64px tiles/stride64 and excludes incomplete edge strips. The inherited label-free normalization is computed separately for each image collection; it is not a strictly inductive training-statistics-only transform. Original sliding-window center singularities and NIfTI write/read behavior are retained. These limitations matter especially for topology.

## 3. CPU tests and the minimal repair

After downloading the pinned baseline, these tests need **no data or trained weights**, and work on CPU:

```bash
python tests/test_dsconv_fix.py DSCNet/DSCNet_2D_opensource/Code/DRIVE/DSCNet
python tests/test_dsconv_shuffle.py DSCNet/DSCNet_2D_opensource/Code/DRIVE/DSCNet
python tests/test_stats.py
python tests/test_power_exact_acceleration.py
python tests/test_posthoc_report.py
python tools/audit_rebuild.py
```

`tests/test_source_cache.py` additionally requires both prepared datasets and the pinned training dependencies; it is deliberately excluded from data-free checks. The defect tests automatically use an available GPU, otherwise CPU. The public package adds copyright/provenance headers while preserving the original mathematical operations.

`modules/dsconv_fix.py` retains each sample's flattened spatial offset. The representative one-line change is `base_y0 - base + x0` → `base_y0 + x0`; apply the analogous change to all four interpolation neighbors. Usage:

```python
import S3_DSConv
from modules.dsconv_fix import patch_dsc
patch_dsc(S3_DSConv)
```

The upstream file stays unchanged. Batch1 original/fixed outputs are bitwise identical in the controls; fixed batch-vs-singleton comparisons use floating-point tolerance. The newer upstream pro operator is different and is not treated as an exactly equivalent repair.

## 4. Reproduce each experiment

The table gives **planning estimates**, not isolated method speed benchmarks. Original runs shared a GPU. Full training from scratch, including earlier configurations reused in the study, can require roughly 50–70 single-GPU hours; calibrate your hardware on a complete run. Commands resume generated checkpoints. They do not start a long-lived scheduler and do not alter system power settings.

| Experiment | Device / expected duration | Command |
|---|---|---|
| Data-free checks | CPU, about 1–3min | Section3 |
| CHASE preparation | CPU, about 2–10min after downloads | `python prepare_chase.py` |
| Synthetic E1,20 seeds / two morphs / five batches | CPU, about 1–5min | `python ops/run_e1_local.py` |
| Whole-network E1 partsA–D | CPU; roughly 2–6h depending on processor | Command below; baseline checkpoint/data required |
| Seed0 tuning and default-LR references | One 24 GB GPU, roughly 8–15h | `python tools/run_experiments.py --phase tuning` |
| E2 original/fixed,5 seeds each, batch8/K8 | One 24 GB GPU, roughly 12–20h, scan runs reused | `python tools/run_experiments.py --phase e2` |
| Main matrix: six DRIVE, five CHASE models,5 seeds | One 24 GB GPU, roughly 20–35h, completed seed0 runs reused | `python tools/run_experiments.py --phase main` |
| Optional sixth CHASE model: SumProd,5 seeds | One 24 GB GPU, roughly 2–4h more | `python tools/run_experiments.py --phase main --include-chase-sumprod` |
| Four exploratory E2 controls | One 24 GB GPU, roughly 5–8h | `python tools/run_experiments.py --phase posthoc` |
| Published-table paired statistics | CPU, seconds to a minute | `python tools/stats_from_tables.py` |
| All fresh-run tables/figures and10,000-resample sensitivity | CPU, about 5–30min; prepared data/predictions required | `python finalize.py` |
| Audit references/counts | CPU, seconds offline / several minutes online | `python tools/audit_rebuild.py --online` |

Preview the exact training arguments without launching computation:

```bash
python tools/run_experiments.py --phase all --dry-run
```

Default commands perform new **validation-only** LR selection from completed seed0 scans. `--use-published-lr` explicitly selects the reported LR choices from `configs/lr_selected.json` instead. The selected DRIVE DSCNet/plugin LR2e-4 is inherited from the earlier baseline tuning; DRIVE U-Net, CHASE U-Net/DSCNet and E2 use their own declared scans. All variants preserve 400 epochs, ROI224, AdamW betas(0.9,0.95), weight_decay0.01, constant LR and fixed seeds0–4. Plugin insertion is b2,b3,b4. Capacity matching applies to the convolution control; Mamba parameters are reported independently.

The six architecture commands cover matched U-Net, DSCNet, `+conv`, `+sumprod`, `+trop` and `+mamba` for either dataset. **CHASE SumProd was not run in the reported primary study**. The optional command provides the full six-model/two-dataset/five-seed extension, but it must not be described as an existing result. The primary finalizer intentionally retains the original five-model CHASE comparison and its Holm family; optional SumProd outputs remain separate generated runs.

For an individual unchanged E2 run:

```bash
python train_clean.py --run drive_dscnet_none_fix_b8_lr0.001_s0 --dataset drive --variant dscnet --kind none --insert b2,b3,b4 --lr 0.001 --seed 0 --epochs 400 --bs 8 --roi_per_image 8 --dsconv fix --diag_test
```

E2 uses 128 sampled ROIs per epoch but 16 optimizer steps, matching batch1/K1 step count. The fixed seed0 LR scan is2e-4/5e-4/1e-3; the original uses the same winner. Validation selects the checkpoint at epochs201,206,...,396. Diagnostic test evaluations at those 40 epochs are recorded only for analysis and never select a checkpoint/LR. Every run evaluates the selected and final checkpoints separately; E2 also packs same-image inference tiles in groups of up to8.

For the whole-network E1 experiment, first obtain the DRIVE DSCNet seed0 checkpoint by running the main phase. The following output paths are **generated by the commands**, not included weights:

```bash
python experiments/e1_whole_network/e1_bug.py DSCNet/DSCNet_2D_opensource runs/drive_dscnet_none_orig_b1_lr0.0002_s0/best.pt runs/e1_whole_network ABCD
python results/figures/render_e1_local.py --input-dir results/e1_synthetic --output-dir reproduced/e1
```

The reported whole-network E1 results use the earlier validation-selected batch1 model at the same LR/seed. Packed batch8 outputs were computed through exact two-sample pairs to limit CPU memory; the included equivalence JSON records the comparison with a true batch8 forward. This experiment changes inference batching, not training. PartsB/C have their own ROI128/96 synthetic-gradient limitations. Set `E1_START` to a zero-based test-image offset for an interrupted partD; retain earlier rows when combining a resumed result. The output JSONs are numeric; generated images and predictions stay outside version control.

## 5. Statistics, figures and audit reconstruction

`analysis_stats.py` first averages seeds within each image, then uses paired two-sided Wilcoxon and10,000 image-bootstrap samples (seed20261006). Holm correction is applied per declared dataset/contrast family. Differences are rounded to12 digits before rank tests. Five seeds and20/8 test images limit generalization; nonsignificance does not establish equivalence. The power/sensitivity grids fail nominal5% null calibration and **do not provide validated sample-size recommendations**.

`tools/stats_from_tables.py` recomputes the primary contrasts from released per-image CSVs without training or datasets. `finalize.py` handles fresh per-run metrics, losses, diagnostics and qualitative predictions, writing a **separate `reproduced` directory**, not overwriting published results. Its figure2–7 plotting code is included. Example commands for the standalone paper layouts:

```bash
python results/figures/fig0_study_overview.py reproduced/fig0_study_overview
python results/figures/fig1_workflow.py reproduced/fig1_workflow
python results/figures/fig1_study_overview.py results/tables/per_seed_drive.csv results/tables/per_seed_chase.csv reproduced/fig1_study_overview
python results/figures/fig4_magnitudes_v2.py
```

Create the output directory first (`mkdir -p reproduced`). The last script renders the rounded published magnitudes in the current directory; run it in an output directory if retaining a clean checkout. Publication layout scripts include hard-coded schematic/rounded numbers; the numeric CSVs/JSONs are the source of quantitative reporting.

The audit covers 14 reviewed segmentation repositories (12 assessable vessel training paths) and 11 Python Dynamic Snake ports. Four retain the same cancellation; seven lack this specific pattern, without being certified generally correct. See `audit/audit_method_codex.md` for convenience sampling, excluded scopes and code-copy dependence. `tools/audit_rebuild.py` verifies IDs/counts offline; `--online` fetches referenced immutable files **only in memory**, checks SHA256 and line ranges, and emits counts. It does not publish complete sources or automatically replace the original manual classifications. Evidence links remain usable even though upstream source snapshots are excluded.

## 6. Included results and attribution

* `results/tables`: all delivered CSV/Markdown tables, including per-image/per-seed results and exploratory controls.
* `results/figures`: delivered PDF/PNG figures, captions and plotting scripts; numbering includes alternative manuscript layouts.
* `results/numbers.json`: primary numerical results and historical relative per-run provenance.
* `results/numbers_E1_E6_codex.json`: independent synthetic E1/audit numbers; not merged into original E2 inference.
* `experiments/e1_whole_network`: the supplied whole-network E1 script and numeric JSONs.
* `audit`: source-reference evidence tables and method, without complete third-party source text.
* `configs`: reported LR choices, run configuration table and separate exploratory configuration plan.

Historical `runs` references in the numeric records refer to generated experiment outputs. They are not included weights or datasets. Public integration changes paths, attribution, English text and reporting entry points; it does not change the released measured metrics. Code is MIT, copyright Heng Zhang. Adapted DSCNet files retain Copyright (c) 2025 Yaolei Qi; see `LICENSE` and `THIRD_PARTY_NOTICES.md`.
