# NMF-PAC

**Non-negative Matrix Factorization-based Prototype pseudo-labeling with Adaptive
Thresholding and Contrastive consistency** for unsupervised domain adaptation (UDA)
in medical image classification.

NMF-PAC builds class prototypes from labeled source features via rank-1 NMF, assigns
target pseudo-labels from the **non-negative reconstruction responsibilities** of each
target sample to the prototypes, filters them with a **class-adaptive dynamic
threshold**, maintains the prototype library by EMA, and refines the class-conditional
decision boundary with a **symmetric cross-domain InfoNCE** loss. Training alternates
between a pseudo-label mining stage (network frozen) and a model optimization stage.

> This repository contains the reference implementation of **NMF-PAC** and the
> **knee-MRI** cross-dataset task (MRNet-v1.0 ↔ KneeMRI) as a runnable example. The
> method core (`uda_core/`) is modality-agnostic; the X-ray and breast-ultrasound
> tasks in the paper use the same code with different data configurations.

---

## Repository structure

```
nmf-pac/
├── uda_core/                 # method core (modality-agnostic)
│   ├── config.py             #   CLI / hyper-parameters (build_parser)
│   ├── trainer.py            #   two-stage alternating training loop
│   ├── prototypes.py         #   NMF class prototypes + responsibility assignment
│   ├── thresholds.py         #   class-adaptive EMA thresholding
│   ├── contrastive.py        #   symmetric cross-domain InfoNCE
│   ├── data_builders.py, pseudo_dataset.py, repro.py
│   └── train_full_supervised.py
├── custom_net.py             # spatial-attention-enhanced ResNet-50 backbone
├── data.py                   # dataset loaders (.npy slices + CSV splits)
├── proto_kmeans.py           # K-means prototype baseline / K search
├── nmf_lib_assign.py         # NMF helpers (min-max, component align, row-normalize)
├── calib.py                  # temperature calibration utilities
├── train_online_end2end.py   # >>> main entry: NMF-PAC adaptation
├── train_test_src.py         # source-only supervised pretraining
├── scripts/                  # reproduction & analysis (edit DATA_ROOT before use)
├── requirements.txt · LICENSE · .gitignore
```

`data/`, `outputs/`, and model checkpoints (`*.pth`) are **not** tracked — see below.

---

## Installation

```bash
git clone https://github.com/RuiPingYe-lim/nmf-pac.git
cd nmf-pac
pip install -r requirements.txt
```
A CUDA-enabled PyTorch is recommended. The reported results were produced with
Python 3.12.3 and PyTorch 2.7.0+cu128 (CUDA 12.8, cuDNN 9.7.1); scikit-learn 1.9.0,
NumPy 2.2.6. Earlier versions (Python 3.10 / PyTorch 1.12+) also run, but the exact
numbers are only reproducible under the environment above with `--deterministic`.

---

## Data preparation

The public datasets are **not redistributed** here. Download them from the original
sources and preprocess each case into per-case `.npy` slices plus CSV split files.

| Modality | Datasets | Source |
|---|---|---|
| Knee MRI (this repo's example) | MRNet-v1.0, KneeMRI | Stanford MRNet; KneeMRI (Štajduhar et al.) |
| Knee X-ray | OAI kneeKL224, MedicalExpert-II | OAI; Mendeley Data |
| Breast ultrasound | BUSI, BrEaST | BUSI; BrEaST |

Expected layout (per dataset):

```
data/<dataset>/knees_npy/<case_id>.npy   # preprocessed image tensor(s) per case
data/<dataset>/train_0.csv               # columns: case_id,label
data/<dataset>/valid_0.csv
data/<dataset>/test_0.csv
```

Splits are fixed before training; target labels are used **only** for final evaluation
(never for training, pseudo-labeling, thresholding, or checkpoint selection).

---

## Usage

### 1. Source-only pretraining (produces the initialization checkpoint)

```bash
python train_test_src.py \
  --npyfile_src data/MRNet-v1.0/knees_npy --src_csv data/MRNet-v1.0/train_0.csv \
  --src_val_csv data/MRNet-v1.0/valid_0.csv \
  --plane sagittal --resize 224 \
  --backbone custom_resnet50_space \
  --lr 3e-4 --weight_decay 6e-4 --batch_size 32 \
  --save_dir outputs/src_only_mrnet
```

### 2. NMF-PAC adaptation (MRNet-v1.0 → KneeMRI)

```bash
python train_online_end2end.py \
  --npy_src data/MRNet-v1.0/knees_npy --csv_src data/MRNet-v1.0/train_0.csv \
  --csv_src_val data/MRNet-v1.0/valid_0.csv \
  --npy_tgt data/KneeMRI/knees_npy --csv_tgt data/KneeMRI/train_0.csv \
  --npy_tgt_test data/KneeMRI/knees_npy --csv_tgt_test data/KneeMRI/test_0.csv \
  --plane sagittal --resize 224 --num_classes 2 \
  --init_from outputs/src_only_mrnet/custom_resnet50_space_best.pth \
  --use_nmf_pseudo --proto_init nmf \
  --rounds 60 --K 1 --beta_loss frobenius \
  --proto_m 0.97 --nmf_assign_iters 100 \
  --lr 1e-4 --wd 4e-4 --seed 42 \
  --save_dir outputs/nmfpac_mrnet2knee_seed42
```

> **Important:** `--use_nmf_pseudo --proto_init nmf` are **required** — they select
> NMF prototype initialization and NNLS responsibility assignment. Without them the
> defaults (`kmeans` + softmax similarity) run a different, non-NMF-PAC variant.

Key hyper-parameters (defaults in `uda_core/config.py`): `--K` prototypes per class
(rank-1 = 1), `--beta_loss` NMF reconstruction loss (`frobenius`/`kullback-leibler`),
`--proto_m` prototype EMA momentum, `--nmf_assign_iters` NNLS iterations for
responsibility estimation, `--lam_lo`/`--lam_hi` adaptive-threshold quantile bounds
(Section 3.5), `--rounds` target scans, `--seed` random seed.
`--csv_tgt_test` supplies target labels used **only** to report target-test AUC per
round; checkpoint selection uses the source-validation AUC only.

### 3. Evaluation

Evaluation is performed inside the adaptation run itself. After every round the run
evaluates the current model on the source-validation split and on the target test
split and prints both to stdout. The checkpoint reported in the paper is the one with
the highest **source-validation** AUC; at the end of training its metrics are written
to `<save_dir>/metrics_summary.json`. Target labels are never used for training,
pseudo-labeling, thresholding, or checkpoint selection — only to compute the reported
target-test numbers.

```
<save_dir>/
├── metrics_summary.json     # selection_rule, best_round, and the metrics below
│                            #   best_src_val_metrics
│                            #   tgt_test_metrics_at_best_src_val   <-- reported in the paper
│                            #   last_round_{src_val,tgt_test}_metrics
├── best_by_src_val.json     # identical copy, kept for backward compatibility
├── best_by_src_val.pth      # the selected checkpoint
└── args.json, run_manifest.json   # exact hyper-parameters and environment
```

Per-round curves (Fig. 3/4) are parsed from the training stdout by
`scripts/parse_threshold_dynamics.py`; multi-seed tables are aggregated by
`scripts/collect_multiseed_results.py`, which reads `metrics_summary.json`.

No separate evaluation script is provided, so the reported numbers can only come from
the run that produced them.

---

## Reproducing the paper

The `scripts/` folder contains the exact runs used in the paper. **Edit `DATA_ROOT`
(and `PROJ_ROOT`, `OUTPUTS_ROOT`) at the top of each script — or `export` them — before
running**, since they default to local paths.

```bash
export DATA_ROOT=/path/to/data
bash scripts/run_online.sh                    # main MRNet→KneeMRI (multi-seed)
bash scripts/run_online_knee2mrnet.sh         # main KneeMRI→MRNet
bash scripts/run_ablation_mrnet_to_knee.sh    # module-level ablation
bash scripts/run_nmf_sensitivity_mrnet_to_knee.sh   # K / loss / m / T sensitivity
```
`scripts/collect_*.py` aggregate the multi-seed results; `scripts/plot_nmf_sensitivity_2x2.py`,
`scripts/parse_threshold_dynamics.py`, and `scripts/visualize_*.py` produce the figures.

Results are reported as mean±std over seeds `{7, 16, 42}`.

## Baselines

Baseline methods (Deep CORAL, DANN, AdaMatch, ICON, HMA, UDPCS, VirDA) were run from
their **original public repositories** under the same data splits, backbone, and
evaluation protocol; they are not redistributed here. Please refer to the respective
repositories to reproduce them.

## Citation

If you use this code, please cite the paper (under review):

```bibtex
@article{nmfpac,
  title   = {NMF-PAC: Non-negative Matrix Factorization-based Prototype Pseudo-labeling
             with Adaptive Thresholding and Contrastive Consistency for Unsupervised
             Domain Adaptation in Medical Image Classification},
  author  = {Ye, Ruiping and Liu, Weiqiang and Zheng, Yifeng and Chen, Wenlong},
  year    = {2026}
}
```

## License

Released under the [MIT License](LICENSE).
