# PediLite-MS: A Pareto-Efficient Mixed-Scale Inverted Residual CNN for Pediatric Chest Radiograph Classification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official reproducible repository for **PediLite-MS: A Pareto-Efficient Mixed-Scale Inverted Residual CNN for Pediatric Chest Radiograph Classification** (submitted to Elsevier *Image and Vision Computing*).

This repository contains the architecture definition, leak-free grouped split verification utilities, training configs, evaluation routines, post-hoc calibration modules, multi-resolution stress tests, and external validation pipelines. It does not redistribute patient radiographs, clinical DICOMs, or PhysioNet access credentials.

---

## Key Highlights

- **Ultra-Compact Pareto Frontier**: **81,491 parameters** (~81.5K), **29.8M FLOPs**, and **1.28 ms latency**, establishing an unmatched Pareto efficiency boundary for pediatric CXR diagnosis.
- **Dual Mixed-Scale Inductive Bias**: Incorporates dual parallel mixed-scale depthwise separable convolutions ($3\times3$ and $5\times5$) within inverted residual stages, resolving receptive field trade-offs between localized consolidations (bacterial) and diffuse interstitial opacities (viral) without channel expansion overhead.
- **Decisive From-Scratch Superiority**: Outperforms canonical lightweight backbones trained from scratch under identical protocols (MobileNetV2: 0.7669, ShuffleNetV2: 0.7550) while reducing parameter footprints by $27\times$ and $15\times$, respectively.
- **Resolving Heavy Model Capacity Mismatch**: Outperforms transfer-learned heavy vision backbones including ConvNeXt-Tiny (0.7545, 27.8M params) and MaxViT-T (0.7390, 30.4M params), demonstrating that tailored compact inductive bias provides superior regularization against overfitting on clinical cohorts.
- **Rigorous Leak-Free Benchmark**: Evaluated across three matched random seeds (42, 2026, 3407) on an audited, filename-grouped split with zero patient-like overlap and zero exact hash duplication.
- **Transparent Cross-Dataset Stress Testing**: Includes frozen zero-shot transportability evaluation on 1,397 independent images from the Vietnamese VinDr-PCXR cohort to delineate the boundaries of domain shift.

---

## Benchmark Results (3 Matched Seeds)

All models are evaluated on the leak-free grouped Kermany test split (4,089 train, 886 val, 881 test). Metrics report mean $\pm$ standard deviation across three matched seeds (42, 2026, 3407) with validation-fitted temperature scaling:

| Category | Model | Pretraining | Accuracy | Macro-F1 | AUROC | ECE | Parameters | FLOPs |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Proposed** | **PediLite-MS** | **Scratch** | **0.7775$\pm$0.0169** | **0.7772$\pm$0.0139** | **0.9119$\pm$0.0114** | **0.0423** | **81,491** | **29.8M** |
| Ablation | PediLite-MS ($3\times3$ only) | Scratch | 0.7507$\pm$0.0288 | 0.7538$\pm$0.0293 | 0.9033$\pm$0.0101 | 0.0464 | 77,651 | 28.4M |
| Ablation | PediLite-MS ($5\times5$ only) | Scratch | 0.7397$\pm$0.0142 | 0.7407$\pm$0.0134 | 0.8923$\pm$0.0091 | 0.0418 | 85,331 | 31.1M |
| Ablation | PediLite-MS (+ SE Attention) | Scratch | 0.7563$\pm$0.0194 | 0.7618$\pm$0.0171 | 0.9096$\pm$0.0057 | 0.0389 | 97,775 | 29.8M |
| Scratch Baselines | ShuffleNetV2-x1.0 | Scratch | 0.7476$\pm$0.0318 | 0.7550$\pm$0.0294 | 0.9070$\pm$0.0058 | 0.0258 | 1,256,679 | 143.9M |
| Scratch Baselines | MobileNetV2 | Scratch | 0.7575$\pm$0.0139 | 0.7669$\pm$0.0110 | 0.9168$\pm$0.0052 | 0.0324 | 2,227,715 | 299.5M |
| Transfer Lightweight | MobileNetV3-Small | ImageNet | 0.7764$\pm$0.0623 | 0.7651$\pm$0.0595 | 0.9088$\pm$0.0210 | 0.0397 | 1,520,931 | 55.5M |
| Transfer Lightweight | MobileNetV2 | ImageNet | 0.7677$\pm$0.0127 | 0.7783$\pm$0.0121 | 0.9210$\pm$0.0009 | 0.0256 | 2,227,715 | 299.5M |
| Transfer Lightweight | ShuffleNetV2-x1.0 | ImageNet | 0.7904$\pm$0.0084 | 0.7979$\pm$0.0059 | 0.9166$\pm$0.0038 | 0.0319 | 1,256,679 | 143.9M |
| Transfer Lightweight | EfficientNet-B0 | ImageNet | 0.8146$\pm$0.0091 | 0.8163$\pm$0.0088 | 0.9263$\pm$0.0030 | 0.0354 | 4,011,391 | 384.5M |
| Heavy Architectures | MaxViT-T | ImageNet | 0.7212$\pm$0.0487 | 0.7390$\pm$0.0455 | 0.9167$\pm$0.0116 | 0.0543 | 30,409,163 | 2011.0M |
| Heavy Architectures | ConvNeXt-Tiny | ImageNet | 0.7374$\pm$0.0133 | 0.7545$\pm$0.0107 | 0.9254$\pm$0.0070 | 0.0434 | 27,820,131 | 4460.0M |
| Heavy Architectures | Swin V2-T | ImageNet | 0.7639$\pm$0.0103 | 0.7748$\pm$0.0093 | 0.9195$\pm$0.0065 | 0.0372 | 27,584,877 | 4360.0M |
| Heavy Architectures | EfficientNetV2-S | ImageNet | 0.7450$\pm$0.0567 | 0.7592$\pm$0.0491 | 0.9234$\pm$0.0101 | 0.0515 | 20,177,651 | 2870.0M |
| Distillation | PediLite-MS $\leftarrow$ B0 | Scratch (KD) | 0.7684$\pm$0.0177 | 0.7686$\pm$0.0153 | 0.9077$\pm$0.0074 | 0.0331 | 81,491 | 29.8M |

---

## Environment Setup

Requirements: Python 3.10+, PyTorch 2.5+, CUDA 12.4 (optional, CPU supported).

```bash
git clone https://github.com/Hajimi-Sudo/pedilite-se-pediatric-cxr.git
cd pedilite-se-pediatric-cxr
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

---

## Dataset Preparation & Split Integrity Audit

### 1. Primary Public Kermany Dataset
Download the Kermany pediatric CXR dataset from [Mendeley Data (v2)](https://doi.org/10.17632/rscbjbr9sj.2).

### 2. Generate Leak-Free Filename-Grouped Split
To eliminate cross-split patient-like overlap:
```bash
PYTHONPATH=src python scripts/15_prepare_subject_split.py   --raw-root /path/to/raw_kermany   --out-root data/corrected_split   --label-mode three_class   --seed 42

PYTHONPATH=src python scripts/13_split_integrity_audit.py   --data-root data/corrected_split   --out-dir results/corrected_split_audit
```

### 3. Optional VinDr-PCXR External Cohort
Access requires a credentialed PhysioNet account: [VinDr-PCXR v1.0.0](https://physionet.org/content/vindr-pcxr/1.0.0/). Set environment credentials `VINDR_USER` and `VINDR_PASSWORD`. Never commit credentials to git.

---

## Reproduction Workflow

### 1. Train Proposed PediLite-MS (3 Matched Seeds)
```bash
PYTHONPATH=src python scripts/02_run_experiment.py   --config configs/pedilite_ms_matched_weighted_3seed.json
```

### 2. Run Systematic Ablation Suite ($3\times3$, $5\times5$, SE Attention)
```bash
PYTHONPATH=src python scripts/02_run_experiment.py   --config configs/pedilite_ms_ablation_3seed.json
```

### 3. Run Scratch & Transfer Baselines
```bash
# Scratch lightweight baselines (MobileNetV2, ShuffleNetV2)
PYTHONPATH=src python scripts/02_run_experiment.py   --config configs/light_family_scratch_3seed.json

# Pretrained lightweight baselines (MobileNetV2, ShuffleNetV2, EfficientNet-B0)
PYTHONPATH=src python scripts/02_run_experiment.py   --config configs/light_family_weighted_3seed.json

# Heavy modern architectures (ConvNeXt-Tiny, Swin V2-T, EfficientNetV2-S)
PYTHONPATH=src python scripts/02_run_experiment.py   --config configs/recent_models_weighted_3seed.json
```

### 4. Multi-Resolution Robustness Analysis (Locked Weights)
Evaluate resolution sensitivity across 128, 160, 192, 224, and 256:
```bash
PYTHONPATH=src python scripts/24_eval_multires.py   --model pedilite_ms   --weights results/pedilite_ms_matched_weighted   --data-root data/corrected_split
```

### 5. Frozen Zero-Shot External Stress Test (VinDr-PCXR)
```bash
PYTHONPATH=src python scripts/19_external_vindr_validation_fast_pr.py   --model pedilite_ms   --weights results/pedilite_ms_matched_weighted   --vindr-root data/vindr_pcxr_test
```

---

## Codebase Architecture

```
pedilite-se-pediatric-cxr/
├── configs/                  # Experiment JSON configs for all 16 models
│   ├── pedilite_ms_matched_weighted_3seed.json
│   ├── pedilite_ms_ablation_3seed.json
│   ├── light_family_scratch_3seed.json
│   ├── light_family_weighted_3seed.json
│   └── recent_models_weighted_3seed.json
├── data/                     # Data layout documentation & split helpers
├── scripts/                  # Split preparation, audit, and benchmark runners
│   ├── 02_run_experiment.py
│   ├── 13_split_integrity_audit.py
│   ├── 15_prepare_subject_split.py
│   ├── 19_external_vindr_validation_fast_pr.py
│   └── 24_eval_multires.py
├── src/pedilite/             # Core library package
│   ├── models.py             # PediLite-MS, inverted residuals, baseline wrappers
│   ├── train.py              # Training engine, calibration, early-stop rules
│   ├── data.py               # Safe dataset discovery & leak-free split loaders
│   ├── metrics.py            # Macro-F1, AUROC, ECE, Brier score
│   └── utils.py              # FLOPs estimator, latency profiler, seed management
├── pyproject.toml            # Project packaging specification
└── requirements.txt          # Python dependencies
```

---

## Scope, Limitations, and Clinical Boundaries

1. **Diagnostic Support vs. Autonomous Triage**: PediLite-MS is designed as a computer-aided diagnostic decision support tool for low-resource environments; it is not intended for autonomous clinical diagnosis or triage.
2. **Pathological Labels**: Labels on public CXR datasets represent image-level diagnostic findings rather than microbiological reference standards (e.g., blood/sputum culture or PCR).
3. **Transportability**: As characterized by our external zero-shot test on the Vietnamese VinDr-PCXR cohort, single-source public CXR models experience significant domain shift and require localized fine-tuning prior to real-world deployment.

---

## Citation

```bibtex
@article{liu2026pedilitem,
  title={PediLite-MS: A Pareto-Efficient Mixed-Scale Inverted Residual CNN for Pediatric Chest Radiograph Classification},
  author={Liu, Cuicui and Dong, Qiyu and Liang, Weijun and Wu, Shuang},
  journal={Image and Vision Computing},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
