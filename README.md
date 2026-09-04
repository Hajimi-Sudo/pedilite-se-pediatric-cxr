# PediLite-SE

Reproducible code for **PediLite-SE: A Parameter-Efficient Attention CNN for
Pediatric Chest Radiograph Classification**.

This repository contains the model definition, corrected split audit, training
configs, evaluation scripts, calibration utilities, and the matched three-seed
ablation in which all squeeze-and-excitation (SE) blocks are removed
(`pedilite_none`). It does not contain patient images, VinDr credentials, or
trained weights.

## Scope and limitations

The Kermany-derived public data support image-level diagnostic categories:
`normal`, `viral pneumonia`, and `bacterial pneumonia`. These labels are not
microbiological reference standards and do not represent clinical severity.
The code is for research reproducibility, not clinical diagnosis or deployment.
The corrected split uses filename-derived subject-like grouping and exact-hash
checks; authoritative patient identifiers were not available.

## Dataset availability

Download the public Kermany dataset from [Mendeley Data, version 2](https://data.mendeley.com/datasets/rscbjbr9sj/2).
Do not commit images or derived patient-level data. Expected layouts and label
inference rules are documented in [`data/README.md`](data/README.md).

VinDr-PCXR is optional and is used only as a frozen cross-dataset stress test.
Access requires the user's PhysioNet credentialed data-use process; set
`VINDR_USER` and `VINDR_PASSWORD` as environment variables and never place
credentials in a config file or commit history.

## Code availability

The source code, resolved configurations, and reproducibility instructions are
available in this repository. This is separate from dataset availability: the
repository does not redistribute patient images, credentials, or trained
weights.

## Environment

The completed server runs used Python 3.12.3, PyTorch 2.5.1+cu124, CUDA 12.4,
and an NVIDIA RTX 2080 Ti. Install the public dependencies with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

For a CPU-only installation, select the appropriate PyTorch wheel first, then
install the remaining requirements.

## Reproduction workflow

1. Audit the downloaded dataset:

   ```bash
   PYTHONPATH=src python scripts/01_audit_dataset.py \
     --data-root /path/to/raw_dataset \
     --label-mode three_class \
     --out results/label_audit.json
   ```

2. Create the corrected filename-derived grouped split:

   ```bash
   PYTHONPATH=src python scripts/15_prepare_subject_split.py \
     --raw-root /path/to/raw_dataset \
     --out-root data/corrected_split \
     --label-mode three_class \
     --seed 42
   PYTHONPATH=src python scripts/13_split_integrity_audit.py \
     --data-root data/corrected_split \
     --out-dir results/corrected_split_audit
   ```

3. Run the three matched weighted PediLite-SE seeds:

   ```bash
   PYTHONPATH=src python scripts/02_run_experiment.py \
     --config configs/viral_recall_weighted_3seed.json
   ```

4. Run the matched no-SE ablation:

   ```bash
   PYTHONPATH=src python scripts/02_run_experiment.py \
     --config configs/viral_recall_weighted_no_se_3seed.json
   ```

5. Optional follow-up protocols for external transportability and baseline
coverage are provided as separate configs:

   ```bash
   PYTHONPATH=src python scripts/02_run_experiment.py \
     --config configs/external_binary_source_3seed.json
   PYTHONPATH=src python scripts/02_run_experiment.py \
     --config configs/corrected_baselines_weighted_3seed.json
   ```

   The first trains task-aligned binary source models for frozen VinDr-PCXR
   evaluation; it does not create viral-versus-bacterial labels on VinDr. The
   second adds matched small-CNN, ECA, CBAM, and EfficientNet-B0 comparisons.
   Both protocols keep the corrected split, validation-only model selection,
   and the prespecified three seeds.

6. Use [`scripts/12_harden_results.py`](scripts/12_harden_results.py),
   [`scripts/14_seed_stats.py`](scripts/14_seed_stats.py), and
   [`scripts/18_diagnostic_metrics.py`](scripts/18_diagnostic_metrics.py) for
   frozen-test summaries. Threshold selection, if used, must be fitted on
   validation predictions only (`scripts/20_threshold_select.py`).

## Main configuration

All configs use a fixed 224 x 224 input, 30 epochs, batch size 32, AdamW,
learning rate 0.001, weight decay 0.0001, dropout 0.2, class-weight power 0.5,
validation viral-recall checkpoint selection, and seeds 42, 2026, and 3407.
The SE ablation changes only the attention blocks: they are replaced with
identity mappings.

## Code map

- `src/pedilite/models.py`: PediLite-SE, `pedilite_none`, and baselines.
- `src/pedilite/data.py`: image discovery and label auditing.
- `src/pedilite/train.py`: training, evaluation, and temperature scaling.
- `src/pedilite/metrics.py`: performance and calibration metrics.
- `scripts/15_prepare_subject_split.py`: corrected grouped split generation.
- `scripts/13_split_integrity_audit.py`: cross-split ID/hash checks.
- `scripts/02_run_experiment.py`: single- or multi-seed experiment runner.
- `scripts/19_external_vindr_validation.py`: frozen VinDr stress test.

## Reproducibility and publication status

The manuscript reports aggregate results and the no-SE ablation. The public
release is hosted at
<https://github.com/Hajimi-Sudo/pedilite-se-pediatric-cxr>. No patient data or
private credentials belong in this repository.
