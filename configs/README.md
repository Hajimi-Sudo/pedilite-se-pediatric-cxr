# Configurations

This directory contains JSON configurations for reproducing the 16-model benchmark, ablations, and stress tests reported in the manuscript:

## Primary PediLite-MS Benchmarks
- `pedilite_ms_matched_weighted_3seed.json`: Proposed PediLite-MS model (3 seeds: 42, 2026, 3407) with weighted cross-entropy and validation viral-recall checkpointing.
- `pedilite_ms_ablation_3seed.json`: Systematic kernel and attention ablations ($3\times3$ only, $5\times5$ only, and mixed-scale with Squeeze-and-Excitation).
- `pedilite_ms_kd_efficientnet_b0_3seed.json`: PediLite-MS trained via response-based knowledge distillation from EfficientNet-B0 teacher.

## Comparative Baselines
- `light_family_scratch_3seed.json`: Canonical lightweight architectures trained from scratch (MobileNetV2, ShuffleNetV2-x1.0).
- `light_family_weighted_3seed.json`: ImageNet-pretrained lightweight baselines (MobileNetV2, ShuffleNetV2-x1.0, EfficientNet-B0).
- `recent_models_weighted_3seed.json`: Modern heavy vision backbones (ConvNeXt-Tiny, Swin V2-T, EfficientNetV2-S).
- `recent_sota_maxvit_t_3seed.json`: Modern multi-axis attention comparator (MaxViT-T).
- `mobilenet_v3_small_3seed.json`: Reference MobileNetV3-Small transfer-learning baseline.

## Quick Checks & Smoke Tests
- `pilot.json`: Smoke-test configuration for verifying local GPU/CPU pipelines.
- `pedilite_ms_ac_smoke_seed42.json`: 3-epoch smoke test for adaptive early-exit research exploration.

Replace `"data_root"` in each configuration with your local leak-free split path before running.
