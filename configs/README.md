# Configurations

- `viral_recall_weighted_3seed.json`: primary weighted PediLite-SE repair.
- `viral_recall_weighted_no_se_3seed.json`: matched PediLite-none ablation with
  all SE blocks removed.
- `mobilenet_v3_small_3seed.json`: matched transfer-learning baseline.
- `pilot.json`: short smoke-test configuration for checking a local setup.

Replace `data_root` in each configuration with a local corrected split path.
The test split must remain frozen until the final evaluation.
