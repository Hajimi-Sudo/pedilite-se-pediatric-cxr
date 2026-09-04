# Data Directory

No dataset is stored in this repository.

Expected external dataset examples:

```text
chest_xray/
  train/
    NORMAL/
    PNEUMONIA/
  val/
    NORMAL/
    PNEUMONIA/
  test/
    NORMAL/
    PNEUMONIA/
```

When pneumonia filenames contain `bacteria` or `virus`, the code can create a three-class diagnostic label set:

- `normal`
- `viral_pneumonia`
- `bacterial_pneumonia`

If the audit finds only binary labels, the experiment must be reported as calibrated diagnostic-risk stratification rather than etiologic or clinical severity grading.
