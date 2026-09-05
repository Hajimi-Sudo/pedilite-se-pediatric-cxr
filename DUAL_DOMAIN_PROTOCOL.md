# Dual-domain PediLite upgrade protocol

## Scientific question

Can a shared compact encoder retain three-class diagnostic typing on the
Mendeley-derived pediatric CXR split while improving binary pneumonia
transportability to VinDr-PCXR?

This is a new multi-domain study. It does not replace the locked single-domain
Children submission results.

## Label contract

- Mendeley head: `normal`, `viral_pneumonia`, `bacterial_pneumonia`.
- VinDr head: `non_pneumonia`, `pneumonia`, derived from the VinDr `Pneumonia`
  label only.
- VinDr cannot provide viral-versus-bacterial supervision in this protocol.

## Required data partitions

Both datasets must have independent train, validation, and test partitions.
The VinDr test set currently used by `19_external_vindr_validation.py` must
remain frozen. It cannot be used for loss weighting, checkpoint selection,
threshold selection, architecture choice, or early stopping.

If VinDr train/validation data are used for adaptation, a separate unseen
dataset is required for an external-transport claim. Otherwise VinDr remains a
stress-test-only dataset.

## Model and training

`src/pedilite/dual_domain.py` provides a shared depthwise-separable PediLite
encoder, two small domain heads, and domain-specific affine adapters. The
training entry point is `scripts/22_run_dual_domain.py` with
`configs/dual_domain_pedilite_se_3seed.json`.

Each seed alternates batches from both domains. The checkpoint is selected from
the validation-only composite macro-F1, with the VinDr loss contribution
prespecified in the config. Temperature scaling is fit separately on each
validation set and applied once to the corresponding test set.

## Promotion criteria

Promote a v2 model only if three matched seeds satisfy all of the following:

- Mendeley macro-F1 is at least the locked weighted PediLite-SE mean (`0.7689`);
- Mendeley viral sensitivity is at least `0.7040`, with no material loss of
  any-pneumonia sensitivity;
- VinDr binary AUROC and sensitivity both improve over the frozen stress-test
  reference, with confidence intervals reported;
- ECE and Brier score are reported for both domains;
- parameter count and latency remain substantially below MobileNetV3-small.

The first positive result is still a bounded two-domain result, not a claim of
universal state-of-the-art performance or clinical deployment readiness.
