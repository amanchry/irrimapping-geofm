# Phase 4 · Multi-model fusion

!!! note "Starter page"
    Optional extension, planned for December 2026 – January 2027 if time allows.

**Research question RQ5:** does fusing the features or outputs of several foundation models improve on the best single model?

## Candidate strategies

| Strategy | Description |
| --- | --- |
| Early fusion | concatenation of the pixel / patch features of several frozen encoders ([E03](../experiments/E03_frozen_geofm.md)) before the decoder |
| Late fusion | combination of the per-pixel class probabilities of the single-model runs (averaging or stacking) |
| Learned fusion | a small network that weights the features of the encoders, trained under the same protocol |

All strategies are compared with the best single model of [Phase 2](../phase2_models/overview.md) on the same
out-of-fold pixels, folds and bootstrap samples ([evaluation protocol](../phase2_models/evaluation.md)). The complementarity
of the models (different pretraining data and objectives, e.g. AgriFM with its temporal design) motivates the
comparison. All inputs are Sentinel-2 only ([D14](../decisions.md#d14-sentinel-2-only-for-every-input)).
