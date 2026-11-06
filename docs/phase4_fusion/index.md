# Phase 4 · Multi-model fusion

!!! note "Starter page"
    Optional extension, planned for December 2026 – January 2027 if time allows.

**Research question RQ5:** does fusing the embeddings of several foundation models improve on the best single model?

## Candidate strategies

| Strategy | Description |
| --- | --- |
| Early fusion | concatenation of the parcel embeddings of several models before the classifier |
| Late fusion | combination of the class probabilities of the single-model classifiers (averaging or stacking) |
| Learned fusion | a small network that weights the models' embeddings, trained under the same protocol |

All strategies are compared with the best single model of [Phase 2](../phase2_models/overview.md) on the same
parcels, folds and bootstrap samples ([evaluation protocol](../phase2_models/evaluation.md)). The complementarity
of the models (e.g. TerraMind with Sentinel-1, AgriFM with its temporal design) motivates the comparison.
