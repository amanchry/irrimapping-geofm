# Phase 3 · Temporal and spatial transferability

!!! note "Starter page"
    Planned for February – March 2027. This page will be extended into an overview and component pages when the work
    begins.

**Objective 2:** evaluate the temporal and spatial transferability of the best model of
[Phase 2](../phase2_models/overview.md).

| RQ | Question | Planned test |
| --- | --- | --- |
| **RQ3** (temporal) | How temporally consistent is the best model across years? | train on one agricultural year and test on the other, within the regions surveyed in both years (Doukkala, El Haouz); year-to-year consistency of the [annual maps](../phase5_maps/index.md) |
| **RQ4** (spatial) | How much does accuracy degrade when the Morocco-trained model is applied to another region without retraining? | zero-shot application to the transfer region; optionally few-shot adaptation with a small number of local labels |

## Open decisions

- **Transfer region** ([D10](../decisions.md#d10-transfer-region-under-review)): Tunisia (54 GMIA-NEXT points, of which
  10 rainfed), eastern Washington (WSDA, irrigated and non-irrigated fields) or Utah (WRLU, 2017–2025, including dry
  crops). See [Data sources](../data_source/index.md#ground-truth).
- Metrics of transfer: absolute accuracy in the new domain and the drop relative to the in-domain spatial
  cross-validation.

## Inputs from earlier phases

- best model and training configuration from Phase 2;
- transfer-region dataset built with the Phase 1 pipeline ([component 1b](../phase1_data/overview.md#components)).
