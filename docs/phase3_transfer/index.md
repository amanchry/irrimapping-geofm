# Phase 3 · Temporal transferability and a second region

!!! note "Starter page"
    Planned for February – March 2027. This page will be extended into an overview and component pages when the work
    begins.

**Objective 2:** evaluate the temporal transferability of the models of [Phase 2](../phase2_models/overview.md) and
whether their results hold in a second, independent region (Utah).

| RQ | Question | Planned test |
| --- | --- | --- |
| **RQ3** (temporal) | How temporally consistent is the best model across years? | **Utah:** train on WRLU 2024, test on WRLU 2021 on the folds held out in training (same fields, paired); Morocco: year-to-year consistency of the [annual maps](../phase5_maps/index.md) |
| **RQ4** (second region) | Do the results of Morocco hold in a second, independent region? | the full Phase 2 comparison repeated in **Utah** with its own labels and spatial folds ([D13](../decisions.md#d13-utah-as-a-second-independent-study-area)); no Morocco → Utah transfer |

## Decided

- **Second region and temporal test:** Utah ([D13](../decisions.md#d13-utah-as-a-second-independent-study-area));
  dataset: [Utah training data](../phase1_data/utah.md). The temporal test holds out space as well, because the
  WRLU labels barely change between years.
- **Metrics:** macro-F1 per region as in Phase 2; temporal: the drop from WRLU 2024 (score A) to WRLU 2021
  (score B) on the same held-out fields.

## Open

- Wording of RQ3 / RQ4 in the report, to be agreed with the supervisors (departure from the proposal).

## Inputs from earlier phases

- best model and training configuration from Phase 2;
- Utah dataset built with the Phase 1 pipeline ([component 1b](../phase1_data/overview.md#components)).
