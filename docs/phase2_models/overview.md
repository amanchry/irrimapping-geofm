# Phase 2 · Model evaluation

Phase 2 compares geospatial foundation models (GeoFMs) and ways of adapting them for classifying **every 10 m pixel**
as irrigated, rainfed or not cropland ([D16](../decisions.md#d16-pixel-wise-classification-for-continuous-maps)), in
Morocco and Utah ([D13](../decisions.md#d13-utah-as-a-second-independent-study-area)). It answers two research
questions under one [evaluation protocol](evaluation.md); every model is scored on the same out-of-fold pixels.

| RQ | Question | Experiments | Status | Page |
| --- | --- | --- | --- | --- |
| **RQ1** | How accurately do frozen GeoFMs classify the pixels, compared with classifiers trained on the Sentinel-2 time series and a network trained from scratch? | E01–E04 | ✅ done for four of six GeoFMs (Presto and TESSERA to run) | [RQ1 · Experiment plan](experiment_plan.md) |
| **RQ2** | To what extent does fine-tuning (LoRA or full) improve on the frozen encoder and on training from scratch, and how does this depend on the number of labelled parcels? | E05 (Prithvi, Utah) | ⏳ frozen, fine-tuned and from scratch running; LoRA later | [RQ2 · Fine-tuning](fine_tuning.md) |

## Workflow

```mermaid
flowchart TD
    subgraph P1["Phase 1 · Data"]
        L["Labelled parcels<br/>Morocco 2023/24 + 2024/25 · Utah 2023/24"]
        M["Cropland mask<br/>(four products)"]
        S["Pixel samples<br/>irrigated · rainfed · not cropland"]
        T["Tile store<br/>128 px tiles · 12 monthly Sentinel-2 composites"]
        F["5 spatial folds<br/>(20 km blocks)"]
        L --> S
        M --> S
        L --> F
    end
    subgraph P2["Phase 2 · Experiments"]
        E1["E01 · random forest,<br/>gradient boosting"]
        E2["E02 · TempCNN"]
        E3["E04 · frozen GeoFMs<br/>+ L-TAE + UPerNet"]
        E4["E03 · U-TAE<br/>from scratch"]
        E5["E05 · Prithvi frozen / LoRA /<br/>fine-tuned / from scratch"]
    end
    S --> E1
    S --> E2
    T --> E3
    T --> E4
    T --> E5
    F --> E1
    E1 --> R["Out-of-fold predictions →<br/>cropland macro-F1 with block-bootstrap CI,<br/>paired tests (Holm)"]
    E2 --> R
    E3 --> R
    E4 --> R
    E5 --> R
    R --> P3["Phase 3 · temporal and spatial transferability"]
    R --> P5["Phase 5 · annual maps"]
```

| Page | Content |
| --- | --- |
| [RQ1 · Experiment plan](experiment_plan.md) | how E01–E04 answer RQ1, status and scores |
| [RQ2 · Fine-tuning](fine_tuning.md) | the ways to adapt a GeoFM, what E05 tests, papers to read |
| [Evaluation protocol](evaluation.md) | task, spatial folds, fairness rules, metrics, confidence intervals, how a result is read |
| [Experiment plan (all experiments)](../experiments/plan.md) | shared settings of E01–E05 |
| [Foundation models](../foundation_models.md) | the six GeoFMs and how each is applied |

The earlier parcel-wise Phase 2 (frozen embeddings pooled over parcels, PCA or L-TAE and classifiers) is in the
[archive](../archive/parcel_wise_classification/workflow.md).
