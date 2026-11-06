# Irrigated Cropland Mapping with Geospatial Foundation Models

**Evaluating geospatial foundation models (GeoFMs) for irrigated cropland mapping in North Africa and their
transferability to data-scarce regions** · MSc thesis, Aman Chaudhary · ITC, University of Twente ·
supervisors Dr. Mahdi Farnaghi and Prof. Dr. Raúl Zurita Milla

The project maps **irrigated** and **rainfed** cropland in Morocco from Sentinel-1 and Sentinel-2 time series. It
compares hand-built satellite features with the embeddings of four foundation models (**Clay v1.5,
Prithvi-EO-2.0, TerraMind, AgriFM**) on the same imagery and the same 8,781 field-surveyed parcels, always tested
on areas the model has never seen.

<div class="grid cards" markdown>

-   :material-school:{ .lg .middle } **Research**

    ---

    What the thesis asks, the method in brief, and where each research question stands.

    [:octicons-arrow-right-24: Thesis design](research/thesis_design.md) ·
    [Method in brief](research/overview.md)

-   :material-cog-outline:{ .lg .middle } **Methodology**

    ---

    The dataset and model pipelines step by step, the evaluation protocol and every decision.

    [:octicons-arrow-right-24: Dataset](dataset/index.md) · [Models](models/index.md) ·
    [Protocol](research/protocol.md) · [Decisions](research/decisions.md)

-   :material-flask-outline:{ .lg .middle } **Experiments**

    ---

    One setup page and one report per experiment, each reproducible from its config file.

    [:octicons-arrow-right-24: All experiments](experiments/index.md)

-   :material-chart-bar:{ .lg .middle } **Results and meetings**

    ---

    The answers so far, and one page per supervisor meeting.

    [:octicons-arrow-right-24: Results](results.md) · [Weekly notes](weekly/index.md)

</div>

## Where the thesis stands

| | |
| --- | --- |
| **Dataset** | ✅ 8,781 parcels (7,585 irrigated, 1,196 rainfed), 5 spatial folds, 2,987 tiles of 12 months × 19 bands |
| **RQ1 · frozen GeoFMs vs S1/S2 baseline** | ⏳ first result ([E01](experiments/E01_frozen_pca_nested/report.md)): no GeoFM beats the baseline (best Clay + SVM 0.910 vs 0.906); main run [E02](experiments/E02_frozen_pca_pastis/index.md) ready |
| **RQ2 – RQ4 · fine-tuning, annual maps, transfer** | 📝 planned ([design](research/thesis_design.md), [decision D11](research/decisions.md#d11-two-kinds-of-head-parcel-benchmark-and-mapping-model)) |
| **Latest changes** | [Progress log](progress.md) |

## Workflow

```mermaid
flowchart TD
    subgraph P1["Part 1 · ml_dataset_prep"]
        GT["<b>Ground data</b><br/>field-surveyed parcels (~10,000)<br/>Dec 2023 – Jan 2025"]
        CL["<b>Data cleaning & labels</b><br/>fix geometry · cropland only<br/>irrigated / rainfed · agricultural year<br/>→ 8,781 labelled parcels"]
        SF["<b>Spatial folds</b><br/>20 km blocks → 5 balanced folds"]
        TS["<b>ML-ready tile store</b><br/>128 × 128 px tiles · 12 months × 19 bands<br/>+ label / parcel-ID / edge rasters"]
        FE["<b>Parcel features</b><br/>monthly medians: S2 bands,<br/>NDVI, NDWI, EVI, S1 VV, VH"]
        C8["<b>Chips p8</b><br/>64 × 64 px around each parcel<br/>aligned to 8-px patches"]
        C16["<b>Chips p16</b><br/>64 × 64 px around each parcel<br/>aligned to 16-px patches"]
        GT --> CL --> SF --> TS
        TS --> FE
        TS --> C8
        TS --> C16
    end

    EE["<b>Google Earth Engine</b><br/>Sentinel-2 L2A · Cloud Score+<br/>Sentinel-1 GRD"] --> TS

    subgraph P2["Part 2 · models_run"]
        BL["<b>S1/S2 baseline</b><br/>204 features per parcel<br/>(17 variables × 12 months)"]
        FM["<b>Foundation models (frozen)</b><br/>Clay v1.5 (p8) · Prithvi-EO-2.0 (p16)<br/>TerraMind (p16) · AgriFM (p16)<br/>→ one embedding per parcel"]
        CF["<b>Classifiers</b><br/>logistic regression · random forest<br/>gradient boosting · SVM<br/>spatial CV: 3 train / 1 val / 1 test · block-bootstrap 95% CI"]
        CM["<b>Compare inputs</b><br/>same parcels · paired differences vs baseline"]
        BL --> CF
        FM --> CF
        CF --> CM
    end

    FE --> BL
    C8 --> FM
    C16 --> FM
    W["<b>Pretrained weights</b><br/>Clay · Prithvi · TerraMind · AgriFM"] --> FM
    SF -. "folds" .-> CF
    CM --> R(["Does a foundation model beat<br/>the Sentinel-1/2 baseline?"])

    classDef part1 fill:#1F5F8B,color:#fff,stroke:#1F5F8B
    classDef data fill:#FFF4E6,stroke:#B4541A,color:#1C2733
    classDef part2 fill:#B4541A,color:#fff,stroke:#B4541A
    classDef ext fill:#F4F1EA,stroke:#8A919A,color:#1C2733
    class GT,CL,SF,TS part1
    class FE,C8,C16 data
    class BL,FM,CF,CM part2
    class EE,W ext
```

Setup, run order and the project folders: [Getting started](setup.md).
