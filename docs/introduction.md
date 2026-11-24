# Introduction

!!! abstract "Research proposal"
    **[:material-file-pdf-box: Research proposal (PDF)](assets/research_proposal.pdf)**

## Why irrigated cropland matters

Irrigated agriculture covers only about **20%** of the world's cultivated land, yet it produces around **40%** of
global food and accounts for about **70%** of all freshwater withdrawals (Wu et al., 2023). With population growth,
climate change and increasing water scarcity, reliable information on where irrigation takes place, and how it
changes from year to year, is essential for water management, agricultural planning and the Sustainable Development
Goals on food (SDG 2) and water (SDG 6).

**Morocco** is a demanding and relevant case. Agriculture provides about 14% of its GDP, 39% of national employment
and 74% of rural employment, and the country has developed more than two million hectares of irrigated land across
seven large schemes. At the same time it is among the most water-stressed countries in the world: agriculture uses
over 80% of the available water, while rainfall has fallen by about 30% over three decades. National programmes
(the Green Morocco Plan 2008–2020 and the Green Generation strategy 2020–2030) expanded drip irrigation from
160,000 ha in 2008 to 585,000 ha in 2019 (Guemouria et al., 2023). Tracking whether this expansion meets its goals,
and where water use is most intensive, requires precise and up-to-date irrigated cropland maps.

Much of the rest of North Africa lacks the dense, field-verified irrigation data that exists for Morocco. Methods that
can be trained where labels are rich and transferred to agro-climatically similar, data-scarce regions would be a
significant step for agricultural monitoring across the region.

## State of the art

| Approach | Examples | Limitation |
| --- | --- | --- |
| Global irrigation products | GMIA (Siebert et al., 2005; ~9 km), GIAM (Thenkabail et al., 2009) | far too coarse for fields of a few hectares |
| 30 m global maps | LGRIP30 (Teluguntla et al., 2023; 86.5% overall accuracy), GMIE (Wu et al., 2023) | static snapshots or decade-maximum extents, no annual series; locally calibrated thresholds |
| Machine learning on Sentinel-1/2 time series | current state of the art for field-scale irrigation mapping | depends on hand-built spectral and temporal features; degrades when applied to new regions; hard to combine sensors (Yang et al., 2026) |

**Geospatial foundation models (GeoFMs)** offer a different route. They are large neural networks pretrained with
self-supervised learning on vast unlabelled satellite archives. They learn rich spatio-temporal representations
directly from the imagery, without hand-built features, can be adapted to a task with few labels, and may transfer
across regions and sensors (Tseng et al., 2023; Szwarcman et al., 2024). Adaptation ranges from a **frozen encoder**
with a light classification head, through **parameter-efficient fine-tuning** such as LoRA, to full fine-tuning.

## Research gap

1. **No annual irrigated cropland maps exist** for Morocco or other North African countries. Global products give
   static extents, not the year-by-year maps needed to monitor irrigation expansion and evaluate policy. This needs a
   model that is accurate *and* consistent from one year to the next.
2. **No geospatial foundation model has been evaluated for irrigated cropland mapping**, and none in Africa's diverse
   agro-hydrological landscapes, from semi-arid to sub-humid climates and from flood to drip and pivot irrigation.
3. **It is unknown which foundation model suits the task best**, how much fine-tuning adds to a frozen encoder, and
   whether a model trained in one region still works in a data-scarce region nearby.

## Objectives and research questions

**Main objective:** to compare the performance and transferability of geospatial foundation models for irrigated
cropland mapping.

| Objective | Research question |
| --- | --- |
| **1 · Compare models and adaptation strategies** in Morocco | **RQ1** What classification accuracy does each foundation model achieve for irrigated cropland mapping across Morocco's agricultural regions, and how does it compare with a conventional Sentinel-based classifier? |
| | **RQ2** To what extent does fine-tuning (LoRA or full fine-tuning) improve on a frozen encoder, compared with training from scratch? |
| **2 · Evaluate temporal and spatial transferability** of the best model | **RQ3** How temporally consistent is the best model across years, and does the annual map series for Morocco (2021–2025) reflect the irrigation trends in national statistics? |
| | **RQ4** When the Morocco-trained model is applied without retraining to another region, how much does accuracy degrade, and do its annual maps stay consistent? |
| **3 · Optional extension:** multi-model fusion | **RQ5** Does fusing the embeddings of several foundation models improve on the best single model? |

!!! note "Research questions as proposed"
    The questions above follow the research proposal; RQ2 is worded as it is now tested. Changes to the design
    since the proposal are recorded in [Decisions](decisions.md), in particular
    [D13](decisions.md#d13-utah-as-a-second-independent-study-area) (Utah as an independent second study area
    instead of a zero-shot transfer region, which changes RQ4),
    [D14](decisions.md#d14-sentinel-2-only-for-every-input) (Sentinel-2 only as model input) and
    [D16](decisions.md#d16-pixel-wise-classification-for-continuous-maps) (every 10 m pixel is classified, not
    every parcel). The current phases are described in the [Research plan](research_plan.md).


## Study area and data

**Morocco.** Five major agricultural regions that together span the country's diversity (Ouassanouan et al., 2026):

| Region | Climate | Irrigation and crops |
| --- | --- | --- |
| El Gharb | humid | surface-water irrigation; rice, vegetables |
| Tadla | semi-arid to sub-humid | large-scale irrigation; sugar beet, cereals |
| Doukkala | Mediterranean | mixed irrigated and rainfed systems |
| El Haouz | semi-arid | groundwater irrigation; olives, citrus |
| Souss | arid | advanced drip systems; high-value export crops |

**Second study area.** The proposal applied the best Morocco-trained model, without retraining, to **Tunisia**,
validated with the GMIA-NEXT reference points (Kebede et al., 2026), which include only 10 rainfed points. Instead,
**Utah (USA)** is studied as an independent second region with its own field-level labels (Water-Related Land Use,
Utah Division of Water Resources), and it also provides the temporal test: models trained on 2024 are tested on 2021
([decision D13](decisions.md#d13-utah-as-a-second-independent-study-area),
[Utah training data](phase1_data/utah.md)).

| Data | Use |
| --- | --- |
| **Field survey**, Ouassanouan et al. (2026): 8,826 georeferenced parcels with crop type, irrigation status and irrigation system (8,781 after cleaning: 7,585 irrigated, 1,196 rainfed) | training and validation in Morocco |
| **Sentinel-2** multispectral (10 m) and **Sentinel-1** SAR (VV, VH) time series | model input in the proposal; every model now uses Sentinel-2 only, Sentinel-1 remains an optional ablation ([D14](decisions.md#d14-sentinel-2-only-for-every-input)) |
| GMIE, LGRIP30, FAO WaPOR, FAO AQUASTAT | comparison of the maps with existing products and national statistics |

## Models compared

Six openly available GeoFMs are evaluated: **Prithvi-EO-2.0** (NASA, IBM), **Clay v1.5** (Clay Foundation),
**TerraMind** (IBM, ESA), **AgriFM** (Li et al., 2025), **Presto** (Tseng et al., 2023) and **TESSERA** (Feng et al.,
2025). The proposal named the first four; Presto and TESSERA, which encode the time series of a single pixel, were
added later. Together they represent the principal GeoFM families: masked-autoencoder Vision Transformers, a
multimodal generative model, an agriculture-specific spatio-temporal model and pixel time-series models. Every model
receives Sentinel-2 input only ([D14](decisions.md#d14-sentinel-2-only-for-every-input)). Architecture, pretraining, configuration and sources are given in
[Foundation models](foundation_models/index.md).

## Approach

```mermaid
flowchart LR
    A["<b>1 · Data preparation</b><br/>Sentinel-2 time series<br/>labelled pixels · spatial folds"] --> B["<b>2 · Model evaluation</b><br/>frozen · LoRA · full fine-tuning · from scratch<br/>vs Sentinel-2 baselines"]
    B --> C["<b>3 · Temporal and spatial transferability</b><br/>Utah 2024 → 2021<br/>Morocco and Utah"]
    B -.-> E["<b>4 · Multi-model fusion</b><br/><i>optional</i>"]
    C --> D["<b>5 · Annual maps</b><br/>Morocco 2021–2025<br/>temporal consistency"]
```

The phases are described in the [Research plan](research_plan.md).

## Expected outcomes

1. A comprehensive evaluation of six geospatial foundation models for irrigated cropland mapping in North Africa,
   against a strong Sentinel-based baseline under spatial validation.
2. A comparison of adaptation strategies for this task: frozen encoder, LoRA, full fine-tuning and training from
   scratch.
3. Annual irrigated cropland maps for Morocco at 10 m resolution for 2021–2025, checked against national statistics.
4. An assessment of the temporal consistency of foundation-model predictions across years.
5. An assessment of how far the best model transfers to a data-scarce region.
6. A reusable, ML-ready dataset of irrigated and rainfed parcels with spatial folds.

!!! info "Full proposal"
    Background, literature, methods, risks and the work plan in detail:
    **[Research proposal (PDF)](assets/research_proposal.pdf)**
