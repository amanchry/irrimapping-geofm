# RQ1 · Experiment plan

This page is the complete plan of the frozen-encoder evaluation of Phase 2 · Model evaluation (RQ1): which experiments are run, what each one
answers, and every setting. It is agreed before the runs start; an experiment page in
[Experiments](../experiments/index.md) is created from it when the experiment is implemented. Rules shared by all
experiments (splits, metrics, fairness) are in the [evaluation protocol](evaluation.md).

**Contents:** [1 Decisions](#1-decisions-this-plan-rests-on) · [2 Shared settings](#2-settings-shared-by-every-experiment) ·
[3 Inputs](#3-inputs) · [4 Design](#4-design-what-is-varied) · [5 Main experiments](#5-main-experiments) ·
[6 Label efficiency](#6-e06--label-efficiency) · [7 Ablations](#7-ablations) · [8 Analyses](#8-analyses-on-every-run) ·
[9 Results pages](#9-results-pages) · [10 Order of work](#10-order-of-work) · [11 Size](#11-number-of-runs) ·
[12 Open points](#12-open-points)

---

## 1. Decisions this plan rests on

| Decision | Choice | Reason |
| --- | --- | --- |
| Input data | **Sentinel-2 only** for every foundation model and every baseline | same optical data for all; Morocco and Utah treated alike; Sentinel-1 coverage of Utah changed between years ([D14](../decisions.md#d14-sentinel-2-only-for-every-input)). Sentinel-1 is an ablation for later ([A5](#7-ablations)). |
| Model inputs | each foundation model receives only the bands it was pretrained on; no fusion designed by us | the comparison tests the pretrained models, not a fusion scheme |
| Classifier heads | three classical classifiers (**logistic regression, random forest, SVM**) and one neural head (**MLP**) | linear probe = standard test of frozen features; RF = standard remote-sensing classifier; SVM = strongest in E01; MLP = trainable head as in GeoFM benchmarks. Gradient boosting is dropped: in E01 it was never the best and always within the CI of SVM ([D6](../decisions.md#d6-four-classifiers-instead-of-one-random-forest)). |
| Temporal handling | **PCA** (concatenated months) and **L-TAE** (learned attention over the months) | [D8](../decisions.md#d8-two-temporal-aggregation-pipelines); L-TAE as in PANGAEA (Marsocci et al. 2024) |
| Neural baseline from scratch | **PSE-TAE** on the parcel pixel sets | built for parcel classification from satellite time series (Garnot et al. 2020) |
| Most important extra | **label efficiency** | the main claim for foundation models is that they need fewer labels |
| Regions | **Morocco** and **Utah 2023/24**, same protocol, analysed separately | [D13](../decisions.md#d13-utah-as-a-second-independent-study-area): two independent study areas |

---

## 2. Settings shared by every experiment

| Setting | Value |
| --- | --- |
| Task | irrigated (1) vs rainfed (2), one prediction per parcel |
| Morocco data | views `views/` (2023/24 + 2024/25 pooled), 8,775 parcels scored (7,579 irrigated, 1,196 rainfed) |
| Utah data | views `views_2023-24/` (WRLU 2024 labels), 6,011 parcels with pixels |
| Time | 12 monthly composites of the agricultural year (Morocco Sep–Aug, Utah Oct–Sep) |
| Splits | 5 spatial folds of 20 km blocks; **PASTIS rotation**: per round 3 folds train, the next fold validates (settings, early stopping), 1 fold tests |
| Tuning | grids below, chosen on the validation fold by macro-F1; same grid for every input |
| Fitted steps | imputation, standardisation, PCA, tuning and early stopping use the training folds only |
| Neural networks | 3 seeds (0, 1, 2) per run; mean ± SD over seeds; the out-of-fold prediction of a parcel is the mean probability of the 3 seeds |
| Metrics | macro-F1 (headline), F1 / precision / recall per class, balanced and overall accuracy, PR-AUC (rainfed), Brier, ECE |
| Uncertainty | block bootstrap, 1,000 draws, same draws for every run; paired differences; Holm over all comparisons of an experiment |
| Comparison set | parcels present in every run of the region |

---

## 3. Inputs

All five inputs are cut from the same tile store. Sentinel-2 only.

| Input | Data per parcel | Bands | Patch | Per parcel | Months | Script |
| --- | --- | --- | --- | --- | --- | --- |
| **S2 baseline** | median of the parcel's inner pixels | 12 S2 bands + NDVI, NDWI, EVI (15 variables) | parcel | 15 × 12 = 180 features | 12 | `03` (view A) |
| **Clay v1.5** | 64 px chip, p8 grid | 10 S2 bands | 8 px (80 m) | 12 × 1,024 | 12 | `02a` |
| **Prithvi-EO-2.0** | 64 px chip, p16 grid | 6 S2 bands (HLS set) | 16 px (160 m) | 12 × 1,024 | 12 | `02b` |
| **TerraMind** | 64 px chip, p16 grid | 12 S2 bands | 16 px (160 m) | 12 × 768 | 12 | `02c` |
| **AgriFM** | 64 px chip, p16 grid | 10 S2 bands | 4 px patch, stride 32 px (320 m) | 1 × 1,024 (whole year) | 1 | `02d` |
| **Pixel sets** (PSE-TAE only) | all inner pixels of the parcel | 10 S2 bands | pixel | N pixels × 12 × 10 | 12 | `03b` (view B) |

Patch embeddings are pooled over the parcel with the patch-share weights ([models pipeline §5](frozen_encoders.md#5-scripts-02a02d--foundation-model-embeddings)).
Clay and Prithvi also receive the parcel location and the date, as in their pretraining; [A4](#7-ablations) tests
whether the location acts as a shortcut.

---

## 4. Design: what is varied

Three things are varied: the **input**, the **temporal handling** and the **head**. Not every combination exists:
L-TAE is a neural network trained with its own loss, so it cannot feed a random forest, and AgriFM gives one vector
per year, so it has no months for L-TAE.

```mermaid
flowchart LR
    IN["<b>Input</b><br/>S2 baseline · Clay · Prithvi<br/>TerraMind · AgriFM"] --> PCA["<b>PCA</b><br/>12 months concatenated<br/>→ PCA 128"]
    IN --> LT["<b>L-TAE</b><br/>attention over<br/>the 12 months"]
    PCA --> CL["<b>Classical</b><br/>LogReg · RF · SVM"]
    PCA --> MLP1["<b>MLP</b>"]
    LT --> MLP2["<b>MLP</b><br/>(trained with L-TAE)"]
    PX["<b>Pixel sets</b><br/>S2, from scratch"] --> PS["<b>PSE-TAE</b>"]
    CL --> E02(["E02"])
    MLP1 --> E05(["E05"])
    MLP2 --> E03(["E03"])
    PS --> E04(["E04"])
```

| Cell | Temporal handling | Head | S2 baseline | Clay | Prithvi | TerraMind | AgriFM | Experiment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a | PCA | LogReg · RF · SVM | ✓ (no PCA) | ✓ | ✓ | ✓ | ✓ | **E02** |
| b | PCA | MLP | ✓ (no PCA) | ✓ | ✓ | ✓ | ✓ | **E05** |
| c | L-TAE | MLP | ✓ | ✓ | ✓ | ✓ | – | **E03** |
| d | from scratch | PSE-TAE | pixel sets | – | – | – | – | **E04** |

**Which comparison answers which question**

| Question | Comparison | Same in both |
| --- | --- | --- |
| Q1 · Do frozen foundation-model embeddings beat Sentinel-2 features? | each GeoFM vs the S2 baseline, within E02, E05 and E03 | head, temporal handling, parcels |
| Q2 · Does the head matter? | E02 vs E05 (classical vs MLP) | input, PCA |
| Q3 · Does learning which months matter help? | E05 vs E03 (PCA vs L-TAE) | input, MLP head |
| Q4 · Is a model trained from scratch on our labels enough? | E04 vs the best GeoFM setting and the best baseline setting | parcels, folds |
| Q5 · Do foundation models need fewer labels? | E06 learning curves | everything except the training size |
| Q6 · Do the conclusions hold in a second region? | every question in Morocco and in Utah | protocol |

The S2 baseline in E03 (L-TAE on the 15 monthly variables) is the key control for Q1: if L-TAE helps the GeoFMs, the
same L-TAE must be given to the raw features, otherwise the gain could come from L-TAE rather than from the
embeddings.

---

## 5. Main experiments

### E02 · PCA + classical classifiers

| | |
| --- | --- |
| **Question** | Q1 with the classical heads; the main table of RQ1 |
| **Inputs** | S2 baseline, Clay, Prithvi, TerraMind, AgriFM |
| **Temporal handling** | months with no clear pixel filled with the training-fold median → 12 months concatenated → standardised → PCA 128 (fitted on the training folds). AgriFM: 1 × 1,024 → PCA 128. Baseline: 180 features, gaps filled by linear interpolation in time per parcel, standardised, no PCA. |
| **Classifiers and grids** | `logreg` C ∈ {0.01, 0.1, 1, 10}, balanced classes · `rf` 500 trees, max_features ∈ {√p, 0.3}, min_samples_leaf ∈ {1, 5}, class_weight ∈ {none, balanced_subsample} · `svm` RBF, C ∈ {1, 10}, γ ∈ {scale, 0.01}, balanced classes |
| **Headline** | logistic regression (linear probe) for the GeoFMs; random forest for the baseline, as in the protocol; all three reported |
| **Runs** | 5 inputs × 2 regions = 10 runs (3 classifiers each) |
| **Script** | `models_run/03_train_evaluate.py` (exists; HGB to be removed, region setting to be added) |

### E05 · PCA + MLP

| | |
| --- | --- |
| **Question** | Q2: does a trainable neural head get more out of the same PCA input than the classical heads? |
| **Inputs** | the same 128 PCA components (baseline: the 180 standardised features) as E02 |
| **Head** | MLP 128 → 256 → 128 → 2, ReLU, batch normalisation, dropout 0.2 (same hidden sizes as the E03 head) |
| **Training** | AdamW, learning rate 1e-3, weight decay 1e-4, batch 128, class-weighted cross-entropy, up to 100 epochs, early stopping on validation macro-F1 (patience 15), 3 seeds |
| **Tuning** | none beyond early stopping (same budget for every input) |
| **Runs** | 5 inputs × 2 regions × 3 seeds |
| **Script** | `models_run/03c_train_ltae.py` (to be written), mode `head: mlp` on the PCA input |

### E03 · L-TAE + MLP

| | |
| --- | --- |
| **Question** | Q3: does attention over the months classify better than concatenation + PCA? |
| **Inputs** | monthly sequences: S2 baseline (12 × 15 variables), Clay (12 × 1,024), Prithvi (12 × 1,024), TerraMind (12 × 768). AgriFM excluded (one vector per year). |
| **Pre-processing** | per-channel standardisation with training-fold statistics; months without a clear pixel are masked in the attention, not filled |
| **Model** | L-TAE (Garnot & Landrieu 2020) with the paper's default size: 16 heads, d_k 8, d_model 256, output MLP 256-128; then a linear layer with 2 outputs; the month position (1–12) as positional encoding |
| **Training** | as E05 (AdamW 1e-3, weight decay 1e-4, batch 128, class-weighted cross-entropy, ≤ 100 epochs, patience 15, 3 seeds) |
| **Runs** | 4 inputs × 2 regions × 3 seeds |
| **Script** | `models_run/03c_train_ltae.py` (to be written), mode `head: ltae` |
| **Compared with** | the E05 run of the same input (same head, PCA instead of L-TAE) and the E02 runs |

### E04 · PSE-TAE from scratch

| | |
| --- | --- |
| **Question** | Q4: does a temporal deep model trained only on our labelled parcels match the frozen foundation models? |
| **Input** | pixel sets: inner pixels of the parcel (edge pixels within 10 m left out; parcels with < 4 inner pixels keep all), 10 S2 bands × 12 months |
| **Model** | PSE-TAE (Garnot et al. 2020): 64 pixels sampled per parcel and month; at prediction the mean of 5 random pixel draws |
| **Training** | Adam 1e-3, weight decay 1e-4, batch 128, ≤ 100 epochs, early stopping on validation macro-F1 (patience 15), 3 seeds |
| **Runs** | `morocco_s2`, `utah_s2` × 3 seeds |
| **Script** | `models_run/03b_train_psetae.py` (exists; seeds loop to be added) |

---

## 6. E06 · Label efficiency

| | |
| --- | --- |
| **Question** | Q5: how much training data does each input need? Foundation models are expected to gain most when labels are few. |
| **Training size** | 10, 25, 50 and 100% of the **training blocks** of each round. Validation and test folds stay complete, so only the amount of training data changes. |
| **Sampling** | whole blocks, drawn separately from blocks with rainfed parcels and from irrigated-only blocks, so every subset keeps rainfed parcels (in Morocco most rainfed parcels lie in a few Doukkala blocks). A draw with fewer than 20 rainfed parcels is redrawn. |
| **Repeats** | 3 draws per fraction; the same draws for every input (paired) |
| **Inputs and heads** | every input with the linear probe (`logreg`) and SVM; the best neural setting of each input from E03/E05; PSE-TAE for the baseline |
| **Result** | learning curves (macro-F1 and rainfed F1 vs training fraction), mean over draws with block-bootstrap CI; the fraction at which each GeoFM reaches the full-data score of the baseline |
| **Runs** | 3 fractions below 100% × 3 draws × (5 inputs × 3 heads + PSE-TAE) × 2 regions; 100% comes from E02–E05 |
| **Script** | a `TRAIN_FRACTION` and `FRACTION_DRAW` setting in scripts 03, 03b and 03c |

---

## 7. Ablations

Short, targeted runs; each changes one setting of a main experiment.

| ID | Question | Change | On | Priority |
| --- | --- | --- | --- | --- |
| **A1** | Does the yearly profile matter at all? | average of the 12 monthly embeddings instead of concat + PCA (`FEATURE_MODE: mean`) | E02, linear probe + SVM | high (cheap) |
| **A2** | Is the PCA size a fragile choice? | PCA 32 / 64 / 128 / 256 | E02, linear probe | medium |
| **A3** | How many months are needed? | 1 peak month / 4 seasonal months / 12 months | best setting of each input | optional |
| **A4** | Do Clay and Prithvi use the location as a shortcut? | location input switched off (Prithvi `USE_COORDS = False`, Clay location set to zero); re-embedding needed | E02, Morocco (rainfed parcels concentrated in Doukkala) | high |
| **A5** | Does Sentinel-1 add information? | TerraMind with Sentinel-1 (`terramind-s1s2`), baseline with radar features (`S1S2_VARIABLES: all`), PSE-TAE `morocco_s2s1` | E02, E03, E04, Morocco only | later |

E01 (nested spatial CV, done before D14 with Sentinel-1 in the baseline and in TerraMind) stays as the check of the
validation design.

---

## 8. Analyses on every run

From the out-of-fold predictions; no new training.

| Analysis | Why | Script |
| --- | --- | --- |
| Paired differences vs the baseline, Holm-adjusted | main test of Q1 | `04_compare_models.py` (Holm fix pending) |
| Results by zone, season, parcel size and patch purity | region shortcut (Morocco rainfed in Doukkala); small parcels inside 160 m patches | `05_breakdown_oof.py` |
| Results by crop group (Utah: rainfed alfalfa vs grain) | where each input fails | `05_breakdown_oof.py` (extension) |
| Random vs spatial CV | how much random CV overstates accuracy | `03_train_evaluate.py` |
| Calibration (Brier, ECE) | usable probabilities for maps | `03`, `03b`, `03c` |
| Month importance (permutation, L-TAE attention) | which part of the season separates the classes | `03`, `03c` |
| Compute cost: parameters, embedding time, training time | practical value | run logs |

---

## 9. Results pages

Every experiment gets its own pages in [Experiments](../experiments/index.md), created when it is implemented and
filled after each run:

- `index.md` (setup): question, what changes against the previous experiment, all settings, how to run.
- `report.md` (results): the answer in one or two sentences first, then the generated tables and figures, then the
  interpretation, caveats and next steps ([protocol §9](evaluation.md#9-reporting-standard)).

Tables and figures are written by `experiments/make_report.py` from the results, so a page always matches its run.
With two regions, every table and figure appears once for Morocco and once for Utah.

| Experiment | Tables | Figures |
| --- | --- | --- |
| **every experiment** | metrics with 95% CI per input × head; differences vs the baseline (Δ macro-F1, CI, Holm) ; per-fold macro-F1 | macro-F1 with CI per input; confusion matrices; random vs spatial CV; results by zone, season, parcel size and purity |
| E02 · PCA + classical | the three classifiers side by side | PCA explained variance per input; month importance (permutation) |
| E05 · PCA + MLP | MLP vs the E02 classifiers on the same input; mean ± SD over seeds | training curves (loss, validation macro-F1) |
| E03 · L-TAE + MLP | L-TAE vs PCA (E05) on the same input | attention weights per month and class; training curves |
| E04 · PSE-TAE | PSE-TAE vs the best GeoFM and the best baseline setting | training curves |
| E06 · Label efficiency | macro-F1 and rainfed F1 per training fraction; fraction at which each GeoFM reaches the full-data baseline | learning curves per input (one panel per head) |
| Ablations | each ablation vs the setting it changes | one comparison figure each |

The [Results](../results.md) page collects the one-line answer of every experiment; the RQ1 synthesis (step 6
below) compares the experiments and the two regions.

## 10. Order of work

One experiment at a time: implement, check on a small run, run in both regions, write the report, then the next.

| Step | Work | Needs |
| --- | --- | --- |
| 0a | ✅ AgriFM normalisation: chip statistics of each region ([D15](../decisions.md#d15-agrifm-normalised-with-the-statistics-of-the-regions-chips)); config statistics as sensitivity run | – |
| 0b | Code: `REGION` / `VIEWS_NAME` in scripts 02a–02d ✅; remove HGB from script 03 and add the region to scripts 03, 04, 05 (before E02) | – |
| 0c | Embeddings, Sentinel-2 only: **Morocco** TerraMind (the old file had Sentinel-1) and AgriFM (new normalisation) again; Clay and Prithvi reused. **Utah** all four from `views_2023-24/` (cluster GPU) | 0a, 0b |
| 1 | **E02** Morocco + Utah; report page with tables and figures (`make_report.py` extended to two regions) | 0c |
| 2 | Write `03c_train_ltae.py` (heads `mlp` and `ltae`); **E05**, then **E03**; reports | 1 |
| 3 | **E04** PSE-TAE (3 seeds) | – (can run any time on the cluster) |
| 4 | **E06** label efficiency | 1–3 (best neural setting) |
| 5 | Ablations **A1**, **A4**, **A2** (A3, A5 if time allows) | 1 |
| 6 | RQ1 synthesis over both regions | 1–5 |

---

## 11. Number of runs

| Experiment | Runs per region | Both regions | Hardware |
| --- | --- | --- | --- |
| E02 | 5 (× 3 classifiers) | 10 | CPU, minutes each |
| E05 | 5 × 3 seeds | 30 | CPU or GPU, minutes |
| E03 | 4 × 3 seeds | 24 | GPU |
| E04 | 1 × 3 seeds | 6 | GPU |
| E06 | 3 fractions × 3 draws × 16 settings | 288 | mostly CPU |
| Ablations A1, A2, A4 | ~15 | ~25 | CPU (+ GPU for A4 embeddings) |

E06 is the largest; it reuses the code of E02–E05 and only adds the training-block subsampling.

---

## 12. Open points

- Headline classifier: the protocol keeps random forest for the baseline and the linear probe for the GeoFMs; with
  SVM the strongest classifier in E01, all three are always reported.
- Prithvi-EO-2.0 is pretrained on 30 m HLS and applied to 10 m Sentinel-2 (stated as a limitation).
- Fine-tuning (LoRA, RQ2), the 2024 → 2021 temporal test in Utah (RQ3) and the mapping models follow after RQ1.

## References

- Garnot, V. S. F., Landrieu, L., Giordano, S., & Chehata, N. (2020). Satellite image time series classification with pixel-set encoders and temporal self-attention. *CVPR*. https://arxiv.org/abs/1911.07757
- Garnot, V. S. F., & Landrieu, L. (2020). Lightweight temporal self-attention for classifying satellite images time series. *AALTD workshop, ECML-PKDD*. https://arxiv.org/abs/2007.00586
- Garnot, V. S. F., & Landrieu, L. (2021). Panoptic segmentation of satellite image time series with convolutional temporal attention networks. *ICCV*. https://arxiv.org/abs/2107.07933
- Marsocci, V., et al. (2024). PANGAEA: A global and inclusive benchmark for geospatial foundation models. https://arxiv.org/abs/2412.04204
