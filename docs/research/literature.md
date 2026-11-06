# Literature notes

The papers the method is built on, each with what it did and what this thesis takes from it. The full reference
lists are in the [proposal](https://github.com/amanchry/irrimapping-geofm/blob/main/thesis_proposal/MScProposal_s3677133_AmanChaudhary.pdf) and at the end of the
[dataset](../dataset/index.md#12-references) and [models](../models/index.md#11-references) pages.

| Topic | Paper | Used for |
| --- | --- | --- |
| GeoFM benchmarking | [PANGAEA](#pangaea--marsocci-et-al-2024) | frozen-encoder evaluation, temporal aggregation (L-TAE) |
| Crop time-series benchmark | [PASTIS](#pastis--garnot--landrieu-2021) | fold rotation (3 / 1 / 1), tile design |
| Temporal attention | [L-TAE](#l-tae--garnot--landrieu-2020) | the L-TAE pipeline (E03) |
| US irrigation data | [IrrMap](#irrmap--mandal-et-al-2025) | candidate transfer data; what not to do in validation |
| Spatial validation | [Roberts 2017, Ploton 2020](#spatial-validation) | spatial folds instead of a random split |
| Comparing classifiers | [Foody 2009](#comparing-accuracies--foody-2009) | paired differences with confidence intervals |
| Ground truth | [Ouassanouan et al. 2026](#ground-truth--ouassanouan-et-al-2026) | the Moroccan field survey |
| Heads and mapping | [GeoFM heads, neural baselines](#heads-and-mapping-what-geofm-crop-studies-use) | parcel benchmark vs mapping models ([D11](decisions.md#d11-two-kinds-of-head-parcel-benchmark-and-mapping-model)) |

---

### PANGAEA · Marsocci et al. (2024)
*PANGAEA: A global and inclusive benchmark for geospatial foundation models.* https://arxiv.org/abs/2412.04204
- Benchmarks many GeoFMs with **frozen encoders** and the same decoder (UPerNet), against supervised baselines
  (UNet, ViT) trained from scratch.
- Multi-temporal tasks with single-date encoders: each date is encoded separately, then combined either by a
  **linear mapping** or by an **L-TAE**; L-TAE was generally better.
- GeoFMs **do not consistently beat** supervised baselines; they help most when labels are few. Adding SAR did not
  always help.
- **Taken here:** frozen evaluation as the main comparison; the L-TAE pipeline (E03); results that equal the
  baseline are a plausible outcome, not a failure.

### PASTIS · Garnot & Landrieu (2021)
*Panoptic segmentation of satellite image time series with convolutional temporal attention networks.* ICCV.
https://arxiv.org/abs/2107.07933
- Sentinel-2 time series of French parcels on 128 × 128 px patches; **5 folds**, used as 3 train / 1 validation /
  1 test in rotation.
- **Taken here:** the same rotation for every model ([D3](decisions.md#d3-validation-pastis-rotation-instead-of-nested-cv));
  non-overlapping 128 px tiles.

### L-TAE · Garnot & Landrieu (2020)
*Lightweight temporal self-attention for classifying satellite images time series.* https://arxiv.org/abs/2007.00586
- A light version of temporal self-attention: a learnt master query per head attends over the dates; few
  parameters, strong results on crop-type mapping.
- **Taken here:** the temporal model of the L-TAE pipeline, for the monthly embeddings and for the monthly S1/S2
  features.

### IrrMap · Mandal et al. (2025)
*IrrMap: a large-scale comprehensive dataset for irrigation method mapping.* KDD 2025.
https://arxiv.org/abs/2505.08273
- Irrigation **method** (flood, sprinkler, drip) from US state datasets on 224 px patches of single dates; no
  rainfed class; random 80 / 20 split per state.
- Our check of their split files: almost every test patch location also appears in training on another date
  (WA 2018: 31,056 of 31,064), so their scores are not spatially independent.
- **Taken here:** the label sources (WSDA, Utah WRLU) as possible transfer data
  ([D10](decisions.md#d10-transfer-region-under-review)); an example of why spatial folds matter.

### Spatial validation
- Roberts, D. R., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or
  phylogenetic structure. *Ecography*, 40, 913–929.
- Ploton, P., et al. (2020). Spatial validation reveals poor predictive performance of large-scale ecological
  mapping models. *Nature Communications*, 11, 4540.
- **Taken here:** blocks larger than the range of spatial dependence, whole blocks per fold
  ([dataset §5](../dataset/index.md#5-script-02--make_spatial_folds)).

### Comparing accuracies · Foody (2009)
*Classification accuracy comparison: hypothesis tests and the use of confidence intervals in evaluations of
difference, equivalence and non-inferiority.* Remote Sensing of Environment, 113, 1658–1663.
- **Taken here:** compare models by the confidence interval of their paired difference, not by their point scores.

### Ground truth · Ouassanouan et al. (2026)
*Crop and irrigation types ground-truth dataset for Moroccan agricultural regions.* Scientific Data, 13, 746.
https://doi.org/10.1038/s41597-026-06993-y
- About 10,000 parcels in El Gharb, Tadla, Doukkala, El Haouz and Souss, Dec 2023 – Jan 2025, with crop type,
  irrigation system and a photo per parcel → [data source](../data_source/index.md).

---

## Heads and mapping: what GeoFM crop studies use

Literature search of 6 Nov 2026: which head is put on a GeoFM for crop classification, which neural baselines are
used, and what wall-to-wall mapping requires. Result: [decision D11](decisions.md#d11-two-kinds-of-head-parcel-benchmark-and-mapping-model).

| Head | Used for | Papers |
| --- | --- | --- |
| Classical ML on frozen embeddings (LogReg, RF, GBDT, SVM, kNN) | per-pixel / per-parcel classification | Presto (Tseng et al. 2023); AlphaEarth irrigated vs rainfed (Yang et al. 2026); AlphaEarth cropland maps (Mughees et al. 2026); Galileo / CropFM + RF (Nedungadi et al. 2026) |
| Small neural head (linear, MLP) | per-pixel classification, fine-tuning | TESSERA, 2-layer MLP per pixel (Feng et al. 2025); Presto / AlphaEarth + RF or MLP (Tong & Wang 2025); WorldCereal, fine-tuned Presto (Butsko et al. 2025) |
| Segmentation decoder (UPerNet, UNet) | wall-to-wall maps from patch-based models | PANGAEA (Marsocci et al. 2024); Prithvi-EO-2.0 (Szwarcman et al. 2024) |

| Neural baseline (from scratch) | Input | Gives maps | Note |
| --- | --- | --- | --- |
| TempCNN (Pelletier et al. 2019) | pixel time series | yes, per pixel | 1D-CNN > RF for irrigated mapping (Bazzi et al. 2019); TempCNN-type models transfer better across regions than GeoFM embeddings (Tong & Wang 2025) |
| L-TAE (Garnot & Landrieu 2020) | pixel or parcel time series | yes | same module as the E03 pipeline |
| U-TAE (Garnot & Landrieu 2021) | image tiles | yes, 10 m | PASTIS baseline; counterpart of GeoFM + UPerNet |
| PSE-TAE (Garnot et al. 2020) | parcel pixel sets | no (needs polygons) | parcel-level only |
| LSTM, Transformer (BreizhCrops, Rußwurm et al. 2020) | pixel time series | yes | similar to TempCNN in benchmarks |

**Main points**
- Several 2025–26 studies find the head choice matters little on frozen embeddings (LogReg within 0.3 pp of a
  boosted ensemble, Mughees et al. 2026), but E01 shows RF is weaker on dense PCA embeddings, so the linear probe
  is the fairer headline for GeoFMs.
- GeoFM embeddings degrade under large geographic shifts (cross-continent) more than simple temporal CNNs
  (Tong & Wang 2025): relevant for RQ4.
- Fine-tuning helps in operational crop mapping (WorldCereal) but can collapse pretrained features without care
  (Nedungadi et al. 2026): relevant for RQ2.

**References**
- Bazzi, H., et al. (2019). Mapping irrigated areas using Sentinel-1 time series in Catalonia, Spain. *Remote Sensing*, 11(15), 1836.
- Butsko, C., Van Tricht, K., Tseng, G., Milli, G., Rolnick, D., Cartuyvels, R., Becker Reshef, I., Szantoi, Z., & Kerner, H. (2025). Deploying geospatial foundation models in the real world: lessons from WorldCereal. https://arxiv.org/abs/2508.00858
- Feng, Z., Atzberger, C., Jaffer, S., et al. (2025). TESSERA: temporal embeddings of surface spectra for Earth representation and analysis. https://arxiv.org/abs/2506.20380
- Mughees, M. A., Montefoschi, G., Chen, Z., & Brovelli, M. A. (2026). From foundation embeddings to cropland maps: label efficiency, temporal transferability and independent human validation. *arXiv preprint* 2609.17138.
- Nedungadi, V., Xiong, X., Rußwurm, M., & Athanasiadis, I. N. (2026). Foundation models meet agriculture: challenges beyond pretraining. *arXiv preprint* 2608.30392.
- Pelletier, C., Webb, G. I., & Petitjean, F. (2019). Temporal convolutional neural network for the classification of satellite image time series. *Remote Sensing*, 11(5), 523.
- Rußwurm, M., Pelletier, C., Zollner, M., Lefèvre, S., & Körner, M. (2020). BreizhCrops: a time series dataset for crop type mapping. *ISPRS Archives*, XLIII-B2-2020, 1545–1551.
- Szwarcman, D., et al. (2024). Prithvi-EO-2.0: a versatile multi-temporal foundation model for Earth observation applications. https://arxiv.org/abs/2412.02732
- Tong, X.-Y., & Wang, S. (2025). Invariant features for global crop type classification. https://arxiv.org/abs/2509.03497
- Tseng, G., Cartuyvels, R., Zvonkov, I., Purohit, M., Rolnick, D., & Kerner, H. (2023). Lightweight, pre-trained transformers for remote sensing timeseries (Presto). https://arxiv.org/abs/2304.14065
- Yang, L., Gao, Y., Zhao, X., Liang, N., Ma, R., Xi, S., Zhang, X., & Wang, R. (2026). Evaluating the performance of AlphaEarth Foundation embeddings for irrigated cropland mapping across regions and years. *Remote Sensing*, 18(7), 1065. https://www.mdpi.com/2072-4292/18/7/1065

