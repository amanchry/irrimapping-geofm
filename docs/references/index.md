# Reference papers

Every paper cited in the [research proposal](../assets/research_proposal.pdf) or used while building the method,
grouped by topic. Papers marked **📄** have a summary page (what they did, key results, what this research takes
from them).

**Add a paper:** one row in the matching table (authors, year, title with link, venue, why it matters here). For an
important one, add a summary page in `docs/references/` from [`_template.md`](_template.md) and link it in the
*Summary* column and in the menu (`docs/SUMMARY.md`). Only references checked against the paper itself.

**Status:** ★ central to this research · ✓ cited / used · ○ background

---

## Geospatial foundation models

| | Paper | Venue | Why it matters here | Summary |
| --- | --- | --- | --- | --- |
| ★ | Szwarcman, D., et al. (2024). [Prithvi-EO-2.0: a versatile multi-temporal foundation model for Earth observation applications](https://arxiv.org/abs/2412.02732) | arXiv | model compared (time and location encoding) | |
| ★ | Clay Foundation (2024). [Clay Foundation Model v1.5](https://huggingface.co/made-with-clay/Clay) | model release | model compared (sensor-flexible MAE) | |
| ★ | Jakubik, J., et al. (2025). [TerraMind: large-scale generative multimodality for Earth observation](https://arxiv.org/abs/2504.11171) | ICCV 2025 | model compared (only one pretrained with SAR) | |
| ★ | Li, W., et al. (2025). [AgriFM: a multi-source temporal remote sensing foundation model for agriculture mapping](https://arxiv.org/abs/2505.21357) | arXiv | model compared (agriculture-specific) | |
| ★ | Brown, C. F., Kazmierski, M. R., Pasquarella, V. J., et al. (2025). [AlphaEarth Foundations: an embedding field model for accurate and efficient global mapping from sparse label data](https://arxiv.org/abs/2507.22291) | arXiv | 10 m annual embeddings; weights not released, so not in the comparison | |
| ✓ | Tseng, G., Cartuyvels, R., Zvonkov, I., Purohit, M., Rolnick, D., & Kerner, H. (2023). [Lightweight, pre-trained transformers for remote sensing timeseries (Presto)](https://arxiv.org/abs/2304.14065) | arXiv | frozen embeddings + RF / LogReg heads | |
| ✓ | Tseng, G., et al. (2025). [Galileo: learning global and local features in pretrained remote sensing models](https://arxiv.org/abs/2502.09356) | arXiv | kNN / linear probing of GeoFMs | |
| ✓ | Feng, Z., Atzberger, C., Jaffer, S., et al. (2025). [TESSERA: temporal embeddings of surface spectra for Earth representation and analysis](https://arxiv.org/abs/2506.20380) | arXiv | pixel embeddings + MLP head | |
| ○ | Cong, Y., et al. (2022). [SatMAE: pre-training transformers for temporal and multi-spectral satellite imagery](https://arxiv.org/abs/2207.08051) | NeurIPS 2022 | background (GeoFM landscape) | |
| ○ | Reed, C. J., et al. (2023). [Scale-MAE: a scale-aware masked autoencoder for multiscale geospatial representation learning](https://arxiv.org/abs/2212.14532) | ICCV 2023 | background (GeoFM landscape) | |

## Benchmarking and adapting GeoFMs

| | Paper | Venue | Why it matters here | Summary |
| --- | --- | --- | --- | --- |
| ★ | Marsocci, V., et al. (2024). [PANGAEA: a global and inclusive benchmark for geospatial foundation models](https://arxiv.org/abs/2412.04204) | arXiv | frozen-encoder protocol; L-TAE for time series; GeoFMs vs supervised baselines | [📄](pangaea.md) |
| ★ | Tong, X.-Y., & Wang, S. (2025). [Invariant features for global crop type classification](https://arxiv.org/abs/2509.03497) | arXiv | GeoFM embeddings vs temporal CNNs under geographic shift (RQ4) | [📄](tong-wang-2025.md) |
| ✓ | Butsko, C., Van Tricht, K., Tseng, G., et al. (2025). [Deploying geospatial foundation models in the real world: lessons from WorldCereal](https://arxiv.org/abs/2508.00858) | arXiv | operational GeoFM crop mapping; fine-tuned vs frozen | |
| ✓ | Marti-Escofet, F., Blumenstiel, B., Scheibenreif, L., Fraccaro, P., & Schindler, K. (2025). [Fine-tune smarter, not harder: parameter-efficient fine-tuning for geospatial foundation models](https://arxiv.org/abs/2504.17397) | arXiv | LoRA and other PEFT methods (RQ2) | |
| ✓ | van der Plas, T. L., et al. (2026). [Better together: evaluating the complementarity of Earth embedding models](https://arxiv.org/abs/2605.18667) | arXiv | embedding fusion (RQ5) | |
| ✓ | Tomotaki-Dawoud, K., Blumenstiel, B., Fraccaro, P., & Jakubik, J. (2025). [Fine-tuning the Prithvi foundation model for crop-type and health mapping in smallholder farms](https://doi.org/10.1016/j.procs.2025.10.507) | Procedia Computer Science | Prithvi fine-tuning for crops | |
| ✓ | Mughees, M. A., Montefoschi, G., Chen, Z., & Brovelli, M. A. (2026). [From foundation embeddings to cropland maps: label efficiency, temporal transferability and independent human validation](https://arxiv.org/abs/2609.17138) | arXiv preprint | classifier heads on embeddings matter little | |
| ✓ | Nedungadi, V., Xiong, X., Rußwurm, M., & Athanasiadis, I. N. (2026). [Foundation models meet agriculture: challenges beyond pretraining](https://arxiv.org/abs/2608.30392) | arXiv preprint | RF on frozen embeddings; risks of fine-tuning | |
| ✓ | Alain, G., & Bengio, Y. (2016). [Understanding intermediate layers using linear classifier probes](https://arxiv.org/abs/1610.01644) | arXiv | the linear probe | |

## Irrigation and cropland mapping

| | Paper | Venue | Why it matters here | Summary |
| --- | --- | --- | --- | --- |
| ★ | Yang, L., Gao, Y., Zhao, X., Liang, N., Ma, R., Xi, S., Zhang, X., & Wang, R. (2026). [Evaluating the performance of AlphaEarth Foundation embeddings for irrigated cropland mapping across regions and years](https://doi.org/10.3390/rs18071065) | Remote Sensing 18(7) | closest study: GeoFM embeddings for irrigated mapping, spatial + temporal transfer | [📄](yang-2026-alphaearth-irrigated.md) |
| ★ | Mandal et al. (2025). [IrrMap: a large-scale comprehensive dataset for irrigation method mapping](https://arxiv.org/abs/2505.08273) | KDD 2025 | US irrigation data; split leakage example | [📄](irrmap.md) |
| ★ | Ouassanouan, Y., Elfarkh, J., Grich, S., Liblab, A., & Chehbouni, A. (2026). [Crop and irrigation types ground-truth dataset for Moroccan agricultural regions](https://doi.org/10.1038/s41597-026-06993-y) | Scientific Data 13 | the labels ([data sources](../data_source/index.md)) | [📄](ouassanouan-2026-morocco-survey.md) |
| ✓ | Wu, B., et al. (2023). [Mapping global maximum irrigation extent at 30 m resolution using the irrigation performances under drought stress (GMIE)](https://doi.org/10.1016/j.gloenvcha.2023.102652) | Global Environmental Change 79 | irrigation statistics; comparison product | |
| ✓ | Teluguntla, P., et al. (2023). [Landsat-derived global rainfed and irrigated-cropland product 30 m V001 (LGRIP30)](https://doi.org/10.5067/COMMUNITY/LGRIP/LGRIP30.001) | NASA LP DAAC | comparison product | |
| ✓ | Siebert, S., et al. (2005). [Development and validation of the global map of irrigation areas (GMIA)](https://doi.org/10.5194/hess-9-535-2005) | HESS 9(5) | first global irrigation map | |
| ✓ | Thenkabail, P. S., et al. (2009). [Global irrigated area map (GIAM), derived from remote sensing](https://doi.org/10.1080/01431160902800928) | IJRS 30(14) | global irrigation map | |
| ✓ | Gao, Q., Zribi, M., Escorihuela, M. J., Baghdadi, N., & Quintana Seguí, P. (2018). [Irrigation mapping using Sentinel-1 time series at field scale](https://doi.org/10.3390/rs10091495) | Remote Sensing 10(9) | Sentinel-1 irrigation mapping; RBF SVM | |
| ✓ | Elwan, E., Le Page, M., Jarlan, L., et al. (2022). [Irrigation mapping on two contrasted climatic contexts using Sentinel-1 and Sentinel-2 data](https://doi.org/10.3390/w14050804) | Water 14(5) | Sentinel-1/2 irrigation mapping incl. Morocco; SVM | |
| ✓ | Kebede, E. A., et al. (2026). [GMIA-NEXT: next-generation global map of irrigated areas](https://doi.org/10.5281/zenodo.17627111) | Zenodo dataset | reference points (transfer test) | |
| ✓ | Van Tricht, K., et al. (2023). [WorldCereal: a dynamic open-source system for global-scale, seasonal, and reproducible crop and irrigation mapping](https://doi.org/10.5194/essd-15-5491-2023) | ESSD 15 | operational crop and irrigation mapping | |
| ✓ | Bazzi, H., et al. (2019). [Mapping irrigated areas using Sentinel-1 time series in Catalonia, Spain](https://doi.org/10.3390/rs11151836) | Remote Sensing 11(15) | 1D-CNN > RF for irrigated mapping | |
| ✓ | Guemouria, Y., Hammani, A., & Bouignane, A. (2023). [Opportunities and challenges of irrigation in Morocco, Spain, and India](https://doi.org/10.1002/wwp2.12148) | World Water Policy 9(4) | Moroccan irrigation context | |
| ○ | Chandrasekharan, K. M., Subasinghe, C., & Haileslassie, A. (2021). [Mapping irrigated and rainfed agriculture in Ethiopia (2015–2016)](https://doi.org/10.5337/2021.206) | IWMI Working Paper 196 | limits of global products in Africa | |
| ○ | Gumma, M. K., et al. (2011). [Mapping irrigated areas of Ghana using fusion of 30 m and 250 m resolution remote-sensing data](https://doi.org/10.3390/rs3040816) | Remote Sensing 3(4) | limits of coarse products | |
| ○ | Potapov, P., et al. (2021). [Global maps of cropland extent and change show accelerated cropland expansion in the twenty-first century](https://doi.org/10.1038/s43016-021-00429-z) | Nature Food 2 | cropland mask | |
| ○ | Zanaga, D., et al. (2022). [ESA WorldCover 10 m 2021 v200](https://doi.org/10.5281/zenodo.7254221) | Zenodo dataset | cropland check | |
| ○ | Kerner, H. R., Nakalembe, C., & Becker-Reshef, I. (2020). [Field-level crop type classification with k nearest neighbors: a baseline for a new Kenya smallholder dataset](https://arxiv.org/abs/2004.03023) | arXiv | data scarcity in Africa | |

## Crop time-series deep learning (neural baselines)

| | Paper | Venue | Why it matters here | Summary |
| --- | --- | --- | --- | --- |
| ★ | Garnot, V. S. F., & Landrieu, L. (2021). [Panoptic segmentation of satellite image time series with convolutional temporal attention networks (PASTIS, U-TAE)](https://arxiv.org/abs/2107.07933) | ICCV 2021 | fold rotation (3 / 1 / 1); tile design; U-TAE baseline | [📄](pastis-utae.md) |
| ★ | Garnot, V. S. F., & Landrieu, L. (2020). [Lightweight temporal self-attention for classifying satellite images time series (L-TAE)](https://arxiv.org/abs/2007.00586) | AALTD, ECML-PKDD | the L-TAE pipeline (E03) | [📄](ltae.md) |
| ✓ | Pelletier, C., Webb, G. I., & Petitjean, F. (2019). [Temporal convolutional neural network for the classification of satellite image time series (TempCNN)](https://doi.org/10.3390/rs11050523) | Remote Sensing 11(5) | pixel-level neural baseline | |
| ✓ | Garnot, V. S. F., Landrieu, L., Giordano, S., & Chehata, N. (2020). [Satellite image time series classification with pixel-set encoders and temporal self-attention (PSE-TAE)](https://arxiv.org/abs/1911.07757) | CVPR 2020 | parcel-level neural baseline | |
| ✓ | Rußwurm, M., Pelletier, C., Zollner, M., Lefèvre, S., & Körner, M. (2020). [BreizhCrops: a time series dataset for crop type mapping](https://arxiv.org/abs/1905.11893) | ISPRS Archives | benchmark of LSTM / Transformer / TempCNN | |
| ○ | Sykas, D., Sdraka, M., Zografakis, D., & Papoutsis, I. (2022). A Sentinel-2 multiyear, multicountry benchmark dataset for crop classification and segmentation with deep learning (Sen4AgriNet) | IEEE JSTARS 15 | tile design | |
| ○ | Kerner, H., et al. (2025). Fields of The World: a machine learning benchmark dataset for global agricultural field boundary segmentation | AAAI 2025 | tile design | |

## Validation, statistics and data processing

| | Paper | Venue | Why it matters here | Summary |
| --- | --- | --- | --- | --- |
| ★ | Roberts, D. R., et al. (2017). [Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure](https://doi.org/10.1111/ecog.02881) | Ecography 40 | spatial blocks instead of random splits | |
| ★ | Ploton, P., et al. (2020). [Spatial validation reveals poor predictive performance of large-scale ecological mapping models](https://doi.org/10.1038/s41467-020-18321-y) | Nature Communications 11 | why random CV overstates accuracy | |
| ✓ | Kattenborn, T., et al. (2022). [Spatially autocorrelated training and validation samples inflate performance assessment of convolutional neural networks](https://doi.org/10.1016/j.ophoto.2022.100018) | ISPRS Open J. Photogramm. Remote Sens. 5 | same for deep models | |
| ✓ | Foody, G. M. (2009). [Classification accuracy comparison: hypothesis tests and the use of confidence intervals](https://doi.org/10.1016/j.rse.2009.03.014) | Remote Sensing of Environment 113 | paired differences with CIs | |
| ✓ | Holm, S. (1979). A simple sequentially rejective multiple test procedure | Scandinavian Journal of Statistics 6 | multiple-test correction | |
| ✓ | Varma, S., & Simon, R. (2006). [Bias in error estimation when using cross-validation for model selection](https://doi.org/10.1186/1471-2105-7-91) | BMC Bioinformatics 7 | nested cross-validation (E01) | |
| ✓ | Valavi, R., Elith, J., Lahoz-Monfort, J. J., & Guillera-Arroita, G. (2019). [blockCV: an R package for generating spatially or environmentally separated folds for k-fold cross-validation of species distribution models](https://doi.org/10.1111/2041-210X.13107) | Methods Ecol. Evol. 10(2) | same principle as the spatial folds | |
| ✓ | Schratz, P., Muenchow, J., Iturritxa, E., Richter, J., & Brenning, A. (2019). [Hyperparameter tuning and performance assessment of statistical and machine-learning algorithms using spatial data](https://doi.org/10.1016/j.ecolmodel.2019.06.002) | Ecol. Model. 406 | spatial folds for tuning as well as testing | |
| ✓ | Karasiak, N., Dejoux, J.-F., Monteil, C., & Sheeren, D. (2022). [Spatial dependence between training and test sets: another pitfall of classification accuracy assessment in remote sensing](https://doi.org/10.1007/s10994-021-05972-1) | Machine Learning 111 | random CV overstates accuracy in remote sensing | |
| ✓ | Orynbaikyzy, A., Gessner, U., & Conrad, C. (2022). [Spatial transferability of random forest models for crop type classification using Sentinel-1 and Sentinel-2](https://doi.org/10.3390/rs14061493) | Remote Sensing 14(6) | closest validation design: held-out regions, inner spatial CV | |
| ✓ | Künsch, H. R. (1989). [The jackknife and the bootstrap for general stationary observations](https://doi.org/10.1214/aos/1176347265) | Ann. Stat. 17(3) | block bootstrap of the confidence intervals | |
| ✓ | Pontius, R. G., & Millones, M. (2011). [Death to Kappa](https://doi.org/10.1080/01431161.2011.552923) | IJRS 32(15) | no kappa coefficient | |
| ✓ | Breiman, L. (2001). [Random forests](https://doi.org/10.1023/A:1010933404324) | Machine Learning 45 | baseline classifier | |
| ✓ | Pasquarella, V. J., Brown, C. F., Czerwinski, W., & Rucklidge, W. J. (2023). Comprehensive quality assessment of optical satellite imagery using weakly supervised video learning (Cloud Score+) | CVPR Workshops 2023 | cloud mask | |

## Topic reviews

- [Heads and mapping: what GeoFM crop studies use](heads-and-mapping.md) — classifier heads on frozen embeddings,
  neural baselines, and what wall-to-wall mapping requires (basis of
  [decision D11](../decisions.md#d11-two-kinds-of-head-parcel-benchmark-and-mapping-model)).
