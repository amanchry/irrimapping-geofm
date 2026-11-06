# Heads and mapping: what GeoFM crop studies use

Literature search of 6 Nov 2026: which head is put on a GeoFM for crop classification, which neural baselines are
used, and what wall-to-wall mapping requires. Result: [decision D11](../decisions.md#d11-two-kinds-of-head-parcel-benchmark-and-mapping-model).

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

## References
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

