# Results

Results of the pixel-wise experiments, each with its principal finding and main figure. The list of all
experiments is on the [Experiments](experiments/index.md) page, all runs side by side on the
[comparison](experiments/comparison.md) page; the archived parcel-wise results are
[here](archive/parcel_wise_classification/results.md).

!!! info "Experiments redesigned (20 November 2026)"
    The experiments now classify the pixels of the surveyed parcels into **two classes, irrigated and rainfed**;
    the cropland mask and the not-cropland class are left to the mapping phase
    ([D19](decisions.md#d19-two-class-experiments-cropland-extent-in-the-mapping-phase)). All earlier pixel-wise results (three
    classes) were removed, and E01–E05 run again, **Morocco first**, then Utah.

## Where each research question stands

| Phase | RQ | Question (short) | Current answer | Evidence |
| --- | --- | --- | --- | --- |
| 2 | RQ1 | Do frozen GeoFMs separate irrigated from rainfed pixels better than classifiers on the Sentinel-2 time series and a network trained from scratch? | Morocco: frozen GeoFMs best (TerraMind 0.927, Clay 0.923, AgriFM 0.921, Prithvi 0.917); significantly better than U-TAE from scratch (0.900), not than the random forest (0.894) | [E01](experiments/E01_s2_pixel_baseline.md)–[E04](experiments/E04_frozen_geofm.md) |
| 2 | RQ2 | Does fine-tuning (LoRA, full) beat the frozen encoder and training from scratch, and how does it depend on the number of labels? | Morocco: no. Frozen, LoRA and full fine-tuning tie (0.90–0.92) at 10–100% of the parcels; pretraining adds 0.08–0.11 over training from scratch | [E05](experiments/E05_prithvi_strategies.md) |
| 3, 5 | RQ3 | Is the best model consistent across years, and do the annual maps follow national statistics? | not started | – |
| 3, 5 | RQ4 | Do the conclusions hold in a second region? | not started (Utah after Morocco) | – |
| 4 | RQ5 | Does fusing several models improve on the best single model? | optional, not started | – |

## Finished pixel-wise experiments

| Experiment | Region | Principal finding | Figure |
| --- | --- | --- | --- |
| [E01](experiments/E01_s2_pixel_baseline.md) · Sentinel-2 pixel baseline | Morocco | Random forest macro-F1 0.894 [0.834, 0.940], rainfed F1 0.810; gradient boosting identical. 23% of the rainfed pixels are taken for irrigated, most in 2024/25 and in Souss. SWIR (B12, B11) and the months February and July–August matter most. | [scores](experiments/figures/scores_e01_e02.png), [importance](experiments/figures/e01_importance.png) |
| [E02](experiments/E02_temporal_nn.md) · TempCNN | Morocco | Macro-F1 0.912 [0.848, 0.955], rainfed F1 0.845: +0.018 over the random forest, not significant (CI includes 0). Finds more rainfed pixels in 2024/25 (recall 0.86 vs 0.66) and Souss (0.68 vs 0.32). | [scores](experiments/figures/scores_e01_e02.png), [per zone](experiments/figures/groups_e01_e02.png) |
| [E03](experiments/E03_utae.md) · U-TAE from scratch | Morocco | Macro-F1 0.900 [0.845, 0.941]: on a par with the pixel models; better in fold 0 and Souss, worse in folds 1 and 4. | [scores](experiments/figures/scores_e03_e04.png) |
| [E04](experiments/E04_frozen_geofm.md) · frozen GeoFMs | Morocco | TerraMind 0.927, Clay 0.923, AgriFM 0.921, Prithvi 0.917 (significantly above U-TAE), TESSERA 0.912, Presto 0.903. Gains in rainfed recall in 2024/25 and Souss. No encoder significantly above the random forest. | [scores](experiments/figures/scores_e03_e04.png), [per zone](experiments/figures/groups_e03_e04.png) |
| [E05](experiments/E05_prithvi_strategies.md) · Prithvi strategies | Morocco | Frozen 0.920, LoRA 0.916, fine-tuned 0.911, from scratch 0.825 (100%). Pretrained strategies hardly change from 10% to 100% of the parcels; fine-tuning adds nothing over the frozen encoder at 1.3–1.6× the GPU time. | [label shares](experiments/figures/e05_label_fraction.png), [curves](experiments/figures/e05_curves.png) |
