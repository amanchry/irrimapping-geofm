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
| 2 | RQ1 | Do frozen GeoFMs separate irrigated from rainfed pixels better than classifiers on the Sentinel-2 time series and a network trained from scratch? | E01–E04 planned (Morocco first) | [E01](experiments/E01_s2_pixel_baseline.md)–[E04](experiments/E04_frozen_geofm.md) |
| 2 | RQ2 | Does fine-tuning (LoRA, full) beat the frozen encoder and training from scratch, and how does it depend on the number of labels? | E05 planned (Morocco first) | [E05](experiments/E05_prithvi_strategies.md) |
| 3, 5 | RQ3 | Is the best model consistent across years, and do the annual maps follow national statistics? | not started | – |
| 3, 5 | RQ4 | Do the conclusions hold in a second region? | not started (Utah after Morocco) | – |
| 4 | RQ5 | Does fusing several models improve on the best single model? | optional, not started | – |

## Finished pixel-wise experiments

_None yet in the two-class design._
