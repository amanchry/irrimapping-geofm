# IrrMap · Mandal et al. (2025)

!!! quote "Citation"
    Mandal et al. (2025). IrrMap: a large-scale comprehensive dataset for irrigation method mapping. *KDD 2025*.
    [arXiv:2505.08273](https://arxiv.org/abs/2505.08273) · data: [Hugging Face](https://huggingface.co/datasets/Nibir/IrrMap)

**In one line:** a large US dataset of image patches labelled with the irrigation **method** (flood, sprinkler,
drip), built from state agricultural databases.

## What they did
- **Labels** from state datasets: Arizona (verified irrigated lands), Utah (Water-Related Land Use), Washington (WSDA
  agricultural land use), Colorado (CDSS).
- **Imagery:** single Sentinel-2 scenes (cloud cover ≤ 5%) and Landsat, 11 Sentinel-2 bands, reprojected to
  EPSG:4326 and cut into **224 × 224 px** patches on a fixed grid.
- **Classes:** 0 none, 1 flood, 2 sprinkler, 3 drip; plus crop group and a cloud mask. **No rainfed class.**
- **Split:** random 80 / 20 per state and year; U-Net models.

## Our check of their code and splits (Nov 2026)
- Every acquisition date is a separate sample, so the same place appears many times. Almost every **test patch
  location also appears in training on another date** (Washington 2018: 31,056 of 31,064; all states: 423,226 of
  423,268). The test set is also used for evaluation after every epoch. Their scores are therefore not spatially
  independent.
- Minor code issues (missing `argparse` import, undefined `src_path`, a constant land-type band).

## What this research takes from it
- **Not usable as our dataset** (no rainfed class), but its label sources are: eastern Washington (WSDA) and Utah
  (WRLU) contain both irrigated and dryland fields — candidate transfer regions
  ([data sources](../data_source/index.md), [decision D10](../decisions.md#d10-transfer-region-under-review)).
- A concrete example of why spatial folds are needed ([protocol §2](../phase2_models/evaluation.md#2-spatial-splits)).
