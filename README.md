# Irrigated Cropland Mapping with Geospatial Foundation Models

**📘 Documentation: [amanchry.github.io/irrimapping-geofm](https://amanchry.github.io/irrimapping-geofm/)**

Irrigated agriculture covers about a fifth of the world's cropland but produces around 40% of its food and accounts for most freshwater withdrawals. Knowing where and how much land is irrigated is therefore essential for water management and food security, especially in water-stressed regions such as North Africa. Yet most countries lack accurate, regularly updated irrigation maps, and existing global products are coarse or static. Geospatial foundation models, large models pretrained on vast archives of satellite imagery, promise accurate maps from few labels and better transfer to new regions. This research evaluates that promise for irrigated cropland mapping, using Morocco as a case study. It asks how accurately these models separate irrigated from rainfed cropland compared with conventional satellite-based methods, how much adapting them to the task helps, how consistent their maps are from year to year, and how well a model trained in Morocco transfers to a region with little ground truth.

| | |
| --- | --- |
| **Research design** | [introduction](https://amanchry.github.io/irrimapping-geofm/introduction/) · [proposal (PDF)](https://amanchry.github.io/irrimapping-geofm/assets/research_proposal.pdf) · [method in brief](https://amanchry.github.io/irrimapping-geofm/research_plan/) · [decisions](https://amanchry.github.io/irrimapping-geofm/decisions/) |
| **Methodology** | [dataset pipeline](https://amanchry.github.io/irrimapping-geofm/phase1_data/dataset/) · [models pipeline](https://amanchry.github.io/irrimapping-geofm/phase2_models/frozen_encoders/) · [evaluation protocol](https://amanchry.github.io/irrimapping-geofm/phase2_models/evaluation/) |
| **Experiments and results** | [experiments](https://amanchry.github.io/irrimapping-geofm/experiments/) · [results](https://amanchry.github.io/irrimapping-geofm/results/) · [weekly notes](https://amanchry.github.io/irrimapping-geofm/weekly/) · [progress log](https://amanchry.github.io/irrimapping-geofm/progress/) |
| **Getting started** | [setup, data, weights, run order](https://amanchry.github.io/irrimapping-geofm/setup/) |



## Quick start

```bash
conda env create -f environment.yml && conda activate condaml
python data_training_prep/01_prepare_labels_Morocco.py      # labels, folds, tile plan: data_training_prep/ 01-03
python data_download/01_plan_blocks.py                 # 10.24 km blocks over the region; then 02a-02c (Earth Engine)
python data_training_prep/04_cut_tiles.py                   # training tiles cut from the blocks
python pixelwise_run/04_make_pixel_samples.py          # then pixelwise_run/ 04-05 and the experiments
python experiments/E01_s2_pixel_baseline/run.py        # experiments E01-E04
python pixelwise_run/05_compare_experiments.py         # comparison of all finished runs
mkdocs serve --livereload -a 127.0.0.1:8001           # preview the documentation
```

Two pipelines, each with numbered scripts: `pixelwise_run/` (current) and `parcelwise_run/` (archive). Every
script has its settings at the top and a description of what it does. Details:
[getting started](https://amanchry.github.io/irrimapping-geofm/setup/).
