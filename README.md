# Irrigated Cropland Mapping with Geospatial Foundation Models

**📘 Documentation: [amanchry.github.io/irrimapping-geofm](https://amanchry.github.io/irrimapping-geofm/)**

Irrigated agriculture covers about a fifth of the world's cropland but produces around 40% of its food and accounts for most freshwater withdrawals. Knowing where and how much land is irrigated is therefore essential for water management and food security, especially in water-stressed regions such as North Africa. Yet most countries lack accurate, regularly updated irrigation maps, and existing global products are coarse or static. Geospatial foundation models, large models pretrained on vast archives of satellite imagery, promise accurate maps from few labels and better transfer to new regions. This thesis evaluates that promise for irrigated cropland mapping, using Morocco as a case study. It asks how accurately these models separate irrigated from rainfed cropland compared with conventional satellite-based methods, how much adapting them to the task helps, how consistent their maps are from year to year, and how well a model trained in Morocco transfers to a region with little ground truth.

| | |
| --- | --- |
| **Research design** | [thesis design](https://amanchry.github.io/irrimapping-geofm/research/thesis_design/) · [method in brief](https://amanchry.github.io/irrimapping-geofm/research/overview/) · [decisions](https://amanchry.github.io/irrimapping-geofm/research/decisions/) |
| **Methodology** | [dataset pipeline](https://amanchry.github.io/irrimapping-geofm/dataset/) · [models pipeline](https://amanchry.github.io/irrimapping-geofm/models/) · [evaluation protocol](https://amanchry.github.io/irrimapping-geofm/research/protocol/) |
| **Experiments and results** | [experiments](https://amanchry.github.io/irrimapping-geofm/experiments/) · [results](https://amanchry.github.io/irrimapping-geofm/results/) · [weekly notes](https://amanchry.github.io/irrimapping-geofm/weekly/) · [progress log](https://amanchry.github.io/irrimapping-geofm/progress/) |
| **Getting started** | [setup, data, weights, run order](https://amanchry.github.io/irrimapping-geofm/setup/) |



## Quick start

```bash
conda env create -f environment.yml && conda activate condaml
python ml_dataset_prep/01_prepare_labels.py      # … 02, 03, 04: the dataset
python models_run/02a_embed_clay.py              # … 02b–02d: foundation-model embeddings
python experiments/run_experiment.py             # an experiment from its config (EXPERIMENT at the top)
mkdocs serve --livereload -a 127.0.0.1:8001      # preview the documentation
```

Every script has its settings at the top and a description of what it does. Details:
[getting started](https://amanchry.github.io/irrimapping-geofm/setup/).
