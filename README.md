# Irrigated Cropland Mapping with Geospatial Foundation Models

**📘 Documentation: [amanchry.github.io/irrimapping-geofm](https://amanchry.github.io/irrimapping-geofm/)**

The project maps **irrigated** and **rainfed** cropland in Morocco from Sentinel-1 and Sentinel-2 time series. It
compares hand-built satellite features with the embeddings of four geospatial foundation models (**Clay v1.5,
Prithvi-EO-2.0, TerraMind, AgriFM**) on the same imagery and the same 8,781 field-surveyed parcels, tested with
spatial cross-validation on areas the models have never seen. Later phases fine-tune the best model, produce annual
irrigated-cropland maps and test transfer to a new region.

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
