# experiments/

One folder per experiment with its `config.yaml` (all settings) and `results/` (written by the scripts).

- `run_experiment.py` runs an experiment from its config: training per input, comparison, report.
- `make_report.py` writes the tables and figures into the experiment's report page in `docs/experiments/`.
- `_template/` holds the config, setup page and report page to start a new experiment.

**Documentation:** setup pages and reports of every experiment, and how to add one:
[https://amanchry.github.io/irrimapping-geofm/experiments/](https://amanchry.github.io/irrimapping-geofm/experiments/) (source: [`docs/experiments/`](../docs/experiments/index.md))
