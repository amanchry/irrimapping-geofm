<!-- Setup page of an experiment: copy to docs/experiments/E<NN>_<short_name>/index.md -->
# E00 · Short title

| | |
| --- | --- |
| **Status** | planned |
| **Research question** | RQ1 |
| **Builds on** | – |
| **Config** | [`config.yaml`](https://github.com/amanchry/irrimapping-geofm/blob/main/experiments/E00_short_name/config.yaml) |
| **Report** | [report](report.md), once the results are in |

## Question
The one question this experiment answers.

## Why
What we expect to learn, and which decision depends on it.

## What changes
| | Previous experiment | This experiment |
| --- | --- | --- |
| setting | value | value |

Everything else (parcels, folds, classifiers, metrics) follows the [evaluation protocol](../../research/protocol.md).

## How to run
```bash
# in experiments/run_experiment.py:  EXPERIMENT = "E00_short_name"
python experiments/run_experiment.py
```
Run time:

## Outcome
Filled in when the results are in: two or three lines, link to the [report](report.md).
