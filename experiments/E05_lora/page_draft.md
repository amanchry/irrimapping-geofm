# E05 · Fine-tuning with LoRA (RQ2)

!!! info "Status: ready to run after E04"
    Script written; the encoder is chosen from the E04 results (`EXPERIMENT_ENCODER`).

**Question (RQ2).** Does adapting the encoder improve on the frozen encoder of E04 with the same decoder, data and
training?

**What changes from E04.** Low-rank adapters (LoRA; Hu et al. 2022) are added to the attention projections of
every encoder block and trained together with the temporal head and the decoder; the pretrained weights stay
frozen.

| | E04 | E05 |
| --- | --- | --- |
| Encoder | frozen | frozen weights + trained LoRA adapters |
| Adapters | – | rank 8, α 16, on the `qkv` (Prithvi, TerraMind, AgriFM, Presto) or `to_qkv` (Clay) projections, and on the feed-forward layers `linear1` / `linear2` for TESSERA (its attention stores q, k, v in one packed weight); B initialised to zero, so the adapted encoder starts identical to the frozen one |
| Batch | 8 tiles | 4 tiles (gradients flow through the encoder) |
| Everything else | | as E04 |

## How to run

Script: [`experiments/E05_lora/run.py`](https://github.com/amanchry/irrimapping-geofm/blob/main/experiments/E05_lora/run.py).
Set the encoder to the best one of E04:

```bash
sbatch --export=ALL,SCRIPT=experiments/E05_lora/run.py,EXPERIMENT_REGION=morocco,EXPERIMENT_ENCODER=clay jobs/step_gpu.sbatch
```

## Outputs

`experiments/E05_lora/results/<region>/<encoder>_lora8/`: as E04.

## Results

*To be filled after the run.* The comparison of interest is the paired difference to the same encoder in E04
(`05_compare_experiments.py` with `REFERENCE = "E04_frozen_geofm/<encoder>"`).

| Region | Encoder | Cropland macro-F1 | 3-class macro-F1 | Δ vs E04 (same encoder) |
| --- | --- | --- | --- | --- |
| Morocco | | | | |
| Utah | | | | |

## References

- Hu, E. J. et al. (2022). LoRA: low-rank adaptation of large language models. *ICLR*.
