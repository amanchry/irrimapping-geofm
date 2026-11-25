# Foundation models in this study

This page describes how the six foundation models are **adapted to this study**: the bands and months they receive,
how a 128 × 128 px tile is cut into tokens, what is merged where, and which parts are frozen or trained. The original
design of each model, as its authors built and pretrained it, is on the model pages
([overview](index.md), [Prithvi-EO-2.0](prithvi.md), [Clay v1.5](clay.md), [TerraMind](terramind.md),
[AgriFM](agrifm.md), [Presto](presto.md), [TESSERA](tessera.md)). How they are trained and scored is on the
experiment pages ([E04](../experiments/E04_frozen_geofm.md), [E05](../experiments/E05_prithvi_strategies.md)).

## Configuration in this study

Every model classifies **every 10 m pixel** ([D16](../decisions.md#d16-pixel-wise-classification-for-continuous-maps)).
The encoders see whole **128 × 128 px tiles** (1.28 km) of **12 monthly Sentinel-2 composites** of the agricultural
year, with the bands, band order, normalisation and weights of their own pretraining. **Every input uses Sentinel-2
only** ([D14](../decisions.md#d14-sentinel-2-only-for-every-input)): every model sees the same optical data, and
Sentinel-1 coverage of Utah changed between the years. All six encoders are loaded by `pixelwise_run/encoders.py`.

```mermaid
flowchart LR
    A["128 px tile<br/>12 months, Sentinel-2"] --> B["Encoder<br/>(frozen, fine-tuned<br/>or from scratch)"]
    B --> C["L-TAE over the months<br/>(image models)"]
    C --> D["Feature pyramid<br/>+ UPerNet head"]
    D --> E["irrigated / rainfed<br/>per 10 m pixel"]
```

What each encoder receives, model by model, is in the table under the [tile figure](#what-each-encoder-sees-of-a-tile).

### Frozen, fine-tuned or trained from scratch

How much of the encoder is trained differs between the experiments ([RQ2 · Fine-tuning](../phase2_models/fine_tuning.md)
explains the strategies):

| Experiment | Models | Encoder | Trained |
| --- | --- | --- | --- |
| [E04](../experiments/E04_frozen_geofm.md) (RQ1) | all six | **frozen**: pretrained weights, no gradient updates, evaluation mode | head only (1.7–3.2 M weights, depending on the encoder) |
| [E05](../experiments/E05_prithvi_strategies.md) (RQ2) | Prithvi-EO-2.0, Morocco first | **frozen**, **LoRA** (low-rank adapters, rank 16, on the attention and MLP layers), **fully fine-tuned** (every weight, encoder learning rate 10⁻⁵, head 10⁻⁴) or **from scratch** (same network from random weights); 10, 25, 50 or 100% of the training parcels | head only (frozen), 5.5 M adapters + head (LoRA), or all ≈ 306 M weights |


## What each encoder sees of a tile

![How each model cuts one tile and its 12 months into tokens](../figures/geofm_tokens.png)

*What each encoder receives in this study: the bands, how one 128 × 128 px tile (1.28 km) of 12 monthly composites
is cut into tokens (one cell = one token, the labelled cell shaded), and how the 12 months are grouped per encoder pass
(one box = one pass). Presto and TESSERA run once per pixel (16,384 runs per tile); their tokens are made from that
pixel's series: 6 band groups × 12 months = 72 tokens plus a location token (Presto), or one token per clear month,
padded to 8 or 16 (TESSERA). Drawn by `tools/a8_geofm_tokens_figure.py`.*

| | Prithvi-EO-2.0 | Clay v1.5 | TerraMind | AgriFM | Presto | TESSERA v2 |
| --- | --- | --- | --- | --- | --- | --- |
| **Family** | image ViT | image ViT | image ViT, multimodal | image video-Swin | pixel transformer | pixel transformer |
| **Bands** | 6: B2 B3 B4 B8A B11 B12 | 10: B2–B8A, B11, B12 | 12: B1–B12 without B10 | 10: B2–B8A, B11, B12 | 10 + NDVI | 10 (TESSERA order) |
| **Normalisation** | Prithvi's pretraining statistics | Clay's Sentinel-2 statistics | TerraMind's pretraining statistics | the region's band statistics (D15) | reflectance / 10,000 | TESSERA's statistics |
| **One token** | 16 × 16 px, 1 month | 8 × 8 px, 1 month | 16 × 16 px, 1 month | 4 × 4 px × 4 months | 1 band group, 1 month, 1 pixel | 1 month of 1 pixel |
| **Months per pass** | 4, with date and location; 3 passes | 1, with week and location; 12 passes | 1, no date; 12 passes | all 12 in one pass | all 12 (cloudy months interpolated) | the clear months, dated (15th) |
| **Encoder weights** | 304 M | 311 M | 87 M | 88 M | 0.4 M | 21 M |
| **Output on a tile** | 1024 × 8 × 8 per month | 1024 × 16 × 16 per month | 768 × 8 × 8 per month | maps of 32², 16², 8², 4² | 128 per pixel | 128 per pixel |
| **Into the head** | L-TAE → 4-scale pyramid → UPerNet | L-TAE → 4-scale pyramid → UPerNet | L-TAE → 4-scale pyramid → UPerNet | its 4 maps → UPerNet | pooled to 4 scales → UPerNet | pooled to 4 scales → UPerNet |

Encoder weights counted on the loaded encoders of this study (`pixelwise_run/encoders.py`); TerraMind and AgriFM
publish no count.


Encoder weights counted on the loaded encoders of this study (`pixelwise_run/encoders.py`); TerraMind and AgriFM
publish no count. Step-by-step figures of each encoder's set-up: [E04](../experiments/E04_frozen_geofm.md#how-e04-works-step-by-step).

## What this means for the comparison

- **Spatial unit.** The image ViTs (Prithvi, TerraMind 160 m; Clay 80 m) decide on a coarse grid that the head
  upsamples to 10 m, so field edges are soft; AgriFM starts at 40 m; Presto and TESSERA decide per 10 m pixel but see
  no neighbours, which the UPerNet head adds afterwards.
- **Time.** Prithvi, Clay and TerraMind encode months separately; the L-TAE combines them. AgriFM, Presto and
  TESSERA combine the months inside the encoder.
- **Pretraining gap.** Prithvi was pretrained on 30 m HLS, AgriFM partly on 30 m and 250 m data, the others at 10 m;
  Presto, TESSERA and Prithvi saw time series, Clay and TerraMind single dates.
- **Inputs left out.** TerraMind, Presto and TESSERA were pretrained with Sentinel-1 (and Presto with weather and
  terrain); here they receive Sentinel-2 only.


## Settings used in this study (details)

One subsection per model: what is taken from the original design and what is changed for the tiles of this study. The original design is on the model's own page.

### Prithvi-EO-2.0

Prithvi-EO-2.0 is pretrained on HLS time series and encodes acquisition date and location explicitly. The six HLS
bands (Blue, Green, Red, Narrow NIR, SWIR 1, SWIR 2; Sentinel-2 B2, B3, B4, B8A, B11, B12) are standardised with the
model's pretraining statistics. Following the pretraining configuration, four frames are processed per pass
(September–December, January–April, May–August); each frame retains its own tokens, giving one vector per month.
The model is loaded through TerraTorch (`prithvi_eo_v2_300_tl`). **Limitation:** pretraining used 30 m HLS data,
whereas 10 m Sentinel-2 is used here; the resulting 160 m patch is closer to typical parcel size than the 480 m patch
at native resolution.

Original design: [Prithvi-EO-2.0](prithvi.md).

#### From one tile to a class per 10 m pixel (Prithvi-EO-2.0)

![Prithvi-EO-2.0 in this study, step by step](figures/prithvi_pipeline_steps.png)

*One 128 × 128 px tile through Prithvi-EO-2.0 and the trained head, with the shape of the data after every step (one tile; grey = frozen encoder, orange = trained head). Schematic: the maps are illustrations, the sizes are those of the model (`pixelwise_run/encoders.py`, `pixelwise_run/tile_common.py`). Drawn by `tools/a11_geofm_pipeline_figures.py`.*

| Step | What happens | Output for one tile | Trained in E04 |
| --- | --- | --- | --- |
| 0 · Input | 12 monthly composites, 6 bands, standardised with Prithvi's statistics; pixels without a clear observation set to 0 and masked | 12 × 6 × 128 × 128 | – |
| 1 · Encoder | 3 passes of 4 months; each month cut into 8 × 8 patches of 16 × 16 px; a patch (16 × 16 px × 6 bands = 1,536 numbers) is projected to a token of 1,024 values, date and location embeddings are added, and 24 transformer layers let every token attend to the 256 tokens of its pass | 12 × 1,024 × 8 × 8 | no (304 M weights) |
| 2 · L-TAE | at each of the 64 patch positions the 12 monthly vectors are reduced to 256 values and weighted by 16 attention heads (months without a clear pixel in the patch masked) | 256 × 8 × 8 | yes (0.36 M) |
| 3 · Feature pyramid | four map sizes from the 8 × 8 map: × 4 and × 2 by transposed convolutions, × 1 by a 1 × 1 convolution, × ½ by max-pooling | 128 × 32², 16², 8², 4² | yes (0.39 M) |
| 4 · UPerNet | pyramid pooling on the 4 × 4 map, top-down sums from coarse to fine, fusion of the four sizes at 32 × 32, 1 × 1 convolution to the two classes | 2 × 32 × 32 (40 m cells) | yes (1.72 M) |
| 5 · Upsampling | bilinear interpolation × 4; the loss is the class-weighted cross-entropy at the labelled pixels only | 2 × 128 × 128 (10 m) | – |

**Width of a token.** The 1,024 values per token are the width of the encoder (ViT-Large), a property of the model,
not of the input: every patch, whatever its content, becomes 1,024 values, and only the values differ. The input
sets the number of tokens (8 × 8 per month for a 128 px tile with 16 px patches), not their length.

**One vector per 160 m patch, not per pixel.** The encoder describes the 256 pixels of a 16 × 16 px patch with one
vector per month; no per-pixel embedding exists inside Prithvi. The 10 m classes are produced by the head: the
pyramid and UPerNet decide on 40 m cells using the neighbouring patches and the coarser context, and the final
bilinear interpolation spreads those decisions to 10 m. Field edges inside a patch are therefore resolved only as far
as the head can infer them from the patch and its neighbours.

**Why a pyramid.** A plain vision transformer keeps one resolution throughout (8 × 8 here), whereas segmentation
heads combine fine maps, which locate boundaries, with coarse maps, which carry context. The simple feature pyramid
of ViTDet (Li et al., 2022), used for plain ViTs in PANGAEA (Marsocci et al., 2024), derives four sizes from the last
layer alone. The upsampled maps are learnt from the 8 × 8 map and add no new spatial information.

**How UPerNet merges the sizes** (Xiao et al., 2018). (1) *Pyramid pooling*: the 4 × 4 map is averaged to 1 × 1,
2 × 2 and 4 × 4 cells, brought back to 4 × 4 and concatenated, which gives every cell the context of the whole tile;
pool sizes 1, 2, 4 replace the usual 1, 2, 3, 6 because the coarsest map has only 4 × 4 cells. (2) *Top-down path*:
each coarser map is upsampled and added to the next finer one (4 → 8 → 16 → 32), so that the fine maps also carry
context. (3) *Fusion*: the four maps are upsampled to 32 × 32, concatenated and mixed by a 3 × 3 convolution; a 1 × 1
convolution then gives two class scores per 40 m cell.

**L-TAE** (Garnot & Landrieu, 2020). Each head has one learned query that scores the 12 months of a patch; the
softmax of the scores weights the months, and the weighted sums of the 16 heads form the 256-value output. The
weights are learned from the labels, so months that separate irrigated from rainfed fields (for example the dry
summer, when only irrigated fields stay green) can receive more weight.

### Clay v1.5

Clay encodes each band together with its central wavelength, which allows a single model to process imagery from
different sensors. Ten Sentinel-2 bands (B2–B8A, B11, B12) are standardised with the statistics of Clay's
`metadata.yaml`. Additional inputs are the pixel size (10 m), the acquisition date (week of year, sine/cosine
encoded; monthly composites are dated mid-month) and the location (latitude and longitude, sine/cosine encoded).
The model processes one date per forward pass, so the twelve months are embedded independently. With 8 px patches,
a 128 px tile yields a 16 × 16 token grid (8 × 8 for the 64 px chips of the archived parcel benchmark).

Original design: [Clay v1.5](clay.md).

#### From one tile to a class per 10 m pixel (Clay v1.5)

![Clay v1.5 in this study, step by step](figures/clay_pipeline_steps.png)

*One 128 × 128 px tile through Clay v1.5 and the trained head, with the shape of the data after every step (one tile; grey = frozen encoder, orange = trained head). Schematic: the maps are illustrations, the sizes are those of the model (`pixelwise_run/encoders.py`, `pixelwise_run/tile_common.py`). Drawn by `tools/a11_geofm_pipeline_figures.py`.*

| Step | What happens | Output for one tile | Trained in E04 |
| --- | --- | --- | --- |
| 0 · Input | 12 monthly composites, 10 bands, standardised with Clay's statistics; pixels without a clear observation set to 0 and masked | 12 × 10 × 128 × 128 | – |
| 1 · Encoder | 12 passes of one month; each month cut into 16 × 16 patches of 8 × 8 px; a patch (8 × 8 px × 10 bands = 640 numbers) is projected to 1,024 values with weights generated from the bands' wavelengths; week, location and pixel size are added; 24 transformer layers over the 256 tokens of the month | 12 × 1,024 × 16 × 16 | no (311 M) |
| 2 · L-TAE | at each of the 256 patch positions the 12 monthly vectors are reduced to 256 values and weighted by 16 heads | 256 × 16 × 16 | yes (0.36 M) |
| 3 · Feature pyramid | × 4, × 2, × 1, × ½ of the 16 × 16 map | 128 × 64², 32², 16², 8² | yes (0.39 M) |
| 4 · UPerNet | pyramid pooling on the 8 × 8 map, top-down sums, fusion at 64 × 64, two class scores | 2 × 64 × 64 (20 m cells) | yes (1.72 M) |
| 5 · Upsampling | bilinear × 2; loss at the labelled pixels only | 2 × 128 × 128 (10 m) | – |

**What differs from Prithvi.** Clay's 8 px patches give a 16 × 16 token grid (80 m), so the head decides on 20 m
cells, the finest of the three image ViTs. The patch projection is not a fixed matrix: its weights are generated
from each band's central wavelength, which lets one model read different sensors. Each month is a separate pass, so
attention never links months inside the encoder; the L-TAE alone combines them. The feature pyramid, UPerNet and L-TAE steps are explained in detail under [Prithvi-EO-2.0](#from-one-tile-to-a-class-per-10-m-pixel-prithvi-eo-20).

### TerraMind-1.0-base

TerraMind is trained to generate any of nine modalities from any other and therefore learns a joint representation
of optical, radar and auxiliary data. The pixel-wise experiments supply only `S2L2A` (B1–B12), standardised with
TerraMind's pretraining statistics ([D14](../decisions.md#d14-sentinel-2-only-for-every-input)). The Sentinel-1
ablation of the archived parcel benchmark also supplied `S1GRD` (VV, VH in dB) and averaged the tokens of the two
modalities (`merge_method = "mean"`). One date is processed per pass; 16 px patches give an 8 × 8 token grid on a
128 px tile (4 × 4 on a 64 px chip).

Original design: [TerraMind-1.0-base](terramind.md).

#### From one tile to a class per 10 m pixel (TerraMind)

![TerraMind in this study, step by step](figures/terramind_pipeline_steps.png)

*One 128 × 128 px tile through TerraMind-1.0-base and the trained head, with the shape of the data after every step (one tile; grey = frozen encoder, orange = trained head). Schematic: the maps are illustrations, the sizes are those of the model (`pixelwise_run/encoders.py`, `pixelwise_run/tile_common.py`). Drawn by `tools/a11_geofm_pipeline_figures.py`.*

| Step | What happens | Output for one tile | Trained in E04 |
| --- | --- | --- | --- |
| 0 · Input | 12 monthly composites, 12 bands (S2L2A), standardised with TerraMind's statistics; pixels without a clear observation set to 0 and masked | 12 × 12 × 128 × 128 | – |
| 1 · Encoder | 12 passes of one month, no date or location; each month cut into 8 × 8 patches of 16 × 16 px; a patch (16 × 16 px × 12 bands = 3,072 numbers) is projected to 768 values; 12 transformer layers over the 64 tokens of the month | 12 × 768 × 8 × 8 | no (87 M) |
| 2 · L-TAE | at each of the 64 patch positions the 12 monthly vectors are reduced to 256 values and weighted by 16 heads | 256 × 8 × 8 | yes (0.30 M) |
| 3 · Feature pyramid | × 4, × 2, × 1, × ½ of the 8 × 8 map | 128 × 32², 16², 8², 4² | yes (0.39 M) |
| 4 · UPerNet | pyramid pooling on the 4 × 4 map, top-down sums, fusion at 32 × 32, two class scores | 2 × 32 × 32 (40 m cells) | yes (1.72 M) |
| 5 · Upsampling | bilinear × 4; loss at the labelled pixels only | 2 × 128 × 128 (10 m) | – |

**What differs from Prithvi.** The same 16 px patches and 8 × 8 grid, but a smaller encoder (ViT-Base: 12 layers,
768 values per token) and no date or location input: the encoder does not know which month it encodes, so the
month order enters only through the L-TAE's positional encoding. Only one of TerraMind's nine modalities is given
(Sentinel-2 L2A). The feature pyramid, UPerNet and L-TAE steps are explained in detail under [Prithvi-EO-2.0](#from-one-tile-to-a-class-per-10-m-pixel-prithvi-eo-20).

### AgriFM

AgriFM is designed for crop mapping and models the temporal dimension within the network. The full 12-month stack
of ten Sentinel-2 bands is processed in a single pass. Input tokens span four time steps and 4 × 4 px (the
pretrained patch embedding); the final feature map has a stride of 32 px
([D12](../decisions.md#d12-agrifm-with-its-pretrained-4-px-patch-embedding)). Because time is compressed inside the
network, there is one set of features per season: in the pixel-wise experiments its four-level feature pyramid goes
directly to the UPerNet head; the archived parcel benchmark pooled it to one vector per parcel. Only the encoder is
used, through the mmengine registries of the cloned repository.

**Normalisation** ([D15](../decisions.md#d15-agrifm-normalised-with-the-statistics-of-the-regions-chips)). Neither
the AgriFM paper nor the repository documents the normalisation of the pretraining data; the only published
statistics belong to the repository's fine-tuning example (`configs/cropland_mapping.py`, blue 4,179 ± 4,042). The
cloud-masked composites of this study are on a different scale (Morocco blue 714 ± 344), so with the config
statistics the visible and NIR bands would use only 8–30% of the expected spread. AgriFM is therefore normalised per
band with the mean and standard deviation of the region's own chips (`NORM = "data"`; statistics saved to
`agrifm_norm_stats_<tag>.json`), the usual choice when the pretraining statistics are unknown. The config
statistics are kept as a sensitivity run (`NORM = "config"`, file `agrifm-cfgnorm_*.npz`).

Original design: [AgriFM](agrifm.md).

#### From one tile to a class per 10 m pixel (AgriFM)

![AgriFM in this study, step by step](figures/agrifm_pipeline_steps.png)

*One 128 × 128 px tile through AgriFM and the trained head, with the shape of the data after every step (one tile; grey = frozen encoder, orange = trained head). Schematic: the maps are illustrations, the sizes are those of the model (`pixelwise_run/encoders.py`, `pixelwise_run/tile_common.py`). Drawn by `tools/a11_geofm_pipeline_figures.py`.*

| Step | What happens | Output for one tile | Trained in E04 |
| --- | --- | --- | --- |
| 0 · Input | 12 monthly composites, 10 bands, standardised with the region's band statistics (D15); pixels without a clear observation set to 0 | 12 × 10 × 128 × 128 | – |
| 1 · 3-D patches | the 12-month stack is cut into blocks of 4 months × 4 × 4 px (640 numbers), each projected to 128 values | 128 × 3 × 32 × 32 (3,072 tokens) | no |
| 2 · Swin stages | windowed attention (8 time steps × 7 × 7 tokens); after each stage 2 × 2 × 2 tokens are merged, halving space and time; the patch embedding and the outputs of stages 1–3, with remaining time steps stacked as channels, form four maps | 384 × 32², 512 × 16², 512 × 8², 1,024 × 4² | no (88 M in all) |
| 3 · UPerNet | pyramid pooling on the 4 × 4 map, top-down sums, fusion at 32 × 32, two class scores | 2 × 32 × 32 (40 m cells) | yes (3.23 M) |
| 4 · Upsampling | bilinear × 4; loss at the labelled pixels only | 2 × 128 × 128 (10 m) | – |

**What differs from the image ViTs.** AgriFM reads the whole year in one pass, as a 12-frame video, and reduces time
inside the network, so it needs neither an L-TAE nor an added pyramid: its hierarchical Swin stages already produce
four map sizes, which go straight to UPerNet (whose larger input widths explain its 3.23 M weights). The finest map
(32 × 32, 40 m) is the patch embedding itself, before any attention, so the finest level carries little context;
the coarser maps come from 2, 4 and 22 Swin blocks. The fourth stage (2 blocks at 4 × 4) runs but its output is not
used. The UPerNet steps are explained under [Prithvi-EO-2.0](#from-one-tile-to-a-class-per-10-m-pixel-prithvi-eo-20).

### Presto

Presto tokenises a pixel's time series by channel group and month (Sentinel-1, three Sentinel-2 groups, NIR, SWIR,
ERA5, SRTM, NDVI, Dynamic World) and adds the month and the location. Sentinel-2 only
([D14](../decisions.md#d14-sentinel-2-only-for-every-input)): the Sentinel-1, ERA5, SRTM and Dynamic World tokens are
masked, which the model supports (masked tokens are removed). Sentinel-2 is divided by 10,000 as in pretraining.
Presto requires the same tokens to be masked for all pixels of a batch, so months without a clear observation are
filled by linear interpolation in time. The embedding is the normalised mean of the encoder tokens (128-d).

Original design: [Presto](presto.md).

#### From one tile to a class per 10 m pixel (Presto)

![Presto in this study, step by step](figures/presto_pipeline_steps.png)

*One 128 × 128 px tile through Presto and the trained head, with the shape of the data after every step (one tile; grey = frozen encoder, orange = trained head). Schematic: the maps are illustrations, the sizes are those of the model (`pixelwise_run/encoders.py`, `pixelwise_run/tile_common.py`). Drawn by `tools/a11_geofm_pipeline_figures.py`.*

| Step | What happens | Output for one tile | Trained in E04 |
| --- | --- | --- | --- |
| 0 · Input | 12 monthly composites, 10 bands / 10,000 plus NDVI; cloudy months filled by linear interpolation | 12 × 11 × 128 × 128 | – |
| 1 · Encoder, once per pixel | each pixel's series is cut into 6 band groups × 12 months = 72 tokens plus a location token; 2 transformer layers (width 128); the mean of the tokens is the pixel's embedding; 16,384 runs per tile | 128 × 128 × 128 | no (0.4 M) |
| 2 · Pyramid | average pooling of the 10 m map by 1, 2, 4 and 8 (no weights) | 128 × 128², 64², 32², 16² | – |
| 3 · UPerNet | pyramid pooling on the 16 × 16 map, top-down sums, fusion at 128 × 128, two class scores | 2 × 128 × 128 (10 m) | yes (1.72 M) |
| 4 · Output | no upsampling: the scores are already per 10 m pixel; loss at the labelled pixels only | 2 × 128 × 128 (10 m) | – |

**What differs from the image models.** Presto never sees a neighbouring pixel: one run encodes one pixel, and its
tokens are that pixel's band groups and months, not image patches. The encoder therefore delivers a true 10 m map,
and the pyramid is plain average pooling, which gives each pixel the context of its 20, 40 and 80 m neighbourhood;
spatial patterns enter only through the UPerNet head. 16,384 runs per tile is cheap because the encoder is small
(0.4 M weights). The UPerNet steps are explained under
[Prithvi-EO-2.0](#from-one-tile-to-a-class-per-10-m-pixel-prithvi-eo-20).

### TESSERA v2

TESSERA encodes the Sentinel-2 and Sentinel-1 observations of a pixel as unordered sets dated by their day of year
and pools them with attention into a 128-d embedding (Matryoshka-ordered: the first 16, 32 or 64 dimensions are
usable on their own). The Medium student is used, as recommended by the authors. TESSERA was trained on single
acquisitions; here each observation is a monthly composite dated the 15th of its month, and only clear months are
used, padded or subsampled to the observation-count bin of the reference code (8 or 16). Sentinel-2 only: the
Sentinel-1 branch receives one empty observation, as the reference code does without Sentinel-1.

Original design: [TESSERA v2](tessera.md).

#### From one tile to a class per 10 m pixel (TESSERA v2)

![TESSERA v2 in this study, step by step](figures/tessera_pipeline_steps.png)

*One 128 × 128 px tile through TESSERA v2 and the trained head, with the shape of the data after every step (one tile; grey = frozen encoder, orange = trained head). Schematic: the maps are illustrations, the sizes are those of the model (`pixelwise_run/encoders.py`, `pixelwise_run/tile_common.py`). Drawn by `tools/a11_geofm_pipeline_figures.py`.*

| Step | What happens | Output for one tile | Trained in E04 |
| --- | --- | --- | --- |
| 0 · Input | 12 monthly composites, 10 bands in TESSERA's order, standardised with its statistics; only clear months kept, each dated the 15th | 12 × 10 × 128 × 128 | – |
| 1 · Encoder, once per pixel | one token per clear month (10 bands + day of year), padded to 8 or 16 by repeating months; 4 transformer layers (width 256) and attention pooling; joined with the empty Sentinel-1 branch and reduced to 128 values; 16,384 runs per tile | 128 × 128 × 128 | no (21 M) |
| 2 · Pyramid | average pooling of the 10 m map by 1, 2, 4 and 8 (no weights) | 128 × 128², 64², 32², 16² | – |
| 3 · UPerNet | pyramid pooling on the 16 × 16 map, top-down sums, fusion at 128 × 128, two class scores | 2 × 128 × 128 (10 m) | yes (1.72 M) |
| 4 · Output | no upsampling; loss at the labelled pixels only | 2 × 128 × 128 (10 m) | – |

**What differs from Presto.** Both encode one pixel per run, but TESSERA's tokens are whole observations (all bands
of one clear month) rather than band groups, cloudy months are left out instead of interpolated, and each token
carries its day of year, so an irregular set of dates is handled directly. Padding repeats some clear months (for 10
clear months, six are repeated to reach 16 tokens, following the reference inference code). The UPerNet steps are
explained under [Prithvi-EO-2.0](#from-one-tile-to-a-class-per-10-m-pixel-prithvi-eo-20).


## Earlier parcel-wise benchmark (archived)

The first benchmark ([archive](../archive/parcel_wise_classification/index.md)) classified whole parcels in Morocco
with the four image models, all frozen. Each model received a 64 × 64 px chip (640 m) centred on the parcel, and the
parcel embedding was the mean of the patch tokens covering the parcel, weighted by the fraction of each patch inside
the parcel ([Models pipeline §5](../archive/parcel_wise_classification/models_pipeline.md#5-scripts-06a06d--foundation-model-embeddings)).
These embeddings cannot give wall-to-wall maps, which is why the study moved to the pixel-wise set-up
([D16](../decisions.md#d16-pixel-wise-classification-for-continuous-maps)).

| Input | Chips | Bands | Patch | Temporal handling | Output per parcel | Script |
| --- | --- | --- | --- | --- | --- | --- |
| S2 baseline | parcel features (inner pixels) | 12 S2 bands, NDVI, NDWI, EVI | parcel median | 12 months concatenated | 180 features | `03` |
| Clay v1.5 | p8 | 10 S2 bands | 8 px (80 m) | one month per forward pass | 12 × 1024 | `02a` |
| Prithvi-EO-2.0 | p16 | 6 S2 bands (HLS set) | 16 px (160 m) | four months per pass, three passes | 12 × 1024 | `02b` |
| TerraMind | p16 | 12 S2 bands | 16 px (160 m) | one month per forward pass | 12 × 768 | `02c` |
| AgriFM | p16 | 10 S2 bands | 4 px pretrained patch embedding; output stride 32 px (320 m) | full 12-month stack in one pass | 1 × 1024 | `02d` |

An optional ablation of that benchmark (Morocco only) added Sentinel-1 where a model was pretrained on it: TerraMind
with Sentinel-1 (`MODALITIES = ["S2L2A", "S1GRD"]` in script 02c, written as `terramind-s1s2_agyear_tiles.npz`) and
the baseline with radar features (`S1S2_VARIABLES = "all"` in script 03).


## Weights

All weights are stored in `checkpoints/<model>/` (excluded from version control) and are downloaded automatically on
the first use (`pixelwise_run/encoders.py`).

| Model | Folder | Downloaded by |
| --- | --- | --- |
| Clay v1.5 | `checkpoints/clay/` (`clay-v1.5.ckpt`, `metadata.yaml`) | `pixelwise_run/encoders.py` (archive: `parcelwise_run/06a_embed_clay.py`) |
| Prithvi-EO-2.0-300M-TL | `checkpoints/prithvi/` (Hugging Face cache) | `pixelwise_run/encoders.py` (archive: `parcelwise_run/06b_embed_prithvi.py`) |
| TerraMind-1.0-base | `checkpoints/terramind/` (Hugging Face cache) | `pixelwise_run/encoders.py` (archive: `parcelwise_run/06c_embed_terramind.py`) |
| AgriFM | `checkpoints/agrifm/` (`AgriFM.pth`); code in `external/AgriFM` | `pixelwise_run/encoders.py` (archive: `parcelwise_run/06d_embed_agrifm.py`) |
| Presto | `checkpoints/presto/` (`default_model.pt`, from the repository); code in `external/presto` | `pixelwise_run/encoders.py` |
| TESSERA v2 Medium | `checkpoints/tessera/ckpt/` (`student_medium.pt`, Hugging Face); code in `external/tessera` | `pixelwise_run/encoders.py` |

Installation of the required packages is described in [Environment setup](../setup.md).


## References

- Garnot, V. S. F. & Landrieu, L. (2020). Lightweight temporal self-attention for classifying satellite images time
  series (L-TAE). *AALTD*.
- Li, Y., Mao, H., Girshick, R. & He, K. (2022). Exploring plain vision transformer backbones for object detection
  (ViTDet). *ECCV*. arXiv:2203.16527.
- Marsocci, V. et al. (2024). PANGAEA: a global and inclusive benchmark for geospatial foundation models.
  arXiv:2412.04204.
- Xiao, T., Liu, Y., Zhou, B., Jiang, Y. & Sun, J. (2018). Unified perceptual parsing for scene understanding.
  *ECCV*.
