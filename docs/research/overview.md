# 2 · The method in plain language

The question in one line: **is a field irrigated or rainfed?**, and does a pretrained foundation model answer it
better than the classic way of describing a field by its monthly satellite values?

> **One field, one label.** A surveyor visited each field and recorded whether it is irrigated. The model's job is
> to give the same answer from satellite images alone, for fields in areas it has never seen.

---

## The chain

```mermaid
flowchart TD
    A["<b>Field survey</b><br/>~10,000 fields visited<br/>Dec 2023 – Jan 2025"] --> B["<b>Clean labels</b><br/>cropland only · irrigated or rainfed<br/>8,781 fields"]
    B --> C["<b>Spatial folds</b><br/>Morocco cut into 20 km squares,<br/>grouped into 5 folds"]
    S["<b>Satellites</b><br/>Sentinel-2 (optical) · Sentinel-1 (radar)"] --> D
    C --> D["<b>One shared image store</b><br/>1.28 km tiles · 12 monthly pictures<br/>Sep – Aug of the field's season"]
    D --> E["<b>Classic description</b><br/>each field's monthly band values<br/>and indices: 204 numbers"]
    D --> F["<b>Foundation models</b><br/>each reads a 640 m window around the field<br/>and returns a summary vector"]
    E --> G["<b>Same four classifiers</b><br/>logistic regression · random forest<br/>gradient boosting · SVM"]
    F --> G
    G --> H["<b>Test on unseen areas</b><br/>train on 3 folds, tune on 1, test on 1,<br/>five times"]
    H --> I["<b>Compare</b><br/>same fields, same tests:<br/>which description is better?"]

    classDef a fill:#1F5F8B,color:#fff,stroke:#1F5F8B
    classDef b fill:#FFF4E6,stroke:#B4541A,color:#1C2733
    classDef c fill:#B4541A,color:#fff,stroke:#B4541A
    class A,B,C,D a
    class S,E,F b
    class G,H,I c
```

## Step by step

**1. Labels.** The Moroccan field survey (Ouassanouan et al. 2026) records crop and irrigation method for each
field. Every cropland field becomes *irrigated* (flood, sprinkler, drip, pivot …) or *rainfed*. Rainfed fields are
only 14% of the data, so every score treats both classes as equally important.

**2. Spatial folds.** Neighbouring fields look alike. If a test field has a training field next door, the model
can look good without having learnt anything general. So the country is cut into 20 km squares and whole squares
are kept together in one of five folds. A model is always tested on squares it never saw.

**3. One image store for every model.** For each field's agricultural year (September to August) the images are
averaged into 12 monthly pictures: Sentinel-2 bands with clouds removed, and Sentinel-1 radar. They are cut into a
fixed grid of 1.28 km tiles that do not overlap. The baseline and every foundation model read exactly these
pixels, so differences come from the method, not the data.

**4. Two ways to describe a field.**
- *Classic:* the median of the field's pixels for each band, three vegetation and water indices and the radar,
  for each of the 12 months: 204 numbers. Months that were fully cloudy are filled from the months around them.
- *Foundation model:* a 64 × 64-pixel window (640 m) around the field goes into the frozen model. It returns a
  vector for each small patch; the patches that cover the field are averaged, weighted by how much of each patch is
  inside the field. Clay, Prithvi and TerraMind do this for each month; AgriFM reads the whole year at once.

**5. From 12 months to one vector.** A foundation model gives up to 12 × 1,024 numbers per field. Two ways to
combine the months are tested:
- *PCA pipeline:* put the 12 months side by side and compress them to 128 numbers (PCA learnt on the training
  folds only).
- *L-TAE pipeline (next):* a small attention network learns which months matter, as in the PANGAEA benchmark.

**6. Classifiers.** The same four classifiers are trained on every description, each with a small set of settings
to choose from. Logistic regression on frozen embeddings is the standard "linear probe": it shows what the
encoder already knows.

**7. Testing.** Five rounds: three folds train, the next fold picks the best setting, the remaining fold is the
test. After five rounds every field has been predicted once, by a model that never saw its area.

**8. Comparing.** All inputs are scored on the same fields. Uncertainty comes from re-drawing whole 20 km squares
1,000 times; the same draws are used for every input, so differences are measured fairly. A model counts as
better only if the difference is clearly above zero after correcting for the number of comparisons.

## What the results can and cannot say

- They say how well each description maps **new areas inside the surveyed regions** of Morocco.
- They do not yet say how a model does **in another year** or **in another country**: that is RQ3 and RQ4.
- Region and year overlap in the survey (Gharb and Tadla only 2023/24, Souss only 2024/25), so a "hard region"
  may partly be a "hard year".

