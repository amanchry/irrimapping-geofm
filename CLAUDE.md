# CLAUDE.md — working rules for this project

MSc thesis: irrigated vs rainfed cropland mapping in Morocco with geospatial foundation models (Clay, Prithvi,
TerraMind, AgriFM) against a Sentinel-1/2 baseline. Code in `ml_dataset_prep/`, `models_run/`, `experiments/`,
`tools/`; **all documentation in `docs/`**, published with MkDocs to https://amanchry.github.io/irrimapping-geofm/.

## 1. Every change is documented — in `docs/`, in the same task

Whenever code is changed, something is run, a result comes in, or the method changes, update the matching page(s)
**before finishing**, then add an entry to the progress log. Never write documentation outside `docs/` (the root
`README.md` stays a short summary with links to the site; the code folders keep a 5-line pointer README).

| What happened | Update |
| --- | --- |
| Any change (code, settings, data, docs) | `docs/progress.md`: new entry at the top of *Change log* (`### YYYY-MM-DD (short title)` + bullets: what, why, where); update *Current status* and *Open points* if they change |
| Dataset scripts (`ml_dataset_prep/`) changed | `docs/dataset/index.md` (script section, settings, outputs, diagrams) |
| Model scripts (`models_run/`) changed | `docs/models/index.md` |
| Method or comparison rules changed | `docs/research/decisions.md` (new `D<n>` row + section: before / now / why / where) and `docs/research/protocol.md` if a rule changes; `docs/research/overview.md` if the plain-language story changes |
| New experiment | see §2 |
| Experiment finished | §2 step 5; then `docs/results.md` (status table per RQ) and the current weekly page |
| Supervisor meeting | `docs/weekly/YYYY-Www.md` from `docs/weekly/_template.md`; add it to `docs/weekly/index.md` and `docs/SUMMARY.md` |
| Paper read / reference used | `docs/research/literature.md` (what it did, what we take from it); only references checked against the paper |
| Research question or plan changed | `docs/research/thesis_design.md` (RQ status table, phases) |
| New page | add it to `docs/SUMMARY.md` (the navigation) |

## 2. Experiments

Every result that goes into the thesis comes from an experiment. One experiment = one question, one varied thing.

1. `experiments/E<NN>_<short_name>/config.yaml` from `experiments/_template/config.yaml` (id, title, short_title,
   status, question, `changes_from`, inputs, steps, only the settings that differ; same names as in the scripts).
2. Setup page `docs/experiments/E<NN>_<short_name>/index.md` from `experiments/_template/README.md`: question, why,
   *What changes* table vs the experiment it builds on, how to run, run time.
3. Run: `EXPERIMENT = "E<NN>_..."` in `experiments/run_experiment.py` → `python experiments/run_experiment.py`
   (QUICK_TEST first). Results go to `experiments/<name>/results/`.
4. `make_report.py` (the last step of every experiment) writes `docs/experiments/<name>/report.md` (tables, figures
   between the AUTO markers), the experiment list, the results page and the **Experiments menu** in
   `docs/SUMMARY.md` (entries under "All experiments"). Never edit between `<!-- AUTO:...:START/END -->`
   markers by hand. `docs/SUMMARY.md` must stay a plain nested list (no HTML comments: they break the menu).
5. Finished: write *Answer*, *Key findings*, *Interpretation*, *Caveats*, *Next steps* in `report.md` (numbers taken
   from the generated tables, checked); `status: done` + one-line `answer:` in the config; *Outcome* on the setup
   page; run `make_report.py` again.

`listed: false` keeps an experiment out of the site (currently E04 PSE-TAE: parked for the next phase — do not show
PSE-TAE on overview pages or workflow diagrams until asked).

## 3. Writing and code style (the author's preferences)

- Concise, plain English; answer first. Tables for comparisons, Mermaid diagrams **in the markdown** for flows (no
  images of diagrams). Workflow diagrams start at the labelled ground data; no inspection step in them.
- Only real, verified references (author, year, title, venue, link). Mark anything unverified as TODO.
- Scripts: independent, **settings at the top, no command-line arguments**, a description at the top (what it does,
  outputs, how to run), short comments. Experiment mode via the `EXPERIMENT` setting + `config.yaml`.
- Inside `docs/`, link pages and images with relative paths; link code, configs and result tables with GitHub URLs
  (`https://github.com/amanchry/irrimapping-geofm/blob/main/...`). Keep figures of a page next to it (`figures/`).
- Headings: anchors follow GitHub's style (`#d3-validation-pastis-rotation-instead-of-nested-cv`).

## 4. Before finishing a task

- `mkdocs build --strict` must pass (no broken links, no page missing from the navigation).
- The progress log has an entry for the change.
- Never commit `secrets/` (Earth Engine key) or data; commit or push only when asked. A push to `main` republishes
  the website (`.github/workflows/docs.yml`).
