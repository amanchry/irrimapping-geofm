"""Copy the figures written by the pipeline scripts and experiments into the documentation (docs/).

The dataset and model pages show these figures next to the step that produces them. Run this script after
re-running a pipeline step, so the website shows the current figures. Large figures are reduced to MAX_WIDTH
pixels (aspect ratio kept), and image-heavy figures are saved as JPEG, to keep the website light.

Run:  python tools/a6_copy_doc_figures.py
"""
from pathlib import Path
from PIL import Image

# ---------------------------------------------------------------- settings
ROOT = Path(__file__).resolve().parents[1]               # project folder
DOCS = ROOT / "docs"
MAX_WIDTH = 1600                                         # px; wider figures are downscaled
FIGURES = {                                              # source (in the project) -> target (in docs/)
    "outputs/morocco/dataset/01_labels/parcels_map.png":                 "phase1_data/figures/parcels_map.png",
    "outputs/morocco/dataset/01_labels/class_balance_and_size.png":      "phase1_data/figures/class_balance_and_size.png",
    "outputs/morocco/dataset/01_labels/survey_timing.png":               "phase1_data/figures/survey_timing.png",
    "outputs/morocco/dataset/02_folds/class_similarity_by_distance.png": "phase1_data/figures/class_similarity_by_distance.png",
    "outputs/morocco/dataset/02_folds/spatial_folds.png":                "phase1_data/figures/spatial_folds.png",
    "outputs/morocco/dataset/02_folds/spatial_folds_blocks.png":         "phase1_data/figures/spatial_folds_blocks.png",
    "outputs/morocco/dataset/02_folds/train_test_distance.png":          "phase1_data/figures/train_test_distance.png",
    "outputs/morocco/study/01_inspection/observations_per_month.png":    "phase1_data/figures/observations_per_month.png",
    "outputs/morocco/study/01_inspection/purity.png":                    "phase1_data/figures/purity.png",
    "outputs/morocco/study/01_inspection/seasonal_profiles.png":         "phase1_data/figures/seasonal_profiles.png",
    "outputs/morocco/study/01_inspection/examples_irrigated.png":        "phase1_data/figures/examples_irrigated.jpg",
    "outputs/morocco/study/01_inspection/examples_rainfed.png":          "phase1_data/figures/examples_rainfed.jpg",
    "experiments/E01_frozen_pca_nested/results/s1s2-all_agyear_tiles-inner-asc/cv_random_vs_spatial.png": "experiments/E01_frozen_pca_nested/figures/cv_random_vs_spatial_baseline.png",
    "experiments/E01_frozen_pca_nested/results/s1s2-all_agyear_tiles-inner-asc/importance.png":           "experiments/E01_frozen_pca_nested/figures/importance_baseline.png",
}
# --------------------------------------------------------------------------

for src, dst in FIGURES.items():
    s, d = ROOT / src, DOCS / dst
    if not s.exists():
        print(f"missing  {src}")
        continue
    d.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(s)
    if img.width > MAX_WIDTH:
        img = img.resize((MAX_WIDTH, round(img.height * MAX_WIDTH / img.width)), Image.LANCZOS)
    if d.suffix == ".jpg":                               # photo-like figures: JPEG keeps them small
        img.convert("RGB").save(d, quality=85, optimize=True)
    else:
        img.save(d, optimize=True)
    print(f"{d.stat().st_size // 1024:6d} KB  {dst}")
