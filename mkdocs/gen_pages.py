"""
gen_pages.py  —  collect the project's markdown pages into the MkDocs site (run by mkdocs, not by hand)
=====================================================================================================

What it does (at every `mkdocs serve / build / gh-deploy`, via the mkdocs-gen-files plugin)
-----------------------------------------------------------------------------------------
1. Mirrors the documentation pages into the site at the SAME paths as in the repository
   (README.md, docs/..., ml_dataset_prep/README.md, models_run/README.md, experiments/...), so every
   relative link that works on GitHub also works on the site. Nothing is copied on disk.
2. Copies the images the pages show (figures of the experiments, dataset maps).
3. Links to anything that is not a page or an image (scripts, CSV tables, folders, config files) are
   pointed to the file on GitHub.
4. Writes the navigation (SUMMARY.md, read by mkdocs-literate-nav). Experiments are added
   automatically from experiments/E*/config.yaml; "listed: false" leaves an experiment out.

To add a page: add it to PAGES / NAV below.
"""
import os
import re
from pathlib import Path

import mkdocs_gen_files

# ================================ SETTINGS ================================
ROOT = Path(__file__).resolve().parents[1]                     # project folder
REPO_URL = "https://github.com/amanchry/irrimapping-geofm"    # links to scripts, tables and folders
BRANCH = "main"
IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}
# ==========================================================================


def experiments():
    """[(id, short title, folder)] of the listed experiments, in folder order."""
    out = []
    for cfg in sorted((ROOT / "experiments").glob("E*/config.yaml")):
        text = cfg.read_text()
        if re.search(r"^listed:\s*false", text, re.M):
            continue
        eid = re.search(r"^id:\s*(\S+)", text, re.M).group(1)
        short = (re.search(r"^short_title:\s*(.+?)\s*(#.*)?$", text, re.M)
                 or re.search(r"^title:\s*(.+)$", text, re.M)).group(1).strip().strip('"')
        out.append((eid, short, cfg.parent.relative_to(ROOT).as_posix()))
    return out


def weeks():
    return sorted((p for p in (ROOT / "docs" / "weekly").glob("*.md") if p.stem[:1].isdigit()), reverse=True)


# Navigation: (title, page) and (section, [children]); a section with a page has that page as its index
NAV = [
    ("Home", "README.md"),
    ("Research", "docs/README.md", [
        ("Thesis design", "docs/01_thesis_design.md"),
        ("Overview", "docs/02_overview.md"),
        ("Evaluation protocol", "docs/03_protocol.md"),
        ("Decisions", "docs/04_decisions.md"),
        ("Literature notes", "docs/research/README.md"),
    ]),
    ("Results", None, [
        ("Results overview", "docs/05_results.md"),
        ("Experiments", "experiments/README.md",
         [(f"{eid} · {short}", f"{folder}/README.md",
           [("Report", f"{folder}/REPORT.md")] if (ROOT / folder / "REPORT.md").exists() else [])
          for eid, short, folder in experiments()]),
    ]),
    ("Meetings", "docs/weekly/README.md", [(f"Week {p.stem}", p.relative_to(ROOT).as_posix()) for p in weeks()]),
    ("Code reference", None, [
        ("Part 1 · Dataset", "ml_dataset_prep/README.md"),
        ("Part 2 · Models", "models_run/README.md"),
        ("Data source", "docs/data_source/readme.md"),
        ("Progress log", "PROGRESS.md"),
    ]),
]


def nav_pages(items):
    for item in items:
        if item[1]:
            yield item[1]
        if len(item) > 2:
            yield from nav_pages(item[2])


def summary(items, depth=0):
    lines = []
    for item in items:
        title, page = item[0], item[1]
        lines.append("    " * depth + (f"* [{title}]({page})" if page else f"* {title}"))
        if len(item) > 2:
            lines += summary(item[2], depth + 1)
    return lines


PAGES = set(nav_pages(NAV))
LINK = re.compile(r"(!?\[[^\]]*\]\()([^)\s]+)((?:\s+\"[^\"]*\")?\))|((?:src|href)=\")([^\"]+)(\")")
CODE = re.compile(r"(```.*?```)", re.S)                       # fenced code blocks are left as they are


def rewrite(text, page):
    """Fix the relative links of one page for the site (pages and images stay, other files -> GitHub)."""
    here = (ROOT / page).parent

    def fix(target):
        if re.match(r"^(https?:|mailto:|#|data:)", target):
            return target
        path, _, anchor = target.partition("#")
        dest = (here / path).resolve()
        try:
            rel = dest.relative_to(ROOT).as_posix()
        except ValueError:
            return target
        if dest.is_dir() and f"{rel}/README.md" in PAGES:              # folder with a page -> that page
            new = Path(f"{rel}/README.md")
        elif rel in PAGES:
            return target
        elif dest.suffix.lower() in IMAGE_TYPES and dest.exists():
            with mkdocs_gen_files.open(rel, "wb") as f:                # images: copied into the site
                f.write(dest.read_bytes())
            return target
        else:                                                         # scripts, tables, folders -> GitHub
            kind = "tree" if dest.is_dir() or path.endswith("/") else "blob"
            return f"{REPO_URL}/{kind}/{BRANCH}/{rel}" + (f"#{anchor}" if anchor else "")
        out = os.path.relpath(ROOT / new, here).replace(os.sep, "/")
        return out + (f"#{anchor}" if anchor else "")

    def sub(m):
        if m.group(1):
            return m.group(1) + fix(m.group(2)) + m.group(3)
        return m.group(4) + fix(m.group(5)) + m.group(6)

    parts = CODE.split(text)
    return "".join(p if i % 2 else LINK.sub(sub, p) for i, p in enumerate(parts))


for page in sorted(PAGES):
    src = ROOT / page
    if not src.exists():
        continue
    with mkdocs_gen_files.open(page, "w") as f:
        f.write(rewrite(src.read_text(encoding="utf-8"), page))

with mkdocs_gen_files.open("SUMMARY.md", "w") as f:
    f.write("\n".join(summary(NAV)) + "\n")
