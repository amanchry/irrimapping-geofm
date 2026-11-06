# mkdocs/

Build files of the documentation website (not a page of it).

- `gen_pages.py` — run by MkDocs at every build: puts the markdown pages of the repository into the site at their
  own paths, copies their images, points other links to GitHub, and writes the navigation.
- `pages/` — the MkDocs `docs_dir`: only the style sheet; all pages come from `gen_pages.py`.

Build and publish: see the top of [`../mkdocs.yml`](../mkdocs.yml).
