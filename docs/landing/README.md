# GLASS research landing page

The research cover page is deployed at `/GLASS/`; Sphinx documentation lives at
`/GLASS/docs/`. See the build and review instructions below.

## Review locally

From the repository root, with `docs/requirements.txt` and Doxygen installed:

```sh
make -C docs doxygen html SPHINXOPTS="-W --keep-going"
python docs/build_site.py
python docs/check_site.py docs/build/site
python -m http.server 8000 --directory docs/build/site
```

Open `http://localhost:8000/` for the cover and `/docs/` for the reference.
`build_site.py` refuses an existing output: pass `--output` with a fresh path
for subsequent previews. Relative links also work at the `/GLASS/` prefix.
Old HTML URLs redirect to matching `/docs/` pages with query strings and
fragments preserved. Old assets, downloads, and `/badges/` remain accessible.

The workflow deploys only pushes to `main`. PRs build and check without
deploying. This branch does not update the public site.

## Paper provenance and release

Content follows `plancherb1/glass-paper-arxiv` release `afc0149`.
Exact figure commit and PDF/PNG hashes are in
`../source/_static/paper/provenance.json`. PDFs are byte-identical copies;
1800-pixel PNGs are rendered previews, not regenerated experiments. Refresh:

```sh
python docs/import_paper_figures.py --paper-dir ../glass-paper-arxiv
```

The robotics table is transcribed from `figs/robotics_operators_table_orin.tex`.
`overview.svg` adapts `tex/tikz_overview.tex` for the web. Captions retain the
populations, native-only host candidates, and 16× clipping disclosure.

When arXiv announces the paper, update the pending Paper button, release
notice, and BibTeX together; add the actual identifier and URL to citation
metadata. No identifier or conference acceptance is invented in this draft.

## Template attribution

Layout and styling adapt A2R Lab's GATO cover page (`gh-pages`), itself adapted
from Nerfies under CC BY-SA 4.0. The footer retains those attributions; GLASS
software remains MIT licensed. Font families, green accents, serif headings,
figure cards, resource buttons, and citation copying follow GATO; content,
responsive layout, and navigation are tailored to GLASS.
