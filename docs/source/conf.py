# Configuration file for the GLASS Sphinx documentation builder.
#
# GLASS is a header-only CUDA/C++ library. There is no Python API to autodoc, so
# the API reference is produced by Doxygen (XML output) and surfaced here through
# Breathe. See ../Doxyfile and ../Makefile.
#
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------

project = "GLASS"
copyright = "2024, A2R Lab"
author = "A2R Lab"
release = "1.0.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "breathe",
    "sphinx.ext.autosectionlabel",
    "sphinx_design",
    "myst_parser",
]

# Keep section labels useful without generating duplicate labels for repeated
# headings such as "Parameters" and "Example".
autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 2

# MyST (Markdown) parser — lets us include/port the existing .md docs.
myst_enable_extensions = ["colon_fence", "dollarmath"]
myst_heading_anchors = 4

# -- Breathe (Doxygen bridge) ------------------------------------------------

breathe_projects = {"GLASS": "../doxygen/xml"}
breathe_default_project = "GLASS"
breathe_default_members = ()
breathe_domain_by_extension = {"cuh": "cpp", "cu": "cpp"}

templates_path = ["_templates"]
exclude_patterns = []

# The inline tier sub-namespaces (warp / thread / *_detail) are reopened across
# many headers — exactly as in the C++ source — so listing those files
# separately in the API reference makes Breathe re-emit the same namespace
# wrapper per file, and the per-tier overloads of one op (warp::posv in
# trsm.cuh vs thread::posv in posv.cuh) lose their scoping in that rendering
# and collide as bare names. Sphinx flags every re-emission as a duplicate
# declaration — structural to the doxygenfile-per-header layout, not a content
# error; the rendered HTML is correct. Sphinx ≥8.2 gives these a suppressible
# subtype (duplicate_declaration.{cpp,c}), so the docs build is
# zero-warning and CI runs sphinx with -W: any NEW warning (broken ref, bad
# markup, genuine content duplication rendered elsewhere) fails the build
# loudly instead of hiding in a warning baseline.
suppress_warnings = [
    "docutils",
    "duplicate_declaration.cpp",
    "duplicate_declaration.c",
]

# Enable numref / numbered figures.
numfig = True

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_favicon = "_static/favicon/favicon.ico"
html_theme_options = {
    "navigation_depth": 4,
    "github_url": "https://github.com/A2R-Lab/GLASS",
    "use_edit_page_button": True,
    "logo": {
        "image_light": "_static/a2r_lab.png",
        "image_dark": "_static/a2r_lab.png",
    },
    "collapse_navigation": True,
    "header_links_before_dropdown": 6,
    "navbar_end": [
        "search-button",
        "theme-switcher",
        "navbar-icon-links",
    ],
    "navbar_persistent": [],
    "show_version_warning_banner": True,
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "_static/favicon/favicon.ico"

html_context = {
    "display_github": True,
    "github_user": "A2R-Lab",
    "github_repo": "GLASS",
    "github_version": "main",
    "conf_py_path": "/docs/source/",
    "doc_path": "docs/source",
}
