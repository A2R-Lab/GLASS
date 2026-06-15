# Minimal makefile for the GLASS Sphinx documentation.
#
# GLASS is a header-only CUDA library, so the API reference is produced by
# Doxygen (XML) and surfaced in Sphinx via Breathe. `make all` regenerates the
# Doxygen XML first, then builds the HTML site.

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
DOXYGEN       ?= doxygen
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

all: clean doxygen html

# Regenerate the Doxygen XML consumed by Breathe. Non-fatal if doxygen is
# missing locally, but the API pages will be empty without it.
doxygen:
	@$(DOXYGEN) Doxyfile

clean:
	@rm -rf $(BUILDDIR)/* doxygen/xml

.PHONY: help Makefile all doxygen clean

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
