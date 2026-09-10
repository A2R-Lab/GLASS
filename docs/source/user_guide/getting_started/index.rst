Getting Started
===============

New to GLASS? Start here. GLASS is header-only — there is nothing to build to
*use* it; add the repo root to your include path and ``#include "glass.cuh"``.
These two pages get you from a clean checkout to your first working kernel and
explain the single-block execution model the rest of the docs assume.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Installation
      :link: installation
      :link-type: doc

      Header-only include model, the CUDA toolkit requirement, and the optional
      MathDx setup for the ``glass::nvidia::`` backend.

   .. grid-item-card:: Library Overview
      :link: library_overview
      :link-type: doc

      What GLASS is, the single-block execution model, and the call surfaces —
      with a guide to choosing the right backend.

.. toctree::
   :maxdepth: 1
   :hidden:

   installation
   library_overview
