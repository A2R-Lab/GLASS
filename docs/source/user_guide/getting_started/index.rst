Getting Started
===============

New to GLASS? Start here. GLASS is header-only — there is nothing to build to
*use* it; add the repo root to your include path and ``#include "glass.cuh"``.
These pages get you from a clean checkout to your first working kernel,
explain the single-block execution model the rest of the docs assume, and
list the devices and toolkits the library has been checked on.

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

   .. grid-item-card:: Supported Platforms
      :link: supported_platforms
      :link-type: doc

      Which devices and toolkits carry a signed receipt, which have been
      exercised, and which only compile.

.. toctree::
   :maxdepth: 1
   :hidden:

   installation
   library_overview
   supported_platforms
