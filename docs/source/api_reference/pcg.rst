Preconditioned Conjugate Gradient (``glass::pcg``)
====================================================

A single-block **preconditioned conjugate gradient** solver for a
block-tridiagonal symmetric-positive-definite system ``S x = b``, with a
block-tridiagonal preconditioner ``Pinv`` applied as ``z = Pinv·r``. One CUDA
block solves one system (launch one block per independent solve); it uses only
``__syncthreads()`` — no cooperative groups.

``S`` and ``Pinv`` use the ``[L|D|R]`` block-tridiagonal layout of
:doc:`banded`, and all vectors use the same padded
``(knot_points + 2)·state_size`` layout. Size the dynamic shared memory with
``glass::pcg_scratch_bytes<T, state_size, knot_points>(threads)``; it returns
bytes, ready to pass as the launch argument. Any thread count is legal (the
fast dot reduction bounds its shuffle mask to the active lanes of a ragged
last warp); multiples of 32 are the fast path and reproduce the historical
results bit-for-bit.

See :doc:`../user_guide/concepts/block_tridiagonal` for the layout and a worked
walkthrough.

.. doxygenfile:: src/base/pcg/solve.cuh
   :no-link:
