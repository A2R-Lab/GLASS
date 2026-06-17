Block-tridiagonal Matvec (``glass::banded::``)
==============================================

Single-block matrix-vector product for a **block-tridiagonal** matrix — the
sparsity pattern of the KKT / Schur systems that arise in trajectory
optimization and MPC. The matrix is stored as ``NumBlockRows`` contiguous
``BlockSize × (3·BlockSize)`` row-major strips laid out ``[L | D | R]``
(left / diagonal / right blocks), and the vectors use a **padded** layout
``(NumBlockRows + 2)·BlockSize`` (one ``BlockSize`` pad block on each end) so the
edge block-rows need no special case — their absent ``L`` / ``R`` simply multiply
the zero pad.

See :doc:`../user_guide/concepts/block_tridiagonal` for the layout in detail, and
:doc:`pcg` for the conjugate-gradient solver built on top of this matvec.

.. doxygenfile:: src/base/banded/bdmv.cuh
