#pragma once
#include <cstdint>

// Indexed batched small square GEMM: for each pair p in [0, pairs),
//   C[c_idx[p]] = A[a_idx[p]] * B[b_idx[p]]
// where every matrix is DIM x DIM, column-major (element (i,j) at i + j*DIM),
// and the *_idx arrays select which matrix slot in the contiguous base buffers
// participates in pair p.  A_base / B_base / C_base are flat arrays of DIM*DIM
// matrices; a_idx[p] is a MATRIX index, so the matrix lives at offset
// a_idx[p]*DIM*DIM (likewise B and C).  DIM is compile-time (default 4), so the
// inner contraction loop is fully unrolled.
//
// One block-stride loop over the flattened `pairs*DIM*DIM` output elements:
// thread `rank` owns global element e, decoded as pair = e/(DIM*DIM),
// el = e%(DIM*DIM), row = el%DIM, col = el/DIM.  This is the indexed/gather
// analogue of gemm_strided — useful for assembling many independent 4x4
// (e.g. SE(3) / spatial-transform) products selected by index lists, computing
// all `pairs` concurrently in a single block.
//
// ── Generalizations (template flags, all default to the original behavior) ────
// These mirror the L2 segmented-gemv TRANSPOSE / ATOMIC_Y additions and give the
// matrix-valued backward passes (e.g. the world-frame second-order Step-5
// propagation Xᵀ·M·X → parent, and IA-block accumulation) first-party support.
//
// TRANSPOSE_A (default false): pair p computes  C_p = A_pᵀ · B_p, i.e. the left
//   factor is read transposed (A_p[k][row] instead of A_p[row][k]).  This is the
//   Xᵀ-on-the-left direction the GRiD backward recursions propagate.  A_p is
//   still stored DIM×DIM col-major; only the index it is read at changes.
//
// TRANSPOSE_B (default false): pair p reads the right factor transposed
//   (B_p[col][k] instead of B_p[k][col]), giving C_p = A_p · B_pᵀ.  Combine with
//   TRANSPOSE_A for C_p = A_pᵀ · B_pᵀ.  All matrices are square DIM×DIM, so the
//   output is always DIM×DIM regardless of the transpose flags.
//
// ATOMIC_C (default false): the per-element result is accumulated into C via
//   atomicAdd instead of being written.  This RELAXES the distinct-c_idx
//   requirement: several pairs may target the SAME c_idx slot (the parent-block
//   accumulation where many children add into one parent), and their products
//   are atomically summed.  Each thread owns a single output element and issues
//   exactly one atomicAdd to ITS OWN element Cp[el], so there is no intra-tile
//   race — overlap is only ACROSS pairs sharing a c slot, which the atomic
//   resolves.  Under ATOMIC_C the operation is the pure accumulate
//   C[c_idx[p]] += A_p(ᵀ) · B_p(ᵀ); the caller must PRE-ZERO (or pre-load) the
//   touched C slots, exactly as a beta term would — no beta is applied here (a
//   read-modify-write scaled prior C would race the concurrent atomicAdds, the
//   same rule the L2 ATOMIC_Y path documents).
//
// Index convention (all IDX_T, MATRIX indices not element offsets):
//   a_idx[p] : matrix slot of the left factor   in A_base   (len `pairs`)
//   b_idx[p] : matrix slot of the right factor  in B_base   (len `pairs`)
//   c_idx[p] : matrix slot of the destination   in C_base   (len `pairs`)
// Without ATOMIC_C, distinct pairs MUST target distinct c_idx slots (each output
// written once); with ATOMIC_C they may share c_idx freely (summed atomically).
// a_idx / b_idx may repeat or alias freely.  No alpha/beta — pure C = A*B
// overwrite (or, under ATOMIC_C, pure += accumulate into pre-zeroed C).

/**
 * @brief Indexed/gather batched square GEMM: `C[c_idx[p]] = op(A[a_idx[p]]) * op(B[b_idx[p]])`.
 *
 * For each pair `p` in `[0, pairs)` multiplies two `DIM x DIM` column-major
 * matrices selected by index into flat base buffers (`a_idx[p]` is a MATRIX
 * index, so the matrix lives at offset `a_idx[p] * DIM * DIM`), computing all
 * pairs concurrently in a single block. This is the indexed/gather analogue of
 * `gemm_strided`, useful for assembling many independent small (e.g. 4x4
 * SE(3)) products from index lists. No alpha/beta — a pure overwrite
 * (`C = op(A) * op(B)`) unless `ATOMIC_C` is set.
 *
 * Layout flags read the factors transposed in place (matrices stay square
 * `DIM x DIM`, so the output is always `DIM x DIM`):
 * `TRANSPOSE_A` gives `A_p^T * B_p`, `TRANSPOSE_B` gives `A_p * B_p^T`, and both
 * give `A_p^T * B_p^T`. Without `ATOMIC_C`, distinct pairs MUST target distinct
 * `c_idx` slots (each output written once); `a_idx` / `b_idx` may alias freely.
 *
 * @tparam T  Scalar type.
 * @tparam DIM  Compile-time matrix dimension (square; inner loop fully unrolled).
 * @tparam TRANSPOSE_A  If true, the left factor is read transposed (`C_p = A_p^T * ...`).
 * @tparam TRANSPOSE_B  If true, the right factor is read transposed (`... * B_p^T`).
 * @tparam ATOMIC_C  If true, accumulate via atomicAdd (`C[c_idx[p]] += ...`), allowing
 *                   several pairs to share a `c_idx` slot; the caller must PRE-ZERO
 *                   (or pre-load) the touched C slots, and no beta is applied.
 * @tparam IDX_T  Index type of the *_idx arrays.
 * @param pairs   Number of independent GEMMs.
 * @param a_idx   Per-pair matrix slot of the left factor in `A_base` (length `pairs`).
 * @param b_idx   Per-pair matrix slot of the right factor in `B_base` (length `pairs`).
 * @param c_idx   Per-pair matrix slot of the destination in `C_base` (length `pairs`).
 * @param A_base  Flat array of `DIM x DIM` left-factor matrices.
 * @param B_base  Flat array of `DIM x DIM` right-factor matrices.
 * @param C_base  Flat array of `DIM x DIM` destination matrices (written, or accumulated if ATOMIC_C).
 */
template <typename T, uint32_t DIM = 4, bool TRANSPOSE_A = false,
          bool TRANSPOSE_B = false, bool ATOMIC_C = false, typename IDX_T = int>
__device__ void gemm_batched_indexed(
    uint32_t pairs, const IDX_T* a_idx, const IDX_T* b_idx, const IDX_T* c_idx,
    const T* A_base, const T* B_base, T* C_base)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    constexpr uint32_t MAT = DIM * DIM;
    uint32_t total = pairs * MAT;
    for (uint32_t e = rank; e < total; e += size) {
        uint32_t pair = e / MAT;
        uint32_t el   = e % MAT;
        uint32_t row  = el % DIM;
        uint32_t col  = el / DIM;
        const T* Ap = A_base + static_cast<uint32_t>(a_idx[pair]) * MAT;
        const T* Bp = B_base + static_cast<uint32_t>(b_idx[pair]) * MAT;
        T* Cp       = C_base + static_cast<uint32_t>(c_idx[pair]) * MAT;
        T res = static_cast<T>(0);
        #pragma unroll
        for (uint32_t ind = 0; ind < DIM; ind++) {
            // A_pᵀ reads A_p[ind][row] = Ap[ind + row*DIM]; else A_p[row][ind].
            T a = TRANSPOSE_A ? Ap[ind + row * DIM] : Ap[row + ind * DIM];
            // B_pᵀ reads B_p[col][ind] = Bp[col + ind*DIM]; else B_p[ind][col].
            T b = TRANSPOSE_B ? Bp[col + ind * DIM] : Bp[ind + col * DIM];
            res += a * b;
        }
        if (ATOMIC_C) atomicAdd(&Cp[el], res);
        else          Cp[el] = res;
    }
}
