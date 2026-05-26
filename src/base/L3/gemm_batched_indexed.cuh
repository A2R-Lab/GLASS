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
// analogue of row_strided_gemm — useful for assembling many independent 4x4
// (e.g. SE(3) / spatial-transform) products selected by index lists, computing
// all `pairs` concurrently in a single block.
//
// Index convention (all IDX_T, MATRIX indices not element offsets):
//   a_idx[p] : matrix slot of the left factor   in A_base   (len `pairs`)
//   b_idx[p] : matrix slot of the right factor  in B_base   (len `pairs`)
//   c_idx[p] : matrix slot of the destination   in C_base   (len `pairs`)
// Distinct pairs MUST target distinct c_idx slots (each output written once);
// a_idx / b_idx may repeat or alias freely.  No alpha/beta — pure C = A*B
// overwrite (matches the assemble-a-fresh-product use case).

template <typename T, uint32_t DIM = 4, typename IDX_T = int>
__device__ void indexed_batched_gemm(
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
        for (uint32_t ind = 0; ind < DIM; ind++)
            res += Ap[row + ind * DIM] * Bp[ind + col * DIM];
        Cp[el] = res;
    }
}
