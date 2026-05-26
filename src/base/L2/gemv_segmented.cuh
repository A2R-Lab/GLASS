#pragma once
#include <cstdint>

// Segmented (batched) compile-time-size column-major GEMV.
//
// Computes `segments` independent small GEMVs concurrently in a single block,
// each of the form  y_seg = alpha * A_seg * x_seg + beta * y_seg, where every
// A_seg is M rows x N cols, column-major with leading dimension ROW_STRIDE
// (A_seg[i][j] = A[seg_a_off[seg] + i + j*ROW_STRIDE]).  M, N, ROW_STRIDE are
// compile-time so the inner column loop is fully unrolled.  This is the
// segmented analogue of row_strided_gemv<T,M,N,ROW_STRIDE>: instead of one
// matrix it walks `segments` of them, with per-segment base offsets supplied by
// the descriptor arrays seg_a_off / seg_x_off / seg_y_off.
//
// The kernel runs ONE block-stride loop over the flattened `segments*M` output
// rows; thread `rank` owns row `r`, decoded as seg = r / M, row = r % M, so all
// segments are computed simultaneously by the block (good when each GEMV is too
// small (<= ~6xN) to fill the block on its own).
//
// When FUSE_SCALED_ADD is true an extra per-segment scaled vector add is fused
// into the same pass: after computing the GEMV result for output row, we add
// S[seg_s_off[seg] + row] * scalar[seg].  S has the same per-segment layout as
// y (contiguous M-vectors at seg_s_off), and scalar is one value per segment.
// This costs no extra global write — it folds into the single y[] store.
//
// Descriptor / index convention (all IDX_T, element offsets not byte offsets):
//   seg_a_off[seg] : base element index of A_seg within A   (len `segments`)
//   seg_x_off[seg] : base element index of x_seg within x    (len `segments`)
//   seg_y_off[seg] : base element index of y_seg within y    (len `segments`)
//   seg_s_off[seg] : base element index of S_seg within S    (len `segments`, FUSE only)
//   scalar[seg]    : per-segment fused-add multiplier         (len `segments`, FUSE only)
// x_seg is N contiguous values; y_seg / S_seg are M contiguous values.
// Segments may overlap or alias freely in A/x; y_seg ranges must be disjoint
// (each output row is written exactly once).

template <typename T, uint32_t M, uint32_t N, uint32_t ROW_STRIDE = M,
          bool FUSE_SCALED_ADD = false, typename IDX_T = int>
__device__ void segmented_row_strided_gemv(
    uint32_t segments,
    const IDX_T* seg_a_off, const IDX_T* seg_x_off, const IDX_T* seg_y_off,
    const T* A, const T* x, T* y, T alpha, T beta,
    const IDX_T* seg_s_off = nullptr, const T* S = nullptr, const T* scalar = nullptr)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    uint32_t total = segments * M;
    for (uint32_t r = rank; r < total; r += size) {
        uint32_t seg = r / M;
        uint32_t row = r % M;
        uint32_t a_base = static_cast<uint32_t>(seg_a_off[seg]);
        uint32_t x_base = static_cast<uint32_t>(seg_x_off[seg]);
        uint32_t y_base = static_cast<uint32_t>(seg_y_off[seg]);
        T res = static_cast<T>(0);
        for (uint32_t col = 0; col < N; col++)
            res += A[a_base + row + col * ROW_STRIDE] * x[x_base + col];
        T out = alpha * res + beta * y[y_base + row];
        if (FUSE_SCALED_ADD)
            out += S[static_cast<uint32_t>(seg_s_off[seg]) + row] * scalar[seg];
        y[y_base + row] = out;
    }
}

// No-beta overload: y_seg = alpha * A_seg * x_seg (+ fused scaled add).
template <typename T, uint32_t M, uint32_t N, uint32_t ROW_STRIDE = M,
          bool FUSE_SCALED_ADD = false, typename IDX_T = int>
__device__ void segmented_row_strided_gemv(
    uint32_t segments,
    const IDX_T* seg_a_off, const IDX_T* seg_x_off, const IDX_T* seg_y_off,
    const T* A, const T* x, T* y, T alpha,
    const IDX_T* seg_s_off = nullptr, const T* S = nullptr, const T* scalar = nullptr)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    uint32_t total = segments * M;
    for (uint32_t r = rank; r < total; r += size) {
        uint32_t seg = r / M;
        uint32_t row = r % M;
        uint32_t a_base = static_cast<uint32_t>(seg_a_off[seg]);
        uint32_t x_base = static_cast<uint32_t>(seg_x_off[seg]);
        uint32_t y_base = static_cast<uint32_t>(seg_y_off[seg]);
        T res = static_cast<T>(0);
        for (uint32_t col = 0; col < N; col++)
            res += A[a_base + row + col * ROW_STRIDE] * x[x_base + col];
        T out = alpha * res;
        if (FUSE_SCALED_ADD)
            out += S[static_cast<uint32_t>(seg_s_off[seg]) + row] * scalar[seg];
        y[y_base + row] = out;
    }
}
