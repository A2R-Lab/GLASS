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
// y (contiguous OUT_ROWS-vectors at seg_s_off), and scalar is one value per
// segment.  This costs no extra global write — it folds into the single y[]
// store, so FUSE_SCALED_ADD is only available on the non-atomic path (it relies
// on the single read-modify-write of y).
//
// ── Generalizations (template flags, all default to the original behavior) ────
// TRANSPOSE (default false): when true, each segment computes the TRANSPOSED
//   product  y_seg[n] = alpha * Σ_i A_seg[i][n] * x_seg[i] (+ beta*y_seg[n]) for
//   n in [0,N) — i.e.  Aᵀ_seg · x_seg.  A_seg is still M×N col-major with
//   leading dimension ROW_STRIDE (A_seg[i][n] = A[a_off + i + n*ROW_STRIDE]),
//   but now x_seg has M contiguous values and y_seg has N contiguous values.
//   The block-stride loop runs over `segments*N` outputs; thread `rank` owns
//   output n, decoded as seg = r / N, n = r % N.  This is the leaf→root
//   (backward-pass) direction GRiD's recursions need (the Xᵀ·f map).
//
// ATOMIC_Y (default false): when true, the per-output result is accumulated
//   into y via atomicAdd instead of being written.  This lets segments share or
//   overlap their y ranges (the parent-accumulation / tree-reduction case where
//   several child segments add into the SAME parent rows).  Under ATOMIC_Y the
//   operation is the pure accumulate  y_seg += alpha*(A_seg·x_seg)  (or the
//   transposed Aᵀ_seg·x_seg).  ── beta is intentionally NOT applied in the
//   atomic path: scaling the prior y by beta is ill-defined under concurrent
//   atomicAdds (the read of y races the writes), so the caller must pre-scale /
//   pre-zero y before the call and treat this as a pure accumulator.  The atomic
//   overload therefore takes no beta argument.  FUSE_SCALED_ADD is likewise
//   unavailable under ATOMIC_Y (it folds into a non-atomic store); accumulate
//   the scaled vector separately if needed.
//
// Output-row count per segment is  OUT_ROWS = (TRANSPOSE ? N : M).
//
// Descriptor / index convention (all IDX_T, element offsets not byte offsets):
//   seg_a_off[seg] : base element index of A_seg within A   (len `segments`)
//   seg_x_off[seg] : base element index of x_seg within x    (len `segments`)
//   seg_y_off[seg] : base element index of y_seg within y    (len `segments`)
//   seg_s_off[seg] : base element index of S_seg within S    (len `segments`, FUSE only)
//   scalar[seg]    : per-segment fused-add multiplier         (len `segments`, FUSE only)
// Non-transpose: x_seg is N contiguous values; y_seg / S_seg are M contiguous.
// Transpose:     x_seg is M contiguous values; y_seg / S_seg are N contiguous.
// Segments may overlap or alias freely in A/x.  Without ATOMIC_Y, y_seg ranges
// must be DISJOINT (each output row written exactly once); with ATOMIC_Y they
// may overlap freely (results are atomically summed).

template <typename T, uint32_t M, uint32_t N, uint32_t ROW_STRIDE = M,
          bool FUSE_SCALED_ADD = false, bool TRANSPOSE = false,
          bool ATOMIC_Y = false, typename IDX_T = int>
__device__ void segmented_row_strided_gemv(
    uint32_t segments,
    const IDX_T* seg_a_off, const IDX_T* seg_x_off, const IDX_T* seg_y_off,
    const T* A, const T* x, T* y, T alpha, T beta,
    const IDX_T* seg_s_off = nullptr, const T* S = nullptr, const T* scalar = nullptr)
{
    static_assert(!(ATOMIC_Y && FUSE_SCALED_ADD),
                  "FUSE_SCALED_ADD folds into a non-atomic store; not available under ATOMIC_Y");
    constexpr uint32_t OUT_ROWS = TRANSPOSE ? N : M;   // output rows per segment
    constexpr uint32_t CONTRACT = TRANSPOSE ? M : N;   // contracted dimension
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    uint32_t total = segments * OUT_ROWS;
    for (uint32_t r = rank; r < total; r += size) {
        uint32_t seg = r / OUT_ROWS;
        uint32_t out = r % OUT_ROWS;                    // output index within segment
        uint32_t a_base = static_cast<uint32_t>(seg_a_off[seg]);
        uint32_t x_base = static_cast<uint32_t>(seg_x_off[seg]);
        uint32_t y_base = static_cast<uint32_t>(seg_y_off[seg]);
        T res = static_cast<T>(0);
        #pragma unroll
        for (uint32_t k = 0; k < CONTRACT; k++) {
            // Non-transpose: A[i=out][j=k]  → A[out + k*ROW_STRIDE]
            // Transpose:     A[i=k][n=out]  → A[k   + out*ROW_STRIDE]
            uint32_t a_idx = TRANSPOSE ? (a_base + k + out * ROW_STRIDE)
                                       : (a_base + out + k * ROW_STRIDE);
            res += A[a_idx] * x[x_base + k];
        }
        if (ATOMIC_Y) {
            atomicAdd(&y[y_base + out], alpha * res);
        } else {
            T val = alpha * res + beta * y[y_base + out];
            if (FUSE_SCALED_ADD)
                val += S[static_cast<uint32_t>(seg_s_off[seg]) + out] * scalar[seg];
            y[y_base + out] = val;
        }
    }
}

// No-beta overload: y_seg = alpha * A_seg * x_seg (+ fused scaled add), or its
// transpose.  Doubles as the ATOMIC_Y entry point (atomic accumulate takes no
// beta — see the header note on beta-under-atomic).
template <typename T, uint32_t M, uint32_t N, uint32_t ROW_STRIDE = M,
          bool FUSE_SCALED_ADD = false, bool TRANSPOSE = false,
          bool ATOMIC_Y = false, typename IDX_T = int>
__device__ void segmented_row_strided_gemv(
    uint32_t segments,
    const IDX_T* seg_a_off, const IDX_T* seg_x_off, const IDX_T* seg_y_off,
    const T* A, const T* x, T* y, T alpha,
    const IDX_T* seg_s_off = nullptr, const T* S = nullptr, const T* scalar = nullptr)
{
    static_assert(!(ATOMIC_Y && FUSE_SCALED_ADD),
                  "FUSE_SCALED_ADD folds into a non-atomic store; not available under ATOMIC_Y");
    constexpr uint32_t OUT_ROWS = TRANSPOSE ? N : M;
    constexpr uint32_t CONTRACT = TRANSPOSE ? M : N;
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    uint32_t total = segments * OUT_ROWS;
    for (uint32_t r = rank; r < total; r += size) {
        uint32_t seg = r / OUT_ROWS;
        uint32_t out = r % OUT_ROWS;
        uint32_t a_base = static_cast<uint32_t>(seg_a_off[seg]);
        uint32_t x_base = static_cast<uint32_t>(seg_x_off[seg]);
        uint32_t y_base = static_cast<uint32_t>(seg_y_off[seg]);
        T res = static_cast<T>(0);
        #pragma unroll
        for (uint32_t k = 0; k < CONTRACT; k++) {
            uint32_t a_idx = TRANSPOSE ? (a_base + k + out * ROW_STRIDE)
                                       : (a_base + out + k * ROW_STRIDE);
            res += A[a_idx] * x[x_base + k];
        }
        if (ATOMIC_Y) {
            atomicAdd(&y[y_base + out], alpha * res);
        } else {
            T val = alpha * res;
            if (FUSE_SCALED_ADD)
                val += S[static_cast<uint32_t>(seg_s_off[seg]) + out] * scalar[seg];
            y[y_base + out] = val;
        }
    }
}
