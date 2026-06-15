#pragma once
#include <cstdint>
#include <cstdio>
#include <type_traits>
#include "./tuning_table.cuh"

// glass::nvidia query helpers that do NOT depend on cuBLASDx — safe to
// include in builds without the cuBLASDx headers. Companion to query.cuh
// (which provides cuBLASDx-dependent helpers like gemm_min_block_threads).

#ifndef SMS
#define SMS 860
#endif

// -- backend dispatch (P1-3) ------------------------------------------------

// Returns true iff cuBLASDx is expected to outperform the SIMT path for the
// given (T, M, N, K) on SM_VAL. Consults the per-SM lookup table in
// tuning_table.cuh; falls back to a conservative shape heuristic for shapes
// not measured for that SM.
//
// Used by glass::nvidia::gemm to decide whether to dispatch to cuBLASDx
// (requires a DEFINE_NVIDIA_GEMM* specialization) or fall through to the
// SIMT base path (no DEFINE needed). Today only T==float is tuned; other
// types always return false (SIMT).
//
// To regenerate tuning_table.cuh for your hardware:
//   python bench/autotune.py --sm AUTO --out src/nvidia/tuning_table.cuh
/**
 * @brief Compile-time backend decision for `gemm`: cuBLASDx vs SIMT (host-callable).
 *
 * Returns true iff cuBLASDx is expected to beat the SIMT path for this
 * (T,M,N,K) on SM_VAL. Consults the per-SM tuning_table.cuh, falling back to a
 * conservative shape heuristic for unmeasured shapes. Drives the gemm<>
 * primary template's dispatch. Only T==float is tuned; other types return
 * false. No cuBLASDx dependency. constexpr.
 *
 * @tparam T      Scalar type.
 * @tparam M      Rows of A / C.
 * @tparam N      Columns of B / C.
 * @tparam K      Inner dimension.
 * @tparam SM_VAL Target SM architecture (default = SMS).
 * @return true to route to cuBLASDx, false for SIMT.
 */
template <typename T, uint32_t M, uint32_t N, uint32_t K,
          uint32_t SM_VAL = SMS>
constexpr bool should_use_cublasdx() {
    if constexpr (!std::is_same<T, float>::value) {
        return false;
    } else {
        // tuning_table.cuh declares `namespace _glass_tuning` at the include
        // site (inside glass::nvidia), so the unqualified name resolves here.
        return _glass_tuning::cublasdx_wins<M, N, K, SM_VAL>();
    }
}

// Diagnostic helper. Prints which backend
// `glass::nvidia::gemm<T,M,N,K,...,SM_VAL>` will dispatch to. Callable from
// host or device (printf works in CUDA device code).
//
//   glass::nvidia::print_dispatch<float, 4, 4, 4>();
//   // -> "glass::nvidia::gemm<T,4,4,4,SM=860>: SIMT fallback"
//
// Useful when debugging "why is my GEMM slow" or "why does the linker complain
// that gemm<float,32,32,32> is undefined" (answer: the heuristic identified it
// as cuBLASDx territory and you need to call DEFINE_NVIDIA_GEMM(32,32,32)).
/**
 * @brief Print which backend `gemm<T,M,N,K,...>` dispatches to (host/device).
 *
 * Diagnostic helper: prints "cuBLASDx (needs DEFINE_NVIDIA_GEMM*)" or "SIMT
 * fallback" for the given shape. Callable from host or device. Useful when
 * debugging an undefined-symbol or unexpectedly-slow GEMM.
 *
 * @tparam T      Scalar type.
 * @tparam M      Rows of A / C.
 * @tparam N      Columns of B / C.
 * @tparam K      Inner dimension.
 * @tparam SM_VAL Target SM architecture (default = SMS).
 */
template <typename T, uint32_t M, uint32_t N, uint32_t K,
          uint32_t SM_VAL = SMS>
__host__ __device__ inline void print_dispatch() {
    printf("glass::nvidia::gemm<T,%u,%u,%u,SM=%u>: %s\n",
           M, N, K, SM_VAL,
           should_use_cublasdx<T, M, N, K, SM_VAL>()
               ? "cuBLASDx (needs DEFINE_NVIDIA_GEMM*)"
               : "SIMT fallback");
}

// -- gemm_batched_1d queries (SIMT-only) ------------------------------------

// True iff BLOCK_THREADS is at least TC*BATCH (the minimum for the SIMT-only
// 1D-launch batched GEMM). Unlike gemm_block_threads_valid, this does not
// consult cuBLASDx — the requirement is purely "enough threads to give every
// batch element TC of them".
/**
 * @brief True iff BLOCK_THREADS suffices for `gemm_batched_1d` (host-callable).
 *
 * Returns whether BLOCK_THREADS >= TC*BATCH (every batch element gets TC
 * threads). Unlike gemm_block_threads_valid this does not consult cuBLASDx —
 * the SIMT-only requirement is purely "enough threads". constexpr.
 *
 * @tparam T             Scalar type.
 * @tparam M             Rows of A / C.
 * @tparam N             Columns of B / C.
 * @tparam K             Inner dimension.
 * @tparam BATCH         Number of independent GEMMs.
 * @tparam TC            Threads per batch element.
 * @tparam BLOCK_THREADS Candidate launch thread count to validate.
 * @return true if BLOCK_THREADS >= TC*BATCH.
 */
template <typename T, uint32_t M, uint32_t N, uint32_t K,
          uint32_t BATCH, uint32_t TC, uint32_t BLOCK_THREADS>
constexpr bool gemm_batched_1d_block_threads_valid()
{
    return BLOCK_THREADS >= TC * BATCH;
}

// -- round-2 backend-dispatch siblings (Gaps A/B/C + gemm_batched_1d) -------
//
// These mirror should_use_cublasdx<> for the round-2 APIs:
//   * gemv<>             (Gap A)  → cublasdx_wins_gemv
//   * row_strided_gemv<> (Gap B)  → cublasdx_wins_row_strided_gemv
//   * row_strided_gemm<> (Gap C)  → cublasdx_wins_row_strided_gemm
//   * gemm_batched_1d    (NEW-1)  → cublasdx_wins_batched
//
// Each routes through a per-API primary template in tuning_table.cuh
// that exposes the same "default heuristic + per-shape specialization +
// optional local override" structure as the gemm decision. Today only
// T==float is tuned; other types always return false (SIMT).

/**
 * @brief Compile-time backend decision for `gemv`: cuBLASDx vs SIMT (host-callable).
 *
 * Sibling of should_use_cublasdx<> for the gemv API; consults
 * tuning_table.cuh. Only T==float is tuned; other types return false.
 * constexpr.
 *
 * @tparam T      Scalar type.
 * @tparam M      Rows of A.
 * @tparam N      Columns of A.
 * @tparam SM_VAL Target SM architecture (default = SMS).
 * @return true to route to cuBLASDx, false for SIMT.
 */
template <typename T, uint32_t M, uint32_t N, uint32_t SM_VAL = SMS>
constexpr bool should_use_cublasdx_gemv() {
    if constexpr (!std::is_same<T, float>::value) return false;
    else return _glass_tuning::cublasdx_wins_gemv<M, N, SM_VAL>();
}

/**
 * @brief Compile-time backend decision for `row_strided_gemv` (host-callable).
 *
 * Sibling of should_use_cublasdx<> for the strided-GEMV API. Only T==float is
 * tuned; other types return false. constexpr.
 *
 * @tparam T          Scalar type.
 * @tparam M          Rows of A.
 * @tparam N          Columns of A.
 * @tparam ROW_STRIDE Leading dimension (row stride) of A.
 * @tparam SM_VAL     Target SM architecture (default = SMS).
 * @return true to route to cuBLASDx, false for SIMT.
 */
template <typename T, uint32_t M, uint32_t N, uint32_t ROW_STRIDE,
          uint32_t SM_VAL = SMS>
constexpr bool should_use_cublasdx_row_strided_gemv() {
    if constexpr (!std::is_same<T, float>::value) return false;
    else return _glass_tuning::cublasdx_wins_row_strided_gemv<M, N, ROW_STRIDE, SM_VAL>();
}

/**
 * @brief Compile-time backend decision for `row_strided_gemm` (host-callable).
 *
 * Sibling of should_use_cublasdx<> for the strided-GEMM API. Only T==float is
 * tuned; other types return false. constexpr.
 *
 * @tparam T      Scalar type.
 * @tparam M      Rows of A / C.
 * @tparam N      Columns of B / C.
 * @tparam K      Inner dimension.
 * @tparam A_RS   Leading dimension (row stride) of A.
 * @tparam B_RS   Leading dimension (row stride) of B.
 * @tparam SM_VAL Target SM architecture (default = SMS).
 * @return true to route to cuBLASDx, false for SIMT.
 */
template <typename T, uint32_t M, uint32_t N, uint32_t K,
          uint32_t A_RS, uint32_t B_RS, uint32_t SM_VAL = SMS>
constexpr bool should_use_cublasdx_row_strided_gemm() {
    if constexpr (!std::is_same<T, float>::value) return false;
    else return _glass_tuning::cublasdx_wins_row_strided_gemm<M, N, K, A_RS, B_RS, SM_VAL>();
}

/**
 * @brief Compile-time backend hint for batched GEMM: cuBLASDx vs SIMT (host-callable).
 *
 * Sibling of should_use_cublasdx<> reporting whether cuBLASDx would win for a
 * batched GEMM shape (used as a tuning signal; gemm_batched_1d itself is
 * SIMT-only and does not auto-dispatch). Only T==float is tuned. constexpr.
 *
 * @tparam T      Scalar type.
 * @tparam M      Rows of A / C.
 * @tparam N      Columns of B / C.
 * @tparam K      Inner dimension.
 * @tparam BATCH  Number of independent GEMMs.
 * @tparam SM_VAL Target SM architecture (default = SMS).
 * @return true if cuBLASDx is expected to win, false for SIMT.
 */
template <typename T, uint32_t M, uint32_t N, uint32_t K, uint32_t BATCH,
          uint32_t SM_VAL = SMS>
constexpr bool should_use_cublasdx_batched() {
    if constexpr (!std::is_same<T, float>::value) return false;
    else return _glass_tuning::cublasdx_wins_batched<M, N, K, BATCH, SM_VAL>();
}

// -- per-API print_dispatch* diagnostics ------------------------------------
// HEAD's print_dispatch<> is gemm-only. These siblings report the dispatch
// decision for each round-2 API. Useful when codegen produces a mix of
// shapes and you want to know which routes which way.

/**
 * @brief Print which backend `gemv<T,M,N,...>` dispatches to (host/device).
 *
 * Diagnostic sibling of print_dispatch<> for the gemv API.
 *
 * @tparam T      Scalar type.
 * @tparam M      Rows of A.
 * @tparam N      Columns of A.
 * @tparam SM_VAL Target SM architecture (default = SMS).
 */
template <typename T, uint32_t M, uint32_t N, uint32_t SM_VAL = SMS>
__host__ __device__ inline void print_dispatch_gemv() {
    printf("glass::nvidia::gemv<T,%u,%u,SM=%u>: %s\n",
           M, N, SM_VAL,
           should_use_cublasdx_gemv<T, M, N, SM_VAL>()
               ? "cuBLASDx (needs DEFINE_NVIDIA_GEMV*)"
               : "SIMT fallback");
}

/**
 * @brief Print which backend `row_strided_gemm<...>` dispatches to (host/device).
 *
 * Diagnostic sibling of print_dispatch<> for the strided-GEMM API.
 *
 * @tparam T      Scalar type.
 * @tparam M      Rows of A / C.
 * @tparam N      Columns of B / C.
 * @tparam K      Inner dimension.
 * @tparam A_RS   Leading dimension (row stride) of A.
 * @tparam B_RS   Leading dimension (row stride) of B.
 * @tparam SM_VAL Target SM architecture (default = SMS).
 */
template <typename T, uint32_t M, uint32_t N, uint32_t K,
          uint32_t A_RS = M, uint32_t B_RS = N, uint32_t SM_VAL = SMS>
__host__ __device__ inline void print_dispatch_row_strided_gemm() {
    printf("glass::nvidia::row_strided_gemm<T,%u,%u,%u,A_RS=%u,B_RS=%u,SM=%u>: %s\n",
           M, N, K, A_RS, B_RS, SM_VAL,
           should_use_cublasdx_row_strided_gemm<T, M, N, K, A_RS, B_RS, SM_VAL>()
               ? "cuBLASDx (needs DEFINE_NVIDIA_GEMM*)"
               : "SIMT fallback");
}

/**
 * @brief Print which backend `row_strided_gemv<...>` dispatches to (host/device).
 *
 * Diagnostic sibling of print_dispatch<> for the strided-GEMV API.
 *
 * @tparam T          Scalar type.
 * @tparam M          Rows of A.
 * @tparam N          Columns of A.
 * @tparam ROW_STRIDE Leading dimension (row stride) of A.
 * @tparam SM_VAL     Target SM architecture (default = SMS).
 */
template <typename T, uint32_t M, uint32_t N, uint32_t ROW_STRIDE = M,
          uint32_t SM_VAL = SMS>
__host__ __device__ inline void print_dispatch_row_strided_gemv() {
    printf("glass::nvidia::row_strided_gemv<T,%u,%u,RS=%u,SM=%u>: %s\n",
           M, N, ROW_STRIDE, SM_VAL,
           should_use_cublasdx_row_strided_gemv<T, M, N, ROW_STRIDE, SM_VAL>()
               ? "cuBLASDx (needs DEFINE_NVIDIA_GEMV*)"
               : "SIMT fallback");
}

/**
 * @brief Print the batched-GEMM backend hint for a shape (host/device).
 *
 * Diagnostic sibling of print_dispatch<> reporting the batched-GEMM tuning
 * signal (gemm_batched_1d is SIMT-only; the 2D gemm_batched is the cuBLASDx path).
 *
 * @tparam T      Scalar type.
 * @tparam M      Rows of A / C.
 * @tparam N      Columns of B / C.
 * @tparam K      Inner dimension.
 * @tparam BATCH  Number of independent GEMMs.
 * @tparam SM_VAL Target SM architecture (default = SMS).
 */
template <typename T, uint32_t M, uint32_t N, uint32_t K, uint32_t BATCH,
          uint32_t SM_VAL = SMS>
__host__ __device__ inline void print_dispatch_batched() {
    printf("glass::nvidia::gemm_batched_1d<T,%u,%u,%u,BATCH=%u,SM=%u>: %s\n",
           M, N, K, BATCH, SM_VAL,
           should_use_cublasdx_batched<T, M, N, K, BATCH, SM_VAL>()
               ? "cuBLASDx (no auto-dispatch; see 2D gemm_batched)"
               : "SIMT fallback");
}
