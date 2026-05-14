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
template <typename T, uint32_t M, uint32_t N, uint32_t K,
          uint32_t BATCH, uint32_t TC, uint32_t BLOCK_THREADS>
constexpr bool gemm_batched_1d_block_threads_valid()
{
    return BLOCK_THREADS >= TC * BATCH;
}
