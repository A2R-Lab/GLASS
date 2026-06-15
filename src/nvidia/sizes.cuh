#pragma once
/**
 * @file sizes.cuh
 * @brief Reference list of pre-instantiated cuBLASDx sizes for glass::nvidia gemm/gemv.
 *
 * Documents the square shapes (4, 6, 8, 12, 14, 24, 64) the benchmark suite
 * pre-instantiates. Defines no functions itself — to add a new size, call
 * DEFINE_NVIDIA_GEMM(M,N,K) / DEFINE_NVIDIA_GEMV(M,N) in your `.cu` and launch
 * with the thread count + smem reported by the host-side query helpers.
 */
// Pre-instantiated cuBLASDx sizes for glass::nvidia::gemm / gemv.
// Sizes below match the benchmark suite. To add a new size:
//   1. Call DEFINE_NVIDIA_GEMM(M, N, K) or DEFINE_NVIDIA_GEMV(M, N) in your .cu file.
//   2. Launch with the thread count and smem from GEMM::block_dim and
//      glass::nvidia::gemm_smem_size<T,M,N,K>().
//
// All functions are __device__-only and require compile-time sizes.

// Benchmark sizes (square): 4, 6, 8, 12, 14, 24, 64
