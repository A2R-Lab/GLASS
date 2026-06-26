#pragma once
#include <cstdint>

/**
 * @file block_access.cuh
 * @brief Dense d×d block ↔ block-tridiagonal `[L|D|R]` strip movers
 *        (`store_block` / `load_block`).
 *
 * GLASS owns the block-tridiagonal `[L|D|R]` strip layout used by `bdmv` / `pcg`:
 * each block-row is a `d × (3*d)` **row-major** tile, so the `(y,x)` entry of the
 * slot-`s` block lives at `strip[y*band_width + s*d + x]` (`band_width = 3*d`).
 * Consumers (e.g. GATO's Schur assembly) repeatedly hand-roll the transpose /
 * negate remap between a dense d×d block and this strip; these two functions are
 * that remap, once, correctly.
 *
 * Each `(y,x)` element is written by exactly one thread/lane, so there is no race
 * and the ops are trivially thread-count invariant. Block + `warp::`.
 *
 * `TRANSPOSE=false` moves the dense block as `src[y*d + x]` (the "store it as laid
 * out" case); `TRANSPOSE=true` moves `src[x*d + y]` (store/load the transpose —
 * GATO's `phi_k` vs `phi_kᵀ`). `scale` folds in a multiplier (`-1` negates, GATO's
 * common case). `store_block` writes the strip from a dense block; `load_block` is
 * the exact inverse (strip → dense). With matching `TRANSPOSE`/`scale=1`,
 * `load_block ∘ store_block` is the identity.
 */

/**
 * @brief Which sub-block of a `[L|D|R]` block-row strip a mover targets.
 *
 * Column offset within the row-major strip: `LEFT` at `0`, `MAIN` at `d`,
 * `RIGHT` at `2*d` (matching `bdmv`'s `[L|D|R]` order).
 */
enum class BandSlot : uint32_t { LEFT = 0, MAIN = 1, RIGHT = 2 };

/**
 * @brief Store a dense d×d block into a `[L|D|R]` strip slot:
 *        `strip[y*band_width + slot*d + x] = scale * src[(TRANSPOSE? x*d+y : y*d+x)]`.
 *
 * @tparam T             Scalar type (e.g. `float`, `double`).
 * @tparam d             Block dimension (the `L`/`D`/`R` blocks are `d×d`).
 * @tparam band_width    Row length of the strip (default `3*d`).
 * @tparam TRANSPOSE     Store the transpose of the dense block (default false).
 * @tparam TRAILING_SYNC Emit a trailing `__syncthreads()` (default true).
 * @param dst_strip  Start of the block-row strip (row-major, row length `band_width`).
 * @param slot       Which sub-block (`LEFT`/`MAIN`/`RIGHT`) to write.
 * @param src        Dense d×d source block (read-only).
 * @param scale      Multiplier folded into the store (e.g. `-1` to negate).
 */
template <typename T, uint32_t d, uint32_t band_width = 3 * d,
          bool TRANSPOSE = false, bool TRAILING_SYNC = true>
__device__ void store_block(T* dst_strip, BandSlot slot, const T* src, T scale = T(1))
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    const uint32_t off = static_cast<uint32_t>(slot) * d;
    for (uint32_t k = rank; k < d * d; k += size) {
        uint32_t x = k % d, y = k / d;
        dst_strip[y * band_width + off + x] = scale * src[TRANSPOSE ? (x * d + y) : (y * d + x)];
    }
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief Load a dense d×d block from a `[L|D|R]` strip slot (inverse of `store_block`):
 *        `dst[(TRANSPOSE? x*d+y : y*d+x)] = scale * strip[y*band_width + slot*d + x]`.
 *
 * @tparam T             Scalar type (e.g. `float`, `double`).
 * @tparam d             Block dimension.
 * @tparam band_width    Row length of the strip (default `3*d`).
 * @tparam TRANSPOSE     Load the transpose (default false).
 * @tparam TRAILING_SYNC Emit a trailing `__syncthreads()` (default true).
 * @param dst        Dense d×d destination block (overwritten).
 * @param src_strip  Start of the block-row strip (row-major, row length `band_width`).
 * @param slot       Which sub-block (`LEFT`/`MAIN`/`RIGHT`) to read.
 * @param scale      Multiplier folded into the load.
 */
template <typename T, uint32_t d, uint32_t band_width = 3 * d,
          bool TRANSPOSE = false, bool TRAILING_SYNC = true>
__device__ void load_block(T* dst, const T* src_strip, BandSlot slot, T scale = T(1))
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    const uint32_t off = static_cast<uint32_t>(slot) * d;
    for (uint32_t k = rank; k < d * d; k += size) {
        uint32_t x = k % d, y = k / d;
        dst[TRANSPOSE ? (x * d + y) : (y * d + x)] = scale * src_strip[y * band_width + off + x];
    }
    if constexpr (TRAILING_SYNC) __syncthreads();
}

namespace warp {
    /**
     * @brief Single-warp `store_block` — one 32-lane warp strides the `d*d` block.
     *
     * Warp form of `store_block`; `TRAILING_SYNC` gates a closing `__syncwarp()`.
     * Each element written once. Full 32 lanes required.
     * @see ::store_block
     */
    template <typename T, uint32_t d, uint32_t band_width = 3 * d,
              bool TRANSPOSE = false, bool TRAILING_SYNC = true>
    __device__ void store_block(T* dst_strip, BandSlot slot, const T* src, T scale = T(1))
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        const uint32_t off = static_cast<uint32_t>(slot) * d;
        for (uint32_t k = lane; k < d * d; k += 32) {
            uint32_t x = k % d, y = k / d;
            dst_strip[y * band_width + off + x] = scale * src[TRANSPOSE ? (x * d + y) : (y * d + x)];
        }
        if constexpr (TRAILING_SYNC) __syncwarp();
    }

    /**
     * @brief Single-warp `load_block` — one 32-lane warp strides the `d*d` block.
     *
     * Warp form of `load_block`; `TRAILING_SYNC` gates a closing `__syncwarp()`.
     * Full 32 lanes required.
     * @see ::load_block
     */
    template <typename T, uint32_t d, uint32_t band_width = 3 * d,
              bool TRANSPOSE = false, bool TRAILING_SYNC = true>
    __device__ void load_block(T* dst, const T* src_strip, BandSlot slot, T scale = T(1))
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        const uint32_t off = static_cast<uint32_t>(slot) * d;
        for (uint32_t k = lane; k < d * d; k += 32) {
            uint32_t x = k % d, y = k / d;
            dst[TRANSPOSE ? (x * d + y) : (y * d + x)] = scale * src_strip[y * band_width + off + x];
        }
        if constexpr (TRAILING_SYNC) __syncwarp();
    }
}
