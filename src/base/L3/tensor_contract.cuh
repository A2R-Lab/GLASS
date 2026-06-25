#pragma once
#include <cstdint>

// ─── tensor ⊗ vector contractions (contraction-parallel engine consumers) ────
//
// 3-tensor / vector contractions that the serial BLAS surface cannot express in
// one call. Built on the same contraction-parallel engine as glass::gemm_reduced
// (one warp owns one output, its lanes split the contracted axis and combine via
// a warp-shuffle reduce), and identically thread-count invariant: bit-identical
// at any block size, with a sub-warp register path that reproduces the
// warp-shuffle rounding exactly (see gemm_reduced.cuh / reduced_tree32).
//
// Requires gemm_reduced.cuh (reduced_tree32 + glass::warp::reduce) included first
// — guaranteed by glass.cuh include order.

// Which tensor axis is contracted away by tensor_vec_contract.
enum class TensorAxis { K, A, B };

namespace detail {
    // Output / contraction dims for a (K,A,B) tensor contracted on CONTRACT.
    template <TensorAxis C, uint32_t K, uint32_t A, uint32_t B>
    struct tvc_dims {
        // OUT0 x OUT1 output (column-major), contracted length CDIM, vector len CDIM.
        static constexpr uint32_t OUT0 = (C == TensorAxis::K) ? A : K;
        static constexpr uint32_t OUT1 = (C == TensorAxis::K) ? B : (C == TensorAxis::A ? B : A);
        static constexpr uint32_t CDIM = (C == TensorAxis::K) ? K : (C == TensorAxis::A ? A : B);
    };

    // One contraction term v[c] * Tns[k,a,b] for output coords (o0,o1) and
    // contracted index c, resolving (k,a,b) from CONTRACT and the tensor layout.
    template <typename T, TensorAxis C, uint32_t K, uint32_t A, uint32_t B, bool TIN_ROW_MAJOR>
    __device__ __forceinline__ T tvc_term(const T* Tns, const T* v,
                                           uint32_t o0, uint32_t o1, uint32_t c)
    {
        uint32_t k, a, b;
        if (C == TensorAxis::K)      { k = c;  a = o0; b = o1; }
        else if (C == TensorAxis::A) { k = o0; a = c;  b = o1; }
        else                         { k = o0; a = o1; b = c;  }
        const uint32_t idx = k*A*B + (TIN_ROW_MAJOR ? (a*B + b) : (a + b*A));
        return v[c] * Tns[idx];
    }

    // Shared engine: reduce the contracted axis for each output element.
    template <typename T, TensorAxis C, uint32_t K, uint32_t A, uint32_t B,
              bool SYMMETRIC, bool ACCUMULATE, bool TIN_ROW_MAJOR>
    __device__ void tvc_impl(uint32_t rank, uint32_t size,
                             const T* Tns, const T* v, T* Mout)
    {
        using D = tvc_dims<C, K, A, B>;
        constexpr uint32_t OUT0 = D::OUT0, OUT1 = D::OUT1, CDIM = D::CDIM;
        constexpr uint32_t maxel = OUT0 * OUT1;
        static_assert(!SYMMETRIC || (C == TensorAxis::K && A == B),
                      "SYMMETRIC requires CONTRACT==K and a square (A==B) output");

        if (size < 32u) {
            for (uint32_t el = rank; el < maxel; el += size) {
                const uint32_t o0 = el % OUT0, o1 = el / OUT0;
                if (SYMMETRIC && o0 < o1) continue;          // canonical owner writes the mirror
                T p[32];
                #pragma unroll
                for (uint32_t vlane = 0; vlane < 32u; ++vlane) {
                    T acc = static_cast<T>(0);
                    for (uint32_t c = vlane; c < CDIM; c += 32u)
                        acc += tvc_term<T, C, K, A, B, TIN_ROW_MAJOR>(Tns, v, o0, o1, c);
                    p[vlane] = acc;
                }
                const T res = reduced_tree32<T>(p);
                const uint32_t idx = o0 + o1*OUT0;
                Mout[idx] = ACCUMULATE ? (Mout[idx] + res) : res;
                if (SYMMETRIC && o0 != o1) {
                    const uint32_t m = o1 + o0*OUT0;
                    Mout[m] = ACCUMULATE ? (Mout[m] + res) : res;
                }
            }
            return;
        }

        const uint32_t n_warps = size >> 5;
        const uint32_t warp = rank >> 5, lane = rank & 31u;
        if (warp < n_warps) {
            for (uint32_t el = warp; el < maxel; el += n_warps) {
                const uint32_t o0 = el % OUT0, o1 = el / OUT0;
                if (SYMMETRIC && o0 < o1) continue;
                T partial = static_cast<T>(0);
                for (uint32_t c = lane; c < CDIM; c += 32u)
                    partial += tvc_term<T, C, K, A, B, TIN_ROW_MAJOR>(Tns, v, o0, o1, c);
                const T res = warp::reduce<T>(partial);
                if (lane == 0) {
                    const uint32_t idx = o0 + o1*OUT0;
                    Mout[idx] = ACCUMULATE ? (Mout[idx] + res) : res;
                    if (SYMMETRIC && o0 != o1) {
                        const uint32_t m = o1 + o0*OUT0;
                        Mout[m] = ACCUMULATE ? (Mout[m] + res) : res;
                    }
                }
            }
        }
    }

    // vec_tensor_vec engine: s[k] = u^T T_k w = Σ_{a,b} u[a] T[k,a,b] w[b].
    template <typename T, uint32_t K, uint32_t A, uint32_t B, bool ACCUMULATE, bool TIN_ROW_MAJOR>
    __device__ void vtv_impl(uint32_t rank, uint32_t size,
                             const T* Tns, const T* u, const T* w, T* s)
    {
        constexpr uint32_t AB = A * B;     // flattened (a,b) contraction length
        if (size < 32u) {
            for (uint32_t k = rank; k < K; k += size) {
                T p[32];
                #pragma unroll
                for (uint32_t vlane = 0; vlane < 32u; ++vlane) {
                    T acc = static_cast<T>(0);
                    for (uint32_t ab = vlane; ab < AB; ab += 32u) {
                        const uint32_t a = ab % A, b = ab / A;
                        const uint32_t idx = k*A*B + (TIN_ROW_MAJOR ? (a*B + b) : (a + b*A));
                        acc += u[a] * Tns[idx] * w[b];
                    }
                    p[vlane] = acc;
                }
                const T res = reduced_tree32<T>(p);
                s[k] = ACCUMULATE ? (s[k] + res) : res;
            }
            return;
        }
        const uint32_t n_warps = size >> 5;
        const uint32_t warp = rank >> 5, lane = rank & 31u;
        if (warp < n_warps) {
            for (uint32_t k = warp; k < K; k += n_warps) {
                T partial = static_cast<T>(0);
                for (uint32_t ab = lane; ab < AB; ab += 32u) {
                    const uint32_t a = ab % A, b = ab / A;
                    const uint32_t idx = k*A*B + (TIN_ROW_MAJOR ? (a*B + b) : (a + b*A));
                    partial += u[a] * Tns[idx] * w[b];
                }
                const T res = warp::reduce<T>(partial);
                if (lane == 0) s[k] = ACCUMULATE ? (s[k] + res) : res;
            }
        }
    }
} // namespace detail

/**
 * @brief Tensor ⊗ vector contraction: `Mout (+)= Σ_c v[c] · T[..c..]`.
 *
 * Contracts a `(K, A, B)` tensor against a vector along one axis (default the
 * leading `K` axis), producing a matrix. With `CONTRACT = TensorAxis::K`:
 * `Mout[a + b*A] (+)= Σ_k v[k] · Tns[k,a,b]` — the second-order Hessian-fold
 * `Hxx += Σ_i Vx[i]·fxx[i]`. Contracting `A` or `B` instead gives a `(K,B)` or
 * `(K,A)` result. One warp owns each output and its lanes split the contracted
 * axis (warp-shuffle reduce); thread-count invariant at any block size.
 *
 * @tparam T  Scalar type.
 * @tparam K,A,B  Tensor dimensions (slabs K, each A x B).
 * @tparam CONTRACT  Axis contracted away (default K). Output is the other two axes, column-major.
 * @tparam SYMMETRIC  When the K-slabs are symmetric in (a,b): compute the lower triangle and mirror (requires CONTRACT==K, A==B).
 * @tparam ACCUMULATE  Add into Mout (true, default) vs overwrite (false).
 * @tparam TIN_ROW_MAJOR  Each tensor slab is row-major `a*B+b` (true) vs column-major `a+b*A` (false, default).
 * @tparam TRAILING_SYNC  Emit a trailing `__syncthreads()` (default true).
 * @param Tns  Input tensor (K slabs of A x B).
 * @param v    Contraction vector (length = contracted-axis size).
 * @param Mout In/out result matrix (column-major; read only when ACCUMULATE).
 */
template <typename T, uint32_t K, uint32_t A, uint32_t B,
          TensorAxis CONTRACT = TensorAxis::K, bool SYMMETRIC = false,
          bool ACCUMULATE = true, bool TIN_ROW_MAJOR = false, bool TRAILING_SYNC = true>
__device__ void tensor_vec_contract(const T* Tns, const T* v, T* Mout)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    detail::tvc_impl<T, CONTRACT, K, A, B, SYMMETRIC, ACCUMULATE, TIN_ROW_MAJOR>(rank, size, Tns, v, Mout);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief Vector–tensor–vector triple product: `s[k] (+)= u^T · T_k · w`.
 *
 * For each slab `k` of a `(K, A, B)` tensor, forms the bilinear form
 * `s[k] = Σ_{a,b} u[a] · Tns[k,a,b] · w[b]` (second-order curvature along each
 * mode). One warp owns each `s[k]` and its lanes split the flattened `(a,b)`
 * contraction; thread-count invariant at any block size.
 *
 * @tparam T  Scalar type.
 * @tparam K,A,B  Tensor dimensions (slabs K, each A x B).
 * @tparam ACCUMULATE  Add into s (false, default = overwrite).
 * @tparam TIN_ROW_MAJOR  Each tensor slab row-major `a*B+b` (true) vs column-major `a+b*A` (false, default).
 * @tparam TRAILING_SYNC  Emit a trailing `__syncthreads()` (default true).
 * @param Tns  Input tensor (K slabs of A x B).
 * @param u    Left vector (length A).
 * @param w    Right vector (length B).
 * @param s    In/out result vector (length K; read only when ACCUMULATE).
 */
template <typename T, uint32_t K, uint32_t A, uint32_t B,
          bool ACCUMULATE = false, bool TIN_ROW_MAJOR = false, bool TRAILING_SYNC = true>
__device__ void vec_tensor_vec(const T* Tns, const T* u, const T* w, T* s)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    detail::vtv_impl<T, K, A, B, ACCUMULATE, TIN_ROW_MAJOR>(rank, size, Tns, u, w, s);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

// ─── single-warp tensor contractions ─────────────────────────────────────────
namespace warp {
    /**
     * @brief Single-warp tensor ⊗ vector contraction: `Mout (+)= Σ_c v[c] · T[..c..]`.
     *
     * Warp-per-problem analogue of `glass::tensor_vec_contract`; one full 32-lane
     * warp performs the whole contraction. See the block version for semantics.
     *
     * @tparam T,K,A,B,CONTRACT,SYMMETRIC,ACCUMULATE,TIN_ROW_MAJOR  See glass::tensor_vec_contract.
     * @tparam TRAILING_SYNC  Emit a trailing `__syncwarp()` (default true).
     * @param Tns,v,Mout  See glass::tensor_vec_contract.
     */
    template <typename T, uint32_t K, uint32_t A, uint32_t B,
              TensorAxis CONTRACT = TensorAxis::K, bool SYMMETRIC = false,
              bool ACCUMULATE = true, bool TIN_ROW_MAJOR = false, bool TRAILING_SYNC = true>
    __device__ void tensor_vec_contract(const T* Tns, const T* v, T* Mout)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31u;
        detail::tvc_impl<T, CONTRACT, K, A, B, SYMMETRIC, ACCUMULATE, TIN_ROW_MAJOR>(lane, 32u, Tns, v, Mout);
        if constexpr (TRAILING_SYNC) __syncwarp();
    }

    /**
     * @brief Single-warp vector–tensor–vector triple product: `s[k] (+)= u^T · T_k · w`.
     *
     * Warp-per-problem analogue of `glass::vec_tensor_vec`.
     *
     * @tparam T,K,A,B,ACCUMULATE,TIN_ROW_MAJOR  See glass::vec_tensor_vec.
     * @tparam TRAILING_SYNC  Emit a trailing `__syncwarp()` (default true).
     * @param Tns,u,w,s  See glass::vec_tensor_vec.
     */
    template <typename T, uint32_t K, uint32_t A, uint32_t B,
              bool ACCUMULATE = false, bool TIN_ROW_MAJOR = false, bool TRAILING_SYNC = true>
    __device__ void vec_tensor_vec(const T* Tns, const T* u, const T* w, T* s)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31u;
        detail::vtv_impl<T, K, A, B, ACCUMULATE, TIN_ROW_MAJOR>(lane, 32u, Tns, u, w, s);
        if constexpr (TRAILING_SYNC) __syncwarp();
    }
}
