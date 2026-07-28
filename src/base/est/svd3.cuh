#pragma once
#include <cstdint>

// ─── 3x3 spectral kit: eig3 / svd3 / closest_rotation (estimation family) ────
//
// The estimation/registration primitives every GPU perception-and-fitting
// stack hand-rolls: symmetric 3x3 eigendecomposition, general 3x3 SVD, and
// the closest-rotation projection (= the polar decomposition's rotation
// factor with the det fix = the Kabsch/Wahba/Umeyama best-fit rotation).
// Batched ICP, point-cloud alignment, covariance ellipsoids, inertia
// principal axes, and rotation-matrix re-orthonormalization are all this one
// kit applied per-problem.
//
// DESIGN — deterministic serial cores, all three tiers:
//   * `eig3` is a FIXED-SWEEP serial cyclic Jacobi — the 3x3 sibling of the
//     block-scope `glass::eigh` (same rotation formulas, same fixed-sweep
//     no-convergence-check policy, `eigh_sweeps<T>()` sweeps of the pair
//     cycle (0,1),(0,2),(1,2)). Fixed schedule + fixed sweeps ⇒ bit-identical
//     across thread counts and runs.
//   * `svd3` routes through `eig3(AᵀA)` — 3x3-appropriate and branch-light.
//     PRECISION NOTE: the AᵀA squaring means small singular values are
//     accurate to ~‖A‖·√ε (f32: ~3e-4·‖A‖) and directions degrade as σ
//     approaches that floor; the rank cutoff below reflects it. Use f64 where
//     tight small-σ accuracy matters.
//   * Rank-deficient inputs are completed deterministically: left vectors
//     with `σ_k ≤ tol·σ_0` (tol = 1e-3 f32 / 1e-7 f64) are rebuilt by
//     Gram-Schmidt / cross-product completion (an arbitrary-but-fixed
//     orthonormal choice — the standard SVD freedom).
//
// CONVENTIONS: matrices are 3x3 column-major (GLASS-wide). `eig3` returns the
// spectrum ASCENDING (`np.linalg.eigh` parity, eigenvector signs free);
// `svd3` returns singular values DESCENDING (`np.linalg.svd` parity; U/V are
// orthogonal, possibly det −1). `closest_rotation` ALWAYS returns a proper
// rotation (det +1). Outputs must not alias inputs at block/warp scope.

namespace est_detail {
    template <typename T>
    __device__ __forceinline__ T det3(const T *A) {
        return A[0]*(A[4]*A[8] - A[5]*A[7])
             - A[3]*(A[1]*A[8] - A[2]*A[7])
             + A[6]*(A[1]*A[5] - A[2]*A[4]);
    }

    // serial core: A = V·diag(W)·Vᵀ for SYMMETRIC A; W ascending, V columns
    // paired with W. Fixed-sweep cyclic Jacobi (formulas exactly as eigh.cuh:
    // theta/t/c/s incl. the theta==0 → t=1 branch and the apq==0 skip).
    template <typename T>
    __device__ __forceinline__ void eig3_core(const T *A, T *W, T *V) {
        T B[9];
        #pragma unroll
        for (uint32_t i = 0; i < 9; ++i) {
            B[i] = A[i];
            V[i] = (i % 4 == 0) ? static_cast<T>(1) : static_cast<T>(0);
        }
        constexpr uint32_t P[3] = {0u, 0u, 1u}, Q[3] = {1u, 2u, 2u};
        const uint32_t sweeps = eigh_sweeps<T>();
        for (uint32_t sw = 0; sw < sweeps; ++sw) {
            for (uint32_t k = 0; k < 3; ++k) {
                const uint32_t p = P[k], q = Q[k];
                const T apq = B[p + q*3];
                if (apq == static_cast<T>(0)) continue;
                const T theta = (B[q + q*3] - B[p + p*3]) / (static_cast<T>(2)*apq);
                const T at = (theta < static_cast<T>(0)) ? -theta : theta;
                T t = static_cast<T>(1) / (at + sqrt(static_cast<T>(1) + theta*theta));
                if (theta < static_cast<T>(0)) t = -t;
                if (theta == static_cast<T>(0)) t = static_cast<T>(1);
                const T c = static_cast<T>(1) / sqrt(static_cast<T>(1) + t*t);
                const T s = t*c;
                #pragma unroll
                for (uint32_t j = 0; j < 3; ++j) {   // row rotation
                    const T bpj = B[p + j*3], bqj = B[q + j*3];
                    B[p + j*3] = c*bpj - s*bqj;
                    B[q + j*3] = s*bpj + c*bqj;
                }
                #pragma unroll
                for (uint32_t i = 0; i < 3; ++i) {   // column rotation + V accumulation
                    const T bip = B[i + p*3], biq = B[i + q*3];
                    B[i + p*3] = c*bip - s*biq;
                    B[i + q*3] = s*bip + c*biq;
                    const T vip = V[i + p*3], viq = V[i + q*3];
                    V[i + p*3] = c*vip - s*viq;
                    V[i + q*3] = s*vip + c*viq;
                }
            }
        }
        W[0] = B[0]; W[1] = B[4]; W[2] = B[8];
        // ascending sort with eigenvector column swaps (3-element bubble; ties stable)
        #pragma unroll
        for (uint32_t pass = 0; pass < 3; ++pass) {
            const uint32_t a = pass & 1u, b = a + 1u;   // (0,1), (1,2), (0,1)
            if (W[a] > W[b]) {
                const T tw = W[a]; W[a] = W[b]; W[b] = tw;
                #pragma unroll
                for (uint32_t i = 0; i < 3; ++i) {
                    const T tv = V[i + a*3]; V[i + a*3] = V[i + b*3]; V[i + b*3] = tv;
                }
            }
        }
    }

    template <typename T>
    __device__ __forceinline__ T dot3(const T *a, const T *b)
    { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }

    // A deterministic unit vector orthogonal to unit u: cross with the axis of
    // u's smallest component (never near-parallel).
    template <typename T>
    __device__ __forceinline__ void perp3(const T *u, T *out) {
        const T ax = (u[0] < static_cast<T>(0)) ? -u[0] : u[0];
        const T ay = (u[1] < static_cast<T>(0)) ? -u[1] : u[1];
        const T az = (u[2] < static_cast<T>(0)) ? -u[2] : u[2];
        T e[3] = {static_cast<T>(0), static_cast<T>(0), static_cast<T>(0)};
        e[(ax <= ay && ax <= az) ? 0u : (ay <= az ? 1u : 2u)] = static_cast<T>(1);
        // out = u × e, normalized
        out[0] = u[1]*e[2] - u[2]*e[1];
        out[1] = u[2]*e[0] - u[0]*e[2];
        out[2] = u[0]*e[1] - u[1]*e[0];
        const T inv = static_cast<T>(1) / sqrt(dot3(out, out));
        out[0] *= inv; out[1] *= inv; out[2] *= inv;
    }

    // serial core: A = U·diag(S)·Vᵀ, S descending (see file header for the
    // AᵀA route, precision note, and deficient-rank completion policy).
    template <typename T>
    __device__ __forceinline__ void svd3_core(const T *A, T *U, T *S, T *V) {
        const T zero = static_cast<T>(0);
        // B = AᵀA (symmetric; column dots of A)
        T B[9];
        #pragma unroll
        for (uint32_t j = 0; j < 3; ++j)
            #pragma unroll
            for (uint32_t i = j; i < 3; ++i) {
                const T d = dot3(A + i*3, A + j*3);
                B[i + j*3] = d; B[j + i*3] = d;
            }
        T lam[3], Vw[9];
        eig3_core(B, lam, Vw);                        // ascending
        #pragma unroll
        for (uint32_t k = 0; k < 3; ++k) {            // flip to descending
            const T lv = lam[2 - k];
            S[k] = sqrt((lv > zero) ? lv : zero);
            #pragma unroll
            for (uint32_t i = 0; i < 3; ++i) V[i + k*3] = Vw[i + (2 - k)*3];
        }
        // Left vectors: w_k = A·v_k (= σ_k·u_k), guarded + Gram-Schmidt polished.
        const T tol = (sizeof(T) == 8) ? static_cast<T>(1e-7) : static_cast<T>(1e-3);
        T w[9];
        #pragma unroll
        for (uint32_t k = 0; k < 3; ++k)
            lie_detail::mat3_vec_core(A, V + k*3, w + k*3);
        // u0
        const T n0 = sqrt(dot3(w, w));
        if (S[0] > zero && n0 > zero) {
            const T inv = static_cast<T>(1)/n0;
            U[0] = w[0]*inv; U[1] = w[1]*inv; U[2] = w[2]*inv;
        } else {
            U[0] = static_cast<T>(1); U[1] = zero; U[2] = zero;
        }
        // u1
        bool built = false;
        if (S[1] > tol*S[0]) {
            T t1[3];
            const T d0 = dot3(U, w + 3);
            t1[0] = w[3] - d0*U[0]; t1[1] = w[4] - d0*U[1]; t1[2] = w[5] - d0*U[2];
            const T nt = sqrt(dot3(t1, t1));
            if (nt > tol*S[0]) {
                const T inv = static_cast<T>(1)/nt;
                U[3] = t1[0]*inv; U[4] = t1[1]*inv; U[5] = t1[2]*inv;
                built = true;
            }
        }
        if (!built) perp3(U, U + 3);
        // u2
        built = false;
        if (S[2] > tol*S[0]) {
            T t2[3];
            const T d0 = dot3(U, w + 6), d1 = dot3(U + 3, w + 6);
            t2[0] = w[6] - d0*U[0] - d1*U[3];
            t2[1] = w[7] - d0*U[1] - d1*U[4];
            t2[2] = w[8] - d0*U[2] - d1*U[5];
            const T nt = sqrt(dot3(t2, t2));
            if (nt > tol*S[0]) {
                const T inv = static_cast<T>(1)/nt;
                U[6] = t2[0]*inv; U[7] = t2[1]*inv; U[8] = t2[2]*inv;
                built = true;
            }
        }
        if (!built) {                                  // right-handed completion
            U[6] = U[1]*U[5] - U[2]*U[4];
            U[7] = U[2]*U[3] - U[0]*U[5];
            U[8] = U[0]*U[4] - U[1]*U[3];
        }
    }

    // serial core: R = U·diag(1, 1, det(U)·det(V))·Vᵀ from svd3(A).
    template <typename T>
    __device__ __forceinline__ void closest_rotation_core(const T *A, T *R) {
        T U[9], S[3], V[9];
        svd3_core(A, U, S, V);
        const T d = (det3(U)*det3(V) < static_cast<T>(0))
                  ? static_cast<T>(-1) : static_cast<T>(1);
        #pragma unroll
        for (uint32_t j = 0; j < 3; ++j)
            #pragma unroll
            for (uint32_t i = 0; i < 3; ++i)
                R[i + j*3] = U[i]*V[j] + U[i + 3]*V[j + 3] + d*U[i + 6]*V[j + 6];
    }
} // namespace est_detail

/**
 * @brief Symmetric 3x3 eigendecomposition: `A = V·diag(W)·Vᵀ`, W ASCENDING.
 *
 * Fixed-sweep serial cyclic Jacobi (`eigh_sweeps<T>()` sweeps — the 3x3
 * sibling of `glass::eigh`; deterministic, bit-identical across thread counts
 * and runs). `A` must be symmetric; only its stored values are read (no
 * symmetrization). NumPy equivalent: `W, V = np.linalg.eigh(A)` (up to
 * eigenvector signs).
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam TRAILING_SYNC  Emit a trailing `__syncthreads()` (default true).
 * @param A  Input symmetric 3x3 (9 elements, column-major).
 * @param W  Output eigenvalues, ascending (3 elements).
 * @param V  Output eigenvectors (9 elements, column-major; column k ↔ W[k]).
 */
template <typename T, bool TRAILING_SYNC = true>
__device__ void eig3(const T *A, T *W, T *V)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    T tw[3], tv[9];
    est_detail::eig3_core(A, tw, tv);
    quat_detail::copy_out<T, 3>(rank, size, tw, W);
    quat_detail::copy_out<T, 9>(rank, size, tv, V);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief General 3x3 SVD: `A = U·diag(S)·Vᵀ`, S DESCENDING (σ₁ ≥ σ₂ ≥ σ₃ ≥ 0).
 *
 * Routed through `eig3(AᵀA)` with guarded left-vector recovery and
 * deterministic Gram-Schmidt / cross-product completion for rank-deficient
 * inputs (see the file header; note the √ε small-σ precision floor of the
 * AᵀA squaring — prefer f64 for tight small-σ needs). `U`/`V` are orthogonal
 * but may have det −1 (standard SVD semantics); `closest_rotation` applies
 * the proper-rotation det fix. NumPy equivalent:
 * `U, S, Vt = np.linalg.svd(A)` (up to paired column signs).
 *
 * @tparam T,TRAILING_SYNC  See `eig3`.
 * @param A  Input 3x3 (9 elements, column-major; any matrix).
 * @param U  Output left singular vectors (9 elements, column-major).
 * @param S  Output singular values, descending (3 elements).
 * @param V  Output right singular vectors (9 elements, column-major, NOT transposed).
 */
template <typename T, bool TRAILING_SYNC = true>
__device__ void svd3(const T *A, T *U, T *S, T *V)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    T tu[9], ts[3], tv[9];
    est_detail::svd3_core(A, tu, ts, tv);
    quat_detail::copy_out<T, 9>(rank, size, tu, U);
    quat_detail::copy_out<T, 3>(rank, size, ts, S);
    quat_detail::copy_out<T, 9>(rank, size, tv, V);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief Closest proper rotation to a 3x3 matrix (Frobenius sense):
 *        `R = U·diag(1, 1, det(U)·det(V))·Vᵀ` from `svd3(A)`.
 *
 * ONE op, three classic jobs:
 *   1. Re-orthonormalize a drifted rotation matrix (`A ≈ R` after integration).
 *   2. The polar decomposition's rotation factor (`A = R·S`, S symmetric PSD
 *      when det(A) ≥ 0).
 *   3. The Kabsch / Wahba / Umeyama best-fit rotation: feed the cross
 *      covariance `M = Σ b_i·a_iᵀ` (centered correspondences, as `glass::ger`
 *      accumulations) and `R = argmin_R Σ‖b_i − R·a_i‖²`.
 * The det fix guarantees `R ∈ SO(3)` (det +1) for ANY input including
 * det(A) < 0. Unique when σ₂ + σ₃ > 0 (i.e. rank ≥ 2 and not the degenerate
 * reflection tie); the deterministic completion picks a fixed representative
 * otherwise.
 *
 * @tparam T,TRAILING_SYNC  See `eig3`.
 * @param A  Input 3x3 (9 elements, column-major).
 * @param R  Output rotation (9 elements, column-major, det +1; no aliasing).
 */
template <typename T, bool TRAILING_SYNC = true>
__device__ void closest_rotation(const T *A, T *R)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    T tr[9];
    est_detail::closest_rotation_core(A, tr);
    quat_detail::copy_out<T, 9>(rank, size, tr, R);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

// ─── single-thread 3x3 spectral ops ──────────────────────────────────────────
namespace thread {
    /** @brief Single-thread symmetric 3x3 eigendecomposition. See `glass::eig3`. */
    template <typename T>
    __device__ void eig3(const T *A, T *W, T *V)
    {
        T tw[3], tv[9];
        est_detail::eig3_core(A, tw, tv);
        for (uint32_t i = 0; i < 3; i++) W[i] = tw[i];
        for (uint32_t i = 0; i < 9; i++) V[i] = tv[i];
    }

    /** @brief Single-thread 3x3 SVD. See `glass::svd3`. */
    template <typename T>
    __device__ void svd3(const T *A, T *U, T *S, T *V)
    {
        T tu[9], ts[3], tv[9];
        est_detail::svd3_core(A, tu, ts, tv);
        for (uint32_t i = 0; i < 9; i++) U[i] = tu[i];
        for (uint32_t i = 0; i < 3; i++) S[i] = ts[i];
        for (uint32_t i = 0; i < 9; i++) V[i] = tv[i];
    }

    /** @brief Single-thread closest proper rotation. See `glass::closest_rotation`. */
    template <typename T>
    __device__ void closest_rotation(const T *A, T *R)
    {
        T tr[9];
        est_detail::closest_rotation_core(A, tr);
        for (uint32_t i = 0; i < 9; i++) R[i] = tr[i];
    }
}

// ─── single-warp 3x3 spectral ops ────────────────────────────────────────────
namespace warp {
    /** @brief Single-warp symmetric 3x3 eigendecomposition. See `glass::eig3`. */
    template <typename T>
    __device__ void eig3(const T *A, T *W, T *V)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        T tw[3], tv[9];
        est_detail::eig3_core(A, tw, tv);
        quat_detail::copy_out<T, 3>(lane, 32u, tw, W);
        quat_detail::copy_out<T, 9>(lane, 32u, tv, V);
        __syncwarp();
    }

    /** @brief Single-warp 3x3 SVD. See `glass::svd3`. */
    template <typename T>
    __device__ void svd3(const T *A, T *U, T *S, T *V)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        T tu[9], ts[3], tv[9];
        est_detail::svd3_core(A, tu, ts, tv);
        quat_detail::copy_out<T, 9>(lane, 32u, tu, U);
        quat_detail::copy_out<T, 3>(lane, 32u, ts, S);
        quat_detail::copy_out<T, 9>(lane, 32u, tv, V);
        __syncwarp();
    }

    /** @brief Single-warp closest proper rotation. See `glass::closest_rotation`. */
    template <typename T>
    __device__ void closest_rotation(const T *A, T *R)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        T tr[9];
        est_detail::closest_rotation_core(A, tr);
        quat_detail::copy_out<T, 9>(lane, 32u, tr, R);
        __syncwarp();
    }
}
