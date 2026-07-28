#pragma once
#include <cstdint>

// ─── SO(3) maps and Jacobians (robotics ops, Lie family) ─────────────────────
//
// The rotation-group toolkit: hat map (`skew`), exponential (Rodrigues) and
// logarithm, and the right/left Jacobians of the exponential with their
// inverses. Every estimator (GTSAM/Sophus/manif/Pinocchio/Ceres) hand-rolls
// this family with subtly different conventions and small-angle series — and
// GPU simulators typically ship none of it. Convention here:
//
//   * 3x3 matrices are COLUMN-MAJOR (GLASS-wide).
//   * `so3_exp(φ)` is the Rodrigues formula on the FULL rotation vector φ
//     (axis·angle, radians): `R = I + a·[φ]ₓ + b·[φ]ₓ²`, a = sinθ/θ,
//     b = (1−cosθ)/θ², θ = |φ|.
//   * The RIGHT Jacobian `Jr(φ) = I − b·[φ]ₓ + c·[φ]ₓ²` (b as above,
//     c = (θ−sinθ)/θ³) maps body-frame tangent perturbations:
//     `exp(φ + δ) ≈ exp(φ)·exp(Jr(φ)·δ)`. The LEFT Jacobian
//     `Jl(φ) = Jr(−φ) = I + b·[φ]ₓ + c·[φ]ₓ²` is also the SE(3) translation
//     "V matrix": `exp_se3(ρ,φ)` translates by `Jl(φ)·ρ`.
//   * `so3_log` returns the canonical branch |φ| ≤ π. It routes through the
//     Shepperd quaternion extraction + `2·atan2(|v|, w)`, which is
//     numerically stable across the whole range INCLUDING θ near π (where the
//     naive `vee(R−Rᵀ)` formula loses the axis).
//   * Small-angle Taylor heads keep every map smooth through θ = 0 (thresholds
//     documented per function; validated against mpmath ground truth).
//
// Tiers: block/warp/thread share one serial core per op (redundant-core +
// strided copy-out; see quat.cuh's header note). Outputs must not alias inputs
// at block/warp scope.

namespace lie_detail {
    // serial core: S = [v]ₓ (3x3 column-major hat map).
    template <typename T>
    __device__ __forceinline__ void skew_core(const T *v, T *S) {
        const T zero = static_cast<T>(0);
        // column-major: S[c*3 + r] = S(r,c)
        S[0] = zero;   S[3] = -v[2];  S[6] =  v[1];
        S[1] =  v[2];  S[4] = zero;   S[7] = -v[0];
        S[2] = -v[1];  S[5] =  v[0];  S[8] = zero;
    }

    // serial core: C = A·B, 3x3 column-major (C must not alias A/B).
    template <typename T>
    __device__ __forceinline__ void mat3_mul_core(const T *A, const T *B, T *C) {
        #pragma unroll
        for (uint32_t c = 0; c < 3; ++c) {
            #pragma unroll
            for (uint32_t r = 0; r < 3; ++r) {
                C[c*3 + r] = A[r]*B[c*3] + A[3 + r]*B[c*3 + 1] + A[6 + r]*B[c*3 + 2];
            }
        }
    }

    // serial core: out = A·v, 3x3 column-major times 3-vector (out must not alias v).
    template <typename T>
    __device__ __forceinline__ void mat3_vec_core(const T *A, const T *v, T *out) {
        out[0] = A[0]*v[0] + A[3]*v[1] + A[6]*v[2];
        out[1] = A[1]*v[0] + A[4]*v[1] + A[7]*v[2];
        out[2] = A[2]*v[0] + A[5]*v[1] + A[8]*v[2];
    }

    // serial core: M = I + ca·S + cb·S² for S = [φ]ₓ — the shared shape of
    // exp / Jr / Jl (coefficients differ per map).
    template <typename T>
    __device__ __forceinline__ void rodrigues_core(const T *phi, T ca, T cb, T *M) {
        T S[9];  skew_core(phi, S);
        T S2[9]; mat3_mul_core(S, S, S2);
        #pragma unroll
        for (uint32_t i = 0; i < 9; ++i) {
            const T id = (i % 4 == 0) ? static_cast<T>(1) : static_cast<T>(0);
            M[i] = id + ca*S[i] + cb*S2[i];
        }
    }

    // Rodrigues coefficients a = sinθ/θ, b = (1−cosθ)/θ², c = (θ−sinθ)/θ³ with
    // the θ→0 Taylor heads (a→1, b→1/2, c→1/6). Threshold 1e-8 matches the
    // validated GRiD emitters this family is promoted from.
    template <typename T>
    __device__ __forceinline__ void rodrigues_coefs(T theta, T &a, T &b, T &c) {
        if (theta < static_cast<T>(1e-8)) {
            a = static_cast<T>(1);
            b = static_cast<T>(0.5);
            c = static_cast<T>(1.0/6.0);
        } else {
            const T t2 = theta*theta;
            a = sin(theta)/theta;
            b = (static_cast<T>(1) - cos(theta))/t2;
            c = (theta - sin(theta))/(t2*theta);
        }
    }

    template <typename T>
    __device__ __forceinline__ T vec3_norm(const T *v) {
        return sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    }

    // serial core: R = exp([φ]ₓ).
    template <typename T>
    __device__ __forceinline__ void so3_exp_core(const T *phi, T *R) {
        T a, b, c; rodrigues_coefs(vec3_norm(phi), a, b, c);
        rodrigues_core(phi, a, b, R);
    }

    // serial core: Jr(φ) = I − b·S + c·S².
    template <typename T>
    __device__ __forceinline__ void so3_right_jacobian_core(const T *phi, T *J) {
        T a, b, c; rodrigues_coefs(vec3_norm(phi), a, b, c);
        rodrigues_core(phi, -b, c, J);
    }

    // serial core: Jl(φ) = Jr(−φ) = I + b·S + c·S² (the SE(3) "V matrix").
    template <typename T>
    __device__ __forceinline__ void so3_left_jacobian_core(const T *phi, T *J) {
        T a, b, c; rodrigues_coefs(vec3_norm(phi), a, b, c);
        rodrigues_core(phi, b, c, J);
    }

    // Inverse-Jacobian S² coefficient e = 1/θ² − cot(θ/2)/(2θ), written in the
    // half-angle form so the only removable singularity is θ→0 (series head
    // 1/12 + θ²/720 + θ⁴/30240); the naive (1+cosθ)/(2θ sinθ) form is 0/0 at
    // θ = π, where this one is exactly 1/π². (θ → 2π is a GENUINE singularity
    // of Jr⁻¹ — out of the |φ| ≤ π canonical range this library works in.)
    template <typename T>
    __device__ __forceinline__ T inv_jacobian_coef(T theta) {
        if (theta < static_cast<T>(1e-4)) {
            const T t2 = theta*theta;
            return static_cast<T>(1.0/12.0) + t2*(static_cast<T>(1.0/720.0)
                 + t2*static_cast<T>(1.0/30240.0));
        }
        const T half = static_cast<T>(0.5)*theta;
        return static_cast<T>(1)/(theta*theta)
             - cos(half)/(static_cast<T>(2)*theta*sin(half));
    }

    // serial core: Jr(φ)⁻¹ = I + S/2 + e·S².
    template <typename T>
    __device__ __forceinline__ void so3_right_jacobian_inv_core(const T *phi, T *J) {
        const T e = inv_jacobian_coef(vec3_norm(phi));
        rodrigues_core(phi, static_cast<T>(0.5), e, J);
    }

    // serial core: Jl(φ)⁻¹ = I − S/2 + e·S².
    template <typename T>
    __device__ __forceinline__ void so3_left_jacobian_inv_core(const T *phi, T *J) {
        const T e = inv_jacobian_coef(vec3_norm(phi));
        rodrigues_core(phi, static_cast<T>(-0.5), e, J);
    }

    // serial core: φ = log(R), canonical branch |φ| ≤ π. Route through the
    // Shepperd quaternion (stable at EVERY rotation) then the quaternion log
    // (quat_detail::quat_log_core) — no near-π axis loss, no trace clamping
    // games. Shepperd already yields w >= 0, so the log's cover fold is a no-op.
    template <typename T>
    __device__ __forceinline__ void so3_log_core(const T *R, T *phi) {
        T q[4];
        quat_detail::rot_to_quat_core<T, QuatLayout::xyzw>(R, q);
        quat_detail::quat_log_core<T, QuatLayout::xyzw>(q, phi);
    }
} // namespace lie_detail

/**
 * @brief Hat map: `S = [v]ₓ` (3x3 column-major skew-symmetric matrix).
 *
 * `S·x = v × x`. The SO(3) building block under every op in this family.
 * NumPy equivalent: `np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])`
 * (flatten Fortran-order for the column-major array).
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam TRAILING_SYNC  Emit a trailing `__syncthreads()` (default true).
 * @param v  Input 3-vector.
 * @param S  Output 3x3 skew matrix (9 elements, column-major; no aliasing).
 */
template <typename T, bool TRAILING_SYNC = true>
__device__ void skew(const T *v, T *S)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    T tmp[9]; lie_detail::skew_core(v, tmp);
    quat_detail::copy_out<T, 9>(rank, size, tmp, S);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief SO(3) exponential (Rodrigues): `R = exp([φ]ₓ)`.
 *
 * `φ` is the full rotation vector (axis·angle). Taylor heads below θ = 1e-8
 * keep the map smooth through the identity. NumPy equivalent:
 * `Rotation.from_rotvec(phi).as_matrix()` (Fortran-order flatten).
 *
 * @tparam T,TRAILING_SYNC  See `skew`.
 * @param phi  Rotation vector (3 elements).
 * @param R    Output 3x3 rotation (9 elements, column-major; no aliasing).
 */
template <typename T, bool TRAILING_SYNC = true>
__device__ void so3_exp(const T *phi, T *R)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    T tmp[9]; lie_detail::so3_exp_core(phi, tmp);
    quat_detail::copy_out<T, 9>(rank, size, tmp, R);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief SO(3) logarithm: `φ = log(R)`, canonical branch `|φ| ≤ π`.
 *
 * Implemented through the Shepperd quaternion extraction plus
 * `φ = 2·atan2(|v|, w)·v̂` — numerically stable across the WHOLE range,
 * including θ near π where the textbook `θ/(2 sinθ)·vee(R−Rᵀ)` formula loses
 * the axis. Inverse of `so3_exp` (round-trip exact to floating tolerance for
 * |φ| < π; at exactly θ = π the axis sign is the canonical-cover choice).
 * NumPy equivalent: `Rotation.from_matrix(R).as_rotvec()`.
 *
 * @tparam T,TRAILING_SYNC  See `skew`.
 * @param R    Input 3x3 rotation (9 elements, column-major).
 * @param phi  Output rotation vector (3 elements; no aliasing at block/warp scope).
 */
template <typename T, bool TRAILING_SYNC = true>
__device__ void so3_log(const T *R, T *phi)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    T tmp[3]; lie_detail::so3_log_core(R, tmp);
    quat_detail::copy_out<T, 3>(rank, size, tmp, phi);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief SO(3) right Jacobian: `Jr(φ) = I − b·[φ]ₓ + c·[φ]ₓ²`.
 *
 * `b = (1−cosθ)/θ²`, `c = (θ−sinθ)/θ³`. Maps additive tangent perturbations to
 * the manifold: `exp(φ+δ) ≈ exp(φ)·exp(Jr(φ)·δ)` — the object every on-manifold
 * optimizer and covariance propagation needs. Identity: `Jr(φ) = Jl(−φ)`.
 *
 * @tparam T,TRAILING_SYNC  See `skew`.
 * @param phi  Rotation vector (3 elements).
 * @param J    Output 3x3 Jacobian (9 elements, column-major; no aliasing).
 */
template <typename T, bool TRAILING_SYNC = true>
__device__ void so3_right_jacobian(const T *phi, T *J)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    T tmp[9]; lie_detail::so3_right_jacobian_core(phi, tmp);
    quat_detail::copy_out<T, 9>(rank, size, tmp, J);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief Inverse right Jacobian: `Jr(φ)⁻¹ = I + [φ]ₓ/2 + e·[φ]ₓ²`.
 *
 * `e = 1/θ² − cot(θ/2)/(2θ)` (half-angle form — finite at θ = π, exactly 1/π²;
 * series `1/12 + θ²/720 + …` below θ = 1e-4). `Jr⁻¹` pulls manifold
 * differences back to the tangent (IMU preintegration, on-manifold GN steps).
 * Genuinely singular only at θ = 2π, outside the canonical |φ| ≤ π range.
 *
 * @tparam T,TRAILING_SYNC  See `skew`.
 * @param phi  Rotation vector (3 elements).
 * @param J    Output 3x3 inverse Jacobian (9 elements, column-major; no aliasing).
 */
template <typename T, bool TRAILING_SYNC = true>
__device__ void so3_right_jacobian_inv(const T *phi, T *J)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    T tmp[9]; lie_detail::so3_right_jacobian_inv_core(phi, tmp);
    quat_detail::copy_out<T, 9>(rank, size, tmp, J);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief SO(3) left Jacobian: `Jl(φ) = Jr(−φ) = I + b·[φ]ₓ + c·[φ]ₓ²`.
 *
 * Also the SE(3) translation "V matrix": `exp_se3(ρ,φ)` translates by
 * `Jl(φ)·ρ` — some codebases (including the GRiD emitters this is promoted
 * from) carry it under that name; it is the SAME matrix.
 *
 * @tparam T,TRAILING_SYNC  See `skew`.
 * @param phi  Rotation vector (3 elements).
 * @param J    Output 3x3 Jacobian (9 elements, column-major; no aliasing).
 */
template <typename T, bool TRAILING_SYNC = true>
__device__ void so3_left_jacobian(const T *phi, T *J)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    T tmp[9]; lie_detail::so3_left_jacobian_core(phi, tmp);
    quat_detail::copy_out<T, 9>(rank, size, tmp, J);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief Inverse left Jacobian: `Jl(φ)⁻¹ = I − [φ]ₓ/2 + e·[φ]ₓ²`.
 *
 * Same `e` coefficient (and the same θ = 2π caveat) as
 * `so3_right_jacobian_inv`; `Jl⁻¹(φ) = Jr⁻¹(−φ)`.
 *
 * @tparam T,TRAILING_SYNC  See `skew`.
 * @param phi  Rotation vector (3 elements).
 * @param J    Output 3x3 inverse Jacobian (9 elements, column-major; no aliasing).
 */
template <typename T, bool TRAILING_SYNC = true>
__device__ void so3_left_jacobian_inv(const T *phi, T *J)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    T tmp[9]; lie_detail::so3_left_jacobian_inv_core(phi, tmp);
    quat_detail::copy_out<T, 9>(rank, size, tmp, J);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

// ─── single-thread SO(3) ops ─────────────────────────────────────────────────
namespace thread {
    // One thread owns the whole 3x3/3-vector result — the same serial cores,
    // serial copy-out. No barriers, no threadIdx read; register operands fine.

    /** @brief Single-thread hat map. See `glass::skew`. */
    template <typename T>
    __device__ void skew(const T *v, T *S) { lie_detail::skew_core(v, S); }

    /** @brief Single-thread Rodrigues exponential. See `glass::so3_exp`. */
    template <typename T>
    __device__ void so3_exp(const T *phi, T *R) { lie_detail::so3_exp_core(phi, R); }

    /** @brief Single-thread SO(3) log (canonical branch). See `glass::so3_log`. */
    template <typename T>
    __device__ void so3_log(const T *R, T *phi)
    {
        T tmp[3]; lie_detail::so3_log_core(R, tmp);
        phi[0] = tmp[0]; phi[1] = tmp[1]; phi[2] = tmp[2];
    }

    /** @brief Single-thread right Jacobian. See `glass::so3_right_jacobian`. */
    template <typename T>
    __device__ void so3_right_jacobian(const T *phi, T *J)
    { lie_detail::so3_right_jacobian_core(phi, J); }

    /** @brief Single-thread inverse right Jacobian. See `glass::so3_right_jacobian_inv`. */
    template <typename T>
    __device__ void so3_right_jacobian_inv(const T *phi, T *J)
    { lie_detail::so3_right_jacobian_inv_core(phi, J); }

    /** @brief Single-thread left Jacobian (SE(3) "V matrix"). See `glass::so3_left_jacobian`. */
    template <typename T>
    __device__ void so3_left_jacobian(const T *phi, T *J)
    { lie_detail::so3_left_jacobian_core(phi, J); }

    /** @brief Single-thread inverse left Jacobian. See `glass::so3_left_jacobian_inv`. */
    template <typename T>
    __device__ void so3_left_jacobian_inv(const T *phi, T *J)
    { lie_detail::so3_left_jacobian_inv_core(phi, J); }
}

// ─── single-warp SO(3) ops ───────────────────────────────────────────────────
namespace warp {
    // One 32-lane warp owns the result: same serial cores, lane-strided
    // copy-out, `__syncwarp()` close. Outputs must not alias inputs.

    /** @brief Single-warp hat map. See `glass::skew`. */
    template <typename T>
    __device__ void skew(const T *v, T *S)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        T tmp[9]; lie_detail::skew_core(v, tmp);
        quat_detail::copy_out<T, 9>(lane, 32u, tmp, S);
        __syncwarp();
    }

    /** @brief Single-warp Rodrigues exponential. See `glass::so3_exp`. */
    template <typename T>
    __device__ void so3_exp(const T *phi, T *R)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        T tmp[9]; lie_detail::so3_exp_core(phi, tmp);
        quat_detail::copy_out<T, 9>(lane, 32u, tmp, R);
        __syncwarp();
    }

    /** @brief Single-warp SO(3) log (canonical branch). See `glass::so3_log`. */
    template <typename T>
    __device__ void so3_log(const T *R, T *phi)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        T tmp[3]; lie_detail::so3_log_core(R, tmp);
        quat_detail::copy_out<T, 3>(lane, 32u, tmp, phi);
        __syncwarp();
    }

    /** @brief Single-warp right Jacobian. See `glass::so3_right_jacobian`. */
    template <typename T>
    __device__ void so3_right_jacobian(const T *phi, T *J)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        T tmp[9]; lie_detail::so3_right_jacobian_core(phi, tmp);
        quat_detail::copy_out<T, 9>(lane, 32u, tmp, J);
        __syncwarp();
    }

    /** @brief Single-warp inverse right Jacobian. See `glass::so3_right_jacobian_inv`. */
    template <typename T>
    __device__ void so3_right_jacobian_inv(const T *phi, T *J)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        T tmp[9]; lie_detail::so3_right_jacobian_inv_core(phi, tmp);
        quat_detail::copy_out<T, 9>(lane, 32u, tmp, J);
        __syncwarp();
    }

    /** @brief Single-warp left Jacobian (SE(3) "V matrix"). See `glass::so3_left_jacobian`. */
    template <typename T>
    __device__ void so3_left_jacobian(const T *phi, T *J)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        T tmp[9]; lie_detail::so3_left_jacobian_core(phi, tmp);
        quat_detail::copy_out<T, 9>(lane, 32u, tmp, J);
        __syncwarp();
    }

    /** @brief Single-warp inverse left Jacobian. See `glass::so3_left_jacobian_inv`. */
    template <typename T>
    __device__ void so3_left_jacobian_inv(const T *phi, T *J)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        T tmp[9]; lie_detail::so3_left_jacobian_inv_core(phi, tmp);
        quat_detail::copy_out<T, 9>(lane, 32u, tmp, J);
        __syncwarp();
    }
}
