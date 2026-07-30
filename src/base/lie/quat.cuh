#pragma once
#include <cstdint>

// ─── quaternion primitives (robotics ops, Lie family) ────────────────────────
//
// Unit-quaternion algebra: Hamilton product, conjugate, normalize, exponential,
// vector rotation, quaternion↔rotation-matrix conversion, and the SO(3)
// quaternion retract. These recur in every GPU robotics codebase (rigid-body
// dynamics integrators, batched IK, sampling-based control, motion generation)
// and are classically reimplemented with silently-different conventions — the
// point of this header is ONE tested home with the conventions pinned:
//
//   * ALL quaternions are HAMILTON quaternions (i·j = k). The STORAGE order is
//     a compile-time `QuatLayout` tag: `xyzw` (default; Eigen/Warp/cuRobo/GRiD
//     storage) or `wxyz` (MuJoCo/Ceres/GTSAM/ROS message order). The math is
//     written once against accessor indices, so both layouts share one body.
//   * Rotation matrices are 3x3 COLUMN-MAJOR (GLASS-wide convention).
//   * `quat_exp` takes the FULL rotation vector φ (axis·angle) and internally
//     halves it: `q = [sin(|φ|/2)·φ̂ ; cos(|φ|/2)]`. (Some codebases pass the
//     pre-halved vector — check twice when porting call sites.)
//   * Rotation action is `R(q)·p = q ⊗ [p;0] ⊗ q⁻¹` (body→world for a body
//     orientation quaternion).
//
// Tiers: every array-shaped op has block (`glass::`), warp (`glass::warp::`),
// and thread (`glass::thread::`) forms sharing one serial core: each active
// thread computes the (tiny, fixed-size) result redundantly into registers and
// the tier strides the copy-out — deterministic and thread-count invariant by
// construction (no reductions, no cross-thread data flow). Outputs must NOT
// alias inputs at block/warp scope (a strided writer may overwrite an operand
// another thread is still reading); the thread:: tier buffers through
// registers, so aliasing is safe there.

/**
 * @brief Compile-time quaternion storage layout tag.
 *
 * `xyzw` — vector part first, scalar last (Eigen / NVIDIA Warp / cuRobo / GRiD
 * storage order; the GLASS default). `wxyz` — scalar first (MuJoCo / Ceres /
 * GTSAM / ROS geometry_msgs order). Both are HAMILTON quaternions; the tag only
 * moves where each component is stored.
 */
enum class QuatLayout { xyzw, wxyz };

namespace quat_detail {
    // storage indices for a layout — formulas are written once against these.
    template <QuatLayout L> struct layout;
    template <> struct layout<QuatLayout::xyzw> {
        static constexpr uint32_t X = 0, Y = 1, Z = 2, W = 3;
    };
    template <> struct layout<QuatLayout::wxyz> {
        static constexpr uint32_t X = 1, Y = 2, Z = 3, W = 0;
    };

    // serial core: Hamilton product out = a ⊗ b (out must not alias a/b).
    template <typename T, QuatLayout L>
    __device__ __forceinline__ void quat_mul_core(const T *a, const T *b, T *out) {
        using QL = layout<L>;
        const T ax = a[QL::X], ay = a[QL::Y], az = a[QL::Z], aw = a[QL::W];
        const T bx = b[QL::X], by = b[QL::Y], bz = b[QL::Z], bw = b[QL::W];
        out[QL::X] = aw*bx + ax*bw + ay*bz - az*by;
        out[QL::Y] = aw*by - ax*bz + ay*bw + az*bx;
        out[QL::Z] = aw*bz + ax*by - ay*bx + az*bw;
        out[QL::W] = aw*bw - ax*bx - ay*by - az*bz;
    }

    // serial core: q = exp([φ/2]) — φ is the FULL rotation vector.
    template <typename T, QuatLayout L>
    __device__ __forceinline__ void quat_exp_core(const T *phi, T *out) {
        using QL = layout<L>;
        const T hx = static_cast<T>(0.5)*phi[0], hy = static_cast<T>(0.5)*phi[1],
                hz = static_cast<T>(0.5)*phi[2];
        const T theta = sqrt(hx*hx + hy*hy + hz*hz);
        T sinc, cos_t;
        if (theta < static_cast<T>(1e-12)) {   // sin(θ)/θ and cos(θ) Taylor heads
            sinc  = static_cast<T>(1) - theta*theta/static_cast<T>(6);
            cos_t = static_cast<T>(1) - static_cast<T>(0.5)*theta*theta;
        } else {
            sinc  = sin(theta)/theta;
            cos_t = cos(theta);
        }
        out[QL::X] = sinc*hx; out[QL::Y] = sinc*hy; out[QL::Z] = sinc*hz;
        out[QL::W] = cos_t;
    }

    // serial core: unit normalize (optionally canonicalize the double cover to w >= 0).
    template <typename T, QuatLayout L, bool CANONICAL>
    __device__ __forceinline__ void quat_normalize_core(const T *q, T *out) {
        using QL = layout<L>;
        const T n = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
        T s = static_cast<T>(1) / n;
        if constexpr (CANONICAL) { if (q[QL::W] < static_cast<T>(0)) s = -s; }
        out[0] = q[0]*s; out[1] = q[1]*s; out[2] = q[2]*s; out[3] = q[3]*s;
    }

    // serial core: p' = R(q)·p via p + 2w(v×p) + 2 v×(v×p), v = vec(q). Unit q assumed.
    template <typename T, QuatLayout L>
    __device__ __forceinline__ void quat_rotate_core(const T *q, const T *p, T *out) {
        using QL = layout<L>;
        const T x = q[QL::X], y = q[QL::Y], z = q[QL::Z], w = q[QL::W];
        // c = v × p
        const T cx = y*p[2] - z*p[1];
        const T cy = z*p[0] - x*p[2];
        const T cz = x*p[1] - y*p[0];
        // p + 2w·c + 2 v×c
        out[0] = p[0] + static_cast<T>(2)*(w*cx + y*cz - z*cy);
        out[1] = p[1] + static_cast<T>(2)*(w*cy + z*cx - x*cz);
        out[2] = p[2] + static_cast<T>(2)*(w*cz + x*cy - y*cx);
    }

    // serial core: R (3x3 column-major) from a UNIT quaternion.
    template <typename T, QuatLayout L>
    __device__ __forceinline__ void quat_to_rot_core(const T *q, T *R) {
        using QL = layout<L>;
        const T x = q[QL::X], y = q[QL::Y], z = q[QL::Z], w = q[QL::W];
        const T xx = x*x, yy = y*y, zz = z*z;
        const T xy = x*y, xz = x*z, yz = y*z;
        const T wx = w*x, wy = w*y, wz = w*z;
        const T one = static_cast<T>(1), two = static_cast<T>(2);
        // column-major: R[c*3 + r] = R(r,c)
        R[0] = one - two*(yy + zz); R[3] = two*(xy - wz);       R[6] = two*(xz + wy);
        R[1] = two*(xy + wz);       R[4] = one - two*(xx + zz); R[7] = two*(yz - wx);
        R[2] = two*(xz - wy);       R[5] = two*(yz + wx);       R[8] = one - two*(xx + yy);
    }

    // serial core: quaternion from a rotation matrix (Shepperd max-pivot: branch
    // on the largest of trace/diagonal so the divisor is never small). R is 3x3
    // column-major with leading dimension LDA (LDA=4 reads the rotation block of
    // a column-major 4x4 homogeneous transform in place); the result is unit up
    // to the orthonormality of R, with the canonical w >= 0 sign.
    template <typename T, QuatLayout L, uint32_t LDA = 3>
    __device__ __forceinline__ void rot_to_quat_core(const T *R, T *q) {
        using QL = layout<L>;
        // column-major reads: R(r,c) = R[c*LDA + r]
        const T r00 = R[0],       r10 = R[1],       r20 = R[2];
        const T r01 = R[LDA],     r11 = R[LDA + 1], r21 = R[LDA + 2];
        const T r02 = R[2*LDA],   r12 = R[2*LDA + 1], r22 = R[2*LDA + 2];
        const T tr = r00 + r11 + r22;
        T x, y, z, w;
        if (tr > static_cast<T>(0)) {
            T s = sqrt(tr + static_cast<T>(1)) * static_cast<T>(2);   // 4w
            w = static_cast<T>(0.25)*s;
            x = (r21 - r12)/s; y = (r02 - r20)/s; z = (r10 - r01)/s;
        } else if (r00 > r11 && r00 > r22) {
            T s = sqrt(static_cast<T>(1) + r00 - r11 - r22) * static_cast<T>(2);   // 4x
            w = (r21 - r12)/s;
            x = static_cast<T>(0.25)*s;
            y = (r01 + r10)/s; z = (r02 + r20)/s;
        } else if (r11 > r22) {
            T s = sqrt(static_cast<T>(1) + r11 - r00 - r22) * static_cast<T>(2);   // 4y
            w = (r02 - r20)/s;
            x = (r01 + r10)/s;
            y = static_cast<T>(0.25)*s;
            z = (r12 + r21)/s;
        } else {
            T s = sqrt(static_cast<T>(1) + r22 - r00 - r11) * static_cast<T>(2);   // 4z
            w = (r10 - r01)/s;
            x = (r02 + r20)/s; y = (r12 + r21)/s;
            z = static_cast<T>(0.25)*s;
        }
        if (w < static_cast<T>(0)) { w = -w; x = -x; y = -y; z = -z; }   // canonical cover
        q[QL::X] = x; q[QL::Y] = y; q[QL::Z] = z; q[QL::W] = w;
    }

    // serial core: SO(3) retract on the quaternion chart —
    // q_new = normalize(q ⊗ exp([φ/2])). Renormalizing every step keeps the
    // integrated quaternion on the unit sphere (drift-free).
    template <typename T, QuatLayout L>
    __device__ __forceinline__ void quat_retract_core(const T *q, const T *phi, T *q_new) {
        T dq[4]; quat_exp_core<T, L>(phi, dq);
        T qm[4]; quat_mul_core<T, L>(q, dq, qm);
        quat_normalize_core<T, L, false>(qm, q_new);
    }

    // serial core: φ = log(q) — the rotation vector of a UNIT quaternion,
    // canonical branch |φ| ≤ π (the double cover folds via the w-sign flip, so
    // q and −q give the same shortest-path answer). `φ = 2·atan2(|v|, w)·v̂`
    // with the series head `2/w − 2|v|²/(3w³)` below |v| = 1e-8 keeping the
    // map smooth through the identity.
    template <typename T, QuatLayout L>
    __device__ __forceinline__ void quat_log_core(const T *q, T *phi) {
        using QL = layout<L>;
        T x = q[QL::X], y = q[QL::Y], z = q[QL::Z], w = q[QL::W];
        if (w < static_cast<T>(0)) { x = -x; y = -y; z = -z; w = -w; }
        const T n = sqrt(x*x + y*y + z*z);
        T s;   // scale = θ/|v| with θ = 2·atan2(|v|, w)
        if (n < static_cast<T>(1e-8)) {
            s = static_cast<T>(2)/w - static_cast<T>(2)*n*n/(static_cast<T>(3)*w*w*w);
        } else {
            s = static_cast<T>(2)*atan2(n, w)/n;
        }
        phi[0] = s*x; phi[1] = s*y; phi[2] = s*z;
    }

    // tier glue: strided copy-out of a register tmp (the redundant-core pattern).
    template <typename T, uint32_t N>
    __device__ __forceinline__ void copy_out(uint32_t rank, uint32_t size,
                                             const T *tmp, T *out) {
        for (uint32_t i = rank; i < N; i += size) out[i] = tmp[i];
    }

    // tier glue: strided copy-out of a contiguous 3x3 register tmp into a
    // column-major destination with leading dimension LDA (only the nine
    // rotation entries are written — LDA=4 targets the rotation block of a 4x4
    // homogeneous transform without touching its translation row/column).
    template <typename T, uint32_t LDA>
    __device__ __forceinline__ void copy_out_mat3(uint32_t rank, uint32_t size,
                                                  const T *tmp, T *out) {
        for (uint32_t i = rank; i < 9; i += size)
            out[(i/3)*LDA + (i%3)] = tmp[i];
    }
} // namespace quat_detail

/**
 * @brief Hamilton quaternion product: `out = a ⊗ b`.
 *
 * Composition of rotations: `R(out) = R(a)·R(b)` (apply `b` first in the body
 * frame of `a`). Layout via the `L` tag (default `xyzw`); both operands and the
 * result share one layout. `out` must not alias `a`/`b` at block/warp scope.
 * NumPy equivalent (xyzw, scipy): `(Rotation.from_quat(a) * Rotation.from_quat(b)).as_quat()`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam L  Quaternion storage layout (default `QuatLayout::xyzw`).
 * @tparam TRAILING_SYNC  Emit a trailing `__syncthreads()` (default true).
 * @param a    Left quaternion (4 elements).
 * @param b    Right quaternion (4 elements).
 * @param out  Result quaternion (4 elements; no aliasing).
 */
template <typename T, QuatLayout L = QuatLayout::xyzw, bool TRAILING_SYNC = true>
__device__ void quat_mul(const T *a, const T *b, T *out)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    T tmp[4]; quat_detail::quat_mul_core<T, L>(a, b, tmp);
    quat_detail::copy_out<T, 4>(rank, size, tmp, out);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief Quaternion conjugate (= inverse for unit quaternions): `out = a*`.
 *
 * Negates the vector part. NumPy equivalent (xyzw): `[-x, -y, -z, w]`.
 *
 * @tparam T,L,TRAILING_SYNC  See `quat_mul`.
 * @param a    Input quaternion (4 elements).
 * @param out  Conjugated quaternion (4 elements; no aliasing at block/warp scope).
 */
template <typename T, QuatLayout L = QuatLayout::xyzw, bool TRAILING_SYNC = true>
__device__ void quat_conj(const T *a, T *out)
{
    using QL = quat_detail::layout<L>;
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    T tmp[4];
    tmp[QL::X] = -a[QL::X]; tmp[QL::Y] = -a[QL::Y]; tmp[QL::Z] = -a[QL::Z];
    tmp[QL::W] =  a[QL::W];
    quat_detail::copy_out<T, 4>(rank, size, tmp, out);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief Normalize a quaternion to unit length: `out = q / |q|`.
 *
 * With `CANONICAL = true` the result is also flipped onto the `w >= 0` half of
 * the double cover (`q` and `-q` encode the same rotation) — useful before
 * comparing or interpolating quaternions. In-place (`out == q`) is safe at
 * thread:: scope only.
 *
 * @tparam T,L,TRAILING_SYNC  See `quat_mul`.
 * @tparam CANONICAL  Also canonicalize the sign to `w >= 0` (default false).
 * @param q    Input quaternion (4 elements, nonzero).
 * @param out  Unit quaternion (4 elements).
 */
template <typename T, QuatLayout L = QuatLayout::xyzw, bool CANONICAL = false,
          bool TRAILING_SYNC = true>
__device__ void quat_normalize(const T *q, T *out)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    T tmp[4]; quat_detail::quat_normalize_core<T, L, CANONICAL>(q, tmp);
    quat_detail::copy_out<T, 4>(rank, size, tmp, out);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief Quaternion exponential of a rotation vector: `out = exp([φ/2])`.
 *
 * `φ` is the FULL rotation vector (axis · angle, radians); the half-angle is
 * taken internally: `out = [sin(|φ|/2)·φ̂ ; cos(|φ|/2)]`, with the `sin(θ)/θ`
 * Taylor head below θ = 1e-12 so the derivative is exact through φ = 0.
 * NumPy equivalent (xyzw): `Rotation.from_rotvec(phi).as_quat()`.
 *
 * @tparam T,L,TRAILING_SYNC  See `quat_mul`.
 * @param phi  Rotation vector (3 elements).
 * @param out  Unit quaternion (4 elements; no aliasing at block/warp scope).
 */
template <typename T, QuatLayout L = QuatLayout::xyzw, bool TRAILING_SYNC = true>
__device__ void quat_exp(const T *phi, T *out)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    T tmp[4]; quat_detail::quat_exp_core<T, L>(phi, tmp);
    quat_detail::copy_out<T, 4>(rank, size, tmp, out);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief Quaternion logarithm: the rotation vector `φ` of a unit quaternion.
 *
 * The inverse of `quat_exp` on the canonical branch `|φ| ≤ π`: `φ = 2·log(q)`
 * with the double cover folded (q and −q return the SAME shortest-path
 * vector). Routed through `2·atan2(|v|, w)` — stable across the whole range
 * including θ near π — with a series head below `|v| = 1e-8` keeping the map
 * smooth through the identity. `q` must be unit length. NumPy equivalent
 * (xyzw): `Rotation.from_quat(q).as_rotvec()`.
 *
 * @tparam T,L,TRAILING_SYNC  See `quat_mul`.
 * @param q    Unit quaternion (4 elements).
 * @param phi  Output rotation vector (3 elements; no aliasing at block/warp scope).
 */
template <typename T, QuatLayout L = QuatLayout::xyzw, bool TRAILING_SYNC = true>
__device__ void quat_log(const T *q, T *phi)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    T tmp[3]; quat_detail::quat_log_core<T, L>(q, tmp);
    quat_detail::copy_out<T, 3>(rank, size, tmp, phi);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief Rotate a 3-vector by a unit quaternion: `out = R(q)·p`.
 *
 * Uses the cross-product form `p + 2w(v×p) + 2v×(v×p)` (no 3x3 materialized;
 * 18 mul + 12 add vs 15 mul + 15 add through the matrix, but with no scratch).
 * `q` must be unit length. NumPy equivalent (xyzw):
 * `Rotation.from_quat(q).apply(p)`.
 *
 * @tparam T,L,TRAILING_SYNC  See `quat_mul`.
 * @param q    Unit quaternion (4 elements).
 * @param p    Input vector (3 elements).
 * @param out  Rotated vector (3 elements; no aliasing at block/warp scope).
 */
template <typename T, QuatLayout L = QuatLayout::xyzw, bool TRAILING_SYNC = true>
__device__ void quat_rotate(const T *q, const T *p, T *out)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    T tmp[3]; quat_detail::quat_rotate_core<T, L>(q, p, tmp);
    quat_detail::copy_out<T, 3>(rank, size, tmp, out);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief Rotation matrix (3x3 column-major) from a unit quaternion.
 *
 * `LDA` is the destination's leading dimension (row stride between columns;
 * default 3 = contiguous). `LDA = 4` writes the rotation block of a
 * column-major 4x4 homogeneous transform IN PLACE — only the nine rotation
 * entries are touched (the `gemv`/`gemm` ROW_STRIDE pattern extended to the
 * Lie corner). NumPy equivalent (xyzw): `Rotation.from_quat(q).as_matrix()`
 * (flatten Fortran-order for the column-major array).
 *
 * @tparam T,L,TRAILING_SYNC  See `quat_mul`.
 * @tparam LDA  Destination leading dimension (default 3).
 * @param q  Unit quaternion (4 elements).
 * @param R  Output rotation (column-major, leading dimension LDA; no aliasing).
 */
template <typename T, QuatLayout L = QuatLayout::xyzw, uint32_t LDA = 3, bool TRAILING_SYNC = true>
__device__ void quat_to_rot(const T *q, T *R)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    T tmp[9]; quat_detail::quat_to_rot_core<T, L>(q, tmp);
    quat_detail::copy_out_mat3<T, LDA>(rank, size, tmp, R);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief Unit quaternion from a rotation matrix (Shepperd max-pivot extraction).
 *
 * Branches on the largest of the trace and the three diagonal entries so the
 * divisor is never small — numerically safe for every rotation including the
 * θ = π family. Result is canonicalized to `w >= 0`. `R` is column-major with
 * leading dimension `LDA` (default 3 = contiguous; `LDA = 4` reads the
 * rotation block of a column-major 4x4 homogeneous transform in place — no
 * repack). NumPy equivalent (xyzw): `Rotation.from_matrix(R).as_quat()` (up
 * to the double-cover sign).
 *
 * @tparam T,L,TRAILING_SYNC  See `quat_mul`.
 * @tparam LDA  Source leading dimension (default 3).
 * @param R  Input rotation matrix (column-major, leading dimension LDA).
 * @param q  Output unit quaternion (4 elements; no aliasing at block/warp scope).
 */
template <typename T, QuatLayout L = QuatLayout::xyzw, uint32_t LDA = 3, bool TRAILING_SYNC = true>
__device__ void rot_to_quat(const T *R, T *q)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    T tmp[4]; quat_detail::rot_to_quat_core<T, L, LDA>(R, tmp);
    quat_detail::copy_out<T, 4>(rank, size, tmp, q);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief Normalize a quaternion and return the three rotation-matrix columns.
 *
 * `u`/`v`/`w` receive columns 0/1/2 of `R(q/|q|)` — the body frame's three axes
 * in the parent frame. The explicit normalize makes this safe on raw stored
 * quaternions (e.g. parsed poses). Collision-geometry / frame-setup helper.
 *
 * @tparam T,L,TRAILING_SYNC  See `quat_mul`.
 * @param q  Quaternion (4 elements; NOT required to be unit).
 * @param u  Output column 0 (3 elements).
 * @param v  Output column 1 (3 elements).
 * @param w  Output column 2 (3 elements).
 */
template <typename T, QuatLayout L = QuatLayout::xyzw, bool TRAILING_SYNC = true>
__device__ void quat_to_basis(const T *q, T *u, T *v, T *w)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    T qn[4]; quat_detail::quat_normalize_core<T, L, false>(q, qn);
    T R[9];  quat_detail::quat_to_rot_core<T, L>(qn, R);
    for (uint32_t i = rank; i < 9; i += size) {
        T *dst = (i < 3) ? u : (i < 6) ? v : w;
        dst[i % 3] = R[i];
    }
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief SO(3) retract on the quaternion chart: `q_new = normalize(q ⊗ exp([φ/2]))`.
 *
 * The manifold update `q ⊞ φ` (body-frame tangent `φ`, radians): one integrator
 * step of a spherical joint / orientation state under angular velocity is
 * `quat_retract(q, ω·dt, q_new)`. The trailing renormalize keeps repeated
 * retracts on the unit sphere (drift-free). NumPy equivalent (xyzw):
 * `(Rotation.from_quat(q) * Rotation.from_rotvec(phi)).as_quat()`.
 *
 * @tparam T,L,TRAILING_SYNC  See `quat_mul`.
 * @param q      Current unit quaternion (4 elements).
 * @param phi    Tangent step — full rotation vector (3 elements).
 * @param q_new  Updated unit quaternion (4 elements; no aliasing at block/warp scope).
 */
template <typename T, QuatLayout L = QuatLayout::xyzw, bool TRAILING_SYNC = true>
__device__ void quat_retract(const T *q, const T *phi, T *q_new)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    T tmp[4]; quat_detail::quat_retract_core<T, L>(q, phi, tmp);
    quat_detail::copy_out<T, 4>(rank, size, tmp, q_new);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

// ─── single-thread quaternion ops ────────────────────────────────────────────
namespace thread {
    // One thread owns the whole (4/9-element) result: the SAME serial cores as
    // the block/warp tiers with a plain serial copy-out. No barriers, no
    // shuffles, no threadIdx read; operands may be thread-local register
    // arrays, and in-place aliasing is safe (the core buffers via registers).

    /** @brief Single-thread `out = a ⊗ b`. See `glass::quat_mul`. */
    template <typename T, QuatLayout L = QuatLayout::xyzw>
    __device__ void quat_mul(const T *a, const T *b, T *out)
    {
        T tmp[4]; quat_detail::quat_mul_core<T, L>(a, b, tmp);
        for (uint32_t i = 0; i < 4; i++) out[i] = tmp[i];
    }

    /** @brief Single-thread conjugate. See `glass::quat_conj`. */
    template <typename T, QuatLayout L = QuatLayout::xyzw>
    __device__ void quat_conj(const T *a, T *out)
    {
        using QL = quat_detail::layout<L>;
        const T x = a[QL::X], y = a[QL::Y], z = a[QL::Z], w = a[QL::W];
        out[QL::X] = -x; out[QL::Y] = -y; out[QL::Z] = -z; out[QL::W] = w;
    }

    /** @brief Single-thread normalize (optional `w>=0` canonicalization). See `glass::quat_normalize`. */
    template <typename T, QuatLayout L = QuatLayout::xyzw, bool CANONICAL = false>
    __device__ void quat_normalize(const T *q, T *out)
    {
        quat_detail::quat_normalize_core<T, L, CANONICAL>(q, out);
    }

    /** @brief Single-thread `exp([φ/2])`. See `glass::quat_exp`. */
    template <typename T, QuatLayout L = QuatLayout::xyzw>
    __device__ void quat_exp(const T *phi, T *out)
    {
        quat_detail::quat_exp_core<T, L>(phi, out);
    }

    /** @brief Single-thread quaternion logarithm. See `glass::quat_log`. */
    template <typename T, QuatLayout L = QuatLayout::xyzw>
    __device__ void quat_log(const T *q, T *phi)
    {
        T tmp[3]; quat_detail::quat_log_core<T, L>(q, tmp);
        phi[0] = tmp[0]; phi[1] = tmp[1]; phi[2] = tmp[2];
    }

    /** @brief Single-thread `R(q)·p`. See `glass::quat_rotate`. */
    template <typename T, QuatLayout L = QuatLayout::xyzw>
    __device__ void quat_rotate(const T *q, const T *p, T *out)
    {
        T tmp[3]; quat_detail::quat_rotate_core<T, L>(q, p, tmp);
        out[0] = tmp[0]; out[1] = tmp[1]; out[2] = tmp[2];
    }

    /** @brief Single-thread quaternion → column-major 3x3 (LDA-strided). See `glass::quat_to_rot`. */
    template <typename T, QuatLayout L = QuatLayout::xyzw, uint32_t LDA = 3>
    __device__ void quat_to_rot(const T *q, T *R)
    {
        if constexpr (LDA == 3) {
            quat_detail::quat_to_rot_core<T, L>(q, R);
        } else {
            T tmp[9]; quat_detail::quat_to_rot_core<T, L>(q, tmp);
            quat_detail::copy_out_mat3<T, LDA>(0u, 1u, tmp, R);
        }
    }

    /** @brief Single-thread column-major 3x3 (LDA-strided) → quaternion (Shepperd). See `glass::rot_to_quat`. */
    template <typename T, QuatLayout L = QuatLayout::xyzw, uint32_t LDA = 3>
    __device__ void rot_to_quat(const T *R, T *q)
    {
        quat_detail::rot_to_quat_core<T, L, LDA>(R, q);
    }

    /** @brief Single-thread normalize + rotation columns. See `glass::quat_to_basis`. */
    template <typename T, QuatLayout L = QuatLayout::xyzw>
    __device__ void quat_to_basis(const T *q, T *u, T *v, T *w)
    {
        T qn[4]; quat_detail::quat_normalize_core<T, L, false>(q, qn);
        T R[9];  quat_detail::quat_to_rot_core<T, L>(qn, R);
        for (uint32_t i = 0; i < 3; i++) { u[i] = R[i]; v[i] = R[3+i]; w[i] = R[6+i]; }
    }

    /** @brief Single-thread SO(3) quaternion retract. See `glass::quat_retract`. */
    template <typename T, QuatLayout L = QuatLayout::xyzw>
    __device__ void quat_retract(const T *q, const T *phi, T *q_new)
    {
        T tmp[4]; quat_detail::quat_retract_core<T, L>(q, phi, tmp);
        for (uint32_t i = 0; i < 4; i++) q_new[i] = tmp[i];
    }
}

// ─── single-warp quaternion ops ──────────────────────────────────────────────
namespace warp {
    // One 32-lane warp owns the result: the same serial cores, lane-strided
    // copy-out, `__syncwarp()` close. For warp-per-problem kernels. Outputs must
    // not alias inputs (lanes write while others may still read).

    /** @brief Single-warp `out = a ⊗ b`. See `glass::quat_mul`. */
    template <typename T, QuatLayout L = QuatLayout::xyzw>
    __device__ void quat_mul(const T *a, const T *b, T *out)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        T tmp[4]; quat_detail::quat_mul_core<T, L>(a, b, tmp);
        quat_detail::copy_out<T, 4>(lane, 32u, tmp, out);
        __syncwarp();
    }

    /** @brief Single-warp conjugate. See `glass::quat_conj`. */
    template <typename T, QuatLayout L = QuatLayout::xyzw>
    __device__ void quat_conj(const T *a, T *out)
    {
        using QL = quat_detail::layout<L>;
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        T tmp[4];
        tmp[QL::X] = -a[QL::X]; tmp[QL::Y] = -a[QL::Y]; tmp[QL::Z] = -a[QL::Z];
        tmp[QL::W] =  a[QL::W];
        quat_detail::copy_out<T, 4>(lane, 32u, tmp, out);
        __syncwarp();
    }

    /** @brief Single-warp normalize. See `glass::quat_normalize`. */
    template <typename T, QuatLayout L = QuatLayout::xyzw, bool CANONICAL = false>
    __device__ void quat_normalize(const T *q, T *out)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        T tmp[4]; quat_detail::quat_normalize_core<T, L, CANONICAL>(q, tmp);
        quat_detail::copy_out<T, 4>(lane, 32u, tmp, out);
        __syncwarp();
    }

    /** @brief Single-warp `exp([φ/2])`. See `glass::quat_exp`. */
    template <typename T, QuatLayout L = QuatLayout::xyzw>
    __device__ void quat_exp(const T *phi, T *out)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        T tmp[4]; quat_detail::quat_exp_core<T, L>(phi, tmp);
        quat_detail::copy_out<T, 4>(lane, 32u, tmp, out);
        __syncwarp();
    }

    /** @brief Single-warp quaternion logarithm. See `glass::quat_log`. */
    template <typename T, QuatLayout L = QuatLayout::xyzw>
    __device__ void quat_log(const T *q, T *phi)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        T tmp[3]; quat_detail::quat_log_core<T, L>(q, tmp);
        quat_detail::copy_out<T, 3>(lane, 32u, tmp, phi);
        __syncwarp();
    }

    /** @brief Single-warp `R(q)·p`. See `glass::quat_rotate`. */
    template <typename T, QuatLayout L = QuatLayout::xyzw>
    __device__ void quat_rotate(const T *q, const T *p, T *out)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        T tmp[3]; quat_detail::quat_rotate_core<T, L>(q, p, tmp);
        quat_detail::copy_out<T, 3>(lane, 32u, tmp, out);
        __syncwarp();
    }

    /** @brief Single-warp quaternion → column-major 3x3 (LDA-strided). See `glass::quat_to_rot`. */
    template <typename T, QuatLayout L = QuatLayout::xyzw, uint32_t LDA = 3>
    __device__ void quat_to_rot(const T *q, T *R)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        T tmp[9]; quat_detail::quat_to_rot_core<T, L>(q, tmp);
        quat_detail::copy_out_mat3<T, LDA>(lane, 32u, tmp, R);
        __syncwarp();
    }

    /** @brief Single-warp 3x3 (LDA-strided) → quaternion (Shepperd). See `glass::rot_to_quat`. */
    template <typename T, QuatLayout L = QuatLayout::xyzw, uint32_t LDA = 3>
    __device__ void rot_to_quat(const T *R, T *q)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        T tmp[4]; quat_detail::rot_to_quat_core<T, L, LDA>(R, tmp);
        quat_detail::copy_out<T, 4>(lane, 32u, tmp, q);
        __syncwarp();
    }

    /** @brief Single-warp normalize + rotation columns. See `glass::quat_to_basis`. */
    template <typename T, QuatLayout L = QuatLayout::xyzw>
    __device__ void quat_to_basis(const T *q, T *u, T *v, T *w)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        T qn[4]; quat_detail::quat_normalize_core<T, L, false>(q, qn);
        T R[9];  quat_detail::quat_to_rot_core<T, L>(qn, R);
        for (uint32_t i = lane; i < 9; i += 32u) {
            T *dst = (i < 3) ? u : (i < 6) ? v : w;
            dst[i % 3] = R[i];
        }
        __syncwarp();
    }

    /** @brief Single-warp SO(3) quaternion retract. See `glass::quat_retract`. */
    template <typename T, QuatLayout L = QuatLayout::xyzw>
    __device__ void quat_retract(const T *q, const T *phi, T *q_new)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        T tmp[4]; quat_detail::quat_retract_core<T, L>(q, phi, tmp);
        quat_detail::copy_out<T, 4>(lane, 32u, tmp, q_new);
        __syncwarp();
    }
}
