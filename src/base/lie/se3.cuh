#pragma once
#include <cstdint>

// ─── SE(3) retract and its derivatives (robotics ops, Lie family) ────────────
//
// The floating-base / free-flyer toolkit: the SE(3) manifold update
// ("retract" / boxplus) on a position+quaternion pose block, the Barfoot Q
// coupling block, the two 6x6 first-derivative blocks of the retract
// (w.r.t. the base pose and w.r.t. the tangent step — Pinocchio's
// `dIntegrate` ARG0/ARG1), and the full 6x6x6 second-derivative tensor.
// Promoted from GRiD's floating-base integrator emitters, where this chain is
// Pinocchio-validated (and the second derivative mpmath-complex-step
// validated); the formulas are ported verbatim, restated on GLASS's
// column-major storage.
//
// CONVENTIONS (read this — it is the classic cross-library trap):
//   * The SE(3) TANGENT is passed as two explicit 3-vectors `rho` (linear) and
//     `phi` (angular) — no packed 6-vector, so no ordering ambiguity on input.
//   * OUTPUT 6x6/6x6x6 blocks are indexed with the tangent ordered
//     [ρ(3); φ(3)] — LINEAR-FIRST, matching Pinocchio's `dIntegrate`. This is
//     the opposite half of the field from the Featherstone spatial-vector
//     convention `[ω; v]` used by `motion_cross`/`force_cross` (angular-first):
//     the two families serve different callers and each keeps its native
//     literature convention. Do not mix them without a permutation.
//   * The pose block is 7 elements `[p(3); quat(4)]`, quaternion layout via
//     the `QuatLayout` tag (default `xyzw` — the Pinocchio `nq` layout).
//   * Matrices are column-major (3x3 `M[c*3+r]`, 6x6 `J[c*6+r]`); the Hessian
//     tensor is six STACKED column-major 6x6 slices: `J2[k*36 + c*6 + r]` =
//     ∂J(r,c)/∂w_k with w = [ρ; φ].
//   * The second-derivative chain computes in DOUBLE internally regardless of
//     `T` (a tiny once-per-step block; keeps a float32 kernel matching a
//     float64 oracle — the validated GRiD design decision, kept).
//
// Tiers: block/warp/thread share the serial cores (redundant-core + strided
// copy-out; the Hessian parallelizes over the 6 tangent directions instead —
// one direction per active thread). Outputs must not alias inputs at
// block/warp scope.

namespace lie_detail {
    // serial core: the Barfoot Q coupling block of the SE(3) exponential
    // derivative (Pinocchio sign convention: Q = −Sola(ρ, −φ); ported verbatim
    // from the pin.dIntegrate-validated GRiD emitter). 3x3 column-major.
    template <typename T>
    __device__ __forceinline__ void se3_Q_block_core(const T *rho, const T *phi, T *Q) {
        const T phi_neg[3] = {-phi[0], -phi[1], -phi[2]};
        T Px[9];  skew_core(phi_neg, Px);
        T Rx[9];  skew_core(rho, Rx);
        T Px2[9];      mat3_mul_core(Px, Px, Px2);
        T Rx_Px[9];    mat3_mul_core(Rx, Px, Rx_Px);
        T Px_Rx[9];    mat3_mul_core(Px, Rx, Px_Rx);
        T Px_Rx_Px[9]; mat3_mul_core(Px, Rx_Px, Px_Rx_Px);
        T Px2_Rx[9];   mat3_mul_core(Px2, Rx, Px2_Rx);
        T Rx_Px2[9];   mat3_mul_core(Rx_Px, Px, Rx_Px2);      // Rx·Px·Px
        T Px_Rx_Px2[9]; mat3_mul_core(Px_Rx, Px2, Px_Rx_Px2); // Px·Rx·Px·Px
        T Px2_Rx_Px[9]; mat3_mul_core(Px2, Rx_Px, Px2_Rx_Px); // Px·Px·Rx·Px
        const T t = vec3_norm(phi);
        T sola[9];
        if (t < static_cast<T>(1e-4)) {
            #pragma unroll
            for (uint32_t i = 0; i < 9; ++i)
                sola[i] = static_cast<T>(0.5)*Rx[i]
                        + static_cast<T>(1.0/6.0)*(Px_Rx[i] + Rx_Px[i] + Px_Rx_Px[i])
                        - static_cast<T>(1.0/24.0)*(Px2_Rx[i] + Rx_Px2[i]
                                                    - static_cast<T>(3)*Px_Rx_Px[i]);
        } else {
            const T t2 = t*t, t3 = t2*t, t4 = t3*t, t5 = t4*t;
            const T c1 = (t - sin(t)) / t3;
            const T c2 = (static_cast<T>(1) - static_cast<T>(0.5)*t2 - cos(t)) / t4;
            // Negative sign matches Barfoot's (2θ−3sinθ+θcosθ)/(2θ⁵) coefficient;
            // verified against pin.dIntegrate(ARG1) in the GRiD arc.
            const T c3 = static_cast<T>(-0.5)
                       * (c2 - static_cast<T>(3)*(t - sin(t) - t3/static_cast<T>(6)) / t5);
            #pragma unroll
            for (uint32_t i = 0; i < 9; ++i)
                sola[i] = static_cast<T>(0.5)*Rx[i]
                        + c1*(Px_Rx[i] + Rx_Px[i] + Px_Rx_Px[i])
                        - c2*(Px2_Rx[i] + Rx_Px2[i] - static_cast<T>(3)*Px_Rx_Px[i])
                        + c3*(Px_Rx_Px2[i] + Px2_Rx_Px[i]);
        }
        #pragma unroll
        for (uint32_t i = 0; i < 9; ++i) Q[i] = -sola[i];
    }

    // serial core: SE(3) retract on the [p(3); quat(4)] pose block.
    template <typename T, QuatLayout L>
    __device__ __forceinline__ void se3_retract_core(const T *pose, const T *rho,
                                                     const T *phi, T *out) {
        // orientation: q_new = normalize(q ⊗ exp([φ/2]))
        lie_detail::quat_retract_core<T, L>(pose + 3, phi, out + 3);
        // position: p_new = p + R(q)·(Jl(φ)·ρ)   (body-frame tangent)
        T V[9];  so3_left_jacobian_core(phi, V);
        T pl[3]; mat3_vec_core(V, rho, pl);
        T R[9];  lie_detail::quat_to_rot_core<T, L>(pose + 3, R);
        T pw[3]; mat3_vec_core(R, pl, pw);
        out[0] = pose[0] + pw[0];
        out[1] = pose[1] + pw[1];
        out[2] = pose[2] + pw[2];
    }

    // serial core: SE(3) difference (boxminus) — the exact inverse of
    // se3_retract_core: the tangent [ρ; φ] with
    // se3_retract(pose_from, ρ, φ) == pose_to. Canonical branch |φ| ≤ π
    // (quat_log's w-sign fold), matching Pinocchio `difference` on the
    // free-flyer.
    template <typename T, QuatLayout L>
    __device__ __forceinline__ void se3_difference_core(const T *pose_from, const T *pose_to,
                                                        T *rho, T *phi) {
        using QL = lie_detail::layout<L>;
        // φ = log(q_from⁻¹ ⊗ q_to)
        const T *qf = pose_from + 3;
        T qf_conj[4];
        qf_conj[QL::X] = -qf[QL::X]; qf_conj[QL::Y] = -qf[QL::Y];
        qf_conj[QL::Z] = -qf[QL::Z]; qf_conj[QL::W] =  qf[QL::W];
        T q_rel[4]; lie_detail::quat_mul_core<T, L>(qf_conj, pose_to + 3, q_rel);
        lie_detail::quat_log_core<T, L>(q_rel, phi);
        // ρ = Jl(φ)⁻¹ · R(q_from)ᵀ · (p_to − p_from)   (undo the body-frame V·ρ)
        const T dp[3] = {pose_to[0] - pose_from[0],
                         pose_to[1] - pose_from[1],
                         pose_to[2] - pose_from[2]};
        T R[9];  lie_detail::quat_to_rot_core<T, L>(qf, R);
        T pl[3];   // Rᵀ·dp (R column-major: row i of Rᵀ = column i of R)
        #pragma unroll
        for (uint32_t i = 0; i < 3; ++i)
            pl[i] = R[i*3 + 0]*dp[0] + R[i*3 + 1]*dp[1] + R[i*3 + 2]*dp[2];
        T Vinv[9]; so3_left_jacobian_inv_core(phi, Vinv);
        mat3_vec_core(Vinv, pl, rho);
    }

    // serial core: d(retract)/d(base pose) = Ad_{exp(−[ρ;φ])} as a 6x6
    // column-major block [[R⁻, off],[0, R⁻]] in [ρ; φ] tangent order.
    template <typename T>
    __device__ __forceinline__ void se3_retract_jacobian_q_core(const T *rho, const T *phi,
                                                                T *J) {
        const T phi_neg[3] = {-phi[0], -phi[1], -phi[2]};
        T R_inv[9]; so3_exp_core(phi_neg, R_inv);
        T V_neg[9]; so3_left_jacobian_core(phi_neg, V_neg);
        T Vr[3];    mat3_vec_core(V_neg, rho, Vr);
        const T p_inv[3] = {-Vr[0], -Vr[1], -Vr[2]};
        T Px[9];  skew_core(p_inv, Px);
        T off[9]; mat3_mul_core(Px, R_inv, off);
        #pragma unroll
        for (uint32_t i = 0; i < 36; ++i) J[i] = static_cast<T>(0);
        #pragma unroll
        for (uint32_t c = 0; c < 3; ++c) {
            #pragma unroll
            for (uint32_t r = 0; r < 3; ++r) {
                J[c*6 + r]           = R_inv[c*3 + r];   // (0:3, 0:3)
                J[(c + 3)*6 + r]     = off[c*3 + r];     // (0:3, 3:6)
                J[(c + 3)*6 + r + 3] = R_inv[c*3 + r];   // (3:6, 3:6)
            }
        }
    }

    // serial core: d(retract)/d(tangent) = the SE(3) right Jacobian as a 6x6
    // column-major block [[Jr, Q],[0, Jr]] in [ρ; φ] tangent order.
    template <typename T>
    __device__ __forceinline__ void se3_retract_jacobian_v_core(const T *rho, const T *phi,
                                                                T *J) {
        T Jr[9]; so3_right_jacobian_core(phi, Jr);
        T Q[9];  se3_Q_block_core(rho, phi, Q);
        #pragma unroll
        for (uint32_t i = 0; i < 36; ++i) J[i] = static_cast<T>(0);
        #pragma unroll
        for (uint32_t c = 0; c < 3; ++c) {
            #pragma unroll
            for (uint32_t r = 0; r < 3; ++r) {
                J[c*6 + r]           = Jr[c*3 + r];      // (0:3, 0:3)
                J[(c + 3)*6 + r]     = Q[c*3 + r];       // (0:3, 3:6)
                J[(c + 3)*6 + r + 3] = Jr[c*3 + r];      // (3:6, 3:6)
            }
        }
    }

    // ---- second-derivative chain (DOUBLE internals; ported verbatim from the
    //      mpmath-complex-step-validated GRiD emitters) ------------------------

    // SE(3)-exp coefficient VALUES c[5] = {b, c, a, c2, c3} in Rodrigues terms
    // (c[0]=(1−cos t)/t², c[1]=(t−sin t)/t³, c[2]=sin t/t, c[3], c[4] the
    // Q-block coefficients) and their θ-DERIVATIVES dc[5]. Series below t=0.2.
    __device__ __forceinline__ void se3_d2_coefs(double t, double c[5], double dc[5]) {
        if (t >= 0.2) {
            double s = sin(t), co = cos(t), t2 = t*t, t3 = t2*t, t4 = t3*t, t5 = t4*t, t6 = t5*t;
            c[0] = (1.0 - co)/t2; c[1] = (t - s)/t3; c[2] = s/t;
            c[3] = (1.0 - 0.5*t2 - co)/t4;
            c[4] = -0.5*(c[3] - 3.0*(t - s - t3/6.0)/t5);
            dc[0] = (t*s - 2.0*(1.0 - co))/t3;
            dc[1] = ((1.0 - co)*t - 3.0*(t - s))/t4;
            dc[2] = (t*co - s)/t2;
            dc[3] = (t*s + t2 + 4.0*co - 4.0)/t5;
            dc[4] = -0.5*(dc[3] - 3.0*(-4.0*t - t*co + 5.0*s + t3/3.0)/t6);
        } else {
            double x = t*t;   // even series (value) / odd series (derivative)
            c[0] = 0.5 + x*(-1.0/24 + x*(1.0/720 + x*(-1.0/40320 + x*(1.0/3628800))));
            c[1] = 1.0/6 + x*(-1.0/120 + x*(1.0/5040 + x*(-1.0/362880 + x*(1.0/39916800))));
            c[2] = 1.0 + x*(-1.0/6 + x*(1.0/120 + x*(-1.0/5040 + x*(1.0/362880))));
            c[3] = -1.0/24 + x*(1.0/720 + x*(-1.0/40320 + x*(1.0/3628800 + x*(-1.0/479001600))));
            c[4] = 1.0/120 + x*(-1.0/2520 + x*(1.0/120960 + x*(-1.0/9979200 + x*(1.0/1245404160))));
            dc[0] = t*(-1.0/12 + x*(1.0/180 + x*(-1.0/6720 + x*(1.0/453600))));
            dc[1] = t*(-1.0/60 + x*(1.0/1260 + x*(-1.0/60480 + x*(1.0/4989600))));
            dc[2] = t*(-1.0/3 + x*(1.0/30 + x*(-1.0/840 + x*(1.0/45360))));
            dc[3] = t*(1.0/360 + x*(-1.0/10080 + x*(1.0/604800 + x*(-1.0/59875200))));
            dc[4] = t*(-1.0/1260 + x*(1.0/30240 + x*(-1.0/1663200 + x*(1.0/155675520))));
        }
    }

    // d Jr(φ)/d φ_k (3x3). Structured chain rule: coeff'(θ)·φ_k/θ on the
    // scalars plus the skew-product derivatives.
    __device__ __forceinline__ void se3_d2_dJr(const double phi[3], double t,
                                               const double c[5], const double dc[5],
                                               int k, double out[9]) {
        double S[9]; skew_core<double>(phi, S);
        double ek[3] = {0, 0, 0}; ek[k] = 1.0;
        double Ek[9]; skew_core<double>(ek, Ek);
        double S2[9];  mat3_mul_core<double>(S, S, S2);
        double EkS[9]; mat3_mul_core<double>(Ek, S, EkS);
        double SEk[9]; mat3_mul_core<double>(S, Ek, SEk);
        double invt = (t > 1e-30) ? 1.0/t : 0.0;
        double da = dc[0]*phi[k]*invt, db = dc[1]*phi[k]*invt;
        #pragma unroll
        for (uint32_t i = 0; i < 9; ++i)
            out[i] = -da*S[i] - c[0]*Ek[i] + db*S2[i] + c[1]*(EkS[i] + SEk[i]);
    }

    // d exp(−φ)/d φ_k (3x3).
    __device__ __forceinline__ void se3_d2_dRinv(const double phi[3], double t,
                                                 const double c[5], const double dc[5],
                                                 int k, double out[9]) {
        double S[9]; skew_core<double>(phi, S);
        double ek[3] = {0, 0, 0}; ek[k] = 1.0;
        double Ek[9]; skew_core<double>(ek, Ek);
        double S2[9];  mat3_mul_core<double>(S, S, S2);
        double EkS[9]; mat3_mul_core<double>(Ek, S, EkS);
        double SEk[9]; mat3_mul_core<double>(S, Ek, SEk);
        double invt = (t > 1e-30) ? 1.0/t : 0.0;
        double ds = dc[2]*phi[k]*invt, da = dc[0]*phi[k]*invt;
        #pragma unroll
        for (uint32_t i = 0; i < 9; ++i)
            out[i] = -ds*S[i] - c[2]*Ek[i] + da*S2[i] + c[0]*(EkS[i] + SEk[i]);
    }

    // d Q(ρ,φ)/d w_kk (kk<3 → ρ direction, exact; else φ direction).
    __device__ __forceinline__ void se3_d2_dQ(const double rho[3], const double phi[3],
                                              double t, const double c[5], const double dc[5],
                                              int kk, double out[9]) {
        double nphi[3] = {-phi[0], -phi[1], -phi[2]};
        double Px[9];  skew_core<double>(nphi, Px);
        double Px2[9]; mat3_mul_core<double>(Px, Px, Px2);
        if (kk < 3) {
            double ek[3] = {0, 0, 0}; ek[kk] = 1.0;
            double Ek[9]; skew_core<double>(ek, Ek);
            double PxEk[9];    mat3_mul_core<double>(Px, Ek, PxEk);
            double EkPx[9];    mat3_mul_core<double>(Ek, Px, EkPx);
            double PxEkPx[9];  mat3_mul_core<double>(PxEk, Px, PxEkPx);
            double Px2Ek[9];   mat3_mul_core<double>(Px2, Ek, Px2Ek);
            double EkPx2[9];   mat3_mul_core<double>(Ek, Px2, EkPx2);
            double PxEkPx2[9]; mat3_mul_core<double>(PxEk, Px2, PxEkPx2);
            double Px2EkPx[9]; mat3_mul_core<double>(Px2Ek, Px, Px2EkPx);
            #pragma unroll
            for (uint32_t i = 0; i < 9; ++i)
                out[i] = -(0.5*Ek[i] + c[1]*(PxEk[i] + EkPx[i] + PxEkPx[i])
                         - c[3]*(Px2Ek[i] + EkPx2[i] - 3.0*PxEkPx[i])
                         + c[4]*(PxEkPx2[i] + Px2EkPx[i]));
            return;
        }
        int k = kk - 3;
        double Rx[9]; skew_core<double>(rho, Rx);
        double ek[3] = {0, 0, 0}; ek[k] = 1.0;
        double Ek[9]; skew_core<double>(ek, Ek);
        double dPx[9];
        #pragma unroll
        for (uint32_t i = 0; i < 9; ++i) dPx[i] = -Ek[i];
        double dPx2a[9]; mat3_mul_core<double>(dPx, Px, dPx2a);
        double dPx2b[9]; mat3_mul_core<double>(Px, dPx, dPx2b);
        double dPx2[9];
        #pragma unroll
        for (uint32_t i = 0; i < 9; ++i) dPx2[i] = dPx2a[i] + dPx2b[i];
        // base products for A1, A2, A3
        double PxRx[9];    mat3_mul_core<double>(Px, Rx, PxRx);
        double RxPx[9];    mat3_mul_core<double>(Rx, Px, RxPx);
        double PxRxPx[9];  mat3_mul_core<double>(PxRx, Px, PxRxPx);
        double Px2Rx[9];   mat3_mul_core<double>(Px2, Rx, Px2Rx);
        double RxPx2[9];   mat3_mul_core<double>(RxPx, Px, RxPx2);
        double PxRxPx2[9]; mat3_mul_core<double>(PxRx, Px2, PxRxPx2);
        double Px2RxPx[9]; mat3_mul_core<double>(Px2Rx, Px, Px2RxPx);
        // derivative products
        double dPxRx[9];    mat3_mul_core<double>(dPx, Rx, dPxRx);
        double RxdPx[9];    mat3_mul_core<double>(Rx, dPx, RxdPx);
        double dPxRxPx[9];  mat3_mul_core<double>(dPxRx, Px, dPxRxPx);
        double PxRxdPx[9];  mat3_mul_core<double>(PxRx, dPx, PxRxdPx);
        double dPx2Rx[9];   mat3_mul_core<double>(dPx2, Rx, dPx2Rx);
        double RxdPx2[9];   mat3_mul_core<double>(Rx, dPx2, RxdPx2);
        double dPxRxPx2[9]; mat3_mul_core<double>(dPxRx, Px2, dPxRxPx2);
        double PxRxdPx2[9]; mat3_mul_core<double>(PxRx, dPx2, PxRxdPx2);
        double dPx2RxPx[9]; mat3_mul_core<double>(dPx2Rx, Px, dPx2RxPx);
        double Px2RxdPx[9]; mat3_mul_core<double>(Px2Rx, dPx, Px2RxdPx);
        double invt = (t > 1e-30) ? 1.0/t : 0.0;
        double dc1 = dc[1]*phi[k]*invt, dc2 = dc[3]*phi[k]*invt, dc3 = dc[4]*phi[k]*invt;
        #pragma unroll
        for (uint32_t i = 0; i < 9; ++i) {
            double A1 = PxRx[i] + RxPx[i] + PxRxPx[i];
            double A2 = Px2Rx[i] + RxPx2[i] - 3.0*PxRxPx[i];
            double A3 = PxRxPx2[i] + Px2RxPx[i];
            double dA1 = dPxRx[i] + RxdPx[i] + dPxRxPx[i] + PxRxdPx[i];
            double dA2 = dPx2Rx[i] + RxdPx2[i] - 3.0*(dPxRxPx[i] + PxRxdPx[i]);
            double dA3 = (dPxRxPx2[i] + PxRxdPx2[i]) + (dPx2RxPx[i] + Px2RxdPx[i]);
            out[i] = -(dc1*A1 + c[1]*dA1 - dc2*A2 - c[3]*dA2 + dc3*A3 + c[4]*dA3);
        }
    }

    // d off/d w_kk, off = skew(−Jr·ρ)·exp(−φ) (the base-pose coupling block).
    __device__ __forceinline__ void se3_d2_doff(const double rho[3], const double phi[3],
                                                double t, const double c[5], const double dc[5],
                                                int kk, double out[9]) {
        double S[9];  skew_core<double>(phi, S);
        double S2[9]; mat3_mul_core<double>(S, S, S2);
        double Jr[9], R[9];
        #pragma unroll
        for (uint32_t i = 0; i < 9; ++i) {
            double id = (i % 4 == 0) ? 1.0 : 0.0;
            Jr[i] = id - c[0]*S[i] + c[1]*S2[i];
            R[i]  = id - c[2]*S[i] + c[0]*S2[i];
        }
        if (kk < 3) {
            // column kk of −Jr (column-major storage: Jr[kk*3 + r])
            double col[3] = {-Jr[kk*3], -Jr[kk*3 + 1], -Jr[kk*3 + 2]};
            double Sc[9]; skew_core<double>(col, Sc);
            mat3_mul_core<double>(Sc, R, out);
            return;
        }
        int k = kk - 3;
        double p[3]; mat3_vec_core<double>(Jr, rho, p);
        p[0] = -p[0]; p[1] = -p[1]; p[2] = -p[2];
        double dJr[9]; se3_d2_dJr(phi, t, c, dc, k, dJr);
        double dp[3];  mat3_vec_core<double>(dJr, rho, dp);
        dp[0] = -dp[0]; dp[1] = -dp[1]; dp[2] = -dp[2];
        double dR[9]; se3_d2_dRinv(phi, t, c, dc, k, dR);
        double Sdp[9]; skew_core<double>(dp, Sdp);
        double Sp[9];  skew_core<double>(p, Sp);
        double t1[9]; mat3_mul_core<double>(Sdp, R, t1);
        double t2[9]; mat3_mul_core<double>(Sp, dR, t2);
        #pragma unroll
        for (uint32_t i = 0; i < 9; ++i) out[i] = t1[i] + t2[i];
    }

    // serial per-direction slice of the Hessian: fills J2 slice k (36 entries,
    // column-major 6x6) = ∂J/∂w_k for J the IS_Q ? jacobian_q : jacobian_v block.
    template <typename T, bool IS_Q>
    __device__ __forceinline__ void se3_retract_hessian_slice(
        const double rho[3], const double phi[3], double t,
        const double c[5], const double dc[5], int k, T *J2_slice) {
        double dA[9], dB[9];
        if constexpr (IS_Q) {
            if (k >= 3) se3_d2_dRinv(phi, t, c, dc, k - 3, dA);
            else { for (uint32_t i = 0; i < 9; ++i) dA[i] = 0.0; }
            se3_d2_doff(rho, phi, t, c, dc, k, dB);
        } else {
            if (k >= 3) se3_d2_dJr(phi, t, c, dc, k - 3, dA);
            else { for (uint32_t i = 0; i < 9; ++i) dA[i] = 0.0; }
            se3_d2_dQ(rho, phi, t, c, dc, k, dB);
        }
        // column-major [[dA, dB],[0, dA]]
        #pragma unroll
        for (uint32_t cc = 0; cc < 3; ++cc) {
            #pragma unroll
            for (uint32_t r = 0; r < 3; ++r) {
                J2_slice[cc*6 + r]           = static_cast<T>(dA[cc*3 + r]);
                J2_slice[(cc + 3)*6 + r]     = static_cast<T>(dB[cc*3 + r]);
                J2_slice[cc*6 + r + 3]       = static_cast<T>(0);
                J2_slice[(cc + 3)*6 + r + 3] = static_cast<T>(dA[cc*3 + r]);
            }
        }
    }

    // tier-shared body: Hessian parallelized over the 6 tangent directions.
    template <typename T, bool IS_Q>
    __device__ __forceinline__ void se3_retract_hessian_impl(
        uint32_t rank, uint32_t size, const T *rho, const T *phi, T *J2) {
        const double rd[3] = {(double)rho[0], (double)rho[1], (double)rho[2]};
        const double pd[3] = {(double)phi[0], (double)phi[1], (double)phi[2]};
        const double t = sqrt(pd[0]*pd[0] + pd[1]*pd[1] + pd[2]*pd[2]);
        double c[5], dc[5]; se3_d2_coefs(t, c, dc);
        // unroll 1: unroll copies of this body may contract FMAs differently,
        // which would break bit-identity across thread counts (slice k as
        // "iteration 2 of one thread" vs "iteration 1 of thread k").
        #pragma unroll 1
        for (uint32_t k = rank; k < 6; k += size)
            se3_retract_hessian_slice<T, IS_Q>(rd, pd, t, c, dc, (int)k, J2 + k*36);
    }
} // namespace lie_detail

/**
 * @brief Barfoot Q coupling block of the SE(3) exponential derivative (3x3).
 *
 * The off-diagonal block of the SE(3) right Jacobian
 * `[[Jr(φ), Q(ρ,φ)],[0, Jr(φ)]]`; Pinocchio sign convention (validated against
 * `pin.dIntegrate` ARG1 in the GRiD arc this is promoted from). Series form
 * below θ = 1e-4.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam TRAILING_SYNC  Emit a trailing `__syncthreads()` (default true).
 * @param rho  Linear tangent (3 elements).
 * @param phi  Angular tangent (3 elements).
 * @param Q    Output 3x3 block (9 elements, column-major; no aliasing).
 */
template <typename T, bool TRAILING_SYNC = true>
__device__ void se3_Q_block(const T *rho, const T *phi, T *Q)
{
    uint32_t rank = flat_rank();
    uint32_t size = flat_size();
    T tmp[9]; lie_detail::se3_Q_block_core(rho, phi, tmp);
    lie_detail::copy_out<T, 9>(rank, size, tmp, Q);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief SE(3) retract on a `[p(3); quat(4)]` pose block:
 *        `pose_new = pose ⊞ (ρ, φ)`.
 *
 * The free-flyer manifold update (one floating-base integrator step is
 * `se3_retract(pose, v_lin·dt, ω·dt, pose_new)` with a BODY-frame twist):
 * orientation `q_new = normalize(q ⊗ exp([φ/2]))`, position
 * `p_new = p + R(q)·(Jl(φ)·ρ)`. Matches Pinocchio's `integrate` on the
 * free-flyer joint. Joint-space tails (revolute q += v·dt) are a plain vector
 * add — compose with `glass::axpy`, they are not this op's job.
 *
 * @tparam T  Scalar type.
 * @tparam L  Quaternion layout inside the pose block (default `xyzw` — the
 *            Pinocchio nq layout `[x,y,z, qx,qy,qz,qw]`).
 * @tparam TRAILING_SYNC  Emit a trailing `__syncthreads()` (default true).
 * @param pose      Input pose block (7 elements: position then quaternion).
 * @param rho       Linear tangent step (3 elements, body frame).
 * @param phi       Angular tangent step (3 elements, body frame).
 * @param pose_new  Output pose block (7 elements; no aliasing at block/warp scope).
 */
template <typename T, QuatLayout L = QuatLayout::xyzw, bool TRAILING_SYNC = true>
__device__ void se3_retract(const T *pose, const T *rho, const T *phi, T *pose_new)
{
    uint32_t rank = flat_rank();
    uint32_t size = flat_size();
    T tmp[7]; lie_detail::se3_retract_core<T, L>(pose, rho, phi, tmp);
    lie_detail::copy_out<T, 7>(rank, size, tmp, pose_new);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief SE(3) difference (boxminus): the tangent `[ρ; φ]` with
 *        `se3_retract(pose_from, ρ, φ) == pose_to` — the exact inverse of the
 *        retract, equal to Pinocchio `difference(model, q_from, q_to)` on the
 *        free-flyer block. Canonical branch |φ| ≤ π.
 *
 * `φ = log(q_from⁻¹ ⊗ q_to)`, `ρ = Jl(φ)⁻¹ · R(q_from)ᵀ · (p_to − p_from)`
 * (body-frame tangent, linear-first — the same conventions as the retract).
 *
 * @tparam T  Scalar type.
 * @tparam L  Quaternion layout inside the pose blocks (default `xyzw`).
 * @tparam TRAILING_SYNC  Emit a trailing `__syncthreads()` (default true).
 * @param pose_from  Base pose block (7 elements: position then quaternion).
 * @param pose_to    Target pose block (7 elements).
 * @param rho  Output linear tangent (3 elements; no aliasing at block scope).
 * @param phi  Output angular tangent (3 elements; no aliasing at block scope).
 */
template <typename T, QuatLayout L = QuatLayout::xyzw, bool TRAILING_SYNC = true>
__device__ void se3_difference(const T *pose_from, const T *pose_to, T *rho, T *phi)
{
    uint32_t rank = flat_rank();
    uint32_t size = flat_size();
    T tr[3], tp[3]; lie_detail::se3_difference_core<T, L>(pose_from, pose_to, tr, tp);
    lie_detail::copy_out<T, 3>(rank, size, tr, rho);
    lie_detail::copy_out<T, 3>(rank, size, tp, phi);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief 6x6 derivative of the SE(3) retract w.r.t. the BASE POSE:
 *        `J = Ad_{exp(−[ρ;φ])}` (Pinocchio `dIntegrate` ARG0).
 *
 * Column-major, tangent ordered `[ρ; φ]` (linear-first): block form
 * `[[R⁻, [−Jl(−φ)ρ]ₓ·R⁻],[0, R⁻]]` with `R⁻ = exp(−[φ]ₓ)`.
 *
 * @tparam T,TRAILING_SYNC  See `se3_Q_block`.
 * @param rho  Linear tangent (3 elements).
 * @param phi  Angular tangent (3 elements).
 * @param J    Output 6x6 block (36 elements, column-major; no aliasing).
 */
template <typename T, bool TRAILING_SYNC = true>
__device__ void se3_retract_jacobian_q(const T *rho, const T *phi, T *J)
{
    uint32_t rank = flat_rank();
    uint32_t size = flat_size();
    T tmp[36]; lie_detail::se3_retract_jacobian_q_core(rho, phi, tmp);
    lie_detail::copy_out<T, 36>(rank, size, tmp, J);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief 6x6 derivative of the SE(3) retract w.r.t. the TANGENT: the SE(3)
 *        right Jacobian `J = [[Jr(φ), Q(ρ,φ)],[0, Jr(φ)]]` (Pinocchio
 *        `dIntegrate` ARG1).
 *
 * Column-major, tangent ordered `[ρ; φ]` (linear-first).
 *
 * @tparam T,TRAILING_SYNC  See `se3_Q_block`.
 * @param rho  Linear tangent (3 elements).
 * @param phi  Angular tangent (3 elements).
 * @param J    Output 6x6 block (36 elements, column-major; no aliasing).
 */
template <typename T, bool TRAILING_SYNC = true>
__device__ void se3_retract_jacobian_v(const T *rho, const T *phi, T *J)
{
    uint32_t rank = flat_rank();
    uint32_t size = flat_size();
    T tmp[36]; lie_detail::se3_retract_jacobian_v_core(rho, phi, tmp);
    lie_detail::copy_out<T, 36>(rank, size, tmp, J);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief 6x6x6 second derivative of the SE(3) retract:
 *        `J2[k·36 + c·6 + r] = ∂J(r,c)/∂w_k`, w = [ρ; φ].
 *
 * The closed-form SE(3)-exp second derivative (structured chain rule on the
 * Rodrigues/Q coefficient series; small-angle series below θ = 0.2), for
 * `IS_Q = true` differentiating the base-pose block (`se3_retract_jacobian_q`)
 * and for `IS_Q = false` the tangent block (`se3_retract_jacobian_v`).
 * Computed in DOUBLE internally regardless of `T` (tiny once-per-step block;
 * keeps float32 kernels matching float64 oracles — the validated design this
 * is promoted from, mpmath-complex-step ground truth ≈1e-14). Layout: six
 * stacked column-major 6x6 slices, one per tangent direction `k`.
 *
 * Parallelism: the block/warp tiers stride the 6 directions over the active
 * threads (each direction's slice is computed serially by one thread).
 *
 * @tparam T  Scalar type of the interface (output cast from double).
 * @tparam IS_Q  Differentiate the ARG0 (base-pose) block instead of ARG1.
 * @tparam TRAILING_SYNC  Emit a trailing `__syncthreads()` (default true).
 * @param rho  Linear tangent (3 elements).
 * @param phi  Angular tangent (3 elements).
 * @param J2   Output tensor (216 elements; no aliasing).
 */
template <typename T, bool IS_Q, bool TRAILING_SYNC = true>
__device__ void se3_retract_hessian(const T *rho, const T *phi, T *J2)
{
    uint32_t rank = flat_rank();
    uint32_t size = flat_size();
    lie_detail::se3_retract_hessian_impl<T, IS_Q>(rank, size, rho, phi, J2);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

// ═══════════════════════════════════════════════════════════════════════
// warp:: — one warp per problem (32 lanes, __shfl_*_sync)
// ═══════════════════════════════════════════════════════════════════════

namespace warp {
    /** @brief Single-warp Barfoot Q block. See `glass::se3_Q_block`. */
    template <typename T>
    __device__ void se3_Q_block(const T *rho, const T *phi, T *Q)
    {
        uint32_t lane = (flat_rank()) & 31;
        T tmp[9]; lie_detail::se3_Q_block_core(rho, phi, tmp);
        lie_detail::copy_out<T, 9>(lane, 32u, tmp, Q);
        __syncwarp();
    }

    /** @brief Single-warp SE(3) retract. See `glass::se3_retract`. */
    template <typename T, QuatLayout L = QuatLayout::xyzw>
    __device__ void se3_retract(const T *pose, const T *rho, const T *phi, T *pose_new)
    {
        uint32_t lane = (flat_rank()) & 31;
        T tmp[7]; lie_detail::se3_retract_core<T, L>(pose, rho, phi, tmp);
        lie_detail::copy_out<T, 7>(lane, 32u, tmp, pose_new);
        __syncwarp();
    }

    /** @brief Single-warp SE(3) difference (boxminus). See `glass::se3_difference`. */
    template <typename T, QuatLayout L = QuatLayout::xyzw>
    __device__ void se3_difference(const T *pose_from, const T *pose_to, T *rho, T *phi)
    {
        uint32_t lane = (flat_rank()) & 31;
        T tr[3], tp[3]; lie_detail::se3_difference_core<T, L>(pose_from, pose_to, tr, tp);
        lie_detail::copy_out<T, 3>(lane, 32u, tr, rho);
        lie_detail::copy_out<T, 3>(lane, 32u, tp, phi);
        __syncwarp();
    }

    /** @brief Single-warp retract Jacobian w.r.t. the base pose. See `glass::se3_retract_jacobian_q`. */
    template <typename T>
    __device__ void se3_retract_jacobian_q(const T *rho, const T *phi, T *J)
    {
        uint32_t lane = (flat_rank()) & 31;
        T tmp[36]; lie_detail::se3_retract_jacobian_q_core(rho, phi, tmp);
        lie_detail::copy_out<T, 36>(lane, 32u, tmp, J);
        __syncwarp();
    }

    /** @brief Single-warp retract Jacobian w.r.t. the tangent. See `glass::se3_retract_jacobian_v`. */
    template <typename T>
    __device__ void se3_retract_jacobian_v(const T *rho, const T *phi, T *J)
    {
        uint32_t lane = (flat_rank()) & 31;
        T tmp[36]; lie_detail::se3_retract_jacobian_v_core(rho, phi, tmp);
        lie_detail::copy_out<T, 36>(lane, 32u, tmp, J);
        __syncwarp();
    }

    /** @brief Single-warp retract Hessian (double internals). See `glass::se3_retract_hessian`. */
    template <typename T, bool IS_Q>
    __device__ void se3_retract_hessian(const T *rho, const T *phi, T *J2)
    {
        uint32_t lane = (flat_rank()) & 31;
        lie_detail::se3_retract_hessian_impl<T, IS_Q>(lane, 32u, rho, phi, J2);
        __syncwarp();
    }
}

// ═══════════════════════════════════════════════════════════════════════
// thread:: — one problem per thread (serial, register-resident)
// ═══════════════════════════════════════════════════════════════════════

namespace thread {
    /** @brief Single-thread Barfoot Q block. See `glass::se3_Q_block`. */
    template <typename T>
    __device__ void se3_Q_block(const T *rho, const T *phi, T *Q)
    { lie_detail::se3_Q_block_core(rho, phi, Q); }

    /** @brief Single-thread SE(3) retract. See `glass::se3_retract`. */
    template <typename T, QuatLayout L = QuatLayout::xyzw>
    __device__ void se3_retract(const T *pose, const T *rho, const T *phi, T *pose_new)
    {
        T tmp[7]; lie_detail::se3_retract_core<T, L>(pose, rho, phi, tmp);
        for (uint32_t i = 0; i < 7; i++) pose_new[i] = tmp[i];
    }

    /** @brief Single-thread SE(3) difference (boxminus). See `glass::se3_difference`. */
    template <typename T, QuatLayout L = QuatLayout::xyzw>
    __device__ void se3_difference(const T *pose_from, const T *pose_to, T *rho, T *phi)
    { lie_detail::se3_difference_core<T, L>(pose_from, pose_to, rho, phi); }

    /** @brief Single-thread retract Jacobian w.r.t. the base pose. See `glass::se3_retract_jacobian_q`. */
    template <typename T>
    __device__ void se3_retract_jacobian_q(const T *rho, const T *phi, T *J)
    { lie_detail::se3_retract_jacobian_q_core(rho, phi, J); }

    /** @brief Single-thread retract Jacobian w.r.t. the tangent. See `glass::se3_retract_jacobian_v`. */
    template <typename T>
    __device__ void se3_retract_jacobian_v(const T *rho, const T *phi, T *J)
    { lie_detail::se3_retract_jacobian_v_core(rho, phi, J); }

    /** @brief Single-thread retract Hessian (double internals). See `glass::se3_retract_hessian`. */
    template <typename T, bool IS_Q>
    __device__ void se3_retract_hessian(const T *rho, const T *phi, T *J2)
    { lie_detail::se3_retract_hessian_impl<T, IS_Q>(0u, 1u, rho, phi, J2); }
}
