#pragma once
#include <cstdint>

// ─── Riccati feedback gain  K = (R + BᵀPB)⁻¹ (BᵀPA) ──────────────────────────
//
// The control-update solve at the heart of an LQR / iLQR backward pass, composed
// from the library's own primitives: a symmetric congruence (R + BᵀPB), a
// bilinear form (BᵀPA), and a (optionally regularized, checked) SPD solve. One
// block, column-major. Requires congruence.cuh + posv.cuh (included first).

/**
 * @brief Shared-memory floats needed by `riccati_gain` `s_scratch`.
 *
 * Holds the NU×NU control-Hessian `S = R + BᵀPB` plus the larger of the two
 * congruence/bilinear products (`P·B` is NX×NU, `P·A` is NX×NX).
 *
 * @tparam NX  State dimension.
 * @tparam NU  Control dimension.
 * @return Number of `T` elements for `s_scratch`.
 */
template <uint32_t NX, uint32_t NU>
__host__ __device__ constexpr uint32_t riccati_smem_count() {
    return NU*NU + NX * (NX >= NU ? NX : NU);
}

/**
 * @brief LQR/iLQR feedback gain: `K = (R + BᵀPB)⁻¹ (BᵀPA)`.
 *
 * Forms the control Hessian `S = R + BᵀPB` (symmetric congruence), the coupling
 * `G = BᵀPA` (bilinear), then solves `S·K = G` for the `NU×NX` gain by Cholesky
 * (multi-RHS). With `REGULARIZE`, shifts `S` by `rho·I` before factoring (and
 * always reports a non-PD `S` via `s_fail`) so an iLQR caller can escalate `rho`
 * and retry. Single block, column-major; thread-count invariant within the
 * surface. On return `Kgain` holds `K` (the inputs `P,A,B,R` are unchanged).
 *
 * @tparam T  Scalar type (prefer `double` for ill-conditioned `S`).
 * @tparam NX  State dimension (`P` is NX×NX, `A` is NX×NX, `B` is NX×NU).
 * @tparam NU  Control dimension (`R` is NU×NU, `K` is NU×NX). Assumes `NX >= NU`.
 * @tparam REGULARIZE  If true, add `rho·I` to `S` before the solve (default false).
 * @tparam TRAILING_SYNC  Emit a trailing `__syncthreads()` (default true).
 * @param P  Cost-to-go Hessian (NX×NX, symmetric, column-major).
 * @param A  State Jacobian (NX×NX, column-major).
 * @param B  Control Jacobian (NX×NU, column-major).
 * @param R  Control cost (NU×NU, SPD, column-major).
 * @param Kgain  Out gain `K` (NU×NX, column-major).
 * @param s_scratch  Shared scratch of `riccati_smem_count<NX,NU>()` elements.
 * @param rho     Diagonal shift on `S` when REGULARIZE (ignored otherwise).
 * @param s_fail  Optional flag: set to 1 if `S` (after the shift) is not PD, else 0.
 */
template <typename T, uint32_t NX, uint32_t NU,
          bool REGULARIZE = false, bool TRAILING_SYNC = true>
__device__ void riccati_gain(const T* P, const T* A, const T* B, const T* R,
                             T* Kgain, T* s_scratch, T rho = T(0), int* s_fail = nullptr)
{
    T* S   = s_scratch;                 // NU x NU control Hessian
    T* scr = s_scratch + NU*NU;         // congruence/bilinear product scratch

    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t i = rank; i < NU*NU; i += size) S[i] = R[i];   // S = R
    __syncthreads();

    // S += Bᵀ·P·B  (symmetric congruence, accumulate onto S=R)
    congruence_sym<T, NX, NU, /*ACCUMULATE=*/true>(static_cast<T>(1), B, P, static_cast<T>(1), S, scr);
    // G = Bᵀ·P·A  -> Kgain  (general bilinear, NU x NX)
    bilinear<T, NX, NU, NX>(static_cast<T>(1), B, P, A, static_cast<T>(0), Kgain, scr);
    // solve S·K = G in place on Kgain (NX right-hand sides); checked + optional shift
    posv<T, NU, NX, REGULARIZE, /*CHECK=*/true>(S, Kgain, rho, s_fail);

    if constexpr (TRAILING_SYNC) __syncthreads();
}

namespace warp {
    /**
     * @brief Single-warp LQR/iLQR feedback gain `K = (R + BᵀPB)⁻¹ (BᵀPA)`.
     *
     * Warp-per-knot parity with the block `glass::riccati_gain`: one 32-lane warp
     * forms `S = R + BᵀPB` (`warp::congruence_sym`), `G = BᵀPA` (`warp::bilinear`),
     * then solves `S·K = G` for the `NU×NX` gain with the checked, optionally
     * regularized `warp::posv` (NRHS=NX). Every sub-op is `__syncwarp`-scoped, so
     * independent warps may run distinct knots of a batched backward pass
     * concurrently in one block. On return `Kgain` holds `K`; `P,A,B,R` unchanged.
     *
     * @tparam T  Scalar type (prefer `double` for ill-conditioned `S`).
     * @tparam NX  State dimension (`P`,`A` are NX×NX, `B` is NX×NU).
     * @tparam NU  Control dimension (`R` is NU×NU, `K` is NU×NX). Assumes `NX >= NU`.
     * @tparam REGULARIZE  If true, add `rho·I` to `S` before the solve (default false).
     * @tparam TRAILING_SYNC  Emit a trailing `__syncwarp()` (default true).
     * @param P,A,B,R  Inputs (column-major; see the block overload).
     * @param Kgain  Out gain `K` (NU×NX, column-major).
     * @param s_scratch Shared scratch of `riccati_smem_count<NX,NU>()` elements (per warp).
     * @param rho    Diagonal shift on `S` when REGULARIZE (ignored otherwise).
     * @param s_fail Optional flag: set to 1 if `S` (after the shift) is not PD, else 0.
     */
    template <typename T, uint32_t NX, uint32_t NU,
              bool REGULARIZE = false, bool TRAILING_SYNC = true>
    __device__ void riccati_gain(const T* P, const T* A, const T* B, const T* R,
                                 T* Kgain, T* s_scratch, T rho = T(0), int* s_fail = nullptr)
    {
        T* S   = s_scratch;                 // NU x NU control Hessian
        T* scr = s_scratch + NU*NU;         // congruence/bilinear product scratch

        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        for (uint32_t i = lane; i < NU*NU; i += 32) S[i] = R[i];   // S = R
        __syncwarp();

        congruence_sym<T, NX, NU, /*ACCUMULATE=*/true>(static_cast<T>(1), B, P, static_cast<T>(1), S, scr);
        bilinear<T, NX, NU, NX>(static_cast<T>(1), B, P, A, static_cast<T>(0), Kgain, scr);
        posv<T, NU, NX, REGULARIZE, /*CHECK=*/true>(S, Kgain, rho, s_fail);

        if constexpr (TRAILING_SYNC) __syncwarp();
    }
}
