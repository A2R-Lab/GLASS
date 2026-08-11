#pragma once
#include <cstdint>
#include <cstddef>

// ─── Riccati feedback gain  K = (R + BᵀPB)⁻¹ (BᵀPA) ──────────────────────────
//
// The control-update solve at the heart of an LQR / iLQR backward pass, composed
// from the library's own primitives: a symmetric congruence (R + BᵀPB), a
// bilinear form (BᵀPA), and a (optionally regularized, checked) SPD solve. One
// block, column-major. Requires congruence.cuh + posv.cuh (included first).

/**
 * @brief Scratch size in bytes for `riccati_gain` `s_scratch`.
 *
 * Holds the NU×NU control-Hessian `S = R + BᵀPB` plus the larger of the two
 * congruence/bilinear products (`P·B` is NX×NU, `P·A` is NX×NX).
 *
 * @tparam T   Element type.
 * @tparam NX  State dimension.
 * @tparam NU  Control dimension.
 * @return Bytes to allocate for `riccati_gain`'s `s_scratch`.
 */
template <typename T, uint32_t NX, uint32_t NU>
__host__ __device__ constexpr std::size_t riccati_scratch_bytes() {
    return (NU*NU + NX * (NX >= NU ? NX : NU)) * sizeof(T);
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
 * @param s_scratch  Shared scratch of `riccati_scratch_bytes<T,NX,NU>()` bytes.
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

    uint32_t rank = flat_rank();
    uint32_t size = flat_size();
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

// ═══════════════════════════════════════════════════════════════════════
// warp:: — one warp per problem (32 lanes, __shfl_*_sync)
// ═══════════════════════════════════════════════════════════════════════

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
     * @param s_scratch Shared scratch of `riccati_scratch_bytes<T,NX,NU>()` bytes (per warp).
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

        uint32_t lane = (flat_rank()) & 31;
        for (uint32_t i = lane; i < NU*NU; i += 32) S[i] = R[i];   // S = R
        __syncwarp();

        congruence_sym<T, NX, NU, /*ACCUMULATE=*/true>(static_cast<T>(1), B, P, static_cast<T>(1), S, scr);
        bilinear<T, NX, NU, NX>(static_cast<T>(1), B, P, A, static_cast<T>(0), Kgain, scr);
        posv<T, NU, NX, REGULARIZE, /*CHECK=*/true>(S, Kgain, rho, s_fail);

        if constexpr (TRAILING_SYNC) __syncwarp();
    }
}

// ═══════════════════════════════════════════════════════════════════════
// thread:: — one problem per thread (serial, register-resident)
// ═══════════════════════════════════════════════════════════════════════

namespace thread {
    /**
     * @brief Single-thread LQR/iLQR feedback gain `K = (R + BᵀPB)⁻¹ (BᵀPA)`.
     *
     * Thread-per-knot analogue of `glass::riccati_gain`: ONE thread forms
     * `S = R + BᵀPB` (`thread::congruence_sym`), `G = BᵀPA` (`thread::bilinear`),
     * then solves `S·K = G` — composing the tier's own pieces exactly as the
     * block/warp twins compose theirs. Because the multi-RHS `posv` bodies are
     * block/warp-scoped (BlockBarrier / warp trsm) and `thread::posv` is
     * single-RHS, the solve leg is spelled out from the tier's existing
     * primitives with the identical algorithm and order the flagged `posv`
     * runs: optional `rho·I` shift → checked `thread::potrf` → per-column
     * forward/back substitution (`thread::potrs`, NX columns). No barriers, no
     * shuffles, no `threadIdx` read, so it is safe in ragged-tail
     * thread-per-problem launches; operands and `scratch` may be thread-local
     * register arrays (the implied `T[NX*NX]` P stays register-resident only
     * under the tier's N<=7 element-count ceiling — see CLAUDE.md; larger dims
     * still compute correctly but spill to local memory). Same algorithm and
     * operand order as the `glass::` twin, agreeing to a few ULP (cross-tier
     * bit-identity is NOT guaranteed). On return `Kgain` holds `K`; `P,A,B,R`
     * are unchanged.
     *
     * `scratch` is ALGORITHMIC workspace (it holds `S` and the congruence /
     * bilinear products the contraction re-reads), not cross-lane staging — the
     * thread tier keeps the caller-provided pointer, intended as a thread-local
     * array of `riccati_scratch_bytes<T,NX,NU>()` bytes. No `TRAILING_SYNC`
     * parameter, matching the tier's precedent.
     *
     * @tparam T  Scalar type (prefer `double` for ill-conditioned `S`).
     * @tparam NX  State dimension (`P`,`A` are NX×NX, `B` is NX×NU).
     * @tparam NU  Control dimension (`R` is NU×NU, `K` is NU×NX). Assumes `NX >= NU`.
     * @tparam REGULARIZE  If true, add `rho·I` to `S` before the solve (default false).
     * @param P,A,B,R  Inputs (column-major; see the block overload).
     * @param Kgain  Out gain `K` (NU×NX, column-major).
     * @param scratch  Workspace of `riccati_scratch_bytes<T,NX,NU>()` bytes (per thread).
     * @param rho    Diagonal shift on `S` when REGULARIZE (ignored otherwise).
     * @param s_fail Optional flag: set to 1 if `S` (after the shift) is not PD, else 0.
     */
    template <typename T, uint32_t NX, uint32_t NU, bool REGULARIZE = false>
    __device__ void riccati_gain(const T* P, const T* A, const T* B, const T* R,
                                 T* Kgain, T* scratch, T rho = T(0), int* s_fail = nullptr)
    {
        T* S   = scratch;                 // NU x NU control Hessian
        T* scr = scratch + NU*NU;         // congruence/bilinear product scratch

        for (uint32_t i = 0; i < NU*NU; ++i) S[i] = R[i];   // S = R

        // S += Bᵀ·P·B  (symmetric congruence, accumulate onto S=R)
        thread::congruence_sym<T, NX, NU, /*ACCUMULATE=*/true>(static_cast<T>(1), B, P, static_cast<T>(1), S, scr);
        // G = Bᵀ·P·A  -> Kgain  (general bilinear, NU x NX)
        thread::bilinear<T, NX, NU, NX>(static_cast<T>(1), B, P, A, static_cast<T>(0), Kgain, scr);
        // solve S·K = G in place on Kgain: the flagged posv path, one thread —
        // optional rho·I shift, checked factor, then NX forward/back solves.
        if constexpr (REGULARIZE) {
            for (uint32_t i = 0; i < NU; ++i) S[i*NU + i] += rho;       // rho·I (Marquardt)
        }
        thread::potrf<T, NU, /*CHECK=*/true>(S, s_fail);                // S -> L (lower)
        for (uint32_t c = 0; c < NX; ++c)
            thread::potrs<T, NU>(S, Kgain + c*NU);                      // per-column L Lᵀ x = g
    }
}
