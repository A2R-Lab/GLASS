// ============================================================================
// INTERNAL / NOT EXPORTED — single-block box-constrained QP solver.
//
//   minimize   0.5 xᵀ P x + qᵀ x      subject to   l ≤ x ≤ u
//
// Projected-gradient descent with Armijo backtracking line search. One CUDA
// block solves one QP; threads cooperate over the GLASS L1/L3 primitives.
// Intended for small n (e.g. DDP forward-pass control bounds). P is assumed
// symmetric positive-definite (a QP Hessian).
//
// VALIDATED (2026-06-15) by test/test_qp.py: matches a SciPy bounded-min
// reference and the closed-form unconstrained / separable cases, satisfies the
// projected-gradient KKT condition, is thread-count invariant (1..256), and is
// compute-sanitizer clean (memcheck + racecheck). Note: plain projected gradient
// converges only linearly — fine for the small, well-conditioned control QPs it
// targets; an accelerated/active-set method would be faster (future work).
//
// This header is deliberately NOT included by glass.cuh — QP is optimization,
// not linear algebra (see docs/open-tasks/qp_solver_scope.md). It is an internal
// utility, not part of the public GLASS API.
// ============================================================================
#pragma once

#include "../base/L1/clip.cuh"
#include "../base/L1/copy.cuh"
#include "../base/L1/dot.cuh"
#include "../base/L1/infnorm.cuh"
#include "../base/L1/elementwise_logic.cuh"
#include "../base/L3/gemm.cuh"
#include <cstddef>
#include <cstdint>

namespace glass {
namespace internal {

// Tunable solver parameters (were #defines in the legacy cpqp).
template <typename T>
struct QPParams {
    int max_iter = 1000;      // outer iterations and line-search backtracks
    T   tol      = T(1e-5);   // projected-gradient inf-norm stopping tolerance
    T   alpha0   = T(1);      // initial line-search step
    T   c        = T(1e-4);   // Armijo sufficient-decrease constant
    T   beta     = T(0.5);    // backtracking shrink factor
};

// Outcome of a solve.
template <typename T>
struct QPResult {
    bool converged;   // projected-gradient inf-norm fell below tol
    int  iters;       // outer iterations performed
    T    grad_norm;   // final projected-gradient inf-norm
};

// Scratch elements (of type T) the solver needs: five length-n work vectors
// (grad, xnew, d, Px, tmp). `tmp` is the reduction scratch for the dot products
// — low_memory::dot writes the full length-n buffer and reduces into [0], so the
// accumulators must be length n, not single scalars.
template <typename T>
__host__ __device__ constexpr std::size_t box_qp_scratch_size(std::uint32_t n)
{
    return static_cast<std::size_t>(5u) * n;
}

// f(x) = 0.5 xᵀP x + qᵀx. Uses Px and tmp (each length n) as scratch. Returns
// the value (identical on every thread).
template <typename T>
__device__ T qp_objective(std::uint32_t n, T *P, T *x, T *q, T *Px, T *tmp)
{
    // Px = P @ x  (n×n · n×1). P symmetric ⇒ xᵀP x == xᵀ(P x).
    gemm<T>(n, n, 1, (T)1, P, x, Px);
    __syncthreads();
    low_memory::dot<T>(n, Px, x, tmp);     // tmp[0] = xᵀP x (uses tmp[0..n-1])
    T xPx = tmp[0];
    low_memory::dot<T>(n, q, x, tmp);      // tmp[0] = qᵀx
    T qx = tmp[0];
    return (T)0.5 * xPx + qx;
}

// Solve the box QP. x: in = initial guess, out = solution (in-place, projected
// onto [l,u]). scratch must hold >= box_qp_scratch_size<T>(n) elements.
template <typename T>
__device__ QPResult<T> box_qp(std::uint32_t n, T *P, T *q, T *l, T *u, T *x,
                              T *scratch, const QPParams<T> &params = QPParams<T>())
{
    // Carve the scratch arena (five length-n buffers).
    T *grad = scratch;          // gradient at x (persists across line search)
    T *xnew = grad + n;         // candidate iterate / projected point
    T *d    = xnew + n;         // step (xnew - x) or projected-gradient vector
    T *Px   = d + n;            // objective P@x scratch
    T *tmp  = Px + n;           // length-n reduction scratch for dot products

    // Feasible start.
    clip<T>(n, x, l, u);
    __syncthreads();

    QPResult<T> out{false, 0, T(0)};

    for (int it = 0; it < params.max_iter; ++it) {
        out.iters = it + 1;

        // grad = P x + q
        gemm<T>(n, n, 1, (T)1, P, x, grad);
        __syncthreads();
        elementwise_add<T>(n, grad, q, grad);   // grad = grad + q
        __syncthreads();

        // Optimality: projected gradient  pg = x - clip(x - grad, l, u).
        elementwise_sub<T>(n, x, grad, xnew);   // xnew = x - grad
        __syncthreads();
        clip<T>(n, xnew, l, u);                 // xnew = proj(x - grad)
        __syncthreads();
        elementwise_sub<T>(n, x, xnew, d);      // d = pg
        __syncthreads();
        infnorm<T>(n, d);                       // d[0] = ||pg||_inf (destructive)
        __syncthreads();
        out.grad_norm = d[0];
        if (d[0] <= params.tol) { out.converged = true; break; }

        // Armijo projected backtracking line search.
        T f_x = qp_objective<T>(n, P, x, q, Px, tmp);
        __syncthreads();
        T alpha = params.alpha0;
        bool stepped = false;
        for (int ls = 0; ls < params.max_iter; ++ls) {
            // xnew = proj(x - alpha * grad)
            elementwise_mult_scalar<T>(n, grad, alpha, xnew);  // xnew = alpha*grad
            __syncthreads();
            elementwise_sub<T>(n, x, xnew, xnew);              // xnew = x - alpha*grad
            __syncthreads();
            clip<T>(n, xnew, l, u);
            __syncthreads();
            elementwise_sub<T>(n, xnew, x, d);                 // d = xnew - x
            __syncthreads();
            low_memory::dot<T>(n, grad, d, tmp);               // tmp[0] = grad·d (<=0)
            __syncthreads();
            T gd = tmp[0];
            T f_new = qp_objective<T>(n, P, xnew, q, Px, tmp);
            __syncthreads();
            // Sufficient decrease: f(xnew) <= f(x) + c * grad·(xnew - x).
            if (f_new <= f_x + params.c * gd) { stepped = true; break; }
            alpha *= params.beta;
        }

        // x <- xnew (if the line search made no progress, xnew == proj(x) == x).
        copy<T>(n, xnew, x);
        __syncthreads();
        if (!stepped) {            // step collapsed to ~0; can't improve further
            out.converged = (out.grad_norm <= params.tol);
            break;
        }
    }
    return out;
}

}  // namespace internal
}  // namespace glass
