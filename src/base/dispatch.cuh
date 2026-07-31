/**
 * @file dispatch.cuh
 * @brief The bare `glass::op` face for ops with measured non-block bodies.
 *
 * Included by glass.cuh at `namespace glass` scope, AFTER the block tier and
 * the `using namespace block;` re-export. C++ qualified lookup finds names
 * declared directly in a namespace before names reachable through
 * using-directives, so each function defined here SHADOWS the whole
 * `block::` overload set of that name on the bare face — which is why every
 * public block overload of a shadowed name must be restated here (dispatched
 * or forwarded verbatim). Ops with no moved cell (everything not in this
 * file) keep resolving straight to `glass::block::` through the directive.
 *
 * CONTRACT (see glass-dispatch.cuh): the bare face keeps the block-scope
 * calling contract — every thread of the block enters, any thread count and
 * block shape is legal, results are published block-wide on return. Only the
 * implementation BODY changes per measured (op, N, dtype) cell:
 *
 *   body::block           the block:: twin, verbatim.
 *   body::warp_in_block   warp 0 executes the warp:: twin, then __syncthreads.
 *                         Requires blockDim.x >= 32 (a full x-major first
 *                         warp); narrower launches fall back to block.
 *   body::thread_in_block thread 0 executes the thread:: twin, then
 *                         __syncthreads. Legal at every block shape.
 *
 * Dispatch applies ONLY to the configurations the body sweep measured: the
 * compile-time-size spellings, default flags (no TRANSPOSE/ROW_MAJOR/CHECK/
 * REGULARIZE), TRAILING_SYNC=true, square gemv — everything else forwards to
 * block:: verbatim (softmax dispatches on its runtime n; its table verdict is
 * bounded by the largest measured n). Numerical results of a moved cell match
 * the block tier to reduction-order tolerance, NOT bit-exactly —
 * determinism-sensitive callers pin `glass::block::` explicitly.
 *
 * MAINTENANCE: adding a public overload to a block:: op that has moved cells?
 * It must be restated here or the bare face silently loses it
 * (test_defaults.cu pins the surface).
 */

namespace dispatch_detail {

/// True when the block's x-major first warp is a full hardware warp, i.e.
/// linear ranks 0..31 are exactly threadIdx.x 0..31 of the y==z==0 slice.
__device__ __forceinline__ bool full_first_warp() { return blockDim.x >= 32u; }

/// All threads enter; warp 0 runs `f`; block-wide sync publishes the result.
template <typename F>
__device__ __forceinline__ void warp0(F f) {
    if (threadIdx.x < 32u && threadIdx.y == 0u && threadIdx.z == 0u) f();
    __syncthreads();
}

/// All threads enter; thread 0 runs `f`; block-wide sync publishes the result.
template <typename F>
__device__ __forceinline__ void thread0(F f) {
    if (threadIdx.x == 0u && threadIdx.y == 0u && threadIdx.z == 0u) f();
    __syncthreads();
}

}  // namespace dispatch_detail

// ─── dot ─────────────────────────────────────────────────────────────────────
// Contract (matches block::dot): destructive reduction scratch on y; the
// result is y[0] after return. Non-block bodies write only y[0].

/// Runtime-size spelling: always the block body (the table is keyed on
/// compile-time N; runtime-n callers wanting the measured pick should use the
/// `<T, N>` spelling).
template <typename T, bool TRAILING_SYNC = true>
__device__ void dot(uint32_t n, T *x, T *y) {
    block::dot<T, TRAILING_SYNC>(n, x, y);
}

template <typename T, uint32_t N, bool TRAILING_SYNC = true>
__device__ void dot(T *x, T *y) {
    constexpr body b = TRAILING_SYNC ? dispatch_body(op::dot, N, sizeof(T) == 8)
                                     : body::block;
    if constexpr (b == body::warp_in_block) {
        if (dispatch_detail::full_first_warp()) {
            dispatch_detail::warp0([&] {
                T r = warp::dot<T, N>(x, y);
                if (threadIdx.x == 0u) y[0] = r;
            });
        } else {
            block::dot<T, N, TRAILING_SYNC>(x, y);
        }
    } else if constexpr (b == body::thread_in_block) {
        dispatch_detail::thread0([&] { y[0] = thread::dot<T, N>(x, y); });
    } else {
        block::dot<T, N, TRAILING_SYNC>(x, y);
    }
}

// ─── gemv ────────────────────────────────────────────────────────────────────
// Only the measured configuration dispatches: compile-time square (M == N),
// column-major, no transpose, trailing sync. Everything else is block.

template <typename T, bool TRANSPOSE = false, bool ROW_MAJOR = false, bool TRAILING_SYNC = true>
__device__ void gemv(uint32_t m, uint32_t n, T alpha, const T *A, const T *x, T beta, T *y) {
    block::gemv<T, TRANSPOSE, ROW_MAJOR, TRAILING_SYNC>(m, n, alpha, A, x, beta, y);
}

template <typename T, bool TRANSPOSE = false, bool ROW_MAJOR = false, bool TRAILING_SYNC = true>
__device__ void gemv(uint32_t m, uint32_t n, T alpha, const T *A, const T *x, T *y) {
    block::gemv<T, TRANSPOSE, ROW_MAJOR, TRAILING_SYNC>(m, n, alpha, A, x, y);
}

template <typename T, uint32_t M, uint32_t N, bool TRANSPOSE = false, bool ROW_MAJOR = false, bool TRAILING_SYNC = true>
__device__ void gemv(T alpha, const T *A, const T *x, T beta, T *y) {
    constexpr bool plain = (M == N) && !TRANSPOSE && !ROW_MAJOR && TRAILING_SYNC;
    constexpr body b = plain ? dispatch_body(op::gemv, N, sizeof(T) == 8)
                             : body::block;
    if constexpr (b == body::warp_in_block) {
        if (dispatch_detail::full_first_warp()) {
            dispatch_detail::warp0([&] {
                warp::gemv<T, M, N, TRANSPOSE, ROW_MAJOR>(alpha, A, x, beta, y);
            });
        } else {
            block::gemv<T, M, N, TRANSPOSE, ROW_MAJOR, TRAILING_SYNC>(alpha, A, x, beta, y);
        }
    } else if constexpr (b == body::thread_in_block) {
        dispatch_detail::thread0([&] {
            thread::gemv<T, M, N, TRANSPOSE, ROW_MAJOR>(alpha, A, x, beta, y);
        });
    } else {
        block::gemv<T, M, N, TRANSPOSE, ROW_MAJOR, TRAILING_SYNC>(alpha, A, x, beta, y);
    }
}

template <typename T, uint32_t M, uint32_t N, bool TRANSPOSE = false, bool ROW_MAJOR = false, bool TRAILING_SYNC = true>
__device__ void gemv(T alpha, const T *A, const T *x, T *y) {
    constexpr bool plain = (M == N) && !TRANSPOSE && !ROW_MAJOR && TRAILING_SYNC;
    constexpr body b = plain ? dispatch_body(op::gemv, N, sizeof(T) == 8)
                             : body::block;
    if constexpr (b == body::warp_in_block) {
        if (dispatch_detail::full_first_warp()) {
            dispatch_detail::warp0([&] {
                warp::gemv<T, M, N, TRANSPOSE, ROW_MAJOR>(alpha, A, x, y);
            });
        } else {
            block::gemv<T, M, N, TRANSPOSE, ROW_MAJOR, TRAILING_SYNC>(alpha, A, x, y);
        }
    } else if constexpr (b == body::thread_in_block) {
        dispatch_detail::thread0([&] {
            thread::gemv<T, M, N, TRANSPOSE, ROW_MAJOR>(alpha, A, x, y);
        });
    } else {
        block::gemv<T, M, N, TRANSPOSE, ROW_MAJOR, TRAILING_SYNC>(alpha, A, x, y);
    }
}

// ─── potrf (op::chol) ────────────────────────────────────────────────────────
// Only the plain compile-time CHECK=false cell dispatches; CHECK builds and
// every runtime/K-way-fused spelling forward to block.

template <typename T, bool CHECK = false>
__device__ void potrf(uint32_t n, T *s_A, int *s_fail = nullptr) {
    block::potrf<T, CHECK>(n, s_A, s_fail);
}

template <typename T>
__device__ void potrf(uint32_t K, const uint32_t *dims, uint32_t MAX_DIM, T **mats) {
    block::potrf<T>(K, dims, MAX_DIM, mats);
}

template <typename T>
__device__ void potrf(uint32_t dimA, uint32_t dimB, uint32_t MAX_DIM, T *A, T *B) {
    block::potrf<T>(dimA, dimB, MAX_DIM, A, B);
}

template <typename T>
__device__ void potrf(uint32_t dimA, uint32_t dimB, uint32_t dimC, uint32_t MAX_DIM,
                      T *A, T *B, T *C) {
    block::potrf<T>(dimA, dimB, dimC, MAX_DIM, A, B, C);
}

template <typename T, uint32_t N, bool CHECK = false>
__device__ void potrf(T *s_A, int *s_fail = nullptr) {
    constexpr body b = CHECK ? body::block
                             : dispatch_body(op::chol, N, sizeof(T) == 8);
    if constexpr (b == body::warp_in_block) {
        if (dispatch_detail::full_first_warp()) {
            dispatch_detail::warp0([&] { warp::potrf<T, N, CHECK>(s_A, s_fail); });
        } else {
            block::potrf<T, N, CHECK>(s_A, s_fail);
        }
    } else if constexpr (b == body::thread_in_block) {
        dispatch_detail::thread0([&] { thread::potrf<T, N, CHECK>(s_A, s_fail); });
    } else {
        block::potrf<T, N, CHECK>(s_A, s_fail);
    }
}

// ─── trsv ────────────────────────────────────────────────────────────────────
// Only the measured Lower/NonUnit/no-transpose compile-time cell dispatches.

template <typename T, FillMode FILL = FillMode::Lower, Diag DIAG = Diag::NonUnit, bool TRANSPOSE = false>
__device__ void trsv(uint32_t n, const T *A, T *x) {
    block::trsv<T, FILL, DIAG, TRANSPOSE>(n, A, x);
}

template <typename T, uint32_t N, FillMode FILL = FillMode::Lower, Diag DIAG = Diag::NonUnit, bool TRANSPOSE = false>
__device__ void trsv(const T *A, T *x) {
    constexpr bool plain = (FILL == FillMode::Lower) && (DIAG == Diag::NonUnit)
                           && !TRANSPOSE;
    constexpr body b = plain ? dispatch_body(op::trsv, N, sizeof(T) == 8)
                             : body::block;
    if constexpr (b == body::warp_in_block) {
        if (dispatch_detail::full_first_warp()) {
            dispatch_detail::warp0([&] {
                warp::trsv<T, N, FILL, DIAG, TRANSPOSE>(A, x);
            });
        } else {
            block::trsv<T, N, FILL, DIAG, TRANSPOSE>(A, x);
        }
    } else if constexpr (b == body::thread_in_block) {
        dispatch_detail::thread0([&] {
            thread::trsv<T, N, FILL, DIAG, TRANSPOSE>(A, x);
        });
    } else {
        block::trsv<T, N, FILL, DIAG, TRANSPOSE>(A, x);
    }
}

// ─── posv ────────────────────────────────────────────────────────────────────
// Only the plain compile-time single-RHS cell dispatches; the flagged
// multi-RHS family and the runtime spelling forward to block.

template <typename T>
__device__ void posv(uint32_t n, T *A, T *b) {
    block::posv<T>(n, A, b);
}

template <typename T, uint32_t N>
__device__ void posv(T *A, T *b) {
    constexpr body bd = dispatch_body(op::posv, N, sizeof(T) == 8);
    if constexpr (bd == body::warp_in_block) {
        if (dispatch_detail::full_first_warp()) {
            dispatch_detail::warp0([&] { warp::posv<T, N>(A, b); });
        } else {
            block::posv<T, N>(A, b);
        }
    } else if constexpr (bd == body::thread_in_block) {
        dispatch_detail::thread0([&] { thread::posv<T, N>(A, b); });
    } else {
        block::posv<T, N>(A, b);
    }
}

template <typename T, bool REGULARIZE = false, bool CHECK = false, bool REG_DIAG = false>
__device__ void posv(uint32_t n, uint32_t nrhs, T *A, T *B, T rho = T(0),
                     int *s_fail = nullptr) {
    block::posv<T, REGULARIZE, CHECK, REG_DIAG>(n, nrhs, A, B, rho, s_fail);
}

template <typename T, uint32_t N, uint32_t NRHS, bool REGULARIZE = false, bool CHECK = false, bool REG_DIAG = false>
__device__ void posv(T *A, T *B, T rho = T(0), int *s_fail = nullptr) {
    block::posv<T, N, NRHS, REGULARIZE, CHECK, REG_DIAG>(A, B, rho, s_fail);
}

// ─── eig3 ────────────────────────────────────────────────────────────────────
// Fixed 3x3 — the table cell is (op::eig3, N=3, dtype).

template <typename T, bool TRAILING_SYNC = true>
__device__ void eig3(const T *A, T *W, T *V) {
    constexpr body b = TRAILING_SYNC ? dispatch_body(op::eig3, 3u, sizeof(T) == 8)
                                     : body::block;
    if constexpr (b == body::warp_in_block) {
        if (dispatch_detail::full_first_warp()) {
            dispatch_detail::warp0([&] { warp::eig3<T>(A, W, V); });
        } else {
            block::eig3<T, TRAILING_SYNC>(A, W, V);
        }
    } else if constexpr (b == body::thread_in_block) {
        dispatch_detail::thread0([&] { thread::eig3<T>(A, W, V); });
    } else {
        block::eig3<T, TRAILING_SYNC>(A, W, V);
    }
}

// ─── softmax ─────────────────────────────────────────────────────────────────
// The one runtime-size dispatch: the table verdict is bounded by the largest
// measured n, so unmeasured large n stay on the block body. The warp/thread
// twins are scratch-free; `s_scratch` is simply unused on those bodies.

template <typename T, bool TRAILING_SYNC = true>
__device__ void softmax(uint32_t n, T alpha, const T *x, T *y, T *s_scratch) {
    if constexpr (TRAILING_SYNC) {
        const body b = dispatch_body(op::softmax, n, sizeof(T) == 8);
        if (b == body::warp_in_block && dispatch_detail::full_first_warp()) {
            dispatch_detail::warp0([&] { warp::softmax<T>(n, alpha, x, y); });
            return;
        }
        if (b == body::thread_in_block) {
            dispatch_detail::thread0([&] { thread::softmax<T>(n, alpha, x, y); });
            return;
        }
    }
    block::softmax<T, TRAILING_SYNC>(n, alpha, x, y, s_scratch);
}
