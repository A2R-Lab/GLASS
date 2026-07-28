// test_robotics.cu — driver for the robotics-op families (lie/quat, spatial 6D,
// projections/AL, geometry, softmax/argreduce).
//
// Three launch models over the SAME inputs, selected by <model>:
//
//   block   <<<P, tpb>>>          — block p owns problem p (glass:: surface).
//                                   tpb comes from the CLI so the pytest suite
//                                   sweeps {1, 32, 64, 256} and asserts
//                                   BIT-IDENTICAL output (thread-count
//                                   invariance; tpb=1 is also the block1
//                                   oracle for the cross-tier ULP checks).
//   warp    <<<ceil(P/4), 128>>>  — warp w owns problem w (glass::warp::).
//   thread  <<<ceil(P/64), 64>>>  — thread p owns problem p (glass::thread::).
//
// The three tiers share one serial core per op (redundant-core + strided
// copy-out), so cross-tier agreement is asserted tightly in pytest (ULP for
// pure-arithmetic maps, tight-allclose for sqrt/trig chains — the policy of
// test_thread.py). Scalar tier-free ops (angle/proj scalars, geometry) have no
// tier variants; the driver computes them on one writer lane per problem in
// every model.
//
// Usage: ./test_robotics <op> <model> <dtype> <P> <tpb> <flag0> <flag1> <files...>
//   dtype: f32 f64. Operand .bin files are float32 (the harness convention);
//   widened to T on load; printed at native precision (round-trip exact).
//   flag0/flag1 are op-specific (quat_mul: flag0=layout(0 xyzw/1 wxyz);
//   quat_normalize: flag0=CANONICAL; mcross_mul: flag0=AXIS+1, flag1=HAS_BETA;
//   fcross_mul: flag1=HAS_BETA; soc_*/softmax/logsumexp/arg*: flag0=n;
//   interval_scalars: flag0=soft).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include "helpers.cuh"
#include "../../glass.cuh"

// Fixed op constants mirrored by test_robotics.py:
#define ALPHA_MUL 1.5   // alpha for the fused cross applies
#define BETA_MUL  0.5   // beta when HAS_BETA
#define SOC_RHO   0.7   // rho for al_soc_value
#define AL_RHO    1.3   // rho for the interval AL scalars
#define AL_SIGMA  0.3   // sigma when flag0 = soft
#define RB_MU     0.8   // relaxed-barrier mu
#define RB_DELTA  0.15  // relaxed-barrier delta
#define SH_ETA    0.2   // smooth-hinge eta
#define SM_ALPHA  (-0.75)  // softmax/logsumexp alpha

enum Op {
    OP_QUAT_MUL, OP_QUAT_CONJ, OP_QUAT_NORMALIZE, OP_QUAT_EXP, OP_QUAT_ROTATE,
    OP_QUAT_TO_ROT, OP_ROT_TO_QUAT, OP_QUAT_TO_BASIS, OP_QUAT_RETRACT,
    OP_SKEW, OP_SO3_EXP, OP_SO3_LOG, OP_SO3_RJAC, OP_SO3_RJAC_INV,
    OP_SO3_LJAC, OP_SO3_LJAC_INV,
    OP_SE3_Q_BLOCK, OP_SE3_RETRACT, OP_SE3_JAC_Q, OP_SE3_JAC_V,
    OP_SE3_HESS_Q, OP_SE3_HESS_V,
    OP_MOTION_CROSS, OP_FORCE_CROSS, OP_FORCE_CROSS_DUAL,
    OP_MCROSS_MUL, OP_FCROSS_MUL,
    OP_SOC_PROJECT, OP_SOC_SCALARS, OP_INTERVAL_SCALARS, OP_RBAR,
    OP_SMOOTH_HINGE, OP_ANGLE,
    OP_SPHERE_SPHERE, OP_SPHERE_BOX, OP_TRANSFORM_SPHERE, OP_FRAME, OP_SEGMENT,
    OP_SOFTMAX, OP_LOGSUMEXP, OP_ARGMAX, OP_ARGMIN,
    OP_MOTION_XFORM, OP_FORCE_XFORM, OP_MXFORM_MUL, OP_FXFORM_MUL,
    OP_SPATIAL_INERTIA, OP_SINERTIA_MUL,
    OP_QUAT_LOG, OP_QUAT_ERROR, OP_POSE_ERROR, OP_QUAT_ANGLE, OP_LOG_COSH,
    OP_EIG3, OP_SVD3, OP_CLOSEST_ROT,
    OP_ARGMAX_FAST, OP_ARGMIN_FAST,
    OP_COUNT
};

struct OpInfo { const char* name; int in0, in1, in2, out; };
// -1 sizes mean "flag0 elements" (runtime-n ops).
static const OpInfo OPS[OP_COUNT] = {
    {"quat_mul", 4, 4, 0, 4},          {"quat_conj", 4, 0, 0, 4},
    {"quat_normalize", 4, 0, 0, 4},    {"quat_exp", 3, 0, 0, 4},
    {"quat_rotate", 4, 3, 0, 3},       {"quat_to_rot", 4, 0, 0, 9},
    {"rot_to_quat", 9, 0, 0, 4},       {"quat_to_basis", 4, 0, 0, 9},
    {"quat_retract", 4, 3, 0, 4},
    {"skew", 3, 0, 0, 9},              {"so3_exp", 3, 0, 0, 9},
    {"so3_log", 9, 0, 0, 3},           {"so3_rjac", 3, 0, 0, 9},
    {"so3_rjac_inv", 3, 0, 0, 9},      {"so3_ljac", 3, 0, 0, 9},
    {"so3_ljac_inv", 3, 0, 0, 9},
    {"se3_q_block", 3, 3, 0, 9},       {"se3_retract", 7, 3, 3, 7},
    {"se3_jac_q", 3, 3, 0, 36},        {"se3_jac_v", 3, 3, 0, 36},
    {"se3_hess_q", 3, 3, 0, 216},      {"se3_hess_v", 3, 3, 0, 216},
    {"motion_cross", 6, 0, 0, 36},     {"force_cross", 6, 0, 0, 36},
    {"force_cross_dual", 6, 0, 0, 36},
    {"mcross_mul", 6, 6, 6, 6},        {"fcross_mul", 6, 6, 6, 6},
    {"soc_project", -1, 0, 0, -1},     {"soc_scalars", -1, -1, 0, 3},
    {"interval_scalars", 5, 0, 0, 4},  {"rbar", 3, 0, 0, 3},
    {"smooth_hinge", 1, 0, 0, 2},      {"angle", 3, 0, 0, 4},
    {"sphere_sphere", 8, 0, 0, 4},     {"sphere_box", 7, 0, 0, 4},
    {"transform_sphere", 4, 3, 4, 4},  {"frame", 3, 0, 0, 6},
    {"segment", 12, 0, 0, 9},
    {"softmax", -1, 0, 0, -1},         {"logsumexp", -1, 0, 0, 1},
    {"argmax", -1, 0, 0, 2},           {"argmin", -1, 0, 0, 2},
    // wave 2: transforms pack (E|r) as one 12-elem input; inertia is pi[10].
    {"motion_xform", 12, 0, 0, 36},    {"force_xform", 12, 0, 0, 36},
    {"mxform_mul", 12, 6, 6, 6},       {"fxform_mul", 12, 6, 6, 6},
    {"spatial_inertia", 10, 0, 0, 36}, {"sinertia_mul", 10, 6, 6, 6},
    {"quat_log", 4, 0, 0, 3},          {"quat_error", 4, 4, 0, 3},
    {"pose_error", 7, 7, 0, 6},        {"quat_angle", 4, 4, 0, 1},
    {"log_cosh", 1, 0, 0, 2},
    {"eig3", 9, 0, 0, 12},             {"svd3", 9, 0, 0, 21},
    {"closest_rot", 9, 0, 0, 9},
    {"argmax_fast", -1, 0, 0, 2},      {"argmin_fast", -1, 0, 0, 2},
};

// ─── dtype-generic I/O ───────────────────────────────────────────────────────
template <typename T>
static T* read_dev(const char* path, long n) {
    float* h = read_host_vec(path, (int)n);
    T* hT = (T*)malloc(n * sizeof(T));
    for (long i = 0; i < n; i++) hT[i] = (T)h[i];
    free(h);
    T* d; cudaMalloc(&d, n * sizeof(T));
    cudaMemcpy(d, hT, n * sizeof(T), cudaMemcpyHostToDevice);
    free(hT);
    return d;
}
template <typename T>
static T* alloc_dev(long n) { T* d; cudaMalloc(&d, n * sizeof(T)); cudaMemset(d, 0, n * sizeof(T)); return d; }

template <typename T> __global__ void print_kernelT(const T* d, long n) {
    for (long i = 0; i < n; i++) {
        if constexpr (sizeof(T) == 8) printf("%.17g", (double)d[i]);
        else                          printf("%.9g",  (double)d[i]);
        if (i < n - 1) printf(" ");
    }
    printf("\n");
}
template <typename T> static void print_dev(const T* d, long n) {
    print_kernelT<T><<<1,1>>>(d, n); cudaDeviceSynchronize();
}

// ─── tier dispatch ───────────────────────────────────────────────────────────
// M: 0 = block (glass::), 1 = warp (glass::warp::), 2 = thread (glass::thread::).
#define TIER3(FN, ...)                                          \
    do {                                                        \
        if constexpr (M == 0)      glass::FN<T>(__VA_ARGS__);   \
        else if constexpr (M == 1) glass::warp::FN<T>(__VA_ARGS__); \
        else                       glass::thread::FN<T>(__VA_ARGS__); \
    } while (0)

template <typename T, int M>
__device__ void dev_tier_op(int op, int flag0, int flag1,
                            const T* a, const T* b, const T* c, T* out,
                            T* smem, bool writer, uint32_t nrt) {
    switch (op) {
        case OP_QUAT_MUL:
            if (flag0 == 0) { TIER3(quat_mul, a, b, out); }
            else {
                if constexpr (M == 0)      glass::quat_mul<T, glass::QuatLayout::wxyz>(a, b, out);
                else if constexpr (M == 1) glass::warp::quat_mul<T, glass::QuatLayout::wxyz>(a, b, out);
                else                       glass::thread::quat_mul<T, glass::QuatLayout::wxyz>(a, b, out);
            }
            break;
        case OP_QUAT_CONJ:      TIER3(quat_conj, a, out); break;
        case OP_QUAT_NORMALIZE:
            if (flag0 == 0) { TIER3(quat_normalize, a, out); }
            else {
                if constexpr (M == 0)      glass::quat_normalize<T, glass::QuatLayout::xyzw, true>(a, out);
                else if constexpr (M == 1) glass::warp::quat_normalize<T, glass::QuatLayout::xyzw, true>(a, out);
                else                       glass::thread::quat_normalize<T, glass::QuatLayout::xyzw, true>(a, out);
            }
            break;
        case OP_QUAT_EXP:       TIER3(quat_exp, a, out); break;
        case OP_QUAT_ROTATE:    TIER3(quat_rotate, a, b, out); break;
        case OP_QUAT_TO_ROT:    TIER3(quat_to_rot, a, out); break;
        case OP_ROT_TO_QUAT:    TIER3(rot_to_quat, a, out); break;
        case OP_QUAT_TO_BASIS:  TIER3(quat_to_basis, a, out, out + 3, out + 6); break;
        case OP_QUAT_RETRACT:   TIER3(quat_retract, a, b, out); break;
        case OP_SKEW:           TIER3(skew, a, out); break;
        case OP_SO3_EXP:        TIER3(so3_exp, a, out); break;
        case OP_SO3_LOG:        TIER3(so3_log, a, out); break;
        case OP_SO3_RJAC:       TIER3(so3_right_jacobian, a, out); break;
        case OP_SO3_RJAC_INV:   TIER3(so3_right_jacobian_inv, a, out); break;
        case OP_SO3_LJAC:       TIER3(so3_left_jacobian, a, out); break;
        case OP_SO3_LJAC_INV:   TIER3(so3_left_jacobian_inv, a, out); break;
        case OP_SE3_Q_BLOCK:    TIER3(se3_Q_block, a, b, out); break;
        case OP_SE3_RETRACT:    TIER3(se3_retract, a, b, c, out); break;
        case OP_SE3_JAC_Q:      TIER3(se3_retract_jacobian_q, a, b, out); break;
        case OP_SE3_JAC_V:      TIER3(se3_retract_jacobian_v, a, b, out); break;
        case OP_SE3_HESS_Q:
            if constexpr (M == 0)      glass::se3_retract_hessian<T, true>(a, b, out);
            else if constexpr (M == 1) glass::warp::se3_retract_hessian<T, true>(a, b, out);
            else                       glass::thread::se3_retract_hessian<T, true>(a, b, out);
            break;
        case OP_SE3_HESS_V:
            if constexpr (M == 0)      glass::se3_retract_hessian<T, false>(a, b, out);
            else if constexpr (M == 1) glass::warp::se3_retract_hessian<T, false>(a, b, out);
            else                       glass::thread::se3_retract_hessian<T, false>(a, b, out);
            break;
        case OP_MOTION_CROSS:     TIER3(motion_cross, a, out); break;
        case OP_FORCE_CROSS:      TIER3(force_cross, a, out); break;
        case OP_FORCE_CROSS_DUAL: TIER3(force_cross_dual, a, out); break;
        case OP_MCROSS_MUL: {
            const T al = (T)ALPHA_MUL, be = (T)BETA_MUL;
            // flag0 = AXIS+1 (0 = dense), flag1 = HAS_BETA. HAS_BETA seeds out from c.
            if (flag1 && writer) for (int i = 0; i < 6; i++) out[i] = c[i];
            if constexpr (M == 0) __syncthreads(); else if constexpr (M == 1) __syncwarp();
            #define MCM_CASE(AX)                                                              \
                if (flag1) {                                                                  \
                    if constexpr (M == 0)      glass::motion_cross_mul<T, AX, true>(al, a, b, be, out);        \
                    else if constexpr (M == 1) glass::warp::motion_cross_mul<T, AX, true>(al, a, b, be, out);  \
                    else                       glass::thread::motion_cross_mul<T, AX, true>(al, a, b, be, out);\
                } else {                                                                      \
                    if constexpr (M == 0)      glass::motion_cross_mul<T, AX, false>(al, a, b, be, out);       \
                    else if constexpr (M == 1) glass::warp::motion_cross_mul<T, AX, false>(al, a, b, be, out); \
                    else                       glass::thread::motion_cross_mul<T, AX, false>(al, a, b, be, out);\
                }
            switch (flag0) {
                case 0: MCM_CASE(-1); break;
                case 1: MCM_CASE(0); break;  case 2: MCM_CASE(1); break;
                case 3: MCM_CASE(2); break;  case 4: MCM_CASE(3); break;
                case 5: MCM_CASE(4); break;  default: MCM_CASE(5); break;
            }
            #undef MCM_CASE
            break;
        }
        case OP_FCROSS_MUL: {
            const T al = (T)ALPHA_MUL, be = (T)BETA_MUL;
            if (flag1 && writer) for (int i = 0; i < 6; i++) out[i] = c[i];
            if constexpr (M == 0) __syncthreads(); else if constexpr (M == 1) __syncwarp();
            if (flag1) {
                if constexpr (M == 0)      glass::force_cross_mul<T, true>(al, a, b, be, out);
                else if constexpr (M == 1) glass::warp::force_cross_mul<T, true>(al, a, b, be, out);
                else                       glass::thread::force_cross_mul<T, true>(al, a, b, be, out);
            } else {
                if constexpr (M == 0)      glass::force_cross_mul<T, false>(al, a, b, be, out);
                else if constexpr (M == 1) glass::warp::force_cross_mul<T, false>(al, a, b, be, out);
                else                       glass::thread::force_cross_mul<T, false>(al, a, b, be, out);
            }
            break;
        }
        case OP_SOC_PROJECT:    TIER3(soc_project, a, out, (int32_t)nrt); break;
        case OP_SOFTMAX:
            if constexpr (M == 0)      glass::softmax<T>(nrt, (T)SM_ALPHA, a, out, smem);
            else if constexpr (M == 1) glass::warp::softmax<T>(nrt, (T)SM_ALPHA, a, out);
            else                       glass::thread::softmax<T>(nrt, (T)SM_ALPHA, a, out);
            break;
        case OP_LOGSUMEXP:
            if constexpr (M == 0)      glass::logsumexp<T>(nrt, (T)SM_ALPHA, a, out, smem);
            else if constexpr (M == 1) { T r = glass::warp::logsumexp<T>(nrt, (T)SM_ALPHA, a); if (writer) out[0] = r; }
            else                       { out[0] = glass::thread::logsumexp<T>(nrt, (T)SM_ALPHA, a); }
            break;
        case OP_ARGMAX:
        case OP_ARGMIN: {
            const bool mn = (op == OP_ARGMIN);
            if constexpr (M == 0) {
                __shared__ uint32_t s_idx[1];
                __shared__ T s_val[1];
                if (mn) glass::argmin<T>(nrt, a, s_idx, s_val, smem);
                else    glass::argmax<T>(nrt, a, s_idx, s_val, smem);
                if (writer) { out[0] = (T)s_idx[0]; out[1] = s_val[0]; }
            } else if constexpr (M == 1) {
                uint32_t i = mn ? glass::warp::argmin<T>(nrt, a) : glass::warp::argmax<T>(nrt, a);
                if (writer) { out[0] = (T)i; out[1] = a[i]; }
            } else {
                uint32_t i = mn ? glass::thread::argmin<T>(nrt, a) : glass::thread::argmax<T>(nrt, a);
                out[0] = (T)i; out[1] = a[i];
            }
            break;
        }
        case OP_MOTION_XFORM: TIER3(motion_transform, a, a + 9, out); break;
        case OP_FORCE_XFORM:  TIER3(force_transform, a, a + 9, out); break;
        case OP_MXFORM_MUL:
        case OP_FXFORM_MUL: {
            const T al = (T)ALPHA_MUL, be = (T)BETA_MUL;
            // flag0 = INVERSE, flag1 = HAS_BETA (seeds out from c).
            if (flag1 && writer) for (int i = 0; i < 6; i++) out[i] = c[i];
            if constexpr (M == 0) __syncthreads(); else if constexpr (M == 1) __syncwarp();
            #define XFM_CASE(FN, INV, HB)                                                     \
                if constexpr (M == 0)      glass::FN<T, INV, HB>(al, a, a + 9, b, be, out);        \
                else if constexpr (M == 1) glass::warp::FN<T, INV, HB>(al, a, a + 9, b, be, out);  \
                else                       glass::thread::FN<T, INV, HB>(al, a, a + 9, b, be, out);
            #define XFM_DISPATCH(FN)                                                          \
                if (!flag0 && !flag1)      { XFM_CASE(FN, false, false) }                      \
                else if (!flag0 && flag1)  { XFM_CASE(FN, false, true) }                       \
                else if (flag0 && !flag1)  { XFM_CASE(FN, true, false) }                       \
                else                       { XFM_CASE(FN, true, true) }
            if (op == OP_MXFORM_MUL) { XFM_DISPATCH(motion_transform_mul) }
            else                     { XFM_DISPATCH(force_transform_mul) }
            #undef XFM_DISPATCH
            #undef XFM_CASE
            break;
        }
        case OP_SPATIAL_INERTIA: TIER3(spatial_inertia, a, out); break;
        case OP_SINERTIA_MUL: {
            const T al = (T)ALPHA_MUL, be = (T)BETA_MUL;
            if (flag1 && writer) for (int i = 0; i < 6; i++) out[i] = c[i];
            if constexpr (M == 0) __syncthreads(); else if constexpr (M == 1) __syncwarp();
            if (flag1) {
                if constexpr (M == 0)      glass::spatial_inertia_mul<T, true>(al, a, b, be, out);
                else if constexpr (M == 1) glass::warp::spatial_inertia_mul<T, true>(al, a, b, be, out);
                else                       glass::thread::spatial_inertia_mul<T, true>(al, a, b, be, out);
            } else {
                if constexpr (M == 0)      glass::spatial_inertia_mul<T, false>(al, a, b, be, out);
                else if constexpr (M == 1) glass::warp::spatial_inertia_mul<T, false>(al, a, b, be, out);
                else                       glass::thread::spatial_inertia_mul<T, false>(al, a, b, be, out);
            }
            break;
        }
        case OP_QUAT_LOG:    TIER3(quat_log, a, out); break;
        case OP_QUAT_ERROR:  TIER3(quat_error, a, b, out); break;
        case OP_POSE_ERROR:  TIER3(pose_error, a, b, out); break;
        case OP_EIG3:        TIER3(eig3, a, out, out + 3); break;
        case OP_SVD3:        TIER3(svd3, a, out, out + 9, out + 12); break;
        case OP_CLOSEST_ROT: TIER3(closest_rotation, a, out); break;
        case OP_ARGMAX_FAST:
        case OP_ARGMIN_FAST: {
            // Block model exercises the _fast variant; warp/thread models fall
            // back to their (only) argreduce forms — the cross-model equality
            // check then pins _fast against the same answer.
            const bool mn = (op == OP_ARGMIN_FAST);
            if constexpr (M == 0) {
                __shared__ uint32_t s_idx[1];
                __shared__ T s_val[1];
                if (mn) glass::argmin_fast<T>(nrt, a, s_idx, s_val, smem);
                else    glass::argmax_fast<T>(nrt, a, s_idx, s_val, smem);
                if (writer) { out[0] = (T)s_idx[0]; out[1] = s_val[0]; }
            } else if constexpr (M == 1) {
                uint32_t i = mn ? glass::warp::argmin<T>(nrt, a) : glass::warp::argmax<T>(nrt, a);
                if (writer) { out[0] = (T)i; out[1] = a[i]; }
            } else {
                uint32_t i = mn ? glass::thread::argmin<T>(nrt, a) : glass::thread::argmax<T>(nrt, a);
                out[0] = (T)i; out[1] = a[i];
            }
            break;
        }
        default: break;   // scalar tier-free ops handled by dev_scalar_op
    }
}

// Scalar tier-free ops: one writer lane per problem computes the whole answer.
template <typename T>
__device__ void dev_scalar_op(int op, int flag0, const T* a, const T* b, const T* c,
                              T* out, uint32_t m) {
    switch (op) {
        case OP_SOC_SCALARS:
            out[0] = glass::soc_tail_norm<T>(a, (int32_t)m);
            out[1] = glass::soc_violation<T>(a, (int32_t)m);
            out[2] = glass::al_soc_value<T>(a, b, (T)SOC_RHO, (int32_t)m);
            break;
        case OP_INTERVAL_SCALARS: {
            const T sg = flag0 ? (T)AL_SIGMA : (T)0;
            out[0] = glass::interval_violation<T>(a[0], a[1], a[2]);
            out[1] = glass::al_interval_value<T>(a[0], a[1], a[2], a[3], a[4], (T)AL_RHO, sg);
            T gr, h;
            glass::al_interval_grad_hess<T>(a[0], a[1], a[2], a[3], a[4], (T)AL_RHO, sg, gr, h);
            out[2] = gr; out[3] = h;
            break;
        }
        case OP_RBAR:
            out[0] = glass::relaxed_barrier_interval_value<T>(a[0], a[1], a[2], (T)RB_MU, (T)RB_DELTA);
            out[1] = glass::relaxed_barrier_interval_grad<T>(a[0], a[1], a[2], (T)RB_MU, (T)RB_DELTA);
            out[2] = glass::relaxed_barrier_interval_hess<T>(a[0], a[1], a[2], (T)RB_MU, (T)RB_DELTA);
            break;
        case OP_SMOOTH_HINGE:
            out[0] = glass::smooth_hinge<T>(a[0], (T)SH_ETA);
            out[1] = glass::smooth_hinge_grad<T>(a[0], (T)SH_ETA);
            break;
        case OP_ANGLE:
            out[0] = glass::angle_wrap<T>(a[0]);
            out[1] = glass::angle_diff<T>(a[0], a[1]);
            out[2] = glass::angle_lerp<T>(a[0], a[1], a[2]);
            out[3] = glass::clamp_unit<T>(a[0]);
            break;
        case OP_SPHERE_SPHERE:
            out[0] = glass::sphere_sphere_dist<T>(a, a[6], a + 3, a[7], out + 1);
            break;
        case OP_SPHERE_BOX:
            out[0] = glass::sphere_box_dist<T>(a, a[6], a + 3, out + 1);
            break;
        case OP_TRANSFORM_SPHERE:
            glass::transform_sphere<T>(a, b, c, out);
            break;
        case OP_FRAME:
            glass::frame_from_vector<T>(a, out, out + 3);
            break;
        case OP_SEGMENT: {
            T s, t;
            out[0] = glass::segment_segment_closest<T>(a, a + 3, a + 6, a + 9, s, t, out + 3, out + 6);
            out[1] = s; out[2] = t;
            break;
        }
        case OP_QUAT_ANGLE:
            out[0] = glass::quat_angle<T>(a, b);
            break;
        case OP_LOG_COSH:
            out[0] = glass::log_cosh<T>(a[0]);
            out[1] = glass::log_cosh_grad<T>(a[0]);
            break;
        default: break;
    }
}

static bool is_scalar_op(int op) {
    switch (op) {
        case OP_SOC_SCALARS: case OP_INTERVAL_SCALARS: case OP_RBAR:
        case OP_SMOOTH_HINGE: case OP_ANGLE: case OP_SPHERE_SPHERE:
        case OP_SPHERE_BOX: case OP_TRANSFORM_SPHERE: case OP_FRAME:
        case OP_SEGMENT: case OP_QUAT_ANGLE: case OP_LOG_COSH:
            return true;
        default: return false;
    }
}

// ─── model kernels ───────────────────────────────────────────────────────────
template <typename T>
__global__ void k_block(int op, int flag0, int flag1, int P,
                        long s0, long s1, long s2, long so,
                        const T* i0, const T* i1, const T* i2, T* out,
                        bool scalar, uint32_t nrt) {
    extern __shared__ char raw[];
    T* smem = reinterpret_cast<T*>(raw);
    int p = blockIdx.x;
    if (p >= P) return;
    const T* a = i0 + (size_t)p*s0;
    const T* b = (s1 > 0) ? i1 + (size_t)p*s1 : nullptr;
    const T* c = (s2 > 0) ? i2 + (size_t)p*s2 : nullptr;
    T* o = out + (size_t)p*so;
    const bool writer = (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0);
    if (scalar) { if (writer) dev_scalar_op<T>(op, flag0, a, b, c, o, nrt); }
    else        dev_tier_op<T, 0>(op, flag0, flag1, a, b, c, o, smem, writer, nrt);
}

template <typename T>
__global__ void k_warp(int op, int flag0, int flag1, int P,
                       long s0, long s1, long s2, long so,
                       const T* i0, const T* i1, const T* i2, T* out,
                       bool scalar, uint32_t nrt) {
    int gw = (int)((blockIdx.x*blockDim.x + threadIdx.x) >> 5);
    int lane = (int)(threadIdx.x & 31);
    if (gw >= P) return;
    const T* a = i0 + (size_t)gw*s0;
    const T* b = (s1 > 0) ? i1 + (size_t)gw*s1 : nullptr;
    const T* c = (s2 > 0) ? i2 + (size_t)gw*s2 : nullptr;
    T* o = out + (size_t)gw*so;
    if (scalar) { if (lane == 0) dev_scalar_op<T>(op, flag0, a, b, c, o, nrt); }
    else        dev_tier_op<T, 1>(op, flag0, flag1, a, b, c, o, nullptr, lane == 0, nrt);
}

template <typename T>
__global__ void k_thread(int op, int flag0, int flag1, int P,
                         long s0, long s1, long s2, long so,
                         const T* i0, const T* i1, const T* i2, T* out,
                         bool scalar, uint32_t nrt) {
    int p = (int)(blockIdx.x*blockDim.x + threadIdx.x);
    if (p >= P) return;
    const T* a = i0 + (size_t)p*s0;
    const T* b = (s1 > 0) ? i1 + (size_t)p*s1 : nullptr;
    const T* c = (s2 > 0) ? i2 + (size_t)p*s2 : nullptr;
    T* o = out + (size_t)p*so;
    if (scalar) dev_scalar_op<T>(op, flag0, a, b, c, o, nrt);
    else        dev_tier_op<T, 2>(op, flag0, flag1, a, b, c, o, nullptr, true, nrt);
}

// ─── host driver ─────────────────────────────────────────────────────────────
template <typename T>
static int run(int op, const char* model, int P, int tpb, int flag0, int flag1,
               int nfiles, char** files) {
    const OpInfo& info = OPS[op];
    const uint32_t nrt = (info.in0 == -1) ? (uint32_t)flag0 : 0u;
    const long s0 = (info.in0 == -1) ? flag0 : info.in0;
    const long s1 = (info.in1 == -1) ? flag0 : info.in1;
    const long s2 = info.in2;
    const long so = (info.out == -1) ? flag0 : info.out;
    int need = (s0 > 0) + (s1 > 0) + (s2 > 0);
    if (nfiles != need) { fprintf(stderr, "expected %d files, got %d\n", need, nfiles); return 1; }
    int fi = 0;
    T* i0 = (s0 > 0) ? read_dev<T>(files[fi++], (long)P*s0) : nullptr;
    T* i1 = (s1 > 0) ? read_dev<T>(files[fi++], (long)P*s1) : nullptr;
    T* i2 = (s2 > 0) ? read_dev<T>(files[fi++], (long)P*s2) : nullptr;
    T* out = alloc_dev<T>((long)P*so);
    const bool scalar = is_scalar_op(op);
    const size_t smem = 8192;   // covers softmax n-scratch and argreduce (key+idx per thread)
    if (strcmp(model, "block") == 0) {
        k_block<T><<<P, tpb, smem>>>(op, flag0, flag1, P, s0, s1, s2, so, i0, i1, i2, out, scalar, nrt);
    } else if (strcmp(model, "warp") == 0) {
        int blocks = (P + 3) / 4;
        k_warp<T><<<blocks, 128>>>(op, flag0, flag1, P, s0, s1, s2, so, i0, i1, i2, out, scalar, nrt);
    } else if (strcmp(model, "thread") == 0) {
        int blocks = (P + 63) / 64;
        k_thread<T><<<blocks, 64>>>(op, flag0, flag1, P, s0, s1, s2, so, i0, i1, i2, out, scalar, nrt);
    } else {
        fprintf(stderr, "unknown model %s\n", model); return 1;
    }
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { fprintf(stderr, "kernel failed: %s\n", cudaGetErrorString(err)); return 1; }
    print_dev<T>(out, (long)P*so);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 8) {
        fprintf(stderr, "usage: %s <op> <model> <dtype> <P> <tpb> <flag0> <flag1> <files...>\n", argv[0]);
        return 1;
    }
    int op = -1;
    for (int i = 0; i < OP_COUNT; i++) if (strcmp(argv[1], OPS[i].name) == 0) { op = i; break; }
    if (op < 0) { fprintf(stderr, "unknown op %s\n", argv[1]); return 1; }
    const char* model = argv[2];
    const char* dtype = argv[3];
    int P = atoi(argv[4]);
    int tpb = atoi(argv[5]);
    int flag0 = atoi(argv[6]);
    int flag1 = atoi(argv[7]);
    char** files = argv + 8;
    int nfiles = argc - 8;
    if (strcmp(dtype, "f64") == 0) return run<double>(op, model, P, tpb, flag0, flag1, nfiles, files);
    return run<float>(op, model, P, tpb, flag0, flag1, nfiles, files);
}
