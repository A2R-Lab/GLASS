// timing_common.cuh — shared measurement core for the bench harnesses.
//
// Methodology (documented in bench/TUNING.md, audited 2026-08-11):
//   * one untimed launch + sync doubles as warmup and LAUNCH-FAILURE probe
//     (a failed launch returns the 1e30 FAIL sentinel, never a fake fast time);
//   * 3 trials, each = wall-clock (CLOCK_MONOTONIC) around `reps` back-to-back
//     async launches + one sync, normalized to ns/problem;
//   * the reported value is the MIN of the 3 trials (noise on a quiet GPU is
//     one-sided additive), and the SPREAD (worst/best − 1) of the same 3
//     trials is exposed via tc_last_spread_pct() so every capture can prove it
//     was clean — tune.py warns when a row's spread exceeds the decision
//     margin it is about to resolve;
//   * MUTATION INVARIANT: reps run with NO restore between them, so in-place
//     ops (potrf/ldlt/…) re-process their own output from rep 2 on. That is
//     a steady-throughput characterization ONLY for branch-free ops whose
//     control flow is data-independent. It is not sufficient evidence for an
//     in-place solver default: POTRF/TRSV/POSV defaults come instead from the
//     symmetric fresh-input sweep in bench_solver_ladder.cu. Do NOT time a
//     CHECK-gated or pivoted op through this loop — its data-dependent branches
//     would time garbage.
#pragma once
#include <cstdio>
#include <ctime>
#include <cuda_runtime.h>

static inline double tc_elapsed_ms(struct timespec a, struct timespec b) {
    return (double)(b.tv_sec - a.tv_sec) * 1e3 + (double)(b.tv_nsec - a.tv_nsec) * 1e-6;
}

// Trial spread of the most recent tc_time_ns_per_prob call, in percent
// ((worst − best)/best over the 3 trials; 0 on FAIL).
static double tc_g_spread_pct = 0.0;
static inline double tc_last_spread_pct() { return tc_g_spread_pct; }

// pre_trial runs UNTIMED before the probe and before each trial (followed by a
// device sync), so a harness can restore pristine inputs per trial. In-place
// ops (chol/trsv/posv) drift their inputs across launches; without a per-TRIAL
// reset the first trial times the pristine->drifted transient while later
// trials time the drifted steady state — inflating spread and, worse, timing
// different contenders against different input trajectories (measured ~13%
// apart on identical code by position alone, audit 2026-08-15). With the hook
// every trial times the same trajectory for every contender.
template <typename F, typename P>
static double tc_time_ns_per_prob_pre(F launch, P pre_trial, int reps, int nprob) {
    cudaGetLastError();                          // clear any sticky prior error
    pre_trial();
    launch(); cudaDeviceSynchronize();
    // An infeasible config fails to LAUNCH; without this probe the empty launch
    // times as ~350ns total and poisons the argmin (caught 2026-07-18).
    if (cudaGetLastError() != cudaSuccess) { tc_g_spread_pct = 0.0; return 1e30; }
    double best = 1e30, worst = 0.0;
    for (int t = 0; t < 3; t++) {
        pre_trial();
        cudaDeviceSynchronize();                 // keep the reset out of the timed region
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int r = 0; r < reps; r++) launch();
        cudaDeviceSynchronize();
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ns = tc_elapsed_ms(t0, t1) * 1e6 / ((double)reps * nprob);
        if (ns < best) best = ns;
        if (ns > worst) worst = ns;
    }
    tc_g_spread_pct = (best > 0.0 && best < 1e29) ? (worst / best - 1.0) * 100.0 : 0.0;
    return best;
}

template <typename F>
static double tc_time_ns_per_prob(F launch, int reps, int nprob) {
    return tc_time_ns_per_prob_pre(launch, []{}, reps, nprob);
}

// ~0.7 s FMA busy-loop across all SMs: brings an idle GPU to steady boost
// clocks BEFORE the first timed cell. One untimed launch is not enough from
// idle, and without this the first rows of a sweep run measurably cooler than
// the last (min-of-3 only partially self-corrects). Call once from main().
static __global__ void tc_warm_kernel(float* sink, int inner) {
    float a = 1.0f + threadIdx.x * 1e-6f, b = 1.000001f;
    for (int i = 0; i < inner; i++) a = fmaf(a, b, 1e-9f);
    if (a == 0.0f) sink[threadIdx.x & 255] = a;   // never true; defeats DCE
}

static void tc_warm_gpu(int target_ms = 700) {
    int dev = 0, sms = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    float* sink = nullptr;
    cudaMalloc(&sink, 256 * sizeof(float));
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    do {
        tc_warm_kernel<<<sms * 4, 256>>>(sink, 200000);
        cudaDeviceSynchronize();
        clock_gettime(CLOCK_MONOTONIC, &t1);
    } while (tc_elapsed_ms(t0, t1) < (double)target_ms);
    cudaFree(sink);
    printf("# gpu warmed: %.0f ms busy-loop before first timed cell\n",
           tc_elapsed_ms(t0, t1));
}
