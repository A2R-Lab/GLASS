GLASS restricted-pointer audit — 2026-10-28
=========================================
Scope: investigate whether issue #44's caller-restrict-sensitive cross-thread
handoff pattern appears outside trsv. No library implementation was changed.
Baseline: 9e57178ca146fa1cecae7e27c22260603b993aa6.
Environment: RTX 5090 (sm_120), nvcc 13.2 V13.2.86, driver 615.71.09,
-O3 -std=c++17. These results do not establish behavior on other toolchains.

Method
------
A force-inlined device wrapper takes distinct __restrict__ pointers to shared
matrix, RHS, scratch, and pivot arrays. Arrays do not overlap. Remove only the
qualifier with -DRESTRICT= for controls. The kernel initializes shared inputs,
calls the library, and synchronizes before copying outputs for host checking.
Scratch is initialized to -777 to expose accidental stale scratch reads; it is
workspace, not an input with a required initial value.

The main matrix covers float/double, dimensions 3/7/17, three deterministic
inputs, and block sizes 1/2/3/7/16/31/32/33/64/96/128. Warp routines run only
at 32 threads. Prefix scan runs only when its documented n <= blockDim.x
precondition holds. Reference factors and solutions are constructed on the CPU;
inversion is checked through A*Ainv; reductions and normalization use separate
CPU sums. Threshold on abs(actual-expected)/(1+abs(expected)): 2e-5 float,
1e-11 double. Nonfinite errors fail. These are exploratory harnesses, not the
repository's permanent regression suite. Numerical FAIL rows do NOT change
program exit status; inspect CSV results. CUDA errors return nonzero.

Results
-------
The completed restricted sweep has 4,926 cases across 27 call variants:
910 fail. The 5,322-case unrestricted extended sweep passes entirely.
An additional 72-case unrestricted warp-softmax/eigensolver sweep also passes.
All 5,394 unrestricted control cases pass. See summary.csv for per-op counts.
Two fatal restricted variants are deliberately omitted from extended-safe.cu;
they must run separately with timeouts. Partial runs must not be counted as
complete coverage.

Confirmed numerical failures (restricted / unrestricted failure counts):
- trsv compile-time and constant-folded runtime calls: 120/198 each versus 0.
- single-RHS potrs and posv: 120/198 each versus 0.
- block softmax: 180/198 versus 0.
- block logsumexp: 100/198 versus 0.
- bare glass::softmax (measured dispatch): 150/198 versus 0.
The sampled trsv failures occur at n=3/7; n=17 passes. This shape dependence
is a reason not to interpret a passing shape as a general safety guarantee.

Minimal new softmax example: x=[0,0,0], double, n=3. One thread returns
[1/3,1/3,1/3]. Two threads with restrict return [1/3,1,1/3] (sum=5/3).
The same two-thread call without restrict returns the correct result.

Confirmed restricted-path execution failures:
- inv_pivoted<float,3>: one thread passes; the next two-thread launch does not
  finish within 10 seconds. Isolated repro also times out under synccheck at
  15 seconds. Synccheck produced no diagnostic before timeout: a divergent
  barrier is a hypothesis, not a sanitizer-confirmed root cause.
- pivoted ldlt<float,3> followed by ldlt_solve: one thread passes; two threads
  produce an illegal memory access. Compute Sanitizer reports an out-of-bounds
  shared read in ldlt.cuh:169, thread 1, after s_idx[0] supplies imax. The error
  occurs in factorization, before the solve. See ldlt-memcheck.log.
- syev<float,3>: one thread passes; the next two-thread launch times out at
  15 seconds. The unrestricted extra-control run passes all 66 syev cases.
  The attempted double/n=7 restricted syev cases were never reached.
These establish failures correlated with the qualifier. They do not by
themselves prove the exact optimizer transformation or a universal repair.

No failures in the completed restricted samples of: trsm (one/two RHS),
potrf (runtime/compile-time), multi-RHS potrs/posv (including NRHS=1),
non-pivoted inv, gesv, unpivoted ldlt+solve, reduce, the internal nrm2_impl
engine, vector_norm, inclusive prefix scan, full-block cgrps potrf/trsm/reduce,
warp trsv/potrf. Six extra warp-softmax cases pass. These are bounded negative
results, not certification of all inputs or caller contexts.

Static follow-up map
--------------------
- trsv.cuh:54-67: pivot write/read and changing elimination ownership.
- softmax.cuh:97-112 and 138 onward: shared max/sum scratch consumption.
- inv.cuh:175-200: shared pivot choice feeds a branch containing a barrier.
- ldlt.cuh:167-169: iamax writes index scratch, immediately consumed by peers.
- syev.cuh:148 onward: shared rotation and convergence control values.
- bdsv.cuh composes potrf/trsm/trsv; direct end-to-end restricted testing remains.
- pcg/solve.cuh has shared alpha/beta/rho handoffs; targeted testing remains.
- internal/box_qp.cuh composes reductions and shared control; testing remains.
- Fused/batched factorization, checked paths, tiled cgrps, global-memory inputs,
  vendor wrappers, true dynamic sizes, all triangle/transpose/unit-diagonal
  variants, trailing-sync=false, larger kernels/register pressure, other GPUs
  and compilers remain outside this dynamic sweep.
- agent_debugging_guide.md section 1g says full-block twins are safe today;
  that statement is contradicted by these results and needs correction.
barrier-inventory.txt is a lexical inventory (including comments), not a claim
that every listed site is defective or has received full semantic review.

Improved correctness and performance plan
-----------------------------------------
Purpose: eliminate numerical, memory-access and synchronization bugs while
preserving GLASS's execution contracts and measured performance. The results
above are exploratory evidence; the steps below are planned work, not checks
already completed. pytest-gpu-proof's next release version is 0.5.0.

Weaknesses in the original audit and plan
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Case counts overstate breadth: many cases share one implementation and three
  closely related constructed inputs. The passing controls help attribute the
  behavior, but are not an independent implementation oracle.
- Restrict sensitivity is an observation, not a complete diagnosis. Removing a
  qualifier changes optimization. We have not established the exact load/store
  transformation, ruled out all caller-contract mistakes, or justified one
  repair for all failures. Review aliasing, bounds, initialization, alignment,
  participation and publication rules before attributing a compiler defect.
- A timeout establishes noncompletion within a bound, not a proven deadlock or
  divergent barrier. The pivoted-inverse synccheck run did not diagnose a cause.
- The exploratory runner returns success for numerical FAIL rows. Fatal cases
  abort later tests; their missing rows must not be counted as passes.
- The runtime overload tests use compile-time-known dimensions and therefore
  do not exercise genuinely dynamic loop bounds. The extra trailing barrier
  in the harness can hide publication failures at function return.
- Eigensolver residual checks alone are insufficient: zero or non-orthonormal
  eigenvectors can satisfy a residual test. Sorting, orthogonality and spectral
  agreement need independent checks. Constructed SPD matrices also miss much
  of pivot selection, indefinite factorization and conditioning behavior.
- One GPU/compiler, shared-memory inputs, one inlining shape and limited sizes
  leave substantial gaps. Sampled passing operations must not be called safe.
- The original plan had no measured performance baseline or explicit completion
  criteria, and risked coupling an urgent GLASS fix to a tooling release.

Tonight's execution target
~~~~~~~~~~~~~~~~~~~~~~~~~~
User direction: perform broad fixes tonight, then push the completed work.
This is authorization to implement and push, not just to continue planning.
Work through the reproduced solver, normalization, pivot-index and eigensolver
failures together with related shared-helper fixes and direct-consumer checks.
Use the phases below as execution gates within that effort. The time target
must not turn failed or incomplete checks into passes.

Separate reproduced release blockers from speculative audit candidates so the
scope stays assessable. This distinction does not narrow tonight's broad-fix
objective: investigate related mechanisms and fix supported findings in the
same effort. Keep a visible list of remaining candidates and explain any
unresolved reproduced failure before declaring the work ready to push.

Before pushing implementation changes, satisfy repository requirements: commit
source/test changes, regenerate the manifest as needed, generate and verify a
fresh full signed GPU receipt at the committed source, then commit the receipt.
Push reviewable commits with the validation results and remaining platform gaps.
Treat package publication as its own release step; a Git push alone does not
publish pytest-gpu-proof 0.5.0 or establish that a release is ready.

Additional diagnosis and lasting regression coverage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Minimize representative failures into standalone CUDA examples without GLASS
  dependencies. Preserve valid aliasing and synchronization contracts, inputs,
  flags and failure behavior. Use these to distinguish implementation mistakes
  from compiler behavior; prepare a compiler-report reproducer if justified.
- Vary restrict independently on matrix, output/RHS, scratch and pivot pointers,
  then test relevant combinations. Record the smallest triggering combination
  and inspect its generated code before choosing the scope of a repair.
- Maintain a compact permanent suite of minimal regressions for every relevant
  change, plus a broader seeded randomized and downstream-kernel sweep for
  deeper validation. Run the broader sweep during tonight's validation as
  feasible; save seeds and concrete failing inputs, and minimize failures into
  the permanent suite. Define future periodic execution separately rather than
  assuming a scheduled job already exists.

Phase 1 — make the evidence reliable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Reduce each failing family to a small, valid public-API reproducer. Audit
   disjointness/allowed aliasing, scratch size/alignment, initialization and
   thread participation. Vary scratch contents (zero, nonzero, poison) without
   treating uninitialized required inputs as valid tests.
2. Run each fatal case in a separate process with a per-case timeout, logging
   its parameters before launch. Distinguish numerical failure, CUDA error,
   timeout, skipped, and not-run outcomes. Check launch and synchronization
   errors. Make any failed oracle or incomplete expected case fail the runner.
3. Reproduce failures repeatedly in clean processes. Use targeted memcheck,
   racecheck and synccheck where applicable; instrumentation can alter code
   generation, so retain ordinary optimized runs as the primary reproduction.
4. Use independent oracles: NumPy/SciPy plus normalized backward residuals for
   solves, factor reconstruction with permutations for LU/LDLT, A*Ainv and
   Ainv*A for inverses. Check eigensolver residual, V^T V, eigenvalue ordering
   and reference spectrum; compare invariant subspaces for repeated values.
   For softmax check nonnegativity, normalization, shift invariance and a stable
   reference; for logsumexp use a stable reference and shift identities.
5. Classify findings separately as reproduced failure, supported mechanism,
   static candidate, or no failure in tested cases. Preserve inputs, build
   flags, architecture, compiler version, source SHA and case outcomes.
Exit: trustworthy minimal regressions for each observed failure, with unknown
root causes explicitly recorded and no silent success for missing/failed cases.

Phase 2 — diagnose and repair by mechanism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Prioritize current production impact and severity: wrong triangular solves and
normalization, invalid pivot indices, then nonterminating inversion/eigensolver
paths. Treat all as release blockers for the affected advertised interfaces.

Trace who writes and reads each shared value at each algorithm step. Separate
scalar broadcast, mutable-vector ownership changes, reduction scratch reuse,
and shared indices/branch decisions. Inspect PTX/SASS around the failing
handoff when needed; compare restricted/unrestricted builds and qualifiers on
individual pointers. Verify synchronization and memory visibility independently
of the algebra. Audit direct consumers of affected shared helpers (including
index reductions) rather than only the outer routines that failed.

Evaluate narrowly scoped repairs: register/shuffle broadcast where the execution
scope permits it, explicit shared staging, stable element ownership, or targeted
compiler-visible accesses. A warp shuffle alone cannot publish across warps.
Do not remove callers' restrict qualifiers as the final fix, insert blanket
volatile accesses, or add redundant barriers without a measured justification.
Preserve arbitrary legal block sizes, thread/warp/group contracts, supported
aliasing, and TRAILING_SYNC semantics. Single-thread paths should retain their
register-resident behavior. Keep fixes and regression tests reviewable by family.
Exit: each fixed minimal regression passes with and without restrict; the
mechanism and scope are explained, including remaining uncertainties.

Phase 3 — broaden coverage where failures could propagate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Use a risk-ranked matrix rather than an unbounded Cartesian product:
- Dimensions around algorithm/unrolling boundaries and warp boundaries, odd
  and larger sizes; true runtime sizes passed from host input; supported zero
  and one-size cases. Compile-time and runtime forms are separate obligations.
- Lower/upper, transpose, unit/nonunit, one/multiple RHS, checked/unchecked,
  pivot/no-pivot, SPD/indefinite, pivot-requiring matrices, ill-conditioned and
  scaled inputs, repeated/clustered eigenvalues, and documented failure inputs.
- Shared and global memory; documented in-place aliasing and distinct buffers;
  repeated calls and scratch reuse; inlined/noinline and consumer-sized kernels
  with realistic register pressure. Do not create invalid restrict aliasing.
- Explicit block, bare measured dispatch, warp, thread, full-block and supported
  tiled cgrps forms; 1D and multidimensional block layouts where supported.
- Test default trailing publication with a dependent consumer and no extra
  caller barrier; separately test TRAILING_SYNC=false with the required caller
  synchronization. Include adjacent producer/consumer operations.
- End-to-end bdsv, PCG and box_qp, plus representative downstream generated
  solver kernels. Check solution residuals, convergence/status and iteration
  behavior; a primitive-level pass does not establish composition correctness.
- Fused/batched variants and vendor-wrapper boundaries where these shared
  helpers or scratch conventions are reused.

Repeat critical regressions on the reported sm_86/CUDA 13.3 setup and the local
sm_120/CUDA 13.2 setup when available. Record unavailable configurations as gaps.
Exercise supported optimization/build settings; use lower optimization only as
an investigative control, not a workaround for failure at production settings.
Exit: affected public families and direct consumers have explicit coverage;
untested configurations remain visible instead of being implied by case counts.

Phase 4 — protect performance and tune only after correctness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Capture before/after timings using the repository's benchmark workflow on a
quiet GPU, with warmup, repeated measurements, identical compiler/flags/inputs,
and recorded variance. Include representative dimensions, precisions and thread
counts, plus composition-heavy kernels. Inspect registers, spills, shared-memory
usage, barrier counts and occupancy when explaining regressions.

Compare fixed implementations against correct alternatives. Timing a baseline
that returns wrong answers is informative about cost, not evidence that its
speed must be preserved. Set workload-specific acceptance thresholds after
measuring baseline noise; document significant regressions and their tradeoffs.
Do not trade determinism or correctness for a faster benchmark. If the repair
changes relative performance, regenerate measured dispatch choices with the
owning tuning scripts rather than hand-editing tables.
Exit: correctness gates pass and performance costs are measured and reviewed;
any retuned defaults are regenerated and validated on the corrected code.

Phase 5 — integrate, document and release
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Register permanent regressions in test compilation/cache dependencies, affected
receipt shards and behavioral coverage obligations. Regenerate the public
manifest after header or generated-dispatch changes. Correct the debugging
guide's blanket claim that full-block forms are safe. Document mechanisms,
supported calling contracts and remaining platform gaps without overclaiming.

Run the full required correctness and documentation checks, targeted sanitizer
checks, downstream smoke tests and a fresh full eight-shard GPU receipt before
shipping GLASS. A receipt records the selected execution; it does not substitute
for sufficient tests or establish correctness on untested configurations.

Prepare pytest-gpu-proof 0.5.0 in parallel: version/changelog, complete checks,
CI, package smoke tests, publication and installation verification. Prefer pinning
it in GLASS before the final full run so nvcc metadata is recorded. Do not delay
an urgent GLASS correctness fix solely for the tooling release; if tooling is
blocked, record the compiler/build provenance separately and use the currently
supported attestation workflow, then schedule adoption explicitly. Publishing
and merging are separate actions from this saved plan update.

Completion checklist
~~~~~~~~~~~~~~~~~~~~
[ ] Minimal failures independently reproduced and harness contracts checked.
[ ] Runner fails on numerical errors and records every expected case outcome.
[ ] Root causes explained or remaining uncertainty explicitly documented.
[ ] Repairs and regression tests cover affected interfaces and compositions.
[ ] Critical compiler/GPU matrix run; unavailable environments documented.
[ ] Performance baselines and repair costs measured; retuning validated if needed.
[ ] Required checks, documentation, manifest and full fresh receipt complete.
[ ] pytest-gpu-proof 0.5.0 release/adoption status explicitly resolved.
All items above remain planned unless backed by new recorded results.

Reproduction (from GLASS root)
-----------------------------
D=docs/audits/2026-10-28-restrict
nvcc -O3 -std=c++17 -arch=sm_120 -I. "$D/extended-safe.cu" -o /tmp/glass-audit
 timeout 20 /tmp/glass-audit > /tmp/glass-audit.csv
nvcc -O3 -std=c++17 -arch=sm_120 -DRESTRICT= -I. "$D/extended.cu" -o /tmp/glass-control
 timeout 20 /tmp/glass-control > /tmp/glass-control.csv

For isolated failures compile pivot-repro.cu, ldlt-repro.cu, or extra.cu with
those flags, then use timeout 15 on each executable. Add -lineinfo and run
compute-sanitizer --tool memcheck for the LDLT reproducer. extra.cu first tests
warp softmax, then syev. For the small softmax repro compile softmax-repro.cu;
its control macro is -DAUDIT_RESTRICT= (not -DRESTRICT=).

Artifacts are local and uncommitted. No issue comments, PRs, merges, releases,
or signed receipts were created during the audit.


RESUMED IMPLEMENTATION — 2026-10-28
-----------------------------------
The OP23 dispatch defect was corrected in all exploratory .cu files. Original
pivot-synccheck.log describes the OLD two-call composition, not standalone inv.
Corrected standalone inv_pivoted<float,3> still stalls at seed 0 / two threads;
all 33 no-restrict controls pass. This re-establishes the standalone finding.

Empty asm memory clobbers around barriers did not repair the failures. Targeted
volatile handoff loads did: volatile-experiment.csv and status JSON account for
5,538 cases, all passing. The actual library implementation uses
cooperative_detail::load with a normal-load ThreadBarrier overload. The same
5,538 cases pass in fixed-sweep.csv. These were collected BEFORE the subsequent
getrf/laswp pivot reload addition, and retain the exploratory oracle limitations.

The permanent test has TWO kernel contexts. Generalizing the first kernel for
runtime/global-memory choices suppressed ALL original failures on the baseline.
The minimal shared-memory context was therefore preserved as a separate kernel;
it detects the original numerical, memory-error and timeout regressions on the
unmodified source. The more general kernel checks runtime dimensions, global
memory, multidimensional blocks and immediate consumers with no caller barrier.
Python uses NumPy/SciPy plus inverse round trips and eigensystem reconstruction /
orthogonality, and each case is a separate subprocess with a 20-second timeout.

The expanded context also exposed a gesv LU-pivot handoff failure at float/double
N=3, two threads (earlier exploratory gesv had passed). Reloading the pivot in
getrf and laswp repairs 14 targeted precision/thread cases. This is why sampled
passes are not a blanket claim of immunity.

The existing bdsv, pcg and box_qp test drivers now use restricted kernel pointers
and -O3. Their 53 external-oracle tests pass. The resource canary matches all 18
baseline kernels exactly, with no stack/spills, before the getrf addition.
Strict documentation build passed before final source completion.

Full final regression, sanitizer, timing, signed receipt and push remain pending
at this entry. No performance claim is made from these correctness results.

Focused final checks: 78 optimized cases passed (including all triangle/transpose/
unit-diagonal flag combinations, explicit caller trailing barriers, nontrivial
inverse row swaps and indefinite 2x2 LDLT pivots). All 21 compute-sanitizer runs
(memcheck, racecheck, synccheck x seven affected families, double N=7, 33 threads)
passed; instrumentation can change codegen, so this supplements the plain runs.
All 669 public overload contracts and all 22 behavioral obligations are present;
resource canary remains exactly unchanged at 18 zero-spill/stack kernels.
Timing characterization and its limits are saved in timing-summary.txt.
Full fresh signed run begins only after committing source; it remains the push gate.
