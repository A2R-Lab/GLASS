Execution Plans (``glass-defaults.cuh``)
========================================

``glass::recommend()`` turns the measured implementation ladder into one
``constexpr`` value. It returns:

* the measured implementation family and execution scope for this shape; and
* a ready-to-use, legal launch packing.

The packing fields are defaults, not a claim that every caller's optimal block
size was measured. Applications may retune them within the explicit operation's
documented launch contract.

The decision is host-side or code-generation-time. A thread, warp, block, and
NVIDIA block implementation require different CUDA launches; ``recommend()``
does not dispatch a device call for you.

The contract
------------

.. code-block:: cuda

   enum class glass::family { native, nvidia };
   enum class glass::scope { thread, warp, block };
   enum class glass::dependency_set { native_only, mathdx };
   struct glass::execution_plan {
       family implementation;
       scope execution_scope;
       uint32_t block_threads;
       uint32_t problems_per_block;
       uint32_t shared_bytes;
   };

   template <glass::op Op, typename T, uint32_t... Dims>
   constexpr glass::execution_plan glass::recommend(
       glass::dependency_set dependencies = glass::dependency_set::native_only,
       uint32_t sm = GLASS_TARGET_SM);

Shape arguments follow the operation's mathematical order:

* Square ladder operations: ``recommend<op, T, N>()``
* Rectangular GEMV: ``recommend<op::gemv, T, M, N>()``
* Rectangular GEMM: ``recommend<op::gemm, T, M, N, K>()``

``native_only`` is deliberately the default. Pass ``dependency_set::mathdx``
to admit measured ``glass::nvidia::block`` and ``glass::nvidia::thread``
candidates. Each measured architecture carries a paired native-only table from
the same capture, rather than approximating vendor-winning cells with a size
heuristic. The result is independent of header include order.

Using a plan
------------

.. code-block:: cuda

   constexpr auto plan = glass::recommend<glass::op::potrf, float, N>(
       glass::dependency_set::mathdx);

   if constexpr (plan.implementation == glass::family::nvidia &&
                 plan.execution_scope == glass::scope::thread) {
       // one problem per thread
       // glass::nvidia::thread::potrf<float, N>(...)
   } else if constexpr (plan.implementation == glass::family::nvidia) {
       // one problem per block; query the explicit wrapper's exact requirements
       // glass::nvidia::block::potrf<float, N>(...)
   } else if constexpr (plan.execution_scope == glass::scope::warp) {
       // one problem per warp
       // glass::warp::potrf<float, N>(...)
   } else if constexpr (plan.execution_scope == glass::scope::thread) {
       // one problem per thread
       // glass::thread::potrf<float, N>(...)
   } else {
       // one problem per block
       // glass::block::potrf<float, N>(...)
   }

For native plans, ``block_threads`` and ``problems_per_block`` are complete
launch guidance. NVIDIA block descriptors own shape-specific thread and shared
memory requirements; those fields use
``execution_plan::dynamic_requirement`` when the explicit backend query must be
consulted.

``GLASS_TARGET_SM``
-------------------

``GLASS_TARGET_SM`` selects the measured architecture table and the MathDx
descriptor architecture from one build setting. It defaults to the shipped
sm_120 seed. Define it explicitly when targeting another GPU, for example
``-DGLASS_TARGET_SM=870``. The historical ``SMS`` macro remains an input alias
for existing build systems.

The shipped tables currently cover sm_120 and sm_87. Unmeasured architectures
use conservative generic choices until ``bench/tune.py`` adds a measured table.
``examples/08_backend_picker.cu`` is a complete native-only launcher.

This launch-level plan is separate from ``glass::dispatch_body()``. The latter
selects a measured thread-0, warp-0, or full-block implementation *inside* the
fixed block-scope contract of bare ``glass::op`` calls. See
:doc:`../user_guide/concepts/namespaces`.
