Robotics operators & conventions
================================

GLASS carries a set of **robotics-specialized small operators** alongside the
BLAS/LAPACK surface — the fixed-size device primitives that every GPU robotics
stack otherwise hand-rolls, each with its own (frequently colliding) sign and
storage conventions. They are organized by the workload they serve:

* **Rigid-body dynamics** — the Featherstone spatial 6-D cross products
  (``motion_cross`` / ``force_cross`` / ``force_cross_dual`` and their fused
  ``*_mul`` applies) that every RNEA/ABA/CRBA kernel and analytic dynamics
  gradient is built from.
* **Manifold states (floating bases, orientations)** — the SO(3)/SE(3)/
  quaternion family: ``skew``, ``so3_exp``/``so3_log``, the right/left
  Jacobians and inverses, the SE(3) retract with its 6x6 first derivatives and
  6x6x6 second derivative, and the quaternion primitive set
  (``quat_mul``/``quat_exp``/``quat_rotate``/``quat_to_rot``/…).
* **Constrained trajectory optimization** — second-order-cone projection
  (``soc_project``), PHR augmented-Lagrangian scalars, the relaxed log-barrier,
  and the planar-angle utilities (``angle_wrap``/``angle_diff``/``angle_lerp``).
* **Sampling-based control** — max-shifted ``softmax``/``logsumexp`` (the MPPI
  path-integral weight update) and the signed ``argmax``/``argmin``
  index-payload reductions.
* **Collision checking** — sphere-sphere/sphere-box signed distances with
  gradients, ``transform_sphere``, the branchless ``frame_from_vector`` tangent
  basis, and ``segment_segment_closest``.

Why these live in GLASS rather than in each consumer: the ops are small,
recur across dynamics/planning/control/estimation, and are **error-prone to
hand-roll** — sign conventions, storage order, and small-angle branches are
exactly where independent implementations silently disagree. One tested,
convention-pinned home removes that class of bug (the formulas here are
promoted from Pinocchio-validated generated code and from numpy-oracle-
validated solver kernels, then re-validated against scipy and
finite-difference identities in ``test/test_robotics.py``).

Conventions (read before calling anything)
------------------------------------------

The conventions below are **load-bearing**; a wrong guess at any of them
produces plausible-looking wrong numbers, not crashes.

Spatial (Featherstone) vectors — angular first
   A spatial motion vector is ``v = [ω(3); v_lin(3)]`` and a spatial force
   vector is ``f = [n(3); f_lin(3)]`` (moment first). This is the
   Featherstone/MuJoCo ordering used by the ``motion_cross``/``force_cross``
   family. 6x6 matrices are column-major.

SE(3) tangents — separate ``(ρ, φ)`` arguments, linear-first blocks
   The Lie-family ops never take a packed 6-vector twist: the linear part
   ``ρ`` and angular part ``φ`` are separate 3-vector arguments, so there is
   no input-ordering trap. Output 6x6/6x6x6 blocks index the tangent as
   ``[ρ; φ]`` (linear first), matching Pinocchio's ``dIntegrate``. Note this
   is the OPPOSITE half of the field from the Featherstone family above —
   each family keeps its native literature convention; permute explicitly
   when crossing between them.

Quaternions — Hamilton, compile-time storage layout
   All quaternions are Hamilton quaternions. The storage order is the
   compile-time ``QuatLayout`` tag: ``xyzw`` (default — Eigen/Warp/cuRobo
   storage) or ``wxyz`` (MuJoCo/Ceres/GTSAM/ROS). The math is written once
   against accessor indices; the two layouts are pure storage permutations
   of each other (a pytest gate). ``quat_exp`` takes the FULL rotation
   vector and halves internally.

Matrices — column-major, GLASS-wide
   3x3 and 6x6 outputs are column-major (``M[c*3 + r]`` / ``J[c*6 + r]``);
   the SE(3) Hessian is six stacked column-major 6x6 slices
   (``J2[k*36 + c*6 + r]``).

Small-angle branches
   Every trigonometric map carries a Taylor head so it is smooth through
   θ = 0 (thresholds documented per function); ``so3_log`` routes through the
   Shepperd quaternion extraction so it is stable through θ = π; the
   inverse-Jacobian coefficient uses the half-angle form that is finite at
   θ = π. The SE(3) second-derivative chain computes in double regardless of
   the interface precision (validated against mpmath complex-step ground
   truth in its source project).

Tiers
-----

Every **array-shaped** robotics op spans the three SIMT interfaces — block
(``glass::``), warp (``glass::warp::``), thread (``glass::thread::``) — from
one shared serial core: each active thread computes the small fixed-size
result redundantly in registers and the tier strides the copy-out. That
construction is thread-count **bit-invariant** at block scope (asserted at
1/32/64/256 threads) and keeps the three tiers within FMA-contraction jitter
of each other (asserted at ≤4 ulp for arithmetic maps, tight relative
tolerance for trig chains). Outputs must not alias inputs at block/warp scope.

**Scalar-returning** ops (the angle utilities, the AL/barrier scalars, the
geometry distances) are *tier-free*: they read no ``threadIdx`` and return by
value, so the same ``glass::`` function is correct at any scope — there is
nothing for a tier variant to change, and none exist by design.

Fused vs composed
-----------------

Each fused micro-kernel equals the composition of general kernels it
replaces — ``motion_cross_mul(v, x)`` is exactly ``gemv(motion_cross(v), x)``
without the 36-element temporary, ``so3_exp`` is exactly
``quat_to_rot(quat_exp(φ))`` up to rounding — and the test suite asserts
those equivalences. A hand-rolled copy of any of these formulas computes the
same thing at the same speed; what it cannot give you is the pinned
convention, the identity test suite, and the three tiers. Examples
``18``–``22`` under ``examples/`` walk one use case per family.
