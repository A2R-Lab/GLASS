Robotics operators
==================

The robotics-specialized small-operator families — spatial 6-D ops (cross
products, coordinate transforms, and the 10-parameter inertia: rigid-body
dynamics), the SO(3)/SE(3)/quaternion Lie family with pose-error metrics
(manifold states, their derivatives, and goal costs),
projection/cone/augmented-Lagrangian scalars (constrained trajectory
optimization), geometry distance primitives (sphere-decomposed collision
checking), the 3x3 estimation kit (``eig3``/``svd3``/``closest_rotation``:
ICP, alignment, re-orthonormalization), and the sampling-planner L1 additions
(``softmax``/``logsumexp``/``argmax``/``argmin`` + ``_fast`` twins).

**Read the conventions first**: :doc:`../user_guide/concepts/robotics_conventions`
pins the angular-first spatial ordering, the linear-first SE(3) tangent blocks,
the ``QuatLayout`` storage tag, column-major storage, and the small-angle
branch policy. Every array-shaped op below also exists as ``glass::warp::``
and ``glass::thread::`` (same serial core, one tier per problem-packing
granularity); scalar ops are tier-free.

Spatial 6-D cross products
--------------------------

.. doxygenfile:: src/base/spatial/cross.cuh
   :no-link:

Spatial coordinate transforms
-----------------------------

.. doxygenfile:: src/base/spatial/transform.cuh
   :no-link:

Spatial inertia (10 parameters)
-------------------------------

.. doxygenfile:: src/base/spatial/inertia.cuh
   :no-link:

Quaternions
-----------

.. doxygenfile:: src/base/lie/quat.cuh
   :no-link:

Pose errors
-----------

.. doxygenfile:: src/base/lie/pose.cuh
   :no-link:

SO(3): exp / log / Jacobians
----------------------------

.. doxygenfile:: src/base/lie/so3.cuh
   :no-link:

SE(3): retract + derivatives
----------------------------

.. doxygenfile:: src/base/lie/se3.cuh
   :no-link:

Planar angles
-------------

.. doxygenfile:: src/base/lie/angle.cuh
   :no-link:

Cones & projections
-------------------

.. doxygenfile:: src/base/proj/cone.cuh
   :no-link:

Interval / AL / barrier scalars
-------------------------------

.. doxygenfile:: src/base/proj/interval.cuh
   :no-link:

Geometry distances
------------------

.. doxygenfile:: src/base/geom/sphere.cuh
   :no-link:

.. doxygenfile:: src/base/geom/frame.cuh
   :no-link:

.. doxygenfile:: src/base/geom/segment.cuh
   :no-link:

3x3 estimation kit
------------------

.. doxygenfile:: src/base/est/svd3.cuh
   :no-link:

Fused Gauss-Newton / LM step
----------------------------

.. doxygenfile:: src/base/L3/gn_step.cuh
   :no-link:

Sampling reductions (L1)
------------------------

.. doxygenfile:: src/base/L1/softmax.cuh
   :no-link:

.. doxygenfile:: src/base/L1/argreduce.cuh
   :no-link:
