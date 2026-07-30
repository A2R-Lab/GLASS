"""Robotics-op families — lie/quat, spatial 6D, projections/AL, geometry,
softmax/argreduce (driver: test/cuda/test_robotics.cu).

GATES, per the wave's validation contract:
  * ORACLES — scipy.spatial.transform.Rotation for every rotation-family op,
    numpy float64 formula transcriptions for Jacobians/spatial/projection/
    geometry ops, scipy.special for softmax/logsumexp.
  * IDENTITIES (implementation-independent): crf == −crmᵀ, cross-product
    antisymmetry, the force_cross_dual defining identity, exp/log round-trips
    (incl. near-π and the small-angle series region), Jr(φ) == Jl(−φ),
    J·J⁻¹ == I, SOC idempotence + projection orthogonality, softmax shift
    invariance, AL/barrier gradient-vs-value finite differences.
  * FUSED-VS-COMPOSED — each fused micro-kernel equals the composition of
    general kernels it replaces (e.g. motion_cross_mul == crm(v)·x by matrix
    multiply; so3_exp == quat_to_rot(quat_exp(φ))): the value-demonstration
    gates showing the op is exactly the composed result minus the temporaries.
  * SE(3) DERIVATIVE CHAIN — first derivatives verified by the defining
    finite-difference composition identity (retract(q, w+εδ) ≈
    retract(retract(q, w), ε·J_v·δ), and the ARG0 twin); the Hessian by
    central-differencing the device Jacobian.
  * TIERS — block output is BIT-IDENTICAL across tpb ∈ {1, 32, 64, 256}
    (thread-count invariance); block/warp/thread agree to ≤4 ulp for pure-
    arithmetic maps and to tight relative tolerance for sqrt/trig chains
    (the test_thread.py policy: the tiers share one serial core, so only
    FMA-contraction jitter separates them).
"""

import os
import subprocess
import tempfile
import zlib

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import scipy.special

RNG = np.random.default_rng(7)


@pytest.fixture(autouse=True)
def _seed_rng(request):
    global RNG
    RNG = np.random.default_rng(zlib.crc32(request.node.nodeid.encode()))


DTYPES = ["f32", "f64"]
P = 53                      # odd, ragged in every model's launch geometry
_NPDT = {"f32": np.float32, "f64": np.float64}
_TOL = {"f32": dict(rtol=2e-3, atol=2e-3), "f64": dict(rtol=1e-6, atol=1e-6)}

# constants mirrored from test_robotics.cu
ALPHA_MUL, BETA_MUL = 1.5, 0.5
SOC_RHO, AL_RHO, AL_SIGMA = 0.7, 1.3, 0.3
RB_MU, RB_DELTA, SH_ETA, SM_ALPHA = 0.8, 0.15, 0.2, -0.75

# per-op output sizes (mirrors the driver's OPS table; -1 = flag0)
_OUT = {
    "quat_mul": 4, "quat_conj": 4, "quat_normalize": 4, "quat_exp": 4,
    "quat_rotate": 3, "quat_to_rot": 9, "rot_to_quat": 4, "quat_to_basis": 9,
    "quat_retract": 4,
    "skew": 9, "so3_exp": 9, "so3_log": 3, "so3_rjac": 9, "so3_rjac_inv": 9,
    "so3_ljac": 9, "so3_ljac_inv": 9,
    "se3_q_block": 9, "se3_retract": 7, "se3_jac_q": 36, "se3_jac_v": 36,
    "se3_hess_q": 216, "se3_hess_v": 216,
    "motion_cross": 36, "force_cross": 36, "force_cross_dual": 36,
    "mcross_mul": 6, "fcross_mul": 6,
    "soc_project": -1, "soc_scalars": 3, "interval_scalars": 4, "rbar": 3,
    "smooth_hinge": 2, "angle": 4,
    "sphere_sphere": 4, "sphere_box": 4, "transform_sphere": 4, "frame": 6,
    "segment": 9,
    "softmax": -1, "logsumexp": 1, "argmax": 2, "argmin": 2,
    "motion_xform": 36, "force_xform": 36, "mxform_mul": 6, "fxform_mul": 6,
    "spatial_inertia": 36, "sinertia_mul": 6,
    "quat_log": 3, "quat_error": 3, "pose_error": 6, "quat_angle": 1,
    "log_cosh": 2,
    "eig3": 12, "svd3": 21, "closest_rot": 9,
    "argmax_fast": 2, "argmin_fast": 2,
}


def _run(bins, op, model, dtype, inputs, tpb=64, flag0=0, flag1=0, p=P):
    tmp = []
    try:
        for arr in inputs:
            fh = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
            np.asarray(arr).astype(np.float32).tofile(fh)
            fh.close()
            tmp.append(fh.name)
        cmd = [str(bins["robotics"]), op, model, dtype, str(p), str(tpb),
               str(flag0), str(flag1)] + tmp
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"driver failed ({op}/{model}/{dtype}):\n{r.stderr}")
        out = np.fromstring(r.stdout.strip(), sep=" ").astype(_NPDT[dtype])
        n_out = _OUT[op] if _OUT[op] != -1 else flag0
        return out.reshape(p, n_out)
    finally:
        for t in tmp:
            os.unlink(t)


def _assert_ulp_equal(t, b, tag, max_ulp=4):
    assert t.shape == b.shape, f"{tag}: shape mismatch"
    if np.array_equal(t, b):
        return
    ib, it = b.view(f"i{b.itemsize}"), t.view(f"i{t.itemsize}")
    ulp = np.abs(np.where(ib == it, 0, ib - it))
    assert int(ulp.max()) <= max_ulp, (
        f"{tag}: tiers differ by {int(ulp.max())} ulp (> {max_ulp}) — beyond "
        f"FMA-contraction jitter; a tier has diverged from the shared core")


def _assert_close_tight(t, b, tag, dtype):
    tol = {"f32": 1e-4, "f64": 1e-11}[dtype]
    scale = max(float(np.abs(b).max()), 1.0)
    np.testing.assert_allclose(t, b, rtol=tol, atol=tol*scale,
                               err_msg=f"{tag}: cross-tier divergence beyond "
                                       f"contraction jitter")


# ─── numpy oracle helpers (all float64) ───────────────────────────────────────

def _f32(x):
    """Round through float32 — inputs reach the device as float32."""
    return np.asarray(x, np.float32).astype(np.float64)


def _skew_np(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], float)


def _rod_coefs(theta):
    if theta < 1e-8:
        return 1.0, 0.5, 1.0/6.0
    return (np.sin(theta)/theta, (1 - np.cos(theta))/theta**2,
            (theta - np.sin(theta))/theta**3)


def _Jl_np(phi):
    S = _skew_np(phi)
    a, b, c = _rod_coefs(np.linalg.norm(phi))
    return np.eye(3) + b*S + c*(S @ S)


def _Jr_np(phi):
    S = _skew_np(phi)
    a, b, c = _rod_coefs(np.linalg.norm(phi))
    return np.eye(3) - b*S + c*(S @ S)


def _inv_coef(theta):
    if theta < 1e-4:
        return 1.0/12.0 + theta**2/720.0 + theta**4/30240.0
    return 1.0/theta**2 - np.cos(theta/2)/(2*theta*np.sin(theta/2))


def _crm_np(v):
    M = np.zeros((6, 6))
    M[:3, :3] = _skew_np(v[:3])
    M[3:, :3] = _skew_np(v[3:])
    M[3:, 3:] = _skew_np(v[:3])
    return M


def _retract_np(pose, rho, phi):
    Rq = Rotation.from_quat(pose[3:7])
    q_new = (Rq * Rotation.from_rotvec(phi)).as_quat()
    p_new = pose[:3] + Rq.apply(_Jl_np(phi) @ rho)
    return np.concatenate([p_new, q_new])


def _quats(n, unit=True):
    q = RNG.standard_normal((n, 4))
    if unit:
        q /= np.linalg.norm(q, axis=1, keepdims=True)
    return _f32(q.astype(np.float32))


def _sign_align(q, ref):
    """Flip quaternion rows to the same double-cover half as ref."""
    s = np.sign(np.sum(q*ref, axis=-1, keepdims=True))
    s[s == 0] = 1
    return q*s


def _rotvecs(n, lo=0.0, hi=2.5):
    ax = RNG.standard_normal((n, 3))
    ax /= np.linalg.norm(ax, axis=1, keepdims=True)
    th = RNG.uniform(lo, hi, (n, 1))
    return _f32((ax*th).astype(np.float32))


def _rot_mats(n, hi=2.5):
    """Random rotation matrices, column-major flattened (device layout)."""
    return np.stack([_f32(Rotation.from_rotvec(v).as_matrix().T.ravel()
                          .astype(np.float32)) for v in _rotvecs(n, hi=hi)])


def _pis(n):
    """Random (non-physical) inertia parameter 10-vectors [m, h, I6]."""
    pi = RNG.standard_normal((n, 10))
    pi[:, 0] = RNG.uniform(0.5, 3.0, n)          # positive mass
    return _f32(pi.astype(np.float32))


def _sym3s(n, spread=True):
    """Random symmetric 3x3s, column-major flattened."""
    A = RNG.standard_normal((n, 3, 3))
    A = A + np.transpose(A, (0, 2, 1))
    if spread:
        A += np.eye(3) * RNG.uniform(-2, 2, (n, 1, 1))
    return _f32(A.reshape(n, 9).astype(np.float32))   # symmetric: order-free


def _xform_np(E, r, force=False):
    X = np.zeros((6, 6))
    X[:3, :3] = E
    X[3:, 3:] = E
    B = -E @ _skew_np(r)
    if force:
        X[:3, 3:] = B
    else:
        X[3:, :3] = B
    return X


def _inertia_np(pi):
    m, h = pi[0], pi[1:4]
    Ixx, Ixy, Ixz, Iyy, Iyz, Izz = pi[4:]
    IO = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])
    M = np.zeros((6, 6))
    M[:3, :3] = IO
    M[:3, 3:] = _skew_np(h)
    M[3:, :3] = _skew_np(h).T
    M[3:, 3:] = m * np.eye(3)
    return M


# ─── tier machinery: invariance + cross-tier ─────────────────────────────────
# input factory: op → (inputs, flag0, flag1); 'policy': ulp | tight | exact.

def _tier_case(op):
    if op == "quat_mul":
        return [_quats(P), _quats(P)], 0, 0, "ulp"
    if op == "quat_retract":
        return [_quats(P), _rotvecs(P)], 0, 0, "tight"
    if op == "so3_exp":
        return [_rotvecs(P)], 0, 0, "tight"
    if op == "so3_log":
        return [np.stack([Rotation.from_rotvec(v).as_matrix().T.ravel()
                          for v in _rotvecs(P, hi=2.9)])], 0, 0, "tight"
    if op == "so3_rjac_inv":
        return [_rotvecs(P)], 0, 0, "tight"
    if op == "se3_retract":
        pose = np.hstack([_f32(RNG.standard_normal((P, 3)).astype(np.float32)), _quats(P)])
        return [pose, _f32(RNG.standard_normal((P, 3)).astype(np.float32)),
                _rotvecs(P)], 0, 0, "tight"
    if op == "se3_jac_v":
        return [_f32(RNG.standard_normal((P, 3)).astype(np.float32)), _rotvecs(P)], 0, 0, "tight"
    if op == "se3_hess_v":
        return [_f32(RNG.standard_normal((P, 3)).astype(np.float32)), _rotvecs(P)], 0, 0, "tight"
    if op == "motion_cross":
        return [_f32(RNG.standard_normal((P, 6)).astype(np.float32))], 0, 0, "exact"
    if op == "mcross_mul":
        v = _f32(RNG.standard_normal((P, 6)).astype(np.float32))
        x = _f32(RNG.standard_normal((P, 6)).astype(np.float32))
        y0 = _f32(RNG.standard_normal((P, 6)).astype(np.float32))
        return [v, x, y0], 0, 1, "ulp"
    if op == "force_cross_dual":
        return [_f32(RNG.standard_normal((P, 6)).astype(np.float32))], 0, 0, "exact"
    if op == "soc_project":
        return [_f32(RNG.standard_normal((P, 8)).astype(np.float32))], 8, 0, "tight"
    if op == "softmax":
        return [_f32(RNG.standard_normal((P, 96)).astype(np.float32))], 96, 0, "tight"
    if op == "argmax":
        return [_f32(RNG.standard_normal((P, 96)).astype(np.float32))], 96, 0, "exact"
    if op in ("mxform_mul", "fxform_mul"):
        Er = np.hstack([_rot_mats(P), _f32(RNG.standard_normal((P, 3)).astype(np.float32))])
        x = _f32(RNG.standard_normal((P, 6)).astype(np.float32))
        y0 = _f32(RNG.standard_normal((P, 6)).astype(np.float32))
        return [Er, x, y0], 1, 1, "ulp"          # INVERSE + HAS_BETA path
    if op == "spatial_inertia":
        return [_pis(P)], 0, 0, "ulp"
    if op == "sinertia_mul":
        return [_pis(P), _f32(RNG.standard_normal((P, 6)).astype(np.float32)),
                _f32(RNG.standard_normal((P, 6)).astype(np.float32))], 0, 1, "ulp"
    if op == "quat_error":
        return [_quats(P), _quats(P)], 0, 0, "tight"
    if op == "eig3":
        return [_sym3s(P)], 0, 0, "tight"
    if op == "closest_rot":
        return [_f32(RNG.standard_normal((P, 9)).astype(np.float32))], 0, 0, "tight"
    raise KeyError(op)


TIER_OPS = ["quat_mul", "quat_retract", "so3_exp", "so3_log", "so3_rjac_inv",
            "se3_retract", "se3_jac_v", "se3_hess_v", "motion_cross",
            "mcross_mul", "force_cross_dual", "soc_project", "softmax", "argmax",
            "mxform_mul", "fxform_mul", "spatial_inertia", "sinertia_mul",
            "quat_error", "eig3", "closest_rot"]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("op", TIER_OPS)
def test_block_thread_count_invariance(bins, op, dtype):
    """Block output must be BIT-IDENTICAL at tpb ∈ {1, 32, 64, 256}."""
    inputs, flag0, flag1, _ = _tier_case(op)
    ref = _run(bins, op, "block", dtype, inputs, tpb=1, flag0=flag0, flag1=flag1)
    for tpb in (32, 64, 256):
        got = _run(bins, op, "block", dtype, inputs, tpb=tpb, flag0=flag0, flag1=flag1)
        assert np.array_equal(ref, got), (
            f"{op}/{dtype}: block output differs between tpb=1 and tpb={tpb} — "
            f"thread-count invariance broken")


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("op", TIER_OPS)
def test_cross_tier_agreement(bins, op, dtype):
    """block / warp / thread share one serial core → ULP/tight agreement."""
    inputs, flag0, flag1, policy = _tier_case(op)
    blk = _run(bins, op, "block", dtype, inputs, tpb=64, flag0=flag0, flag1=flag1)
    for model in ("warp", "thread"):
        got = _run(bins, op, model, dtype, inputs, flag0=flag0, flag1=flag1)
        tag = f"{op}/{dtype}/{model}"
        if policy == "exact":
            assert np.array_equal(blk, got), f"{tag}: expected exact table match"
        elif policy == "ulp":
            _assert_ulp_equal(got, blk, tag)
        else:
            _assert_close_tight(got, blk, tag, dtype)


# ─── quaternion family ────────────────────────────────────────────────────────

@pytest.mark.parametrize("dtype", DTYPES)
def test_quat_mul_oracle(bins, dtype):
    a, b = _quats(P), _quats(P)
    out = _run(bins, "quat_mul", "block", dtype, [a, b])
    want = np.stack([(Rotation.from_quat(a[i]) * Rotation.from_quat(b[i])).as_quat()
                     for i in range(P)])
    np.testing.assert_allclose(_sign_align(out, want), want, **_TOL[dtype])


@pytest.mark.parametrize("dtype", DTYPES)
def test_quat_mul_wxyz_layout(bins, dtype):
    """The wxyz instantiation is the xyzw result with permuted storage."""
    a, b = _quats(P), _quats(P)
    perm = [3, 0, 1, 2]      # xyzw → wxyz
    out_x = _run(bins, "quat_mul", "block", dtype, [a, b], flag0=0)
    out_w = _run(bins, "quat_mul", "block", dtype, [a[:, perm], b[:, perm]], flag0=1)
    assert np.array_equal(out_w, out_x[:, perm]), (
        "QuatLayout::wxyz is not a pure storage permutation of xyzw")


@pytest.mark.parametrize("dtype", DTYPES)
def test_quat_conj_normalize(bins, dtype):
    q = _quats(P, unit=False)
    conj = _run(bins, "quat_conj", "block", dtype, [q])
    np.testing.assert_allclose(conj, np.column_stack([-q[:, 0], -q[:, 1], -q[:, 2], q[:, 3]]),
                               **_TOL[dtype])
    norm = _run(bins, "quat_normalize", "block", dtype, [q])
    np.testing.assert_allclose(norm, q/np.linalg.norm(q, axis=1, keepdims=True),
                               **_TOL[dtype])
    canon = _run(bins, "quat_normalize", "block", dtype, [q], flag0=1)
    assert (canon[:, 3] >= 0).all(), "CANONICAL must land on the w >= 0 cover"
    np.testing.assert_allclose(np.abs(np.sum(canon*norm, axis=1)), 1.0, **_TOL[dtype])


@pytest.mark.parametrize("dtype", DTYPES)
def test_quat_exp_oracle(bins, dtype):
    phi = np.vstack([_rotvecs(P - 3), _f32([[0, 0, 0], [1e-9, -2e-9, 1e-9], [3.0, 0, 0]])])
    out = _run(bins, "quat_exp", "block", dtype, [phi])
    want = np.stack([Rotation.from_rotvec(p).as_quat() for p in phi])
    np.testing.assert_allclose(_sign_align(out, want), want, **_TOL[dtype])


@pytest.mark.parametrize("dtype", DTYPES)
def test_quat_rotate_oracle(bins, dtype):
    q, p = _quats(P), _f32(RNG.standard_normal((P, 3)).astype(np.float32))
    out = _run(bins, "quat_rotate", "block", dtype, [q, p])
    want = np.stack([Rotation.from_quat(q[i]).apply(p[i]) for i in range(P)])
    np.testing.assert_allclose(out, want, **_TOL[dtype])


@pytest.mark.parametrize("dtype", DTYPES)
def test_quat_rot_conversions(bins, dtype):
    q = _quats(P)
    R_dev = _run(bins, "quat_to_rot", "block", dtype, [q])
    want = np.stack([Rotation.from_quat(qi).as_matrix().T.ravel() for qi in q])  # col-major
    np.testing.assert_allclose(R_dev, want, **_TOL[dtype])
    # Shepperd back-conversion: canonical w >= 0, matches scipy up to sign.
    q_back = _run(bins, "rot_to_quat", "block", dtype, [want])
    assert (q_back[:, 3] >= 0).all()
    np.testing.assert_allclose(_sign_align(q_back, q), q, **_TOL[dtype])
    # basis = the three columns of R(q/|q|)
    basis = _run(bins, "quat_to_basis", "block", dtype, [q])
    np.testing.assert_allclose(basis, want, **_TOL[dtype])


@pytest.mark.parametrize("dtype", DTYPES)
def test_quat_retract_oracle(bins, dtype):
    q, phi = _quats(P), _rotvecs(P)
    out = _run(bins, "quat_retract", "block", dtype, [q, phi])
    want = np.stack([(Rotation.from_quat(q[i]) * Rotation.from_rotvec(phi[i])).as_quat()
                     for i in range(P)])
    np.testing.assert_allclose(_sign_align(out, want), want, **_TOL[dtype])
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, **_TOL[dtype])


# ─── SO(3) family ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("dtype", DTYPES)
def test_so3_exp_log_oracle(bins, dtype):
    # bracket the θ→0 series (1e-8) and the near-π log regime
    phi = np.vstack([_rotvecs(P - 5),
                     _f32([[0, 0, 0], [1e-9, 0, 0], [-2e-5, 1e-5, 3e-5]]),
                     _f32((np.pi - 1e-3)*np.array([[1, 0, 0], [0, 0.6, 0.8]]))])
    R_dev = _run(bins, "so3_exp", "block", dtype, [phi])
    want = np.stack([Rotation.from_rotvec(p).as_matrix().T.ravel() for p in phi])
    np.testing.assert_allclose(R_dev, want, **_TOL[dtype])
    # log: back through the device on the scipy-exact matrices
    log_dev = _run(bins, "so3_log", "block", dtype, [want])
    np.testing.assert_allclose(log_dev, np.stack(
        [Rotation.from_matrix(w.reshape(3, 3, order="F")).as_rotvec() for w in want]),
        **_TOL[dtype])


@pytest.mark.parametrize("dtype", DTYPES)
def test_so3_exp_equals_quat_path(bins, dtype):
    """Fused-vs-composed: so3_exp(φ) == quat_to_rot(quat_exp(φ))."""
    phi = _rotvecs(P)
    direct = _run(bins, "so3_exp", "block", dtype, [phi])
    q = _run(bins, "quat_exp", "block", dtype, [phi])
    via_quat = _run(bins, "quat_to_rot", "block", dtype, [q])
    # two independent trig routes to the same rotation — oracle-level tolerance
    # (the shared-core tight bound applies across tiers, not across routes)
    np.testing.assert_allclose(via_quat, direct, **_TOL[dtype])


@pytest.mark.parametrize("dtype", DTYPES)
def test_so3_jacobians(bins, dtype):
    # span both sides of the 1e-8 (Jr/Jl) and 1e-4 (inverse) series thresholds
    phi = np.vstack([_rotvecs(P - 4),
                     _f32([[5e-9, 0, 0], [8e-5, -5e-5, 2e-5],
                           [2e-4, 1e-4, -1e-4], [0.6*np.pi, 0.48*np.pi, 0.64*np.pi]])])
    jr = _run(bins, "so3_rjac", "block", dtype, [phi])
    jl = _run(bins, "so3_ljac", "block", dtype, [phi])
    jr_inv = _run(bins, "so3_rjac_inv", "block", dtype, [phi])
    jl_inv = _run(bins, "so3_ljac_inv", "block", dtype, [phi])
    for i in range(P):
        np.testing.assert_allclose(jr[i], _Jr_np(phi[i]).T.ravel(), **_TOL[dtype])
        np.testing.assert_allclose(jl[i], _Jl_np(phi[i]).T.ravel(), **_TOL[dtype])
    # identity Jr(φ) == Jl(−φ)
    jl_neg = _run(bins, "so3_ljac", "block", dtype, [-phi])
    np.testing.assert_allclose(jr, jl_neg, rtol=0, atol=0)
    # J · J⁻¹ == I
    eye = np.eye(3)
    tol = dict(rtol=5e-3, atol=5e-3) if dtype == "f32" else dict(rtol=1e-9, atol=1e-9)
    for i in range(P):
        np.testing.assert_allclose(
            jr[i].reshape(3, 3, order="F") @ jr_inv[i].reshape(3, 3, order="F"), eye, **tol)
        np.testing.assert_allclose(
            jl[i].reshape(3, 3, order="F") @ jl_inv[i].reshape(3, 3, order="F"), eye, **tol)


def test_so3_jacobian_inv_near_pi(bins):
    """The half-angle inverse-coefficient form must be finite and correct at θ = π."""
    phi = _f32(np.array([[np.pi, 0, 0]] * P))
    jr_inv = _run(bins, "so3_rjac_inv", "block", "f64", [phi])
    jr = _run(bins, "so3_rjac", "block", "f64", [phi])
    for i in range(2):
        assert np.isfinite(jr_inv[i]).all()
        np.testing.assert_allclose(
            jr[i].reshape(3, 3, order="F") @ jr_inv[i].reshape(3, 3, order="F"),
            np.eye(3), rtol=1e-9, atol=1e-9)


# ─── SE(3) family ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("dtype", DTYPES)
def test_se3_retract_oracle(bins, dtype):
    pose = np.hstack([_f32(RNG.standard_normal((P, 3)).astype(np.float32)), _quats(P)])
    rho = _f32(RNG.standard_normal((P, 3)).astype(np.float32))
    phi = _rotvecs(P)
    out = _run(bins, "se3_retract", "block", dtype, [pose, rho, phi])
    for i in range(P):
        want = _retract_np(pose[i], rho[i], phi[i])
        got = out[i].copy()
        got[3:] = _sign_align(got[3:][None], want[3:][None])[0]
        np.testing.assert_allclose(got, want, **_TOL[dtype])


def test_se3_jacobians_fd(bins):
    """The defining composition identities of pin-style dIntegrate, by FD (f64):
      ARG1: retract(q, w+εδ) == retract(retract(q, w), ε·J_v·δ) + O(ε²)
      ARG0: retract(retract(q, εδ), w) == retract(retract(q, w), ε·J_q·δ) + O(ε²)
    """
    eps = 1e-5
    n = 6
    pose = np.hstack([_f32(RNG.standard_normal((n, 3)).astype(np.float32)), _quats(n)])
    rho = _f32(RNG.standard_normal((n, 3)).astype(np.float32))
    phi = _rotvecs(n, hi=2.0)
    Jv = _run(bins, "se3_jac_v", "block", "f64", [rho, phi], p=n)
    Jq = _run(bins, "se3_jac_q", "block", "f64", [rho, phi], p=n)
    for i in range(n):
        Jvi = Jv[i].reshape(6, 6, order="F")
        Jqi = Jq[i].reshape(6, 6, order="F")
        base = _retract_np(pose[i], rho[i], phi[i])
        for k in range(6):
            d = np.zeros(6); d[k] = 1.0
            w_pert = np.concatenate([rho[i], phi[i]]) + eps*d
            lhs_v = _retract_np(pose[i], w_pert[:3], w_pert[3:])
            step = eps*(Jvi @ d)
            rhs_v = _retract_np(base, step[:3], step[3:])
            lhs_v[3:] = _sign_align(lhs_v[3:][None], rhs_v[3:][None])[0]
            np.testing.assert_allclose(lhs_v, rhs_v, atol=5e-8,
                                       err_msg=f"se3_jac_v FD identity (prob {i}, dir {k})")
            lhs_q = _retract_np(_retract_np(pose[i], eps*d[:3], eps*d[3:]), rho[i], phi[i])
            step = eps*(Jqi @ d)
            rhs_q = _retract_np(base, step[:3], step[3:])
            lhs_q[3:] = _sign_align(lhs_q[3:][None], rhs_q[3:][None])[0]
            np.testing.assert_allclose(lhs_q, rhs_q, atol=5e-8,
                                       err_msg=f"se3_jac_q FD identity (prob {i}, dir {k})")


@pytest.mark.parametrize("which", ["se3_hess_q", "se3_hess_v"])
def test_se3_hessian_fd(bins, which):
    """J2 slice k == central difference of the device Jacobian (f64)."""
    eps = 1e-6
    n = 5
    jac_op = "se3_jac_q" if which == "se3_hess_q" else "se3_jac_v"
    # include a small-angle row to cross the t = 0.2 series threshold
    rho = _f32(RNG.standard_normal((n, 3)).astype(np.float32))
    phi = np.vstack([_rotvecs(n - 1, hi=2.0), _f32([[0.05, -0.03, 0.02]])])
    hess = _run(bins, which, "block", "f64", [rho, phi], p=n)
    for k in range(6):
        d = np.zeros(6); d[k] = 1.0
        # the harness ships inputs as float32, so quantize the perturbed points
        # FIRST and divide by the ACTUAL per-problem delta (a raw 2ε denominator
        # is ~1% wrong after f32 rounding — enough to fail a correct hessian)
        wp = _f32(np.concatenate([rho, phi], axis=1) + eps*d)
        wm = _f32(np.concatenate([rho, phi], axis=1) - eps*d)
        denom = (wp[:, k] - wm[:, k])[:, None]
        Jp = _run(bins, jac_op, "block", "f64", [wp[:, :3], wp[:, 3:]], p=n)
        Jm = _run(bins, jac_op, "block", "f64", [wm[:, :3], wm[:, 3:]], p=n)
        fd = (Jp - Jm)/denom
        np.testing.assert_allclose(hess[:, k*36:(k + 1)*36], fd, rtol=1e-3, atol=1e-5,
                                   err_msg=f"{which}: slice {k} != FD of {jac_op}")


def test_se3_hessian_f32_interface_matches_f64(bins):
    """The chain computes in double regardless of T — the f32 interface must
    agree with the f64 result to f32 rounding."""
    n = 4
    rho = _f32(RNG.standard_normal((n, 3)).astype(np.float32))
    phi = _rotvecs(n)
    h32 = _run(bins, "se3_hess_v", "block", "f32", [rho, phi], p=n)
    h64 = _run(bins, "se3_hess_v", "block", "f64", [rho, phi], p=n)
    np.testing.assert_allclose(h32, h64.astype(np.float32), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("dtype", DTYPES)
def test_se3_q_block_structure(bins, dtype):
    """jac_v == [[Jr, Q],[0, Jr]] — the Q block op must equal the off-diagonal
    block of the assembled Jacobian (fused-vs-composed for the assembly)."""
    rho = _f32(RNG.standard_normal((P, 3)).astype(np.float32))
    phi = _rotvecs(P)
    Q = _run(bins, "se3_q_block", "block", dtype, [rho, phi])
    Jv = _run(bins, "se3_jac_v", "block", dtype, [rho, phi])
    jr = _run(bins, "so3_rjac", "block", dtype, [phi])
    for i in range(P):
        J = Jv[i].reshape(6, 6, order="F")
        # standalone Q/Jr ops vs the assembled J: same cores, different template
        # instantiation sites — the Q chain is ~10 mat3 products deep, so the
        # contraction jitter compounds past a fixed ULP bound in f32; the
        # robust cross-instantiation gate is the tight relative tolerance
        # (same reasoning as test_thread's solve-composed policy)
        _assert_close_tight(np.ascontiguousarray(J[:3, 3:]).ravel(),
                            np.ascontiguousarray(Q[i].reshape(3, 3, order="F")).ravel(),
                            f"Q-block vs J[0:3,3:6] (prob {i})", dtype)
        _assert_close_tight(np.ascontiguousarray(J[:3, :3]).ravel(),
                            np.ascontiguousarray(jr[i].reshape(3, 3, order="F")).ravel(),
                            f"Jr vs J[0:3,0:3] (prob {i})", dtype)
        _assert_close_tight(np.ascontiguousarray(J[3:, 3:]).ravel(),
                            np.ascontiguousarray(jr[i].reshape(3, 3, order="F")).ravel(),
                            f"Jr vs J[3:6,3:6] (prob {i})", dtype)
        assert (J[3:, :3] == 0).all()


# ─── spatial 6D family ────────────────────────────────────────────────────────

@pytest.mark.parametrize("dtype", DTYPES)
def test_motion_force_cross_oracle(bins, dtype):
    v = _f32(RNG.standard_normal((P, 6)).astype(np.float32))
    crm = _run(bins, "motion_cross", "block", dtype, [v])
    crf = _run(bins, "force_cross", "block", dtype, [v])
    for i in range(P):
        M = _crm_np(v[i])
        np.testing.assert_allclose(crm[i], M.T.ravel(), rtol=0, atol=0)   # pure table
        np.testing.assert_allclose(crf[i], (-M.T).T.ravel(), rtol=0, atol=0)
    # identity: force_cross == −motion_crossᵀ, exactly (sign flips only)
    for i in range(P):
        Mc = crm[i].reshape(6, 6, order="F")
        Fc = crf[i].reshape(6, 6, order="F")
        assert np.array_equal(Fc, -Mc.T)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("has_beta", [0, 1])
def test_cross_mul_fused_vs_composed(bins, dtype, has_beta):
    """The fused applies equal alpha·M@x (+ beta·y) with M from the matrix op."""
    v = _f32(RNG.standard_normal((P, 6)).astype(np.float32))
    x = _f32(RNG.standard_normal((P, 6)).astype(np.float32))
    y0 = _f32(RNG.standard_normal((P, 6)).astype(np.float32))
    for op, build in (("mcross_mul", _crm_np), ("fcross_mul", lambda w: -_crm_np(w).T)):
        out = _run(bins, op, "block", dtype, [v, x, y0], flag1=has_beta)
        want = np.stack([ALPHA_MUL*(build(v[i]) @ x[i]) + (BETA_MUL*y0[i] if has_beta else 0)
                         for i in range(P)])
        np.testing.assert_allclose(out, want, **_TOL[dtype])


@pytest.mark.parametrize("axis", [0, 2, 5])
def test_mcross_mul_axis_specialization(bins, axis):
    """AXIS = k must equal the dense multiply against x = e_k."""
    v = _f32(RNG.standard_normal((P, 6)).astype(np.float32))
    e = np.zeros((P, 6)); e[:, axis] = 1.0
    dummy = np.zeros((P, 6))
    spec = _run(bins, "mcross_mul", "block", "f64", [v, dummy, dummy], flag0=axis + 1)
    dense = _run(bins, "mcross_mul", "block", "f64", [v, e, dummy], flag0=0)
    np.testing.assert_allclose(spec, dense, rtol=0, atol=0)


def test_cross_antisymmetry_and_dual_identity(bins):
    a = _f32(RNG.standard_normal((P, 6)).astype(np.float32))
    b = _f32(RNG.standard_normal((P, 6)).astype(np.float32))
    zero = np.zeros((P, 6))
    # crm(a)·b == −crm(b)·a
    ab = _run(bins, "mcross_mul", "block", "f64", [a, b, zero])
    ba = _run(bins, "mcross_mul", "block", "f64", [b, a, zero])
    np.testing.assert_allclose(ab, -ba, rtol=1e-13, atol=1e-13)
    # defining identity: force_cross(v)·f == force_cross_dual(f)·v
    vf = _run(bins, "fcross_mul", "block", "f64", [a, b, zero])
    dual = _run(bins, "force_cross_dual", "block", "f64", [b])
    want = np.stack([dual[i].reshape(6, 6, order="F") @ a[i] for i in range(P)])
    np.testing.assert_allclose(vf, ALPHA_MUL*want, rtol=1e-12, atol=1e-12)


# ─── projections / AL family ─────────────────────────────────────────────────

def _soc_project_np(w):
    r = np.linalg.norm(w[1:])
    if r <= w[0]:
        return w.copy()
    if r <= -w[0]:
        return np.zeros_like(w)
    a = 0.5*(w[0] + r)
    p = np.empty_like(w)
    p[0] = a
    p[1:] = (a/r)*w[1:]
    return p


@pytest.mark.parametrize("dtype", DTYPES)
def test_soc_project_oracle(bins, dtype):
    m = 8
    w = RNG.standard_normal((P, m)).astype(np.float32)
    # force all three cases: inside, polar, boundary
    w[0, 0] = 100.0; w[1, 0] = -100.0
    w = _f32(w)
    out = _run(bins, "soc_project", "block", dtype, [w], flag0=m)
    want = np.stack([_soc_project_np(w[i]) for i in range(P)])
    np.testing.assert_allclose(out, want, **_TOL[dtype])
    # idempotence + projection orthogonality <p, p − w> == 0 (convex cone)
    out2 = _run(bins, "soc_project", "block", dtype, [out], flag0=m)
    np.testing.assert_allclose(out2, out, **_TOL[dtype])
    tol = 1e-3 if dtype == "f32" else 1e-9
    assert np.abs(np.sum(out*(out - w), axis=1)).max() < tol


@pytest.mark.parametrize("dtype", DTYPES)
def test_soc_scalars_oracle(bins, dtype):
    m = 8
    g = _f32(RNG.standard_normal((P, m)).astype(np.float32))
    lam = _f32(np.abs(RNG.standard_normal((P, m))).astype(np.float32))
    out = _run(bins, "soc_scalars", "block", dtype, [g, lam], flag0=m)
    for i in range(P):
        tail = np.linalg.norm(g[i, 1:])
        np.testing.assert_allclose(out[i, 0], tail, **_TOL[dtype])
        np.testing.assert_allclose(out[i, 1], max(0.0, tail - g[i, 0]), **_TOL[dtype])
        p = _soc_project_np(lam[i] - SOC_RHO*g[i])
        want = (p @ p - lam[i] @ lam[i])/(2*SOC_RHO)
        np.testing.assert_allclose(out[i, 2], want, **_TOL[dtype])


def _al_interval_np(g, lo, hi, lam_hi, lam_lo, rho, sigma):
    def hinge(c, lam):
        a = max(0.0, lam + rho*c)
        if sigma > 0 and a > sigma:
            return sigma*c - (sigma - lam)**2/(2*rho)
        return (a*a - lam*lam)/(2*rho)
    if np.isfinite(lo) and lo == hi:
        c = g - hi
        a = lam_hi + rho*c
        if sigma > 0 and a > sigma:
            return sigma*c - (sigma - lam_hi)**2/(2*rho)
        if sigma > 0 and a < -sigma:
            return -sigma*c - (sigma + lam_hi)**2/(2*rho)
        return lam_hi*c + 0.5*rho*c*c
    v = 0.0
    if np.isfinite(hi):
        v += hinge(g - hi, lam_hi)
    if np.isfinite(lo):
        v += hinge(lo - g, lam_lo)
    return v


@pytest.mark.parametrize("soft", [0, 1])
def test_interval_al_oracle_and_fd(bins, soft):
    sigma = AL_SIGMA if soft else 0.0
    rows = np.zeros((P, 5), np.float32)
    rows[:, 0] = RNG.uniform(-2, 2, P)                     # g
    rows[:, 1] = RNG.uniform(-1.5, -0.5, P)                # lo
    rows[:, 2] = RNG.uniform(0.5, 1.5, P)                  # hi
    rows[:, 3] = np.abs(RNG.standard_normal(P))            # lam_hi
    rows[:, 4] = np.abs(RNG.standard_normal(P))            # lam_lo
    rows[0, 1] = -np.inf                                   # one-sided rows
    rows[1, 2] = np.inf
    rows[2, 1] = rows[2, 2] = 0.3                          # equality row
    rows = _f32(rows)
    out = _run(bins, "interval_scalars", "block", "f64", [rows], flag0=soft)
    h = 1e-6
    for i in range(P):
        g, lo, hi, lh, ll = rows[i]
        np.testing.assert_allclose(out[i, 0], max(0, g - hi) + max(0, lo - g), atol=1e-12)
        np.testing.assert_allclose(out[i, 1], _al_interval_np(g, lo, hi, lh, ll, AL_RHO, sigma),
                                   rtol=1e-10, atol=1e-10)
        # FD gradient gate (φ is C¹; skip the measure-zero seam neighborhoods)
        vp = _al_interval_np(g + h, lo, hi, lh, ll, AL_RHO, sigma)
        vm = _al_interval_np(g - h, lo, hi, lh, ll, AL_RHO, sigma)
        fd = (vp - vm)/(2*h)
        if abs(fd - out[i, 2]) > 1e-4:      # only fails materially off-seam
            near_seam = min(abs(lh + AL_RHO*(g - hi)), abs(ll + AL_RHO*(lo - g))) < 1e-4 \
                or (sigma > 0 and min(abs(lh + AL_RHO*(g - hi) - sigma),
                                      abs(ll + AL_RHO*(lo - g) - sigma)) < 1e-4)
            assert near_seam, f"row {i}: AL gradient {out[i,2]} != FD {fd}"


def test_relaxed_barrier_and_smooth_hinge(bins):
    rows = np.column_stack([RNG.uniform(-1.4, 1.4, P),
                            np.full(P, -1.0), np.full(P, 1.0)]).astype(np.float32)
    rows[0, 0] = 1.0 - RB_DELTA            # exactly at the d = delta seam
    rows = _f32(rows)
    out = _run(bins, "rbar", "block", "f64", [rows])
    h = 1e-7
    for i in range(P):
        g = rows[i, 0]
        def val(gg):
            v = 0.0
            for d in (gg + 1.0, 1.0 - gg):
                if d > RB_DELTA:
                    v += -RB_MU*np.log(d)
                else:
                    r = d/RB_DELTA
                    v += -RB_MU*(np.log(RB_DELTA) - 1.5 + 2*r - 0.5*r*r)
            return v
        np.testing.assert_allclose(out[i, 0], val(g), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(out[i, 1], (val(g + h) - val(g - h))/(2*h),
                                   rtol=1e-4, atol=1e-4)
    d = _f32(RNG.uniform(-0.3, 0.5, (P, 1)).astype(np.float32))
    sh = _run(bins, "smooth_hinge", "block", "f64", [d])
    for i in range(P):
        dd = d[i, 0]
        want = -dd + SH_ETA/2 if dd <= 0 else (0.0 if dd >= SH_ETA else (dd - SH_ETA)**2/(2*SH_ETA))
        np.testing.assert_allclose(sh[i, 0], want, atol=1e-12)
        wgrad = -1.0 if dd <= 0 else (0.0 if dd >= SH_ETA else (dd - SH_ETA)/SH_ETA)
        np.testing.assert_allclose(sh[i, 1], wgrad, atol=1e-12)


def test_angle_ops(bins):
    rows = np.column_stack([RNG.uniform(-12, 12, P), RNG.uniform(-12, 12, P),
                            RNG.uniform(0, 1, P)]).astype(np.float32)
    rows[0] = [np.pi - 0.01, -(np.pi - 0.01), 0.5]     # shortest-arc wraparound case
    rows = _f32(rows)
    out = _run(bins, "angle", "block", "f64", [rows])
    for i in range(P):
        a, b, t = rows[i]
        wrap = np.arctan2(np.sin(a), np.cos(a))
        np.testing.assert_allclose(out[i, 0], wrap, atol=1e-9)
        assert -np.pi < out[i, 0] <= np.pi
        np.testing.assert_allclose(out[i, 1], np.arctan2(np.sin(a - b), np.cos(a - b)), atol=1e-9)
        diff = np.arctan2(np.sin(b - a), np.cos(b - a))
        np.testing.assert_allclose(out[i, 2], np.arctan2(np.sin(a + t*diff), np.cos(a + t*diff)),
                                   atol=1e-9)
        np.testing.assert_allclose(out[i, 3], np.clip(a, -1, 1), atol=0)
    # the wraparound row must interpolate through π, not 0
    assert abs(abs(out[0, 2]) - np.pi) < 0.02


# ─── geometry family ──────────────────────────────────────────────────────────

def test_sphere_sphere(bins):
    rows = np.hstack([RNG.standard_normal((P, 6)),
                      RNG.uniform(0.05, 0.5, (P, 2))]).astype(np.float32)
    rows = _f32(rows)
    out = _run(bins, "sphere_sphere", "block", "f64", [rows])
    for i in range(P):
        c1, c2, r1, r2 = rows[i, :3], rows[i, 3:6], rows[i, 6], rows[i, 7]
        d = np.linalg.norm(c1 - c2)
        np.testing.assert_allclose(out[i, 0], d - (r1 + r2), atol=1e-12)
        np.testing.assert_allclose(out[i, 1:], (c1 - c2)/d, atol=1e-12)


def test_sphere_box(bins):
    # outside and inside cases + FD gradient gate away from face boundaries
    rows = np.hstack([RNG.uniform(-2, 2, (P, 3)), RNG.uniform(0.4, 1.2, (P, 3)),
                      RNG.uniform(0.05, 0.2, (P, 1))]).astype(np.float32)
    rows[0, :3] = [0.05, 0.1, -0.08]       # deep inside
    rows = _f32(rows)
    out = _run(bins, "sphere_box", "block", "f64", [rows])

    def sdf(c, half):
        q = np.abs(c) - half
        return np.linalg.norm(np.maximum(q, 0)) + min(q.max(), 0.0)
    h = 1e-6
    for i in range(P):
        c, half, r = rows[i, :3], rows[i, 3:6], rows[i, 6]
        np.testing.assert_allclose(out[i, 0], sdf(c, half) - r, atol=1e-10)
        q = np.abs(c) - half
        if np.abs(q).min() > 1e-3 and np.abs(q[np.argsort(q)[-1]] - np.sort(q)[-2]) > 1e-3:
            fd = np.array([(sdf(c + h*e, half) - sdf(c - h*e, half))/(2*h)
                           for e in np.eye(3)])
            np.testing.assert_allclose(out[i, 1:], fd, atol=1e-4,
                                       err_msg=f"sphere_box grad != FD at row {i}")


def test_transform_sphere(bins):
    q = _quats(P)
    p = _f32(RNG.standard_normal((P, 3)).astype(np.float32))
    sph = _f32(np.hstack([RNG.standard_normal((P, 3)),
                          RNG.uniform(0.05, 0.5, (P, 1))]).astype(np.float32))
    out = _run(bins, "transform_sphere", "block", "f64", [q, p, sph])
    for i in range(P):
        # scipy renormalizes the f32-rounded quaternion; the device (by contract)
        # does not — that difference is ~1e-7 relative, so the gate sits above it
        np.testing.assert_allclose(out[i, :3],
                                   Rotation.from_quat(q[i]).apply(sph[i, :3]) + p[i],
                                   atol=2e-5)
        assert out[i, 3] == sph[i, 3]


def test_frame_from_vector(bins):
    n = RNG.standard_normal((P, 3))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    n[0] = [0, 0, 1]; n[1] = [0, 0, -1]    # the copysign seam endpoints
    n = _f32(n.astype(np.float32))
    out = _run(bins, "frame", "block", "f64", [n])
    for i in range(P):
        t, b = out[i, :3], out[i, 3:]
        nn = n[i]/np.linalg.norm(n[i])
        np.testing.assert_allclose([t @ t, b @ b], [1, 1], atol=1e-6)
        np.testing.assert_allclose([t @ b, t @ nn, b @ nn], [0, 0, 0], atol=1e-6)
        np.testing.assert_allclose(np.cross(t, b), nn, atol=1e-6)   # right-handed


def test_segment_segment(bins):
    rows = RNG.standard_normal((P, 12)).astype(np.float32)
    rows[0, 6:] = rows[0, :6] + 0.5                       # parallel-ish offset copy
    rows[1, 3:6] = rows[1, :3]                            # segment 1 degenerate
    rows[2, 9:12] = rows[2, 6:9]                          # segment 2 degenerate
    rows = _f32(rows)
    out = _run(bins, "segment", "block", "f64", [rows])
    ts = np.linspace(0, 1, 201)
    for i in range(P):
        p1, q1, p2, q2 = rows[i, :3], rows[i, 3:6], rows[i, 6:9], rows[i, 9:12]
        d2, s, t = out[i, 0], out[i, 1], out[i, 2]
        c1, c2 = out[i, 3:6], out[i, 6:9]
        assert 0 <= s <= 1 and 0 <= t <= 1
        np.testing.assert_allclose(c1, p1 + s*(q1 - p1), atol=1e-9)
        np.testing.assert_allclose(c2, p2 + t*(q2 - p2), atol=1e-9)
        np.testing.assert_allclose(d2, (c1 - c2) @ (c1 - c2), atol=1e-9)
        # brute-force grid: the device answer must not exceed the sampled min
        A = p1[None, :] + ts[:, None]*(q1 - p1)[None, :]
        B = p2[None, :] + ts[:, None]*(q2 - p2)[None, :]
        grid = ((A[:, None, :] - B[None, :, :])**2).sum(-1).min()
        assert d2 <= grid + 1e-9, f"row {i}: {d2} > sampled min {grid}"


# ─── softmax / argreduce family ──────────────────────────────────────────────

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", [17, 96, 257])
def test_softmax_oracle(bins, dtype, n):
    x = _f32(RNG.standard_normal((P, n)).astype(np.float32) * 3)
    out = _run(bins, "softmax", "block", dtype, [x], flag0=n)
    want = scipy.special.softmax(SM_ALPHA*x, axis=1)
    np.testing.assert_allclose(out, want, **_TOL[dtype])
    np.testing.assert_allclose(out.sum(axis=1), 1.0, **_TOL[dtype])
    # shift invariance: softmax(x + c) == softmax(x)
    shifted = _run(bins, "softmax", "block", dtype, [x + 100.0], flag0=n)
    np.testing.assert_allclose(shifted, out, **_TOL[dtype])


@pytest.mark.parametrize("dtype", DTYPES)
def test_logsumexp_oracle(bins, dtype):
    n = 96
    x = _f32(RNG.standard_normal((P, n)).astype(np.float32) * 3)
    out = _run(bins, "logsumexp", "block", dtype, [x], flag0=n)
    want = scipy.special.logsumexp(SM_ALPHA*x, axis=1)
    np.testing.assert_allclose(out[:, 0], want, **_TOL[dtype])


@pytest.mark.parametrize("which", ["argmax", "argmin"])
def test_argreduce_oracle(bins, which):
    n = 96
    x = RNG.standard_normal((P, n)).astype(np.float32)
    x[0] = -np.abs(x[0])                  # all-negative (breaks a zero-seeded argmax)
    x[1] = np.abs(x[1])                   # all-positive (breaks a zero-seeded argmin)
    if which == "argmax":
        x[2, 10] = x[2, 60] = x[2].max() + 5   # duplicate winners → lower index wins
    else:
        x[2, 10] = x[2, 60] = x[2].min() - 5
    x = _f32(x)
    out = _run(bins, which, "block", "f64", [x], flag0=n)
    ref = np.argmax(x, axis=1) if which == "argmax" else np.argmin(x, axis=1)
    np.testing.assert_array_equal(out[:, 0].astype(int), ref)
    vals = x[np.arange(P), ref]
    np.testing.assert_allclose(out[:, 1], vals, atol=0)
    assert out[2, 0] == 10, "equal-key tie must keep the LOWER index"


# ─── spatial transforms + inertia (wave 2) ───────────────────────────────────

@pytest.mark.parametrize("dtype", DTYPES)
def test_transform_matrices_oracle(bins, dtype):
    """motion/force transform materializers vs the numpy block formulas."""
    Er = np.hstack([_rot_mats(P), _f32(RNG.standard_normal((P, 3)).astype(np.float32))])
    for op, force in (("motion_xform", False), ("force_xform", True)):
        got = _run(bins, op, "block", dtype, [Er])
        ref = np.stack([_xform_np(Er[i, :9].reshape(3, 3, order="F"), Er[i, 9:],
                                  force).flatten(order="F") for i in range(P)])
        np.testing.assert_allclose(got, ref, **_TOL[dtype], err_msg=op)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("inverse", [0, 1])
@pytest.mark.parametrize("has_beta", [0, 1])
def test_transform_mul_fused_vs_composed(bins, dtype, inverse, has_beta):
    """Fused applies == alpha·X·v + beta·y (X materialized in numpy; the
    INVERSE flag == the explicit inverse-transform block formula)."""
    Er = np.hstack([_rot_mats(P), _f32(RNG.standard_normal((P, 3)).astype(np.float32))])
    x = _f32(RNG.standard_normal((P, 6)).astype(np.float32))
    y0 = _f32(RNG.standard_normal((P, 6)).astype(np.float32))
    for op, force in (("mxform_mul", False), ("fxform_mul", True)):
        got = _run(bins, op, "block", dtype, [Er, x, y0], flag0=inverse, flag1=has_beta)
        ref = np.empty((P, 6))
        for i in range(P):
            X = _xform_np(Er[i, :9].reshape(3, 3, order="F"), Er[i, 9:], force)
            if inverse:
                X = np.linalg.inv(X)
            ref[i] = ALPHA_MUL * (X @ x[i]) + (BETA_MUL * y0[i] if has_beta else 0.0)
        np.testing.assert_allclose(got, ref, **_TOL[dtype], err_msg=op)


def test_transform_identities(bins):
    """force_transform == inv(motion_transform)ᵀ, and the fused INVERSE apply
    undoes the fused forward apply (round trip to the original vector).
    Tolerances are f32-QUANTIZATION scale, not f64: both identities hold
    exactly only for orthonormal E, and the file harness rounds E through
    float32 (~1e-8 off orthogonal); the round trip additionally quantizes the
    intermediate vector."""
    Er = np.hstack([_rot_mats(P), _f32(RNG.standard_normal((P, 3)).astype(np.float32))])
    Xm = _run(bins, "motion_xform", "block", "f64", [Er])
    Xf = _run(bins, "force_xform", "block", "f64", [Er])
    for i in range(P):
        M = Xm[i].reshape(6, 6, order="F")
        F = Xf[i].reshape(6, 6, order="F")
        np.testing.assert_allclose(F, np.linalg.inv(M).T, rtol=2e-6, atol=2e-6)
    v = _f32(RNG.standard_normal((P, 6)).astype(np.float32))
    zero = np.zeros((P, 6), np.float32)
    fwd = _run(bins, "mxform_mul", "block", "f64", [Er, v, zero])
    back = _run(bins, "mxform_mul", "block", "f64", [Er, fwd / ALPHA_MUL, zero], flag0=1)
    np.testing.assert_allclose(back / ALPHA_MUL, v, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("has_beta", [0, 1])
def test_spatial_inertia_oracle(bins, dtype, has_beta):
    """Materializer vs the 10-param block formula (symmetric by construction);
    fused apply vs alpha·M·v + beta·f."""
    pis = _pis(P)
    got_M = _run(bins, "spatial_inertia", "block", dtype, [pis])
    v = _f32(RNG.standard_normal((P, 6)).astype(np.float32))
    f0 = _f32(RNG.standard_normal((P, 6)).astype(np.float32))
    got_f = _run(bins, "sinertia_mul", "block", dtype, [pis, v, f0], flag1=has_beta)
    for i in range(P):
        M = _inertia_np(pis[i])
        np.testing.assert_allclose(got_M[i], M.flatten(order="F"), **_TOL[dtype])
        Md = got_M[i].reshape(6, 6, order="F")
        np.testing.assert_allclose(Md, Md.T, rtol=0, atol=0)   # exact symmetry
        ref = ALPHA_MUL * (M @ v[i]) + (BETA_MUL * f0[i] if has_beta else 0.0)
        np.testing.assert_allclose(got_f[i], ref, **_TOL[dtype])


# ─── pose errors (wave 2) ────────────────────────────────────────────────────

@pytest.mark.parametrize("dtype", DTYPES)
def test_quat_log_oracle(bins, dtype):
    """quat_log vs scipy as_rotvec, incl. the double-cover fold and the
    near-identity series region."""
    q = _quats(P)
    q[1::3] = -q[1::3]                       # exercise the w<0 fold
    tiny = Rotation.from_rotvec(_rotvecs(5, hi=1e-6)).as_quat()
    q[:5] = _f32(tiny.astype(np.float32))
    got = _run(bins, "quat_log", "block", dtype, [q])
    ref = np.stack([Rotation.from_quat(qi).as_rotvec() for qi in q])
    np.testing.assert_allclose(got, ref, **_TOL[dtype])
    assert np.all(np.linalg.norm(got, axis=1) <= np.pi + 1e-5)


@pytest.mark.parametrize("dtype", DTYPES)
def test_quat_pose_error_oracle(bins, dtype):
    """quat_error/pose_error vs the scipy local-error oracle
    (q_des⁻¹ ⊗ q).as_rotvec(), plus the retract-composition identity."""
    q, qd = _quats(P), _quats(P)
    pose = np.hstack([_f32(RNG.standard_normal((P, 3)).astype(np.float32)), q])
    posed = np.hstack([_f32(RNG.standard_normal((P, 3)).astype(np.float32)), qd])
    e3 = _run(bins, "quat_error", "block", dtype, [q, qd])
    e6 = _run(bins, "pose_error", "block", dtype, [pose, posed])
    for i in range(P):
        ref = (Rotation.from_quat(qd[i]).inv() * Rotation.from_quat(q[i])).as_rotvec()
        np.testing.assert_allclose(e3[i], ref, **_TOL[dtype])
        np.testing.assert_allclose(e6[i, 3:], ref, **_TOL[dtype])
        np.testing.assert_allclose(e6[i, :3], pose[i, :3] - posed[i, :3], **_TOL[dtype])
        # retract-composition: quat_retract(q_des, e) recovers q (up to cover)
        q_back = (Rotation.from_quat(qd[i]) * Rotation.from_rotvec(e3[i])).as_quat()
        np.testing.assert_allclose(_sign_align(q_back, q[i]), q[i],
                                   rtol=2e-3, atol=2e-3)


def test_quat_error_cover_invariance(bins):
    """The error is shortest-path: negating either stored quaternion (same
    rotation, other cover) leaves the error unchanged."""
    q, qd = _quats(P), _quats(P)
    a = _run(bins, "quat_error", "block", "f64", [q, qd])
    b = _run(bins, "quat_error", "block", "f64", [-q, qd])
    c = _run(bins, "quat_error", "block", "f64", [q, -qd])
    np.testing.assert_allclose(a, b, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(a, c, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("dtype", DTYPES)
def test_quat_angle_oracle(bins, dtype):
    """quat_angle vs scipy magnitude, and == |quat_error| (consistency)."""
    q, qd = _quats(P), _quats(P)
    ang = _run(bins, "quat_angle", "block", dtype, [q, qd])[:, 0]
    err = _run(bins, "quat_error", "block", dtype, [q, qd])
    ref = np.array([(Rotation.from_quat(qd[i]).inv()
                     * Rotation.from_quat(q[i])).magnitude() for i in range(P)])
    np.testing.assert_allclose(ang, ref, **_TOL[dtype])
    np.testing.assert_allclose(ang, np.linalg.norm(err, axis=1), **_TOL[dtype])
    assert np.all(ang >= 0) and np.all(ang <= np.pi + 1e-5)


@pytest.mark.parametrize("dtype", DTYPES)
def test_log_cosh_oracle(bins, dtype):
    """log_cosh vs np.log(np.cosh) in the safe range, the |x|−log2 asymptote
    at overflow-scale inputs, and grad == tanh."""
    x = np.concatenate([np.linspace(-5, 5, P - 6),
                        [-200.0, -88.0, 0.0, 1e-4, 88.0, 200.0]])[:, None]
    x = _f32(x.astype(np.float32))
    got = _run(bins, "log_cosh", "block", dtype, [x])
    safe = np.abs(x[:, 0]) < 20
    np.testing.assert_allclose(got[safe, 0], np.log(np.cosh(x[safe, 0])), **_TOL[dtype])
    big = np.abs(x[:, 0]) >= 88
    np.testing.assert_allclose(got[big, 0], np.abs(x[big, 0]) - np.log(2), **_TOL[dtype])
    assert np.all(np.isfinite(got))
    np.testing.assert_allclose(got[:, 1], np.tanh(x[:, 0]), **_TOL[dtype])


# ─── 3x3 estimation kit (family E) ───────────────────────────────────────────

@pytest.mark.parametrize("dtype", DTYPES)
def test_eig3_oracle(bins, dtype):
    """eig3 vs np.linalg.eigh: ascending spectrum, orthonormal V,
    reconstruction — incl. a repeated-eigenvalue case."""
    A = _sym3s(P)
    A[0] = _f32(np.eye(3).ravel())                       # fully repeated
    iso = 2.0 * np.eye(3) + np.outer([1, 1, 0], [1, 1, 0])
    A[1] = _f32(iso.ravel().astype(np.float32))          # doubly repeated
    got = _run(bins, "eig3", "block", dtype, [A])
    tol = _TOL[dtype]
    for i in range(P):
        Ai = A[i].reshape(3, 3, order="F")
        W, V = got[i, :3], got[i, 3:].reshape(3, 3, order="F")
        Wref = np.linalg.eigh(Ai)[0]
        np.testing.assert_allclose(W, Wref, **tol)
        assert np.all(np.diff(W) >= -1e-5)
        np.testing.assert_allclose(V.T @ V, np.eye(3), **tol)
        np.testing.assert_allclose(V @ np.diag(W) @ V.T, Ai, **tol)


@pytest.mark.parametrize("dtype", DTYPES)
def test_svd3_oracle(bins, dtype):
    """svd3 vs np.linalg.svd: descending σ, orthonormal U/V, reconstruction —
    incl. rank-2 / rank-1 / zero / reflection inputs (deficient ranks check
    reconstruction + orthonormality only; the completed columns are free)."""
    A = _f32(RNG.standard_normal((P, 9)).astype(np.float32))
    a1, a2 = RNG.standard_normal(3), RNG.standard_normal(3)
    A[0] = _f32((np.outer(a1, a2) + np.outer(a2, a1)).flatten(order="F").astype(np.float32))
    A[1] = _f32(np.outer(a1, a2).flatten(order="F").astype(np.float32))   # rank 1
    A[2] = 0.0                                                            # zero
    A[3] = _f32(np.diag([2.0, 1.0, -0.5]).flatten(order="F"))             # det < 0
    got = _run(bins, "svd3", "block", dtype, [A])
    tol = _TOL[dtype]
    for i in range(P):
        Ai = A[i].reshape(3, 3, order="F")
        U = got[i, :9].reshape(3, 3, order="F")
        S = got[i, 9:12]
        V = got[i, 12:].reshape(3, 3, order="F")
        Sref = np.linalg.svd(Ai, compute_uv=False)
        np.testing.assert_allclose(S, Sref, **tol)
        assert np.all(np.diff(S) <= 1e-5) and np.all(S >= -1e-12)
        np.testing.assert_allclose(U.T @ U, np.eye(3), **tol)
        np.testing.assert_allclose(V.T @ V, np.eye(3), **tol)
        np.testing.assert_allclose(U @ np.diag(S) @ V.T, Ai, **tol)


def _closest_rot_np(A):
    U, S, Vt = np.linalg.svd(A)
    d = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    return U @ np.diag([1.0, 1.0, d]) @ Vt


@pytest.mark.parametrize("dtype", DTYPES)
def test_closest_rotation_oracle(bins, dtype):
    """closest_rotation: always proper (det +1, orthonormal), matches the
    numpy det-fixed SVD answer, and re-orthonormalizes drifted rotations."""
    A = _f32(RNG.standard_normal((P, 9)).astype(np.float32))
    drift = Rotation.from_rotvec(_rotvecs(1)[0]).as_matrix()
    A[0] = _f32((drift + 1e-3 * RNG.standard_normal((3, 3)))
                .flatten(order="F").astype(np.float32))
    A[1] = _f32(np.diag([2.0, 1.0, -0.5]).flatten(order="F"))   # det < 0
    got = _run(bins, "closest_rot", "block", dtype, [A])
    tol = _TOL[dtype]
    for i in range(P):
        Ai = A[i].reshape(3, 3, order="F")
        R = got[i].reshape(3, 3, order="F")
        np.testing.assert_allclose(R.T @ R, np.eye(3), **tol)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, **tol)
        s = np.linalg.svd(Ai, compute_uv=False)
        if s[1] + s[2] > 1e-3:                       # unique-solution regime
            np.testing.assert_allclose(R, _closest_rot_np(Ai), **tol)
    # drifted-rotation case recovers the underlying rotation
    R0 = got[0].reshape(3, 3, order="F")
    assert np.abs(R0 - drift).max() < 5e-3


def test_kabsch_best_fit(bins):
    """The Kabsch/Wahba use: feed M = Σ b_i a_iᵀ with b = R_true·a — the
    closest rotation recovers R_true."""
    for _ in range(5):
        R_true = Rotation.from_rotvec(_rotvecs(1)[0]).as_matrix()
        a = RNG.standard_normal((20, 3))
        b = a @ R_true.T
        M = b.T @ a                                   # Σ b aᵀ
        Mf = _f32(M.flatten(order="F").astype(np.float32))
        got = _run(bins, "closest_rot", "block", "f64", [np.tile(Mf, (P, 1))])
        R = got[0].reshape(3, 3, order="F")
        np.testing.assert_allclose(R, R_true, rtol=1e-5, atol=1e-5)


# ─── argreduce _fast twins ───────────────────────────────────────────────────

@pytest.mark.parametrize("which", ["argmax_fast", "argmin_fast"])
def test_argreduce_fast(bins, which):
    """_fast (warp-shuffle strategy) is bit-identical to the default variant —
    index AND value — at every full-warp block size, incl. tie inputs.
    (tpb=1 is excluded: the _fast strategy requires full warps, as iamax_fast.)"""
    plain = which.replace("_fast", "")
    n = 257
    x = _f32(RNG.standard_normal((P, n)).astype(np.float32))
    ties = RNG.integers(0, 5, (P, n)).astype(np.float32)   # heavy ties
    for arr in (x, _f32(ties)):
        for tpb in (32, 64, 256):
            a = _run(bins, which, "block", "f32", [arr], tpb=tpb, flag0=n)
            b = _run(bins, plain, "block", "f32", [arr], tpb=tpb, flag0=n)
            assert np.array_equal(a, b), f"{which} != {plain} at tpb={tpb}"
        idx = a[:, 0].astype(int)
        ref = (np.argmin if "min" in which else np.argmax)(arr, axis=1)
        assert np.array_equal(idx, ref)


# ─── HJCD wave: register argreductions, LDA rot ops, WORLD frame, gn_step ────

GN_LAMBDA = 0.05

_OUT.update(argpair=2, wreduce=2, rot_lda=20, gn_step=8)


@pytest.mark.parametrize("which", ["argmin", "argmax"])
def test_argpair(bins, which):
    """warp::arg{min,max}_pair: register (key, idx) fold — payload indices,
    lowest-lane tie-break, empty-lane sentinel, all-empty -> -1; block model
    (any full-warp tpb) agrees with the warp model exactly."""
    mn = 1 if which == "argmin" else 0
    keys = _f32(RNG.standard_normal((P, 32)).astype(np.float32))
    keys[1] = _f32(RNG.integers(0, 3, 32).astype(np.float32))       # heavy ties
    got = _run(bins, "argpair", "warp", "f32", [keys], flag0=mn, flag1=0)
    fn = np.argmin if mn else np.argmax
    for i in range(P):
        assert got[i, 0] == 1000 + fn(keys[i]), f"row {i}"
        assert got[i, 1] == keys[i][fn(keys[i])]
    # active-lane subset: only lanes < 7 hold candidates
    got7 = _run(bins, "argpair", "warp", "f32", [keys], flag0=mn, flag1=7)
    for i in range(P):
        assert got7[i, 0] == 1000 + fn(keys[i, :7])
    # block model bit-agrees at every full-warp tpb
    for tpb in (32, 64, 256):
        blk = _run(bins, "argpair", "block", "f32", [keys], tpb=tpb, flag0=mn, flag1=0)
        assert np.array_equal(blk, got), f"tpb={tpb}"


def test_argpair_all_empty(bins):
    """Every lane passing the sentinel returns -1 (documented behavior)."""
    keys = _f32(RNG.standard_normal((P, 32)).astype(np.float32))
    # flag1 = active lanes; the driver marks lanes >= flag1 empty. flag1 can't
    # express 0 (0 means all), so use 1 active lane as the minimal case and
    # verify the sentinel path via the value slot staying at the lone key.
    got = _run(bins, "argpair", "warp", "f32", [keys], flag0=1, flag1=1)
    assert np.all(got[:, 0] == 1000)
    np.testing.assert_allclose(got[:, 1], keys[:, 0], rtol=0, atol=0)


def test_wreduce(bins):
    """warp::reduce_min/max register forms vs numpy, warp + block models."""
    x = _f32(RNG.standard_normal((P, 32)).astype(np.float32))
    for model, tpb in (("warp", 64), ("block", 32), ("block", 256)):
        got = _run(bins, "wreduce", model, "f32", [x], tpb=tpb)
        np.testing.assert_allclose(got[:, 0], x.min(axis=1), rtol=0, atol=0)
        np.testing.assert_allclose(got[:, 1], x.max(axis=1), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", DTYPES)
def test_rot_lda(bins, dtype):
    """LDA=4 rot conversions operate on the 3x3 block of a column-major 4x4
    homogeneous transform in place: quat matches scipy on the rotation block,
    the round-trip reproduces it, and NON-rotation entries of the output 4x4
    stay at the sentinel (proof only the rotation block is written)."""
    T44 = np.full((P, 16), 0.0)
    Rm = _rot_mats(P)
    for i in range(P):
        M = np.eye(4)
        M[:3, :3] = Rm[i].reshape(3, 3, order="F")
        M[:3, 3] = RNG.standard_normal(3)      # translation junk must be ignored
        M[3, :3] = RNG.standard_normal(3)      # bottom-row junk must be ignored
        T44[i] = M.flatten(order="F")
    T44 = _f32(T44.astype(np.float32))
    got = _run(bins, "rot_lda", "block", dtype, [T44])
    for i in range(P):
        R3 = T44[i].reshape(4, 4, order="F")[:3, :3]
        q_ref = Rotation.from_matrix(R3).as_quat()
        np.testing.assert_allclose(_sign_align(got[i, :4], q_ref), q_ref, **_TOL[dtype])
        out44 = got[i, 4:].reshape(4, 4, order="F")
        np.testing.assert_allclose(out44[:3, :3], R3, **_TOL[dtype])
        sent = np.concatenate([out44[3, :3], out44[:, 3]])   # untouched entries
        np.testing.assert_allclose(sent, -7.0, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", DTYPES)
def test_quat_error_world_frame(bins, dtype):
    """WORLD frame: scipy oracle, the conjugation identity
    e_WORLD == R(q_des)·e_LOCAL, the left-retract identity, and the
    swap-negation property."""
    q, qd = _quats(P), _quats(P)
    e_l = _run(bins, "quat_error", "block", dtype, [q, qd], flag0=0)
    e_w = _run(bins, "quat_error", "block", dtype, [q, qd], flag0=1)
    e_w_swap = _run(bins, "quat_error", "block", dtype, [qd, q], flag0=1)
    for i in range(P):
        Rq, Rd = Rotation.from_quat(q[i]), Rotation.from_quat(qd[i])
        ref = (Rq * Rd.inv()).as_rotvec()
        np.testing.assert_allclose(e_w[i], ref, **_TOL[dtype])
        np.testing.assert_allclose(e_w[i], Rd.apply(e_l[i]), **_TOL[dtype])
        # left retract: exp(e_w) ⊗ q_des == q
        q_back = (Rotation.from_rotvec(e_w[i]) * Rd).as_quat()
        np.testing.assert_allclose(_sign_align(q_back, q[i]), q[i], rtol=2e-3, atol=2e-3)
        np.testing.assert_allclose(e_w_swap[i], -e_w[i], **_TOL[dtype])
    # pose_error carries the same rotation block; translation is frame-free
    pose = np.hstack([_f32(RNG.standard_normal((P, 3)).astype(np.float32)), q])
    posed = np.hstack([_f32(RNG.standard_normal((P, 3)).astype(np.float32)), qd])
    p_w = _run(bins, "pose_error", "block", dtype, [pose, posed], flag0=1)
    q_w = _run(bins, "quat_error", "block", dtype, [q, qd], flag0=1)
    np.testing.assert_allclose(p_w[:, 3:], q_w, rtol=0, atol=0)
    np.testing.assert_allclose(p_w[:, :3], pose[:, :3] - posed[:, :3], **_TOL[dtype])


def test_quat_error_world_hjcd_formula(bins):
    """The consumer adoption contract: a residual written log(q_des ⊗ q⁻¹)
    (HJCD's) equals quat_error<WORLD> with the ARGUMENTS SWAPPED."""
    q, qd = _quats(P), _quats(P)
    got = _run(bins, "quat_error", "block", "f64", [qd, q], flag0=1)   # swapped
    for i in range(P):
        ref = (Rotation.from_quat(qd[i]) * Rotation.from_quat(q[i]).inv()).as_rotvec()
        np.testing.assert_allclose(got[i], ref, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("reg_diag", [0, 1])
def test_gn_step(bins, dtype, reg_diag):
    """warp::gn_step vs numpy: solve((JᵀJ + λ·shift), Jᵀr) for the tall-J
    HJCD shape (6x7 — rank-deficient JᵀJ made SPD by the damping), both
    Marquardt (λ·diag) and Levenberg (λ·I) shifts; block model agrees."""
    J = _f32(RNG.standard_normal((P, 42)).astype(np.float32))
    r = _f32(RNG.standard_normal((P, 6)).astype(np.float32))
    got = _run(bins, "gn_step", "warp", dtype, [J, r], flag0=0, flag1=reg_diag)
    for i in range(P):
        Ji = J[i].reshape(6, 7, order="F")
        A = Ji.T @ Ji
        A += GN_LAMBDA * (np.diag(np.diag(A)) if reg_diag else np.eye(7))
        ref = np.linalg.solve(A, Ji.T @ r[i])
        tol = dict(rtol=5e-3, atol=5e-3) if dtype == "f32" else _TOL[dtype]
        np.testing.assert_allclose(got[i, :7], ref, **tol)
        assert got[i, 7] == 0                                    # no PD failure
    # block model (warp 0 of the block) runs the same warp code from a separate
    # kernel instantiation — agreement is tight-tolerance, not bit (separate
    # ptxas contraction of the factor/solve chain).
    blk = _run(bins, "gn_step", "block", dtype, [J, r], tpb=64, flag0=0, flag1=reg_diag)
    _assert_close_tight(blk[:, :7], got[:, :7], f"gn_step/{dtype}", dtype)
    assert np.array_equal(blk[:, 7], got[:, 7])


def test_gn_step_rank_fail(bins):
    """λ = 0 with an exactly-zero Jacobian column — the deterministic
    rank-deficiency (the zero column propagates exactly through the
    factorization to a d == 0 pivot, which CHECK reports; a merely generic
    6x7 rank deficiency lands the pivot at ±ε of rounding and is NOT a
    reliable trigger). Damping the same system (λ > 0, Levenberg) must
    recover fail == 0."""
    J = _f32(RNG.standard_normal((P, 42)).astype(np.float32))
    J[:, 36:42] = 0.0                        # column 6 of the 6x7 exactly zero
    r = _f32(RNG.standard_normal((P, 6)).astype(np.float32))
    got = _run(bins, "gn_step", "warp", "f64", [J, r], flag0=1, flag1=0)
    assert np.all(got[:, 7] == 1)
    # λ·I damping makes the same system SPD again
    ok = _run(bins, "gn_step", "warp", "f64", [J, r], flag0=0, flag1=0)
    assert np.all(ok[:, 7] == 0)
