"""Optimized inlined restricted callers, with external oracles and bounded runs.

Each subprocess owns one case: a divergent barrier cannot prevent later cases
from being collected. The default trailing barrier is exercised by a different
thread consuming the output immediately on return. Runtime overloads receive a
kernel argument rather than a constant disguised as a runtime dimension.
"""
import os
import shlex
import subprocess

import numpy as np
import pytest
from scipy.special import logsumexp, softmax

OPS = ["trsv_ct", "trsv_rt", "trsm", "potrf", "potrs_1", "posv_1",
       "potrs_2", "posv_2", "inv", "gesv", "ldlt_solve", "reduce", "nrm2",
       "softmax", "cgrps_potrf", "cgrps_trsm", "warp_trsv", "warp_potrf",
       "logsumexp", "vector_norm", "prefix_inclusive", "cgrps_reduce",
       "potrf_ct", "inv_pivoted", "ldlt_pivot_solve", "trsm_1",
       "potrs_multi1", "posv_multi1", "bare_softmax", "warp_softmax", "syev"] + [f"trsv_flags_{i}" for i in range(8)] + ["trsv_no_tail", "softmax_no_tail"]


def check_case(binary, op, bits, n, seed, threads, global_memory=False, multidim=False):
    command = shlex.split(os.environ.get("GLASS_RUN_PREFIX", "")) + [
        str(binary), str(op), str(bits), str(n), str(seed), str(threads),
        str(int(global_memory)), str(int(multidim))]
    result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, (command, result.stdout, result.stderr)
    lines = [line for line in result.stdout.splitlines()
             if not line.startswith("=========")]
    out = np.fromstring(" ".join(lines), sep=" ")
    assert out.size == 2*n*n + 2*n, (command, result.stdout)
    assert np.isfinite(out).all(), command
    dtype = np.float32 if bits == 32 else np.float64
    tol = 3e-5 if bits == 32 else 2e-11
    L = np.zeros((n, n))
    for j in range(n):
        for i in range(j, n):
            L[i, j] = 2 + i % 3 if i == j else ((i*7+j*3+seed) % 9-4)*0.13
    A = L @ L.T
    if seed >= 10 and op in (24, 30):
        A = np.zeros((n, n))
        A[0, 1] = A[1, 0] = 2
        for i in range(2, n):
            A[i, i] = (-1 if i % 2 else 1) * (i + 1)
    if seed >= 10 and op == 23:
        A[[0, 1]] = A[[1, 0]]
    x = np.array([((i*3+seed) % 11-5)*0.17 for i in range(2*n)])
    if seed >= 10 and op in (13, 18, 28, 29, 40):
        x[:n] += 1000
    matrix = out[:n*n].reshape(n, n, order="F")
    right = out[n*n:2*n*n].reshape(n, n, order="F")
    vector = out[2*n*n:]
    def close(actual, expected):
        np.testing.assert_allclose(actual, expected, atol=tol, rtol=tol,
                                   err_msg=str(command))
    if op in (3, 14, 17, 22):
        close(np.tril(matrix), np.linalg.cholesky(A.astype(dtype)))
    elif op in (8, 23):
        close(right, np.linalg.inv(A.astype(dtype)))
        close(A @ right, np.eye(n))
        close(right @ A, np.eye(n))
    elif op in (11, 21):
        close(vector[0], np.sum(x[:n].astype(dtype), dtype=np.float64))
    elif op == 12:
        close(vector[0], np.linalg.norm(x[:n].astype(dtype)))
    elif op in (13, 28, 29, 40):
        close(out[:n], softmax(x[:n].astype(dtype)))
        close(np.sum(out[:n]), 1)
        assert (out[:n] >= 0).all()
    elif op == 18:
        close(out[0], logsumexp(x[:n].astype(dtype)))
    elif op == 19:
        close(out[0], np.linalg.norm(x[:n].astype(dtype)))
    elif op == 20:
        close(out[:n], np.cumsum(x[:n].astype(dtype), dtype=np.float64))
    elif op == 30:
        close(vector[:n], np.linalg.eigvalsh(A.astype(dtype)))
        close(right.T @ right, np.eye(n))
        close((right * vector[:n]) @ right.T, A)
    else:
        cols = 2 if op in (2, 6, 7, 9, 15) else 1
        close(vector[:n*cols], x[:n*cols])
        system = L if op in (0, 1, 2, 15, 16, 25, 39) else A
        if 31 <= op <= 38:
            flags = op - 31
            system = (L.T if flags & 1 else L).copy()
            if flags & 4:
                np.fill_diagonal(system, 1)
            if flags & 2:
                system = system.T
        for col in range(cols):
            b = system @ x[col*n:(col+1)*n]
            residual = np.linalg.norm(system @ vector[col*n:(col+1)*n] - b)
            assert residual <= tol * (1 + np.linalg.norm(b)), command


@pytest.mark.parametrize("op", range(len(OPS)), ids=OPS)
@pytest.mark.parametrize("bits", [32, 64])
@pytest.mark.parametrize("n", [3, 7, 17])
def test_restricted_callers(bins, op, bits, n):
    counts = [32] if op in (16, 17, 29) else [1, 2, 7, 31, 32, 33, 64]
    for threads in counts:
        if op == 20 and threads < n:
            continue  # prefix's documented launch contract
        check_case(bins["restrict"], op, bits, n, 0, threads, global_memory=2)


@pytest.mark.parametrize("op", [0, 1, 4, 5, 13, 18, 23, 24, 28, 30], ids=lambda i: OPS[i])
@pytest.mark.parametrize("bits", [32, 64])
@pytest.mark.parametrize("global_memory,multidim", [(True, False), (False, True), (True, True)])
def test_restricted_memory_and_geometry(bins, op, bits, global_memory, multidim):
    for threads in (2, 33):
        check_case(bins["restrict"], op, bits, 7, 2, threads, global_memory, multidim)


@pytest.mark.parametrize("op", [13, 18, 23, 24, 30, 40], ids=lambda i: OPS[i])
@pytest.mark.parametrize("bits", [32, 64])
def test_restricted_pivots_and_shift(bins, op, bits):
    for threads in (1, 2, 7, 33, 64):
        check_case(bins["restrict"], op, bits, 7, 10, threads, global_memory=2)
