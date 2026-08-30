"""Oracle tests for ``glass::nvidia::thread`` cuSOLVERDx wrappers.

The CUDA runner launches seven independent problems in one block.  This file
reconstructs every deterministic input and validates every public operation in
both supported precisions and at two sizes.
"""

import subprocess

import numpy as np
import pytest


BATCHES = 7
OPS = ("potrf", "trsm", "posv", "potrs", "getrf", "getrs", "gesv", "geqrf", "gels")


def _spd(n, dtype):
    out = []
    for b in range(BATCHES):
        r = np.empty((n, n), dtype=dtype)
        for i in range(n):
            for k in range(n):
                r[i, k] = dtype(0.03) * dtype(1 + ((i + 2 * k + b) % 5))
        a = r @ r.T
        a[np.diag_indices(n)] += dtype(n) + dtype(0.2) * dtype(b)
        out.append(a)
    return np.stack(out)


def _general(n, dtype):
    out = np.empty((BATCHES, n, n), dtype=dtype)
    for b in range(BATCHES):
        for i in range(n):
            for j in range(n):
                v = dtype(0.02) * dtype((i + 2 * j + 3 * b) % 7)
                if i == j:
                    v += dtype(n) + dtype(0.25) * dtype(b)
                out[b, i, j] = v
    return out


def _lower(n, dtype):
    out = np.zeros((BATCHES, n, n), dtype=dtype)
    for b in range(BATCHES):
        for i in range(n):
            for j in range(n):
                if i == j:
                    out[b, i, j] = dtype(1.5) + dtype(0.1) * dtype(i + b)
                elif i > j:
                    out[b, i, j] = dtype(0.03) * dtype(1 + ((i + j + b) % 5))
    return out


def _rhs(rows, dtype):
    out = np.empty((BATCHES, rows, 2), dtype=dtype)
    for b in range(BATCHES):
        for i in range(rows):
            for j in range(2):
                out[b, i, j] = dtype(0.2) + dtype(0.04) * dtype((2 * i + 3 * j + b) % 6)
    return out


def _rect(m, n, dtype):
    out = np.empty((BATCHES, m, n), dtype=dtype)
    for b in range(BATCHES):
        for i in range(m):
            for j in range(n):
                v = dtype(0.04) * dtype(1 + ((i + 3 * j + b) % 7))
                if i == j:
                    v += dtype(1.5) + dtype(0.05) * dtype(b)
                out[b, i, j] = v
    return out


def _run(binary, op, precision, n):
    result = subprocess.run(
        [str(binary), op, precision, str(n)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    return np.fromstring(result.stdout, sep=" ", dtype=np.float64)


def _assert_operation(binary, op, precision, dtype, n):
    raw = _run(binary, op, precision, n)
    rtol = 4e-4 if dtype == np.float32 else 2e-11
    atol = 5e-5 if dtype == np.float32 else 2e-12
    rhs = _rhs(n, dtype).astype(np.float64)

    if op == "potrf":
        got = raw.reshape(BATCHES, n, n).transpose(0, 2, 1)
        original = _spd(n, dtype).astype(np.float64)
        for b in range(BATCHES):
            lower = np.tril(got[b])
            np.testing.assert_allclose(lower @ lower.T, original[b], rtol=rtol, atol=atol)
        return

    if op == "trsm":
        got = raw.reshape(BATCHES, 2, n).transpose(0, 2, 1)
        lower = _lower(n, dtype).astype(np.float64)
        for b in range(BATCHES):
            np.testing.assert_allclose(lower[b] @ got[b], 0.7 * rhs[b], rtol=rtol, atol=atol)
        return

    if op in {"posv", "potrs"}:
        got = raw.reshape(BATCHES, 2, n).transpose(0, 2, 1)
        if op == "posv":
            matrix = _spd(n, dtype).astype(np.float64)
        else:
            lower = _lower(n, dtype).astype(np.float64)
            matrix = lower @ lower.transpose(0, 2, 1)
        for b in range(BATCHES):
            np.testing.assert_allclose(matrix[b] @ got[b], rhs[b], rtol=rtol, atol=atol)
        return

    if op == "getrf":
        got = raw.reshape(BATCHES, n, n).transpose(0, 2, 1)
        original = _general(n, dtype).astype(np.float64)
        for b in range(BATCHES):
            lower = np.tril(got[b], -1) + np.eye(n)
            upper = np.triu(got[b])
            np.testing.assert_allclose(lower @ upper, original[b], rtol=rtol, atol=atol)
        return

    if op in {"getrs", "gesv"}:
        got = raw.reshape(BATCHES, 2, n).transpose(0, 2, 1)
        matrix = _general(n, dtype).astype(np.float64)
        for b in range(BATCHES):
            np.testing.assert_allclose(matrix[b] @ got[b], rhs[b], rtol=rtol, atol=atol)
        return

    m = n + 2
    matrix = _rect(m, n, dtype).astype(np.float64)
    if op == "geqrf":
        got = raw.reshape(BATCHES, n, m).transpose(0, 2, 1)
        for b in range(BATCHES):
            vendor_r = np.triu(got[b][:n, :])
            _, reference_r = np.linalg.qr(matrix[b], mode="reduced")
            np.testing.assert_allclose(
                np.abs(vendor_r), np.abs(reference_r), rtol=rtol, atol=atol
            )
        return

    got = raw.reshape(BATCHES, 2, m).transpose(0, 2, 1)[:, :n, :]
    rhs_rect = _rhs(m, dtype).astype(np.float64)
    for b in range(BATCHES):
        reference, *_ = np.linalg.lstsq(matrix[b], rhs_rect[b], rcond=None)
        np.testing.assert_allclose(got[b], reference, rtol=rtol, atol=atol)


@pytest.mark.parametrize("n", [4, 8])
@pytest.mark.parametrize("precision,dtype", [("f32", np.float32), ("f64", np.float64)])
@pytest.mark.parametrize("op", OPS)
def test_nvidia_thread_operation(bin_nvidia_thread, op, precision, dtype, n):
    _assert_operation(bin_nvidia_thread, op, precision, dtype, n)


@pytest.mark.parametrize("n", [6, 12, 16, 24, 32])
@pytest.mark.parametrize("precision,dtype", [("f32", np.float32), ("f64", np.float64)])
@pytest.mark.parametrize("op", ["potrf", "trsm", "posv"])
def test_nvidia_thread_timing_domain(bin_nvidia_thread, op, precision, dtype, n):
    """Every timing candidate is admitted by its numerical oracle first."""
    _assert_operation(bin_nvidia_thread, op, precision, dtype, n)
