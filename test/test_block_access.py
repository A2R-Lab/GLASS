"""store_block / load_block — dense d×d ↔ [L|D|R] strip movers (Phase F).

Replaces GATO's hand-rolled Schur remap loops. The index map (verified against
bdmv.cuh's row-major strip and GATO schur_linsys.cuh):
    strip[y*band_width + slot*d + x] = scale * src[(TRANSPOSE? x*d+y : y*d+x)]
with band_width = 3*d, slots LEFT=0 / MAIN=1 / RIGHT=2.

Each (y,x) is written once (no reduction) → byte-identical across the full thread
sweep. Warp forms validated at one 32-lane warp.
"""

import os
import subprocess
import tempfile

import numpy as np
import pytest

from conftest import THREAD_SWEEP

WARP = 32
DIMS = [3, 6, 7]
SLOTS = [0, 1, 2]  # LEFT, MAIN, RIGHT


def _write(arr):
    f = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
    np.asarray(arr, dtype=np.float32).tofile(f)
    f.close()
    return f.name


def _run(bins, args):
    cmd = [str(bins["block_access"])] + [str(a) for a in args]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"runner failed:\n{res.stderr}")
    return np.fromstring(res.stdout.strip(), sep=" ").astype(np.float32)


def _expected_strip(src, d, slot, transpose, scale):
    bw = 3 * d
    out = np.zeros((d, bw), dtype=np.float32)
    for k in range(d * d):
        x, y = k % d, k // d
        out[y, slot * d + x] = scale * src[(x * d + y) if transpose else (y * d + x)]
    return out.ravel()


def _expected_dense(strip, d, slot, transpose, scale):
    bw = 3 * d
    s = strip.reshape(d, bw)
    out = np.zeros(d * d, dtype=np.float32)
    for k in range(d * d):
        x, y = k % d, k // d
        out[(x * d + y) if transpose else (y * d + x)] = scale * s[y, slot * d + x]
    return out


@pytest.mark.parametrize("d", DIMS)
@pytest.mark.parametrize("slot", SLOTS)
@pytest.mark.parametrize("transpose", [0, 1])
@pytest.mark.parametrize("scale", [1.0, -1.0])
def test_store_block_sweep(bins, d, slot, transpose, scale):
    """store_block matches the index-map oracle and is byte-identical across the sweep."""
    src = (np.arange(d * d, dtype=np.float32) + 1.0)
    sf = _write(src)
    try:
        expected = _expected_strip(src, d, slot, transpose, scale)
        ref = None
        for t in THREAD_SWEEP:
            out = _run(bins, ["store", t, d, slot, transpose, scale, sf])
            assert np.allclose(out, expected, atol=1e-6), f"d={d} slot={slot} tr={transpose} t={t}"
            if ref is None:
                ref = out
            else:
                assert np.array_equal(out, ref), f"d={d} t={t}: non-invariant"
    finally:
        os.unlink(sf)


@pytest.mark.parametrize("d", DIMS)
@pytest.mark.parametrize("slot", SLOTS)
@pytest.mark.parametrize("transpose", [0, 1])
def test_load_block_sweep(bins, d, slot, transpose):
    """load_block (strip → dense) matches the oracle, byte-identical across the sweep."""
    bw = 3 * d
    rng = np.random.default_rng(d * 10 + slot)
    strip = rng.standard_normal(d * bw).astype(np.float32)
    sf = _write(strip)
    scale = -1.0 if (slot == 1) else 1.0
    try:
        expected = _expected_dense(strip, d, slot, transpose, scale)
        ref = None
        for t in THREAD_SWEEP:
            out = _run(bins, ["load", t, d, slot, transpose, scale, sf])
            assert np.allclose(out, expected, atol=1e-6), f"d={d} slot={slot} tr={transpose} t={t}"
            if ref is None:
                ref = out
            else:
                assert np.array_equal(out, ref), f"d={d} t={t}: non-invariant"
    finally:
        os.unlink(sf)


@pytest.mark.parametrize("d", DIMS)
@pytest.mark.parametrize("slot", SLOTS)
def test_roundtrip_identity(bins, d, slot):
    """load_block ∘ store_block == identity (same TRANSPOSE, scale 1)."""
    src = np.random.default_rng(d).standard_normal(d * d).astype(np.float32)
    sf = _write(src)
    try:
        strip = _run(bins, ["store", 64, d, slot, 0, 1.0, sf])
        stf = _write(strip)
        try:
            back = _run(bins, ["load", 64, d, slot, 0, 1.0, stf])
            assert np.allclose(back, src, atol=1e-6)
        finally:
            os.unlink(stf)
    finally:
        os.unlink(sf)


@pytest.mark.parametrize("d", DIMS)
@pytest.mark.parametrize("slot", SLOTS)
@pytest.mark.parametrize("transpose", [0, 1])
@pytest.mark.parametrize("op", ["store", "load"])
def test_warp_matches_block(bins, op, d, slot, transpose):
    """Warp store/load_block == block form at one warp."""
    bw = 3 * d
    rng = np.random.default_rng(100 + d + slot)
    n_in = d * bw if op == "load" else d * d
    src = rng.standard_normal(n_in).astype(np.float32)
    sf = _write(src)
    scale = -1.0
    try:
        block = _run(bins, [op, 64, d, slot, transpose, scale, sf])
        warp = _run(bins, [op + "_warp", WARP, d, slot, transpose, scale, sf])
        assert np.array_equal(block, warp), f"{op} d={d} slot={slot} tr={transpose}: warp != block"
    finally:
        os.unlink(sf)


def test_gato_schur_patterns(bins):
    """The exact GATO patterns: phi_kᵀ (direct), phi_k (transpose), theta negate.
    d=STATE_SIZE arbitrary here (7); mirrors schur_linsys.cuh's three writes."""
    d = 7
    src = np.random.default_rng(0).standard_normal(d * d).astype(np.float32)
    sf = _write(src)
    try:
        # phi_kᵀ → RIGHT slot, direct
        right = _run(bins, ["store", 33, d, 2, 0, 1.0, sf]).reshape(d, 3 * d)
        # phi_k → LEFT slot, transpose
        left = _run(bins, ["store", 33, d, 0, 1, 1.0, sf]).reshape(d, 3 * d)
        # -theta → MAIN slot, transpose + negate
        main = _run(bins, ["store", 33, d, 1, 1, -1.0, sf]).reshape(d, 3 * d)
        # rtol covers the harness's %.8g text round-trip (device values are exact).
        for y in range(d):
            for x in range(d):
                assert np.isclose(right[y, 2 * d + x], src[y * d + x], rtol=1e-6, atol=1e-7)
                assert np.isclose(left[y, x], src[x * d + y], rtol=1e-6, atol=1e-7)
                assert np.isclose(main[y, d + x], -src[x * d + y], rtol=1e-6, atol=1e-7)
    finally:
        os.unlink(sf)
