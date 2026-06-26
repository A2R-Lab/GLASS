// test_block_access.cu — dense d×d block ↔ [L|D|R] strip movers
// (glass::store_block / load_block, block + warp).
//
//   store <threads> <d> <slot> <transpose> <scale> <src.bin>   (src = dense d*d)
//         → prints the full strip (d * 3d, row-major), pre-zeroed.
//   load  <threads> <d> <slot> <transpose> <scale> <strip.bin> (strip = d*3d)
//         → prints the dense d*d block.
//   _warp variants run on one 32-lane warp.
// d ∈ {3,6,7}; slot 0=LEFT 1=MAIN 2=RIGHT; transpose 0/1.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#include "helpers.cuh"
#include "../../glass.cuh"

template <uint32_t d, bool TRANSPOSE, bool WARP, bool LOAD>
__global__ void k_blk(uint32_t slot_i, float scale, const float* in, float* out) {
    glass::BandSlot slot = static_cast<glass::BandSlot>(slot_i);
    if constexpr (LOAD) {
        if constexpr (WARP) glass::warp::load_block<float, d, 3 * d, TRANSPOSE>(out, in, slot, scale);
        else                glass::load_block<float, d, 3 * d, TRANSPOSE>(out, in, slot, scale);
    } else {
        if constexpr (WARP) glass::warp::store_block<float, d, 3 * d, TRANSPOSE>(out, slot, in, scale);
        else                glass::store_block<float, d, 3 * d, TRANSPOSE>(out, slot, in, scale);
    }
}

template <uint32_t d>
int run_d(int THREADS, int slot, bool tr, bool warp, bool load, float scale,
          const float* in, float* out, int out_n) {
#define K(TR, WP, LD) k_blk<d, TR, WP, LD><<<1, THREADS>>>(slot, scale, in, out)
    if (load) {
        if (warp) { tr ? K(true, true, true)   : K(false, true, true); }
        else      { tr ? K(true, false, true)  : K(false, false, true); }
    } else {
        if (warp) { tr ? K(true, true, false)  : K(false, true, false); }
        else      { tr ? K(true, false, false) : K(false, false, false); }
    }
#undef K
    cudaDeviceSynchronize();
    print_device_vec(out, out_n);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 8) { fprintf(stderr, "usage: %s <store|load[_warp]> <threads> <d> <slot> <transpose> <scale> <in.bin>\n", argv[0]); return 1; }
    const char* op = argv[1];
    int THREADS = atoi(argv[2]);
    int d       = atoi(argv[3]);
    int slot    = atoi(argv[4]);
    bool tr     = atoi(argv[5]) != 0;
    float scale = (float)atof(argv[6]);
    bool warp   = (strstr(op, "_warp") != nullptr);
    bool load   = (strncmp(op, "load", 4) == 0);

    int dd = d * d, strip = d * 3 * d;
    int in_n  = load ? strip : dd;
    int out_n = load ? dd    : strip;
    float* d_in = read_device_vec(argv[7], in_n);
    float* d_out = alloc_device_vec(out_n);   // zeroed: store leaves non-slot strip cols at 0

    if      (d == 3) return run_d<3>(THREADS, slot, tr, warp, load, scale, d_in, d_out, out_n);
    else if (d == 6) return run_d<6>(THREADS, slot, tr, warp, load, scale, d_in, d_out, out_n);
    else if (d == 7) return run_d<7>(THREADS, slot, tr, warp, load, scale, d_in, d_out, out_n);
    fprintf(stderr, "unsupported d=%d (use 3,6,7)\n", d);
    return 1;
}
