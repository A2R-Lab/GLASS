// smoke_vendor.cu — vendor-floor probe: instantiate a minimal cuBLASDx block
// GEMM descriptor for the arch given by -DSM_TARGET=<cc*10> (e.g. 620, 720).
// cuBLASDx's own headers refuse cc < 7.0 at compile time; this TU exists to
// capture that refusal verbatim for the backwards-reach table. Compile-only:
//   nvcc -std=c++17 -arch=sm_<xx> -DSM_TARGET=<xx0> -cubin \
//        -I$MATHDX_ROOT/include -I$MATHDX_ROOT/external/cutlass/include \
//        smoke_vendor.cu -o /dev/null
#include <cublasdx.hpp>

using GEMM = decltype(cublasdx::Size<16, 16, 16>() +
                      cublasdx::Precision<float>() +
                      cublasdx::Type<cublasdx::type::real>() +
                      cublasdx::Function<cublasdx::function::MM>() +
                      cublasdx::SM<SM_TARGET>() +
                      cublasdx::Block());

__global__ void probe(float* a, float* b, float* c) {
    GEMM().execute(1.0f, a, b, 0.0f, c);
}
