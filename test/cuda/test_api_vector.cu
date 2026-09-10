#include "../../glass-cgrps.cuh"
#include <cuda_runtime.h>
#include <cstdio>

// Compile-only overload canary. The kernel is launched with run=false, so the
// null placeholders are never dereferenced; every call below is nevertheless
// parsed and instantiated by nvcc. Numerical behavior remains the job of the
// focused L1 tests and their independent host oracles.
__global__ void compile_vector_contracts(bool run, float* p, uint32_t* u) {
    if (!run) return;

    glass::block::argmax(4u, p, u, p);
    glass::block::argmax_fast(4u, p, u, p);
    glass::block::argmin_fast(4u, p, u, p);
    (void)glass::warp::argmax_pair(1.0f, 0u);
    (void)glass::warp::argmin_pair(1.0f, 0u);
    glass::block::asum_fast<float, 4>(p, p);
    (void)glass::warp::asum<float, 4>(p);
    (void)glass::thread::asum(4u, p);

    glass::block::axpy<float, 4>(1.0f, p, p, p);
    glass::block::axpby<float, 4>(1.0f, p, 1.0f, p, p);
    glass::warp::axpy<float, 4>(1.0f, p, p);
    glass::thread::axpy(4u, 1.0f, p, p);
    glass::block::clip<float, 4>(p, p, p);
    glass::block::copy(4u, 1.0f, p, p);
    glass::block::copy<float, 4>(1.0f, p, p);
    glass::warp::copy<float, 4>(p, p);
    glass::thread::copy(4u, p, p);
    glass::block::dot(4u, p, p);
    glass::block::dot_fast<float, 4>(p, p, p, p);
    (void)glass::thread::dot(4u, p, p);

    glass::block::elementwise_less_than(4u, p, p, p);
    glass::block::elementwise_more_than(4u, p, p, p);
    glass::block::elementwise_less_than_or_eq(4u, p, p, p);
    glass::block::elementwise_and(4u, p, p, p);
    glass::block::elementwise_mult_scalar(4u, p, 1.0f, p);
    glass::block::elementwise_max_scalar(4u, p, 1.0f, p);
    glass::block::elementwise_min_scalar(4u, p, 1.0f, p);
    glass::block::elementwise_max<float, 4>(p, p, p);
    glass::block::elementwise_min<float, 4>(p, p, p);
    glass::block::elementwise_abs<float, 4>(p, p);
    glass::block::elementwise_mult<float, 4>(p, p, p);
    glass::block::elementwise_sub<float, 4>(p, p, p);
    glass::block::elementwise_add<float, 4>(p, p, p);

    glass::block::iamax<float, 4>(p, u, p);
    glass::block::iamax<float, 4>(p, u, p, p);
    glass::block::iamax_lowmem<float, 4>(p, u);
    glass::block::iamax_lowmem(4u, p, u, p);
    glass::block::iamax_lowmem<float, 4>(p, u, p);
    glass::block::iamax_fast<float, 4>(p, u, p);
    glass::block::iamax_fast(4u, p, u, p, p);
    glass::block::iamax_fast<float, 4>(p, u, p, p);
    (void)glass::warp::iamax<float, 4>(p);

    glass::block::set_identity<float, 4>(p);
    glass::block::add_identity<float, 4>(p, 1.0f);
    glass::block::add_identity_partial<float, 4, 2>(p, 1.0f);
    glass::block::infnorm<float, 4>(p);
    glass::block::vector_norm<float, 4>(p, p);
    glass::block::vector_norm_fast<float, 4>(p, p, p);
    glass::block::nrm1_diff_fast<float, 4>(p, p, p, p);
    (void)glass::warp::nrm1_diff<float, 4>(p, p);
    (void)glass::warp::nrm2<float, 4>(p);
    (void)glass::thread::nrm1_diff(4u, p, p);
    (void)glass::thread::nrm2(4u, p);
    glass::block::reduce_lowmem<float, 4>(p);
    glass::block::reduce_fast<float, 4>(p, p);
    glass::warp::reduce<float, 4>(p);
    glass::thread::reduce(4u, p);
    (void)glass::thread::reduce(1.0f);
    glass::thread::rot(4u, p, p, 1.0f, 0.0f);
    glass::block::scal(4u, 1.0f, p, p);
    glass::block::scal<float, 4>(1.0f, p, p);
    glass::warp::scal<float, 4>(1.0f, p);
    glass::thread::scal(4u, 1.0f, p);
    glass::block::set_const<float, 4>(1.0f, p);
    glass::block::swap<float, 4>(p, p);
    glass::thread::symmetrize(2u, p);
    glass::block::transpose(4u, p);
    glass::block::transpose<float, 2, 3>(p, p);
    glass::block::transpose<float, 4>(p);

    glass::cgrps::axpy<float, 4>(1.0f, p, p);
    glass::cgrps::axpy<float, 4>(1.0f, p, p, p);
    glass::cgrps::copy(4u, 1.0f, p, p);
    glass::cgrps::copy<float, 4>(p, p);
    glass::cgrps::scal(4u, 1.0f, p, p);
    glass::cgrps::scal<float, 4>(1.0f, p);
    glass::cgrps::swap<float, 4>(p, p);
    glass::cgrps::reduce<float, 4>(p);
    glass::cgrps::dot<float, 4>(p, p);
    glass::cgrps::transpose(4u, p);
    glass::cgrps::elementwise_less_than(4u, p, p, p);
    glass::cgrps::elementwise_more_than(4u, p, p, p);
    glass::cgrps::elementwise_less_than_or_eq(4u, p, p, p);
    glass::cgrps::elementwise_and(4u, p, p, p);
    glass::cgrps::elementwise_mult_scalar(4u, p, 1.0f, p);
    glass::cgrps::elementwise_max_scalar(4u, p, 1.0f, p);
    glass::cgrps::elementwise_min_scalar(4u, p, 1.0f, p);
}

int main() {
    std::puts("1");
    return 0;
}
