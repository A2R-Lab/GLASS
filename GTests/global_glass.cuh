#include <cstdint>
#include <cooperative_groups.h>
#include "../glass.cuh"


namespace cgrps = cooperative_groups;

template <typename T>
__global__
void global_axpy(std::uint32_t n, 
          T alpha, 
          T *x, 
          T *y, 
          cgrps::thread_group g = cgrps::this_thread_block())
{
	glass::axpy<T>(alpha, x, y, g);
}

template <typename T>
__global__
void global_axpy(std::uint32_t n, 
          T alpha, 
          T *x, 
          T *y, 
          T *z, 
          cgrps::thread_group g = cgrps::this_thread_block())
{
    glass::axpy<T>(n, alpha, x, y, z, g);
}

template <typename T>
__global__
void global_copy(std::uint32_t n, 
          T *x, 
          T *y, 
          cgrps::thread_group g = cgrps::this_thread_block())
{
    glass::copy<T>(n, x, y, g);
}


template <typename T>
__global__
void global_copy(std::uint32_t n,
          T alpha,
          T *x, 
          T *y, 
          cgrps::thread_group g = cgrps::this_thread_block())
{
    glass::copy<T>(n, x, y, g);
}

template <typename T>
__global__
void global_dot(uint32_t n, 
          T *x, 
          T *y)
{
    glass::dot<T>(n, x, y);
}

template <typename T>
__global__
void global_dot(T *out,
         uint32_t n, 
         T *x, 
         T *y)
{
	glass::dot<T>(out, n, x, y);
}

template <typename T>
__global__
void global_loadIdentity(uint32_t dimA, 
                  T *A,
                  cgrps::thread_group g = cgrps::this_thread_block()){
	glass::loadIdentity<T>(dimA, A, g);
}

template <typename T>
__global__
void global_loadIdentity(uint32_t dimA, 
                  T *A, 
                  uint32_t dimB, 
                  T *B,
                  cgrps::thread_group g = cgrps::this_thread_block()){
	glass::loadIdentity<T>(dimA, A, dimB, B, g);
}

template <typename T>
__global__
void global_loadIdentity(uint32_t dimA, 
                  T *A, 
                  uint32_t dimB, 
                  T *B, 
                  uint32_t dimC, 
                  T *C,
                  cgrps::thread_group g = cgrps::this_thread_block()){
	
	glass::loadIdentity<T>(dimA, A, dimB, B, dimC, C, g);
}

template <typename T>
__global__
void global_addI(uint32_t n,
          T *A,
          T alpha,
          cgrps::thread_group g = cgrps::this_thread_block())
{
	glass::addI<T>(n, A, alpha, g);
}

template <typename T>
__global__
void global_reduce(uint32_t n,
            T *x,
            cgrps::thread_group g){
            
            glass::reduce<T>(n, x, g);
}

template <typename T>
__global__
void global_reduce(T *out,
            uint32_t n,
            T *x,
            cgrps::thread_group g)
{
	glass::reduce<T>(out, n, x, g);
}

template <typename T>
__global__
void global_scal(std::uint32_t n, 
          T alpha, 
          T *x, 
          cgrps::thread_group g = cgrps::this_thread_block())
{
	glass::scal<T>(n, alpha, x, g);
}

template <typename T>
__global__
void global_swap(std::uint32_t n, 
          T alpha, 
          T *x, 
          T *y, 
          cgrps::thread_group g)
{
    glass::swap<T>(n, alpha, x, y, g);
}

template <typename T, bool TRANSPOSE = false>
__global__
void global_gemv(std::uint32_t m,
          std::uint32_t n,
          T alpha,
          T *A,
          T *x,
          T beta, 
          T *y, 
          cgrps::thread_group g = cgrps::this_thread_block())
{
        glass::gemv<T, TRANSPOSE>(m, n, alpha, A, x, beta, y, g);
}

template <typename T, bool TRANSPOSE = false>
__global__
void global_gemv(std::uint32_t m,
          std::uint32_t n,
          T alpha,
          T *A,
          T *x,
          T *y, 
          cgrps::thread_group g = cgrps::this_thread_block())
{
        glass::gemv<T, TRANSPOSE>(m, n, alpha, A, x, y, g);
}

/*template <typename T> 
__global__ 
void global_cholDecomp_InPlace_c (std::uint32_t n,
                        T *s_A,
                        cgrps::thread_group g = cgrps::this_thread_block())
{
    glass::cholDecomp_InPlace_c<T>(n, s_A, g);
}*/

template <typename T, bool TRANSPOSE_B>
__global__
void global_gemm(std::uint32_t m,
          std::uint32_t n,
          std::uint32_t k,
          T alpha, 
          T *A, 
          T *B,
          T beta,
          T *C, 
          cgrps::thread_group g)
{
	glass::gemm<T, TRANSPOSE_B>(m, n, k, alpha, A, B, beta, C, g);
}

template <typename T, bool TRANSPOSE_B = false>
__global__
void global_gemm(std::uint32_t m,
          std::uint32_t n,
          std::uint32_t k,
          T alpha, 
          T *A, 
          T *B,
          T *C, 
          cgrps::thread_group g = cgrps::this_thread_block())
{
	glass::gemm<T, TRANSPOSE_B>(m, n, k, alpha, A, B, C, g);
}

template <typename T>
__global__
void global_invertMatrix(uint32_t dimA, T *A, T *s_temp, cgrps::thread_group g = cgrps::this_thread_block()){ 
	glass::invertMatrix<T>(dimA, A, s_temp, g);
}

template <typename T>
__global__
void global_invertMatrix(uint32_t dimA, T *A, uint32_t dimB, T *B, T *s_temp, cgrps::thread_group g = cgrps::this_thread_block()){
	glass::invertMatrix<T>(dimA, A, dimB, B, s_temp, g);
}

template <typename T>
__global__
void invertMatrix(uint32_t dimA, T *A, uint32_t dimB, T *B, uint32_t dimC, T *C, T *s_temp, cgrps::thread_group g = cgrps::this_thread_block()){
	glass::invertMatrix<T>(dimA, A, dimB, B, dimC, C, s_temp, g);
}
















