
#include "../glass.cuh"


template <typename T>
__global__
void global_axpy(uint32_t n, 
          T alpha, 
          T *x, 
          T *y)
{
	glass::column_major::axpy<T>(n, alpha, x, y);
}

template <typename T>
__global__
void global_axpy(uint32_t n, 
          T alpha, 
          T *x, 
          T *y, 
          T *z)
{
    glass::column_major::axpy<T>(n, alpha, x, y, z);
}

template <typename T>
__global__
void global_clip(uint32_t n, 
          T *x,
          T *l, 
          T *u)
{
    glass::column_major::clip<T>(n, x, l, u);
}

template <typename T>
__global__
void global_copy(uint32_t n, 
          T *x, 
          T *y)
{
    glass::column_major::copy<T>(n, x, y);
}


template <typename T>
__global__
void global_copy(uint32_t n,
          T alpha,
          T *x, 
          T *y)
{
    glass::column_major::copy<T>(n, alpha, x, y);
}

template <typename T>
__global__
void global_dot(uint32_t n, 
          T *x, 
          T *y)
{
    glass::column_major::dot<T>(n, x, y);
}

template <typename T>
__global__
void global_dot(T *out,
         uint32_t n, 
         T *x, 
         T *y)
{
	glass::column_major::dot<T>(out, n, x, y);
}

template <typename T>
__global__
void global_infnorm(uint32_t n, 
         T *x)
{
	glass::column_major::infnorm<T>(n, x);
}

template <typename T>
__global__
void global_loadIdentity(uint32_t dimA, 
                  T *A){
	glass::column_major::loadIdentity<T>(dimA, A);
}

template <typename T>
__global__
void global_loadIdentity(uint32_t dimA, 
                  T *A, 
                  uint32_t dimB, 
                  T *B){
	glass::column_major::loadIdentity<T>(dimA, A, dimB, B);
}

template <typename T>
__global__
void global_loadIdentity(uint32_t dimA, 
                  T *A, 
                  uint32_t dimB, 
                  T *B, 
                  uint32_t dimC, 
                  T *C){
	
	glass::column_major::loadIdentity<T>(dimA, A, dimB, B, dimC, C);
}

template <typename T>
__global__
void global_addI(uint32_t n,
          T *A,
          T alpha)
{
	glass::column_major::addI<T>(n, A, alpha);
}

template <typename T>
__global__
void  global_l2norm(const uint32_t n,
                T *x)
{
    glass::column_major::l2norm<T>(n, x);
}

template <typename T>
__global__
void global_reduce(uint32_t n,
            T *x){
            
            glass::column_major::reduce<T>(n, x);
}

template <typename T>
__global__
void global_reduce(T *out,
            uint32_t n,
            T *x)
{
	glass::column_major::reduce<T>(out, n, x);
}

template <typename T>
__global__
void global_scal(uint32_t n, 
          T alpha, 
          T *x)
{
	glass::column_major::scal<T>(n, alpha, x);
}

template <typename T>
__global__
void global_set_const(uint32_t n,
            T alpha,
            T *x)
{
    glass::column_major::set_const<T>(n, alpha, x);
}

template <typename T>
__global__
void global_swap(uint32_t n, 
          T alpha, 
          T *x, 
          T *y)
{
    glass::column_major::swap<T>(n, alpha, x, y);
}

template <typename T, bool TRANSPOSE = false>
__global__
void global_gemv(uint32_t m,
          uint32_t n,
          T alpha,
          T *A,
          T *x,
          T beta, 
          T *y)
{
        glass::column_major::gemv<T, TRANSPOSE>(m, n, alpha, A, x, beta, y);
}

template <typename T, bool TRANSPOSE = false>
__global__
void global_gemv(uint32_t m,
          uint32_t n,
          T alpha,
          T *A,
          T *x,
          T *y)
{
        glass::column_major::gemv<T, TRANSPOSE>(m, n, alpha, A, x, y);
}

template <typename T> 
__global__ 
void global_cholDecomp_InPlace_c (uint32_t n,
                        T *s_A)
{
    glass::chol_InPlace<T>(n, s_A);
}

template <typename T, bool TRANSPOSE_B>
__global__
void global_gemm(uint32_t m,
          uint32_t n,
          uint32_t k,
          T alpha, 
          T *A, 
          T *B,
          T beta,
          T *C)
{
	glass::column_major::gemm<T, TRANSPOSE_B>(m, n, k, alpha, A, B, beta, C);
}

template <typename T, bool TRANSPOSE_B = false>
__global__
void global_gemm(uint32_t m,
          uint32_t n,
          uint32_t k,
          T alpha, 
          T *A, 
          T *B,
          T *C)
{
	glass::column_major::gemm<T, TRANSPOSE_B>(m, n, k, alpha, A, B, C);
}

template <typename T>
__global__
void global_invertMatrix(uint32_t dimA, T *A, T *s_temp){ 
	glass::column_major::invertMatrix<T>(dimA, A, s_temp);
}

template <typename T>
__global__
void global_invertMatrix(uint32_t dimA, T *A, uint32_t dimB, T *B, T *s_temp){
	glass::column_major::invertMatrix<T>(dimA, A, dimB, B, s_temp);
}

template <typename T>
__global__
void invertMatrix(uint32_t dimA, T *A, uint32_t dimB, T *B, uint32_t dimC, T *C, T *s_temp){
	glass::column_major::invertMatrix<T>(dimA, A, dimB, B, dimC, C, s_temp);
}