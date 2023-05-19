/*      L1      */
template <typename T>
__device__
void axpy(std::uint32_t n, 
          T alpha, 
          T *x, 
          T *y, 
          cgrps::thread_group g);

template <typename T>
__device__
void swap(std::uint32_t n, 
          T alpha, 
          T *x, 
          T *y, 
          cgrps::thread_group g);

template <typename T>
__device__
void dot(std::uint32_t n, 
          T *x, 
          T *y, 
          cgrps::thread_group g);


template <typename T>
__device__
void copy(std::uint32_t n, 
          T *x, 
          T *y, 
          cgrps::thread_group g);

template <typename T>
__device__
void scal(std::uint32_t n, 
          T alpha, 
          T *x, 
          cgrps::thread_group g);

template <typename T>
__device__
void loadIdentity(uint32_t dimA, T *A);

template <typename T>
__device__
void loadIdentity(uint32_t dimA, 
                  T *A, 
                  uint32_t dimB, 
                  T *B);

template <typename T>
__device__
void loadIdentity(uint32_t dimA, 
                  T *A, 
                  uint32_t dimB, 
                  T *B, 
                  uint32_t dimC, 
                  T *C);


/*      L2      */
template <typename T>
__device__
void gemv(std::uint32_t m,
          std::uint32_t n,
          T alpha,
          T *A,
          T *x,
          T beta, 
          T *y, 
          cgrps::thread_group g);


/*      L3      */
template <typename T, bool TRANSPOSE_B>
__device__
void gemm(std::uint32_t m,
          std::uint32_t n,
          std::uint32_t k,
          T alpha, 
          T *A, 
          T *B,
          T beta,
          T *C, 
          cgrps::thread_group g);

template <typename T>
__device__
void invertMatrix(uint32_t dimA, 
                  T *A, 
                  T *s_temp, 
                  cgrps::thread_group g);

template <typename T>
__device__
void invertMatrix(uint32_t dimA, 
                  T *A, 
                  uint32_t dimB, 
                  T *B, 
                  uint32_t dimMax, 
                  T *s_temp, 
                  cgrps::thread_group g);

template <typename T>
__device__
void invertMatrix(uint32_t dimA, 
                  T *A, 
                  uint32_t dimB, 
                  T *B, 
                  uint32_t dimC, 
                  T *C, 
                  uint32_t dimMax, 
                  T *s_temp, 
                  cgrps::thread_group g);
