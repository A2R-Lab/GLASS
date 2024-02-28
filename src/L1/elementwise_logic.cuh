#include <cstdint>
#include <cooperative_groups.h>
namespace cgrps = cooperative_groups;

template <typename T>
__device__
void elementwise_less_than(uint32_t N, T* a, T* b, T* c,
                cgrps::thread_group g = cgrps::this_thread_block()) {
    for (int i = g.thread_rank(); i < N; i += g.size()) {
        c[i] = a[i] < b[i];
    }
}

template <typename T>
__device__
void elementwise_less_than_scalar(uint32_t N, T* a, T b, T* c,
                cgrps::thread_group g = cgrps::this_thread_block()) {
    for (int i = g.thread_rank(); i < N; i += g.size()) {
        c[i] = a[i] < b;
    }
}

template <typename T>
__device__
void elementwise_and(uint32_t N, T* a, T* b, T* c,
                cgrps::thread_group g = cgrps::this_thread_block()) {
    for (int i = g.thread_rank(); i < N; i += g.size()) {
        c[i] = a[i] && b[i];
    }
}

template <typename T>
__device__
void elementwise_not(uint32_t N, T* a, T* c,
                cgrps::thread_group g = cgrps::this_thread_block()) {
    for (int i = g.thread_rank(); i < N; i += g.size()) {
        c[i] = !a[i];
    }
}

template <typename T>
__device__
void elementwise_mult(uint32_t N, T* a, T* b, T* c,
                cgrps::thread_group g = cgrps::this_thread_block()) {
    for (int i = g.thread_rank(); i < N; i += g.size()) {
        c[i] = a[i] * b[i];
    }
}

template <typename T>
__device__
void elementwise_max_scalar(uint32_t N, T* a, T b, T* c,
                cgrps::thread_group g = cgrps::this_thread_block()) {
    for (int i = g.thread_rank(); i < N; i += g.size()) {
        c[i] = max(a[i], b);
    }
}

template <typename T>
__device__
void elementwise_min_scalar(uint32_t N, T* a, T b, T* c,
                cgrps::thread_group g = cgrps::this_thread_block()) {
    for (int i = g.thread_rank(); i < N; i += g.size()) {
        c[i] = min(a[i], b);
    }
}