#ifndef ELEMENT_H
#define ELEMENT_H

#include <cstdint>
#include <cooperative_groups.h>
namespace cgrps = cooperative_groups;

template <typename T>
__device__
void elementwise_max(uint32_t N, T* a, T* b, T* c,
                cgrps::thread_group g = cgrps::this_thread_block()) {
    for (int i = g.thread_rank(); i < N; i += g.size()) {
        c[i] = max(a[i], b[i]);
    }
}

template <typename T>
__device__
void elementwise_min(uint32_t N, T* a, T* b, T* c,
                cgrps::thread_group g = cgrps::this_thread_block()) {
    for (int i = g.thread_rank(); i < N; i += g.size()) {
        c[i] = min(a[i], b[i]);
    }
}


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
void elementwise_more_than(uint32_t N, T* a, T* b, T* c,
                cgrps::thread_group g = cgrps::this_thread_block()) {
    for (int i = g.thread_rank(); i < N; i += g.size()) {
        c[i] = a[i] > b[i];
    }
}

template <typename T>
__device__
void elementwise_less_than_or_eq(uint32_t N, T* a, T* b, T* c,
                cgrps::thread_group g = cgrps::this_thread_block()) {
    for (int i = g.thread_rank(); i < N; i += g.size()) {
        c[i] = a[i] <= b[i];
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
void elementwise_abs(uint32_t N, T* a, T* b,
                cgrps::thread_group g = cgrps::this_thread_block()) {
    for (int i = g.thread_rank(); i < N; i += g.size()) {
        b[i] = abs(a[i]);
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
void elementwise_sub(uint32_t N, T* a, T* b, T* c,
                cgrps::thread_group g = cgrps::this_thread_block()) {
    for (int i = g.thread_rank(); i < N; i += g.size()) {
        c[i] = a[i] - b[i];
    }
}

template <typename T>
__device__
void elementwise_add(uint32_t N, T* a, T* b, T* c,
                cgrps::thread_group g = cgrps::this_thread_block()) {
    for (int i = g.thread_rank(); i < N; i += g.size()) {
        c[i] = a[i] + b[i];
    }
}

template <typename T>
__device__
void elementwise_mult_scalar(uint32_t N, T* a, T b, T* c,
                cgrps::thread_group g = cgrps::this_thread_block()) {
    for (int i = g.thread_rank(); i < N; i += g.size()) {
        c[i] = a[i] * b;
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

// === glass::simple variants ===
namespace simple {
#define GLASS_SIMPLE_RANK \
    uint32_t rank = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y; \
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;

    template <typename T>
    __device__ void elementwise_max(uint32_t N, T* a, T* b, T* c)
    { GLASS_SIMPLE_RANK for (uint32_t i = rank; i < N; i += size) c[i] = max(a[i], b[i]); }

    template <typename T>
    __device__ void elementwise_min(uint32_t N, T* a, T* b, T* c)
    { GLASS_SIMPLE_RANK for (uint32_t i = rank; i < N; i += size) c[i] = min(a[i], b[i]); }

    template <typename T>
    __device__ void elementwise_less_than(uint32_t N, T* a, T* b, T* c)
    { GLASS_SIMPLE_RANK for (uint32_t i = rank; i < N; i += size) c[i] = a[i] < b[i]; }

    template <typename T>
    __device__ void elementwise_more_than(uint32_t N, T* a, T* b, T* c)
    { GLASS_SIMPLE_RANK for (uint32_t i = rank; i < N; i += size) c[i] = a[i] > b[i]; }

    template <typename T>
    __device__ void elementwise_less_than_or_eq(uint32_t N, T* a, T* b, T* c)
    { GLASS_SIMPLE_RANK for (uint32_t i = rank; i < N; i += size) c[i] = a[i] <= b[i]; }

    template <typename T>
    __device__ void elementwise_less_than_scalar(uint32_t N, T* a, T b, T* c)
    { GLASS_SIMPLE_RANK for (uint32_t i = rank; i < N; i += size) c[i] = a[i] < b; }

    template <typename T>
    __device__ void elementwise_and(uint32_t N, T* a, T* b, T* c)
    { GLASS_SIMPLE_RANK for (uint32_t i = rank; i < N; i += size) c[i] = a[i] && b[i]; }

    template <typename T>
    __device__ void elementwise_not(uint32_t N, T* a, T* c)
    { GLASS_SIMPLE_RANK for (uint32_t i = rank; i < N; i += size) c[i] = !a[i]; }

    template <typename T>
    __device__ void elementwise_abs(uint32_t N, T* a, T* b)
    { GLASS_SIMPLE_RANK for (uint32_t i = rank; i < N; i += size) b[i] = abs(a[i]); }

    template <typename T>
    __device__ void elementwise_mult(uint32_t N, T* a, T* b, T* c)
    { GLASS_SIMPLE_RANK for (uint32_t i = rank; i < N; i += size) c[i] = a[i] * b[i]; }

    template <typename T>
    __device__ void elementwise_sub(uint32_t N, T* a, T* b, T* c)
    { GLASS_SIMPLE_RANK for (uint32_t i = rank; i < N; i += size) c[i] = a[i] - b[i]; }

    template <typename T>
    __device__ void elementwise_add(uint32_t N, T* a, T* b, T* c)
    { GLASS_SIMPLE_RANK for (uint32_t i = rank; i < N; i += size) c[i] = a[i] + b[i]; }

    template <typename T>
    __device__ void elementwise_mult_scalar(uint32_t N, T* a, T b, T* c)
    { GLASS_SIMPLE_RANK for (uint32_t i = rank; i < N; i += size) c[i] = a[i] * b; }

    template <typename T>
    __device__ void elementwise_max_scalar(uint32_t N, T* a, T b, T* c)
    { GLASS_SIMPLE_RANK for (uint32_t i = rank; i < N; i += size) c[i] = max(a[i], b); }

    template <typename T>
    __device__ void elementwise_min_scalar(uint32_t N, T* a, T b, T* c)
    { GLASS_SIMPLE_RANK for (uint32_t i = rank; i < N; i += size) c[i] = min(a[i], b); }

#undef GLASS_SIMPLE_RANK
}
// ===

#endif