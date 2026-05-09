#pragma once
#include <cstdint>

#define _GLASS_RS \
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y; \
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;

template <typename T> __device__ void elementwise_max(uint32_t N, T *a, T *b, T *c)
{ _GLASS_RS for (uint32_t i = rank; i < N; i += size) c[i] = max(a[i], b[i]); }

template <typename T> __device__ void elementwise_min(uint32_t N, T *a, T *b, T *c)
{ _GLASS_RS for (uint32_t i = rank; i < N; i += size) c[i] = min(a[i], b[i]); }

template <typename T> __device__ void elementwise_less_than(uint32_t N, T *a, T *b, T *c)
{ _GLASS_RS for (uint32_t i = rank; i < N; i += size) c[i] = a[i] < b[i]; }

template <typename T> __device__ void elementwise_more_than(uint32_t N, T *a, T *b, T *c)
{ _GLASS_RS for (uint32_t i = rank; i < N; i += size) c[i] = a[i] > b[i]; }

template <typename T> __device__ void elementwise_less_than_or_eq(uint32_t N, T *a, T *b, T *c)
{ _GLASS_RS for (uint32_t i = rank; i < N; i += size) c[i] = a[i] <= b[i]; }

template <typename T> __device__ void elementwise_less_than_scalar(uint32_t N, T *a, T b, T *c)
{ _GLASS_RS for (uint32_t i = rank; i < N; i += size) c[i] = a[i] < b; }

template <typename T> __device__ void elementwise_and(uint32_t N, T *a, T *b, T *c)
{ _GLASS_RS for (uint32_t i = rank; i < N; i += size) c[i] = a[i] && b[i]; }

template <typename T> __device__ void elementwise_not(uint32_t N, T *a, T *c)
{ _GLASS_RS for (uint32_t i = rank; i < N; i += size) c[i] = !a[i]; }

template <typename T> __device__ void elementwise_abs(uint32_t N, T *a, T *b)
{ _GLASS_RS for (uint32_t i = rank; i < N; i += size) b[i] = abs(a[i]); }

template <typename T> __device__ void elementwise_mult(uint32_t N, T *a, T *b, T *c)
{ _GLASS_RS for (uint32_t i = rank; i < N; i += size) c[i] = a[i]*b[i]; }

template <typename T> __device__ void elementwise_sub(uint32_t N, T *a, T *b, T *c)
{ _GLASS_RS for (uint32_t i = rank; i < N; i += size) c[i] = a[i] - b[i]; }

template <typename T> __device__ void elementwise_add(uint32_t N, T *a, T *b, T *c)
{ _GLASS_RS for (uint32_t i = rank; i < N; i += size) c[i] = a[i] + b[i]; }

template <typename T> __device__ void elementwise_mult_scalar(uint32_t N, T *a, T b, T *c)
{ _GLASS_RS for (uint32_t i = rank; i < N; i += size) c[i] = a[i]*b; }

template <typename T> __device__ void elementwise_max_scalar(uint32_t N, T *a, T b, T *c)
{ _GLASS_RS for (uint32_t i = rank; i < N; i += size) c[i] = max(a[i], b); }

template <typename T> __device__ void elementwise_min_scalar(uint32_t N, T *a, T b, T *c)
{ _GLASS_RS for (uint32_t i = rank; i < N; i += size) c[i] = min(a[i], b); }

// compile-time size overloads
template <typename T, uint32_t N> __device__ void elementwise_max(T *a, T *b, T *c)
{ _GLASS_RS for (uint32_t i = rank; i < N; i += size) c[i] = max(a[i], b[i]); }

template <typename T, uint32_t N> __device__ void elementwise_min(T *a, T *b, T *c)
{ _GLASS_RS for (uint32_t i = rank; i < N; i += size) c[i] = min(a[i], b[i]); }

template <typename T, uint32_t N> __device__ void elementwise_abs(T *a, T *b)
{ _GLASS_RS for (uint32_t i = rank; i < N; i += size) b[i] = abs(a[i]); }

template <typename T, uint32_t N> __device__ void elementwise_mult(T *a, T *b, T *c)
{ _GLASS_RS for (uint32_t i = rank; i < N; i += size) c[i] = a[i]*b[i]; }

template <typename T, uint32_t N> __device__ void elementwise_sub(T *a, T *b, T *c)
{ _GLASS_RS for (uint32_t i = rank; i < N; i += size) c[i] = a[i] - b[i]; }

template <typename T, uint32_t N> __device__ void elementwise_add(T *a, T *b, T *c)
{ _GLASS_RS for (uint32_t i = rank; i < N; i += size) c[i] = a[i] + b[i]; }

#undef _GLASS_RS
