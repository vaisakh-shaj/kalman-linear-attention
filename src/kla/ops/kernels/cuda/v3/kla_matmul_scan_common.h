/******************************************************************************
 * KLA Matmul Scan — Common Utilities
 *   kla_matmul_scan_common.h
 *
 * Load/store helpers, type conversion, constexpr utilities.
 * Adapted from Mamba-1's selective_scan_common.h
 ******************************************************************************/
#pragma once

#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <algorithm>

#ifndef MAX_DSTATE
#define MAX_DSTATE 64
#endif

#ifndef KLA_EPS
#define KLA_EPS 1e-12f
#endif

// Arithmetic operators for CUDA vector types (needed by CUB BlockReduce::Sum)
inline __device__ float2 operator+(const float2 &a, const float2 &b) {
    return {a.x + b.x, a.y + b.y};
}

inline __device__ float4 operator+(const float4 &a, const float4 &b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}

// Constexpr utilities (matches Mamba's pattern)
constexpr size_t custom_max(std::initializer_list<size_t> ilist) {
    return std::max(ilist);
}

constexpr int constexpr_min(int a, int b) { return a < b ? a : b; }

// Vectorized type mapping
template<int kNBytes> struct BytesToType {};
template<> struct BytesToType<4>  { using Type = float; };
template<> struct BytesToType<8>  { using Type = float2; };
template<> struct BytesToType<16> { using Type = float4; };

// ============================================================================
// Cooperative load via CUB BlockLoad
// ============================================================================
template<typename Ktraits, typename T>
__device__ __forceinline__ void load_input(
    const T *__restrict__ input,
    T (&vals)[Ktraits::kNItems],
    typename Ktraits::BlockLoadT::TempStorage &smem_load,
    int seqlen_remaining
) {
    if constexpr (Ktraits::kIsEvenLen) {
        typename Ktraits::BlockLoadT(smem_load).Load(
            reinterpret_cast<const T *>(input), vals);
    } else {
        typename Ktraits::BlockLoadT(smem_load).Load(
            reinterpret_cast<const T *>(input), vals, seqlen_remaining, T(0));
    }
}

// ============================================================================
// Cooperative store via CUB BlockStore
// ============================================================================
template<typename Ktraits, typename T>
__device__ __forceinline__ void store_output(
    T *__restrict__ output,
    T (&vals)[Ktraits::kNItems],
    typename Ktraits::BlockStoreT::TempStorage &smem_store,
    int seqlen_remaining
) {
    if constexpr (Ktraits::kIsEvenLen) {
        typename Ktraits::BlockStoreT(smem_store).Store(
            reinterpret_cast<T *>(output), vals);
    } else {
        typename Ktraits::BlockStoreT(smem_store).Store(
            reinterpret_cast<T *>(output), vals, seqlen_remaining);
    }
}
