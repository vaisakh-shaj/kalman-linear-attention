/******************************************************************************
 * Shared device pieces for the exact CUDA KLA scans.
 *   kla_scan_common.cuh
 *
 * Layout. One thread owns one (b, m, s) triple. A block is
 * [BLOCK_S, ROWS]: BLOCK_S = next_pow2(d_state) lanes covering every state of
 * one channel, ROWS channels stacked on top, aiming at 256 threads. The
 * read-out sums over s, so all of a channel's states have to sit in one block
 * -- which is where MAX_DSTATE comes from.
 *
 * BLOCK_S <= 32 keeps a row inside one warp, so the read-out reduction is a
 * __shfl_xor_sync butterfly with no shared memory at all. That is the path
 * every real config takes (d_state is 16 by default and rarely past 32), and it
 * is the one to optimize. Past 32 a row spans two warps and the reduction goes
 * through shared memory instead.
 *
 * Portability. Nothing here is newer than sm_80: no wgmma, no TMA, no clusters,
 * no distributed shared memory. The scan is float32 scalar FMA work, so no
 * tensor-core path applies in the first place. Shared memory stays under the
 * 48 KB a consumer Ampere/Ada part gives without an opt-in carveout -- sizing
 * for an H200's 228 KB/SM would simply fail to launch on a 4090.
 ******************************************************************************/
#pragma once

#include <cuda_runtime.h>

#define KLA_EPS 1e-12f
#define KLA_MAX_DSTATE 64
#define KLA_TG_THREADS 256
#define KLA_CHUNK 16  // checkpoint stride; also the replay buffers' depth
#define KLA_ITEMS 8   // timesteps each thread of cuda_chunk walks serially
//
// KLA_CHUNK and KLA_ITEMS are independent and must stay so. One is where the
// backward resumes; the other is how deep one thread of the chunk forward
// walks. They do not divide each other, and giving them a single name is
// exactly the bug this comment exists to prevent -- the Metal sources carried
// it for a while.

// ---------------------------------------------------------------- the step
//
// lambda_t = lambda_{t-1}/den_t + phi_t,  den_t = a^2 + p.lambda_{t-1}
// alpha_t  = a/den_t
// eta_t    = alpha_t.eta_{t-1} + r_t
//
// Applied, never composed. That is what leaves the adjoint elementary:
// d lambda_t / d lambda_{t-1} = a^2/den_t^2, a scalar, where differentiating a
// trace-normalized prefix product would need a 4x4 Jacobian chain.

__device__ __forceinline__ float kla_den(float a2, float p, float lam_prev) {
    return fmaxf(a2 + p * lam_prev, KLA_EPS);
}

__device__ __forceinline__ float kla_lambda_step(float lam_prev, float phi, float den) {
    return lam_prev / den + phi;
}

// ------------------------------------------------------- 2x2 Moebius algebra
//
// Here rather than beside the one kernel that composes, because cuda_chunk and
// cuda_pscan both compose and must compose identically -- two copies of a trace
// normalization is two things to drift.
//
// Same representation and the same trace normalization as the torch reference
// (kla.ops.kla_ops._mobius_combine_tracenorm) and the triton kernels:
// float4(A, B, C, D) is [[A, B], [C, D]], acting as lambda -> (A.l+B)/(C.l+D).

__device__ __forceinline__ float4 kla_mob_identity() {
    return make_float4(1.0f, 0.0f, 0.0f, 1.0f);
}

// The map "R after L": the matrix product R*L, divided by its own trace. The
// quotient is invariant under that rescaling, so it only buys exponent range.
__device__ __forceinline__ float4 kla_mob_compose(float4 L, float4 R) {
    const float a = R.x * L.x + R.y * L.z;
    const float b = R.x * L.y + R.y * L.w;
    const float c = R.z * L.x + R.w * L.z;
    const float d = R.z * L.y + R.w * L.w;
    const float inv = 1.0f / fmaxf(a + d, KLA_EPS);
    return make_float4(a * inv, b * inv, c * inv, d * inv);
}

__device__ __forceinline__ float kla_mob_apply(float4 P, float lam) {
    return (P.x * lam + P.y) / fmaxf(P.z * lam + P.w, KLA_EPS);
}

__device__ __forceinline__ float2 kla_aff_identity() {
    return make_float2(1.0f, 0.0f);
}

__device__ __forceinline__ float2 kla_aff_compose(float2 L, float2 R) {
    return make_float2(R.x * L.x, R.x * L.y + R.y);
}

// ------------------------------------------------------------- reductions
//
// Sum a float2 across the BLOCK_S lanes holding one channel. Every thread must
// call it, padding lanes included: they carry neutral values and take part, so
// the butterfly stays collective. Only the storing lane reads the result.

template <int BLOCK_S>
__device__ __forceinline__ float2 kla_sum_over_states(float2 v, float2 *smem,
                                                      int sx, int row) {
    if (BLOCK_S <= 32) {
#pragma unroll
        for (int off = BLOCK_S >> 1; off > 0; off >>= 1) {
            v.x += __shfl_xor_sync(0xffffffffu, v.x, off, BLOCK_S);
            v.y += __shfl_xor_sync(0xffffffffu, v.y, off, BLOCK_S);
        }
        return v;
    }
    // A row wider than a warp cannot shuffle across itself.
    float2 *r = smem + row * BLOCK_S;
    r[sx] = v;
    __syncthreads();
#pragma unroll
    for (int off = BLOCK_S >> 1; off > 0; off >>= 1) {
        if (sx < off) {
            r[sx].x += r[sx + off].x;
            r[sx].y += r[sx + off].y;
        }
        __syncthreads();
    }
    float2 out = r[0];
    __syncthreads();
    return out;
}

// Indexing. msi/si and y/yvar are [B, L, M]; k/q are [B, L, S]; a/p are [M, S];
// the state and the checkpoints are [B, M, S] and [B, M, NCK, S].
__device__ __forceinline__ int kla_blm(int b, int t, int m, int L, int M) {
    return (b * L + t) * M + m;
}
__device__ __forceinline__ int kla_bls(int b, int t, int s, int L, int S) {
    return (b * L + t) * S + s;
}
__device__ __forceinline__ int kla_bms(int b, int m, int s, int M, int S) {
    return (b * M + m) * S + s;
}
__device__ __forceinline__ int kla_ck(int b, int m, int c, int s, int M, int NCK,
                                      int S) {
    return ((b * M + m) * NCK + c) * S + s;
}
