/******************************************************************************
 * KLA Metal kernels — shared prelude
 *   kla_common.metal
 *
 * Prepended to every KLA shader by :mod:`kla.ops.kernels.mps._shaders`, which
 * also emits the specialization defines this file expects:
 *
 *   KLA_BLOCK_S   lanes per channel row = next_pow2(d_state)
 *   KLA_ROWS      channel rows per threadgroup
 *   KLA_CHUNK     timesteps recomputed per checkpoint in the fused backward
 *
 * The thread mapping is the same everywhere: one thread owns one
 * ``(batch b, channel m, state s)`` triple and walks the sequence serially,
 * with ``thread_position_in_grid`` = ``(s, m, b)``. A threadgroup is
 * ``[KLA_BLOCK_S, KLA_ROWS]``, so the ``KLA_BLOCK_S`` lanes of one row hold
 * every state of one channel — which is what makes the read-out sum over ``s``
 * a lane reduction rather than a second kernel.
 *
 * Two reductions follow from that layout, and both are used by the fused
 * kernels only (the composed scans are pure elementwise-in-(b,m,s)):
 *
 *   kla_sum_over_states    along x — the read-out y = sum_s q.mean, and the
 *                          d(v.Lambda^v) / d(Lambda^v) gradients.
 *   kla_sum_over_channels  along y — the d(k) / d(q) gradients, which contract
 *                          over m rather than s.
 *
 * A row of KLA_BLOCK_S lanes sits inside one SIMD-group whenever
 * KLA_BLOCK_S <= 32 (threads are assigned to SIMD-groups in
 * thread_index_in_threadgroup order, and the row length divides 32), so the
 * x reduction is a barrier-free shuffle butterfly there. Past 32 lanes the
 * generator pins KLA_ROWS to 1 and the reduction goes through threadgroup
 * memory instead.
 *
 * Both helpers must be called by *every* thread in the group, including the
 * padding threads at s >= d_state or m >= d_inner: the shuffles need a full
 * SIMD-group and the threadgroup path has barriers in it. So no kernel here
 * returns early — out-of-range threads load neutral values and skip only their
 * stores.
 ******************************************************************************/

#include <metal_stdlib>
using namespace metal;

// Matches EPS in kla.ops.kla_ops, and KLA_EPS in the CUDA kernels.
constant constexpr float KLA_EPS = 1e-12f;

#define KLA_TG_THREADS (KLA_BLOCK_S * KLA_ROWS)

// ---------------------------------------------------------------- addressing

inline uint kla_blms(uint b, uint t, uint m, uint s, uint L, uint M, uint S) {
    return ((b * L + t) * M + m) * S + s;  // [B, L, M, S]
}

inline uint kla_blm(uint b, uint t, uint m, uint L, uint M) {
    return (b * L + t) * M + m;  // [B, L, M]
}

inline uint kla_bls(uint b, uint t, uint s, uint L, uint S) {
    return (b * L + t) * S + s;  // [B, L, S]
}

inline uint kla_bms(uint b, uint m, uint s, uint M, uint S) {
    return (b * M + m) * S + s;  // [B, M, S]
}

// ---------------------------------------------------------------- reductions

// Sum across the KLA_BLOCK_S lanes of one channel row (the state axis).
// Returns the total in every lane of the row.
inline float2 kla_sum_over_states(float2 v,
                                  threadgroup float2 *scratch,
                                  uint tidx,
                                  uint sx) {
#if KLA_BLOCK_S == 1
    return v;
#elif KLA_BLOCK_S <= 32
    // The row is SIMD-group resident: a shuffle butterfly leaves the full sum
    // in every lane with no threadgroup memory and no barrier.
    for (uint off = KLA_BLOCK_S / 2; off > 0; off >>= 1) {
        v.x += simd_shuffle_xor(v.x, ushort(off));
        v.y += simd_shuffle_xor(v.y, ushort(off));
    }
    return v;
#else
    // Wider than a SIMD-group; KLA_ROWS is 1 here, so the row is the group.
    const uint base = tidx - sx;
    scratch[tidx] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = KLA_BLOCK_S / 2; off > 0; off >>= 1) {
        float2 acc = (sx < off) ? scratch[tidx] + scratch[tidx + off] : float2(0.0f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sx < off) { scratch[tidx] = acc; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float2 total = scratch[base];
    threadgroup_barrier(mem_flags::mem_threadgroup);  // scratch reusable again
    return total;
#endif
}

// Sum across the KLA_ROWS lanes holding the same state (the channel axis).
// Returns the total in every lane of the column.
inline float2 kla_sum_over_channels(float2 v,
                                    threadgroup float2 *scratch,
                                    uint tidx,
                                    uint my) {
#if KLA_ROWS == 1
    return v;
#else
    const uint col = tidx - my * KLA_BLOCK_S;
    scratch[tidx] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = KLA_ROWS / 2; off > 0; off >>= 1) {
        float2 acc = (my < off) ? scratch[tidx] + scratch[tidx + off * KLA_BLOCK_S]
                                : float2(0.0f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (my < off) { scratch[tidx] = acc; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float2 total = scratch[col];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
#endif
}

// ------------------------------------------------------------- filter algebra
//
// One step of the precision recurrence, shared by the fused forward and the
// fused backward's recompute so the two cannot drift.
//
//   predict  lambda^- = lambda' / (a^2 + p.lambda')
//   update   lambda   = lambda^- + phi
//
// which is the Moebius map lambda = (A.lambda' + B)/(C.lambda' + D) with
// A = (1 + p.phi)/a^2, B = phi, C = p/a^2, D = 1. Applying the map directly is
// what the lane-per-state layout buys: the CUDA and triton kernels have to
// *compose* the 2x2 matrices (and trace-normalize to keep the entries O(1))
// because their threads span time, whereas here time is the serial axis.
inline float kla_lambda_step(float lam_prev, float phi, float p_s, float inv_a2,
                             thread float &A, thread float &C, thread float &den) {
    A = (1.0f + p_s * phi) * inv_a2;
    C = p_s * inv_a2;
    den = fmax(C * lam_prev + 1.0f, KLA_EPS);
    return (A * lam_prev + phi) / den;
}
