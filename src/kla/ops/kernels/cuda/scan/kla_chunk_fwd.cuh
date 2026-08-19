/******************************************************************************
 * cuda_chunk -- the forward, time as a *parallel* axis.
 *   kla_chunk_fwd.cuh
 *
 * For the case cuda_recurrent cannot fill: batch-1 prefill, where B*M*S alone
 * leaves the GPU short of threads.
 *
 * One block owns one (batch, channel) pair and streams the sequence in tiles of
 * ROWS * ITEMS timesteps, carrying (lambda, eta) across tiles. Inside a tile
 * the ROWS threads of a state column split the timesteps between them, so the
 * grid is ROWS times wider than the serial kernel's.
 *
 * Cost. Parallelizing over time forces *composition*: a thread cannot apply the
 * Moebius map without already knowing the running lambda, which is the serial
 * dependency itself. So each element is composed once (phase A) and applied
 * once (phase C) rather than applied once -- about 4x the arithmetic of
 * kla_recurrent_fwd.cuh, bought in exchange for ROWS times the parallelism.
 * Walking ITEMS steps serially per thread is what keeps it at 4x rather than
 * the log(tile) composes a pure Hillis-Steele scan over timesteps costs.
 *
 * Six phases per tile:
 *   A  compose this thread's ITEMS Moebius leaves     -> thread aggregate
 *   B  exclusive scan of those aggregates along time  -> my starting lambda
 *   C  walk my timesteps applying the map             -> lambda, alpha, r
 *   D  compose my affine leaves (alpha, r)            -> thread aggregate
 *   E  exclusive scan of those along time             -> my starting eta
 *   F  walk applying the affine map, read out over s  -> y, yvar
 *
 * D and E cannot be folded into A and B: alpha_t reads lambda_{t-1}, so the
 * affine leaves do not exist until phase C has produced lambda.
 *
 * Forward only, and that is the whole design rather than a gap: it writes the
 * same [B,M,NCK,S] checkpoints at the same stride as kla_recurrent_fwd.cuh, and
 * kla_scan_bwd.cuh replays from them. The adjoint walks the serial state lanes
 * either way, so composing here buys the backward nothing and costs it nothing.
 *
 * Shared memory. Two scan buffers, the read-out buffer and a broadcast slot:
 * at 256 threads that is ~8 KB, comfortably inside the 48 KB a consumer
 * Ampere/Ada part gives without an opt-in carveout.
 ******************************************************************************/
#pragma once

#include "kla_scan_common.cuh"

// ----------------------------------------------------------------- tile scans
//
// Exclusive scan across the ROWS threads holding one state column, so the
// result maps the tile's incoming carry to this thread's starting value.
// Hillis-Steele: log2(ROWS) rounds over ROWS elements, negligible next to the
// ITEMS serial steps each element already cost. Every thread must reach every
// barrier, so nothing here is predicated on s < d_state or on the sequence
// bound.

template <int BLOCK_S, int ROWS>
__device__ __forceinline__ float4 kla_tile_scan_mobius(float4 mine, float4 *smem,
                                                       int tid, int ty) {
    if (ROWS == 1) return kla_mob_identity();
    smem[tid] = mine;
    __syncthreads();
#pragma unroll
    for (int off = 1; off < ROWS; off <<= 1) {
        const float4 prev =
            (ty >= off) ? smem[tid - off * BLOCK_S] : kla_mob_identity();
        __syncthreads();
        if (ty >= off) mine = kla_mob_compose(prev, mine);
        smem[tid] = mine;
        __syncthreads();
    }
    const float4 excl = (ty > 0) ? smem[tid - BLOCK_S] : kla_mob_identity();
    __syncthreads();
    return excl;
}

template <int BLOCK_S, int ROWS>
__device__ __forceinline__ float2 kla_tile_scan_affine(float2 mine, float2 *smem,
                                                       int tid, int ty) {
    if (ROWS == 1) return kla_aff_identity();
    smem[tid] = mine;
    __syncthreads();
#pragma unroll
    for (int off = 1; off < ROWS; off <<= 1) {
        const float2 prev =
            (ty >= off) ? smem[tid - off * BLOCK_S] : kla_aff_identity();
        __syncthreads();
        if (ty >= off) mine = kla_aff_compose(prev, mine);
        smem[tid] = mine;
        __syncthreads();
    }
    const float2 excl = (ty > 0) ? smem[tid - BLOCK_S] : kla_aff_identity();
    __syncthreads();
    return excl;
}

// Broadcast the last thread-row's value to the whole state column.
template <int ROWS>
__device__ __forceinline__ float kla_tile_broadcast(float v, float *smem, int sx,
                                                    int ty) {
    __syncthreads();
    if (ty == ROWS - 1) smem[sx] = v;
    __syncthreads();
    const float out = smem[sx];
    __syncthreads();
    return out;
}

template <int BLOCK_S, int ROWS, int ITEMS>
__global__ void kla_chunk_fwd_kernel(
    float *__restrict__ y,        // out [B, L, M]
    float *__restrict__ yvar,     // out [B, L, M]
    float *__restrict__ lam_fin,  // out [B, M, S]
    float *__restrict__ eta_fin,  // out [B, M, S]
    float *__restrict__ lam_ck,   // out [B, M, NCK, S] (see store_ck)
    float *__restrict__ eta_ck,   // out [B, M, NCK, S]
    const float *__restrict__ msi, const float *__restrict__ si,
    const float *__restrict__ k, const float *__restrict__ qw,
    const float *__restrict__ a, const float *__restrict__ p,
    const float *__restrict__ lam0, const float *__restrict__ eta0,
    int L, int M, int S, int NCK, int store_ck, int prior) {
    __shared__ float4 mob_s[BLOCK_S * ROWS];
    __shared__ float2 aff_s[BLOCK_S * ROWS];
    __shared__ float2 red_s[BLOCK_S * ROWS];
    __shared__ float bcast_s[BLOCK_S];

    const int sx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * BLOCK_S + sx;
    const int m = blockIdx.x;
    const int b = blockIdx.y;
    const bool active = (sx < S);

    const float a_s = active ? a[m * S + sx] : 1.0f;
    const float p_s = active ? p[m * S + sx] : 0.0f;
    const float a2 = fmaxf(a_s * a_s, KLA_EPS);
    const float inv_a2 = 1.0f / a2;

    float carry_lam = active ? lam0[kla_bms(b, m, sx, M, S)] : 1.0f;
    float carry_eta = active ? eta0[kla_bms(b, m, sx, M, S)] : 0.0f;

    // This thread's slice of each tile, and what phase C leaves for phase F.
    float var_h[ITEMS], alpha_h[ITEMS], r_h[ITEMS];

    const int tile_len = ROWS * ITEMS;
    const int t_base = ty * ITEMS;

    for (int tile = 0; tile < L; tile += tile_len) {
        // -- A: compose my leaves. Steps past the end of the sequence, and the
        // padding lanes at s >= d_state, contribute the identity.
        float4 agg = kla_mob_identity();
#pragma unroll
        for (int i = 0; i < ITEMS; ++i) {
            const int t = tile + t_base + i;
            if (t < L && active) {
                const float si_t = si[kla_blm(b, t, m, L, M)];
                const float k_t = k[kla_bls(b, t, sx, L, S)];
                const float phi = fmaxf(si_t * k_t * k_t, KLA_EPS);
                agg = kla_mob_compose(
                    agg, make_float4((1.0f + p_s * phi) * inv_a2, phi,
                                     p_s * inv_a2, 1.0f));
            }
        }

        // -- B/C: where my slice starts, then walk it applying the map.
        const float4 pref = kla_tile_scan_mobius<BLOCK_S, ROWS>(agg, mob_s, tid, ty);
        float lam = kla_mob_apply(pref, carry_lam);
#pragma unroll
        for (int i = 0; i < ITEMS; ++i) {
            const int t = tile + t_base + i;
            float alpha = 1.0f, r_t = 0.0f, var = 0.0f;
            if (t < L && active) {
                const float si_t = si[kla_blm(b, t, m, L, M)];
                const float msi_t = msi[kla_blm(b, t, m, L, M)];
                const float k_t = k[kla_bls(b, t, sx, L, S)];
                const float phi = fmaxf(si_t * k_t * k_t, KLA_EPS);
                const float lam_prev = lam;
                // The value *entering* step t is what kla_scan_bwd resumes
                // from, so the store precedes the update -- as it does in
                // kla_recurrent_fwd.cuh, at the same stride and the same index.
                if (store_ck != 0 && (t % KLA_CHUNK) == 0) {
                    lam_ck[kla_ck(b, m, t / KLA_CHUNK, sx, M, NCK, S)] = lam_prev;
                }
                const float den = kla_den(a2, p_s, lam_prev);
                lam = kla_lambda_step(lam_prev, phi, den);
                // The gain reads lambda_{t-1}, so it lags the update by one.
                alpha = a_s / den;
                r_t = msi_t * k_t;
                var = 1.0f / fmaxf(lam, KLA_EPS);
            }
            var_h[i] = var;
            alpha_h[i] = alpha;
            r_h[i] = r_t;
        }
        // The last thread holds lambda at the tile boundary (or at L, if the
        // sequence ended inside this tile -- later threads then never updated).
        carry_lam = kla_tile_broadcast<ROWS>(lam, bcast_s, sx, ty);

        // -- D/E: the same two steps for the affine recurrence, which could not
        // run earlier because its leaves need the lambda phase C just produced.
        float2 aagg = kla_aff_identity();
#pragma unroll
        for (int i = 0; i < ITEMS; ++i) {
            aagg = kla_aff_compose(aagg, make_float2(alpha_h[i], r_h[i]));
        }
        const float2 apref =
            kla_tile_scan_affine<BLOCK_S, ROWS>(aagg, aff_s, tid, ty);

        // -- F: walk applying the affine map, and read out along the state axis.
        float eta = apref.x * carry_eta + apref.y;
#pragma unroll
        for (int i = 0; i < ITEMS; ++i) {
            const int t = tile + t_base + i;
            if (store_ck != 0 && active && t < L && (t % KLA_CHUNK) == 0) {
                eta_ck[kla_ck(b, m, t / KLA_CHUNK, sx, M, NCK, S)] = eta;
            }
            eta = alpha_h[i] * eta + r_h[i];
            const float var_f = var_h[i];
            const float mean_f = eta * var_f;
            const float var = (prior != 0) ? (a2 * var_f + p_s) : var_f;
            const float mean = (prior != 0) ? (a_s * mean_f) : mean_f;
            // Lanes with nothing to contribute carry q_t = 0, so the reduction
            // stays collective without them affecting the sum.
            const float q_t = (t < L && active) ? qw[kla_bls(b, t, sx, L, S)] : 0.0f;
            const float2 out = kla_sum_over_states<BLOCK_S>(
                make_float2(mean * q_t, var * q_t * q_t), red_s, sx, ty);
            if (sx == 0 && t < L) {
                y[kla_blm(b, t, m, L, M)] = out.x;
                yvar[kla_blm(b, t, m, L, M)] = out.y;
            }
        }
        carry_eta = kla_tile_broadcast<ROWS>(eta, bcast_s, sx, ty);
    }

    // carry_lam/carry_eta are broadcast, so every row holds them; pick one.
    if (active && ty == 0) {
        lam_fin[kla_bms(b, m, sx, M, S)] = carry_lam;
        eta_fin[kla_bms(b, m, sx, M, S)] = carry_eta;
    }
}
