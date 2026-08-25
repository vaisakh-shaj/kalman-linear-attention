/******************************************************************************
 * cuda_merged_chunk -- kla_chunk_fwd.cuh's six phases, in three.
 *   kla_merged_chunk_fwd.cuh
 *
 * Same shape as kla_chunk_fwd.cuh: one block per (batch, channel), streaming
 * the sequence in tiles of ROWS * ITEMS timesteps, with the ROWS threads of a
 * state column splitting each tile's timesteps between them. Same grid, same
 * carry, same checkpoints. What changes is that there is one scan instead of
 * two, because the leaf composed in phase A is the 3x3 map of kla_merged.cuh,
 * which carries eta alongside lambda.
 *
 *   A  compose this thread's ITEMS 3x3 leaves            -> thread aggregate
 *   B  exclusive scan of those along time, applied to
 *      the tile carry                                    -> my (lambda, eta)
 *   C  walk my timesteps applying *both* recurrences,
 *      and read out over s                               -> y, yvar
 *
 * against A/B/C for lambda and D/E/F for eta. Three things follow, and they are
 * the reason this file exists:
 *
 *   - One block-wide scan instead of two. kla_tile_scan_* is log2(ROWS)
 *     Hillis-Steele rounds with three __syncthreads each; halved. So are the
 *     two kla_tile_broadcast calls, which become one broadcast of a float2.
 *   - The per-thread arrays go away. var_h/alpha_h/r_h in kla_chunk_fwd.cuh
 *     exist only to carry phase C's output to D and F -- 3 * ITEMS registers,
 *     on a kernel whose whole purpose is occupancy.
 *   - Phase C applies both recurrences in the same walk, so it needs alpha_t
 *     only as a local: it has lambda_{t-1} in hand, exactly as the recurrent
 *     kernel does. The composed map never forms alpha at all.
 *
 * The cost is the aggregate: 8 floats (7 live, 1 padding) against a float4 plus
 * a float2, so shared memory for the scan goes from 6 to 8 floats per thread --
 * one buffer of KlaMerged in place of two of float4/float2, which is 8 KB at
 * 256 threads and still far inside the 48 KB budget.
 *
 * Backward: unchanged and unaware, as for every other forward here. This writes
 * the same [B,M,NCK,S] lambda/eta checkpoints at the same KLA_CHUNK stride with
 * the same convention -- the value *entering* step t -- and kla_scan_bwd.cuh
 * replays a scalar recurrence from them. It never sees a composed map of any
 * size, which is why merging the forward cannot touch it.
 *
 * Unlike kla_chunk_fwd.cuh, both checkpoints are written in the same phase.
 * There they could not be: eta did not exist until the affine phases had run.
 *
 * Transcribed from kernels/mps/merged_chunk_kla_scan.metal.
 ******************************************************************************/
#pragma once

#include "kla_merged.cuh"

// ----------------------------------------------------------------- tile scan
//
// Exclusive scan across the ROWS threads holding one state column, so the
// result maps the tile's incoming carry to this thread's starting state. The
// one scan this kernel runs, where kla_chunk_fwd.cuh runs this and an affine
// one. Every thread must reach every barrier, so nothing here is predicated on
// s < d_state or on the sequence bound.

template <int BLOCK_S, int ROWS>
__device__ __forceinline__ KlaMerged kla_tile_scan_merged(KlaMerged mine,
                                                          KlaMerged *smem, int tid,
                                                          int ty) {
    if (ROWS == 1) return kla_mrg_identity();
    smem[tid] = mine;
    __syncthreads();
#pragma unroll
    for (int off = 1; off < ROWS; off <<= 1) {
        const KlaMerged prev =
            (ty >= off) ? smem[tid - off * BLOCK_S] : kla_mrg_identity();
        __syncthreads();
        if (ty >= off) mine = kla_mrg_compose(prev, mine);
        smem[tid] = mine;
        __syncthreads();
    }
    const KlaMerged excl = (ty > 0) ? smem[tid - BLOCK_S] : kla_mrg_identity();
    __syncthreads();
    return excl;
}

// Broadcast the last thread-row's (lambda, eta) to the whole state column. One
// call per tile, where the two-scan kernel needs one after each of its walks.
template <int ROWS>
__device__ __forceinline__ float2 kla_tile_broadcast2(float2 v, float2 *smem,
                                                      int sx, int ty) {
    __syncthreads();
    if (ty == ROWS - 1) smem[sx] = v;
    __syncthreads();
    const float2 out = smem[sx];
    __syncthreads();
    return out;
}

template <int BLOCK_S, int ROWS, int ITEMS>
__global__ void kla_merged_chunk_fwd_kernel(
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
    __shared__ KlaMerged mrg_s[BLOCK_S * ROWS];
    __shared__ float2 red_s[BLOCK_S * ROWS];
    __shared__ float2 bcast_s[BLOCK_S];

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
    const float inv_a = 1.0f / (a_s + copysignf(KLA_EPS, a_s));

    float2 carry = make_float2(active ? lam0[kla_bms(b, m, sx, M, S)] : 1.0f,
                               active ? eta0[kla_bms(b, m, sx, M, S)] : 0.0f);

    const int tile_len = ROWS * ITEMS;
    const int t_base = ty * ITEMS;

    for (int tile = 0; tile < L; tile += tile_len) {
        // -- A: compose my leaves. Steps past the end of the sequence, and the
        // padding lanes at s >= d_state, contribute the identity.
        KlaMerged agg = kla_mrg_identity();
#pragma unroll
        for (int i = 0; i < ITEMS; ++i) {
            const int t = tile + t_base + i;
            if (t < L && active) {
                const float si_t = si[kla_blm(b, t, m, L, M)];
                const float msi_t = msi[kla_blm(b, t, m, L, M)];
                const float k_t = k[kla_bls(b, t, sx, L, S)];
                const float phi = fmaxf(si_t * k_t * k_t, KLA_EPS);
                agg = kla_mrg_compose(
                    agg, kla_mrg_leaf(phi, msi_t * k_t, p_s, inv_a2, inv_a));
            }
        }

        // -- B: where my slice starts. One scan, and it carries eta with it, so
        // there is nothing left to resolve after this.
        const KlaMerged pref =
            kla_tile_scan_merged<BLOCK_S, ROWS>(agg, mrg_s, tid, ty);
        const float2 st = kla_mrg_apply(pref, carry.x, carry.y);
        float lam = st.x, eta = st.y;

        // -- C: walk my timesteps applying both recurrences, and read out. The
        // gain is a local here: this walk has lambda_{t-1} in hand, which is
        // the property the composed map exists to hand back.
#pragma unroll
        for (int i = 0; i < ITEMS; ++i) {
            const int t = tile + t_base + i;
            float var = 0.0f;
            if (t < L && active) {
                const float si_t = si[kla_blm(b, t, m, L, M)];
                const float msi_t = msi[kla_blm(b, t, m, L, M)];
                const float k_t = k[kla_bls(b, t, sx, L, S)];
                const float phi = fmaxf(si_t * k_t * k_t, KLA_EPS);
                const float lam_prev = lam;
                // The values *entering* step t are what kla_scan_bwd resumes
                // from, so both stores precede both updates.
                if (store_ck != 0 && (t % KLA_CHUNK) == 0) {
                    const int ck = kla_ck(b, m, t / KLA_CHUNK, sx, M, NCK, S);
                    lam_ck[ck] = lam_prev;
                    eta_ck[ck] = eta;
                }
                const float den = kla_den(a2, p_s, lam_prev);
                lam = kla_lambda_step(lam_prev, phi, den);
                eta = (a_s / den) * eta + msi_t * k_t;
                var = 1.0f / fmaxf(lam, KLA_EPS);
            }
            const float mean_f = eta * var;
            const float var_o = (prior != 0) ? (a2 * var + p_s) : var;
            const float mean_o = (prior != 0) ? (a_s * mean_f) : mean_f;
            // Lanes with nothing to contribute carry q_t = 0, so the reduction
            // stays collective without them affecting the sum.
            const float q_t =
                (t < L && active) ? qw[kla_bls(b, t, sx, L, S)] : 0.0f;
            const float2 out = kla_sum_over_states<BLOCK_S>(
                make_float2(mean_o * q_t, var_o * q_t * q_t), red_s, sx, ty);
            if (sx == 0 && t < L) {
                y[kla_blm(b, t, m, L, M)] = out.x;
                yvar[kla_blm(b, t, m, L, M)] = out.y;
            }
        }
        // The last thread holds the state at the tile boundary (or at L, if the
        // sequence ended inside this tile -- later threads then never updated).
        carry = kla_tile_broadcast2<ROWS>(make_float2(lam, eta), bcast_s, sx, ty);
    }

    if (active && ty == 0) {
        lam_fin[kla_bms(b, m, sx, M, S)] = carry.x;
        eta_fin[kla_bms(b, m, sx, M, S)] = carry.y;
    }
}
