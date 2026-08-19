/******************************************************************************
 * cuda_pscan -- the forward as a reduce-then-scan, with no serial carry.
 *   kla_pscan_fwd.cuh
 *
 * kla_recurrent_fwd.cuh walks time serially. kla_chunk_fwd.cuh splits time into
 * tiles and walks the *tiles* serially, one block carrying (lambda, eta) from
 * each tile to the next. This file removes that last serial axis: the sequence
 * is cut into NCK chunks of KLA_CHUNK steps, every chunk is reduced
 * independently, and the cross-chunk dependency is resolved by a parallel scan
 * over the NCK aggregates rather than by anyone waiting for their neighbour.
 *
 * The trade is the usual one. No chunk waits, so the whole sequence is in
 * flight at once and the depth is log(NCK) instead of NCK -- but the aggregates
 * have to be materialized in global memory ([B,M,NCK,S] of them, two scans'
 * worth), and every element is touched about three times instead of once. This
 * is the schedule for a short-and-wide launch that the other two starve on, not
 * the one to reach for by default.
 *
 * It stays "fused" in the sense this repo uses the word: the intermediates are
 * per *chunk*, [B,M,NCK,S], never the per-timestep [B,L,M,S] the unfused torch
 * path builds. At KLA_CHUNK = 16 that is a 16x smaller footprint, and the two
 * lambda/eta arrays it does write are the backward's checkpoints -- which this
 * schedule produces for free, since "the state entering chunk c" is exactly
 * what the scan over aggregates computes.
 *
 * Five kernels, two reduce-scan-apply rounds:
 *
 *   1 mob_reduce   compose each chunk's KLA_CHUNK Moebius leaves   -> mob[c]
 *   2 mob_step     one doubling round over the NCK aggregates      (log2 NCK x)
 *   3 aff_reduce   lambda entering chunk c = apply(mob[c-1], lam0) -> lam_ck
 *                  then walk the chunk to build its affine leaves  -> aff[c]
 *   4 aff_step     one doubling round over those                   (log2 NCK x)
 *   5 apply        eta entering chunk c = aff[c-1](eta0)           -> eta_ck
 *                  then walk the chunk, recomputing lambda, and read out
 *
 * Round two cannot be folded into round one: the affine leaf alpha_t reads
 * lambda_{t-1}, so those leaves do not exist until the Moebius scan has
 * produced lambda. Kernel 5 recomputes lambda rather than reading it back,
 * which is what keeps the intermediates per-chunk instead of per-timestep.
 *
 * The scan over aggregates is Hillis-Steele across kernel launches, ping-ponging
 * two buffers. It is work-inefficient by a log factor, but it runs on an axis
 * KLA_CHUNK times shorter than the sequence, and running it across launches
 * rather than inside one block means nothing bounds NCK -- a scan that had to
 * fit one block would need a second level to cover long sequences. Launch-level
 * ping-pong also needs no grid-wide sync, so there is no cooperative-launch
 * requirement and nothing here is newer than sm_80.
 *
 * Backward: kla_scan_bwd.cuh, unchanged and unaware. lam_ck / eta_ck carry the
 * same layout, the same stride and the same convention (the value *entering*
 * step t, for t a multiple of KLA_CHUNK) as both other forwards write.
 *
 * This is a transcription of pscan_kla_scan.metal, which runs and is checked
 * against the torch reference on an Apple GPU -- forward and gradients, over
 * chunk counts of 1, 2, 3, 5 and 17.
 ******************************************************************************/
#pragma once

#include "kla_scan_common.cuh"

// The first four kernels are pure elementwise over (b, m, c, s), so they take a
// flat grid and decode the index. Only the read-out kernel reduces, and only it
// therefore has a block shape.
__device__ __forceinline__ void kla_pscan_decode(int gid, int M, int NCK, int S,
                                                 int &b, int &m, int &c, int &s) {
    s = gid % S;
    int rest = gid / S;
    c = rest % NCK;
    rest /= NCK;
    m = rest % M;
    b = rest / M;
}

// ------------------------------------------------------- 1: reduce the chunks

__global__ void kla_pscan_mob_reduce_kernel(float4 *__restrict__ mob,
                                            const float *__restrict__ si,
                                            const float *__restrict__ k,
                                            const float *__restrict__ a,
                                            const float *__restrict__ p, int L,
                                            int M, int S, int NCK, int total) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;
    int b, m, c, s;
    kla_pscan_decode(gid, M, NCK, S, b, m, c, s);

    const float a_s = a[m * S + s];
    const float p_s = p[m * S + s];
    const float inv_a2 = 1.0f / fmaxf(a_s * a_s, KLA_EPS);

    // Steps past the end of the sequence contribute the identity, so a partial
    // final chunk composes to exactly what its live prefix does.
    float4 agg = kla_mob_identity();
#pragma unroll
    for (int i = 0; i < KLA_CHUNK; ++i) {
        const int t = c * KLA_CHUNK + i;
        if (t < L) {
            const float si_t = si[kla_blm(b, t, m, L, M)];
            const float k_t = k[kla_bls(b, t, s, L, S)];
            const float phi = fmaxf(si_t * k_t * k_t, KLA_EPS);
            agg = kla_mob_compose(
                agg, make_float4((1.0f + p_s * phi) * inv_a2, phi, p_s * inv_a2,
                                 1.0f));
        }
    }
    mob[gid] = agg;
}

// ------------------------------------------- 2/4: one doubling round each
//
// dst[c] = compose(src[c - off], src[c]) for c >= off, src[c] otherwise. Run
// with off = 1, 2, 4, ... < NCK and the buffers swapped, this leaves the
// inclusive prefix in the last destination. The chunk axis has stride S in the
// flat index, so the neighbour is gid - off*S.

__global__ void kla_pscan_mob_step_kernel(float4 *__restrict__ dst,
                                          const float4 *__restrict__ src, int S,
                                          int NCK, int off, int total) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;
    const int c = (gid / S) % NCK;
    dst[gid] = (c >= off) ? kla_mob_compose(src[gid - off * S], src[gid]) : src[gid];
}

__global__ void kla_pscan_aff_step_kernel(float2 *__restrict__ dst,
                                          const float2 *__restrict__ src, int S,
                                          int NCK, int off, int total) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;
    const int c = (gid / S) % NCK;
    dst[gid] = (c >= off) ? kla_aff_compose(src[gid - off * S], src[gid]) : src[gid];
}

// -------------------------------------- 3: seed lambda, then reduce the affine

__global__ void kla_pscan_aff_reduce_kernel(
    float2 *__restrict__ aff, float *__restrict__ lam_ck,
    const float4 *__restrict__ mob_in, const float *__restrict__ msi,
    const float *__restrict__ si, const float *__restrict__ k,
    const float *__restrict__ a, const float *__restrict__ p,
    const float *__restrict__ lam0, int L, int M, int S, int NCK, int total) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;
    int b, m, c, s;
    kla_pscan_decode(gid, M, NCK, S, b, m, c, s);

    const float a_s = a[m * S + s];
    const float p_s = p[m * S + s];
    const float a2 = fmaxf(a_s * a_s, KLA_EPS);

    // The exclusive prefix maps the initial state to the state entering this
    // chunk. Chunk 0 has none, and is seeded with lam0 itself.
    const float lam_start = lam0[kla_bms(b, m, s, M, S)];
    float lam = (c == 0) ? lam_start
                         : kla_mob_apply(mob_in[kla_ck(b, m, c - 1, s, M, NCK, S)],
                                         lam_start);
    lam_ck[gid] = lam;

    float2 agg = kla_aff_identity();
#pragma unroll
    for (int i = 0; i < KLA_CHUNK; ++i) {
        const int t = c * KLA_CHUNK + i;
        if (t < L) {
            const float si_t = si[kla_blm(b, t, m, L, M)];
            const float msi_t = msi[kla_blm(b, t, m, L, M)];
            const float k_t = k[kla_bls(b, t, s, L, S)];
            const float phi = fmaxf(si_t * k_t * k_t, KLA_EPS);
            // The gain reads lambda_{t-1}, so it is formed before the update.
            const float den = kla_den(a2, p_s, lam);
            agg = kla_aff_compose(agg, make_float2(a_s / den, msi_t * k_t));
            lam = kla_lambda_step(lam, phi, den);
        }
    }
    aff[gid] = agg;
}

// ------------------------------ 5: seed eta, replay the chunk, and read out
//
// The only kernel here that reduces, so the only one with a block shape:
// [BLOCK_S, ROWS] with x over states and y over *chunks*, so the BLOCK_S lanes
// of a row hold every state of one (b, m, chunk) and the read-out sum over s is
// a warp reduction. Every thread must reach every reduction, including the
// padding lanes at s >= d_state and the rows past NCK, so nothing below returns
// early.

template <int BLOCK_S, int ROWS>
__global__ void kla_pscan_apply_kernel(
    float *__restrict__ y, float *__restrict__ yvar, float *__restrict__ lam_fin,
    float *__restrict__ eta_fin, float *__restrict__ eta_ck,
    const float *__restrict__ lam_ck, const float2 *__restrict__ aff_in,
    const float *__restrict__ msi, const float *__restrict__ si,
    const float *__restrict__ k, const float *__restrict__ q,
    const float *__restrict__ a, const float *__restrict__ p,
    const float *__restrict__ eta0, int L, int M, int S, int NCK, int prior) {
    __shared__ float2 red_s[KLA_TG_THREADS];

    const int s = threadIdx.x, ty = threadIdx.y;
    const int c = blockIdx.x * ROWS + ty;
    const int m = blockIdx.y, b = blockIdx.z;
    const bool active = (s < S) && (c < NCK);

    const float a_s = active ? a[m * S + s] : 1.0f;
    const float p_s = active ? p[m * S + s] : 0.0f;
    const float a2 = fmaxf(a_s * a_s, KLA_EPS);

    const int agg = active ? kla_ck(b, m, c, s, M, NCK, S) : 0;
    float lam = active ? lam_ck[agg] : 1.0f;

    const float eta_start = active ? eta0[kla_bms(b, m, s, M, S)] : 0.0f;
    float eta = eta_start;
    if (active && c > 0) {
        const float2 pref = aff_in[kla_ck(b, m, c - 1, s, M, NCK, S)];
        eta = pref.x * eta_start + pref.y;
    }
    if (active) eta_ck[agg] = eta;

#pragma unroll
    for (int i = 0; i < KLA_CHUNK; ++i) {
        const int t = c * KLA_CHUNK + i;
        const bool live = active && (t < L);
        float var = 0.0f;
        if (live) {
            const float si_t = si[kla_blm(b, t, m, L, M)];
            const float msi_t = msi[kla_blm(b, t, m, L, M)];
            const float k_t = k[kla_bls(b, t, s, L, S)];
            const float phi = fmaxf(si_t * k_t * k_t, KLA_EPS);
            const float den = kla_den(a2, p_s, lam);
            lam = kla_lambda_step(lam, phi, den);
            eta = (a_s / den) * eta + msi_t * k_t;
            var = 1.0f / fmaxf(lam, KLA_EPS);
        }
        const float mean_f = eta * var;
        const float var_o = prior ? (a2 * var + p_s) : var;
        const float mean_o = prior ? (a_s * mean_f) : mean_f;
        // Lanes with nothing to contribute carry q_t = 0, so the reduction
        // stays collective without them affecting the sum.
        const float q_t = live ? q[kla_bls(b, t, s, L, S)] : 0.0f;
        const float2 out = kla_sum_over_states<BLOCK_S>(
            make_float2(mean_o * q_t, var_o * q_t * q_t), red_s, s, ty);
        if (s == 0 && live) {
            y[kla_blm(b, t, m, L, M)] = out.x;
            yvar[kla_blm(b, t, m, L, M)] = out.y;
        }
    }

    // The last chunk holds the state at L, whatever the tail alignment.
    if (active && c == NCK - 1) {
        lam_fin[kla_bms(b, m, s, M, S)] = lam;
        eta_fin[kla_bms(b, m, s, M, S)] = eta;
    }
}
