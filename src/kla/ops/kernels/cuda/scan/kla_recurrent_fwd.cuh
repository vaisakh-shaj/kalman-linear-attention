/******************************************************************************
 * cuda_recurrent -- the forward, one thread per (b, m, s), time serial.
 *   kla_recurrent_fwd.cuh
 *
 * The Moebius map is *applied* to a running lambda rather than composed with
 * its neighbours, so there is no 2x2 matrix, no trace normalization and no
 * overflow path. Parallelism comes from B*M*S lanes, which is plenty whenever
 * there are sequences or channels to spend -- decode, and training at any real
 * batch size. cuda_chunk is the one to reach for when there are not.
 *
 * Checkpoints. Every KLA_CHUNK steps the thread writes the (lambda, eta) it is
 * *about to* consume into [B, M, NCK, S]. kla_scan_bwd.cuh resumes from exactly
 * that, and the same layout and stride serve every CUDA schedule -- the adjoint
 * does not care how the forward reached the checkpoint.
 ******************************************************************************/
#pragma once

#include "kla_scan_common.cuh"

template <int BLOCK_S, int ROWS>
__global__ void kla_recurrent_fwd_kernel(
    float *__restrict__ y,        // out [B, L, M]
    float *__restrict__ yvar,     // out [B, L, M]
    float *__restrict__ lam_fin,  // out [B, M, S]
    float *__restrict__ eta_fin,  // out [B, M, S]
    float *__restrict__ lam_ck,   // out [B, M, NCK, S] (see store_ck)
    float *__restrict__ eta_ck,   // out [B, M, NCK, S]
    const float *__restrict__ msi,   // v.Lambda^v [B, L, M]
    const float *__restrict__ si,    // Lambda^v   [B, L, M]
    const float *__restrict__ k,     // key   [B, L, S]
    const float *__restrict__ qw,    // query [B, L, S]
    const float *__restrict__ a,     // decay         [M, S]
    const float *__restrict__ p,     // process noise [M, S]
    const float *__restrict__ lam0,  // [B, M, S]
    const float *__restrict__ eta0,  // [B, M, S]
    int L, int M, int S, int NCK, int store_ck, int prior) {
    __shared__ float2 smem[BLOCK_S <= 32 ? 1 : ROWS * BLOCK_S];

    const int sx = threadIdx.x;
    const int row = threadIdx.y;
    const int m = blockIdx.x * ROWS + row;
    const int b = blockIdx.y;
    // Padding lanes exist so the reduction stays collective; they load neutral
    // values, take part in every butterfly, and skip only their stores.
    const bool active = (sx < S) && (m < M);

    const float a_s = active ? a[m * S + sx] : 1.0f;
    const float p_s = active ? p[m * S + sx] : 0.0f;
    const float a2 = fmaxf(a_s * a_s, KLA_EPS);

    float lam = active ? lam0[kla_bms(b, m, sx, M, S)] : 1.0f;
    float eta = active ? eta0[kla_bms(b, m, sx, M, S)] : 0.0f;

    for (int t = 0; t < L; ++t) {
        if (store_ck != 0 && active && (t % KLA_CHUNK) == 0) {
            const int c = kla_ck(b, m, t / KLA_CHUNK, sx, M, NCK, S);
            lam_ck[c] = lam;
            eta_ck[c] = eta;
        }

        const float si_t = active ? si[kla_blm(b, t, m, L, M)] : 0.0f;
        const float msi_t = active ? msi[kla_blm(b, t, m, L, M)] : 0.0f;
        const float k_t = active ? k[kla_bls(b, t, sx, L, S)] : 0.0f;
        const float q_t = active ? qw[kla_bls(b, t, sx, L, S)] : 0.0f;

        const float phi = fmaxf(si_t * k_t * k_t, KLA_EPS);
        const float den = kla_den(a2, p_s, lam);
        const float alpha = a_s / den;
        lam = kla_lambda_step(lam, phi, den);
        eta = alpha * eta + msi_t * k_t;

        float var = 1.0f / fmaxf(lam, KLA_EPS);
        float mean = eta * var;
        if (prior != 0) {  // decode_from_prior: read out one predict step ahead
            mean = a_s * mean;
            var = a2 * var + p_s;
        }
        const float2 out = kla_sum_over_states<BLOCK_S>(
            make_float2(mean * q_t, var * q_t * q_t), smem, sx, row);
        if (sx == 0 && m < M) {
            y[kla_blm(b, t, m, L, M)] = out.x;
            yvar[kla_blm(b, t, m, L, M)] = out.y;
        }
    }

    if (active) {
        lam_fin[kla_bms(b, m, sx, M, S)] = lam;
        eta_fin[kla_bms(b, m, sx, M, S)] = eta;
    }
}
