/******************************************************************************
 * The CUDA backward -- one exact adjoint for every CUDA implementation.
 *   kla_scan_bwd.cuh
 *
 * An adjoint does not have to mirror its forward. All this kernel needs is
 * (lambda, eta) at the checkpoints; whether the forward walked time serially,
 * tiled it, or scanned it changes nothing here. So cuda_recurrent, cuda_chunk
 * and cuda_pscan share this one backward, exactly as the Metal and triton
 * kernels share theirs.
 *
 * Why this is not the v2_* backward. That one differentiates the trace-
 * normalized *composition*: a 4-component adjoint carried through a chain of
 * 4x4 Jacobians, ~64 MACs and 16 floats of shared memory per thread per step.
 * The composed matrix degenerates toward rank-1, so that carry is
 * ill-conditioned in float32 even though the forward -- a scale-invariant ratio
 * of the same matrix -- is not, which is where its 5-15% comes from.
 * Differentiating the *recurrence* instead gives a scalar gain
 *
 *     d lambda_t / d lambda_{t-1} = a^2/den_t^2
 *
 * so the whole backward is two reverse affine recurrences carrying one scalar
 * each. It is not a speed-for-accuracy trade in either direction: this is both
 * exact and strictly less work, and it frees the Jacobian's shared memory,
 * which is what buys occupancy back on the consumer parts.
 *
 * Recompute. The reverse walk needs lambda_t and eta_t at every step and the
 * forward stores neither. It replays KLA_CHUNK steps from each checkpoint into
 * two thread-private arrays, then walks the adjoint back down over them.
 * lambda could be recovered by inverting the leaf instead, but eta cannot:
 * eta_{t-1} = (eta_t - r_t)/alpha_t divides by a gain that is small exactly
 * when the filter is forgetting, so the error would grow geometrically along
 * the reverse walk. (The triton backward, which has no per-thread arrays,
 * avoids eta_{t-1} by a different route -- see kla_scan_bwd.py.)
 *
 * Contractions. d(v.Lambda^v) and d(Lambda^v) contract over s, so they reduce
 * along a row and one lane stores them outright. d(k) and d(q) contract over m,
 * which no single block owns, so they accumulate atomically -- both buffers
 * must be zeroed by the caller. d(a) and d(p) contract over b and t; each
 * thread keeps its own in a register and adds once at the end.
 ******************************************************************************/
#pragma once

#include "kla_scan_common.cuh"

template <int BLOCK_S, int ROWS>
__global__ void kla_scan_bwd_kernel(
    float *__restrict__ dk,     // out [B, L, S] (atomic; zero on entry)
    float *__restrict__ dq,     // out [B, L, S] (atomic; zero on entry)
    float *__restrict__ da,     // out [M, S]    (atomic; zero on entry)
    float *__restrict__ dp,     // out [M, S]    (atomic; zero on entry)
    float *__restrict__ dmsi,   // out [B, L, M]
    float *__restrict__ dsi,    // out [B, L, M]
    float *__restrict__ dlam0,  // out [B, M, S]
    float *__restrict__ deta0,  // out [B, M, S]
    const float *__restrict__ dy,        // [B, L, M]
    const float *__restrict__ dyvar,     // [B, L, M]
    const float *__restrict__ dlam_fin,  // [B, M, S]
    const float *__restrict__ deta_fin,  // [B, M, S]
    const float *__restrict__ msi, const float *__restrict__ si,
    const float *__restrict__ k, const float *__restrict__ qw,
    const float *__restrict__ a, const float *__restrict__ p,
    const float *__restrict__ lam0, const float *__restrict__ eta0,
    const float *__restrict__ lam_ck, const float *__restrict__ eta_ck,
    int L, int M, int S, int NCK, int prior) {
    __shared__ float2 smem[BLOCK_S <= 32 ? 1 : ROWS * BLOCK_S];

    const int sx = threadIdx.x;
    const int row = threadIdx.y;
    const int m = blockIdx.x * ROWS + row;
    const int b = blockIdx.y;
    const bool active = (sx < S) && (m < M);

    const float a_s = active ? a[m * S + sx] : 1.0f;
    const float p_s = active ? p[m * S + sx] : 0.0f;
    const float a2 = fmaxf(a_s * a_s, KLA_EPS);

    float nu_lam = 0.0f, nu_eta = 0.0f;  // adjoints of lambda_t and eta_t
    float da_acc = 0.0f, dp_acc = 0.0f;

    // d(final state) enters at t = L-1, which the first chunk walked holds.
    const float d_lam_fin = active ? dlam_fin[kla_bms(b, m, sx, M, S)] : 0.0f;
    const float d_eta_fin = active ? deta_fin[kla_bms(b, m, sx, M, S)] : 0.0f;

    float lam_h[KLA_CHUNK];  // lambda_{t-1} over the chunk being replayed
    float eta_h[KLA_CHUNK];  // eta_{t-1}, which cannot be recovered by inverting

    for (int c = NCK - 1; c >= 0; --c) {
        const int t0 = c * KLA_CHUNK;
        const int n = min(KLA_CHUNK, L - t0);

        // -- replay the chunk forward from its checkpoint ---------------------
        //
        // Both loops run the full KLA_CHUNK and mask the data instead of
        // trimming the trip count: a constant bound is what lets the compiler
        // unroll and keep lam_h/eta_h in registers rather than spilling them to
        // local memory. Only the last chunk is ever partial, and `n` is uniform
        // across the block, so the masking never diverges.
        float lam = active ? lam_ck[kla_ck(b, m, c, sx, M, NCK, S)] : 1.0f;
        float eta = active ? eta_ck[kla_ck(b, m, c, sx, M, NCK, S)] : 0.0f;
#pragma unroll
        for (int i = 0; i < KLA_CHUNK; ++i) {
            const int t = t0 + i;
            const bool ok = active && (i < n);
            lam_h[i] = lam;
            eta_h[i] = eta;
            const float si_t = ok ? si[kla_blm(b, t, m, L, M)] : 0.0f;
            const float msi_t = ok ? msi[kla_blm(b, t, m, L, M)] : 0.0f;
            const float k_t = ok ? k[kla_bls(b, t, sx, L, S)] : 0.0f;
            const float phi = fmaxf(si_t * k_t * k_t, KLA_EPS);
            const float den = kla_den(a2, p_s, lam);
            const float alpha = a_s / den;
            if (i < n) {
                lam = kla_lambda_step(lam, phi, den);
                eta = alpha * eta + msi_t * k_t;
            }
        }

        // -- walk the adjoint back down the chunk -----------------------------
#pragma unroll
        for (int i = KLA_CHUNK - 1; i >= 0; --i) {
            const int t = t0 + i;
            const bool live = (i < n);
            const float lam_prev = lam_h[i];
            const float eta_prev = eta_h[i];

            const bool ok = active && live;
            const float si_t = ok ? si[kla_blm(b, t, m, L, M)] : 0.0f;
            const float msi_t = ok ? msi[kla_blm(b, t, m, L, M)] : 0.0f;
            const float k_t = ok ? k[kla_bls(b, t, sx, L, S)] : 0.0f;
            const float q_t = ok ? qw[kla_bls(b, t, sx, L, S)] : 0.0f;

            const float raw_phi = si_t * k_t * k_t;
            const float phi = fmaxf(raw_phi, KLA_EPS);
            const float den = kla_den(a2, p_s, lam_prev);
            const float inv_den2 = 1.0f / (den * den);
            const float alpha = a_s / den;
            const float lam_t = kla_lambda_step(lam_prev, phi, den);
            const float eta_t = alpha * eta_prev + msi_t * k_t;

            const float var = 1.0f / fmaxf(lam_t, KLA_EPS);
            const float mean = eta_t * var;
            const float mean_o = (prior != 0) ? a_s * mean : mean;
            const float var_o = (prior != 0) ? (a2 * var + p_s) : var;

            const float dy_t = (m < M && live) ? dy[kla_blm(b, t, m, L, M)] : 0.0f;
            const float dyv_t = (m < M && live) ? dyvar[kla_blm(b, t, m, L, M)] : 0.0f;

            // y = sum_s q.mean_o and yvar = sum_s q^2.var_o, so the incoming
            // scalars reach this lane weighted by q and q^2.
            if (ok) {
                atomicAdd(&dq[kla_bls(b, t, sx, L, S)],
                          dy_t * mean_o + dyv_t * 2.0f * q_t * var_o);
            }
            float d_mean_o = dy_t * q_t;
            float d_var_o = dyv_t * q_t * q_t;

            float d_mean = d_mean_o, d_var = d_var_o;
            if (prior != 0) {
                d_mean = d_mean_o * a_s;
                d_var = d_var_o * a2;
                da_acc += d_mean_o * mean + d_var_o * 2.0f * a_s * var;
                dp_acc += d_var_o;
            }

            float d_eta_dir = d_mean * var;
            float d_lam_dir = -(d_var + d_mean * eta_t) * var * var;
            if (t == L - 1) {
                d_lam_dir += d_lam_fin;
                d_eta_dir += d_eta_fin;
            }

            // The gain at t+1 reads lambda_t, so both multipliers come from the
            // value this step just produced.
            const float den_next = kla_den(a2, p_s, lam_t);
            const float inv_dn2 = 1.0f / (den_next * den_next);
            const float alpha_next = a_s / den_next;
            const float g_next = a2 * inv_dn2;

            // lambda_t also moves alpha_{t+1}, which moves eta_{t+1}; that path
            // carries the eta adjoint one step later.
            // d alpha_{t+1} / d lambda_t = -a.p/den_{t+1}^2; nu_eta still holds
            // the adjoint at t+1 at this point.
            const float src_lam =
                d_lam_dir - eta_t * a_s * p_s * inv_dn2 * nu_eta;
            if (live) {
                nu_eta = d_eta_dir + alpha_next * nu_eta;
                nu_lam = src_lam + g_next * nu_lam;
            }

            // lambda_t = lambda_{t-1}/den_t + phi_t, so d lambda_t/d phi_t = 1.
            const float dphi = (raw_phi > KLA_EPS) ? nu_lam : 0.0f;
            const float dr = nu_eta;

            const float2 red = kla_sum_over_states<BLOCK_S>(
                make_float2(dphi * k_t * k_t, dr * k_t), smem, sx, row);
            if (sx == 0 && m < M && live) {
                dsi[kla_blm(b, t, m, L, M)] = red.x;
                dmsi[kla_blm(b, t, m, L, M)] = red.y;
            }
            if (ok) {
                atomicAdd(&dk[kla_bls(b, t, sx, L, S)],
                          dphi * 2.0f * si_t * k_t + dr * msi_t);
            }
            if (!live) continue;

            da_acc += nu_lam * (-lam_prev * 2.0f * a_s * inv_den2) +
                      nu_eta * eta_prev * ((den - 2.0f * a2) * inv_den2);
            dp_acc += nu_lam * (-lam_prev * lam_prev * inv_den2) +
                      nu_eta * eta_prev * (-a_s * lam_prev * inv_den2);
        }
    }

    if (active) {
        atomicAdd(&da[m * S + sx], da_acc);
        atomicAdd(&dp[m * S + sx], dp_acc);

        // The boundary, one step before t = 0.
        const float l0 = lam0[kla_bms(b, m, sx, M, S)];
        const float e0 = eta0[kla_bms(b, m, sx, M, S)];
        const float den0 = kla_den(a2, p_s, l0);
        const float inv_d02 = 1.0f / (den0 * den0);
        dlam0[kla_bms(b, m, sx, M, S)] =
            a2 * inv_d02 * nu_lam - e0 * a_s * p_s * inv_d02 * nu_eta;
        deta0[kla_bms(b, m, sx, M, S)] = (a_s / den0) * nu_eta;
    }
}
