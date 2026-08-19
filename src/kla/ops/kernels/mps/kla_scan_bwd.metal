/******************************************************************************
 * The Metal backward -- one exact adjoint for both MPS schedules.
 *   kla_scan_bwd.metal
 *
 * An adjoint does not have to mirror its forward. All this kernel needs is
 * (lambda, eta) at the checkpoints; whether the forward walked time serially
 * (recurrent_kla_scan.metal) or split each tile of timesteps across a
 * threadgroup (chunk_kla_scan.metal) changes nothing here. Both write the same
 * [B,M,NCK,S] checkpoints and share this one backward.
 *
 * It is the *exact* adjoint, unlike the equally-fused CUDA v2_* kernels. That
 * falls out of applying the Moebius map instead of composing it: those have to
 * differentiate a trace-normalized prefix product, which is what makes their
 * d(Lambda^v)/d(k)/d(a)/d(p) approximate, whereas a per-step map has the
 * elementary derivative
 *
 *     d lambda_t / d lambda_{t-1} = (A.D - B.C)/den^2 = 1/(a^2.den^2)
 *
 * and the reverse walk is just that gain plus the alpha path, since alpha_t
 * reads lambda_{t-1} as well. The two adjoints are carried in registers, so the
 * backward stays a single kernel like the forward.
 *
 * Recompute. The backward needs lambda_t and eta_t at every step and neither
 * forward stores them. It replays KLA_CHUNK steps from each checkpoint into two
 * thread-private arrays, then runs the adjoint back down over them. lambda
 * could be recovered by inverting the leaf instead, but eta cannot:
 * eta_{t-1} = (eta_t - r_t)/alpha_t divides by a gain that is small exactly
 * when the filter is forgetting, so the error would blow up geometrically along
 * the reverse walk.
 *
 * Contractions. d(v.Lambda^v) and d(Lambda^v) contract over s, so they reduce
 * along a row and one lane stores them outright. d(k) and d(q) contract over m
 * instead, which no single threadgroup owns: they reduce over the KLA_ROWS
 * channels in the group and then accumulate atomically, so both output buffers
 * must be zeroed by the caller. d(a) and d(p) contract over b and t; each
 * thread accumulates its own in a register and adds once at the end.
 ******************************************************************************/

kernel void kla_scan_bwd(
    device atomic_float *dk       [[buffer(0)]],   // accum [B, L, S], zeroed
    device atomic_float *dq       [[buffer(1)]],   // accum [B, L, S], zeroed
    device atomic_float *da       [[buffer(2)]],   // accum [M, S],    zeroed
    device atomic_float *dp       [[buffer(3)]],   // accum [M, S],    zeroed
    device float *dmsi            [[buffer(4)]],   // out [B, L, M]
    device float *dsi             [[buffer(5)]],   // out [B, L, M]
    device float *dlam0           [[buffer(6)]],   // out [B, M, S]
    device float *deta0           [[buffer(7)]],   // out [B, M, S]
    device const float *dy        [[buffer(8)]],   // [B, L, M]
    device const float *dyvar     [[buffer(9)]],   // [B, L, M]
    device const float *dlam_fin  [[buffer(10)]],  // [B, M, S]
    device const float *deta_fin  [[buffer(11)]],  // [B, M, S]
    device const float *msi       [[buffer(12)]],
    device const float *si        [[buffer(13)]],
    device const float *k         [[buffer(14)]],
    device const float *qw        [[buffer(15)]],
    device const float *a         [[buffer(16)]],
    device const float *p         [[buffer(17)]],
    device const float *lam_ck    [[buffer(18)]],  // [B, M, NCK, S]
    device const float *eta_ck    [[buffer(19)]],
    constant int &L_              [[buffer(20)]],
    constant int &M_              [[buffer(21)]],
    constant int &S_              [[buffer(22)]],
    constant int &NCK_            [[buffer(23)]],
    constant int &prior           [[buffer(24)]],  // decode_from_prior
    uint3 gid [[thread_position_in_grid]],
    uint tidx [[thread_index_in_threadgroup]]) {
    threadgroup float2 scratch[KLA_TG_THREADS];

    const uint L = uint(L_), M = uint(M_), S = uint(S_), NCK = uint(NCK_);
    const uint s = gid.x, m = gid.y, b = gid.z;
    const uint sx = tidx % KLA_BLOCK_S;
    const uint my = tidx / KLA_BLOCK_S;
    const bool active = (s < S) && (m < M);

    const float a_s = active ? a[m * S + s] : 1.0f;
    const float p_s = active ? p[m * S + s] : 0.0f;
    const float a2 = fmax(a_s * a_s, KLA_EPS);
    const float inv_a2 = 1.0f / a2;
    // a2 = clamp_min(a*a, EPS) has zero subgradient where the clamp binds.
    const float da2_da = (a_s * a_s > KLA_EPS) ? (2.0f * a_s) : 0.0f;

    // Seeded from the incoming grad of the returned filter state, so a caller
    // that back-propagates through (lam_fin, eta_fin) gets the right answer.
    float carry_lam = active ? dlam_fin[kla_bms(b, m, s, M, S)] : 0.0f;
    float carry_eta = active ? deta_fin[kla_bms(b, m, s, M, S)] : 0.0f;
    float acc_da = 0.0f, acc_dp = 0.0f;

    float lam_h[KLA_CHUNK];  // lambda_{t-1} over the chunk being replayed
    float eta_h[KLA_CHUNK];  // eta_{t-1}

    for (int c = int(NCK) - 1; c >= 0; --c) {
        const uint t0 = uint(c) * KLA_CHUNK;
        const uint ck = ((b * M + m) * NCK + uint(c)) * S + s;

        // Replay the chunk forward from its checkpoint.
        float lam = active ? lam_ck[ck] : 1.0f;
        float eta = active ? eta_ck[ck] : 0.0f;
#pragma unroll
        for (uint j = 0; j < KLA_CHUNK; ++j) {
            lam_h[j] = lam;
            eta_h[j] = eta;
            const uint t = t0 + j;
            if (t < L) {  // threadgroup-uniform
                const float si_t = active ? si[kla_blm(b, t, m, L, M)] : 0.0f;
                const float msi_t = active ? msi[kla_blm(b, t, m, L, M)] : 0.0f;
                const float k_t = active ? k[kla_bls(b, t, s, L, S)] : 0.0f;
                const float phi = fmax(si_t * k_t * k_t, KLA_EPS);
                float A, C, den;
                const float lam_next = kla_lambda_step(lam, phi, p_s, inv_a2, A, C, den);
                const float denom = fmax(a2 + p_s * lam, KLA_EPS);
                eta = (a_s / denom) * eta + msi_t * k_t;
                lam = lam_next;
            }
        }

        // Walk the adjoint back down the chunk.
#pragma unroll
        for (int j = KLA_CHUNK - 1; j >= 0; --j) {
            const uint t = t0 + uint(j);
            if (t >= L) { continue; }  // threadgroup-uniform: reductions stay aligned

            const float lam_prev = lam_h[j];
            const float eta_prev = eta_h[j];

            const float si_t = active ? si[kla_blm(b, t, m, L, M)] : 0.0f;
            const float msi_t = active ? msi[kla_blm(b, t, m, L, M)] : 0.0f;
            const float k_t = active ? k[kla_bls(b, t, s, L, S)] : 0.0f;
            const float q_t = active ? qw[kla_bls(b, t, s, L, S)] : 0.0f;
            const float dy_t = active ? dy[kla_blm(b, t, m, L, M)] : 0.0f;
            const float dyv_t = active ? dyvar[kla_blm(b, t, m, L, M)] : 0.0f;

            // Recompute the step (identical arithmetic to the forward kernel).
            const float raw_phi = si_t * k_t * k_t;
            const float phi = fmax(raw_phi, KLA_EPS);
            const float r_t = msi_t * k_t;
            float A, C, den;
            const float lam_t = kla_lambda_step(lam_prev, phi, p_s, inv_a2, A, C, den);
            const float raw_denom = a2 + p_s * lam_prev;
            const float denom = fmax(raw_denom, KLA_EPS);
            const float alpha = a_s / denom;
            const float eta_t = alpha * eta_prev + r_t;
            const float var = 1.0f / fmax(lam_t, KLA_EPS);
            const float mean = eta_t * var;
            const float var_o = (prior != 0) ? (a2 * var + p_s) : var;
            const float mean_o = (prior != 0) ? (a_s * mean) : mean;

            // Read-out: y = sum_s q.mean_o, yvar = sum_s q^2.var_o.
            const float d_mean_o = dy_t * q_t;
            const float d_var_o = dyv_t * q_t * q_t;
            const float dq_c = dy_t * mean_o + dyv_t * 2.0f * q_t * var_o;

            // ...back through the predict step, if there was one, onto the
            // posterior (mean = eta.var, hence the eta_t term in d_var).
            const float d_mean = (prior != 0) ? (d_mean_o * a_s) : d_mean_o;
            const float d_var =
                ((prior != 0) ? (d_var_o * a2) : d_var_o) + d_mean * eta_t;

            const float mu = carry_lam + ((lam_t > KLA_EPS) ? (-d_var * var * var) : 0.0f);
            const float nu = carry_eta + d_mean * var;

            // eta_t = alpha_t.eta_{t-1} + r_t
            const float dr = nu;
            const float dalpha = nu * eta_prev;
            const float eta_carry = alpha * nu;

            // lambda_t = (A.lambda' + phi)/den,  den = C.lambda' + 1
            const float u = mu / den;
            const float dA = u * lam_prev;
            const float dB = u;                     // B = phi
            const float dC = -u * lam_t * lam_prev; // D = 1 is constant, dD dropped
            const float gain = inv_a2 / (den * den);

            // alpha_t = a/max(a^2 + p.lambda_{t-1}, EPS) also reads lambda_{t-1}.
            const float free = (raw_denom > KLA_EPS) ? 1.0f : 0.0f;
            const float inv_denom2 = free / (denom * denom);
            const float lam_carry = mu * gain + dalpha * (-(a_s * p_s) * inv_denom2);

            // Leaf coefficients back onto phi, p and a^2.
            const float dphi = dA * (p_s * inv_a2) + dB;
            float da2 = -(dA * A + dC * C) * inv_a2;
            acc_dp += (dA * phi + dC) * inv_a2;
            // ... and the alpha path onto a, a^2 and p.
            acc_da += dalpha * (free / denom);
            da2 += dalpha * (-a_s * inv_denom2);
            acc_dp += dalpha * (-a_s * lam_prev * inv_denom2);
            // The predict step reads a and p directly as well.
            if (prior != 0) {
                acc_da += d_mean_o * mean;
                da2 += d_var_o * var;
                acc_dp += d_var_o;
            }
            acc_da += da2 * da2_da;

            // phi = clamp_min(Lambda^v.k^2, EPS) — zero subgradient once floored.
            const float phi_live = (raw_phi > KLA_EPS) ? 1.0f : 0.0f;
            const float dsi_c = dphi * phi_live * k_t * k_t;
            const float dmsi_c = dr * k_t;
            const float dk_c = dphi * phi_live * 2.0f * si_t * k_t + dr * msi_t;

            // Contract over s: one lane per channel owns the result outright.
            const float2 red_s =
                kla_sum_over_states(float2(dmsi_c, dsi_c), scratch, tidx, sx);
            if (s == 0 && m < M) {
                dmsi[kla_blm(b, t, m, L, M)] = red_s.x;
                dsi[kla_blm(b, t, m, L, M)] = red_s.y;
            }
            // Contract over m: partial within the group, then accumulate.
            const float2 red_m =
                kla_sum_over_channels(float2(dk_c, dq_c), scratch, tidx, my);
            if (my == 0 && active) {
                atomic_fetch_add_explicit(&dk[kla_bls(b, t, s, L, S)], red_m.x,
                                          memory_order_relaxed);
                atomic_fetch_add_explicit(&dq[kla_bls(b, t, s, L, S)], red_m.y,
                                          memory_order_relaxed);
            }

            carry_lam = lam_carry;
            carry_eta = eta_carry;
        }
    }

    if (active) {
        dlam0[kla_bms(b, m, s, M, S)] = carry_lam;
        deta0[kla_bms(b, m, s, M, S)] = carry_eta;
        atomic_fetch_add_explicit(&da[m * S + s], acc_da, memory_order_relaxed);
        atomic_fetch_add_explicit(&dp[m * S + s], acc_dp, memory_order_relaxed);
    }
}
