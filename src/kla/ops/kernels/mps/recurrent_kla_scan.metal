/******************************************************************************
 * mps_recurrent -- the forward, one thread per (b, m, s), time serial.
 *   recurrent_kla_scan.metal
 *
 * The whole forward in one kernel, with no [B,L,M,S] intermediate ever reaching
 * device memory:
 *
 *     load k,q,v.Lambda^v,Lambda^v  ->  phi,r  ->  Moebius step -> lambda
 *     ->  gain alpha  ->  affine step -> eta
 *     ->  read-out  y = sum_s q.eta/lambda,  yvar = sum_s q^2/lambda
 *
 * The Moebius map is *applied* to a running lambda rather than composed with
 * its neighbours, so there is no 2x2 matrix, no trace normalization and no
 * overflow path -- and, crucially, the adjoint stays elementary. That is why
 * the backward (kla_scan_bwd.metal) is exact where the equally-fused CUDA
 * v2_* kernels are not.
 *
 * Kernel-internal names follow the CUDA kernels rather than the paper: ``msi``
 * is v.Lambda^v, ``si`` is Lambda^v, ``k`` the observation map and ``qw`` the
 * read-out. ``p`` is the process noise (the CUDA sources call it ``q``, which
 * collides with the query; renamed here).
 *
 * Layout. One thread owns one (b, m, s) triple, a threadgroup is
 * [KLA_BLOCK_S, KLA_ROWS] and the grid is (BLOCK_S, M_padded, B), so the
 * KLA_BLOCK_S lanes of a row hold every state of one channel and the read-out
 * sum over s is a lane reduction (see kla_common.metal). No thread returns
 * early -- the padding lanes at s >= S or m >= M load neutral values, take part
 * in every reduction, and skip only their stores.
 *
 * Checkpoints. Every KLA_CHUNK steps the thread writes the (lambda, eta) it is
 * about to consume into [B,M,NCK,S]. kla_scan_bwd.metal resumes from exactly
 * that, and chunk_kla_scan.metal writes the same thing at the same stride --
 * the adjoint does not care how the forward reached the checkpoint.
 ******************************************************************************/

kernel void kla_recurrent_fwd(
    device float *y            [[buffer(0)]],   // out [B, L, M]
    device float *yvar         [[buffer(1)]],   // out [B, L, M]
    device float *lam_fin      [[buffer(2)]],   // out [B, M, S]
    device float *eta_fin      [[buffer(3)]],   // out [B, M, S]
    device float *lam_ck       [[buffer(4)]],   // out [B, M, NCK, S] (see store_ck)
    device float *eta_ck       [[buffer(5)]],   // out [B, M, NCK, S]
    device const float *msi    [[buffer(6)]],   // v.Lambda^v [B, L, M]
    device const float *si     [[buffer(7)]],   // Lambda^v   [B, L, M]
    device const float *k      [[buffer(8)]],   // key        [B, L, S]
    device const float *qw     [[buffer(9)]],   // query      [B, L, S]
    device const float *a      [[buffer(10)]],  // decay         [M, S]
    device const float *p      [[buffer(11)]],  // process noise [M, S]
    device const float *lam0   [[buffer(12)]],  // [B, M, S]
    device const float *eta0   [[buffer(13)]],  // [B, M, S]
    constant int &L_           [[buffer(14)]],
    constant int &M_           [[buffer(15)]],
    constant int &S_           [[buffer(16)]],
    constant int &NCK_         [[buffer(17)]],
    constant int &store_ck     [[buffer(18)]],  // 0 = inference, skip checkpoints
    constant int &prior        [[buffer(19)]],  // decode_from_prior
    uint3 gid [[thread_position_in_grid]],
    uint tidx [[thread_index_in_threadgroup]]) {
    threadgroup float2 scratch[KLA_TG_THREADS];

    const uint L = uint(L_), M = uint(M_), S = uint(S_), NCK = uint(NCK_);
    const uint s = gid.x, m = gid.y, b = gid.z;
    const uint sx = tidx % KLA_BLOCK_S;
    const bool active = (s < S) && (m < M);

    const float a_s = active ? a[m * S + s] : 1.0f;
    const float p_s = active ? p[m * S + s] : 0.0f;
    const float a2 = fmax(a_s * a_s, KLA_EPS);
    const float inv_a2 = 1.0f / a2;

    float lam = active ? lam0[kla_bms(b, m, s, M, S)] : 1.0f;
    float eta = active ? eta0[kla_bms(b, m, s, M, S)] : 0.0f;

    for (uint t = 0; t < L; ++t) {
        if (store_ck != 0 && active && (t % KLA_CHUNK) == 0) {
            const uint ck = ((b * M + m) * NCK + t / KLA_CHUNK) * S + s;
            lam_ck[ck] = lam;
            eta_ck[ck] = eta;
        }

        const float si_t = active ? si[kla_blm(b, t, m, L, M)] : 0.0f;
        const float msi_t = active ? msi[kla_blm(b, t, m, L, M)] : 0.0f;
        const float k_t = active ? k[kla_bls(b, t, s, L, S)] : 0.0f;
        const float q_t = active ? qw[kla_bls(b, t, s, L, S)] : 0.0f;

        const float phi = fmax(si_t * k_t * k_t, KLA_EPS);
        const float lam_prev = lam;

        float A, C, den;
        lam = kla_lambda_step(lam_prev, phi, p_s, inv_a2, A, C, den);

        // The gain reads lambda_{t-1}, so it lags the precision update by one.
        const float denom = fmax(a2 + p_s * lam_prev, KLA_EPS);
        eta = (a_s / denom) * eta + msi_t * k_t;

        // The filtered posterior, then optionally one predict step ahead of it —
        // the same transform the torch path applies after its scan.
        const float var_f = 1.0f / fmax(lam, KLA_EPS);
        const float mean_f = eta * var_f;
        const float var = (prior != 0) ? (a2 * var_f + p_s) : var_f;
        const float mean = (prior != 0) ? (a_s * mean_f) : mean_f;

        // Padding lanes carry q_t = 0, so they contribute nothing to the sum.
        const float2 out = kla_sum_over_states(float2(mean * q_t, var * q_t * q_t),
                                               scratch, tidx, sx);
        if (s == 0 && m < M) {
            y[kla_blm(b, t, m, L, M)] = out.x;
            yvar[kla_blm(b, t, m, L, M)] = out.y;
        }
    }

    if (active) {
        lam_fin[kla_bms(b, m, s, M, S)] = lam;
        eta_fin[kla_bms(b, m, s, M, S)] = eta;
    }
}
