/******************************************************************************
 * mps_chunk -- the forward, time as a *parallel* axis.
 *   chunk_kla_scan.metal
 *
 * For the case the lane-per-state kernel cannot fill: batch-1 prefill, where
 * B*M*S alone leaves the GPU short of threads.
 *
 * One threadgroup owns one (batch, channel) pair and streams the sequence in
 * tiles of KLA_ROWS * KLA_ITEMS timesteps, carrying (lambda, eta) across tiles.
 * Inside a tile the KLA_ROWS threads of a state column split the timesteps
 * between them, so the grid is KLA_ROWS times wider than the serial kernels'.
 *
 * The specialization defines are reused with tile meanings here:
 *   KLA_BLOCK_S   states per column = next_pow2(d_state)
 *   KLA_ROWS      threads along time  (the parallelism this file exists for)
 *   KLA_ITEMS     timesteps each of those threads walks serially
 *
 * Cost. Parallelizing over time forces *composition*: a thread cannot apply the
 * Moebius map without already knowing the running lambda, which is the serial
 * dependency itself. So each element is composed once (phase A) and applied
 * once (phase C) rather than applied once — about 4x the arithmetic of
 * recurrent_kla_scan.metal, bought in exchange for KLA_ROWS times the parallelism.
 * Walking KLA_ITEMS steps serially per thread is what keeps it at 4x rather
 * than the log(tile) composes a pure Hillis-Steele scan over timesteps costs.
 *
 * Six phases per tile:
 *   A  compose this thread's KLA_ITEMS Moebius leaves    -> thread aggregate
 *   B  exclusive scan of those aggregates along time     -> my starting lambda
 *   C  walk my timesteps applying the map                -> lambda, alpha, r
 *   D  compose my affine leaves (alpha, r)               -> thread aggregate
 *   E  exclusive scan of those along time                -> my starting eta
 *   F  walk applying the affine map, read out over s     -> y, yvar
 *
 * D and E cannot be folded into A and B: alpha_t reads lambda_{t-1}, so the
 * affine leaves do not exist until phase C has produced lambda.
 *
 * Backward. There is none here, and none is wanted: the adjoint does not have
 * to mirror the forward. This kernel checkpoints (lambda, eta) every KLA_CHUNK
 * steps into [B,M,NCK,S] -- the same layout and the same stride
 * recurrent_kla_scan.metal writes -- and kla_scan_bwd replays from them. That walk
 * is over the serial state lanes either way, so composing the map in the
 * forward buys the backward nothing and costs it nothing.
 *
 * lambda and eta are checkpointed in different phases (C and F), because the
 * affine leaves do not exist until the Moebius phase has produced lambda. Both
 * stores land on the same index, and both store the value *entering* step t,
 * which is what the replay resumes from.
 ******************************************************************************/

// ------------------------------------------------------- 2x2 Moebius algebra
//
// Same representation and the same trace normalization as the torch reference
// (kla.ops.kla_ops._mobius_combine_tracenorm) and the triton kernels:
// float4(A, B, C, D) is [[A, B], [C, D]], acting as lambda -> (A.l+B)/(C.l+D).

inline float4 kla_mob_identity() { return float4(1.0f, 0.0f, 0.0f, 1.0f); }

// The map "R after L": the matrix product R*L, divided by its own trace. The
// quotient is invariant under that rescaling, so it only buys exponent range.
inline float4 kla_mob_compose(float4 L, float4 R) {
    const float a = R.x * L.x + R.y * L.z;
    const float b = R.x * L.y + R.y * L.w;
    const float c = R.z * L.x + R.w * L.z;
    const float d = R.z * L.y + R.w * L.w;
    return float4(a, b, c, d) / fmax(a + d, KLA_EPS);
}

inline float kla_mob_apply(float4 P, float lam) {
    return (P.x * lam + P.y) / fmax(P.z * lam + P.w, KLA_EPS);
}

inline float2 kla_aff_identity() { return float2(1.0f, 0.0f); }

inline float2 kla_aff_compose(float2 L, float2 R) {
    return float2(R.x * L.x, R.x * L.y + R.y);
}

// ------------------------------------------------------------- tile scans
//
// Exclusive scan across the KLA_ROWS threads holding one state column, so the
// result maps the tile's incoming carry to this thread's starting value.
// Hillis-Steele: log2(KLA_ROWS) rounds over KLA_ROWS elements, which is
// negligible next to the KLA_ITEMS serial steps each element already cost.
// Every thread must reach every barrier, so nothing here is predicated on
// s < d_state or on the sequence bound.

inline float4 kla_tile_scan_mobius(float4 mine, threadgroup float4 *smem,
                                   uint tid, uint ty) {
#if KLA_ROWS == 1
    return kla_mob_identity();
#else
    smem[tid] = mine;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = 1; off < KLA_ROWS; off <<= 1) {
        const float4 prev =
            (ty >= off) ? smem[tid - off * KLA_BLOCK_S] : kla_mob_identity();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (ty >= off) { mine = kla_mob_compose(prev, mine); }
        smem[tid] = mine;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float4 excl =
        (ty > 0) ? smem[tid - KLA_BLOCK_S] : kla_mob_identity();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return excl;
#endif
}

inline float2 kla_tile_scan_affine(float2 mine, threadgroup float2 *smem,
                                   uint tid, uint ty) {
#if KLA_ROWS == 1
    return kla_aff_identity();
#else
    smem[tid] = mine;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = 1; off < KLA_ROWS; off <<= 1) {
        const float2 prev =
            (ty >= off) ? smem[tid - off * KLA_BLOCK_S] : kla_aff_identity();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (ty >= off) { mine = kla_aff_compose(prev, mine); }
        smem[tid] = mine;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float2 excl =
        (ty > 0) ? smem[tid - KLA_BLOCK_S] : kla_aff_identity();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return excl;
#endif
}

// Broadcast one thread-row's value to the whole state column.
inline float kla_tile_broadcast(float v, threadgroup float *smem, uint sx, uint ty) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ty == KLA_ROWS - 1) { smem[sx] = v; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float out = smem[sx];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return out;
}

kernel void kla_chunk_fwd(
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
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpt  [[thread_position_in_threadgroup]],
    uint  tidx [[thread_index_in_threadgroup]]) {
    threadgroup float4 mob_s[KLA_TG_THREADS];
    threadgroup float2 aff_s[KLA_TG_THREADS];
    threadgroup float2 red_s[KLA_TG_THREADS];
    threadgroup float  bcast_s[KLA_BLOCK_S];

    const uint L = uint(L_), M = uint(M_), S = uint(S_), NCK = uint(NCK_);
    const uint s = tpt.x, ty = tpt.y;
    const uint m = tgid.y, b = tgid.z;
    const bool active = (s < S);

    const float a_s = active ? a[m * S + s] : 1.0f;
    const float p_s = active ? p[m * S + s] : 0.0f;
    const float a2 = fmax(a_s * a_s, KLA_EPS);
    const float inv_a2 = 1.0f / a2;

    float carry_lam = active ? lam0[kla_bms(b, m, s, M, S)] : 1.0f;
    float carry_eta = active ? eta0[kla_bms(b, m, s, M, S)] : 0.0f;

    // This thread's slice of each tile, and what phase C leaves for phase F.
    float var_h[KLA_ITEMS];
    float alpha_h[KLA_ITEMS];
    float r_h[KLA_ITEMS];

    const uint tile_len = KLA_ROWS * KLA_ITEMS;
    const uint t_base = ty * KLA_ITEMS;

    for (uint tile = 0; tile < L; tile += tile_len) {
        // -- A: compose my leaves. Steps past the end of the sequence, and the
        // padding lanes at s >= d_state, contribute the identity.
        float4 agg = kla_mob_identity();
#pragma unroll
        for (uint i = 0; i < KLA_ITEMS; ++i) {
            const uint t = tile + t_base + i;
            if (t < L && active) {
                const float si_t = si[kla_blm(b, t, m, L, M)];
                const float k_t = k[kla_bls(b, t, s, L, S)];
                const float phi = fmax(si_t * k_t * k_t, KLA_EPS);
                agg = kla_mob_compose(
                    agg, float4((1.0f + p_s * phi) * inv_a2, phi, p_s * inv_a2, 1.0f));
            }
        }

        // -- B/C: where my slice starts, then walk it applying the map.
        const float4 pref = kla_tile_scan_mobius(agg, mob_s, tidx, ty);
        float lam = kla_mob_apply(pref, carry_lam);
#pragma unroll
        for (uint i = 0; i < KLA_ITEMS; ++i) {
            const uint t = tile + t_base + i;
            float alpha = 1.0f, r_t = 0.0f, var = 0.0f;
            if (t < L && active) {
                const float si_t = si[kla_blm(b, t, m, L, M)];
                const float msi_t = msi[kla_blm(b, t, m, L, M)];
                const float k_t = k[kla_bls(b, t, s, L, S)];
                const float phi = fmax(si_t * k_t * k_t, KLA_EPS);
                const float lam_prev = lam;
                // The value *entering* step t is what kla_scan_bwd resumes
                // from, so the store precedes the update -- as it does in
                // recurrent_kla_scan.metal, at the same stride and the same index.
                if (store_ck != 0 && (t % KLA_CHUNK) == 0) {
                    lam_ck[((b * M + m) * NCK + t / KLA_CHUNK) * S + s] = lam_prev;
                }
                float A, C, den;
                lam = kla_lambda_step(lam_prev, phi, p_s, inv_a2, A, C, den);
                // The gain reads lambda_{t-1}, so it lags the update by one.
                alpha = a_s / fmax(a2 + p_s * lam_prev, KLA_EPS);
                r_t = msi_t * k_t;
                var = 1.0f / fmax(lam, KLA_EPS);
            }
            var_h[i] = var;
            alpha_h[i] = alpha;
            r_h[i] = r_t;
        }
        // The last thread holds lambda at the tile boundary (or at L, if the
        // sequence ended inside this tile — later threads then never updated).
        carry_lam = kla_tile_broadcast(lam, bcast_s, s, ty);

        // -- D/E: the same two steps for the affine recurrence, which could not
        // run earlier because its leaves need the lambda phase C just produced.
        float2 aagg = kla_aff_identity();
#pragma unroll
        for (uint i = 0; i < KLA_ITEMS; ++i) {
            aagg = kla_aff_compose(aagg, float2(alpha_h[i], r_h[i]));
        }
        const float2 apref = kla_tile_scan_affine(aagg, aff_s, tidx, ty);

        // -- F: walk applying the affine map, and read out along the state axis.
        float eta = apref.x * carry_eta + apref.y;
#pragma unroll
        for (uint i = 0; i < KLA_ITEMS; ++i) {
            const uint t = tile + t_base + i;
            if (store_ck != 0 && active && t < L && (t % KLA_CHUNK) == 0) {
                eta_ck[((b * M + m) * NCK + t / KLA_CHUNK) * S + s] = eta;
            }
            eta = alpha_h[i] * eta + r_h[i];
            const float var_f = var_h[i];
            const float mean_f = eta * var_f;
            const float var = (prior != 0) ? (a2 * var_f + p_s) : var_f;
            const float mean = (prior != 0) ? (a_s * mean_f) : mean_f;
            // Lanes with nothing to contribute carry q_t = 0, so the reduction
            // stays collective without them affecting the sum.
            const float q_t = (t < L && active) ? qw[kla_bls(b, t, s, L, S)] : 0.0f;
            const float2 out = kla_sum_over_states(
                float2(mean * q_t, var * q_t * q_t), red_s, tidx, s);
            if (s == 0 && t < L) {
                y[kla_blm(b, t, m, L, M)] = out.x;
                yvar[kla_blm(b, t, m, L, M)] = out.y;
            }
        }
        carry_eta = kla_tile_broadcast(eta, bcast_s, s, ty);
    }

    if (active && ty == 0) {
        lam_fin[kla_bms(b, m, s, M, S)] = carry_lam;
        eta_fin[kla_bms(b, m, s, M, S)] = carry_eta;
    }
}
