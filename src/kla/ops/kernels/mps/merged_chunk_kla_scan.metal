/******************************************************************************
 * mps_merged_chunk -- chunk_kla_scan.metal's six phases, in three.
 *   merged_chunk_kla_scan.metal
 *
 * Same shape as chunk_kla_scan.metal: one threadgroup per (batch, channel),
 * streaming the sequence in tiles of KLA_ROWS * KLA_ITEMS timesteps, with the
 * KLA_ROWS threads of a state column splitting each tile's timesteps between
 * them. Same grid, same carry, same checkpoints. What changes is that there is
 * one scan instead of two, because the leaf composed in phase A is the 3x3 map
 * of kla_merged.metal, which carries eta alongside lambda.
 *
 *   A  compose this thread's KLA_ITEMS 3x3 leaves          -> thread aggregate
 *   B  exclusive scan of those along time, applied to the
 *      tile carry                                          -> my (lambda, eta)
 *   C  walk my timesteps applying *both* recurrences, and
 *      read out over s                                     -> y, yvar
 *
 * against chunk_kla_scan.metal's A/B/C for lambda and D/E/F for eta. Three
 * things follow, and they are the reason this file exists:
 *
 *   - One threadgroup scan instead of two. kla_tile_scan_* is log2(KLA_ROWS)
 *     Hillis-Steele rounds with two barriers each; halved. So are the two
 *     kla_tile_broadcast calls, which become one broadcast of a float2.
 *   - The per-thread arrays go away. var_h/alpha_h/r_h in chunk_kla_scan.metal
 *     exist only to carry phase C's output to D and F -- 3 * KLA_ITEMS
 *     registers, 24 at KLA_ITEMS=8, on a kernel whose whole purpose is
 *     occupancy.
 *   - Phase C applies both recurrences in the same walk, so it needs alpha_t
 *     only as a local: it has lambda_{t-1} in hand, exactly as the recurrent
 *     kernel does. The composed map never forms alpha at all.
 *
 * The cost is the aggregate: 8 floats (7 live, 1 padding) against a float4 plus
 * a float2, so threadgroup memory for the scan goes from 6 to 8 floats per
 * thread and the arithmetic per compose from ~14 multiplies to ~15. Both are
 * dwarfed by what the second scan cost.
 *
 * Backward: unchanged and unaware, as for both other forwards. This writes the
 * same [B,M,NCK,S] lambda/eta checkpoints at the same KLA_CHUNK stride with the
 * same convention -- the value *entering* step t -- and kla_scan_bwd replays a
 * scalar recurrence from them. It never sees a composed map of any size, which
 * is why merging the forward cannot touch it. If it ever has to, something has
 * leaked.
 *
 * Unlike chunk_kla_scan.metal, both checkpoints are written in the same phase.
 * There they could not be: eta did not exist until the affine phases had run.
 ******************************************************************************/

// ---------------------------------------------------------------- tile scan
//
// Exclusive scan across the KLA_ROWS threads of one state column, so the result
// maps the tile's incoming carry to this thread's starting state. The one scan
// this kernel runs, where chunk_kla_scan.metal runs this and an affine one.
// Every thread must reach every barrier, so nothing here is predicated on
// s < d_state or on the sequence bound.

inline KlaMerged kla_tile_scan_merged(KlaMerged mine, threadgroup KlaMerged *smem,
                                      uint tid, uint ty) {
#if KLA_ROWS == 1
    return kla_mrg_identity();
#else
    smem[tid] = mine;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = 1; off < KLA_ROWS; off <<= 1) {
        const KlaMerged prev =
            (ty >= off) ? smem[tid - off * KLA_BLOCK_S] : kla_mrg_identity();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (ty >= off) { mine = kla_mrg_compose(prev, mine); }
        smem[tid] = mine;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const KlaMerged excl =
        (ty > 0) ? smem[tid - KLA_BLOCK_S] : kla_mrg_identity();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return excl;
#endif
}

// Broadcast the last thread-row's (lambda, eta) to the whole state column. One
// call per tile, where the two-scan kernel needs one after each of its walks.
inline float2 kla_tile_broadcast2(float2 v, threadgroup float2 *smem,
                                  uint sx, uint ty) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ty == KLA_ROWS - 1) { smem[sx] = v; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float2 out = smem[sx];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return out;
}

kernel void kla_merged_chunk_fwd(
    device float *y            [[buffer(0)]],   // out [B, L, M]
    device float *yvar         [[buffer(1)]],   // out [B, L, M]
    device float *lam_fin      [[buffer(2)]],   // out [B, M, S]
    device float *eta_fin      [[buffer(3)]],   // out [B, M, S]
    device float *lam_ck       [[buffer(4)]],   // out [B, M, NCK, S]
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
    threadgroup KlaMerged mrg_s[KLA_TG_THREADS];
    threadgroup float2 red_s[KLA_TG_THREADS];
    threadgroup float2 bcast_s[KLA_BLOCK_S];

    const uint L = uint(L_), M = uint(M_), S = uint(S_), NCK = uint(NCK_);
    const uint s = tpt.x, ty = tpt.y;
    const uint m = tgid.y, b = tgid.z;
    const bool active = (s < S);

    const float a_s = active ? a[m * S + s] : 1.0f;
    const float p_s = active ? p[m * S + s] : 0.0f;
    const float a2 = fmax(a_s * a_s, KLA_EPS);
    const float inv_a2 = 1.0f / a2;
    const float inv_a = 1.0f / (a_s + copysign(KLA_EPS, a_s));

    float2 carry = float2(active ? lam0[kla_bms(b, m, s, M, S)] : 1.0f,
                          active ? eta0[kla_bms(b, m, s, M, S)] : 0.0f);

    const uint tile_len = KLA_ROWS * KLA_ITEMS;
    const uint t_base = ty * KLA_ITEMS;

    for (uint tile = 0; tile < L; tile += tile_len) {
        // -- A: compose my leaves. Steps past the end of the sequence, and the
        // padding lanes at s >= d_state, contribute the identity.
        KlaMerged agg = kla_mrg_identity();
#pragma unroll
        for (uint i = 0; i < KLA_ITEMS; ++i) {
            const uint t = tile + t_base + i;
            if (t < L && active) {
                const float si_t = si[kla_blm(b, t, m, L, M)];
                const float msi_t = msi[kla_blm(b, t, m, L, M)];
                const float k_t = k[kla_bls(b, t, s, L, S)];
                const float phi = fmax(si_t * k_t * k_t, KLA_EPS);
                agg = kla_mrg_compose(
                    agg, kla_mrg_leaf(phi, msi_t * k_t, p_s, inv_a2, inv_a));
            }
        }

        // -- B: where my slice starts. One scan, and it carries eta with it, so
        // there is nothing left to resolve after this.
        const KlaMerged pref = kla_tile_scan_merged(agg, mrg_s, tidx, ty);
        float2 st = kla_mrg_apply(pref, carry.x, carry.y);
        float lam = st.x, eta = st.y;

        // -- C: walk my timesteps applying both recurrences, and read out. The
        // gain is a local here: this walk has lambda_{t-1} in hand, which is
        // the property the composed map exists to hand back.
#pragma unroll
        for (uint i = 0; i < KLA_ITEMS; ++i) {
            const uint t = tile + t_base + i;
            float var = 0.0f;
            if (t < L && active) {
                const float si_t = si[kla_blm(b, t, m, L, M)];
                const float msi_t = msi[kla_blm(b, t, m, L, M)];
                const float k_t = k[kla_bls(b, t, s, L, S)];
                const float phi = fmax(si_t * k_t * k_t, KLA_EPS);
                const float lam_prev = lam;
                // The values *entering* step t are what kla_scan_bwd resumes
                // from, so both stores precede both updates.
                if (store_ck != 0 && (t % KLA_CHUNK) == 0) {
                    const uint ck = ((b * M + m) * NCK + t / KLA_CHUNK) * S + s;
                    lam_ck[ck] = lam_prev;
                    eta_ck[ck] = eta;
                }
                float A, C, den;
                lam = kla_lambda_step(lam_prev, phi, p_s, inv_a2, A, C, den);
                const float alpha = a_s / fmax(a2 + p_s * lam_prev, KLA_EPS);
                eta = alpha * eta + msi_t * k_t;
                var = 1.0f / fmax(lam, KLA_EPS);
            }
            const float mean_f = eta * var;
            const float var_o = (prior != 0) ? (a2 * var + p_s) : var;
            const float mean_o = (prior != 0) ? (a_s * mean_f) : mean_f;
            // Lanes with nothing to contribute carry q_t = 0, so the reduction
            // stays collective without them affecting the sum.
            const float q_t = (t < L && active) ? qw[kla_bls(b, t, s, L, S)] : 0.0f;
            const float2 out = kla_sum_over_states(
                float2(mean_o * q_t, var_o * q_t * q_t), red_s, tidx, s);
            if (s == 0 && t < L) {
                y[kla_blm(b, t, m, L, M)] = out.x;
                yvar[kla_blm(b, t, m, L, M)] = out.y;
            }
        }
        // The last thread holds the state at the tile boundary (or at L, if the
        // sequence ended inside this tile -- later threads then never updated).
        carry = kla_tile_broadcast2(float2(lam, eta), bcast_s, s, ty);
    }

    if (active && ty == 0) {
        lam_fin[kla_bms(b, m, s, M, S)] = carry.x;
        eta_fin[kla_bms(b, m, s, M, S)] = carry.y;
    }
}
