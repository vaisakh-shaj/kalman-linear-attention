/******************************************************************************
 * mps_merged_pscan -- pscan_kla_scan.metal's five kernels, in three.
 *   merged_pscan_kla_scan.metal
 *
 * Same structure as pscan_kla_scan.metal: the sequence is cut into NCK chunks
 * of KLA_CHUNK steps, every chunk is reduced independently, and the cross-chunk
 * dependency is settled by a Hillis-Steele scan over the NCK aggregates rather
 * than by anyone waiting for their neighbour. Same [B,M,NCK,S] footprint, same
 * checkpoints, same backward.
 *
 * What goes away is the second round. That file runs reduce-scan-apply *twice*:
 * once for the Moebius map, and then again for the affine one, because the
 * affine leaf alpha_t reads lambda_{t-1} and so does not exist until the first
 * round has produced lambda. The 3x3 leaf of kla_merged.metal is built from
 * (phi, r, a, p) alone, so there is no second round to run:
 *
 *   1 reduce   compose each chunk's KLA_CHUNK 3x3 leaves        -> mrg[c]
 *   2 step     one doubling round over the NCK aggregates       (log2 NCK x)
 *   3 apply    (lambda, eta) entering chunk c = mrg[c-1] applied to
 *              (lambda0, eta0); write both checkpoints, walk the chunk
 *              applying both recurrences, and read out
 *
 * against five kernels and two rounds. Each element is touched about twice
 * rather than three times, and the scan over aggregates -- the part whose depth
 * is log(NCK) rather than O(1) -- runs once rather than twice.
 *
 * Memory is close to a wash. The aggregate is 8 floats (7 live, 1 padding)
 * against a float4 plus a float2, and it ping-pongs one pair of buffers rather
 * than two: 16 floats per (b,m,c,s) against 12. In exchange lam_ck no longer
 * needs a kernel of its own to produce it -- kernel 3 writes both checkpoints
 * from the one prefix it applies.
 *
 * Backward: kla_scan_bwd, unchanged and unaware. lam_ck / eta_ck carry the same
 * layout, the same stride and the same convention (the value *entering* step t,
 * for t a multiple of KLA_CHUNK) that all four other forwards write.
 ******************************************************************************/

// Aggregate index: [B, M, NCK, S], the checkpoint layout with a wider element.
inline uint kla_magg(uint b, uint m, uint c, uint s, uint M, uint NCK, uint S) {
    return ((b * M + m) * NCK + c) * S + s;
}

// ------------------------------------------------------- 1: reduce the chunks

kernel void kla_merged_pscan_reduce(
    device KlaMerged *mrg    [[buffer(0)]],   // out [B, M, NCK, S]
    device const float *msi  [[buffer(1)]],   // v.Lambda^v [B, L, M]
    device const float *si   [[buffer(2)]],   // Lambda^v   [B, L, M]
    device const float *k    [[buffer(3)]],   // key        [B, L, S]
    device const float *a    [[buffer(4)]],   // decay         [M, S]
    device const float *p    [[buffer(5)]],   // process noise [M, S]
    constant int &L_         [[buffer(6)]],
    constant int &M_         [[buffer(7)]],
    constant int &S_         [[buffer(8)]],
    constant int &NCK_       [[buffer(9)]],
    constant int &total_     [[buffer(10)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(total_)) { return; }
    const uint L = uint(L_), M = uint(M_), S = uint(S_), NCK = uint(NCK_);
    const uint s = gid % S;
    uint rest = gid / S;
    const uint c = rest % NCK;
    rest /= NCK;
    const uint m = rest % M, b = rest / M;

    const float a_s = a[m * S + s];
    const float p_s = p[m * S + s];
    const float inv_a2 = 1.0f / fmax(a_s * a_s, KLA_EPS);
    const float inv_a = 1.0f / (a_s + copysign(KLA_EPS, a_s));

    // Steps past the end of the sequence contribute the identity, so a partial
    // final chunk composes to exactly what its live prefix does.
    KlaMerged agg = kla_mrg_identity();
    for (uint i = 0; i < KLA_CHUNK; ++i) {
        const uint t = c * KLA_CHUNK + i;
        if (t < L) {
            const float si_t = si[kla_blm(b, t, m, L, M)];
            const float msi_t = msi[kla_blm(b, t, m, L, M)];
            const float k_t = k[kla_bls(b, t, s, L, S)];
            const float phi = fmax(si_t * k_t * k_t, KLA_EPS);
            agg = kla_mrg_compose(agg,
                                  kla_mrg_leaf(phi, msi_t * k_t, p_s, inv_a2, inv_a));
        }
    }
    mrg[gid] = agg;
}

// --------------------------------------------------------- 2: one doubling round
//
// dst[c] = compose(src[c - off], src[c]) for c >= off, src[c] otherwise. Run
// with off = 1, 2, 4, ... < NCK and the buffers swapped, this leaves the
// inclusive prefix in the last destination. The chunk axis has stride S in the
// flat index, so the neighbour is gid - off*S. One of these where the two-scan
// implementation runs two.

kernel void kla_merged_pscan_step(
    device KlaMerged *dst       [[buffer(0)]],
    device const KlaMerged *src [[buffer(1)]],
    constant int &S_            [[buffer(2)]],
    constant int &NCK_          [[buffer(3)]],
    constant int &off_          [[buffer(4)]],
    constant int &total_        [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(total_)) { return; }
    const uint S = uint(S_), NCK = uint(NCK_), off = uint(off_);
    const uint c = (gid / S) % NCK;
    dst[gid] = (c >= off) ? kla_mrg_compose(src[gid - off * S], src[gid]) : src[gid];
}

// ------------------------- 3: seed both states, replay the chunk, and read out
//
// The only kernel here that reduces, so it is the only one with a threadgroup
// shape: [KLA_BLOCK_S, KLA_ROWS] with x over states and y over *chunks*, so the
// KLA_BLOCK_S lanes of a row hold every state of one (b, m, chunk) and the
// read-out sum over s is a lane reduction. Every thread must reach every
// reduction, including the padding lanes at s >= d_state and the rows past NCK,
// so nothing below returns early.

kernel void kla_merged_pscan_apply(
    device float *y              [[buffer(0)]],   // out [B, L, M]
    device float *yvar           [[buffer(1)]],   // out [B, L, M]
    device float *lam_fin        [[buffer(2)]],   // out [B, M, S]
    device float *eta_fin        [[buffer(3)]],   // out [B, M, S]
    device float *lam_ck         [[buffer(4)]],   // out [B, M, NCK, S]
    device float *eta_ck         [[buffer(5)]],   // out [B, M, NCK, S]
    device const KlaMerged *mrg_in [[buffer(6)]], // inclusive prefix of the above
    device const float *msi      [[buffer(7)]],
    device const float *si       [[buffer(8)]],
    device const float *k        [[buffer(9)]],
    device const float *qw       [[buffer(10)]],
    device const float *a        [[buffer(11)]],
    device const float *p        [[buffer(12)]],
    device const float *lam0     [[buffer(13)]],  // [B, M, S]
    device const float *eta0     [[buffer(14)]],  // [B, M, S]
    constant int &L_             [[buffer(15)]],
    constant int &M_             [[buffer(16)]],
    constant int &S_             [[buffer(17)]],
    constant int &NCK_           [[buffer(18)]],
    constant int &prior          [[buffer(19)]],  // decode_from_prior
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpt  [[thread_position_in_threadgroup]],
    uint  tidx [[thread_index_in_threadgroup]]) {
    threadgroup float2 red_s[KLA_TG_THREADS];

    const uint L = uint(L_), M = uint(M_), S = uint(S_), NCK = uint(NCK_);
    const uint s = tpt.x, ty = tpt.y;
    const uint c = tgid.y * KLA_ROWS + ty;
    const uint bm = tgid.z;
    const uint m = bm % M, b = bm / M;
    const bool active = (s < S) && (c < NCK);

    const float a_s = active ? a[m * S + s] : 1.0f;
    const float p_s = active ? p[m * S + s] : 0.0f;
    const float a2 = fmax(a_s * a_s, KLA_EPS);
    const float inv_a2 = 1.0f / a2;

    // The exclusive prefix maps the initial state to the state entering this
    // chunk -- both coordinates at once, which is the whole saving. Chunk 0 has
    // no prefix and is seeded with (lambda0, eta0) themselves.
    const uint agg = active ? kla_magg(b, m, c, s, M, NCK, S) : 0u;
    float lam = active ? lam0[kla_bms(b, m, s, M, S)] : 1.0f;
    float eta = active ? eta0[kla_bms(b, m, s, M, S)] : 0.0f;
    if (active && c > 0) {
        const float2 st =
            kla_mrg_apply(mrg_in[kla_magg(b, m, c - 1, s, M, NCK, S)], lam, eta);
        lam = st.x;
        eta = st.y;
    }
    if (active) {
        lam_ck[agg] = lam;
        eta_ck[agg] = eta;
    }

    for (uint i = 0; i < KLA_CHUNK; ++i) {
        const uint t = c * KLA_CHUNK + i;
        const bool live = active && (t < L);
        float var = 0.0f;
        if (live) {
            const float si_t = si[kla_blm(b, t, m, L, M)];
            const float msi_t = msi[kla_blm(b, t, m, L, M)];
            const float k_t = k[kla_bls(b, t, s, L, S)];
            const float phi = fmax(si_t * k_t * k_t, KLA_EPS);
            // The gain reads lambda_{t-1}, which this walk has in hand.
            const float alpha = a_s / fmax(a2 + p_s * lam, KLA_EPS);
            float A, C, den;
            lam = kla_lambda_step(lam, phi, p_s, inv_a2, A, C, den);
            eta = alpha * eta + msi_t * k_t;
            var = 1.0f / fmax(lam, KLA_EPS);
        }
        const float mean_f = eta * var;
        const float var_o = (prior != 0) ? (a2 * var + p_s) : var;
        const float mean_o = (prior != 0) ? (a_s * mean_f) : mean_f;
        // Lanes with nothing to contribute carry q_t = 0, so the reduction stays
        // collective without them affecting the sum.
        const float q_t = live ? qw[kla_bls(b, t, s, L, S)] : 0.0f;
        const float2 out = kla_sum_over_states(
            float2(mean_o * q_t, var_o * q_t * q_t), red_s, tidx, s);
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
