/******************************************************************************
 * mps_pscan -- the forward as a reduce-then-scan, with no serial carry.
 *   pscan_kla_scan.metal
 *
 * recurrent_kla_scan.metal walks time serially. chunk_kla_scan.metal splits time
 * into tiles and walks the *tiles* serially, one threadgroup carrying (lambda,
 * eta) from each tile to the next. This file removes that last serial axis: the
 * sequence is cut into NCK chunks of KLA_CHUNK steps, every chunk is reduced
 * independently, and the cross-chunk dependency is resolved by a parallel scan
 * over the NCK aggregates rather than by anyone waiting for their neighbour.
 *
 * The trade is the usual one. No chunk waits, so the whole sequence is in
 * flight at once and the depth is log(NCK) instead of NCK -- but the aggregates
 * have to be *materialized* in device memory ([B,M,NCK,S] of them, two scans'
 * worth), and every element is touched about three times instead of once. This
 * is the implementation for a short-and-wide launch that the other two starve on, not
 * the one to reach for by default.
 *
 * It stays "fused" in the sense this repo uses the word: the intermediates are
 * per *chunk*, [B,M,NCK,S], never the per-timestep [B,L,M,S] the unfused torch
 * path builds. At KLA_CHUNK = 16 that is a 16x smaller footprint, and the two
 * lambda/eta arrays it does write are the backward's checkpoints -- which the
 * implementation produces for free, since "the state entering chunk c" is exactly
 * what the scan over aggregates computes.
 *
 * Five kernels, two reduce-scan-apply rounds:
 *
 *   1 mob_reduce   compose each chunk's KLA_CHUNK Moebius leaves   -> mob[c]
 *   2 mob_step     one doubling round over the NCK aggregates      (log2 NCK x)
 *   3 aff_reduce   lambda entering chunk c  = apply(mob[c-1], lam0) -> lam_ck
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
 * that is KLA_CHUNK times shorter than the sequence, and it has no threadgroup
 * size ceiling -- NCK is bounded by nothing here, so a scan that had to fit one
 * threadgroup would need a second level to cover long sequences.
 *
 * Backward: kla_scan_bwd, unchanged and unaware. lam_ck / eta_ck carry the same
 * layout, the same stride and the same convention (the value *entering* step t,
 * for t a multiple of KLA_CHUNK) as both other forwards write.
 ******************************************************************************/

// The 2x2 Moebius and affine algebra is chunk_kla_scan.metal's, verbatim in
// meaning: float4(A,B,C,D) acting as lambda -> (A.lambda+B)/(C.lambda+D), and
// kla_pmob_compose(L, R) is "R after L", trace-normalized so the entries stay
// O(1) without leaving linear space.

inline float4 kla_pmob_identity() { return float4(1.0f, 0.0f, 0.0f, 1.0f); }

inline float4 kla_pmob_compose(float4 L, float4 R) {
    const float a = R.x * L.x + R.y * L.z;
    const float b = R.x * L.y + R.y * L.w;
    const float c = R.z * L.x + R.w * L.z;
    const float d = R.z * L.y + R.w * L.w;
    return float4(a, b, c, d) / fmax(a + d, KLA_EPS);
}

inline float kla_pmob_apply(float4 P, float lam) {
    return (P.x * lam + P.y) / fmax(P.z * lam + P.w, KLA_EPS);
}

inline float2 kla_paff_identity() { return float2(1.0f, 0.0f); }

inline float2 kla_paff_compose(float2 L, float2 R) {
    return float2(R.x * L.x, R.x * L.y + R.y);
}

// Aggregate index: [B, M, NCK, S], the checkpoint layout with a wider element.
inline uint kla_agg(uint b, uint m, uint c, uint s, uint M, uint NCK, uint S) {
    return ((b * M + m) * NCK + c) * S + s;
}

// ------------------------------------------------------- 1: reduce the chunks

kernel void kla_pscan_mob_reduce(
    device float4 *mob        [[buffer(0)]],   // out [B, M, NCK, S]
    device const float *si    [[buffer(1)]],   // Lambda^v [B, L, M]
    device const float *k     [[buffer(2)]],   // key      [B, L, S]
    device const float *p     [[buffer(3)]],   // process noise [M, S]
    device const float *a     [[buffer(4)]],   // decay         [M, S]
    constant int &L_          [[buffer(5)]],
    constant int &M_          [[buffer(6)]],
    constant int &S_          [[buffer(7)]],
    constant int &NCK_        [[buffer(8)]],
    constant int &total_      [[buffer(9)]],
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

    // Steps past the end of the sequence contribute the identity, so a partial
    // final chunk composes to exactly what its live prefix does.
    float4 agg = kla_pmob_identity();
    for (uint i = 0; i < KLA_CHUNK; ++i) {
        const uint t = c * KLA_CHUNK + i;
        if (t < L) {
            const float si_t = si[kla_blm(b, t, m, L, M)];
            const float k_t = k[kla_bls(b, t, s, L, S)];
            const float phi = fmax(si_t * k_t * k_t, KLA_EPS);
            agg = kla_pmob_compose(
                agg, float4((1.0f + p_s * phi) * inv_a2, phi, p_s * inv_a2, 1.0f));
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

kernel void kla_pscan_mob_step(
    device float4 *dst        [[buffer(0)]],
    device const float4 *src  [[buffer(1)]],
    constant int &S_          [[buffer(2)]],
    constant int &NCK_        [[buffer(3)]],
    constant int &off_        [[buffer(4)]],
    constant int &total_      [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(total_)) { return; }
    const uint S = uint(S_), NCK = uint(NCK_), off = uint(off_);
    const uint c = (gid / S) % NCK;
    dst[gid] = (c >= off) ? kla_pmob_compose(src[gid - off * S], src[gid]) : src[gid];
}

kernel void kla_pscan_aff_step(
    device float2 *dst        [[buffer(0)]],
    device const float2 *src  [[buffer(1)]],
    constant int &S_          [[buffer(2)]],
    constant int &NCK_        [[buffer(3)]],
    constant int &off_        [[buffer(4)]],
    constant int &total_      [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(total_)) { return; }
    const uint S = uint(S_), NCK = uint(NCK_), off = uint(off_);
    const uint c = (gid / S) % NCK;
    dst[gid] = (c >= off) ? kla_paff_compose(src[gid - off * S], src[gid]) : src[gid];
}

// -------------------------------------- 3: seed lambda, then reduce the affine

kernel void kla_pscan_aff_reduce(
    device float2 *aff          [[buffer(0)]],   // out [B, M, NCK, S]
    device float *lam_ck        [[buffer(1)]],   // out [B, M, NCK, S]
    device const float4 *mob_in [[buffer(2)]],   // inclusive prefix of the above
    device const float *msi     [[buffer(3)]],   // v.Lambda^v [B, L, M]
    device const float *si      [[buffer(4)]],   // Lambda^v   [B, L, M]
    device const float *k       [[buffer(5)]],   // key        [B, L, S]
    device const float *a       [[buffer(6)]],
    device const float *p       [[buffer(7)]],
    device const float *lam0    [[buffer(8)]],   // [B, M, S]
    constant int &L_            [[buffer(9)]],
    constant int &M_            [[buffer(10)]],
    constant int &S_            [[buffer(11)]],
    constant int &NCK_          [[buffer(12)]],
    constant int &total_        [[buffer(13)]],
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
    const float a2 = fmax(a_s * a_s, KLA_EPS);
    const float inv_a2 = 1.0f / a2;

    // The exclusive prefix maps the initial state to the state entering this
    // chunk. Chunk 0 has none, and is seeded with lam0 itself.
    const float lam_start = lam0[kla_bms(b, m, s, M, S)];
    float lam = (c == 0) ? lam_start
                         : kla_pmob_apply(mob_in[kla_agg(b, m, c - 1, s, M, NCK, S)],
                                          lam_start);
    lam_ck[gid] = lam;

    float2 agg = kla_paff_identity();
    for (uint i = 0; i < KLA_CHUNK; ++i) {
        const uint t = c * KLA_CHUNK + i;
        if (t < L) {
            const float si_t = si[kla_blm(b, t, m, L, M)];
            const float msi_t = msi[kla_blm(b, t, m, L, M)];
            const float k_t = k[kla_bls(b, t, s, L, S)];
            const float phi = fmax(si_t * k_t * k_t, KLA_EPS);
            // The gain reads lambda_{t-1}, so it is formed before the update.
            const float alpha = a_s / fmax(a2 + p_s * lam, KLA_EPS);
            agg = kla_paff_compose(agg, float2(alpha, msi_t * k_t));
            float A, C, den;
            lam = kla_lambda_step(lam, phi, p_s, inv_a2, A, C, den);
        }
    }
    aff[gid] = agg;
}

// ------------------------------ 5: seed eta, replay the chunk, and read out
//
// The only kernel here that reduces, so it is the only one with a threadgroup
// shape: [KLA_BLOCK_S, KLA_ROWS] with x over states and y over *chunks*, so the
// KLA_BLOCK_S lanes of a row hold every state of one (b, m, chunk) and the
// read-out sum over s is a lane reduction. Every thread must reach every
// reduction, including the padding lanes at s >= d_state and the rows past
// NCK, so nothing below returns early.

kernel void kla_pscan_apply(
    device float *y             [[buffer(0)]],   // out [B, L, M]
    device float *yvar          [[buffer(1)]],   // out [B, L, M]
    device float *lam_fin       [[buffer(2)]],   // out [B, M, S]
    device float *eta_fin       [[buffer(3)]],   // out [B, M, S]
    device float *eta_ck        [[buffer(4)]],   // out [B, M, NCK, S]
    device const float *lam_ck  [[buffer(5)]],   // in  [B, M, NCK, S]
    device const float2 *aff_in [[buffer(6)]],   // inclusive prefix of the affine
    device const float *msi     [[buffer(7)]],
    device const float *si      [[buffer(8)]],
    device const float *k       [[buffer(9)]],
    device const float *qw      [[buffer(10)]],
    device const float *a       [[buffer(11)]],
    device const float *p       [[buffer(12)]],
    device const float *eta0    [[buffer(13)]],  // [B, M, S]
    constant int &L_            [[buffer(14)]],
    constant int &M_            [[buffer(15)]],
    constant int &S_            [[buffer(16)]],
    constant int &NCK_          [[buffer(17)]],
    constant int &prior         [[buffer(18)]],  // decode_from_prior
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

    const uint agg = active ? kla_agg(b, m, c, s, M, NCK, S) : 0u;
    float lam = active ? lam_ck[agg] : 1.0f;

    const float eta_start = active ? eta0[kla_bms(b, m, s, M, S)] : 0.0f;
    float eta = eta_start;
    if (active && c > 0) {
        const float2 pref = aff_in[kla_agg(b, m, c - 1, s, M, NCK, S)];
        eta = pref.x * eta_start + pref.y;
    }
    if (active) { eta_ck[agg] = eta; }

    for (uint i = 0; i < KLA_CHUNK; ++i) {
        const uint t = c * KLA_CHUNK + i;
        const bool live = active && (t < L);
        float var = 0.0f;
        if (live) {
            const float si_t = si[kla_blm(b, t, m, L, M)];
            const float msi_t = msi[kla_blm(b, t, m, L, M)];
            const float k_t = k[kla_bls(b, t, s, L, S)];
            const float phi = fmax(si_t * k_t * k_t, KLA_EPS);
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
