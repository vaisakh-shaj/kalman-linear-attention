/******************************************************************************
 * The merged 3x3 map -- shared by every "merged" Metal cell.
 *   kla_merged.metal
 *
 * The other implementations compose *two* maps: a 2x2 Moebius map for the
 * precision lambda, and then an affine map for the information vector eta. The
 * second one cannot start until the first has finished, because the Kalman gain
 * alpha_t reads lambda_{t-1} -- so every composing implementation runs two
 * scans, and everything about its phase structure follows from that.
 *
 * Written in homogeneous coordinates the dependency disappears. With lambda =
 * u/v the precision map is already the 2x2 the other files carry, and from
 * C = p/a^2, D = 1 the second coordinate steps as
 *
 *     v_t = C.u_{t-1} + D.v_{t-1} = den_t . v_{t-1} / a^2,
 *
 * so the gain is a ratio of a coordinate the precision scan already has:
 * alpha_t = a/den_t = v_{t-1} / (a . v_t). Putting w = v.eta the v_t cancels and
 * the whole step is one linear map:
 *
 *     [u]   [ A     B     0  ] [u]        A = (1+p.phi)/a^2   B = phi
 *     [v] = [ C     D     0  ] [v]        C = p/a^2           D = 1
 *     [w]   [r.C   r.D   1/a ] [w]        lambda = u/v,  eta = w/v
 *
 * The leaf is built from (phi, r, a, p) alone. Nothing in it reads lambda,
 * which is the entire point: one scan, and the affine phases go away.
 *
 * Lower block-triangular with a scalar (3,3), so a compose is never a full 3x3
 * product -- it is the same 2x2 product as before, plus a 1x2 row and one
 * scalar multiply. ~15 multiplies against ~14 for the 2x2-plus-affine pair, so
 * the arithmetic is a wash and the win is structural.
 *
 * NORMALIZATION IS LOAD-BEARING HERE, more so than for the 2x2. The (3,3) entry
 * accumulates a^-n, which overflows float32 outright for a decaying filter
 * (7e142 at a=0.5 over 200 steps, measured). Dividing all seven entries by the
 * 2x2 block's trace fixes it and is free, because lambda = u/v and eta = w/v are
 * both invariant under a common rescale of (u,v,w). The product of traces grows
 * faster than a^-n, so the normalized s *decays* to zero -- which is the right
 * physics: the initial eta stops mattering. tests/test_merged_algebra.py pins
 * all of this, including the overflow, before any of it reaches a GPU.
 *
 * Seven values are carried, not six. After a compose D = 1 - A, so D looks
 * redundant -- but D is the small entry (roughly a^2/(1+p.phi)), and recovering
 * it as 1 - A would give it an absolute error of one ulp of A, which is a
 * *relative* error of eps/D. At D ~ 1e-3 that is 1e-4 in the entry lambda is
 * most sensitive to, to save one register. Not worth it. The eighth slot is
 * padding, so the aggregate is two float4 loads rather than seven scalar ones;
 * a [...,7] buffer would stride every thread's read across the vector width.
 ******************************************************************************/

struct KlaMerged {
    float4 P;  // (A, B, C, D)   the precision block, exactly as the 2x2 cells
    float4 Q;  // (qa, qb, s, _) the eta row and the (3,3) entry
};

inline KlaMerged kla_mrg_identity() {
    KlaMerged m;
    m.P = float4(1.0f, 0.0f, 0.0f, 1.0f);
    m.Q = float4(0.0f, 0.0f, 1.0f, 0.0f);
    return m;
}

// One timestep. inv_a2 = 1/a^2 and inv_a = 1/a are loop-invariant per lane.
inline KlaMerged kla_mrg_leaf(float phi, float r, float p_s, float inv_a2,
                              float inv_a) {
    const float C = p_s * inv_a2;
    KlaMerged m;
    m.P = float4((1.0f + p_s * phi) * inv_a2, phi, C, 1.0f);
    m.Q = float4(r * C, r, inv_a, 0.0f);  // (r.C, r.D with D = 1, 1/a)
    return m;
}

// The map "R after L", trace-normalized. P = P2.P1, q = q2.P1 + s2.q1, s = s2.s1.
inline KlaMerged kla_mrg_compose(KlaMerged L, KlaMerged R) {
    const float a = R.P.x * L.P.x + R.P.y * L.P.z;
    const float b = R.P.x * L.P.y + R.P.y * L.P.w;
    const float c = R.P.z * L.P.x + R.P.w * L.P.z;
    const float d = R.P.z * L.P.y + R.P.w * L.P.w;
    const float qa = R.Q.x * L.P.x + R.Q.y * L.P.z + R.Q.z * L.Q.x;
    const float qb = R.Q.x * L.P.y + R.Q.y * L.P.w + R.Q.z * L.Q.y;
    const float s = R.Q.z * L.Q.z;
    const float inv = 1.0f / fmax(a + d, KLA_EPS);
    KlaMerged m;
    m.P = float4(a, b, c, d) * inv;
    m.Q = float4(qa, qb, s, 0.0f) * inv;
    return m;
}

// Apply to the homogeneous vector (lam, 1, eta) -> (lambda, eta). Both quotients
// share the denominator C.lam + D, which is the one the 2x2 cells already form.
inline float2 kla_mrg_apply(KlaMerged m, float lam, float eta) {
    const float den = fmax(m.P.z * lam + m.P.w, KLA_EPS);
    return float2((m.P.x * lam + m.P.y) / den,
                  (m.Q.x * lam + m.Q.y + m.Q.z * eta) / den);
}
