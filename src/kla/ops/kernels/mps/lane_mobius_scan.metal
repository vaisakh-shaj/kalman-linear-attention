/******************************************************************************
 * KLA Metal kernels — composed strategy (the "triton-shaped" path)
 *   lane_mobius_scan.metal
 *
 * Three primitives the composed MPS backend builds its forward and its exact
 * backward out of, each consuming pre-built per-(b,l,m,s) coefficients:
 *
 *   kla_mobius_scan_fwd   lambda_t = (A_t.lambda' + B_t)/(C_t.lambda' + D_t)
 *   kla_affine_scan_fwd   eta_t    = alpha_t.eta_{t-1} + r_t        (eta_-1 = 0)
 *   kla_affine_scan_rev   nu_t     = src_t + mult_{t+1}.nu_{t+1}
 *
 * One thread owns one (b, m, s) triple and walks the sequence; the grid is
 * flat, B*M*S wide. There is no cross-thread communication at all here, so
 * unlike the fused kernels these can and do return early on the padding lanes.
 *
 * Two things follow from time being the serial axis rather than the parallel
 * one, and both are departures from the triton kernels these mirror:
 *
 * 1. The Moebius map is *applied*, never *composed*. The triton and CUDA scans
 *    multiply 2x2 matrices and divide by the trace after every compose, because
 *    an associative scan over time has to combine maps that share no common
 *    lambda. Applying the map to a running lambda needs no matrix, no
 *    normalizer, and cannot overflow the way composing two raw leaves can (the
 *    A^2 ~ 1e40 case that tests/test_backends.py pins down at a = 1e-10).
 *
 * 2. Nothing is permuted. The triton wrappers move [B,L,M,S] to [B,M,L,S] so a
 *    program's tile is contiguous; here the natural torch layout is already
 *    right, since adjacent threads read adjacent s (and adjacent m) at a fixed
 *    t. The time stride M*S is walked once per step.
 *
 * kla_affine_scan_rev exists for the same reason: triton's backward runs its
 * reverse recurrence by flipping the sequence, shifting, and reusing the
 * forward scan, which costs three extra [B,L,M,S] materializations. Walking t
 * downwards is free here.
 ******************************************************************************/

kernel void kla_mobius_scan_fwd(
    device float *out         [[buffer(0)]],  // [B, L, M, S] lambda_t
    device const float *A     [[buffer(1)]],  // [B, L, M, S]
    device const float *Bc    [[buffer(2)]],
    device const float *C     [[buffer(3)]],
    device const float *D     [[buffer(4)]],
    device const float *lam0  [[buffer(5)]],  // [B, M, S] initial precision
    constant int &N_          [[buffer(6)]],  // B*M*S, the thread count
    constant int &L_          [[buffer(7)]],
    constant int &M_          [[buffer(8)]],
    constant int &S_          [[buffer(9)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(N_)) { return; }
    const uint L = uint(L_), M = uint(M_), S = uint(S_);

    const uint s = gid % S;
    const uint m = (gid / S) % M;
    const uint b = gid / (S * M);

    float lam = lam0[kla_bms(b, m, s, M, S)];
    uint idx = kla_blms(b, 0, m, s, L, M, S);
    const uint stride = M * S;

    for (uint t = 0; t < L; ++t, idx += stride) {
        const float den = fmax(C[idx] * lam + D[idx], KLA_EPS);
        lam = (A[idx] * lam + Bc[idx]) / den;
        out[idx] = lam;
    }
}

kernel void kla_affine_scan_fwd(
    device float *out          [[buffer(0)]],  // [B, L, M, S] eta_t
    device const float *alpha  [[buffer(1)]],  // [B, L, M, S]
    device const float *r      [[buffer(2)]],
    constant int &N_           [[buffer(3)]],
    constant int &L_           [[buffer(4)]],
    constant int &M_           [[buffer(5)]],
    constant int &S_           [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(N_)) { return; }
    const uint L = uint(L_), M = uint(M_), S = uint(S_);

    const uint s = gid % S;
    const uint m = (gid / S) % M;
    const uint b = gid / (S * M);

    float eta = 0.0f;
    uint idx = kla_blms(b, 0, m, s, L, M, S);
    const uint stride = M * S;

    for (uint t = 0; t < L; ++t, idx += stride) {
        eta = alpha[idx] * eta + r[idx];
        out[idx] = eta;
    }
}

// nu_t = src_t + mult_{t+1}.nu_{t+1}, with nu_L = 0. The adjoint of both
// recurrences above is of this shape (each has a scalar per-step gain), so one
// reverse kernel serves the whole backward.
kernel void kla_affine_scan_rev(
    device float *out          [[buffer(0)]],  // [B, L, M, S] nu_t
    device const float *mult   [[buffer(1)]],  // [B, L, M, S] per-step gain
    device const float *src    [[buffer(2)]],  // [B, L, M, S] injected term
    constant int &N_           [[buffer(3)]],
    constant int &L_           [[buffer(4)]],
    constant int &M_           [[buffer(5)]],
    constant int &S_           [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(N_)) { return; }
    const uint L = uint(L_), M = uint(M_), S = uint(S_);

    const uint s = gid % S;
    const uint m = (gid / S) % M;
    const uint b = gid / (S * M);

    const uint stride = M * S;
    uint idx = kla_blms(b, L - 1, m, s, L, M, S);

    float nu = 0.0f;
    for (uint t = L; t-- > 0; idx -= stride) {
        // mult is read at t+1: the gain that carries nu_{t+1} back onto nu_t.
        nu = src[idx] + ((t + 1 < L) ? mult[idx + stride] * nu : 0.0f);
        out[idx] = nu;
    }
}
