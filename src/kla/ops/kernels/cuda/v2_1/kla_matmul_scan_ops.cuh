/******************************************************************************
 * KLA Matmul Scan — Scan Operators & Gradient Functions
 *   kla_matmul_scan_ops.cuh
 *
 * Linear-space 2×2 matmul with trace-normalization.
 *
 * Forward:
 *   TraceNormMatMulOp  — associative compose: R*L / trace(R*L)
 *   matmul_identity    — I₂ = (1,0,0,1)
 *   build_leaf_matrix  — M_t from (φ, q, 1/a²)
 *   extract_lambda_lin — λ = (A+B)/(C+D)
 *   compute_phi_r      — observation sufficient statistics
 *   build_linear_input — α^lin, denom for η scan
 *   LinearScanOp       — (α₁,η₁)⊕(α₂,η₂) = (α₂α₁, α₂η₁+η₂)
 *
 * Backward:
 *   grad_lambda_extraction — d̄λ → d̄P
 *   grad_through_trace_norm — d̄P_norm → d̄P_raw
 *   grad_matmul_prefix     — d̄P_raw → d̄P_L (propagation)
 *   grad_matmul_element    — d̄P_raw → d̄M_R (parameter grads)
 *
 * Prefix callbacks:
 *   MatmulPrefixCallbackOp — cross-chunk Möbius carry
 *   LinearPrefixCallbackOp — cross-chunk linear carry
 ******************************************************************************/
#pragma once

#include <cuda_runtime.h>
#include <math.h>
#include "kla_matmul_scan_common.h"

// =============================================================================
// Identity elements
// =============================================================================
__device__ __forceinline__ float4 matmul_identity() {
    return make_float4(1.0f, 0.0f, 0.0f, 1.0f);  // I₂ = [[1,0],[0,1]]
}

__device__ __forceinline__ float2 linear_identity() {
    return make_float2(1.0f, 0.0f);
}

// =============================================================================
// TraceNormMatMulOp: 2×2 matmul with trace-normalization
//
// CUB convention: L = running aggregate (prefix), R = new element
// Result = R * L, then divide by trace(R * L) = A_raw + D_raw
//
// Trace-norm keeps entries O(1): A_norm + D_norm = 1 exactly.
// No subtraction → no catastrophic cancellation.
// λ = (A+B)/(C+D) is unchanged (scale-invariant).
// =============================================================================
struct TraceNormMatMulOp {
    __device__ __forceinline__ float4 operator()(const float4 &L, const float4 &R) const {
        // Product = R * L  (R on left, L on right — matrix multiply)
        float A_raw = R.x * L.x + R.y * L.z;
        float B_raw = R.x * L.y + R.y * L.w;
        float C_raw = R.z * L.x + R.w * L.z;
        float D_raw = R.z * L.y + R.w * L.w;

        // Trace-normalize: divide by A + D (always positive for positive matrices)
        float inv_tr = 1.0f / fmaxf(A_raw + D_raw, KLA_EPS);

        return make_float4(
            A_raw * inv_tr,
            B_raw * inv_tr,
            C_raw * inv_tr,
            D_raw * inv_tr
        );
    }
};

// =============================================================================
// LinearScanOp: affine recurrence (α₁,η₁) ⊕ (α₂,η₂) = (α₂·α₁, α₂·η₁ + η₂)
// =============================================================================
struct LinearScanOp {
    __device__ __forceinline__ float2 operator()(const float2 &a, const float2 &b) const {
        return make_float2(b.x * a.x, b.x * a.y + b.y);
    }
};

// =============================================================================
// Prefix callback functors (cross-chunk carry)
// =============================================================================
struct MatmulPrefixCallbackOp {
    float4 running_prefix;
    __device__ MatmulPrefixCallbackOp(float4 prefix) : running_prefix(prefix) {}
    __device__ float4 operator()(float4 block_aggregate) {
        float4 old_prefix = running_prefix;
        running_prefix = TraceNormMatMulOp()(old_prefix, block_aggregate);
        return old_prefix;
    }
};

struct LinearPrefixCallbackOp {
    float2 running_prefix;
    __device__ LinearPrefixCallbackOp(float2 prefix) : running_prefix(prefix) {}
    __device__ float2 operator()(float2 block_aggregate) {
        float2 old_prefix = running_prefix;
        running_prefix = LinearScanOp()(old_prefix, block_aggregate);
        return old_prefix;
    }
    __device__ float2 identity() const { return linear_identity(); }
};

// =============================================================================
// SHARED DEVICE FUNCTIONS — bit-exact between fwd and bwd
// =============================================================================

// Observation sufficient statistics
__device__ __forceinline__ void compute_phi_r(
    float h, float sig_inv, float mu_sig,
    float &phi, float &r, float &raw_phi
) {
    raw_phi = h * h * sig_inv;
    phi = fminf(fmaxf(raw_phi, KLA_EPS), 1000.0f);
    r = mu_sig * h;
}

// Build leaf matrix in linear-space
// M_t = [[(1+qφ)/a², φ], [q/a², 1]]
__device__ __forceinline__ float4 build_leaf_matrix(
    float phi, float q_s, float inv_a2
) {
    float A = (1.0f + q_s * phi) * inv_a2;
    float B = phi;
    float C = q_s * inv_a2;
    float D = 1.0f;
    return make_float4(A, B, C, D);
}

// Extract λ = (A+B)/(C+D)
__device__ __forceinline__ float extract_lambda_lin(const float4 &P) {
    float num = P.x + P.y;
    float den = P.z + P.w;
    return num / fmaxf(den, KLA_EPS);
}

// Linear scan input
__device__ __forceinline__ float2 build_linear_input(
    float a_s, float a2, float q_s, float prev_lambda, float r,
    float &denom
) {
    denom = fmaxf(a2 + q_s * prev_lambda, KLA_EPS);
    float alpha_lin = fmaxf(a_s / denom, KLA_EPS);
    return make_float2(alpha_lin, r);
}

// =============================================================================
// BACKWARD FUNCTIONS
// =============================================================================

// Gradient of λ extraction w.r.t. accumulated matrix P
__device__ __forceinline__ float4 grad_lambda_extraction(
    float dlam, const float4 &P
) {
    float den = P.z + P.w;
    float inv_den = 1.0f / fmaxf(den, KLA_EPS);
    float lam = (P.x + P.y) * inv_den;
    float gnum = dlam * inv_den;
    float gden = -dlam * lam * inv_den;
    return make_float4(gnum, gnum, gden, gden);
}

// Gradient through trace-normalization
// Input:  gP_norm (upstream), P_norm (trace-normalized matrix), trace (pre-norm trace)
// Output: gP_raw
//
// P_norm = P_raw / tau where tau = A_raw + D_raw
// s = sum_i gP_norm_i * P_norm_i
// Diagonal (A,D): gX_raw = (gX_norm - s) / tau
// Off-diag (B,C): gX_raw = gX_norm / tau
__device__ __forceinline__ float4 grad_through_trace_norm(
    const float4 &gP_norm, const float4 &P_norm, float trace
) {
    float inv_tr = 1.0f / fmaxf(trace, KLA_EPS);

    // s = gA_n * A_n + gB_n * B_n + gC_n * C_n + gD_n * D_n
    float s = gP_norm.x * P_norm.x + gP_norm.y * P_norm.y
            + gP_norm.z * P_norm.z + gP_norm.w * P_norm.w;

    return make_float4(
        (gP_norm.x - s) * inv_tr,   // A: diagonal, gets -s
         gP_norm.y      * inv_tr,    // B: off-diagonal, no correction
         gP_norm.z      * inv_tr,    // C: off-diagonal, no correction
        (gP_norm.w - s) * inv_tr     // D: diagonal, gets -s
    );
}

// Gradient w.r.t. P_L (prefix) through raw matmul P_raw = M_R * P_L
// gP_L = M_R^T * gP_raw
__device__ __forceinline__ float4 grad_matmul_prefix(
    const float4 &gP_raw, const float4 &M_R
) {
    return make_float4(
        M_R.x * gP_raw.x + M_R.z * gP_raw.z,
        M_R.x * gP_raw.y + M_R.z * gP_raw.w,
        M_R.y * gP_raw.x + M_R.w * gP_raw.z,
        M_R.y * gP_raw.y + M_R.w * gP_raw.w
    );
}

// Gradient w.r.t. M_R (current element) through raw matmul P_raw = M_R * P_L
// gM_R = gP_raw * P_L^T
__device__ __forceinline__ float4 grad_matmul_element(
    const float4 &gP_raw, const float4 &P_L
) {
    return make_float4(
        gP_raw.x * P_L.x + gP_raw.y * P_L.y,
        gP_raw.x * P_L.z + gP_raw.y * P_L.w,
        gP_raw.z * P_L.x + gP_raw.w * P_L.y,
        gP_raw.z * P_L.z + gP_raw.w * P_L.w
    );
}
