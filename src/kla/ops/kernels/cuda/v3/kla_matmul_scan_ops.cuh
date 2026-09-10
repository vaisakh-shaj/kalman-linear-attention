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
//
// No observation-information floor/cap: match the linear torch recurrence.
// With k=0, phi=0 is a valid pure-prediction step. An artificial floor also
// suppresses useful small-key gradients. Process noise and a² stay floored.
__device__ __forceinline__ void compute_phi_r(
    float h, float sig_inv, float mu_sig,
    float &phi, float &r, float &raw_phi
) {
    raw_phi = h * h * sig_inv;
    phi = raw_phi;  // Linear torch path permits zero information (k == 0).
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
    // This denominator is projectively scaled, so an absolute EPS floor
    // would change the represented recurrence (e.g. precision > 1/EPS).
    // Positive leaves and positive initial precision guarantee den > 0.
    return num / den;
}

// Linear scan input
__device__ __forceinline__ float2 build_linear_input(
    float a_s, float a2, float q_s, float prev_lambda, float r,
    float &denom
) {
    denom = fmaxf(a2 + q_s * prev_lambda, KLA_EPS);
    float alpha_lin = a_s / denom;  // Match torch: no artificial gain floor.
    return make_float2(alpha_lin, r);
}

// Scalar backward formulas live in kla_matmul_bwd_kernel.cuh.
