/******************************************************************************
 * KLA Matmul Backward Kernel — Static a,q, Linear-Space Det-Norm
 *   kla_matmul_bwd_kernel.cuh
 *
 * Grid:  (batch, d_model)
 * Block: kNThreads threads, each owns kNItems contiguous positions
 * Inner: serial loop over d_state
 * Chunks: processed in REVERSE order (c = n_c-1 down to 0)
 *
 * 4 scans per (chunk, state_idx):
 *   Scan #1: Möbius forward recompute   (3-phase sequential, det-norm matmul)
 *   Scan #2: Linear forward recompute   (CUB InclusiveScan)
 *   Scan #3: η reverse scan             (BlockReverseScan, float2)
 *   Scan #4: Möbius backward reverse     (3-phase sequential JVP, corrected Phase 2)
 *
 * Correctness-critical details:
 *   - __syncthreads() between every pair of CUB BlockLoads sharing smem
 *   - Phase 2 of Scan #4 uses per-thread 4×4 Jacobian products (not simple addition)
 *   - Phase 1+3 merged: re-run per-thread JVP scan with suffix as initial carry
 *   - Cross-chunk d̄λ^(S3) carry via persistent smem buffer
 *
 * NO [B,L,M,S] tensor is ever materialized in HBM.
 ******************************************************************************/
#pragma once

#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_reduce.cuh>
#include <ATen/cuda/Atomic.cuh>

#include "kla_matmul_scan.h"
#include "kla_matmul_scan_common.h"
#include "kla_matmul_scan_ops.cuh"
#include "reverse_scan.cuh"
#include "static_switch.h"

// =============================================================================
// Kernel traits
// =============================================================================
template<int kNThreads_, int kNItems_, bool kIsEvenLen_, typename input_t_>
struct KLA_Matmul_bwd_kernel_traits {
    static_assert(kNItems_ % 4 == 0);
    using input_t = input_t_;
    static constexpr int kNThreads  = kNThreads_;
    static constexpr int kNItems    = kNItems_;
    static constexpr int kNBytes    = sizeof(input_t);
    static constexpr int kNElts     = kNBytes == 4 ? 4 : constexpr_min(8, kNItems);
    static constexpr int kNLoads    = kNItems / kNElts;
    static constexpr bool kIsEvenLen = kIsEvenLen_;
    static constexpr int kMinBlocks  = kNThreads == 128 ? 3 : 2;
    static constexpr int kChunkSize  = kNThreads * kNItems;
    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;

    // CUB types
    using BlockLoadT        = cub::BlockLoad<input_t, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockStoreT       = cub::BlockStore<input_t, kNThreads, kNItems, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using LinearBlockScanT  = cub::BlockScan<float2, kNThreads, cub::BLOCK_SCAN_WARP_SCANS>;
    using BlockReverseScanT = BlockReverseScan<float2, kNThreads>;
    using BlockReduceFloat2T = cub::BlockReduce<float2, kNThreads>;

    // SMEM layout
    static constexpr int kSmemIOSize = (int)custom_max({
        sizeof(typename BlockLoadT::TempStorage),
        sizeof(typename BlockStoreT::TempStorage),
        sizeof(typename BlockReduceFloat2T::TempStorage)  // aliased: reduce after scans done
    });
    static constexpr int kSmemScanSize = (int)custom_max({
        (size_t)(kNThreads * sizeof(float4)),
        sizeof(typename LinearBlockScanT::TempStorage)
    });
    static constexpr int kSmemRevScanSize = (int)custom_max({
        sizeof(typename BlockReverseScanT::TempStorage),
        (size_t)(kNThreads * sizeof(float4))   // Möbius bwd scratch (thread aggregates)
    });
    // Jacobian products: kNThreads × 16 floats (4×4 matrix per thread)
    static constexpr int kSmemJacobianSize = kNThreads * 16 * sizeof(float);
    static constexpr int kSmemSizeRaw = kSmemIOSize + kSmemScanSize + kSmemRevScanSize;
    static constexpr int kSmemSize = (kSmemSizeRaw + 15) & ~15;
};

// =============================================================================
// Helper: Apply JVP (Jacobian-vector product) for one backward step
//
// Given gradient carry gP (w.r.t. P_norm at position l+1),
// propagate backward through the trace-normalized compose at position l+1:
//   P_{l+1,norm} = (M_{l+1} * P_l) / trace
//
// Returns gradient w.r.t. P_l (the prefix).
// =============================================================================
__device__ __forceinline__ float4 apply_jvp(
    const float4 &gP,          // gradient w.r.t. P_{l+1,norm}
    const float4 &P_next_norm, // P_{l+1,norm} (trace-normalized accumulated matrix)
    const float4 &M_next,      // M_{l+1} (leaf matrix at next position)
    float trace_next            // trace of raw product at position l+1
) {
    // Step 1: gradient through trace-normalization
    float4 gP_raw = grad_through_trace_norm(gP, P_next_norm, trace_next);
    // Step 2: gradient through raw matmul → gradient w.r.t. P_l (prefix)
    return grad_matmul_prefix(gP_raw, M_next);
}

// =============================================================================
// Helper: Apply JVP and also accumulate the 4×4 Jacobian matrix product
//
// J_thread = J_step^T * J_thread (right-multiply since we scan right-to-left)
//
// The Jacobian J_step^T maps gP_{l+1,norm} → gP_l, i.e. it's the composition
// of grad_through_trace_norm and grad_matmul_prefix as a linear map on R^4.
// =============================================================================
__device__ __forceinline__ float4 apply_jvp_with_jacobian(
    const float4 &gP,
    const float4 &P_next_norm,
    const float4 &M_next,
    float trace_next,
    float (&J)[4][4]  // accumulated Jacobian product, updated in-place
) {
    // Compute the JVP on the gradient vector
    float4 gP_raw = grad_through_trace_norm(gP, P_next_norm, trace_next);
    float4 gP_prev = grad_matmul_prefix(gP_raw, M_next);

    // Also apply the same linear map to each column of J (= J_step^T * J)
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        float4 col = make_float4(J[0][j], J[1][j], J[2][j], J[3][j]);
        float4 col_raw = grad_through_trace_norm(col, P_next_norm, trace_next);
        float4 col_out = grad_matmul_prefix(col_raw, M_next);
        J[0][j] = col_out.x;
        J[1][j] = col_out.y;
        J[2][j] = col_out.z;
        J[3][j] = col_out.w;
    }

    return gP_prev;
}

// =============================================================================
// Helper: 4×4 matrix-vector product  result = J * v
// =============================================================================
__device__ __forceinline__ float4 mat4_vec4(
    const float (&J)[4][4], const float4 &v
) {
    return make_float4(
        J[0][0]*v.x + J[0][1]*v.y + J[0][2]*v.z + J[0][3]*v.w,
        J[1][0]*v.x + J[1][1]*v.y + J[1][2]*v.z + J[1][3]*v.w,
        J[2][0]*v.x + J[2][1]*v.y + J[2][2]*v.z + J[2][3]*v.w,
        J[3][0]*v.x + J[3][1]*v.y + J[3][2]*v.z + J[3][3]*v.w
    );
}

// =============================================================================
// Backward kernel
// =============================================================================
template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads, Ktraits::kMinBlocks)
void kla_matmul_bwd_kernel(KLAMatmulParamsBwd params) {

    constexpr int  kNThreads  = Ktraits::kNThreads;
    constexpr int  kNItems    = Ktraits::kNItems;
    constexpr int  kChunkSize = Ktraits::kChunkSize;
    constexpr bool kIsEvenLen = Ktraits::kIsEvenLen;
    using input_t = typename Ktraits::input_t;

    // =====================================================================
    // Shared memory
    // =====================================================================
    extern __shared__ char smem_[];

    // Zone 1: CUB load/store / BlockReduce (time-multiplexed)
    auto &smem_load  = reinterpret_cast<typename Ktraits::BlockLoadT::TempStorage &>(smem_);
    auto &smem_store = reinterpret_cast<typename Ktraits::BlockStoreT::TempStorage &>(smem_);
    auto &smem_reduce_f2 = reinterpret_cast<typename Ktraits::BlockReduceFloat2T::TempStorage &>(smem_);

    // Zone 2: Möbius fwd scratch / CUB linear scan
    float4 *smem_mob_scratch = reinterpret_cast<float4 *>(smem_ + Ktraits::kSmemIOSize);
    auto &smem_lin_scan = *reinterpret_cast<typename Ktraits::LinearBlockScanT::TempStorage *>(
        smem_ + Ktraits::kSmemIOSize);

    // Zone 3: η reverse scan / Möbius bwd scratch
    auto &smem_rev_scan = *reinterpret_cast<typename Ktraits::BlockReverseScanT::TempStorage *>(
        smem_ + Ktraits::kSmemIOSize + Ktraits::kSmemScanSize);
    float4 *smem_bwd_scratch = reinterpret_cast<float4 *>(
        smem_ + Ktraits::kSmemIOSize + Ktraits::kSmemScanSize);

    // Persistent cross-chunk state (after zones)
    float4 *smem_mob_prefix    = reinterpret_cast<float4 *>(smem_ + Ktraits::kSmemSize);
    float2 *smem_lin_prefix    = reinterpret_cast<float2 *>(smem_mob_prefix + MAX_DSTATE);
    float2 *smem_rev_postfix   = reinterpret_cast<float2 *>(smem_lin_prefix + MAX_DSTATE);
    float  *smem_alpha_shift   = reinterpret_cast<float *>(smem_rev_postfix + MAX_DSTATE);
    float  *smem_lambda_shift  = reinterpret_cast<float *>(smem_alpha_shift + kNThreads + MAX_DSTATE);
    float4 *smem_mob_bwd_postfix = reinterpret_cast<float4 *>(smem_lambda_shift + kNThreads);
    float  *smem_dlam_S3_carry = reinterpret_cast<float *>(smem_mob_bwd_postfix + MAX_DSTATE);
    // Jacobian product storage: kNThreads × 16 floats
    float  *smem_J_products    = reinterpret_cast<float *>(smem_dlam_S3_carry + MAX_DSTATE);
    // Persistent da/dq accumulators (Mamba pattern: BlockReduce per state_idx, accumulate across chunks)
    float  *smem_da            = reinterpret_cast<float *>(smem_J_products + kNThreads * 16);
    float  *smem_dq            = reinterpret_cast<float *>(smem_da + MAX_DSTATE);
    // f4 exchange aliases Zone 2 (not in use during exchange)
    float4 *smem_f4_exchange   = smem_mob_scratch;

    // =====================================================================
    // Grid indices
    // =====================================================================
    const int batch_id = blockIdx.x;
    const int m_id     = blockIdx.y;

    // =====================================================================
    // Load static parameters
    // =====================================================================
    float a_regs[MAX_DSTATE], q_regs[MAX_DSTATE], inv_a2_regs[MAX_DSTATE];
    for (int s = 0; s < params.d_state; ++s) {
        float a_val = reinterpret_cast<float *>(params.a_ptr)[m_id * params.a_d_stride + s];
        float q_val = reinterpret_cast<float *>(params.q_ptr)[m_id * params.q_d_stride + s];
        a_regs[s]     = a_val;
        q_regs[s]     = q_val;
        inv_a2_regs[s] = 1.0f / (a_val * a_val);
    }

    // Zero cross-chunk carries and persistent da/dq accumulators
    if (threadIdx.x == 0) {
        for (int s = 0; s < MAX_DSTATE; ++s) {
            smem_dlam_S3_carry[s] = 0.0f;
            smem_da[s] = 0.0f;
            smem_dq[s] = 0.0f;
        }
    }
    __syncthreads();

    // =====================================================================
    // Main loop: chunks in REVERSE order
    // =====================================================================
    for (int chunk = params.n_chunks - 1; chunk >= 0; --chunk) {
        const int seqlen_remaining = params.seqlen - chunk * kChunkSize;

        // =================================================================
        // Load model-axis data: dy, dyvar, mu_sig, sig_inv [B, M, L]
        // =================================================================
        input_t dy_raw[kNItems], dyvar_raw[kNItems];
        input_t mu_sig_raw[kNItems], sig_inv_raw[kNItems];

        __syncthreads();
        load_input<Ktraits>(reinterpret_cast<input_t *>(params.dy_ptr)
            + batch_id * params.dy_batch_stride + m_id * params.dy_d_stride
            + chunk * kChunkSize,
            dy_raw, smem_load, seqlen_remaining);

        __syncthreads();
        load_input<Ktraits>(reinterpret_cast<input_t *>(params.dyvar_ptr)
            + batch_id * params.dyvar_batch_stride + m_id * params.dyvar_d_stride
            + chunk * kChunkSize,
            dyvar_raw, smem_load, seqlen_remaining);

        __syncthreads();
        load_input<Ktraits>(reinterpret_cast<input_t *>(params.mu_sigma_inv_ptr)
            + batch_id * params.mu_sigma_inv_batch_stride + m_id * params.mu_sigma_inv_d_stride
            + chunk * kChunkSize,
            mu_sig_raw, smem_load, seqlen_remaining);

        __syncthreads();
        load_input<Ktraits>(reinterpret_cast<input_t *>(params.sigma_inv_ptr)
            + batch_id * params.sigma_inv_batch_stride + m_id * params.sigma_inv_d_stride
            + chunk * kChunkSize,
            sig_inv_raw, smem_load, seqlen_remaining);

        float dy_f[kNItems], dyvar_f[kNItems], mu_sig_f[kNItems], sig_inv_f[kNItems];
        #pragma unroll
        for (int i = 0; i < kNItems; ++i) {
            dy_f[i]      = float(dy_raw[i]);
            dyvar_f[i]   = float(dyvar_raw[i]);
            mu_sig_f[i]  = float(mu_sig_raw[i]);
            sig_inv_f[i] = float(sig_inv_raw[i]);
        }

        // Model-axis gradient accumulators (sum over state_idx)
        float dmu_sig_accum[kNItems]  = {0};
        float dsig_inv_accum[kNItems] = {0};

        // =================================================================
        // Inner loop: serial over d_state
        // =================================================================
        __syncthreads();
        for (int state_idx = 0; state_idx < params.d_state; ++state_idx) {

            const float a_s    = a_regs[state_idx];
            const float q_s    = q_regs[state_idx];
            const float inv_a2 = inv_a2_regs[state_idx];
            const float a2     = a_s * a_s;

            // =============================================================
            // Load state-axis: h [B,S,L] and w [B,S,L]
            // =============================================================
            input_t h_raw[kNItems], w_raw[kNItems];

            __syncthreads();
            load_input<Ktraits>(reinterpret_cast<input_t *>(params.h_ptr)
                + batch_id * params.h_batch_stride + state_idx * params.h_dstate_stride
                + chunk * kChunkSize,
                h_raw, smem_load, seqlen_remaining);

            __syncthreads();  // sync between h and w loads
            load_input<Ktraits>(reinterpret_cast<input_t *>(params.w_ptr)
                + batch_id * params.w_batch_stride + state_idx * params.w_dstate_stride
                + chunk * kChunkSize,
                w_raw, smem_load, seqlen_remaining);

            float h_f[kNItems], w_f[kNItems];
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                h_f[i] = float(h_raw[i]);
                w_f[i] = float(w_raw[i]);
            }

            // =============================================================
            // RECOMPUTE Stage 1: φ, r (SHARED FUNCTION — bit-exact)
            // =============================================================
            float phi[kNItems], r[kNItems], raw_phi[kNItems];
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                compute_phi_r(h_f[i], sig_inv_f[i], mu_sig_f[i],
                              phi[i], r[i], raw_phi[i]);
            }

            // =============================================================
            // RECOMPUTE Scan #1: Det-norm matmul forward (3-phase sequential)
            // — IDENTICAL code path to forward kernel —
            // =============================================================
            float4 leaf[kNItems];   // save leaf matrices for backward
            float4 mob_data[kNItems];
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                leaf[i] = build_leaf_matrix(phi[i], q_s, inv_a2);
                mob_data[i] = leaf[i];
                if constexpr (!kIsEvenLen) {
                    if (threadIdx.x * kNItems + i >= seqlen_remaining)
                        mob_data[i] = matmul_identity();
                }
            }

            // Cross-chunk carry (load from boundary for chunk > 0)
            float4 mob_carry;
            if (chunk > 0) {
                float4 *mob_bnd = reinterpret_cast<float4 *>(params.mob_boundary_ptr);
                int bi = ((batch_id * params.n_chunks + (chunk - 1)) * params.d_model + m_id)
                         * params.d_state + state_idx;
                mob_carry = mob_bnd[bi];
            } else {
                mob_carry = matmul_identity();
            }
            const float prev_chunk_lambda = extract_lambda_lin(mob_carry);

            // 3-phase scan (identical to forward)
            {
                TraceNormMatMulOp op;
                float4 thread_agg = matmul_identity();
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    thread_agg = op(thread_agg, mob_data[i]);
                    mob_data[i] = thread_agg;
                }
                __syncthreads();
                smem_mob_scratch[threadIdx.x] = thread_agg;
                __syncthreads();
                if (threadIdx.x == 0) {
                    float4 carry = mob_carry;
                    for (int t = 0; t < kNThreads; ++t) {
                        float4 old_agg = smem_mob_scratch[t];
                        smem_mob_scratch[t] = carry;
                        carry = op(carry, old_agg);
                    }
                }
                __syncthreads();
                float4 my_prefix = smem_mob_scratch[threadIdx.x];
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    mob_data[i] = op(my_prefix, mob_data[i]);
                }
                __syncthreads();
            }

            // Extract λ (trace_vals computed after prev_P exchange below)
            float lambda_vals[kNItems], trace_vals[kNItems];
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                lambda_vals[i] = extract_lambda_lin(mob_data[i]);
            }

            // Compute prev_P[i] = P_{t-1} via smem exchange
            float4 prev_P[kNItems];
            smem_f4_exchange[threadIdx.x] = mob_data[kNItems - 1];
            __syncthreads();
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                if (i > 0) {
                    prev_P[i] = mob_data[i - 1];
                } else if (threadIdx.x > 0) {
                    prev_P[0] = smem_f4_exchange[threadIdx.x - 1];
                } else if (chunk > 0) {
                    prev_P[0] = mob_carry;  // boundary from previous chunk
                } else {
                    prev_P[0] = matmul_identity();
                }
            }
            __syncthreads();

            // Compute trace_vals[i] = trace(leaf[i] * prev_P[i])
            // = A_raw + D_raw of the raw product M_i * P_{i-1}
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                float A_raw = leaf[i].x * prev_P[i].x + leaf[i].y * prev_P[i].z;
                float D_raw = leaf[i].z * prev_P[i].y + leaf[i].w * prev_P[i].w;
                trace_vals[i] = A_raw + D_raw;
            }

            // =============================================================
            // RECOMPUTE Scan #2: Linear forward (CUB)
            // =============================================================
            float prev_lambda[kNItems];
            smem_lambda_shift[threadIdx.x] = lambda_vals[kNItems - 1];
            __syncthreads();
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                if (i > 0) prev_lambda[i] = lambda_vals[i - 1];
                else if (threadIdx.x > 0) prev_lambda[0] = smem_lambda_shift[threadIdx.x - 1];
                else prev_lambda[0] = prev_chunk_lambda;
            }
            __syncthreads();

            float alpha_lin_vals[kNItems], denom_vals[kNItems];
            float2 lin_data[kNItems];
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                lin_data[i] = build_linear_input(a_s, a2, q_s, prev_lambda[i], r[i], denom_vals[i]);
                alpha_lin_vals[i] = lin_data[i].x;
                if constexpr (!kIsEvenLen) {
                    if (threadIdx.x * kNItems + i >= seqlen_remaining)
                        lin_data[i] = linear_identity();
                }
            }

            float2 lin_prefix;
            if (chunk > 0 && threadIdx.x % 32 == 0) {
                float *lb = reinterpret_cast<float *>(params.lin_boundary_ptr);
                int bi = ((batch_id * params.n_chunks + (chunk - 1)) * params.d_model + m_id)
                         * params.d_state + state_idx;
                lin_prefix = make_float2(1.0f, lb[bi]);
            } else {
                lin_prefix = linear_identity();
            }
            LinearPrefixCallbackOp lin_prefix_op(lin_prefix);

            __syncthreads();
            typename Ktraits::LinearBlockScanT(smem_lin_scan).InclusiveScan(
                lin_data, lin_data, LinearScanOp(), lin_prefix_op);

            float eta_vals[kNItems], m_vals[kNItems];
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                eta_vals[i] = lin_data[i].y;
                m_vals[i]   = eta_vals[i] / lambda_vals[i];
            }

            // =============================================================
            // Stages 5+4: Direct gradients
            // =============================================================
            float deta_direct[kNItems], dlam_S4[kNItems], dlam_S5[kNItems];
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                float dm         = dy_f[i] * w_f[i];
                deta_direct[i]   = dm / lambda_vals[i];
                dlam_S4[i]       = -dm * m_vals[i] / lambda_vals[i];
                dlam_S5[i]       = -dyvar_f[i] * w_f[i] * w_f[i]
                                   / (lambda_vals[i] * lambda_vals[i]);
            }

            // =============================================================
            // Scan #3: η reverse scan (BlockReverseScan, float2)
            // =============================================================
            smem_alpha_shift[threadIdx.x == 0 ? state_idx : threadIdx.x + MAX_DSTATE]
                = alpha_lin_vals[0];
            __syncthreads();

            float2 rev_data[kNItems];
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                float alpha_next;
                if (i < kNItems - 1)
                    alpha_next = alpha_lin_vals[i + 1];
                else
                    alpha_next = (threadIdx.x < kNThreads - 1)
                        ? smem_alpha_shift[threadIdx.x + 1 + MAX_DSTATE] : 1.0f;
                rev_data[i] = make_float2(alpha_next, deta_direct[i]);
                if constexpr (!kIsEvenLen) {
                    if (threadIdx.x * kNItems + i >= seqlen_remaining)
                        rev_data[i] = linear_identity();
                }
            }
            __syncthreads();

            float2 rev_postfix;
            if (chunk < params.n_chunks - 1 && threadIdx.x % 32 == 0)
                rev_postfix = smem_rev_postfix[state_idx];
            else
                rev_postfix = linear_identity();
            LinearPrefixCallbackOp rev_prefix_op(rev_postfix);

            typename Ktraits::BlockReverseScanT(smem_rev_scan).InclusiveReverseScan(
                rev_data, rev_data, LinearScanOp(), rev_prefix_op);
            if (threadIdx.x == 0)
                smem_rev_postfix[state_idx] = rev_prefix_op.running_prefix;

            // Extract d̄α^lin and d̄λ^(S3)
            float dalpha_lin_vals[kNItems], dlam_S3[kNItems];
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                float deta_total = rev_data[i].y;
                float eta_prev = (alpha_lin_vals[i] > 1e-10f)
                    ? (eta_vals[i] - r[i]) / alpha_lin_vals[i] : 0.0f;
                dalpha_lin_vals[i] = deta_total * eta_prev;
                dlam_S3[i] = dalpha_lin_vals[i]
                           * (-a_s * q_s / (denom_vals[i] * denom_vals[i]));
            }

            // d̄λ^(S3) shift right by one, with cross-chunk carry
            float dlam_S3_shifted[kNItems];
            float dlam_S3_from_next_chunk;
            if (threadIdx.x == kNThreads - 1)
                dlam_S3_from_next_chunk = smem_dlam_S3_carry[state_idx];
            __syncthreads();
            if (threadIdx.x == 0)
                smem_dlam_S3_carry[state_idx] = dlam_S3[0];  // save for prev chunk
            smem_lambda_shift[threadIdx.x] = dlam_S3[kNItems - 1];
            __syncthreads();
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                if (i < kNItems - 1)
                    dlam_S3_shifted[i] = dlam_S3[i + 1];
                else if (threadIdx.x < kNThreads - 1)
                    dlam_S3_shifted[i] = smem_lambda_shift[threadIdx.x + 1];
                else
                    dlam_S3_shifted[i] = dlam_S3_from_next_chunk;
            }
            __syncthreads();

            // =============================================================
            // Combine d̄λ and compute d̄P^direct
            // =============================================================
            float dlam[kNItems];
            float4 gP_direct[kNItems];
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                dlam[i] = dlam_S4[i] + dlam_S5[i] + dlam_S3_shifted[i];
                gP_direct[i] = grad_lambda_extraction(dlam[i], mob_data[i]);
            }

            // =============================================================
            // Prepare next-position data for Scan #4 JVP
            // next_leaf[i]  = leaf at position i+1
            // next_P[i]     = P_norm at position i+1
            // next_trace[i] = trace of raw product at position i+1
            // =============================================================
            float4 next_leaf[kNItems], next_P_arr[kNItems];
            float  next_trace[kNItems];

            smem_f4_exchange[threadIdx.x] = leaf[0];
            __syncthreads();
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                if (i < kNItems - 1) {
                    next_leaf[i] = leaf[i + 1];
                } else if (threadIdx.x < kNThreads - 1) {
                    next_leaf[i] = smem_f4_exchange[threadIdx.x + 1];
                } else {
                    next_leaf[i] = matmul_identity();
                }
            }
            __syncthreads();

            smem_f4_exchange[threadIdx.x] = mob_data[0];
            __syncthreads();
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                if (i < kNItems - 1) {
                    next_P_arr[i] = mob_data[i + 1];
                    next_trace[i] = trace_vals[i + 1];
                } else if (threadIdx.x < kNThreads - 1) {
                    next_P_arr[i] = smem_f4_exchange[threadIdx.x + 1];
                    // trace of raw product at next pos = trace(next_leaf[i] * mob_data[i])
                    // mob_data[i] is P_i (current pos), next_leaf[i] is M_{i+1}
                    float A_raw = next_leaf[i].x * mob_data[i].x + next_leaf[i].y * mob_data[i].z;
                    float D_raw = next_leaf[i].z * mob_data[i].y + next_leaf[i].w * mob_data[i].w;
                    next_trace[i] = A_raw + D_raw;
                } else {
                    next_P_arr[i] = matmul_identity();
                    next_trace[i] = 2.0f;  // trace(I₂) = 2
                }
            }
            __syncthreads();

            // =============================================================
            // Scan #4: Möbius backward reverse scan
            //   Corrected 3-phase: pre-scan + Phase 2 + Phase 1+3 combined
            //
            // Pre-scan: compute per-thread aggregate b_t and 4×4 Jacobian J_t
            // Phase 2:  thread 0 does affine compose: running = J_t * running + b_t
            // Phase 1+3: re-run per-thread JVP with suffix as initial carry
            // =============================================================
            float4 gP[kNItems];
            {
                // --- Pre-scan pass: all threads compute aggregates ---
                float4 b_t = make_float4(0, 0, 0, 0);
                float J_t[4][4] = {
                    {1,0,0,0}, {0,1,0,0}, {0,0,1,0}, {0,0,0,1}
                };

                #pragma unroll
                for (int i = kNItems - 1; i >= 0; --i) {
                    // Apply JVP: propagate b_t through position i+1
                    bool has_next = (i < kNItems - 1)
                                 || (threadIdx.x < kNThreads - 1);
                    if (has_next) {
                        b_t = apply_jvp_with_jacobian(
                            b_t, next_P_arr[i], next_leaf[i], next_trace[i], J_t);
                    }
                    // Add direct gradient at position i
                    b_t.x += gP_direct[i].x;
                    b_t.y += gP_direct[i].y;
                    b_t.z += gP_direct[i].z;
                    b_t.w += gP_direct[i].w;
                }

                // Write J_t (16 floats) and b_t (4 floats) to smem
                #pragma unroll
                for (int r = 0; r < 4; ++r)
                    #pragma unroll
                    for (int c = 0; c < 4; ++c)
                        smem_J_products[threadIdx.x * 16 + r * 4 + c] = J_t[r][c];

                smem_bwd_scratch[threadIdx.x] = b_t;
                __syncthreads();

                // --- Phase 2: thread 0, right-to-left affine compose ---
                if (threadIdx.x == 0) {
                    float4 running;
                    if (chunk < params.n_chunks - 1)
                        running = smem_mob_bwd_postfix[state_idx];
                    else
                        running = make_float4(0, 0, 0, 0);

                    for (int t = kNThreads - 1; t >= 0; --t) {
                        // Exclusive suffix for thread t
                        float4 old_b = smem_bwd_scratch[t];
                        smem_bwd_scratch[t] = running;  // store suffix

                        // Load J_t
                        float Jt[4][4];
                        #pragma unroll
                        for (int r = 0; r < 4; ++r)
                            #pragma unroll
                            for (int c = 0; c < 4; ++c)
                                Jt[r][c] = smem_J_products[t * 16 + r * 4 + c];

                        // Affine compose: running = J_t * running + b_t
                        float4 Jv = mat4_vec4(Jt, running);
                        running.x = Jv.x + old_b.x;
                        running.y = Jv.y + old_b.y;
                        running.z = Jv.z + old_b.z;
                        running.w = Jv.w + old_b.w;
                    }
                    smem_mob_bwd_postfix[state_idx] = running;
                }
                __syncthreads();

                // --- Phase 1+3 combined: re-scan with suffix ---
                float4 carry = smem_bwd_scratch[threadIdx.x];  // exclusive suffix

                #pragma unroll
                for (int i = kNItems - 1; i >= 0; --i) {
                    bool has_next = (i < kNItems - 1)
                                 || (threadIdx.x < kNThreads - 1);
                    if (has_next) {
                        carry = apply_jvp(
                            carry, next_P_arr[i], next_leaf[i], next_trace[i]);
                    }
                    carry.x += gP_direct[i].x;
                    carry.y += gP_direct[i].y;
                    carry.z += gP_direct[i].z;
                    carry.w += gP_direct[i].w;
                    gP[i] = carry;
                }
                __syncthreads();
            }

            // =============================================================
            // Extract parameter gradients from gP[i] and prev_P[i]
            //
            // gP[i] is the total gradient w.r.t. P_t (trace-normalized).
            // Chain through trace-norm → raw matmul → leaf matrix → (φ, a, q)
            // =============================================================
            float da_thread = 0.0f, dq_thread = 0.0f;  // per-thread accumulators
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                float4 gP_raw = grad_through_trace_norm(gP[i], mob_data[i], trace_vals[i]);
                float4 gM     = grad_matmul_element(gP_raw, prev_P[i]);

                // gM = (gA_R, gB_R, gC_R, gD_R) for the leaf matrix
                // M = [[(1+qφ)/a², φ], [q/a², 1]]
                // d̄φ from Möbius: ∂A/∂φ = q/a², ∂B/∂φ = 1
                float dphi_mob = gM.x * q_s * inv_a2 + gM.y;

                // d̄a from Möbius: ∂A/∂a = -2(1+qφ)/a³, ∂C/∂a = -2q/a³
                float da_mob = gM.x * (-2.0f * (1.0f + q_s * phi[i]) / (a_s * a2))
                             + gM.z * (-2.0f * q_s / (a_s * a2));

                // d̄q from Möbius: ∂A/∂q = φ/a², ∂C/∂q = 1/a²
                float dq_mob = gM.x * phi[i] * inv_a2
                             + gM.z * inv_a2;

                // d̄a, d̄q from linear path
                float D2 = denom_vals[i] * denom_vals[i];
                float da_lin = dalpha_lin_vals[i]
                             * (q_s * prev_lambda[i] - a2) / D2;
                float dq_lin = dalpha_lin_vals[i]
                             * (-a_s * prev_lambda[i]) / D2;

                // Accumulate static param grads (per-thread register, reduced below)
                da_thread += da_mob + da_lin;
                dq_thread += dq_mob + dq_lin;

                // Stage 0 backward: chain rule to raw inputs
                float dr_i     = rev_data[i].y;
                float phi_mask = (raw_phi[i] > KLA_EPS && raw_phi[i] < 1000.0f) ? 1.0f : 0.0f;

                dmu_sig_accum[i]  += dr_i * h_f[i];
                dsig_inv_accum[i] += dphi_mob * h_f[i] * h_f[i] * phi_mask;

                // d̄h, d̄w via atomicAdd (cross-m accumulation)
                int lg = chunk * kChunkSize + threadIdx.x * kNItems + i;
                if (lg < params.seqlen) {
                    float dh_val = 2.0f * dphi_mob * h_f[i] * sig_inv_f[i] * phi_mask
                                 + dr_i * mu_sig_f[i];
                    gpuAtomicAdd(reinterpret_cast<float *>(params.dh_ptr)
                        + batch_id * params.dh_batch_stride
                        + state_idx * params.dh_dstate_stride + lg,
                        dh_val);

                    float dw_val = dy_f[i] * m_vals[i]
                                 + dyvar_f[i] * 2.0f * w_f[i] / lambda_vals[i];
                    gpuAtomicAdd(reinterpret_cast<float *>(params.dw_ptr)
                        + batch_id * params.dw_batch_stride
                        + state_idx * params.dw_dstate_stride + lg,
                        dw_val);
                }
            }

            // BlockReduce da/dq across all threads (Mamba pattern)
            // smem_reduce_f2 aliases Zone 1, which is free here
            __syncthreads();
            float2 da_dq_thread = make_float2(da_thread, dq_thread);
            float2 da_dq_sum = typename Ktraits::BlockReduceFloat2T(smem_reduce_f2).Sum(da_dq_thread);
            if (threadIdx.x == 0) {
                smem_da[state_idx] += da_dq_sum.x;
                smem_dq[state_idx] += da_dq_sum.y;
            }
        }  // end state_idx

        // Store model-axis gradients
        input_t dmu_out[kNItems], dsig_out[kNItems];
        #pragma unroll
        for (int i = 0; i < kNItems; ++i) {
            dmu_out[i]  = input_t(dmu_sig_accum[i]);
            dsig_out[i] = input_t(dsig_inv_accum[i]);
        }
        __syncthreads();
        store_output<Ktraits>(reinterpret_cast<input_t *>(params.dmu_sigma_inv_ptr)
            + batch_id * params.dmu_batch_stride + m_id * params.dmu_d_stride
            + chunk * kChunkSize,
            dmu_out, smem_store, seqlen_remaining);
        __syncthreads();
        store_output<Ktraits>(reinterpret_cast<input_t *>(params.dsigma_inv_ptr)
            + batch_id * params.dsig_batch_stride + m_id * params.dsig_d_stride
            + chunk * kChunkSize,
            dsig_out, smem_store, seqlen_remaining);

    }  // end for chunk

    // Store static parameter gradients via atomicAdd (from persistent smem)
    __syncthreads();  // ensure all BlockReduce writes to smem_da/smem_dq are visible
    if (threadIdx.x == 0) {
        for (int s = 0; s < params.d_state; ++s) {
            gpuAtomicAdd(reinterpret_cast<float *>(params.da_ptr)
                + m_id * params.da_d_stride + s, smem_da[s]);
            gpuAtomicAdd(reinterpret_cast<float *>(params.dq_ptr)
                + m_id * params.dq_d_stride + s, smem_dq[s]);
        }
    }
}

// =============================================================================
// Launch dispatch
// =============================================================================
template<int kNThreads, int kNItems, typename input_t>
void kla_matmul_bwd_launch(KLAMatmulParamsBwd &params, cudaStream_t stream) {
    BOOL_SWITCH(params.seqlen % (kNThreads * kNItems) == 0, kIsEvenLen, [&] {
        using Ktraits = KLA_Matmul_bwd_kernel_traits<kNThreads, kNItems, kIsEvenLen, input_t>;

        constexpr int kSmemSize = Ktraits::kSmemSize
            + MAX_DSTATE * sizeof(float4)                        // smem_mob_prefix
            + MAX_DSTATE * sizeof(float2)                        // smem_lin_prefix
            + MAX_DSTATE * sizeof(float2)                        // smem_rev_postfix
            + (kNThreads + MAX_DSTATE) * sizeof(float)           // smem_alpha_shift
            + kNThreads * sizeof(float)                          // smem_lambda_shift
            + MAX_DSTATE * sizeof(float4)                        // smem_mob_bwd_postfix
            + MAX_DSTATE * sizeof(float)                         // smem_dlam_S3_carry
            + kNThreads * 16 * sizeof(float)                     // smem_J_products
            + MAX_DSTATE * sizeof(float)                         // smem_da
            + MAX_DSTATE * sizeof(float);                        // smem_dq

        dim3 grid(params.batch, params.d_model);
        auto kernel = &kla_matmul_bwd_kernel<Ktraits>;

        if (kSmemSize >= 48 * 1024) {
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
        }
        kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

template<typename input_t>
void kla_matmul_bwd_cuda(KLAMatmulParamsBwd &params, cudaStream_t stream) {
    if (params.seqlen <= 128) {
        kla_matmul_bwd_launch<32, 4, input_t>(params, stream);
    } else {
        kla_matmul_bwd_launch<64, 8, input_t>(params, stream);
    }
}
