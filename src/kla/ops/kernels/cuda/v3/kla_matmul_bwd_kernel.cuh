/******************************************************************************
 * KLA v3 backward: chunk-recomputed forward + two scalar reverse affine scans.
 * Grid (batch, channel); threads own contiguous timesteps; serial state loop.
 * Saves only forward chunk checkpoints in HBM, never a [B,L,M,S] trajectory.
 * Boundary carries target chunk INPUT states, including their first-step VJP.
 * Padding has identity scan maps and contributes no parameter gradients.
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
    static constexpr int kSmemRevScanSize = sizeof(typename BlockReverseScanT::TempStorage);
    static constexpr int kSmemSizeRaw = kSmemIOSize + kSmemScanSize + kSmemRevScanSize;
    static constexpr int kSmemSize = (kSmemSizeRaw + 15) & ~15;
};

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

    // Reverse scans reuse this scratch, with barriers between uses.
    auto &smem_rev_scan = *reinterpret_cast<typename Ktraits::BlockReverseScanT::TempStorage *>(
        smem_ + Ktraits::kSmemIOSize + Ktraits::kSmemScanSize);
    // Carries are gradients w.r.t. the NEXT chunk's input state, already
    // transformed through its first timestep. No boundary Jacobian is omitted.
    float *smem_eta_carry = reinterpret_cast<float *>(smem_ + Ktraits::kSmemSize);
    float *smem_lam_carry = smem_eta_carry + MAX_DSTATE;
    float *smem_da = smem_lam_carry + MAX_DSTATE;
    float *smem_dq = smem_da + MAX_DSTATE;
    float *smem_lambda_shift = smem_dq + MAX_DSTATE;

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
        // Floored to stay bit-identical to the forward (see the note there).
        inv_a2_regs[s] = 1.0f / fmaxf(a_val * a_val, KLA_EPS);
    }

    // Zero cross-chunk carries and persistent da/dq accumulators
    if (threadIdx.x == 0) {
        for (int s = 0; s < MAX_DSTATE; ++s) {
            smem_eta_carry[s] = 0.0f;
            smem_lam_carry[s] = 0.0f;
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
            const float a2     = fmaxf(a_s * a_s, KLA_EPS);  // == 1/inv_a2

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
            // RECOMPUTE Scan #1: Trace-normalized forward (3-phase sequential)
            // — IDENTICAL code path to forward kernel —
            // =============================================================
            float4 mob_data[kNItems];
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                mob_data[i] = build_leaf_matrix(phi[i], q_s, inv_a2);
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

            // Extract precision from the recomputed forward prefixes.
            float lambda_vals[kNItems];
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                lambda_vals[i] = extract_lambda_lin(mob_data[i]);
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

            // Exchange eta_{t-1} directly: reconstructing it as
            // (eta_t-r_t)/alpha_t loses accuracy and fails for alpha_t=0.
            float eta_prev[kNItems];
            __syncthreads();
            smem_lambda_shift[threadIdx.x] = eta_vals[kNItems - 1];
            __syncthreads();
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                if (i > 0) eta_prev[i] = eta_vals[i - 1];
                else if (threadIdx.x > 0) eta_prev[i] = smem_lambda_shift[threadIdx.x - 1];
                else if (chunk > 0) {
                    int bi = ((batch_id * params.n_chunks + chunk - 1) * params.d_model + m_id)
                             * params.d_state + state_idx;
                    eta_prev[i] = reinterpret_cast<float *>(params.lin_boundary_ptr)[bi];
                } else eta_prev[i] = 0.0f;
            }
            __syncthreads();

            // Scan #3: bar_eta_t = direct_t + alpha_{t+1} bar_eta_{t+1}.
            // Publish FIRST item for a next-timestep exchange.
            smem_lambda_shift[threadIdx.x] = alpha_lin_vals[0];
            __syncthreads();
            float2 rev_data[kNItems];
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                int t = threadIdx.x * kNItems + i;
                float next = i + 1 < kNItems ? alpha_lin_vals[i + 1]
                    : (threadIdx.x + 1 < kNThreads ? smem_lambda_shift[threadIdx.x + 1] : 1.0f);
                // At the physical chunk end, carry already targets this state.
                // Padded positions are exact identity maps with zero source.
                if (t + 1 >= seqlen_remaining) next = 1.0f;
                rev_data[i] = t < seqlen_remaining
                    ? make_float2(next, deta_direct[i]) : linear_identity();
            }
            LinearPrefixCallbackOp eta_suffix(make_float2(1.0f, smem_eta_carry[state_idx]));
            __syncthreads();
            typename Ktraits::BlockReverseScanT(smem_rev_scan).InclusiveReverseScan(
                rev_data, rev_data, LinearScanOp(), eta_suffix);
            if (threadIdx.x == 0)
                smem_eta_carry[state_idx] = alpha_lin_vals[0] * rev_data[0].y;
            __syncthreads();

            float dalpha[kNItems], dlam_from_eta[kNItems], gain[kNItems];
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                bool valid = threadIdx.x * kNItems + i < seqlen_remaining;
                float den = denom_vals[i];
                float den_mask = a2 + q_s * prev_lambda[i] >= KLA_EPS ? 1.0f : 0.0f;
                dalpha[i] = valid ? rev_data[i].y * eta_prev[i] : 0.0f;
                dlam_from_eta[i] = -dalpha[i] * a_s * q_s / (den * den) * den_mask;
                // Precision recurrence uses a2+p*lambda_prev, which is >= EPS
                // for supported positive process noise and precision.
                gain[i] = a2 / (den * den);
            }
            // Exchange BOTH the next gain and next information-path source.
            __syncthreads();
            smem_lambda_shift[threadIdx.x] = dlam_from_eta[0];
            __syncthreads();
            float source[kNItems];
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                float next_src = i + 1 < kNItems ? dlam_from_eta[i + 1]
                    : (threadIdx.x + 1 < kNThreads ? smem_lambda_shift[threadIdx.x + 1] : 0.0f);
                source[i] = dlam_S4[i] + dlam_S5[i] + next_src;
            }
            __syncthreads();
            smem_lambda_shift[threadIdx.x] = gain[0];
            __syncthreads();
            float2 lam_rev[kNItems];
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                int t = threadIdx.x * kNItems + i;
                float next = i + 1 < kNItems ? gain[i + 1]
                    : (threadIdx.x + 1 < kNThreads ? smem_lambda_shift[threadIdx.x + 1] : 1.0f);
                if (t + 1 >= seqlen_remaining) next = 1.0f;
                lam_rev[i] = t < seqlen_remaining
                    ? make_float2(next, source[i]) : linear_identity();
            }
            LinearPrefixCallbackOp lam_suffix(make_float2(1.0f, smem_lam_carry[state_idx]));
            __syncthreads();
            typename Ktraits::BlockReverseScanT(smem_rev_scan).InclusiveReverseScan(
                lam_rev, lam_rev, LinearScanOp(), lam_suffix);
            if (threadIdx.x == 0)
                smem_lam_carry[state_idx] = gain[0] * lam_rev[0].y + dlam_from_eta[0];
            __syncthreads();

            // Local scalar chain rule. No matrix adjoints or 4x4 Jacobians.
            float da_thread = 0.0f, dq_thread = 0.0f;
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                if (threadIdx.x * kNItems + i >= seqlen_remaining) continue;
                float den = denom_vals[i];
                float D2 = den * den;
                float lp = prev_lambda[i];
                float dphi_mob = lam_rev[i].y;
                float da2 = a_s * a_s >= KLA_EPS ? 2.0f * a_s : 0.0f;
                float den_mask = a2 + q_s * lp >= KLA_EPS ? 1.0f : 0.0f;
                da_thread += -dphi_mob * lp * da2 / D2
                    + dalpha[i] * (1.0f / den - a_s * da2 / D2 * den_mask);
                dq_thread += -dphi_mob * lp * lp / D2
                    - dalpha[i] * a_s * lp / D2 * den_mask;

                // Stage 0 backward: chain rule to raw inputs
                float dr_i     = rev_data[i].y;
                dmu_sig_accum[i]  += dr_i * h_f[i];
                dsig_inv_accum[i] += dphi_mob * h_f[i] * h_f[i];

                // d̄h, d̄w via atomicAdd (cross-m accumulation)
                int lg = chunk * kChunkSize + threadIdx.x * kNItems + i;
                if (lg < params.seqlen) {
                    float dh_val = 2.0f * dphi_mob * h_f[i] * sig_inv_f[i]
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
            + (4 * MAX_DSTATE + kNThreads) * sizeof(float);

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
