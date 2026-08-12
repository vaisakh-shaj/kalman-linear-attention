/******************************************************************************
 * KLA Matmul Forward Kernel — Static a,q, Linear-Space Det-Norm
 *
 * Rewritten from scratch following Mamba-1's synchronization discipline.
 *
 * Grid:  (batch, d_model)
 * Block: kNThreads threads, each owns kNItems contiguous sequence positions
 * Inner: serial loop over d_state
 *
 * Per (chunk, state_idx):
 *   Stage 1: compute φ, r from inputs
 *   Stage 2: build leaf matrices → 3-phase sequential det-norm matmul scan → λ
 *   Stage 3: build linear scan input → CUB inclusive scan → η
 *   Stage 4: contraction y += (η/λ)·w,  yvar += (1/λ)·w²
 *
 * Memory model:
 *   HBM inputs:  μσ⁻², σ⁻² [B,M,L]  h, w [B,S,L]  a, q [M,S]
 *   HBM outputs: y, yvar [B,M,L]
 *   HBM saved:   mob_boundary [B,nc,M,S,4]  lin_boundary [B,nc,M,S]
 *                 lam_boundary [B,nc,M,S]
 *   SMEM zones:  Zone 1 (IO) — CUB load/store temp
 *                Zone 2 (Scan) — Möbius scratch / CUB linear scan temp
 *   SMEM persistent: mob_prefix [S], lin_prefix [S], lambda_shift [T]
 *   Registers:   all per-item arrays (φ, r, leaf, λ, η, accumulators)
 *
 * Sync discipline:
 *   Every CUB BlockLoad/BlockStore uses SMEM internally.  Two consecutive
 *   cooperative ops sharing the same SMEM region MUST have a __syncthreads()
 *   between them.  The 3-phase sequential scan has syncs between phases.
 *   λ-shift exchange has syncs before write and after read.
 *
 * NO [B,L,M,S] tensor is ever materialized in HBM.
 ******************************************************************************/
#pragma once

#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_scan.cuh>

#include "kla_matmul_scan.h"
#include "kla_matmul_scan_common.h"
#include "kla_matmul_scan_ops.cuh"
#include "static_switch.h"

// =============================================================================
// Kernel traits
// =============================================================================
template<int kNThreads_, int kNItems_, bool kIsEvenLen_, typename input_t_>
struct KLA_Matmul_fwd_kernel_traits {
    static_assert(kNItems_ % 4 == 0);
    using input_t = input_t_;
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kNItems   = kNItems_;
    static constexpr int kNBytes   = sizeof(input_t);
    static constexpr int kNElts    = kNBytes == 4 ? 4 : constexpr_min(8, kNItems);
    static constexpr int kNLoads   = kNItems / kNElts;
    static constexpr bool kIsEvenLen = kIsEvenLen_;
    static constexpr int kMinBlocks  = kNThreads < 128 ? 5 : 3;
    static constexpr int kChunkSize  = kNThreads * kNItems;

    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;

    // CUB types
    using BlockLoadT      = cub::BlockLoad<input_t, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockStoreT     = cub::BlockStore<input_t, kNThreads, kNItems, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using LinearBlockScanT = cub::BlockScan<float2, kNThreads, cub::BLOCK_SCAN_WARP_SCANS>;

    // SMEM layout:
    //   [0, kSmemIOSize)             — Zone 1: CUB load/store temp
    //   [kSmemIOSize, kSmemIOSize + kSmemScanSize) — Zone 2: Möbius scratch / CUB linear scan
    //   [kSmemSize, ...)             — Persistent: mob_prefix, lin_prefix, lambda_shift
    static constexpr int kSmemIOSize = (int)custom_max({
        sizeof(typename BlockLoadT::TempStorage),
        sizeof(typename BlockStoreT::TempStorage)
    });
    static constexpr int kSmemScanSize = (int)custom_max({
        (size_t)(kNThreads * sizeof(float4)),                 // Möbius 3-phase scratch
        sizeof(typename LinearBlockScanT::TempStorage)        // CUB linear scan
    });
    static constexpr int kSmemSizeRaw = kSmemIOSize + kSmemScanSize;
    static constexpr int kSmemSize = (kSmemSizeRaw + 15) & ~15;  // 16-byte aligned
};

// =============================================================================
// Forward kernel
// =============================================================================
template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads, Ktraits::kMinBlocks)
void kla_matmul_fwd_kernel(KLAMatmulParamsFwd params) {

    // =====================================================================
    // Constants
    // =====================================================================
    constexpr int  kNThreads  = Ktraits::kNThreads;
    constexpr int  kNItems    = Ktraits::kNItems;
    constexpr int  kChunkSize = Ktraits::kChunkSize;
    constexpr bool kIsEvenLen = Ktraits::kIsEvenLen;
    using input_t = typename Ktraits::input_t;

    // =====================================================================
    // Shared memory
    // =====================================================================
    extern __shared__ char smem_[];

    // Zone 1: CUB cooperative load/store (time-multiplexed)
    auto &smem_load  = reinterpret_cast<typename Ktraits::BlockLoadT::TempStorage &>(smem_);
    auto &smem_store = reinterpret_cast<typename Ktraits::BlockStoreT::TempStorage &>(smem_);

    // Zone 2: Möbius scratch (float4 per thread) / CUB linear scan (time-multiplexed)
    float4 *smem_mob_scratch = reinterpret_cast<float4 *>(smem_ + Ktraits::kSmemIOSize);
    auto &smem_lin_scan = *reinterpret_cast<typename Ktraits::LinearBlockScanT::TempStorage *>(
        smem_ + Ktraits::kSmemIOSize);

    // Persistent cross-chunk state (lives AFTER zones, never aliased)
    float4 *smem_mob_prefix   = reinterpret_cast<float4 *>(smem_ + Ktraits::kSmemSize);
    float2 *smem_lin_prefix   = reinterpret_cast<float2 *>(smem_mob_prefix + MAX_DSTATE);
    float  *smem_lambda_shift = reinterpret_cast<float *>(smem_lin_prefix + MAX_DSTATE);
    // Total persistent: MAX_DSTATE*16 + MAX_DSTATE*8 + kNThreads*4 bytes

    // =====================================================================
    // Grid indices
    // =====================================================================
    const int batch_id = blockIdx.x;
    const int m_id     = blockIdx.y;

    // =====================================================================
    // Pointers: model-axis [B, M, L]
    // =====================================================================
    input_t *mu_sig = reinterpret_cast<input_t *>(params.mu_sigma_inv_ptr)
        + batch_id * params.mu_sigma_inv_batch_stride
        + m_id     * params.mu_sigma_inv_d_stride;

    input_t *sig_inv = reinterpret_cast<input_t *>(params.sigma_inv_ptr)
        + batch_id * params.sigma_inv_batch_stride
        + m_id     * params.sigma_inv_d_stride;

    // Pointers: state-axis [B, S, L]
    input_t *h_base = reinterpret_cast<input_t *>(params.h_ptr)
        + batch_id * params.h_batch_stride;
    input_t *w_base = reinterpret_cast<input_t *>(params.w_ptr)
        + batch_id * params.w_batch_stride;

    // Pointers: output [B, M, L]
    input_t *y_out = reinterpret_cast<input_t *>(params.y_ptr)
        + batch_id * params.y_batch_stride
        + m_id     * params.y_d_stride;
    input_t *yvar_out = reinterpret_cast<input_t *>(params.yvar_ptr)
        + batch_id * params.yvar_batch_stride
        + m_id     * params.yvar_d_stride;

    // =====================================================================
    // Load static parameters into registers (once per block)
    //   a [M, S] — discrete transition decay
    //   q [M, S] — discrete process noise
    //   Precompute 1/a² for leaf matrix construction
    // =====================================================================
    float a_regs[MAX_DSTATE], q_regs[MAX_DSTATE], inv_a2_regs[MAX_DSTATE];
    for (int s = 0; s < params.d_state; ++s) {
        float a_val = reinterpret_cast<float *>(params.a_ptr)[m_id * params.a_d_stride + s];
        float q_val = reinterpret_cast<float *>(params.q_ptr)[m_id * params.q_d_stride + s];
        a_regs[s]     = a_val;
        q_regs[s]     = q_val;
        // Floor a2 exactly as the torch reference does (a2.clamp_min(EPS)).
        // a_bar = exp(delta*a) underflows toward 0 for a large lambda_log, and an
        // unguarded 1/a2 is the one quantity in this kernel that can reach inf.
        inv_a2_regs[s] = 1.0f / fmaxf(a_val * a_val, KLA_EPS);
    }

    // =====================================================================
    // Main loop: sequential over chunks
    // =====================================================================
    for (int chunk = 0; chunk < params.n_chunks; ++chunk) {
        const int seqlen_remaining = params.seqlen - chunk * kChunkSize;

        // =================================================================
        // Load model-axis inputs (shared across all state_idx iterations)
        //   μσ⁻² [B, M, L] and σ⁻² [B, M, L]
        //   Both use smem_load (Zone 1) → sync between them
        // =================================================================
        input_t mu_sig_vals[kNItems], sig_inv_vals[kNItems];

        __syncthreads();  // sync: ensure previous chunk's store is done
        load_input<Ktraits>(mu_sig + chunk * kChunkSize,
                            mu_sig_vals, smem_load, seqlen_remaining);
        __syncthreads();  // sync: smem_load reused for next cooperative load
        load_input<Ktraits>(sig_inv + chunk * kChunkSize,
                            sig_inv_vals, smem_load, seqlen_remaining);

        // Convert to float (register-only, no sync needed)
        float mu_sig_f[kNItems], sig_inv_f[kNItems];
        #pragma unroll
        for (int i = 0; i < kNItems; ++i) {
            mu_sig_f[i]  = float(mu_sig_vals[i]);
            sig_inv_f[i] = float(sig_inv_vals[i]);
        }

        // Output accumulators (summed over state_idx within this chunk)
        float y_accum[kNItems]    = {0};
        float yvar_accum[kNItems] = {0};

        // =================================================================
        // Inner loop: serial over d_state
        // =================================================================
        __syncthreads();  // sync: smem_load done; state loop will reuse Zone 1
        for (int state_idx = 0; state_idx < params.d_state; ++state_idx) {

            // --- Static params for this (m, s) pair ---
            const float a_s    = a_regs[state_idx];
            const float q_s    = q_regs[state_idx];
            const float inv_a2 = inv_a2_regs[state_idx];
            const float a2     = fmaxf(a_s * a_s, KLA_EPS);  // == 1/inv_a2

            // =============================================================
            // Load state-axis inputs: h [B, S, L] and w [B, S, L]
            //   Both use smem_load (Zone 1) → sync between them
            //   Pattern from Mamba-1: sync before each cooperative load
            //   that reuses the same smem region
            // =============================================================
            input_t h_raw[kNItems], w_raw[kNItems];

            __syncthreads();  // sync: ensure previous state_idx iteration's
                              // Zone 2 usage (Möbius/linear scan) is complete
                              // before Zone 1 is used for loads
            load_input<Ktraits>(
                h_base + state_idx * params.h_dstate_stride + chunk * kChunkSize,
                h_raw, smem_load, seqlen_remaining);

            __syncthreads();  // sync: smem_load reused — h load must complete
                              // before w load overwrites the warp-transpose buffer
            load_input<Ktraits>(
                w_base + state_idx * params.w_dstate_stride + chunk * kChunkSize,
                w_raw, smem_load, seqlen_remaining);

            // Convert to float
            float h_f[kNItems], w_f[kNItems];
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                h_f[i] = float(h_raw[i]);
                w_f[i] = float(w_raw[i]);
            }

            // =============================================================
            // Stage 1: Observation sufficient statistics
            //   φ_l = max(h_l² · σ⁻²_l, ε)
            //   r_l = μσ⁻²_l · h_l
            //   (register-only, no smem, no sync)
            // =============================================================
            float phi[kNItems], r[kNItems], raw_phi[kNItems];
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                compute_phi_r(h_f[i], sig_inv_f[i], mu_sig_f[i],
                              phi[i], r[i], raw_phi[i]);
            }

            // =============================================================
            // Stage 2: Det-normalized Möbius scan (3-phase sequential)
            //
            // The Möbius map λ_t = (A·λ_{t-1} + B)/(C·λ_{t-1} + D)
            // is computed via accumulated matrix products P_t = M_t · P_{t-1}
            // with det-normalization at every compose to keep entries O(1).
            //
            // Uses Zone 2 (smem_mob_scratch) for cross-thread exchange.
            // =============================================================

            // Build leaf matrices in registers
            float4 mob_data[kNItems];
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                mob_data[i] = build_leaf_matrix(phi[i], q_s, inv_a2);
                // Positions beyond seqlen get identity (no-op in the scan)
                if constexpr (!kIsEvenLen) {
                    if (threadIdx.x * kNItems + i >= seqlen_remaining)
                        mob_data[i] = matmul_identity();
                }
            }

            // Cross-chunk carry: load from persistent smem (written by
            // previous chunk's Phase 2).  Only thread 0 uses this in
            // Phase 2, but we also extract prev_chunk_lambda for the
            // λ-shift in Stage 3.
            float4 mob_carry;
            if (chunk > 0)
                mob_carry = smem_mob_prefix[state_idx];
            else
                mob_carry = matmul_identity();
            const float prev_chunk_lambda = extract_lambda_lin(mob_carry);

            // ---- Phase 1: per-thread sequential inclusive scan ----
            // Each thread computes the inclusive prefix product of its
            // kNItems leaf matrices.  All in registers, no smem.
            {
                TraceNormMatMulOp op;
                float4 thread_agg = matmul_identity();

                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    thread_agg = op(thread_agg, mob_data[i]);
                    mob_data[i] = thread_agg;  // inclusive scan result
                }

                // Write thread aggregate to Zone 2 for Phase 2
                __syncthreads();  // sync: Zone 2 may still be in use from
                                  // previous state_idx's linear scan
                smem_mob_scratch[threadIdx.x] = thread_agg;
                __syncthreads();  // sync: all aggregates visible to thread 0

                // ---- Phase 2: single-thread exclusive prefix scan ----
                // Thread 0 scans T thread-aggregates left-to-right,
                // incorporating the cross-chunk carry as the initial prefix.
                // Produces exclusive prefixes: smem_mob_scratch[t] = product
                // of all positions to the LEFT of thread t.
                if (threadIdx.x == 0) {
                    float4 carry = mob_carry;
                    for (int t = 0; t < kNThreads; ++t) {
                        float4 old_agg = smem_mob_scratch[t];
                        smem_mob_scratch[t] = carry;  // exclusive prefix for thread t
                        carry = op(carry, old_agg);   // incorporate thread t's aggregate
                    }
                    // Save the total aggregate as carry for the next chunk
                    smem_mob_prefix[state_idx] = carry;
                }
                __syncthreads();  // sync: exclusive prefixes visible to all threads

                // ---- Phase 3: apply exclusive prefix ----
                // Each thread composes its exclusive prefix with each of its
                // local inclusive scan results.
                // Result: mob_data[i] = global inclusive prefix at position tI+i.
                float4 my_prefix = smem_mob_scratch[threadIdx.x];
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    mob_data[i] = op(my_prefix, mob_data[i]);
                }
                // Note: no sync needed here yet — mob_data is in registers.
                // The next smem usage is smem_lambda_shift (persistent region,
                // not aliased with Zone 2).  Zone 2 is not touched until the
                // CUB linear scan later, which has a sync before it.
            }

            // Extract λ from accumulated matrix: λ = (A+B)/(C+D)
            float lambda_vals[kNItems];
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                lambda_vals[i] = extract_lambda_lin(mob_data[i]);
            }

            // Save boundary states to HBM (last thread, last item in chunk)
            if (threadIdx.x == kNThreads - 1) {
                float4 *mob_bnd = reinterpret_cast<float4 *>(params.mob_boundary_ptr);
                float  *lam_bnd = reinterpret_cast<float *>(params.lam_boundary_ptr);
                const int bnd_idx = ((batch_id * params.n_chunks + chunk)
                                     * params.d_model + m_id)
                                    * params.d_state + state_idx;
                mob_bnd[bnd_idx] = mob_data[kNItems - 1];
                lam_bnd[bnd_idx] = lambda_vals[kNItems - 1];
            }

            // =============================================================
            // Stage 3: Linear scan for η (CUB parallel inclusive scan)
            //
            // The linear scan computes:
            //   η_l = α^lin_l · η_{l-1} + r_l
            // where α^lin_l = a / (a² + q·λ_{l-1})
            // and λ_{l-1} is the EXCLUSIVE (previous) Möbius output.
            //
            // The λ-shift requires cross-thread exchange via smem.
            // =============================================================

            // ---- λ-shift: compute prev_lambda[i] = λ_{l-1} ----
            // Each thread needs the lambda of the PREVIOUS position.
            // Within a thread: prev_lambda[i] = lambda_vals[i-1] for i > 0.
            // For i == 0: need the last lambda of the previous thread,
            // obtained via smem exchange.
            float prev_lambda[kNItems];

            __syncthreads();  // sync: Zone 2 reads in Phase 3 are done
            smem_lambda_shift[threadIdx.x] = lambda_vals[kNItems - 1];
            __syncthreads();  // sync: all threads' last lambdas visible

            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                if (i > 0) {
                    prev_lambda[i] = lambda_vals[i - 1];
                } else if (threadIdx.x > 0) {
                    prev_lambda[0] = smem_lambda_shift[threadIdx.x - 1];
                } else {
                    // Thread 0, item 0: use the last lambda from the
                    // previous chunk (or λ₀ = extract_lambda(I₂) = 1)
                    prev_lambda[0] = prev_chunk_lambda;
                }
            }

            // Build linear scan inputs
            float2 lin_data[kNItems];
            float  denom_vals[kNItems];  // saved for backward
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                float alpha_lin_i;
                lin_data[i] = build_linear_input(
                    a_s, a2, q_s, prev_lambda[i], r[i], denom_vals[i]);
                // Positions beyond seqlen get identity
                if constexpr (!kIsEvenLen) {
                    if (threadIdx.x * kNItems + i >= seqlen_remaining)
                        lin_data[i] = linear_identity();
                }
            }

            // ---- CUB inclusive scan (uses Zone 2: smem_lin_scan) ----
            // Cross-chunk carry via prefix callback.
            float2 lin_running_prefix;
            if (chunk > 0 && threadIdx.x % 32 == 0)
                lin_running_prefix = smem_lin_prefix[state_idx];
            else
                lin_running_prefix = linear_identity();
            LinearPrefixCallbackOp lin_prefix_op(lin_running_prefix);

            __syncthreads();  // sync: smem_lambda_shift reads done; Zone 2
                              // (aliased as smem_lin_scan) about to be used
            typename Ktraits::LinearBlockScanT(smem_lin_scan).InclusiveScan(
                lin_data, lin_data, LinearScanOp(), lin_prefix_op);
            // CUB InclusiveScan has an internal __syncthreads() at the end

            // Save cross-chunk carry (thread 0 writes after CUB scan)
            if (threadIdx.x == 0) {
                smem_lin_prefix[state_idx] = lin_prefix_op.running_prefix;
            }

            // Save η boundary to HBM
            if (threadIdx.x == kNThreads - 1) {
                float *lin_bnd = reinterpret_cast<float *>(params.lin_boundary_ptr);
                const int bnd_idx = ((batch_id * params.n_chunks + chunk)
                                     * params.d_model + m_id)
                                    * params.d_state + state_idx;
                lin_bnd[bnd_idx] = lin_data[kNItems - 1].y;
            }

            // =============================================================
            // Stage 4: Posterior contraction (register-only)
            //
            //   m_l = η_l / λ_l        (posterior mean for this state dim)
            //   y   += m_l · w_l        (sum over state dims)
            //   yvar += (1/λ_l) · w_l²  (sum over state dims)
            // =============================================================
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                const float eta_i = lin_data[i].y;
                const float lam_i = lambda_vals[i];
                const float m_i   = eta_i / lam_i;

                y_accum[i]    += m_i * w_f[i];
                yvar_accum[i] += (1.0f / lam_i) * w_f[i] * w_f[i];
            }

        }  // end for state_idx

        // =================================================================
        // Store outputs to HBM via CUB BlockStore (Zone 1)
        // =================================================================
        input_t y_out_vals[kNItems], yvar_out_vals[kNItems];
        #pragma unroll
        for (int i = 0; i < kNItems; ++i) {
            y_out_vals[i]    = input_t(y_accum[i]);
            yvar_out_vals[i] = input_t(yvar_accum[i]);
        }

        __syncthreads();  // sync: CUB linear scan done; Zone 1 about to be
                          // used for cooperative store
        store_output<Ktraits>(y_out + chunk * kChunkSize,
                              y_out_vals, smem_store, seqlen_remaining);
        __syncthreads();  // sync: smem_store reused for second store
        store_output<Ktraits>(yvar_out + chunk * kChunkSize,
                              yvar_out_vals, smem_store, seqlen_remaining);
        // The __syncthreads() at the top of the next chunk iteration
        // ensures this store completes before smem is reused.

    }  // end for chunk
}

// =============================================================================
// Launch dispatch
// =============================================================================
template<int kNThreads, int kNItems, typename input_t>
void kla_matmul_fwd_launch(KLAMatmulParamsFwd &params, cudaStream_t stream) {
    BOOL_SWITCH(params.seqlen % (kNThreads * kNItems) == 0, kIsEvenLen, [&] {
        using Ktraits = KLA_Matmul_fwd_kernel_traits<kNThreads, kNItems, kIsEvenLen, input_t>;

        constexpr int kSmemSize = Ktraits::kSmemSize
            + MAX_DSTATE * sizeof(float4)            // smem_mob_prefix
            + MAX_DSTATE * sizeof(float2)            // smem_lin_prefix
            + Ktraits::kNThreads * sizeof(float);    // smem_lambda_shift

        dim3 grid(params.batch, params.d_model);
        auto kernel = &kla_matmul_fwd_kernel<Ktraits>;

        if (kSmemSize >= 48 * 1024) {
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
        }
        kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

template<typename input_t>
void kla_matmul_fwd_cuda(KLAMatmulParamsFwd &params, cudaStream_t stream) {
    if (params.seqlen <= 128) {
        kla_matmul_fwd_launch<32, 4, input_t>(params, stream);
    } else {
        kla_matmul_fwd_launch<64, 8, input_t>(params, stream);
    }
}
