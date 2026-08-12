/******************************************************************************
 * KLA Matmul Scan — Parameter Structs (static a,q)
 *   kla_matmul_scan.h
 ******************************************************************************/
#pragma once

#define MAX_DSTATE 64
#define KLA_EPS     1e-12f
#define KLA_LOG_EPS 1e-30f

// =============================================================================
// Forward parameters
// =============================================================================
struct KLAMatmulParamsFwd {
    int batch;
    int seqlen;
    int d_model;
    int d_state;
    int n_chunks;

    // Inputs: model-axis [B, M, L]
    void *mu_sigma_inv_ptr;
    int mu_sigma_inv_batch_stride;
    int mu_sigma_inv_d_stride;

    void *sigma_inv_ptr;
    int sigma_inv_batch_stride;
    int sigma_inv_d_stride;

    // Inputs: state-axis [B, S, L]
    void *h_ptr;
    int h_batch_stride;
    int h_dstate_stride;

    void *w_ptr;
    int w_batch_stride;
    int w_dstate_stride;

    // Static params [M, S]
    void *a_ptr;
    int a_d_stride;

    void *q_ptr;
    int q_d_stride;

    // Outputs: [B, M, L]
    void *y_ptr;
    int y_batch_stride;
    int y_d_stride;

    void *yvar_ptr;
    int yvar_batch_stride;
    int yvar_d_stride;

    // Boundary states (cross-chunk carries)
    void *mob_boundary_ptr;   // [B, n_chunks, M, S, 4]
    void *lin_boundary_ptr;   // [B, n_chunks, M, S]
    void *lam_boundary_ptr;   // [B, n_chunks, M, S]
};

// =============================================================================
// Backward parameters
// =============================================================================
struct KLAMatmulParamsBwd {
    int batch;
    int seqlen;
    int d_model;
    int d_state;
    int n_chunks;

    // Upstream gradients: [B, M, L]
    void *dy_ptr;
    int dy_batch_stride;
    int dy_d_stride;

    void *dyvar_ptr;
    int dyvar_batch_stride;
    int dyvar_d_stride;

    // Saved inputs: model-axis [B, M, L]
    void *mu_sigma_inv_ptr;
    int mu_sigma_inv_batch_stride;
    int mu_sigma_inv_d_stride;

    void *sigma_inv_ptr;
    int sigma_inv_batch_stride;
    int sigma_inv_d_stride;

    // Saved inputs: state-axis [B, S, L]
    void *h_ptr;
    int h_batch_stride;
    int h_dstate_stride;

    void *w_ptr;
    int w_batch_stride;
    int w_dstate_stride;

    // Static params [M, S]
    void *a_ptr;
    int a_d_stride;

    void *q_ptr;
    int q_d_stride;

    // Boundary states (from forward)
    void *mob_boundary_ptr;
    void *lin_boundary_ptr;
    void *lam_boundary_ptr;

    // Output gradients: model-axis [B, M, L]
    void *dmu_sigma_inv_ptr;
    int dmu_batch_stride;
    int dmu_d_stride;

    void *dsigma_inv_ptr;
    int dsig_batch_stride;
    int dsig_d_stride;

    // Output gradients: state-axis [B, S, L] (atomicAdd over M)
    void *dh_ptr;
    int dh_batch_stride;
    int dh_dstate_stride;

    void *dw_ptr;
    int dw_batch_stride;
    int dw_dstate_stride;

    // Output gradients: static params [M, S] (atomicAdd over B, L)
    void *da_ptr;
    int da_d_stride;

    void *dq_ptr;
    int dq_d_stride;
};
