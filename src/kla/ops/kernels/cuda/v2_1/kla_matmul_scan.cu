/******************************************************************************
 * KLA Matmul Scan — Launch Wrappers + PyTorch Binding
 *   kla_matmul_scan.cu
 *
 * Static dynamics: a [M,S] and q [M,S].
 * Linear-space det-normalized Möbius scan.
 ******************************************************************************/

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include "kla_matmul_scan.h"
#include "kla_matmul_fwd_kernel.cuh"
#include "kla_matmul_bwd_kernel.cuh"
#include "static_switch.h"

#define CHECK_SHAPE(x, ...) \
    TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), \
                #x " must have shape (" #__VA_ARGS__ ")")

// =============================================================================
// PyTorch C++ binding: forward
// =============================================================================

std::vector<at::Tensor> kla_matmul_fwd(
    const at::Tensor &mu_sigma_inv,   // (B, M, L)
    const at::Tensor &sigma_inv,      // (B, M, L)
    const at::Tensor &h,              // (B, S, L)
    const at::Tensor &w,              // (B, S, L)
    const at::Tensor &a,              // (M, S) — discrete transition decay
    const at::Tensor &q               // (M, S) — discrete process noise
) {
    TORCH_CHECK(mu_sigma_inv.is_cuda());
    TORCH_CHECK(mu_sigma_inv.dim() == 3);
    TORCH_CHECK(sigma_inv.dim() == 3);
    TORCH_CHECK(h.dim() == 3);
    TORCH_CHECK(w.dim() == 3);
    TORCH_CHECK(a.dim() == 2);
    TORCH_CHECK(q.dim() == 2);

    TORCH_CHECK(mu_sigma_inv.stride(-1) == 1);
    TORCH_CHECK(sigma_inv.stride(-1) == 1);
    TORCH_CHECK(h.stride(-1) == 1);
    TORCH_CHECK(w.stride(-1) == 1);

    const int batch   = mu_sigma_inv.size(0);
    const int d_model = mu_sigma_inv.size(1);
    const int seqlen  = mu_sigma_inv.size(2);
    const int d_state = h.size(1);

    CHECK_SHAPE(mu_sigma_inv, batch, d_model, seqlen);
    CHECK_SHAPE(sigma_inv, batch, d_model, seqlen);
    CHECK_SHAPE(h, batch, d_state, seqlen);
    CHECK_SHAPE(w, batch, d_state, seqlen);
    CHECK_SHAPE(a, d_model, d_state);
    CHECK_SHAPE(q, d_model, d_state);
    TORCH_CHECK(d_state <= MAX_DSTATE, "d_state must be <= ", MAX_DSTATE);

    // Compute chunk size from dispatch
    int kChunkSize;
    if (seqlen <= 128) kChunkSize = 32 * 4;
    else               kChunkSize = 64 * 8;
    const int n_chunks = (seqlen + kChunkSize - 1) / kChunkSize;

    auto opts     = mu_sigma_inv.options();
    auto opts_f32 = opts.dtype(torch::kFloat32);

    at::Tensor y    = torch::empty({batch, d_model, seqlen}, opts);
    at::Tensor yvar = torch::empty({batch, d_model, seqlen}, opts);
    at::Tensor mob_boundary = torch::empty({batch, n_chunks, d_model, d_state, 4}, opts_f32);
    at::Tensor lin_boundary = torch::empty({batch, n_chunks, d_model, d_state}, opts_f32);
    at::Tensor lam_boundary = torch::empty({batch, n_chunks, d_model, d_state}, opts_f32);

    KLAMatmulParamsFwd params;
    memset(&params, 0, sizeof(params));
    params.batch   = batch;
    params.seqlen  = seqlen;
    params.d_model = d_model;
    params.d_state = d_state;
    params.n_chunks = n_chunks;

    params.mu_sigma_inv_ptr          = mu_sigma_inv.data_ptr();
    params.mu_sigma_inv_batch_stride = mu_sigma_inv.stride(0);
    params.mu_sigma_inv_d_stride     = mu_sigma_inv.stride(1);

    params.sigma_inv_ptr          = sigma_inv.data_ptr();
    params.sigma_inv_batch_stride = sigma_inv.stride(0);
    params.sigma_inv_d_stride     = sigma_inv.stride(1);

    params.h_ptr          = h.data_ptr();
    params.h_batch_stride = h.stride(0);
    params.h_dstate_stride = h.stride(1);

    params.w_ptr          = w.data_ptr();
    params.w_batch_stride = w.stride(0);
    params.w_dstate_stride = w.stride(1);

    params.a_ptr      = a.data_ptr();
    params.a_d_stride = a.stride(0);
    params.q_ptr      = q.data_ptr();
    params.q_d_stride = q.stride(0);

    params.y_ptr          = y.data_ptr();
    params.y_batch_stride = y.stride(0);
    params.y_d_stride     = y.stride(1);
    params.yvar_ptr          = yvar.data_ptr();
    params.yvar_batch_stride = yvar.stride(0);
    params.yvar_d_stride     = yvar.stride(1);

    params.mob_boundary_ptr = mob_boundary.data_ptr();
    params.lin_boundary_ptr = lin_boundary.data_ptr();
    params.lam_boundary_ptr = lam_boundary.data_ptr();

    at::cuda::CUDAGuard device_guard{mu_sigma_inv.device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    kla_matmul_fwd_cuda<float>(params, stream);

    return {y, yvar, mob_boundary, lin_boundary, lam_boundary};
}

// =============================================================================
// PyTorch C++ binding: backward
// =============================================================================

std::vector<at::Tensor> kla_matmul_bwd(
    const at::Tensor &dy,              // (B, M, L)
    const at::Tensor &dyvar,           // (B, M, L)
    const at::Tensor &mu_sigma_inv,    // (B, M, L)
    const at::Tensor &sigma_inv,       // (B, M, L)
    const at::Tensor &h,               // (B, S, L)
    const at::Tensor &w,               // (B, S, L)
    const at::Tensor &a,               // (M, S)
    const at::Tensor &q,               // (M, S)
    const at::Tensor &mob_boundary,    // (B, n_chunks, M, S, 4)
    const at::Tensor &lin_boundary,    // (B, n_chunks, M, S)
    const at::Tensor &lam_boundary     // (B, n_chunks, M, S)
) {
    const int batch   = dy.size(0);
    const int d_model = dy.size(1);
    const int seqlen  = dy.size(2);
    const int d_state = h.size(1);
    const int n_chunks = mob_boundary.size(1);

    auto opts     = dy.options();
    auto opts_f32 = opts.dtype(torch::kFloat32);

    at::Tensor dmu_sigma_inv = torch::empty({batch, d_model, seqlen}, opts);
    at::Tensor dsigma_inv    = torch::empty({batch, d_model, seqlen}, opts);
    at::Tensor dh = torch::zeros({batch, d_state, seqlen}, opts_f32);
    at::Tensor dw = torch::zeros({batch, d_state, seqlen}, opts_f32);
    at::Tensor da = torch::zeros({d_model, d_state}, opts_f32);
    at::Tensor dq = torch::zeros({d_model, d_state}, opts_f32);

    KLAMatmulParamsBwd params;
    memset(&params, 0, sizeof(params));
    params.batch   = batch;
    params.seqlen  = seqlen;
    params.d_model = d_model;
    params.d_state = d_state;
    params.n_chunks = n_chunks;

    // Upstream gradients
    params.dy_ptr          = dy.data_ptr();
    params.dy_batch_stride = dy.stride(0);
    params.dy_d_stride     = dy.stride(1);
    params.dyvar_ptr          = dyvar.data_ptr();
    params.dyvar_batch_stride = dyvar.stride(0);
    params.dyvar_d_stride     = dyvar.stride(1);

    // Saved inputs: model-axis
    params.mu_sigma_inv_ptr          = mu_sigma_inv.data_ptr();
    params.mu_sigma_inv_batch_stride = mu_sigma_inv.stride(0);
    params.mu_sigma_inv_d_stride     = mu_sigma_inv.stride(1);
    params.sigma_inv_ptr          = sigma_inv.data_ptr();
    params.sigma_inv_batch_stride = sigma_inv.stride(0);
    params.sigma_inv_d_stride     = sigma_inv.stride(1);

    // Saved inputs: state-axis
    params.h_ptr          = h.data_ptr();
    params.h_batch_stride = h.stride(0);
    params.h_dstate_stride = h.stride(1);
    params.w_ptr          = w.data_ptr();
    params.w_batch_stride = w.stride(0);
    params.w_dstate_stride = w.stride(1);

    // Static params
    params.a_ptr      = a.data_ptr();
    params.a_d_stride = a.stride(0);
    params.q_ptr      = q.data_ptr();
    params.q_d_stride = q.stride(0);

    // Boundaries
    params.mob_boundary_ptr = mob_boundary.data_ptr();
    params.lin_boundary_ptr = lin_boundary.data_ptr();
    params.lam_boundary_ptr = lam_boundary.data_ptr();

    // Output gradients: model-axis
    params.dmu_sigma_inv_ptr = dmu_sigma_inv.data_ptr();
    params.dmu_batch_stride  = dmu_sigma_inv.stride(0);
    params.dmu_d_stride      = dmu_sigma_inv.stride(1);
    params.dsigma_inv_ptr = dsigma_inv.data_ptr();
    params.dsig_batch_stride = dsigma_inv.stride(0);
    params.dsig_d_stride     = dsigma_inv.stride(1);

    // Output gradients: state-axis
    params.dh_ptr          = dh.data_ptr();
    params.dh_batch_stride = dh.stride(0);
    params.dh_dstate_stride = dh.stride(1);
    params.dw_ptr          = dw.data_ptr();
    params.dw_batch_stride = dw.stride(0);
    params.dw_dstate_stride = dw.stride(1);

    // Output gradients: static
    params.da_ptr      = da.data_ptr();
    params.da_d_stride = da.stride(0);
    params.dq_ptr      = dq.data_ptr();
    params.dq_d_stride = dq.stride(0);

    at::cuda::CUDAGuard device_guard{dy.device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    kla_matmul_bwd_cuda<float>(params, stream);

    return {dmu_sigma_inv, dsigma_inv, dh, dw, da, dq};
}

// =============================================================================
// Module registration
// =============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd", &kla_matmul_fwd, "KLA matmul scan forward (CUDA, det-norm)");
    m.def("bwd", &kla_matmul_bwd, "KLA matmul scan backward (CUDA, det-norm)");
}
