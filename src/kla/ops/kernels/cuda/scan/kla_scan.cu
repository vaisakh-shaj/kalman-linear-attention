/******************************************************************************
 * The exact CUDA KLA scans -- launchers and the python entry points.
 *   kla_scan.cu
 *
 * cuda_recurrent, cuda_chunk, cuda_pscan, and the one backward all three share.
 * Every kernel is specialized on BLOCK_S = next_pow2(d_state) so the read-out
 * reduction unrolls to a fixed shape and the backward's replay buffers stay
 * fixed-size registers; ROWS follows from aiming a block at KLA_TG_THREADS.
 * What ROWS *means* differs per implementation -- channels stacked for recurrent,
 * timesteps split for chunk, chunks stacked for pscan -- so each entry point
 * declares its own grid.
 *
 * These replace the v2_* kernels rather than extending them: same algebra,
 * different adjoint. See kla_scan_bwd.cuh for why the new one is both exact and
 * cheaper. v2_1 and v2_2 stay in the tree as the comparison.
 ******************************************************************************/
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <utility>
#include <vector>

#include "kla_chunk_fwd.cuh"
#include "kla_pscan_fwd.cuh"
#include "kla_recurrent_fwd.cuh"
#include "kla_scan_bwd.cuh"

namespace {

int next_pow2(int n) {
    int v = 1;
    while (v < n) v <<= 1;
    return v;
}

void check(const torch::Tensor &t, const char *name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

// A block is always [BLOCK_S, ROWS] aiming at KLA_TG_THREADS. What ROWS *means*
// differs: for the recurrent implementation it stacks channels, for the chunk one it
// spans time, so each entry point declares its own grid. LAUNCH_BODY does that,
// and this pair only supplies the two compile-time constants.
#define KLA_LAUNCH(BS)                                                       \
    case BS: {                                                               \
        constexpr int BS_ = (BS);                                            \
        constexpr int ROWS_ =                                                \
            KLA_TG_THREADS / (BS) > 0 ? KLA_TG_THREADS / (BS) : 1;           \
        LAUNCH_BODY(BS_, ROWS_);                                             \
        break;                                                               \
    }

#define KLA_DISPATCH_BLOCK_S(S_)                                             \
    switch (next_pow2(S_)) {                                                 \
        KLA_LAUNCH(1)                                                        \
        KLA_LAUNCH(2)                                                        \
        KLA_LAUNCH(4)                                                        \
        KLA_LAUNCH(8)                                                        \
        KLA_LAUNCH(16)                                                       \
        KLA_LAUNCH(32)                                                       \
        KLA_LAUNCH(64)                                                       \
        default:                                                             \
            TORCH_CHECK(false, "d_state must be <= ", KLA_MAX_DSTATE);       \
    }

}  // namespace

std::vector<torch::Tensor> recurrent_fwd(torch::Tensor msi, torch::Tensor si,
                                         torch::Tensor k, torch::Tensor q,
                                         torch::Tensor a, torch::Tensor p,
                                         torch::Tensor lam0, torch::Tensor eta0,
                                         bool checkpoints, bool prior) {
    check(msi, "msi"); check(si, "si"); check(k, "k"); check(q, "q");
    check(a, "a"); check(p, "p"); check(lam0, "lam0"); check(eta0, "eta0");

    const int B = msi.size(0), L = msi.size(1), M = msi.size(2);
    const int S = k.size(2);
    TORCH_CHECK(S <= KLA_MAX_DSTATE, "d_state must be <= ", KLA_MAX_DSTATE);

    const at::cuda::OptionalCUDAGuard guard(device_of(msi));
    auto opts = msi.options();
    auto y = torch::empty({B, L, M}, opts);
    auto yvar = torch::empty({B, L, M}, opts);
    auto lam_fin = torch::empty({B, M, S}, opts);
    auto eta_fin = torch::empty({B, M, S}, opts);

    const int NCK = checkpoints ? (L + KLA_CHUNK - 1) / KLA_CHUNK : 1;
    auto ck_shape = checkpoints ? std::vector<int64_t>{B, M, NCK, S}
                                : std::vector<int64_t>{1};
    auto lam_ck = torch::empty(ck_shape, opts);
    auto eta_ck = torch::empty(ck_shape, opts);

    auto stream = at::cuda::getCurrentCUDAStream();
#define LAUNCH_BODY(BS, RW)                                                  \
    kla_recurrent_fwd_kernel<BS, RW>                                         \
        <<<dim3((M + (RW) - 1) / (RW), B), dim3(BS, RW), 0, stream>>>(       \
        y.data_ptr<float>(), yvar.data_ptr<float>(),                         \
        lam_fin.data_ptr<float>(), eta_fin.data_ptr<float>(),                \
        lam_ck.data_ptr<float>(), eta_ck.data_ptr<float>(),                  \
        msi.data_ptr<float>(), si.data_ptr<float>(), k.data_ptr<float>(),    \
        q.data_ptr<float>(), a.data_ptr<float>(), p.data_ptr<float>(),       \
        lam0.data_ptr<float>(), eta0.data_ptr<float>(), L, M, S, NCK,        \
        int(checkpoints), int(prior))
    KLA_DISPATCH_BLOCK_S(S)
#undef LAUNCH_BODY
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {y, yvar, lam_fin, eta_fin, lam_ck, eta_ck};
}

std::vector<torch::Tensor> chunk_fwd(torch::Tensor msi, torch::Tensor si,
                                    torch::Tensor k, torch::Tensor q,
                                    torch::Tensor a, torch::Tensor p,
                                    torch::Tensor lam0, torch::Tensor eta0,
                                    bool checkpoints, bool prior) {
    check(msi, "msi"); check(si, "si"); check(k, "k"); check(q, "q");
    check(a, "a"); check(p, "p"); check(lam0, "lam0"); check(eta0, "eta0");

    const int B = msi.size(0), L = msi.size(1), M = msi.size(2);
    const int S = k.size(2);
    TORCH_CHECK(S <= KLA_MAX_DSTATE, "d_state must be <= ", KLA_MAX_DSTATE);

    const at::cuda::OptionalCUDAGuard guard(device_of(msi));
    auto opts = msi.options();
    auto y = torch::empty({B, L, M}, opts);
    auto yvar = torch::empty({B, L, M}, opts);
    auto lam_fin = torch::empty({B, M, S}, opts);
    auto eta_fin = torch::empty({B, M, S}, opts);

    const int NCK = checkpoints ? (L + KLA_CHUNK - 1) / KLA_CHUNK : 1;
    auto ck_shape = checkpoints ? std::vector<int64_t>{B, M, NCK, S}
                                : std::vector<int64_t>{1};
    auto lam_ck = torch::empty(ck_shape, opts);
    auto eta_ck = torch::empty(ck_shape, opts);

    auto stream = at::cuda::getCurrentCUDAStream();
    // One block per (batch, channel): here ROWS spans *time*, not channels.
#define LAUNCH_BODY(BS, RW)                                                  \
    kla_chunk_fwd_kernel<BS, RW, KLA_ITEMS>                                  \
        <<<dim3(M, B), dim3(BS, RW), 0, stream>>>(                           \
            y.data_ptr<float>(), yvar.data_ptr<float>(),                     \
            lam_fin.data_ptr<float>(), eta_fin.data_ptr<float>(),            \
            lam_ck.data_ptr<float>(), eta_ck.data_ptr<float>(),              \
            msi.data_ptr<float>(), si.data_ptr<float>(), k.data_ptr<float>(),\
            q.data_ptr<float>(), a.data_ptr<float>(), p.data_ptr<float>(),   \
            lam0.data_ptr<float>(), eta0.data_ptr<float>(), L, M, S, NCK,    \
            int(checkpoints), int(prior))
    KLA_DISPATCH_BLOCK_S(S)
#undef LAUNCH_BODY
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {y, yvar, lam_fin, eta_fin, lam_ck, eta_ck};
}

std::vector<torch::Tensor> pscan_fwd(torch::Tensor msi, torch::Tensor si,
                                     torch::Tensor k, torch::Tensor q,
                                     torch::Tensor a, torch::Tensor p,
                                     torch::Tensor lam0, torch::Tensor eta0,
                                     bool prior) {
    check(msi, "msi"); check(si, "si"); check(k, "k"); check(q, "q");
    check(a, "a"); check(p, "p"); check(lam0, "lam0"); check(eta0, "eta0");

    const int B = msi.size(0), L = msi.size(1), M = msi.size(2);
    const int S = k.size(2);
    TORCH_CHECK(S <= KLA_MAX_DSTATE, "d_state must be <= ", KLA_MAX_DSTATE);

    const at::cuda::OptionalCUDAGuard guard(device_of(msi));
    auto opts = msi.options();
    auto y = torch::empty({B, L, M}, opts);
    auto yvar = torch::empty({B, L, M}, opts);
    auto lam_fin = torch::empty({B, M, S}, opts);
    auto eta_fin = torch::empty({B, M, S}, opts);

    // No `checkpoints` flag, unlike the other two forwards: the checkpoints are
    // this implementation's own intermediates, so there is nothing to skip.
    const int NCK = (L + KLA_CHUNK - 1) / KLA_CHUNK;
    auto lam_ck = torch::empty({B, M, NCK, S}, opts);
    auto eta_ck = torch::empty({B, M, NCK, S}, opts);

    // The scan aggregates: a 2x2 Moebius map and an affine pair per chunk, each
    // double-buffered for the ping-pong. torch allocations are 512-byte
    // aligned, so the float4 view is aligned too.
    auto mob_t = torch::empty({B, M, NCK, S, 4}, opts);
    auto mob_alt_t = torch::empty({B, M, NCK, S, 4}, opts);
    auto aff_t = torch::empty({B, M, NCK, S, 2}, opts);
    auto aff_alt_t = torch::empty({B, M, NCK, S, 2}, opts);
    auto *mob = reinterpret_cast<float4 *>(mob_t.data_ptr<float>());
    auto *mob_alt = reinterpret_cast<float4 *>(mob_alt_t.data_ptr<float>());
    auto *aff = reinterpret_cast<float2 *>(aff_t.data_ptr<float>());
    auto *aff_alt = reinterpret_cast<float2 *>(aff_alt_t.data_ptr<float>());

    const int total = B * M * NCK * S;
    const int flat = (total + KLA_TG_THREADS - 1) / KLA_TG_THREADS;
    auto stream = at::cuda::getCurrentCUDAStream();

    kla_pscan_mob_reduce_kernel<<<flat, KLA_TG_THREADS, 0, stream>>>(
        mob, si.data_ptr<float>(), k.data_ptr<float>(), a.data_ptr<float>(),
        p.data_ptr<float>(), L, M, S, NCK, total);
    for (int off = 1; off < NCK; off <<= 1) {
        kla_pscan_mob_step_kernel<<<flat, KLA_TG_THREADS, 0, stream>>>(
            mob_alt, mob, S, NCK, off, total);
        std::swap(mob, mob_alt);
    }

    kla_pscan_aff_reduce_kernel<<<flat, KLA_TG_THREADS, 0, stream>>>(
        aff, lam_ck.data_ptr<float>(), mob, msi.data_ptr<float>(),
        si.data_ptr<float>(), k.data_ptr<float>(), a.data_ptr<float>(),
        p.data_ptr<float>(), lam0.data_ptr<float>(), L, M, S, NCK, total);
    for (int off = 1; off < NCK; off <<= 1) {
        kla_pscan_aff_step_kernel<<<flat, KLA_TG_THREADS, 0, stream>>>(
            aff_alt, aff, S, NCK, off, total);
        std::swap(aff, aff_alt);
    }

    // Here ROWS spans *chunks*, not channels and not timesteps.
#define LAUNCH_BODY(BS, RW)                                                  \
    kla_pscan_apply_kernel<BS, RW>                                           \
        <<<dim3((NCK + (RW) - 1) / (RW), M, B), dim3(BS, RW), 0, stream>>>(  \
            y.data_ptr<float>(), yvar.data_ptr<float>(),                     \
            lam_fin.data_ptr<float>(), eta_fin.data_ptr<float>(),            \
            eta_ck.data_ptr<float>(), lam_ck.data_ptr<float>(), aff,         \
            msi.data_ptr<float>(), si.data_ptr<float>(), k.data_ptr<float>(),\
            q.data_ptr<float>(), a.data_ptr<float>(), p.data_ptr<float>(),   \
            eta0.data_ptr<float>(), L, M, S, NCK, int(prior))
    KLA_DISPATCH_BLOCK_S(S)
#undef LAUNCH_BODY
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {y, yvar, lam_fin, eta_fin, lam_ck, eta_ck};
}

std::vector<torch::Tensor> scan_bwd(torch::Tensor dy, torch::Tensor dyvar,
                                    torch::Tensor dlam_fin, torch::Tensor deta_fin,
                                    torch::Tensor msi, torch::Tensor si,
                                    torch::Tensor k, torch::Tensor q,
                                    torch::Tensor a, torch::Tensor p,
                                    torch::Tensor lam0, torch::Tensor eta0,
                                    torch::Tensor lam_ck, torch::Tensor eta_ck,
                                    bool prior) {
    check(dy, "dy"); check(dyvar, "dyvar"); check(msi, "msi"); check(si, "si");
    check(k, "k"); check(q, "q"); check(lam_ck, "lam_ck"); check(eta_ck, "eta_ck");

    const int B = msi.size(0), L = msi.size(1), M = msi.size(2);
    const int S = k.size(2);
    const int NCK = lam_ck.size(2);

    const at::cuda::OptionalCUDAGuard guard(device_of(msi));
    auto opts = msi.options();
    auto dmsi = torch::empty({B, L, M}, opts);
    auto dsi = torch::empty({B, L, M}, opts);
    auto dlam0 = torch::empty({B, M, S}, opts);
    auto deta0 = torch::empty({B, M, S}, opts);
    // dk/dq contract over channels and da/dp over batch and time; neither axis
    // is owned by a single block, so these four accumulate atomically.
    auto dk = torch::zeros({B, L, S}, opts);
    auto dq = torch::zeros({B, L, S}, opts);
    auto da = torch::zeros_like(a);
    auto dp = torch::zeros_like(p);

    auto stream = at::cuda::getCurrentCUDAStream();
#define LAUNCH_BODY(BS, RW)                                                  \
    kla_scan_bwd_kernel<BS, RW>                                              \
        <<<dim3((M + (RW) - 1) / (RW), B), dim3(BS, RW), 0, stream>>>(       \
        dk.data_ptr<float>(), dq.data_ptr<float>(), da.data_ptr<float>(),    \
        dp.data_ptr<float>(), dmsi.data_ptr<float>(), dsi.data_ptr<float>(), \
        dlam0.data_ptr<float>(), deta0.data_ptr<float>(),                    \
        dy.data_ptr<float>(), dyvar.data_ptr<float>(),                       \
        dlam_fin.data_ptr<float>(), deta_fin.data_ptr<float>(),              \
        msi.data_ptr<float>(), si.data_ptr<float>(), k.data_ptr<float>(),    \
        q.data_ptr<float>(), a.data_ptr<float>(), p.data_ptr<float>(),       \
        lam0.data_ptr<float>(), eta0.data_ptr<float>(),                      \
        lam_ck.data_ptr<float>(), eta_ck.data_ptr<float>(), L, M, S, NCK,    \
        int(prior))
    KLA_DISPATCH_BLOCK_S(S)
#undef LAUNCH_BODY
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dmsi, dsi, dk, dq, da, dp, dlam0, deta0};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("recurrent_fwd", &recurrent_fwd, "KLA recurrent forward (CUDA)");
    m.def("chunk_fwd", &chunk_fwd, "KLA chunk forward, time-parallel (CUDA)");
    m.def("pscan_fwd", &pscan_fwd,
          "KLA parallel-scan forward, no serial carry (CUDA)");
    m.def("bwd", &scan_bwd, "KLA exact backward, shared by every implementation (CUDA)");
}
