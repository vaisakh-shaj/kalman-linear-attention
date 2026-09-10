/******************************************************************************
 * BlockReverseScan — Reverse (right-to-left) inclusive prefix scan
 *
 * 4-phase implementation:
 *   Phase 1: per-thread reverse sequential scan (registers)
 *   Phase 2: write per-thread aggregates to smem (reversed order)
 *   Phase 3: single-thread sequential scan over aggregates (thread 0)
 *   Phase 4: apply exclusive prefix to local results
 *
 * scan_op convention: scan_op(older_aggregate, newer_input)
 *   In a reverse scan, the "newer" element is further LEFT (smaller index).
 *   The aggregate accumulated from the right is "older".
 *
 * This gives the correct reverse recurrence:
 *   η_total[l] = η_dir[l] + α[l+1] · η_total[l+1]
 *
 * where input[l] = (α[l+1], η_dir[l]) and LinearScanOp produces:
 *   (input.x * agg.x, input.x * agg.y + input.y)
 * = (α[l+1] * α_product, α[l+1] * η_accumulated + η_dir[l])  ✓
 ******************************************************************************/
#pragma once

#include <cuda_runtime.h>

template<typename scan_t, int kNThreads>
struct BlockReverseScan {

    struct TempStorage {
        scan_t thread_aggregates[kNThreads];
    };

    TempStorage &temp_storage;

    __device__ BlockReverseScan(TempStorage &ts) : temp_storage(ts) {}

    template<int kNItems, typename ScanOp, typename PrefixOp>
    __device__ void InclusiveReverseScan(
        scan_t (&input)[kNItems],
        scan_t (&output)[kNItems],
        ScanOp scan_op,
        PrefixOp &prefix_op
    ) {
        const int tid = threadIdx.x;

        // Phase 1: per-thread reverse sequential scan
        scan_t local_aggregate = prefix_op.identity();

        #pragma unroll
        for (int i = kNItems - 1; i >= 0; --i) {
            local_aggregate = scan_op(local_aggregate, input[i]);
            output[i] = local_aggregate;
        }

        // Phase 2: write per-thread aggregates in REVERSED order
        int reversed_tid = kNThreads - 1 - tid;
        temp_storage.thread_aggregates[reversed_tid] = local_aggregate;
        __syncthreads();

        // Phase 3: single-thread sequential scan (thread 0)
        // Aggregates are stored in reversed order:
        //   [0] = rightmost thread's aggregate (oldest)
        //   [kNThreads-1] = leftmost thread's aggregate (newest)
        if (tid == 0) {
            scan_t carry = prefix_op.running_prefix;

            for (int t = 0; t < kNThreads; ++t) {
                scan_t thread_agg = temp_storage.thread_aggregates[t];
                temp_storage.thread_aggregates[t] = carry;
                carry = scan_op(carry, thread_agg);
            }

            prefix_op.running_prefix = carry;
        }
        __syncthreads();

        // Phase 4: apply exclusive prefix to local results
        scan_t my_prefix = temp_storage.thread_aggregates[reversed_tid];

        #pragma unroll
        for (int i = 0; i < kNItems; ++i) {
            output[i] = scan_op(my_prefix, output[i]);
        }
    }
};
