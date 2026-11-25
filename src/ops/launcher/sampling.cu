// Implements: include/infernix/ops/sampling.h
// Match: validated contiguous FP32 or BF16 logits, I32 tensors and a shared-layout workspace.
// Algorithm assumptions: launcher and kernels use sampler_multiblock_ok() from
// the same layout authority, so exactly one finite route owns each shape.
#include "ops/launcher/sampling.h"

#include "ops/common/math.h"
#include "ops/kernel/sampling.cuh"
#include "core/device.h"
#include "core/pdl.cuh"

namespace infernix::ops::detail {

__global__ void increment_token_counts_kernel(const std::int32_t* token_ids, std::int32_t count,
                                              std::int32_t* token_counts) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) { atomicAdd(&token_counts[token_ids[index]], 1); }
}

std::size_t sampling_workspace_exact_bytes(std::int32_t token_domain, std::int32_t columns) {
    return make_sampling_workspace_layout(token_domain, columns).bytes;
}

namespace {

template <class Logit>
void sample_batch(const Logit* logits, Tensor& out, std::int32_t token_domain, std::int32_t physical_rows,
                  std::int32_t batch, const SamplingConfig* configs, const std::int32_t* positions,
                  std::int32_t purpose, DeviceSpan workspace, cudaStream_t stream) {
    const SamplingWorkspaceLayout layout = make_sampling_workspace_layout(token_domain, batch);
    if (!layout.multiblock) {
        sample_row_kernel<Logit, SamplerRowSchedule>
            <<<static_cast<unsigned int>(batch), SamplerRowSchedule::kSamplerBlock, 0, stream>>>(
                logits, static_cast<std::int32_t*>(out.data), configs, positions, purpose,
                token_domain, physical_rows);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const std::int32_t partial_blocks =
        div_up(token_domain, SamplerParallelSchedule::kSamplerPartialTileItems);
    const std::int32_t groups       = sampler_group_count(partial_blocks);
    const SamplingWorkspace scratch = layout.bind(workspace);
    const dim3 partial_grid(static_cast<unsigned int>(partial_blocks),
                            static_cast<unsigned int>(batch));
    sampling_partial_topk_kernel<Logit, SamplerParallelSchedule, true>
        <<<partial_grid, SamplerParallelSchedule::kSamplerBlock, 0, stream>>>(
            logits, configs, token_domain, physical_rows, scratch);
    CUDA_CHECK(cudaGetLastError());
    const dim3 group_grid(static_cast<unsigned int>(groups), static_cast<unsigned int>(batch));
    sampling_group_finalize_sample_kernel<SamplerParallelSchedule, true>
        <<<group_grid, SamplerParallelSchedule::kSamplerGroupBlock, 0, stream>>>(
            static_cast<std::int32_t*>(out.data), configs, positions, purpose, token_domain,
            partial_blocks, groups, scratch);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void sample_batch_launch(const Tensor& logits, Tensor& out, std::int32_t token_domain,
                         const SamplingConfig* configs, const Tensor& logical_positions,
                         std::int32_t purpose, DeviceSpan workspace, cudaStream_t stream) {
    const std::int32_t physical_rows = logits.ne[0];
    const std::int32_t batch         = logits.ne[1];
    const auto* positions            = static_cast<const std::int32_t*>(logical_positions.data);
    if (logits.dtype == DType::FP32) {
        sample_batch(static_cast<const float*>(logits.data), out, token_domain, physical_rows, batch, configs,
                     positions, purpose, workspace, stream);
    } else {
        sample_batch(static_cast<const __nv_bfloat16*>(logits.data), out, token_domain, physical_rows, batch,
                     configs, positions, purpose, workspace, stream);
    }
}

void sample_compact_launch(const Tensor& compact_ids, const Tensor& compact_logits,
                           const Tensor& compact_counts, Tensor& out, std::int32_t token_domain,
                           const SamplingConfig* configs, const Tensor& logical_positions,
                           std::int32_t purpose, cudaStream_t stream) {
    const std::int32_t cap   = compact_ids.ne[0];
    const std::int32_t batch = compact_ids.ne[1];
    CUDA_CHECK(pdl::launch_consumer(
        {dim3(static_cast<unsigned int>(batch)), dim3(64), 0, stream}, sample_compact_kernel,
        static_cast<const std::int32_t*>(compact_ids.data),
        static_cast<const float*>(compact_logits.data),
        static_cast<const std::int32_t*>(compact_counts.data),
        static_cast<std::int32_t*>(out.data), configs,
        static_cast<const std::int32_t*>(logical_positions.data), purpose, token_domain, cap));
    CUDA_CHECK(cudaGetLastError());
}

void increment_token_counts_launch(const Tensor& token_ids, Tensor& token_counts,
                                   cudaStream_t stream) {
    constexpr int kBlock = 256;
    const int count      = token_ids.ne[0];
    increment_token_counts_kernel<<<div_up(count, kBlock), kBlock, 0, stream>>>(
        static_cast<const std::int32_t*>(token_ids.data), count,
        static_cast<std::int32_t*>(token_counts.data));
    CUDA_CHECK(cudaGetLastError());
}

void filter_combined_branch_masks_launch(const Tensor& logits, std::int32_t token_domain,
                                         const SamplingConfig* configs, cudaStream_t stream,
                                         const Tensor& drafts, const Tensor& current_extents,
                                         const Tensor& tree_rows) {
    const std::int32_t physical_rows = logits.ne[0];
    const bool has_drafts            = drafts.data != nullptr;
    const std::int32_t cols          = has_drafts ? drafts.ne[0] + 1 : 1;
    const std::int32_t batch         = has_drafts ? drafts.ne[1] : logits.ne[1];
    const std::int32_t drafts_k      = has_drafts ? drafts.ne[0] : 0;
    const dim3 grid(static_cast<unsigned int>(cols), static_cast<unsigned int>(batch));
    const auto* drafts_ptr  = has_drafts ? static_cast<const std::int32_t*>(drafts.data) : nullptr;
    const auto* extents_ptr = current_extents.data != nullptr
                                  ? static_cast<const std::int32_t*>(current_extents.data)
                                  : nullptr;
    const auto* trees_ptr   = tree_rows.data != nullptr
                                  ? static_cast<const SpeculativeTreeRow*>(tree_rows.data)
                                  : nullptr;
    if (logits.dtype == DType::BF16) {
        filter_combined_branch_masks_kernel<__nv_bfloat16>
            <<<grid, 256, 0, stream>>>(static_cast<const __nv_bfloat16*>(logits.data), configs,
                                       drafts_ptr, extents_ptr, trees_ptr, token_domain,
                                       physical_rows, cols, drafts_k);
    } else {
        filter_combined_branch_masks_kernel<float>
            <<<grid, 256, 0, stream>>>(static_cast<const float*>(logits.data), configs, drafts_ptr,
                                       extents_ptr, trees_ptr, token_domain, physical_rows, cols,
                                       drafts_k);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix::ops::detail
