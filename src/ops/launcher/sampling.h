#pragma once

// infernix::ops::detail - private launch prototype for sample.

#include "core/tensor.h"
#include "infernix/ops/sampling.h"

#include <cstdint>

#include <cuda_runtime.h>

namespace infernix::ops::detail {

void sample_batch_launch(const Tensor& logits, Tensor& out, std::int32_t token_domain,
                         const SamplingConfig* configs, const Tensor& logical_positions,
                         std::int32_t purpose, DeviceSpan workspace, cudaStream_t stream);

void sample_compact_launch(const Tensor& compact_ids, const Tensor& compact_logits,
                           const Tensor& compact_counts, Tensor& out, std::int32_t token_domain,
                           const SamplingConfig* configs, const Tensor& logical_positions,
                           std::int32_t purpose, cudaStream_t stream);

void increment_token_counts_launch(const Tensor& token_ids, Tensor& token_counts,
                                   cudaStream_t stream);

void filter_combined_branch_masks_launch(const Tensor& logits, std::int32_t token_domain,
                                         const SamplingConfig* configs, cudaStream_t stream,
                                         const Tensor& drafts, const Tensor& current_extents,
                                         const Tensor& tree_rows);

[[nodiscard]] std::size_t sampling_workspace_exact_bytes(std::int32_t token_domain,
                                                         std::int32_t columns);

} // namespace infernix::ops::detail
