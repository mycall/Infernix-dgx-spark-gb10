// infernix::ops - sample wrapper: public api validation and dispatch.
#include "infernix/ops/sampling.h"
#include "infernix/ops/speculative_tree.h"

#include "ops/common/sampling_workspace.h"
#include "ops/launcher/sampling.h" // detail::sample_batch_launch

#include <algorithm>
#include <stdexcept>

namespace infernix::ops {

std::size_t sampling_workspace_capacity_bytes(std::int32_t token_domain, std::int32_t min_lanes,
                                              std::int32_t max_lanes) {
    if (token_domain <= 0 || min_lanes <= 0 || max_lanes < min_lanes) {
        throw std::invalid_argument("sampling workspace: invalid profile or lane interval");
    }
    if (token_domain <= kSamplerTileItems || min_lanes > kSamplerMaxColumns) { return 0; }
    return detail::sampling_workspace_exact_bytes(token_domain,
                                                  std::min(max_lanes, kSamplerMaxColumns));
}

void sample(const Tensor& logits, Tensor& out, std::int32_t token_domain,
            const SamplingConfig* configs, const Tensor& logical_positions, std::int32_t purpose,
            WorkspaceArena& workspace, cudaStream_t stream) {
    if (logits.dtype != DType::BF16 && logits.dtype != DType::FP32) {
        throw std::invalid_argument("sample: logits must be BF16 or FP32");
    }
    if (out.dtype != DType::I32) { throw std::invalid_argument("sample: out must be I32"); }
    if (logits.ne[2] != 1 || logits.ne[3] != 1) {
        throw std::invalid_argument("sample: logits must be rank-2 [physical_rows,B]");
    }
    if (out.ne[1] != 1 || out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("sample: out must be rank-1 [B]");
    }
    if (logits.ne[0] <= 0) {
        throw std::invalid_argument("sample: physical rows must be positive");
    }
    if (token_domain <= 0 || token_domain > logits.ne[0]) {
        throw std::invalid_argument("sample: token_domain must be in [1, logits.ne[0]]");
    }
    if (out.ne[0] != logits.ne[1]) { throw std::invalid_argument("sample: out shape must be [B]"); }
    if (logical_positions.dtype != DType::I32 || logical_positions.ne[0] != logits.ne[1] ||
        logical_positions.ne[1] != 1 || logical_positions.ne[2] != 1 ||
        logical_positions.ne[3] != 1) {
        throw std::invalid_argument("sample: logical_positions must be I32 [logits.ne[1]]");
    }
    if (logits.ne[1] <= 0) { throw std::invalid_argument("sample: B must be positive"); }
    if (!logits.is_contiguous() || !out.is_contiguous() || !logical_positions.is_contiguous()) {
        throw std::invalid_argument("sample: logits/out/logical_positions must be contiguous");
    }
    if (logits.data == nullptr || out.data == nullptr || logical_positions.data == nullptr) {
        throw std::invalid_argument("sample: tensor data must be non-null");
    }
    if (configs == nullptr) { throw std::invalid_argument("sample: configs must be non-null"); }

    auto scratch_scope = workspace.scope();
    const std::size_t bytes =
        sampling_workspace_capacity_bytes(token_domain, logits.ne[1], logits.ne[1]);
    const DeviceSpan scratch = bytes == 0 ? DeviceSpan{} : workspace.alloc_bytes(bytes);
    detail::sample_batch_launch(logits, out, token_domain, configs, logical_positions, purpose,
                                scratch, stream);
}

void sample_compact(const Tensor& compact_ids, const Tensor& compact_logits,
                    const Tensor& compact_counts, Tensor& out, std::int32_t token_domain,
                    const SamplingConfig* configs, const Tensor& logical_positions,
                    std::int32_t purpose, cudaStream_t stream) {
    const std::int32_t cap   = compact_ids.ne[0];
    const std::int32_t batch = compact_ids.ne[1];
    if (cap < 1 || cap > 64 || batch < 1 || compact_ids.dtype != DType::I32 || compact_ids.ne[2] != 1 ||
        compact_ids.ne[3] != 1 || !compact_ids.is_contiguous() || compact_ids.data == nullptr) {
        throw std::invalid_argument("sample_compact: compact_ids must be contiguous I32 [cap, B] with cap in [1,64]");
    }
    if (compact_logits.dtype != DType::FP32 || compact_logits.ne[0] != cap || compact_logits.ne[1] != batch ||
        compact_logits.ne[2] != 1 || compact_logits.ne[3] != 1 || !compact_logits.is_contiguous() ||
        compact_logits.data == nullptr) {
        throw std::invalid_argument("sample_compact: compact_logits must be contiguous FP32 [cap, B]");
    }
    if (compact_counts.dtype != DType::I32 || compact_counts.ne[0] != batch || compact_counts.ne[1] != 1 ||
        compact_counts.ne[2] != 1 || compact_counts.ne[3] != 1 || !compact_counts.is_contiguous() ||
        compact_counts.data == nullptr) {
        throw std::invalid_argument("sample_compact: compact_counts must be contiguous I32 [B]");
    }
    if (out.dtype != DType::I32 || out.ne[0] != batch || out.ne[1] != 1 || out.ne[2] != 1 || out.ne[3] != 1 ||
        !out.is_contiguous() || out.data == nullptr) {
        throw std::invalid_argument("sample_compact: out must be contiguous I32 [B]");
    }
    if (logical_positions.dtype != DType::I32 || logical_positions.ne[0] != batch || logical_positions.ne[1] != 1 ||
        logical_positions.ne[2] != 1 || logical_positions.ne[3] != 1 || !logical_positions.is_contiguous() ||
        logical_positions.data == nullptr) {
        throw std::invalid_argument("sample_compact: logical_positions must be contiguous I32 [B]");
    }
    if (token_domain <= 0 || configs == nullptr) {
        throw std::invalid_argument("sample_compact: token_domain must be positive and configs non-null");
    }
    detail::sample_compact_launch(compact_ids, compact_logits, compact_counts, out, token_domain, configs,
                                  logical_positions, purpose, stream);
}

void increment_token_counts(const Tensor& token_ids, Tensor& token_counts, cudaStream_t stream) {
    if (token_ids.dtype != DType::I32 || token_ids.ne[0] <= 0 || token_ids.ne[1] != 1 ||
        token_ids.ne[2] != 1 || token_ids.ne[3] != 1 || !token_ids.is_contiguous() ||
        token_ids.data == nullptr) {
        throw std::invalid_argument(
            "increment_token_counts: token_ids must be a contiguous non-empty I32 vector");
    }
    if (token_counts.dtype != DType::I32 || token_counts.ne[0] <= 0 || token_counts.ne[1] != 1 ||
        token_counts.ne[2] != 1 || token_counts.ne[3] != 1 || !token_counts.is_contiguous() ||
        token_counts.data == nullptr) {
        throw std::invalid_argument(
            "increment_token_counts: token_counts must be a contiguous non-empty I32 vector");
    }
    detail::increment_token_counts_launch(token_ids, token_counts, stream);
}

void filter_combined_branch_masks(const Tensor& logits, std::int32_t token_domain,
                                  const SamplingConfig* configs, cudaStream_t stream,
                                  const Tensor& drafts, const Tensor& current_extents,
                                  const Tensor& tree_rows) {
    if (logits.dtype != DType::BF16 && logits.dtype != DType::FP32) {
        throw std::invalid_argument("filter_combined_branch_masks: logits must be BF16 or FP32");
    }
    if (!logits.is_contiguous() || logits.data == nullptr) {
        throw std::invalid_argument(
            "filter_combined_branch_masks: logits must be contiguous and non-null");
    }
    const bool has_drafts = drafts.data != nullptr;
    if (has_drafts) {
        if (drafts.dtype != DType::I32 || !drafts.is_contiguous() || drafts.ne[0] <= 0 ||
            drafts.ne[1] <= 0 || drafts.ne[2] != 1 || drafts.ne[3] != 1) {
            throw std::invalid_argument(
                "filter_combined_branch_masks: drafts must be contiguous I32 [K,B]");
        }
        const std::int32_t cols  = drafts.ne[0] + 1;
        const std::int32_t batch = drafts.ne[1];
        if (logits.ne[0] <= 0 || logits.ne[1] != cols || logits.ne[2] != batch ||
            logits.ne[3] != 1) {
            throw std::invalid_argument(
                "filter_combined_branch_masks: logits must be [physical_rows,K+1,B]");
        }
        if (current_extents.dtype != DType::I32 || !current_extents.is_contiguous() ||
            current_extents.data == nullptr || current_extents.ne[0] != batch ||
            current_extents.ne[1] != 1 || current_extents.ne[2] != 1 ||
            current_extents.ne[3] != 1) {
            throw std::invalid_argument(
                "filter_combined_branch_masks: current_extents must be contiguous I32 [B]");
        }
        if (tree_rows.data != nullptr &&
            (tree_rows.dtype != DType::I32 || !tree_rows.is_contiguous() ||
             tree_rows.ne[0] != kSpeculativeTreeRowWords || tree_rows.ne[1] != batch ||
             tree_rows.ne[2] != 1 || tree_rows.ne[3] != 1)) {
            throw std::invalid_argument(
                "filter_combined_branch_masks: tree_rows must be contiguous I32 [words,B]");
        }
    } else {
        if (current_extents.data != nullptr || tree_rows.data != nullptr) {
            throw std::invalid_argument(
                "filter_combined_branch_masks: extents and tree_rows require drafts");
        }
        if (logits.ne[0] <= 0 || logits.ne[1] <= 0 || logits.ne[2] != 1 || logits.ne[3] != 1) {
            throw std::invalid_argument(
                "filter_combined_branch_masks: logits must be [physical_rows,B]");
        }
    }
    if (token_domain <= 0 || token_domain > logits.ne[0]) {
        throw std::invalid_argument(
            "filter_combined_branch_masks: token_domain exceeds logits rows");
    }
    if (configs == nullptr) {
        throw std::invalid_argument("filter_combined_branch_masks: configs must be non-null");
    }
    detail::filter_combined_branch_masks_launch(logits, token_domain, configs, stream, drafts,
                                                current_extents, tree_rows);
}

} // namespace infernix::ops
