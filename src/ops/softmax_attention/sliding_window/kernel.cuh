#pragma once

#include "ops/softmax_attention/common/context_query.cuh"
#include "core/pdl.cuh"

namespace infernix::ops {

struct SlidingWindowAttentionPolicy {
    static constexpr bool PageMapped = false;

    const std::int32_t* positions;
    int padded_context;
    int window_mask;

    __device__ __forceinline__ int context_count(int value) const { return min(value, window_mask); }

    __device__ __forceinline__ int query_position(int token) const { return positions[token]; }

    __device__ __forceinline__ bool allow_context(int query, int key) const {
        return key >= query - (window_mask);
    }

    __device__ __forceinline__ std::int64_t context_tile(int kv_head, int, int) const {
        return static_cast<std::int64_t>(kContextQueryHeadDim) * padded_context * kv_head;
    }

    __device__ __forceinline__ std::int64_t context_index(std::int64_t tile, int d, int position,
                                                          int) const {
        const int slot = position & (window_mask);
        return tile + d + static_cast<std::int64_t>(kContextQueryHeadDim) * slot;
    }

    template <int Tokens, int KeyBlock>
    __device__ __forceinline__ void prime(int, int, int, int) {}
};

template <int Tokens, int WarpsPerCta, int KeyBlock, bool DirectOutput>
__launch_bounds__(WarpsPerCta * 32, kContextQueryMinBlocksPerSm<WarpsPerCta, KeyBlock>) __global__
    void sliding_window_attention_split_partial_kernel(
        const __nv_bfloat16* q, const __nv_bfloat16* query_k,
        const __nv_bfloat16* query_v, const std::int32_t* __restrict__ positions,
        const std::int32_t* __restrict__ valid_columns, const std::int32_t* __restrict__ lanes,
        const __nv_bfloat16* context_k, const __half* context_v,
        int padded_context, int window_mask, int max_context, int split_capacity, float scale,
        float* __restrict__ partial_acc, float* __restrict__ partial_m,
        float* __restrict__ partial_l, __nv_bfloat16* __restrict__ out) {
    pdl::enter_streaming();
    const int batch = static_cast<int>(blockIdx.z);
    const std::int64_t lane_elements =
        static_cast<std::int64_t>(kContextQueryHeadDim) * padded_context * kContextQueryKVHeads;
    context_k += lane_elements * lanes[batch];
    context_v += lane_elements * lanes[batch];
    const std::int32_t* batch_positions = positions + static_cast<std::int64_t>(Tokens) * batch;
    SlidingWindowAttentionPolicy policy{
        .positions      = batch_positions,
        .padded_context = padded_context,
        .window_mask = window_mask,
    };
    context_query_split_partial_body<SlidingWindowAttentionPolicy, Tokens, WarpsPerCta,
                                     KeyBlock, DirectOutput>(
        q, query_k, query_v, valid_columns, context_k, context_v, policy,
        DirectOutput && valid_columns[batch] == 0 ? 0 : batch_positions[0], max_context,
        split_capacity, scale, partial_acc, partial_m, partial_l, out);
}

template <int Tokens, int KeyBlock, int WarpsPerBlock>
__launch_bounds__(WarpsPerBlock * 32, 2) __global__ void sliding_window_attention_reduce_kernel(
    const float* __restrict__ partial_acc, const float* __restrict__ partial_m,
    const float* __restrict__ partial_l, const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ valid_columns, int window_mask, int max_context,
    int split_capacity, __nv_bfloat16* __restrict__ out) {
    pdl::enter();
    const int length =
        positions[static_cast<std::int64_t>(Tokens) * static_cast<int>(blockIdx.z)];
    context_query_reduce_body<float, Tokens, KeyBlock, WarpsPerBlock, 32>(
        partial_acc, partial_m, partial_l, length, min(length, window_mask), valid_columns,
        max_context, split_capacity, out);
}

} // namespace infernix::ops
