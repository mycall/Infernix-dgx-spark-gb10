#pragma once

// Implements: include/infernix/ops/sampling.h
// Match: contiguous FP32 or BF16 logits, physical stride >= token domain, and at most
// sixteen columns on the multi-block route.
// Algorithm assumptions: 256-thread/2-item partial tiles feed bounded top-20
// group merges through caller-owned workspace; unsupported finite geometries
// use the semantically identical single-block fallback.

#include "ops/kernel/sampling_device.cuh"
#include "infernix/ops/speculative_tree.h"
#include "core/pdl.cuh"

#include <type_traits>

namespace infernix::ops {

template <int SamplerBlock, int ItemsPerThread, int GroupBlock, int GroupItemsPerThread,
          int PartialsPerGroup>
struct TopKSchedule {
    static constexpr int kSamplerBlock                   = SamplerBlock;
    static constexpr int kSamplerTileItems               = 256;
    static constexpr int kSamplerItemsPerThread          = ItemsPerThread;
    static constexpr int kSamplerPartialTileItems        = SamplerBlock * ItemsPerThread;
    static constexpr int kSamplerGroupBlock              = GroupBlock;
    static constexpr int kSamplerGroupItemsPerThread     = GroupItemsPerThread;
    static constexpr int kSamplerGroupTileItems          = GroupBlock * GroupItemsPerThread;
    static constexpr int kSamplerPartialsPerGroup        = PartialsPerGroup;
    static constexpr int kSamplingTileWarps              = SamplerBlock / 32;
    static constexpr int kSamplingTileIndexBits          = 9;
    static constexpr unsigned int kSamplingTileIndexMask = (1u << kSamplingTileIndexBits) - 1u;
    static_assert(kSamplerPartialTileItems == (1 << kSamplingTileIndexBits));
    static_assert((kSamplingTileWarps & (kSamplingTileWarps - 1)) == 0);
};

using SamplerRowSchedule      = TopKSchedule<256, 2, 256, 2, 25>;
using SamplerParallelSchedule = TopKSchedule<128, 4, 128, 4, 25>;

static_assert(SamplerParallelSchedule::kSamplerPartialTileItems == kSamplerPartialTileItems);
static_assert(SamplerParallelSchedule::kSamplerGroupTileItems == kSamplerGroupTileItems);
static_assert(SamplerParallelSchedule::kSamplerPartialsPerGroup == kSamplerPartialsPerGroup);

template <class S>
using SchedulePartialSort =
    cub::BlockMergeSort<unsigned long long, S::kSamplerBlock, S::kSamplerItemsPerThread>;
template <class S>
using ScheduleGroupSort =
    cub::BlockMergeSort<unsigned long long, S::kSamplerGroupBlock, S::kSamplerGroupItemsPerThread>;
template <class S>
using ScheduleTileWarpSort = cub::WarpMergeSort<unsigned long long, S::kSamplerItemsPerThread>;
template <class S>
using ScheduleBf16TileWarpSort = cub::WarpMergeSort<unsigned int, S::kSamplerItemsPerThread>;

template <class S>
struct ScheduleTileTopKStorage {
    typename ScheduleTileWarpSort<S>::TempStorage warp[S::kSamplingTileWarps];
    unsigned long long candidates[S::kSamplingTileWarps * kSamplerCandidateCap];
};

template <class S>
struct ScheduleBf16TileTopKStorage {
    typename ScheduleBf16TileWarpSort<S>::TempStorage warp[S::kSamplingTileWarps];
    unsigned int candidates[S::kSamplingTileWarps * kSamplerCandidateCap];
};

template <class S>
union SchedulePartialTopKStorage {
    ScheduleTileTopKStorage<S> fp32;
    ScheduleBf16TileTopKStorage<S> bf16;
};

// Warp-local lists are already sorted. Merge their upper halves in parallel;
// only the global top cap entries are needed by the next stage.
template <class S, class Key>
__device__ inline Key sampling_merge_tile_lists(Key* candidates, int cap) {
    const int warp = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    Key key        = 0;
#pragma unroll
    for (int span = 1; span < S::kSamplingTileWarps; span *= 2) {
        if (warp < S::kSamplingTileWarps / (2 * span)) {
            const int first   = warp * 2 * span * kSamplerCandidateCap;
            key               = lane < cap ? candidates[first + lane] : Key{0};
            const int reverse = 31 - lane;
            const Key other =
                reverse < cap ? candidates[first + span * kSamplerCandidateCap + reverse] : Key{0};
            key = key > other ? key : other;
#pragma unroll
            for (int distance = 16; distance > 0; distance /= 2) {
                const Key opposite = __shfl_xor_sync(0xffffffffU, key, distance);
                key                = (lane & distance) == 0 ? (key > opposite ? key : opposite)
                                                            : (key < opposite ? key : opposite);
            }
            if (lane < cap) { candidates[first + lane] = key; }
        }
        __syncthreads();
    }
    return key;
}

template <class S>
__device__ inline void
sampling_schedule_store_tile_topk(unsigned long long (&keys)[S::kSamplerItemsPerThread], int cap,
                                  SamplingWorkspace workspace, int col, int partial,
                                  ScheduleTileTopKStorage<S>& storage) {
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    ScheduleTileWarpSort<S>(storage.warp[warp]).Sort(keys, SamplingKeyGreater{});
#pragma unroll
    for (int item = 0; item < S::kSamplerItemsPerThread; ++item) {
        const int rank = lane * S::kSamplerItemsPerThread + item;
        if (rank < kSamplerCandidateCap) {
            storage.candidates[warp * kSamplerCandidateCap + rank] = keys[item];
        }
    }
    __syncthreads();

    const auto key = sampling_merge_tile_lists<S>(storage.candidates, cap);
    if (warp == 0 && lane < cap) {
        workspace.partial_keys[sampling_partial_offset(workspace, col, partial, lane)] = key;
    }
}

template <class S>
__device__ inline void
sampling_schedule_store_bf16_tile_topk(unsigned int (&keys)[S::kSamplerItemsPerThread], int cap,
                                       int tile_start, SamplingWorkspace workspace, int col,
                                       int partial, ScheduleBf16TileTopKStorage<S>& storage) {
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    ScheduleBf16TileWarpSort<S>(storage.warp[warp]).Sort(keys, SamplingUintGreater{});
#pragma unroll
    for (int item = 0; item < S::kSamplerItemsPerThread; ++item) {
        const int rank = lane * S::kSamplerItemsPerThread + item;
        if (rank < kSamplerCandidateCap) {
            storage.candidates[warp * kSamplerCandidateCap + rank] = keys[item];
        }
    }
    __syncthreads();

    const auto key = sampling_merge_tile_lists<S>(storage.candidates, cap);
    if (warp == 0 && lane < cap) {
        workspace.partial_keys[sampling_partial_offset(workspace, col, partial, lane)] =
            sampling_expand_bf16_tile_sort_key(key, tile_start);
    }
}

template <class Logit, class S = SamplerRowSchedule>
__launch_bounds__(S::kSamplerBlock) __global__
    void sample_row_kernel(const Logit* logits, std::int32_t* out,
                           const SamplingConfig* configs, const std::int32_t* logical_positions,
                           std::int32_t purpose, std::int32_t token_domain,
                           std::int32_t physical_rows) {
    pdl::enter();
    const int row            = static_cast<int>(blockIdx.x);
    const std::int64_t base  = static_cast<std::int64_t>(row) * physical_rows;
    const int tid            = threadIdx.x;
    const SamplingConfig cfg = configs[row];

    __shared__ float red_val[S::kSamplerBlock];
    __shared__ int red_idx[S::kSamplerBlock];

    // With penalties disabled this remains the exact raw-logit argmax route.
    if (!(cfg.temperature > 0.0f)) {
        float bv             = -CUDART_INF_F;
        int bi               = INT_MAX;
        const bool penalties = cfg.presence_penalty != 0.0f || cfg.frequency_penalty != 0.0f;
        if (!penalties && cfg.mask.words == nullptr && cfg.forbidden_token < 0) {
            if constexpr (std::is_same_v<Logit, float>) {
                if ((reinterpret_cast<std::uintptr_t>(logits + base) % 16 == 0) && (token_domain >= 32)) {
                    const auto* row4 = reinterpret_cast<const float4*>(logits + base);
                    const int nvec   = token_domain >> 2;
                    int v4           = tid;
                    for (; v4 + 7 * blockDim.x < nvec; v4 += 8 * blockDim.x) {
                        float4 val0 = row4[v4];
                        float4 val1 = row4[v4 + blockDim.x];
                        float4 val2 = row4[v4 + 2 * blockDim.x];
                        float4 val3 = row4[v4 + 3 * blockDim.x];
                        float4 val4 = row4[v4 + 4 * blockDim.x];
                        float4 val5 = row4[v4 + 5 * blockDim.x];
                        float4 val6 = row4[v4 + 6 * blockDim.x];
                        float4 val7 = row4[v4 + 7 * blockDim.x];

                        int b0 = v4 << 2;
                        int s4 = blockDim.x << 2;

                        if (sampling_better(val0.x, b0, bv, bi)) { bv = val0.x; bi = b0; }
                        if (sampling_better(val0.y, b0 + 1, bv, bi)) { bv = val0.y; bi = b0 + 1; }
                        if (sampling_better(val0.z, b0 + 2, bv, bi)) { bv = val0.z; bi = b0 + 2; }
                        if (sampling_better(val0.w, b0 + 3, bv, bi)) { bv = val0.w; bi = b0 + 3; }

                        if (sampling_better(val1.x, b0 + s4, bv, bi)) { bv = val1.x; bi = b0 + s4; }
                        if (sampling_better(val1.y, b0 + s4 + 1, bv, bi)) { bv = val1.y; bi = b0 + s4 + 1; }
                        if (sampling_better(val1.z, b0 + s4 + 2, bv, bi)) { bv = val1.z; bi = b0 + s4 + 2; }
                        if (sampling_better(val1.w, b0 + s4 + 3, bv, bi)) { bv = val1.w; bi = b0 + s4 + 3; }

                        if (sampling_better(val2.x, b0 + 2*s4, bv, bi)) { bv = val2.x; bi = b0 + 2*s4; }
                        if (sampling_better(val2.y, b0 + 2*s4 + 1, bv, bi)) { bv = val2.y; bi = b0 + 2*s4 + 1; }
                        if (sampling_better(val2.z, b0 + 2*s4 + 2, bv, bi)) { bv = val2.z; bi = b0 + 2*s4 + 2; }
                        if (sampling_better(val2.w, b0 + 2*s4 + 3, bv, bi)) { bv = val2.w; bi = b0 + 2*s4 + 3; }

                        if (sampling_better(val3.x, b0 + 3*s4, bv, bi)) { bv = val3.x; bi = b0 + 3*s4; }
                        if (sampling_better(val3.y, b0 + 3*s4 + 1, bv, bi)) { bv = val3.y; bi = b0 + 3*s4 + 1; }
                        if (sampling_better(val3.z, b0 + 3*s4 + 2, bv, bi)) { bv = val3.z; bi = b0 + 3*s4 + 2; }
                        if (sampling_better(val3.w, b0 + 3*s4 + 3, bv, bi)) { bv = val3.w; bi = b0 + 3*s4 + 3; }

                        if (sampling_better(val4.x, b0 + 4*s4, bv, bi)) { bv = val4.x; bi = b0 + 4*s4; }
                        if (sampling_better(val4.y, b0 + 4*s4 + 1, bv, bi)) { bv = val4.y; bi = b0 + 4*s4 + 1; }
                        if (sampling_better(val4.z, b0 + 4*s4 + 2, bv, bi)) { bv = val4.z; bi = b0 + 4*s4 + 2; }
                        if (sampling_better(val4.w, b0 + 4*s4 + 3, bv, bi)) { bv = val4.w; bi = b0 + 4*s4 + 3; }

                        if (sampling_better(val5.x, b0 + 5*s4, bv, bi)) { bv = val5.x; bi = b0 + 5*s4; }
                        if (sampling_better(val5.y, b0 + 5*s4 + 1, bv, bi)) { bv = val5.y; bi = b0 + 5*s4 + 1; }
                        if (sampling_better(val5.z, b0 + 5*s4 + 2, bv, bi)) { bv = val5.z; bi = b0 + 5*s4 + 2; }
                        if (sampling_better(val5.w, b0 + 5*s4 + 3, bv, bi)) { bv = val5.w; bi = b0 + 5*s4 + 3; }

                        if (sampling_better(val6.x, b0 + 6*s4, bv, bi)) { bv = val6.x; bi = b0 + 6*s4; }
                        if (sampling_better(val6.y, b0 + 6*s4 + 1, bv, bi)) { bv = val6.y; bi = b0 + 6*s4 + 1; }
                        if (sampling_better(val6.z, b0 + 6*s4 + 2, bv, bi)) { bv = val6.z; bi = b0 + 6*s4 + 2; }
                        if (sampling_better(val6.w, b0 + 6*s4 + 3, bv, bi)) { bv = val6.w; bi = b0 + 6*s4 + 3; }

                        if (sampling_better(val7.x, b0 + 7*s4, bv, bi)) { bv = val7.x; bi = b0 + 7*s4; }
                        if (sampling_better(val7.y, b0 + 7*s4 + 1, bv, bi)) { bv = val7.y; bi = b0 + 7*s4 + 1; }
                        if (sampling_better(val7.z, b0 + 7*s4 + 2, bv, bi)) { bv = val7.z; bi = b0 + 7*s4 + 2; }
                        if (sampling_better(val7.w, b0 + 7*s4 + 3, bv, bi)) { bv = val7.w; bi = b0 + 7*s4 + 3; }
                    }
                    for (; v4 < nvec; v4 += blockDim.x) {
                        float4 val = row4[v4];
                        int base_idx = v4 << 2;
                        if (sampling_better(val.x, base_idx, bv, bi)) { bv = val.x; bi = base_idx; }
                        if (sampling_better(val.y, base_idx + 1, bv, bi)) { bv = val.y; bi = base_idx + 1; }
                        if (sampling_better(val.z, base_idx + 2, bv, bi)) { bv = val.z; bi = base_idx + 2; }
                        if (sampling_better(val.w, base_idx + 3, bv, bi)) { bv = val.w; bi = base_idx + 3; }
                    }
                    for (int v = nvec << 2; v < token_domain; ++v) {
                        const float x = static_cast<const float*>(logits)[base + v];
                        if (sampling_better(x, v, bv, bi)) {
                            bv = x;
                            bi = v;
                        }
                    }
                } else {
                    for (int v = tid; v < token_domain; v += blockDim.x) {
                        const float x = sampling_logit(logits[base + v]);
                        if (sampling_better(x, v, bv, bi)) {
                            bv = x;
                            bi = v;
                        }
                    }
                }
            } else {
                for (int v = tid; v < token_domain; v += blockDim.x) {
                    const float x = sampling_logit(logits[base + v]);
                    if (sampling_better(x, v, bv, bi)) {
                        bv = x;
                        bi = v;
                    }
                }
            }
        } else {
            for (int v = tid; v < token_domain; v += blockDim.x) {
                const float x = sampling_adjusted_logit(sampling_logit(logits[base + v]), v, cfg);
                if (sampling_better(x, v, bv, bi)) {
                    bv = x;
                    bi = v;
                }
            }
        }
        red_val[tid] = bv;
        red_idx[tid] = bi;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s &&
                sampling_better(red_val[tid + s], red_idx[tid + s], red_val[tid], red_idx[tid])) {
                red_val[tid] = red_val[tid + s];
                red_idx[tid] = red_idx[tid + s];
            }
            __syncthreads();
        }
        if (tid == 0) {
            const int picked = isfinite(red_val[0]) ? red_idx[0] : sampling_report_nonfinite(cfg);
            out[row]         = picked;
            if (cfg.token_counts != nullptr) { atomicAdd(&cfg.token_counts[picked], 1); }
        }
        return;
    }

    const int partial_blocks = div_up(token_domain, kSamplerPartialTileItems);
    const int group_count    = sampler_group_count(partial_blocks);
    // No-op when the scratch/group path owns this shape (see sample_batch_launch).
    if (sampler_multiblock_ok(token_domain, static_cast<int>(gridDim.x), partial_blocks, group_count)) {
        return;
    }

    __shared__ float cand_val[kSamplerCandidateCap];
    __shared__ int cand_idx[kSamplerCandidateCap];
    __shared__ float prob[kSamplerCandidateCap];
    __shared__ int n_support;
    __shared__ float merge_val[S::kSamplerBlock * kSamplerFastCandidates];
    __shared__ int merge_idx[S::kSamplerBlock * kSamplerFastCandidates];

    if (token_domain <= S::kSamplerTileItems) {
        sampling_build_truncated_small(logits, base, token_domain, cfg, red_val, red_idx, cand_val,
                                       cand_idx, prob, &n_support);
    } else {
        sampling_build_truncated_block_fast(logits, base, token_domain, cfg, merge_val, merge_idx,
                                            cand_val, cand_idx, prob, &n_support);
    }

    if (tid != 0) { return; }
    const int support = n_support;
    int picked        = cand_idx[support - 1];
    if (cfg.gumbel_max) {
        picked = sampling_gumbel_pick([&](int j) { return cand_idx[j]; }, [&](int j) { return prob[j]; }, support,
                                      cfg.seed, sampling_produced_position(logical_positions[row], purpose));
    } else {
        const float u = sampling_uniform(cfg.seed, logical_positions[row], purpose, 0u);
        float acc     = 0.0f;
        for (int j = 0; j < support; ++j) {
            acc += prob[j]; // prob is normalized: goal == u
            if (u < acc) {
                picked = cand_idx[j];
                break;
            }
        }
    }
    out[row] = picked;
    if (cfg.token_counts != nullptr) { atomicAdd(&cfg.token_counts[picked], 1); }
}

template <class Logit, class S = SamplerParallelSchedule, bool WarpTiles = true>
__launch_bounds__(S::kSamplerBlock) __global__
    void sampling_partial_topk_kernel(const Logit* logits, const SamplingConfig* cfg_ptr,
                                      std::int32_t token_domain, std::int32_t physical_rows,
                                      SamplingWorkspace workspace) {
    pdl::enter();
    const int col            = static_cast<int>(blockIdx.y);
    const int partial        = static_cast<int>(blockIdx.x);
    const SamplingConfig cfg = cfg_ptr[col];
    if (partial == 0 && threadIdx.x == 0) { workspace.group_done[col] = 0; }

    using SortStorage = std::conditional_t<WarpTiles, SchedulePartialTopKStorage<S>,
                                           typename SchedulePartialSort<S>::TempStorage>;
    __shared__ SortStorage sort_storage;
    __shared__ unsigned long long greedy_warp_keys[S::kSamplerBlock / 32];
    unsigned long long keys[S::kSamplerItemsPerThread];

    const bool greedy       = !(cfg.temperature > 0.0f);
    const bool penalties    = cfg.presence_penalty != 0.0f || cfg.frequency_penalty != 0.0f;
    const int cap           = greedy ? 1 : sampling_candidate_cap(cfg, token_domain);
    const std::int64_t base = static_cast<std::int64_t>(col) * physical_rows;
    const int tile_start    = partial * S::kSamplerPartialTileItems;
    if constexpr (WarpTiles && std::is_same_v<Logit, __nv_bfloat16>) {
        if (!greedy && !penalties && cfg.forbidden_token < 0) {
            unsigned int raw_keys[S::kSamplerItemsPerThread];
#pragma unroll
            for (int item = 0; item < S::kSamplerItemsPerThread; ++item) {
                const int local = item * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
                const int v     = tile_start + local;
                raw_keys[item]  = (v < token_domain &&
                                  (!cfg.mask.words || (cfg.mask.words[v / 32] & (1u << (v % 32)))))
                                      ? sampling_bf16_tile_sort_key(logits[base + v], local)
                                      : 0u;
            }
            sampling_schedule_store_bf16_tile_topk<S>(raw_keys, cap, tile_start, workspace, col,
                                                      partial, sort_storage.bf16);
            return;
        }
    }

#pragma unroll
    for (int item = 0; item < S::kSamplerItemsPerThread; ++item) {
        const int v = tile_start + item * blockDim.x + threadIdx.x;
        if (v < token_domain) {
            const float raw = sampling_logit(logits[base + v]);
            const float x = (penalties || cfg.mask.words || cfg.forbidden_token >= 0)
                                ? sampling_adjusted_logit(raw, v, cfg)
                                : raw;
            keys[item]      = sampling_sort_key(x, v);
        } else {
            keys[item] = 0ull;
        }
    }
    if (greedy) {
        unsigned long long best = keys[0];
#pragma unroll
        for (int item = 1; item < S::kSamplerItemsPerThread; ++item) {
            if (keys[item] > best) { best = keys[item]; }
        }
        best = sampling_block_max_key(best, greedy_warp_keys);
        if (threadIdx.x == 0) {
            const int off               = sampling_partial_offset(workspace, col, partial, 0);
            workspace.partial_keys[off] = best;
        }
        return;
    }
    if constexpr (WarpTiles) {
        sampling_schedule_store_tile_topk<S>(keys, cap, workspace, col, partial, sort_storage.fp32);
    } else {
        SchedulePartialSort<S>(sort_storage).Sort(keys, SamplingKeyGreater{});

#pragma unroll
        for (int item = 0; item < S::kSamplerItemsPerThread; ++item) {
            const int rank = threadIdx.x * S::kSamplerItemsPerThread + item;
            if (rank < cap) {
                const int off               = sampling_partial_offset(workspace, col, partial, rank);
                workspace.partial_keys[off] = keys[item];
            }
        }
    }
}

template <class S = SamplerParallelSchedule, bool DirectGreedy = true>
__launch_bounds__(S::kSamplerGroupBlock) __global__ void sampling_group_finalize_sample_kernel(
    std::int32_t* out, const SamplingConfig* cfg_ptr, const std::int32_t* logical_positions,
    std::int32_t purpose, std::int32_t token_domain, std::int32_t partial_blocks,
    std::int32_t group_count, SamplingWorkspace workspace) {
    pdl::enter();
    const int group          = static_cast<int>(blockIdx.x);
    const int col            = static_cast<int>(blockIdx.y);
    const int tid            = threadIdx.x;
    const SamplingConfig cfg = cfg_ptr[col];
    __shared__ typename ScheduleGroupSort<S>::TempStorage sort_storage;
    __shared__ float cand_val[kSamplerCandidateCap];
    __shared__ int cand_idx[kSamplerCandidateCap];
    __shared__ float prob[kSamplerCandidateCap];
    __shared__ int n_support;
    __shared__ int is_last;
    __shared__ unsigned long long greedy_warp_keys[S::kSamplerGroupBlock / 32];
    unsigned long long keys[S::kSamplerGroupItemsPerThread];

    const bool greedy = !(cfg.temperature > 0.0f);
    const int cap     = greedy ? 1 : sampling_candidate_cap(cfg, token_domain);
    // The preceding partial launch initializes group_done[col], so caller-owned
    // workspace does not rely on prior contents or a separate memset launch.

    const int group_begin = group * S::kSamplerPartialsPerGroup;
    int group_partials    = partial_blocks - group_begin;
    if (group_partials < 0) { group_partials = 0; }
    if (group_partials > S::kSamplerPartialsPerGroup) {
        group_partials = S::kSamplerPartialsPerGroup;
    }

    if constexpr (DirectGreedy) {
        if (greedy) {
            if (group != 0) { return; }
            unsigned long long best = 0ull;
            for (int p = tid; p < partial_blocks; p += blockDim.x) {
                const int off = sampling_partial_offset(workspace, col, p, 0);
                if (workspace.partial_keys[off] > best) { best = workspace.partial_keys[off]; }
            }
            best = sampling_block_max_key(best, greedy_warp_keys);
            if (tid == 0) {
                const int picked = isfinite(sampling_key_float(best)) ? sampling_key_index(best)
                                                                      : sampling_report_nonfinite(cfg);
                out[col]         = picked;
                if (cfg.token_counts != nullptr) { atomicAdd(&cfg.token_counts[picked], 1); }
            }
            return;
        }
    }

    if (greedy) {
        unsigned long long best = 0ull;
        for (int p = tid; p < group_partials; p += blockDim.x) {
            const int off = sampling_partial_offset(workspace, col, group_begin + p, 0);
            if (workspace.partial_keys[off] > best) { best = workspace.partial_keys[off]; }
        }
        best = sampling_block_max_key(best, greedy_warp_keys);
        if (tid == 0) {
            const int out_off = sampling_partial_offset(workspace, col, partial_blocks + group, 0);
            workspace.partial_keys[out_off] = best;
            __threadfence();
            const int done = atomicAdd(&workspace.group_done[col], 1) + 1;
            is_last        = (done == group_count) ? 1 : 0;
        }
        __syncthreads();
        if (!is_last) { return; }

        best = 0ull;
        for (int p = tid; p < group_count; p += blockDim.x) {
            const int off = sampling_partial_offset(workspace, col, partial_blocks + p, 0);
            if (workspace.partial_keys[off] > best) { best = workspace.partial_keys[off]; }
        }
        best = sampling_block_max_key(best, greedy_warp_keys);
        if (tid == 0) {
            const int picked = isfinite(sampling_key_float(best)) ? sampling_key_index(best)
                                                                  : sampling_report_nonfinite(cfg);
            out[col]         = picked;
            if (cfg.token_counts != nullptr) { atomicAdd(&cfg.token_counts[picked], 1); }
            workspace.group_done[col] = 0;
        }
        return;
    }

    const int group_n = group_partials * cap;
#pragma unroll
    for (int item = 0; item < S::kSamplerGroupItemsPerThread; ++item) {
        const int p = item * blockDim.x + tid;
        if (p < group_n) {
            const int partial = group_begin + p / cap;
            const int j       = p - (p / cap) * cap;
            const int off     = sampling_partial_offset(workspace, col, partial, j);
            keys[item]        = workspace.partial_keys[off];
        } else {
            keys[item] = 0ull;
        }
    }
    ScheduleGroupSort<S>(sort_storage).Sort(keys, SamplingKeyGreater{});

#pragma unroll
    for (int item = 0; item < S::kSamplerGroupItemsPerThread; ++item) {
        const int rank = tid * S::kSamplerGroupItemsPerThread + item;
        if (rank < cap) {
            const int out_off =
                sampling_partial_offset(workspace, col, partial_blocks + group, rank);
            workspace.partial_keys[out_off] = keys[item];
        }
    }
    __syncthreads();

    if (tid == 0) {
        __threadfence();
        const int done = atomicAdd(&workspace.group_done[col], 1) + 1;
        is_last        = (done == group_count) ? 1 : 0;
    }
    __syncthreads();
    if (!is_last) { return; }

    const int final_n = group_count * cap;
#pragma unroll
    for (int item = 0; item < S::kSamplerGroupItemsPerThread; ++item) {
        const int p = item * blockDim.x + tid;
        if (p < final_n) {
            const int partial = partial_blocks + p / cap;
            const int j       = p - (p / cap) * cap;
            const int off     = sampling_partial_offset(workspace, col, partial, j);
            keys[item]        = workspace.partial_keys[off];
        } else {
            keys[item] = 0ull;
        }
    }
    ScheduleGroupSort<S>(sort_storage).Sort(keys, SamplingKeyGreater{});

#pragma unroll
    for (int item = 0; item < S::kSamplerGroupItemsPerThread; ++item) {
        const int rank = tid * S::kSamplerGroupItemsPerThread + item;
        if (rank < cap) {
            cand_val[rank] = sampling_key_float(keys[item]);
            cand_idx[rank] = sampling_key_index(keys[item]);
        }
    }
    __syncthreads();

    sampling_normalize_support(cfg, cand_val, cand_idx, prob, &n_support, cap);
    if (tid == 0) {
        const int support = n_support;
        int picked        = cand_idx[support - 1];
        if (cfg.gumbel_max) {
            picked = sampling_gumbel_pick([&](int j) { return cand_idx[j]; }, [&](int j) { return prob[j]; },
                                          support, cfg.seed,
                                          sampling_produced_position(logical_positions[col], purpose));
        } else {
            const float u = sampling_uniform(cfg.seed, logical_positions[col], purpose, 0u);
            float acc     = 0.0f;
            for (int j = 0; j < support; ++j) {
                acc += prob[j];
                if (u < acc) {
                    picked = cand_idx[j];
                    break;
                }
            }
        }
        out[col] = picked;
        if (cfg.token_counts != nullptr) { atomicAdd(&cfg.token_counts[picked], 1); }
        workspace.group_done[col] = 0;
    }
}

__launch_bounds__(64) __global__
    void sample_compact_kernel(const std::int32_t* compact_ids, const float* compact_logits,
                               const std::int32_t* compact_counts, std::int32_t* out,
                               const SamplingConfig* configs, const std::int32_t* logical_positions,
                               std::int32_t purpose, std::int32_t token_domain, std::int32_t cap) {
    pdl::enter();
    const int row            = static_cast<int>(blockIdx.x);
    const int tid            = static_cast<int>(threadIdx.x);
    const SamplingConfig cfg = configs[row];
    const int n              = min(cap, max(0, compact_counts[row]));
    const std::size_t base   = static_cast<std::size_t>(row) * cap;

    __shared__ unsigned long long sort_keys[64];
    __shared__ unsigned long long warp_keys[2];
    __shared__ float cand_val[kSamplerCandidateCap];
    __shared__ int cand_idx[kSamplerCandidateCap];
    __shared__ float prob[kSamplerCandidateCap];
    __shared__ int n_support;

    if (!(cfg.temperature > 0.0f)) {
        unsigned long long key = 0ull;
        if (tid < n) {
            const int id = compact_ids[base + tid];
            if (id >= 0 && id < token_domain) {
                const float x = sampling_adjusted_logit(compact_logits[base + tid], id, cfg);
                key           = sampling_sort_key(x, id);
            }
        }
        const unsigned long long best = sampling_compact_max_key(key, warp_keys);
        if (tid == 0) {
            const bool finite = (best != 0ull) && isfinite(sampling_key_float(best));
            const int picked  = finite ? sampling_key_index(best) : sampling_report_nonfinite(cfg);
            out[row]          = picked;
            if (cfg.token_counts != nullptr) { atomicAdd(&cfg.token_counts[picked], 1); }
        }
        return;
    }

    sampling_compact_build_support(compact_ids + base, compact_logits + base, n, token_domain, cfg, sort_keys,
                                   cand_val, cand_idx, prob, &n_support);
    if (tid != 0) { return; }
    const int support = n_support;
    int picked        = cand_idx[support - 1];
    if (cfg.gumbel_max) {
        picked = sampling_gumbel_pick([&](int j) { return cand_idx[j]; }, [&](int j) { return prob[j]; }, support,
                                      cfg.seed, sampling_produced_position(logical_positions[row], purpose));
    } else {
        const float u = sampling_uniform(cfg.seed, logical_positions[row], purpose, 0u);
        float acc     = 0.0f;
        for (int j = 0; j < support; ++j) {
            acc += prob[j];
            if (u < acc) {
                picked = cand_idx[j];
                break;
            }
        }
    }
    out[row] = picked;
    if (cfg.token_counts != nullptr) { atomicAdd(&cfg.token_counts[picked], 1); }
}

template <class Logit>
__global__ void __launch_bounds__(256)
    filter_combined_branch_masks_kernel(const Logit* logits, const SamplingConfig* configs,
                                        const std::int32_t* drafts, const std::int32_t* extents,
                                        const SpeculativeTreeRow* tree_rows,
                                        std::int32_t token_domain, std::int32_t physical_rows,
                                        std::int32_t cols, std::int32_t drafts_k) {
    const int col = static_cast<int>(blockIdx.x);
    const int row = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);

    const SamplingConfig cfg        = configs[row];
    const std::int32_t branch_token = cfg.branch_tool_token;
    const std::int32_t stride       = cfg.mask.stride;
    std::uint32_t* branch_words     = cfg.branch_mask_words;
    const int nwords                = (token_domain + 31) >> 5;
    if (branch_token < 0 || branch_token >= token_domain || branch_words == nullptr ||
        stride < nwords) {
        return;
    }

    __shared__ int s_skip;
    __shared__ int s_tree_prefix[kSpeculativeTreeMaxPathLength];
    __shared__ int s_tree_prefix_len;
    const std::int32_t* row_drafts =
        drafts != nullptr ? drafts + static_cast<std::size_t>(row) * drafts_k : nullptr;
    const bool is_tree = tree_rows != nullptr && tree_rows[row].tree != 0;

    if (tid == 0) {
        int skip          = 0;
        s_tree_prefix_len = 0;
        if (is_tree) {
            const SpeculativeTreeRow& tr = tree_rows[row];
            if (col >= tr.nodes || row_drafts == nullptr) {
                skip = 1;
            } else {
                int len = 0;
                int cur = col;
                while (cur > 0 && len < kSpeculativeTreeMaxPathLength) {
                    const std::int32_t tok = row_drafts[cur - 1];
                    if (tok == branch_token) {
                        skip = 1;
                        break;
                    }
                    s_tree_prefix[len++] = tok;
                    cur                  = static_cast<int>(tr.parent[cur]);
                }
                s_tree_prefix_len = len;
            }
        } else {
            if (extents != nullptr) {
                int extent = extents[row];
                if (extent < 0) { extent = 0; }
                if (extent > cols - 1) { extent = cols - 1; }
                if (col > extent) { skip = 1; }
            }
            if (!skip && col > 0 && row_drafts != nullptr) {
                for (int j = 0; j < col; ++j) {
                    if (row_drafts[j] == branch_token) {
                        skip = 1;
                        break;
                    }
                }
            }
        }
        s_skip = skip;
    }
    __syncthreads();
    if (s_skip != 0) { return; }

    std::uint32_t* col_words        = branch_words + static_cast<std::size_t>(col) * stride;
    const int branch_word_idx       = branch_token >> 5;
    const std::uint32_t branch_bit  = 1U << (branch_token & 31);
    const std::uint32_t branch_word = col_words[branch_word_idx];
    if ((branch_word & branch_bit) == 0U) { return; }

    const int last_bits           = token_domain & 31;
    const std::uint32_t tail_mask = (last_bits == 0) ? 0xFFFFFFFFU : ((1U << last_bits) - 1U);

    int local_other = 0;
    for (int w = tid; w < nwords; w += blockDim.x) {
        std::uint32_t bits = col_words[w];
        if (w == nwords - 1) { bits &= tail_mask; }
        if (w == branch_word_idx) { bits &= ~branch_bit; }
        if (bits != 0U) {
            local_other = 1;
            break;
        }
    }
    if (__syncthreads_or(local_other) == 0) { return; }

    SamplingConfig raw_cfg        = cfg;
    raw_cfg.mask                  = {};
    raw_cfg.forbidden_token       = -1;
    raw_cfg.forbidden_until_token = -1;

    const std::int32_t* overlay = is_tree ? s_tree_prefix : row_drafts;
    const int overlay_len       = is_tree ? s_tree_prefix_len : (row_drafts != nullptr ? col : 0);
    const std::int64_t base     = (static_cast<std::int64_t>(row) * cols + col) * physical_rows;

    float best_val      = -CUDART_INF_F;
    int best_idx        = INT_MAX;
    int local_nonfinite = 0;

    for (int v = tid; v < token_domain; v += blockDim.x) {
        const float raw = sampling_logit(logits[base + v]);
        if (!isfinite(raw)) {
            local_nonfinite = 1;
            continue;
        }
        const float x = sampling_adjusted_logit(raw, v, raw_cfg, overlay, overlay_len);
        if (!isfinite(x)) {
            local_nonfinite = 1;
            continue;
        }
        if (x > best_val || (x == best_val && v < best_idx)) {
            best_val = x;
            best_idx = v;
        }
    }
    if (__syncthreads_or(local_nonfinite) != 0) { return; }

    const unsigned warp_mask = 0xFFFFFFFFU;
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other_val = __shfl_down_sync(warp_mask, best_val, offset);
        const int other_idx   = __shfl_down_sync(warp_mask, best_idx, offset);
        if (other_idx < token_domain &&
            (other_val > best_val || (other_val == best_val && other_idx < best_idx))) {
            best_val = other_val;
            best_idx = other_idx;
        }
    }

    __shared__ float s_warp_val[8];
    __shared__ int s_warp_idx[8];
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if (lane == 0) {
        s_warp_val[warp] = best_val;
        s_warp_idx[warp] = best_idx;
    }
    __syncthreads();

    if (warp == 0) {
        float wval = lane < 8 ? s_warp_val[lane] : -CUDART_INF_F;
        int widx   = lane < 8 ? s_warp_idx[lane] : INT_MAX;
        for (int offset = 4; offset > 0; offset >>= 1) {
            const float other_val = __shfl_down_sync(warp_mask, wval, offset);
            const int other_idx   = __shfl_down_sync(warp_mask, widx, offset);
            if (other_idx < token_domain &&
                (other_val > wval || (other_val == wval && other_idx < widx))) {
                wval = other_val;
                widx = other_idx;
            }
        }
        if (lane == 0 && widx >= 0 && widx < token_domain) {
            const bool best_allowed = (col_words[widx >> 5] & (1U << (widx & 31))) != 0U;
            if (!best_allowed) { col_words[branch_word_idx] = branch_word & ~branch_bit; }
        }
    }
}

} // namespace infernix::ops
