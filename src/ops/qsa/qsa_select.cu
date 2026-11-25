// QSA block selection: multi-CTA scoring and exact radix selection (qsa_select.h,
// docs/maintainer/qwen3_8-flash-next-design.md §19.3.6 item 2).

#include "ops/qsa/qsa_select.h"

#include "ops/qsa/page_spaces.cuh"

#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "core/pdl.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/score_id_order.cuh"
#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace infernix::ops::detail {
namespace {

using bf16   = __nv_bfloat16;

constexpr int kIndexDim       = 128;
constexpr unsigned kFullMask  = 0xFFFFFFFFU;
constexpr int kLanePartials   = kIndexDim / 32; // products each lane accumulates per dot

// Scoring: a warp step reads kScoreStepBlocks pooled keys and, per column, reduce-scatters the
// kScoreStepBlocks x kScoreHeadSlots = 32 lane partials of its dots.
constexpr int kScoreThreads     = 256;
constexpr int kScoreWarps       = kScoreThreads / 32;
constexpr int kScoreCtasPerSm   = 2;
constexpr int kScoreStepBlocks  = 8;
constexpr int kScoreHeadSlots   = 4;  // index heads (1..4; unused slots hold zero queries)
constexpr int kScoreTileColumns = 16; // columns of one row scored from one key read
static_assert(kScoreStepBlocks * kScoreHeadSlots == 32);

// Selection.
constexpr int kSelectThreads     = 512;
constexpr int kSelectWarps       = kSelectThreads / 32;
constexpr int kScoreBins         = 2048; // ordered score bits [30:20]: exponent and 3 mantissa bits
constexpr int kBinShift          = 20;
constexpr int kCandidateCapacity = 4096;
constexpr std::size_t kCandidateBytes = sizeof(std::uint64_t) * kCandidateCapacity;
constexpr int kMaxSlices         = 8;    // selection CTAs per column
constexpr int kSliceBlocks       = 4096; // selection: blocks per CTA at the largest context
constexpr int kWalkBatch         = 4;    // 128-score chunks each warp keeps in flight
static_assert(kScoreBins == 4 * kSelectThreads, "find_score_bin gives each thread four bins");

struct ScoreTiles {
    std::int32_t count = 0;
    std::uint8_t first[kQsaSelectGroupColumns];   // first column, counted from the group's first
    std::uint8_t columns[kQsaSelectGroupColumns]; // 1..kScoreTileColumns, all of one row
};
static_assert(kQsaSelectGroupColumns <= 256, "ScoreTiles stores group-relative columns in bytes");

// ------------------------------------------------------------------------------------- scoring

// Same addressing as qsa.cu's pool and attention kernels: the pooled key of block b fills the R
// token slots of its block, slot_width = Di / R elements each.
// A page id's space is resolved (design §19.3.11): the pool's pooled plane or the spaces' array.
__device__ __forceinline__ const bf16* pooled_element(const bf16* pool, const QsaPageSpaces& spaces,
                                                      const std::int32_t* table, int token, int slot_width, int slot) {
    int page;
    const bf16* plane = qsa_pooled_plane(spaces, pool, table[token >> kPagedKVPageShift], page);
    return plane + static_cast<std::int64_t>(slot_width) * kPagedKVPageSize * page +
           static_cast<std::int64_t>(slot_width) * (token & kPagedKVPageMask) + slot;
}

// One stage of the reduce-scatter: lanes whose `Half` bit is clear keep values [0, Half), the
// others [Half, 2 Half); each kept value adds its partner lane's copy (own + partner, as
// warp_sum's `v += __shfl_xor_sync(v, Half)`).
template <int Half>
__device__ __forceinline__ void reduce_scatter_stage(float (&v)[32], int lane) {
    const bool upper = (lane & Half) != 0;
#pragma unroll
    for (int i = 0; i < Half; ++i) {
        const float keep = upper ? v[i + Half] : v[i];
        const float send = upper ? v[i] : v[i + Half];
        v[i]             = keep + __shfl_xor_sync(kFullMask, send, Half);
    }
}

// Lane l returns the 32-lane sum of its value l, bitwise equal to warp_sum's result for that value
// (31 shuffles for 32 sums instead of 160). After the stages at offsets 16..o, each value lane l
// keeps is the sum over the lanes l ^ m, m a combination of those offsets, formed by exactly the
// additions the xor butterfly performs at lane l; the butterfly ends with that sum in every lane.
__device__ __forceinline__ float reduce_scatter_32(float (&v)[32], int lane) {
    reduce_scatter_stage<16>(v, lane);
    reduce_scatter_stage<8>(v, lane);
    reduce_scatter_stage<4>(v, lane);
    reduce_scatter_stage<2>(v, lane);
    reduce_scatter_stage<1>(v, lane);
    return v[0];
}

// Grid (CTAs per tile, tiles). A tile is up to 16 consecutive columns of one row; its CTAs split
// the row's blocks in steps of 8, one step per warp at a time. Each pooled key is read once per
// tile and scored for every column of the tile that sees it. With `clear`, the CTAs first zero
// `clear_words` words of selection state (histograms and arrival counters).
__device__ __forceinline__ int qsa_logical_position(const std::int32_t* __restrict__ positions,
                                                    const SpeculativeTreeRow* __restrict__ trees,
                                                    int width, int t) {
    if (trees == nullptr) { return positions[t]; }
    const int seq                  = t / width;
    const SpeculativeTreeRow& tree = trees[seq];
    if (tree.tree == 0) { return positions[t]; }
    const int local = t % width;
    return local < tree.nodes ? (positions[seq * width] + static_cast<int>(tree.depth[local]))
                              : positions[seq * width];
}

__global__ void __launch_bounds__(kScoreThreads, kScoreCtasPerSm)
    qsa_score_kernel(const bf16* index_q, int index_heads, const bf16* pooled,
                     const std::int32_t* __restrict__ tables, int table_stride,
                     const std::int32_t* __restrict__ table_rows, const std::int32_t* __restrict__ positions,
                     const SpeculativeTreeRow* __restrict__ trees, int width, int ratio, int top_blocks,
                     int group_begin, const ScoreTiles tiles, float* __restrict__ scores, int score_stride,
                     std::uint32_t* __restrict__ clear, int clear_words, QsaPageSpaces spaces) {
    pdl::enter_streaming();
    if (clear != nullptr) {
        const int total_threads = static_cast<int>(gridDim.x * gridDim.y) * kScoreThreads;
        const int global_tid =
            static_cast<int>(blockIdx.y * gridDim.x + blockIdx.x) * kScoreThreads + static_cast<int>(threadIdx.x);
        for (int i = global_tid; i < clear_words; i += total_threads) { clear[i] = 0; }
    }
    __shared__ uint2 qs[kScoreTileColumns][kScoreHeadSlots][32]; // lane l: q[h][l + 32 j], j = 0..3 as 4 BF16s
    __shared__ int column_blocks[kScoreTileColumns];
    const int tile = static_cast<int>(blockIdx.y);
    const int c0 = tiles.first[tile], nc = tiles.columns[tile];
    const int t0 = group_begin + c0;
    int tile_blocks = 0;
    for (int c = 0; c < nc; ++c) {
        tile_blocks = max(tile_blocks, (qsa_logical_position(positions, trees, width, t0 + c) + 1) / ratio);
    }
    const int steps      = (tile_blocks + kScoreStepBlocks - 1) / kScoreStepBlocks;
    const int first_step = static_cast<int>(blockIdx.x) * kScoreWarps;
    if (tile_blocks <= top_blocks || first_step >= steps) {
        pdl::trigger_dependents();
        return; // every column of the tile is dense, or this CTA has no step
    }
    if (static_cast<int>(threadIdx.x) < nc) {
        column_blocks[threadIdx.x] =
            (qsa_logical_position(positions, trees, width, t0 + static_cast<int>(threadIdx.x)) + 1) / ratio;
    }
    for (int i = static_cast<int>(threadIdx.x); i < nc * kScoreHeadSlots * kIndexDim; i += kScoreThreads) {
        const int c = i / (kScoreHeadSlots * kIndexDim), h = (i / kIndexDim) % kScoreHeadSlots, d = i % kIndexDim;
        const std::uint16_t value =
            h < index_heads
                ? __bfloat16_as_ushort(index_q[(static_cast<std::size_t>(t0 + c) * index_heads + h) * kIndexDim + d])
                : 0U;
        reinterpret_cast<std::uint16_t*>(&qs[c][h][d % 32])[d / 32] = value;
    }
    __syncthreads();

    const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
    const std::int32_t* table = tables + static_cast<std::int64_t>(table_rows[t0 / width]) * table_stride;
    const int slot_width      = kIndexDim / ratio;
    int token_offset[kLanePartials], slot[kLanePartials];
#pragma unroll
    for (int j = 0; j < kLanePartials; ++j) {
        token_offset[j] = (lane + 32 * j) / slot_width;
        slot[j]         = (lane + 32 * j) % slot_width;
    }
    const bool block_in_page = ratio <= kPagedKVPageSize; // a block's tokens share one page
    const float scale        = rsqrtf(static_cast<float>(kIndexDim));
    for (int step = first_step + warp; step < steps; step += static_cast<int>(gridDim.x) * kScoreWarps) {
        const int b0 = step * kScoreStepBlocks;
        float key[kScoreStepBlocks][kLanePartials];
#pragma unroll
        for (int bb = 0; bb < kScoreStepBlocks; ++bb) {
            const int b = b0 + bb;
            if (b >= tile_blocks) {
#pragma unroll
                for (int j = 0; j < kLanePartials; ++j) { key[bb][j] = 0.0F; }
            } else if (block_in_page) {
                int local;
                const bf16* plane = qsa_pooled_plane(spaces, pooled, table[(ratio * b) >> kPagedKVPageShift], local);
                const std::int64_t page = static_cast<std::int64_t>(slot_width) * kPagedKVPageSize * local;
#pragma unroll
                for (int j = 0; j < kLanePartials; ++j) {
                    const int token = ratio * b + token_offset[j];
                    key[bb][j] = __bfloat162float(plane[page + static_cast<std::int64_t>(slot_width) *
                                                                   (token & kPagedKVPageMask) + slot[j]]);
                }
            } else {
#pragma unroll
                for (int j = 0; j < kLanePartials; ++j) {
                    key[bb][j] = __bfloat162float(
                        *pooled_element(pooled, spaces, table, ratio * b + token_offset[j], slot_width, slot[j]));
                }
            }
        }
        for (int c = 0; c < nc; ++c) {
            const int blocks = column_blocks[c];
            if (b0 >= blocks) { continue; } // warp-uniform: this column sees none of the step
            float4 q[kScoreHeadSlots];
#pragma unroll
            for (int h = 0; h < kScoreHeadSlots; ++h) {
                const uint2 raw = qs[c][h][lane];
                const float2 lo = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.x));
                const float2 hi = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.y));
                q[h]            = make_float4(lo.x, lo.y, hi.x, hi.y);
            }
            // Value bb * 4 + h: this lane's partial of block b0 + bb, head h, accumulated in j order
            // from 0 like the one-warp-per-block kernel's `dot += q[h][lane + 32 j] * key[j]`.
            float v[32];
#pragma unroll
            for (int bb = 0; bb < kScoreStepBlocks; ++bb) {
#pragma unroll
                for (int h = 0; h < kScoreHeadSlots; ++h) {
                    float dot = 0.0F;
                    dot += q[h].x * key[bb][0];
                    dot += q[h].y * key[bb][1];
                    dot += q[h].z * key[bb][2];
                    dot += q[h].w * key[bb][3];
                    v[bb * kScoreHeadSlots + h] = dot;
                }
            }
            // Lane 4 bb + h now holds head h's dot of block b0 + bb; heads join in head order.
            const float relu  = fmaxf(reduce_scatter_32(v, lane), 0.0F);
            const float relu1 = __shfl_down_sync(kFullMask, relu, 1);
            const float relu2 = __shfl_down_sync(kFullMask, relu, 2);
            const float relu3 = __shfl_down_sync(kFullMask, relu, 3);
            const int b       = b0 + lane / kScoreHeadSlots;
            if (lane % kScoreHeadSlots == 0 && b < blocks) {
                float score = 0.0F;
                score += relu;
                if (index_heads > 1) { score += relu1; }
                if (index_heads > 2) { score += relu2; }
                if (index_heads > 3) { score += relu3; }
                scores[static_cast<std::size_t>(c0 + c) * score_stride + b] = score * scale;
            }
        }
    }
    pdl::trigger_dependents();
}

// --------------------------------------------------------------------------- wide-row scoring

// Prompt rows (width >= kWideMinWidth): the same scores as qsa_score_kernel, register-tiled for
// the FP32 pipes instead of shuffles. A thread owns one column's four head dots against two
// blocks (8 dots). It forms each dot's 32 lane partials itself, the chain over j from 0 exactly as
// lane l would, and adds them in the butterfly's tree: the partial of lane l pairs with lane
// l ^ 16 first, then the pairs at offset 8, 4, 2 and 1. Taking the leaves in bit-reversed lane
// order turns that tree into a binary counter, so a six-entry stack per dot holds every open
// subtree. FP32 addition is commutative, so each sum is bitwise the butterfly's.
// A CTA scores a 16-column row tile against key tiles of 32 blocks; the next key tile travels in
// registers while the current one is scored. Queries are staged as FP32 (rows of 32 float4 lanes plus
// one pad), keys as BF16 uint4 pair rows: row p lane l holds the uint2 (four BF16) that lane l of
// blocks p's pair (g, g + 8) scores, widened where used (exactly the values the FP32 staging held),
// so a thread's whole key leaf is one LDS.128. Two CTAs fit an SM (42 KiB
// of shared memory and at most 128 registers per thread each): the kernel is latency-bound on its
// dependent adds, and 16 warps per SM hide twice what 8 did.
// Packed-BF16 query staging (uint2 lanes, 25,344 B, 3 CTAs/SM eligible) was measured and rejected:
// widening every leaf costs one LDS.64 plus four shift/mask ALU ops per head against the FP32
// staging's single LDS.128, and this kernel is warp-instruction-bound, so the 512 extra leaf ops per
// key tile regressed select_us by 10..19% at every wide point even with occupancy raised to 3 CTAs.
constexpr int kWideThreads      = 256;
constexpr int kWideCtasPerSm    = 2;
constexpr int kWideBlocks       = 32;                    // blocks per key tile
constexpr int kWideLaneBlocks   = 8;                     // consecutive blocks across a warp's lanes
constexpr int kWideThreadBlocks = 2;                     // a thread's blocks: g + 8 i, i = 0..1
constexpr int kWideRow          = 32 + 1;                // lanes per staged row (one pad)
constexpr int kWideMinWidth     = 2 * kScoreTileColumns; // rows this narrow keep qsa_score_kernel
constexpr int kWideStack        = 6;                     // open subtrees per dot, plus the new leaf
constexpr std::size_t kWideQueryFloat4 = static_cast<std::size_t>(kScoreHeadSlots) * kScoreTileColumns * kWideRow;
constexpr std::size_t kWideKeyPairs    = static_cast<std::size_t>(kWideBlocks) / 2 * kWideRow; // uint4: one lane's two blocks
constexpr std::size_t kWideSharedBytes = sizeof(float4) * (kWideQueryFloat4 + kWideKeyPairs);
static_assert(kWideThreads == kScoreTileColumns * kWideBlocks / kWideThreadBlocks, "one thread per 2 blocks");
static_assert(kWideThreads * 16 == kWideBlocks * kIndexDim, "a thread loads 16 BF16 of one block's key");

__host__ __device__ constexpr int bit_reverse5(int i) {
    return ((i & 1) << 4) | ((i & 2) << 2) | (i & 4) | ((i & 8) >> 2) | ((i & 16) >> 4);
}
__host__ __device__ constexpr int popcount5(int i) {
    return (i & 1) + ((i >> 1) & 1) + ((i >> 2) & 1) + ((i >> 3) & 1) + ((i >> 4) & 1);
}
__host__ __device__ constexpr int trailing_zeros(int n) {
    int count = 0;
    while (count < 6 && (n & 1) == 0) {
        n >>= 1;
        ++count;
    }
    return count;
}

using WideStacks = float[kScoreHeadSlots][kWideThreadBlocks][kWideStack];

// Four BF16 (lane l's elements l + 32 j, j = 0..3) as FP32, exactly.
__device__ __forceinline__ float4 widen_lane(uint2 raw) {
    return make_float4(__uint_as_float(raw.x << 16), __uint_as_float(raw.x & 0xFFFF0000U), __uint_as_float(raw.y << 16),
                       __uint_as_float(raw.y & 0xFFFF0000U));
}

// Leaf I (in bit-reversed order: lane l = bit_reverse5(I)) of all 8 dots, pushed and folded.
// `key` is the thread's block-pair row as uint4 lanes (store_tile stages a thread's two blocks,
// g and g + 8, as adjacent rows), so one uint2 per block comes from a single LDS.128 per leaf
// instead of two LDS.64.
template <int I>
__device__ __forceinline__ void wide_leaves(const float4* __restrict__ query, const uint4* __restrict__ key,
                                            WideStacks& stack) {
    if constexpr (I < 32) {
        constexpr int l      = bit_reverse5(I);
        constexpr int depth  = popcount5(I);         // open subtrees before this leaf
        constexpr int merges = trailing_zeros(I + 1); // subtrees this leaf completes
        float4 q[kScoreHeadSlots], k[kWideThreadBlocks];
#pragma unroll
        for (int h = 0; h < kScoreHeadSlots; ++h) { q[h] = query[h * kScoreTileColumns * kWideRow + l]; }
        {
            const uint4 kk = key[l];
            k[0]           = widen_lane(make_uint2(kk.x, kk.y));
            k[1]           = widen_lane(make_uint2(kk.z, kk.w));
        }
#pragma unroll
        for (int h = 0; h < kScoreHeadSlots; ++h) {
#pragma unroll
            for (int i = 0; i < kWideThreadBlocks; ++i) {
                // Lane l's partial, as qsa_score_kernel forms it.
                float dot = 0.0F;
                dot += q[h].x * k[i].x;
                dot += q[h].y * k[i].y;
                dot += q[h].z * k[i].z;
                dot += q[h].w * k[i].w;
                stack[h][i][depth] = dot;
#pragma unroll
                for (int m = 0; m < merges; ++m) {
                    stack[h][i][depth - 1 - m] = stack[h][i][depth - 1 - m] + stack[h][i][depth - m];
                }
            }
        }
        wide_leaves<I + 1>(query, key, stack);
    }
}

// Grid (CTAs per tile, tiles), kWideThreads threads, kWideSharedBytes of dynamic shared memory.
// Warp w scores columns 4 (w % 4) .. + 3 (lane / 8) against blocks 16 (w / 4) + lane % 8 + 8 i of
// each key tile. Arguments as qsa_score_kernel's.
__global__ void __launch_bounds__(kWideThreads, kWideCtasPerSm)
    qsa_score_wide_kernel(const bf16* index_q, int index_heads, const bf16* pooled,
                          const std::int32_t* __restrict__ tables, int table_stride,
                          const std::int32_t* __restrict__ table_rows, const std::int32_t* __restrict__ positions,
                          const SpeculativeTreeRow* __restrict__ trees, int width, int ratio, int top_blocks,
                          int group_begin, const ScoreTiles tiles, float* __restrict__ scores, int score_stride,
                          std::uint32_t* __restrict__ clear, int clear_words, QsaPageSpaces spaces) {
    pdl::enter_streaming();
    const int tid = static_cast<int>(threadIdx.x);
    if (clear != nullptr) {
        const int total_threads = static_cast<int>(gridDim.x * gridDim.y) * kWideThreads;
        const int global_tid    = static_cast<int>(blockIdx.y * gridDim.x + blockIdx.x) * kWideThreads + tid;
        for (int i = global_tid; i < clear_words; i += total_threads) { clear[i] = 0; }
    }
    extern __shared__ float4 wide_shared[];
    float4* qs = wide_shared;                                   // [head][column][kWideRow]: lane l = q[l + 32 j], j = 0..3
    uint4* ks  = reinterpret_cast<uint4*>(wide_shared + kWideQueryFloat4); // [pair][kWideRow]: lane l = the two
    // uint2 (block g, block g + kWideLaneBlocks) of lane l's key elements, staged by store_tile.
    __shared__ int column_blocks[kScoreTileColumns];

    const int tile = static_cast<int>(blockIdx.y);
    const int c0 = tiles.first[tile], nc = tiles.columns[tile];
    const int t0 = group_begin + c0;
    int tile_blocks = 0;
    for (int c = 0; c < nc; ++c) {
        tile_blocks = max(tile_blocks, (qsa_logical_position(positions, trees, width, t0 + c) + 1) / ratio);
    }
    const int key_tiles = (tile_blocks + kWideBlocks - 1) / kWideBlocks;
    if (tile_blocks <= top_blocks || static_cast<int>(blockIdx.x) >= key_tiles) {
        pdl::trigger_dependents();
        return; // every column of the tile is dense, or this CTA has no key tile
    }
    if (tid < kScoreTileColumns) {
        column_blocks[tid] = tid < nc ? (qsa_logical_position(positions, trees, width, t0 + tid) + 1) / ratio : 0;
    }
    for (int i = tid; i < kScoreTileColumns * kScoreHeadSlots * kIndexDim; i += kWideThreads) {
        const int c = i / (kScoreHeadSlots * kIndexDim), h = (i / kIndexDim) % kScoreHeadSlots, d = i % kIndexDim;
        const float value =
            c < nc && h < index_heads
                ? __bfloat162float(index_q[(static_cast<std::size_t>(t0 + c) * index_heads + h) * kIndexDim + d])
                : 0.0F;
        reinterpret_cast<float*>(&qs[(h * kScoreTileColumns + c) * kWideRow + d % 32])[d / 32] = value;
    }

    // Key staging: thread t carries elements 16 (t % 8) .. + 15 of block t / 8 of a tile (two
    // 16-byte loads; a block's key is 128 contiguous BF16 because its tokens share a page). Element e
    // goes to lane e % 32 of the block's pair row, uint2 (e / 32) of the lane's block half, as the
    // uint4 rows ordered it: pair (g, g + 8) of the block's 16-block panel, half by panel position.
    const std::int32_t* table = tables + static_cast<std::int64_t>(table_rows[t0 / width]) * table_stride;
    const int slot_width = kIndexDim / ratio;
    const int load_block = tid / 8, load_part = tid % 8;
    auto load_tile = [&](int key_tile, uint4 (&raw)[2]) {
        const int b = key_tile * kWideBlocks + load_block;
        if (b < tile_blocks) {
            int local;
            const bf16* plane = qsa_pooled_plane(spaces, pooled, table[(ratio * b) >> kPagedKVPageShift], local);
            const auto* src   = reinterpret_cast<const uint4*>(
                plane + static_cast<std::int64_t>(slot_width) * kPagedKVPageSize * local +
                static_cast<std::int64_t>(slot_width) * ((ratio * b) & kPagedKVPageMask) + 16 * load_part);
            raw[0] = src[0];
            raw[1] = src[1];
        } else {
            raw[0] = raw[1] = make_uint4(0U, 0U, 0U, 0U);
        }
    };
    auto store_tile = [&](const uint4 (&raw)[2], uint4* buffer) {
        // Pair row (g, g + 8) of the block's 16-block panel; the block's half of each lane's uint4.
        const int pair_row = (load_block % 8) + (load_block / 16) * 8;
        const int half_row = (load_block % 16) / 8;
        auto* row          = reinterpret_cast<std::uint16_t*>(buffer + pair_row * kWideRow);
        const auto* half   = reinterpret_cast<const std::uint16_t*>(raw);
        const int lane0    = 16 * (load_part % 2), j = load_part / 2;
#pragma unroll
        for (int k = 0; k < 16; ++k) { row[(lane0 + k) * 8 + half_row * 4 + j] = half[k]; }
    };

    const int warp = tid / 32, lane = tid % 32;
    const int column     = 4 * (warp % 4) + lane / kWideLaneBlocks;
    const int warp_first = 16 * (warp / 4);                       // first block of the warp in a tile
    const int first      = warp_first + lane % kWideLaneBlocks;    // this thread's first block in a tile
    const float scale    = rsqrtf(static_cast<float>(kIndexDim));

    uint4 raw[2]; // register copy of the key tile one ahead of shared memory
    const int stride_ctas = static_cast<int>(gridDim.x);
    int key_tile = static_cast<int>(blockIdx.x);
    load_tile(key_tile, raw);
    store_tile(raw, ks);
    __syncthreads();
    int warp_blocks = 0; // the most blocks a column of the warp sees
    for (int c = 0; c < 4; ++c) { warp_blocks = max(warp_blocks, column_blocks[4 * (warp % 4) + c]); }
    const int blocks = column_blocks[column];
    const float4* query = qs + column * kWideRow;
    for (; key_tile < key_tiles; key_tile += stride_ctas) {
        const int next = key_tile + stride_ctas;
        if (next < key_tiles) { load_tile(next, raw); }
        const int b0 = key_tile * kWideBlocks;
        if (b0 + warp_first < warp_blocks) { // warp-uniform: some column of the warp sees a block
            WideStacks stack;
            const uint4* key4 = ks + ((lane % kWideLaneBlocks) + (warp / 4) * kWideLaneBlocks) * kWideRow;
            wide_leaves<0>(query, key4, stack);
#pragma unroll
            for (int i = 0; i < kWideThreadBlocks; ++i) {
                const int b = b0 + first + kWideLaneBlocks * i;
                if (b < blocks) {
                    float score = 0.0F;
                    score += fmaxf(stack[0][i][0], 0.0F);
                    if (index_heads > 1) { score += fmaxf(stack[1][i][0], 0.0F); }
                    if (index_heads > 2) { score += fmaxf(stack[2][i][0], 0.0F); }
                    if (index_heads > 3) { score += fmaxf(stack[3][i][0], 0.0F); }
                    scores[static_cast<std::size_t>(c0 + column) * score_stride + b] = score * scale;
                }
            }
        }
        if (next < key_tiles) {
            __syncthreads(); // every warp is done with this tile's keys
            store_tile(raw, ks);
            __syncthreads();
        }
    }
    pdl::trigger_dependents();
}

// ------------------------------------------------------------------------- Tensor Core scoring

// Fast-path BF16 Tensor Core m16n8k16 scoring for width >= 4 (verify, MTP, and prompt chunks).
// Each CTA walks a persistent work-map of (query tile, 64-key block) jobs, staging 64 pooled BF16
// keys with cp.async into bank-conflict-free swizzled shared memory and scoring 4 queries x 4 heads
// per 16 MMA rows with FP32 accumulation.
constexpr int kMmaKeyTile  = 64;
constexpr int kMmaMaxTiles = 64; // at width >= 4, a 128-column pass has at most 34 row tiles

__device__ __forceinline__ void cp_async_cg_l2_retain_16(void* smem_dst, const void* gmem_src) {
    std::uint64_t policy;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(policy));
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, 16, %2;\n"
                 :
                 : "r"(smem_addr(smem_dst)), "l"(gmem_src), "l"(policy));
}

template <int Capacity>
struct MmaWorkMap {
    static_assert(Capacity % 32 == 0);
    int length[Capacity];
    int prefix[Capacity + 1];
    int groups;

    __device__ __forceinline__ int group_for(int job) const {
        int lo = 0, hi = groups;
        while (lo + 1 < hi) {
            const int mid = (lo + hi) >> 1;
            if (prefix[mid] <= job) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        return lo;
    }
};

template <int Tile, bool Vector, int Stages, bool RetainKeys = false>
__global__ void __launch_bounds__(Tile > 16 ? Tile * 8 : 128)
    qsa_score_mma_kernel(const bf16* index_q, int index_heads, const bf16* pooled,
                         const std::int32_t* __restrict__ tables, int table_stride,
                         const std::int32_t* __restrict__ table_rows, const std::int32_t* __restrict__ positions,
                         const SpeculativeTreeRow* __restrict__ trees, int width, int ratio, int top_blocks,
                         int group_begin, int group_columns, const ScoreTiles tiles, float* __restrict__ scores,
                         int score_stride, std::uint32_t* __restrict__ clear, int clear_words, QsaPageSpaces spaces) {
    constexpr int Threads   = Tile > 16 ? Tile * 8 : 128;
    constexpr int KeyGroups = Tile > 16 ? 1 : 16 / Tile;
    constexpr int WarpKeys  = kMmaKeyTile / KeyGroups;
    constexpr int NT        = WarpKeys / 8;
    static_assert(Tile >= 4 && Tile <= 128 && (Tile & (Tile - 1)) == 0);
    static_assert(Stages == 1 || Stages == 2);

    pdl::enter_streaming();
    const int tid = static_cast<int>(threadIdx.x);
    if (clear != nullptr) {
        const int total_threads = static_cast<int>(gridDim.x) * Threads;
        const int global_tid    = static_cast<int>(blockIdx.x) * Threads + tid;
        for (int i = global_tid; i < clear_words; i += total_threads) { clear[i] = 0; }
    }

    __shared__ MmaWorkMap<kMmaMaxTiles> jobs;
    __shared__ int col_blocks[kQsaSelectGroupColumns];
    extern __shared__ __align__(1024) bf16 pipe_keys[];

    int my_blocks = 0;
    if (tid < group_columns) {
        my_blocks       = (qsa_logical_position(positions, trees, width, group_begin + tid) + 1) / ratio;
        col_blocks[tid] = my_blocks;
    }
    if (!__syncthreads_or(my_blocks > top_blocks)) {
        pdl::trigger_dependents();
        return;
    }
    if (tid < tiles.count) {
        const int c0 = tiles.first[tid], nc = tiles.columns[tid];
        int tile_max = 0;
        for (int c = 0; c < nc; ++c) { tile_max = max(tile_max, col_blocks[c0 + c]); }
        jobs.length[tid] = tile_max;
    }
    __syncthreads();
    if (tid < 32) {
        int carry = 0;
#pragma unroll
        for (int part = 0; part < kMmaMaxTiles / 32; ++part) {
            const int g   = part * 32 + tid;
            const int n_b = g < tiles.count ? jobs.length[g] : 0;
            int sum       = n_b > top_blocks ? (n_b + kMmaKeyTile - 1) / kMmaKeyTile : 0;
#pragma unroll
            for (int shift = 1; shift < 32; shift <<= 1) {
                const int x = __shfl_up_sync(kFullMask, sum, shift);
                if (tid >= shift) { sum += x; }
            }
            if (g < tiles.count) { jobs.prefix[g + 1] = carry + sum; }
            carry += __shfl_sync(kFullMask, sum, 31);
        }
        if (tid == 0) {
            jobs.prefix[0] = 0;
            jobs.groups    = tiles.count;
        }
    }
    __syncthreads();

    const int total = jobs.prefix[jobs.groups];
    if (static_cast<int>(blockIdx.x) >= total) {
        pdl::trigger_dependents();
        return;
    }

    const int lane       = tid & 31;
    const int warp       = tid >> 5;
    const int slot_width = kIndexDim / ratio;
    const float scale    = rsqrtf(static_cast<float>(kIndexDim));

    auto stage = [&](int job, int buffer) {
        const int group           = jobs.group_for(job);
        const int c0              = tiles.first[group];
        const int row_idx         = (group_begin + c0) / width;
        const std::int32_t* table = tables + static_cast<std::int64_t>(table_rows[row_idx]) * table_stride;
        const int g0              = (job - jobs.prefix[group]) * kMmaKeyTile;
        const int group_len       = jobs.length[group];
        bf16* dst                 = pipe_keys + buffer * kMmaKeyTile * kIndexDim;
        for (int i = tid; i < kMmaKeyTile * 16; i += Threads) {
            const int k_row = i / 16;
            const int d     = (i % 16) * 8;
            bf16* out       = dst + k_row * kIndexDim + (d ^ ((k_row & 7) << 3));
            if (g0 + k_row < group_len) {
                const int token = ratio * (g0 + k_row);
                int local;
                const bf16* plane = qsa_pooled_plane(spaces, pooled, table[token >> kPagedKVPageShift], local);
                const bf16* in    = plane + static_cast<std::int64_t>(slot_width) * kPagedKVPageSize * local +
                                    static_cast<std::int64_t>(slot_width) * (token & kPagedKVPageMask) + d;
                if constexpr (Vector) {
                    if constexpr (RetainKeys) {
                        cp_async_cg_l2_retain_16(out, in);
                    } else {
                        cp_async<16, Cache::cg>(out, in);
                    }
                } else {
#pragma unroll
                    for (int j = 0; j < 8; ++j) { out[j] = in[j]; }
                }
            } else {
                store_vec(out, make_int4(0, 0, 0, 0));
            }
        }
        if constexpr (Vector) { cp_commit(); }
    };

    if constexpr (Stages == 2) { stage(static_cast<int>(blockIdx.x), 0); }
    unsigned qf[8][4];
    int previous = -1, limit[2], col_slot[2];
    const int grid_stride = static_cast<int>(gridDim.x);
    for (int job = static_cast<int>(blockIdx.x), step_job = 0; job < total; job += grid_stride, ++step_job) {
        const int group  = jobs.group_for(job);
        const int g0     = (job - jobs.prefix[group]) * kMmaKeyTile;
        const int buffer = Stages == 2 ? (step_job & 1) : 0;
        if constexpr (Stages == 1) { stage(job, buffer); }
        if (group != previous) {
            const int c0 = tiles.first[group], nc = tiles.columns[group];
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int row         = (warp / KeyGroups) * 16 + (lane >> 2) + half * 8;
                const int t           = row >> 2;
                const int h           = row & 3;
                col_slot[half]        = c0 + t;
                limit[half]           = t < nc ? col_blocks[c0 + t] : 0;
                const bool active_row = limit[half] > top_blocks && h < index_heads;
                const bf16* q_row =
                    active_row
                        ? index_q + (static_cast<std::size_t>(group_begin + c0 + t) * index_heads + h) * kIndexDim
                        : nullptr;
#pragma unroll
                for (int k = 0; k < 8; ++k) {
#pragma unroll
                    for (int col = 0; col < 2; ++col) {
                        const int d = k * 16 + (lane & 3) * 2 + col * 8;
                        qf[k][half + 2 * col] =
                            active_row ? __ldg(reinterpret_cast<const unsigned*>(q_row + d)) : 0U;
                    }
                }
            }
            previous = group;
        }
        if constexpr (Vector) { cp_wait<0>(); }
        __syncthreads();
        if constexpr (Stages == 2) {
            if (job + grid_stride < total) { stage(job + grid_stride, 1 - buffer); }
        }
        const bf16* ks = pipe_keys + buffer * kMmaKeyTile * kIndexDim;
        float acc[NT][4]{};
#pragma unroll
        for (int k = 0; k < 8; ++k) {
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                unsigned bf[2];
                const int k_row = (warp % KeyGroups) * WarpKeys + n * 8 + (lane & 7);
                const int d     = k * 16 + ((lane >> 3) & 1) * 8;
                ldmatrix_x2(bf[0], bf[1], smem_addr(ks + k_row * kIndexDim + (d ^ ((k_row & 7) << 3))));
                mma_bf16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], qf[k][0], qf[k][1], qf[k][2], qf[k][3],
                         bf[0], bf[1]);
            }
        }
#pragma unroll
        for (int n = 0; n < NT; ++n) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float v = fmaxf(acc[n][j], 0.0F);
                v += __shfl_xor_sync(kFullMask, v, 4);
                v += __shfl_xor_sync(kFullMask, v, 8);
                const int g = g0 + (warp % KeyGroups) * WarpKeys + n * 8 + (lane & 3) * 2 + (j & 1);
                if (((lane >> 2) & 3) == 0 && limit[j >> 1] > top_blocks && g < limit[j >> 1]) {
                    scores[static_cast<std::size_t>(col_slot[j >> 1]) * score_stride + g] = v * scale;
                }
            }
        }
        __syncthreads();
    }
    pdl::trigger_dependents();
}

// ----------------------------------------------------------------------------------- selection

struct SelectShared {
    int warp_totals[kSelectWarps];          // block_exclusive_sum's per-warp inclusive totals
    unsigned radix[256];
    int pick_digit, pick_above, pick_count; // radix_threshold's digit
    int bin, bin_above, bin_count;          // the score bin holding the top-th score
    unsigned candidates;                    // keys appended to this CTA's candidate buffer
    unsigned long long and_key, or_key;     // over this CTA's share of the bin's keys
    int last;                               // this CTA arrived last at its column
};

// Scores are non-negative (a sum of ReLUs times a positive scale), so the ordered bits' top bit is
// set and bits [30:20] order the bins like the scores.
__device__ __forceinline__ int score_bin(float score) {
    return static_cast<int>((ordered_score_bits(score) >> kBinShift) & (kScoreBins - 1));
}

// First block of slice s of S over n blocks; multiples of 4 so float4 loads stay aligned.
__host__ __device__ __forceinline__ int slice_begin(int n, int s, int slices) {
    if (s >= slices) { return n; }
    return static_cast<int>((static_cast<std::int64_t>(n) * s / slices) & ~std::int64_t{3});
}

// Walks scores [lo, hi) of a column (lo a multiple of 4) in chunks of 128, each warp one chunk at
// a time with kWalkBatch chunks' loads in flight. Lane l sees scores base..base+3, base = lo +
// 128 chunk + 4 l; `valid` masks those < hi. Calls are warp-uniform.
template <class Visit>
__device__ __forceinline__ void walk_scores(const float* column, int lo, int hi, Visit&& visit) {
    const int lane = static_cast<int>(threadIdx.x) % 32, warp = static_cast<int>(threadIdx.x) / 32;
    const int chunks = (hi - lo + 127) / 128;
    for (int first = warp; first < chunks; first += kSelectWarps * kWalkBatch) {
        float4 v[kWalkBatch];
#pragma unroll
        for (int u = 0; u < kWalkBatch; ++u) {
            const int base = lo + (first + u * kSelectWarps) * 128 + 4 * lane;
            v[u] = base < hi ? *reinterpret_cast<const float4*>(column + base) : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
        }
#pragma unroll
        for (int u = 0; u < kWalkBatch; ++u) {
            const int chunk = first + u * kSelectWarps;
            if (chunk >= chunks) { break; }
            const int base     = lo + chunk * 128 + 4 * lane;
            const int left     = hi - base;
            const unsigned ok  = left >= 4 ? 0xFU : left > 0 ? (1U << left) - 1 : 0U;
            visit(chunk, base, v[u], ok);
        }
    }
}

// Lanes 8m..8m+7 hold four flags each of scores 32m..32m+31 of a chunk; lane 8m returns the word.
__device__ __forceinline__ unsigned chunk_word(unsigned nibble, int lane) {
    unsigned word = nibble << (4 * (lane & 7));
    word |= __shfl_xor_sync(kFullMask, word, 1);
    word |= __shfl_xor_sync(kFullMask, word, 2);
    word |= __shfl_xor_sync(kFullMask, word, 4);
    return word;
}

__device__ __forceinline__ void score_histogram(const float* column, int lo, int hi, unsigned* hist) {
    walk_scores(column, lo, hi, [&](int, int, float4 v, unsigned ok) {
        const float s[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            if ((ok >> k) & 1U) { atomicAdd(hist + score_bin(s[k]), 1U); }
        }
    });
}

// Block-wide exclusive prefix sum over kSelectThreads: a 32-lane inclusive scan by shuffles per
// warp, the kSelectWarps warp totals scanned by warp 0 in shared memory, then each thread adds its
// warp's prefix. Integer sums are exact, so every result is bit-for-bit cub::BlockScan's. All 512
// threads must call it (the two callers below do; the kernel's early exits precede them).
__device__ __forceinline__ int block_exclusive_sum(int value, SelectShared& sh) {
    const int tid = static_cast<int>(threadIdx.x), lane = tid & 31, warp = tid >> 5;
    int inclusive = value;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int other = __shfl_up_sync(kFullMask, inclusive, o);
        if (lane >= o) { inclusive += other; }
    }
    if (lane == 31) { sh.warp_totals[warp] = inclusive; }
    __syncthreads();
    if (warp == 0) {
        int v = lane < kSelectWarps ? sh.warp_totals[lane] : 0;
#pragma unroll
        for (int o = 1; o < kSelectWarps; o <<= 1) {
            const int other = __shfl_up_sync(kFullMask, v, o);
            if (lane >= o) { v += other; }
        }
        if (lane < kSelectWarps) { sh.warp_totals[lane] = v; }
    }
    __syncthreads();
    return (warp == 0 ? 0 : sh.warp_totals[warp - 1]) + inclusive - value;
}

// The bin holding the top-th highest score: sh.bin, the count above it and its own count.
__device__ void find_score_bin(const unsigned* hist, int top_blocks, SelectShared& sh) {
    const int tid = static_cast<int>(threadIdx.x);
    int count[4], local = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        count[k] = static_cast<int>(hist[kScoreBins - 1 - 4 * tid - k]);
        local += count[k];
    }
    int above = block_exclusive_sum(local, sh);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        if (above < top_blocks && top_blocks <= above + count[k]) {
            sh.bin       = kScoreBins - 1 - 4 * tid - k;
            sh.bin_above = above;
            sh.bin_count = count[k];
        }
        above += count[k];
    }
    __syncthreads();
}

// One pass over scores [lo, hi): scores above the bin set their bit in `bitmap` (word w holds
// scores lo + 32 w ..), the bin's scores append their order keys to `candidates` (up to
// kCandidateCapacity; `count` counts them all) and fold into sh.and_key / sh.or_key.
__device__ void filter_scores(const float* column, int lo, int hi, int bin, unsigned* bitmap,
                              std::uint64_t* candidates, unsigned* count, SelectShared& sh) {
    const int lane              = static_cast<int>(threadIdx.x) % 32;
    unsigned long long and_key  = ~0ULL, or_key = 0;
    walk_scores(column, lo, hi, [&](int chunk, int base, float4 v, unsigned ok) {
        const float s[4] = {v.x, v.y, v.z, v.w};
        unsigned above = 0, inside = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            if ((ok >> k) & 1U) {
                const int b = score_bin(s[k]);
                above |= static_cast<unsigned>(b > bin) << k;
                inside |= static_cast<unsigned>(b == bin) << k;
            }
        }
        const unsigned word = chunk_word(above, lane);
        if ((lane & 7) == 0) { bitmap[chunk * 4 + lane / 8] = word; }
        const int mine = __popc(inside);
        if (__any_sync(kFullMask, mine != 0)) {
            int inclusive = mine;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int other = __shfl_up_sync(kFullMask, inclusive, o);
                if (lane >= o) { inclusive += other; }
            }
            unsigned start = 0;
            if (lane == 31) { start = atomicAdd(count, static_cast<unsigned>(inclusive)); }
            start    = __shfl_sync(kFullMask, start, 31);
            int slot = static_cast<int>(start) + inclusive - mine;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                if ((inside >> k) & 1U) {
                    const std::uint64_t key = score_id_order_key(s[k], base + k);
                    if (slot < kCandidateCapacity) { candidates[slot] = key; }
                    ++slot;
                    and_key &= key;
                    or_key |= key;
                }
            }
        }
    });
    const unsigned and_hi = __reduce_and_sync(kFullMask, static_cast<unsigned>(and_key >> 32));
    const unsigned and_lo = __reduce_and_sync(kFullMask, static_cast<unsigned>(and_key));
    const unsigned or_hi  = __reduce_or_sync(kFullMask, static_cast<unsigned>(or_key >> 32));
    const unsigned or_lo  = __reduce_or_sync(kFullMask, static_cast<unsigned>(or_key));
    if (lane == 0) {
        atomicAnd(&sh.and_key, (static_cast<unsigned long long>(and_hi) << 32) | and_lo);
        atomicOr(&sh.or_key, (static_cast<unsigned long long>(or_hi) << 32) | or_lo);
    }
}

// Keys of the candidate buffer.
struct CandidateKeys {
    const std::uint64_t* keys;
    int count;
    template <class F>
    __device__ __forceinline__ void for_each(F&& f) const {
        for (int i = static_cast<int>(threadIdx.x); i < count; i += kSelectThreads) { f(keys[i]); }
    }
};

// Keys of the bin's scores, read from the column (when the bin exceeds the candidate buffer).
struct BinKeys {
    const float* column;
    int n;
    int bin;
    template <class F>
    __device__ __forceinline__ void for_each(F&& f) const {
        walk_scores(column, 0, n, [&](int, int base, float4 v, unsigned ok) {
            const float s[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                if (((ok >> k) & 1U) && score_bin(s[k]) == bin) { f(score_id_order_key(s[k], base + k)); }
            }
        });
    }
};

// Warp 0: the digit holding the `remaining`-th highest key among sh.radix's 256 bins.
__device__ __forceinline__ void choose_digit(SelectShared& sh, int remaining) {
    const int lane = static_cast<int>(threadIdx.x);
    int count[8], local = 0;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        count[k] = static_cast<int>(sh.radix[255 - 8 * lane - k]);
        local += count[k];
    }
    int inclusive = local;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int other = __shfl_up_sync(kFullMask, inclusive, o);
        if (lane >= o) { inclusive += other; }
    }
    int above = inclusive - local;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        if (above < remaining && remaining <= above + count[k]) {
            sh.pick_digit = 255 - 8 * lane - k;
            sh.pick_above = above;
            sh.pick_count = count[k];
        }
        above += count[k];
    }
}

// The `needed`-th highest of a set of distinct 64-bit keys, as a threshold T: exactly `needed` keys
// of the set are >= T. Radix select, 8 bits a pass from the highest bit in which the keys differ
// (and_keys / or_keys over the set), stopping once a digit's keys are all taken. Block-wide.
template <class Keys>
__device__ std::uint64_t radix_threshold(const Keys& keys, int needed, std::uint64_t and_keys,
                                         std::uint64_t or_keys, SelectShared& sh) {
    const std::uint64_t differ = and_keys ^ or_keys;
    if (differ == 0) { return or_keys; } // a single key
    const int top        = 63 - __clzll(static_cast<long long>(differ));
    std::uint64_t mask   = top == 63 ? 0 : ~((std::uint64_t{2} << top) - 1); // bits every key shares
    std::uint64_t prefix = or_keys & mask;
    int remaining        = needed;
    int high             = top;
    for (;;) {
        const int low             = high >= 7 ? high - 7 : 0;
        const unsigned digit_mask = (2U << (high - low)) - 1;
        for (int i = static_cast<int>(threadIdx.x); i < 256; i += kSelectThreads) { sh.radix[i] = 0; }
        __syncthreads();
        keys.for_each([&](std::uint64_t key) {
            if ((key & mask) == prefix) {
                atomicAdd(&sh.radix[static_cast<unsigned>(key >> low) & digit_mask], 1U);
            }
        });
        __syncthreads();
        if (threadIdx.x < 32) { choose_digit(sh, remaining); }
        __syncthreads();
        remaining -= sh.pick_above;
        const bool done = sh.pick_count == remaining || low == 0;
        prefix |= static_cast<std::uint64_t>(sh.pick_digit) << low;
        mask |= static_cast<std::uint64_t>(digit_mask) << low;
        __syncthreads(); // every thread has read sh.pick_* before the next pass or caller reuses sh
        if (done) { return prefix; }
        high = low - 1;
    }
}

// Writes id_base + (bit positions of words[0..nwords)) in ascending order to out. Block-wide.
__device__ void compact_words(const unsigned* words, int nwords, int id_base, std::int32_t* out,
                              SelectShared& sh) {
    const int per   = (nwords + kSelectThreads - 1) / kSelectThreads;
    const int begin = min(nwords, static_cast<int>(threadIdx.x) * per);
    const int end   = min(nwords, begin + per);
    int local       = 0;
    for (int w = begin; w < end; ++w) { local += __popc(words[w]); }
    int offset = block_exclusive_sum(local, sh);
    for (int w = begin; w < end; ++w) {
        unsigned bits = words[w];
        while (bits != 0) {
            out[offset++] = id_base + 32 * w + (__ffs(static_cast<int>(bits)) - 1);
            bits &= bits - 1;
        }
    }
}

// The column's selection from its complete histogram (dynamic_smem[0, kScoreBins words), every
// thread past the barrier that completed it): finds the bin, filters the column (bitmap above the
// bin, candidates in it), selects the candidates by radix and writes the ids in ascending order.
__device__ void select_from_histogram(const float* column, int n, int top_blocks, unsigned char* dynamic_smem,
                                      std::int32_t* out, SelectShared& sh) {
    const auto* hist = reinterpret_cast<const unsigned*>(dynamic_smem);
    if (threadIdx.x == 0) {
        sh.candidates = 0;
        sh.and_key    = ~0ULL;
        sh.or_key     = 0;
    }
    __syncthreads();
    find_score_bin(hist, top_blocks, sh);
    const int bin = sh.bin, needed = top_blocks - sh.bin_above, in_bin = sh.bin_count;
    auto* candidates = reinterpret_cast<std::uint64_t*>(dynamic_smem); // the histogram is read
    auto* bitmap     = reinterpret_cast<unsigned*>(dynamic_smem + kCandidateBytes);
    filter_scores(column, 0, n, bin, bitmap, candidates, &sh.candidates, sh);
    __syncthreads();
    const bool buffered = in_bin <= kCandidateCapacity;
    const std::uint64_t and_key = sh.and_key, or_key = sh.or_key;
    const std::uint64_t threshold = buffered ? radix_threshold(CandidateKeys{candidates, in_bin}, needed, and_key, or_key, sh)
                                             : radix_threshold(BinKeys{column, n, bin}, needed, and_key, or_key, sh);
    if (buffered) {
        for (int i = static_cast<int>(threadIdx.x); i < in_bin; i += kSelectThreads) {
            const std::uint64_t key = candidates[i];
            if (key >= threshold) {
                const int id = id_from_order_key(key);
                atomicOr(bitmap + id / 32, 1U << (id % 32));
            }
        }
    } else {
        const int lane = static_cast<int>(threadIdx.x) % 32;
        walk_scores(column, 0, n, [&](int chunk, int base, float4 v, unsigned ok) {
            const float s[4] = {v.x, v.y, v.z, v.w};
            unsigned chosen  = 0;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                if (((ok >> k) & 1U) && score_bin(s[k]) == bin && score_id_order_key(s[k], base + k) >= threshold) {
                    chosen |= 1U << k;
                }
            }
            const unsigned word = chunk_word(chosen, lane);
            if ((lane & 7) == 0) { bitmap[chunk * 4 + lane / 8] |= word; }
        });
    }
    __syncthreads();
    compact_words(bitmap, (n + 31) / 32, 0, out, sh);
}

// Grid (slices, columns of the group). Each CTA adds the histogram of its slice of the column into
// the column's global histogram (zeroed by the score kernel); the last CTA to arrive selects the
// column from it.
__global__ void __launch_bounds__(kSelectThreads)
    qsa_select_global_kernel(const std::int32_t* __restrict__ positions,
                             const SpeculativeTreeRow* __restrict__ trees, int width, int ratio, int top_blocks,
                             int group_begin, const float* __restrict__ scores, int score_stride,
                             unsigned* __restrict__ histograms, unsigned* __restrict__ arrivals,
                             std::int32_t* __restrict__ selected, std::int32_t* __restrict__ counts) {
    pdl::enter();
    extern __shared__ __align__(16) unsigned char dynamic_smem[];
    __shared__ SelectShared sh;
    const int c = static_cast<int>(blockIdx.y), t = group_begin + c;
    const int slice = static_cast<int>(blockIdx.x), slices = static_cast<int>(gridDim.x);
    const int n = (qsa_logical_position(positions, trees, width, t) + 1) / ratio;
    if (n <= top_blocks) {
        if (slice == 0 && threadIdx.x == 0) { counts[t] = -1; }
        return;
    }
    const float* column = scores + static_cast<std::size_t>(c) * score_stride;
    auto* hist          = reinterpret_cast<unsigned*>(dynamic_smem); // [kScoreBins], then candidates
    for (int i = static_cast<int>(threadIdx.x); i < kScoreBins; i += kSelectThreads) { hist[i] = 0; }
    __syncthreads();
    score_histogram(column, slice_begin(n, slice, slices), slice_begin(n, slice + 1, slices), hist);
    __syncthreads();
    if (slices > 1) {
        unsigned* column_hist = histograms + static_cast<std::size_t>(c) * kScoreBins;
        for (int i = static_cast<int>(threadIdx.x); i < kScoreBins; i += kSelectThreads) {
            if (hist[i] != 0) { atomicAdd(column_hist + i, hist[i]); }
        }
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) { sh.last = atomicAdd(arrivals + c, 1U) + 1 == static_cast<unsigned>(slices); }
        __syncthreads();
        if (sh.last == 0) { return; }
        __threadfence();

        for (int i = static_cast<int>(threadIdx.x); i < kScoreBins; i += kSelectThreads) {
            hist[i] = __ldcg(column_hist + i);
        }
    }
    select_from_histogram(column, n, top_blocks, dynamic_smem, selected + static_cast<std::size_t>(t) * top_blocks, sh);
    if (threadIdx.x == 0) { counts[t] = top_blocks; }
}

// ------------------------------------------------------------------------------------------ host

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("qsa select: ") + message); }
}

int ceil_div(std::int64_t a, std::int64_t b) { return static_cast<int>((a + b - 1) / b); }

std::size_t align256(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

int multiprocessors() {
    static const int count = [] {
        int device = 0, value = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount, device));
        return value;
    }();
    return count;
}

struct Layout {
    std::int32_t group  = 0; // columns per score pass
    std::int32_t stride = 0; // floats per column score row
    std::size_t scores = 0, state = 0, total = 0;
};

Layout layout(const QsaGeometry& g, std::int32_t columns, std::int32_t max_context) {
    Layout l;
    l.group  = std::min(columns, kQsaSelectGroupColumns);
    l.stride = (max_context / g.ratio + 1 + 31) / 32 * 32;
    l.scores = align256(sizeof(float) * static_cast<std::size_t>(l.group) * l.stride);
    // One arrival counter per column, then kScoreBins histogram words per column.
    l.state = align256(sizeof(std::uint32_t) * static_cast<std::size_t>(l.group) * (1 + kScoreBins));
    l.total = l.scores + l.state;
    return l;
}

ScoreTiles score_tiles(std::int32_t begin, std::int32_t columns, std::int32_t width,
                       std::int32_t max_tile_cols = kScoreTileColumns) {
    ScoreTiles tiles;
    for (std::int32_t c = 0; c < columns;) {
        const std::int32_t t       = begin + c;
        const std::int32_t row_end = (t / width + 1) * width;
        const std::int32_t n       = std::min({max_tile_cols, columns - c, row_end - t});
        tiles.first[tiles.count]   = static_cast<std::uint8_t>(c);
        tiles.columns[tiles.count] = static_cast<std::uint8_t>(n);
        ++tiles.count;
        c += n;
    }
    return tiles;
}

template <int Tile, bool Vector, int Stages, bool RetainKeys = false>
void launch_mma_score_impl(int workers, cudaStream_t stream, const bf16* index, int index_heads,
                           const bf16* pooled, const std::int32_t* tables, int table_stride,
                           const std::int32_t* rows, const std::int32_t* positions,
                           const SpeculativeTreeRow* trees, int width, int ratio,
                           int top_blocks, int begin, int n, const ScoreTiles& tiles, float* scores,
                           int score_stride, std::uint32_t* pass_state, int pass_clear_words,
                           const QsaPageSpaces& spaces) {
    constexpr int Threads          = Tile > 16 ? Tile * 8 : 128;
    constexpr std::size_t SmemSize = static_cast<std::size_t>(Stages) * kMmaKeyTile * kIndexDim * sizeof(bf16);
    CUDA_CHECK(pdl::launch_consumer(
        {dim3(static_cast<unsigned>(workers)), dim3(Threads), SmemSize, stream},
        qsa_score_mma_kernel<Tile, Vector, Stages, RetainKeys>, index, index_heads, pooled, tables,
        table_stride, rows, positions, trees, width, ratio, top_blocks, begin, n, tiles, scores, score_stride,
        pass_state, pass_clear_words, spaces));
}

template <bool Vector>
void launch_mma_score(int tile, int stages, bool retain_keys, int workers, cudaStream_t stream,
                      const bf16* index, int index_heads, const bf16* pooled, const std::int32_t* tables,
                      int table_stride, const std::int32_t* rows, const std::int32_t* positions,
                      const SpeculativeTreeRow* trees, int width, int ratio, int top_blocks, int begin, int n,
                      const ScoreTiles& tiles, float* scores, int score_stride, std::uint32_t* pass_state,
                      int pass_clear_words, const QsaPageSpaces& spaces) {
    const auto submit = [&]<int Tile>() {
        if (stages == 2) {
            launch_mma_score_impl<Tile, Vector, 2>(workers, stream, index, index_heads, pooled, tables,
                                                   table_stride, rows, positions, trees, width, ratio, top_blocks,
                                                   begin, n, tiles, scores, score_stride, pass_state,
                                                   pass_clear_words, spaces);
        } else {
            launch_mma_score_impl<Tile, Vector, 1>(workers, stream, index, index_heads, pooled, tables,
                                                   table_stride, rows, positions, trees, width, ratio, top_blocks,
                                                   begin, n, tiles, scores, score_stride, pass_state,
                                                   pass_clear_words, spaces);
        }
    };
    switch (tile) {
    case 4:
        submit.template operator()<4>();
        break;
    case 8:
        submit.template operator()<8>();
        break;
    case 16:
        submit.template operator()<16>();
        break;
    case 32:
        submit.template operator()<32>();
        break;
    case 64:
        submit.template operator()<64>();
        break;
    default:
        if constexpr (Vector) {
            if (retain_keys) {
                launch_mma_score_impl<64, true, 2, true>(workers, stream, index, index_heads, pooled, tables,
                                                         table_stride, rows, positions, trees, width, ratio,
                                                         top_blocks, begin, n, tiles, scores, score_stride,
                                                         pass_state, pass_clear_words, spaces);
                break;
            }
        }
        launch_mma_score_impl<64, Vector, 2>(workers, stream, index, index_heads, pooled, tables,
                                              table_stride, rows, positions, trees, width, ratio, top_blocks, begin,
                                              n, tiles, scores, score_stride, pass_state, pass_clear_words,
                                              spaces);
        break;
    }
}

// Opts a kernel into `bytes` of dynamic shared memory beyond the default 48 KiB, once per size.
template <class Kernel>
void reserve_shared(Kernel kernel, std::size_t bytes, std::atomic<std::size_t>& reserved) {
    if (bytes <= 48 * 1024 || bytes <= reserved.load(std::memory_order_acquire)) { return; }
    int device = 0, limit = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    require(bytes <= static_cast<std::size_t>(limit), "max_context needs more shared memory than a CTA has");
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)));
    std::size_t seen = reserved.load(std::memory_order_relaxed);
    while (seen < bytes && !reserved.compare_exchange_weak(seen, bytes, std::memory_order_acq_rel)) {}
}

std::atomic<std::size_t> g_global_reserved{0};
std::atomic<std::size_t> g_wide_reserved{0};

} // namespace

std::size_t qsa_select_scratch_bytes(const QsaGeometry& geometry, std::int32_t columns, std::int32_t max_context) {
    require(columns > 0 && max_context > 0 && geometry.ratio > 0, "columns and max_context must be positive");
    return layout(geometry, columns, max_context).total;
}

QsaSelectScores qsa_select_scores(const void* scratch, const QsaGeometry& geometry, std::int32_t max_context) {
    return {static_cast<const float*>(scratch), layout(geometry, 1, max_context).stride};
}

void qsa_select(const Tensor& index_q, const Tensor& pooled_pages, const QsaBatch& batch,
                const QsaGeometry& geometry, std::int32_t max_context, void* scratch,
                std::size_t scratch_bytes, const QsaSelectOutput& output, cudaStream_t stream,
                const QsaPageSpaces& spaces, bool allow_mma_score,
                const SpeculativeTreeRow* tree_rows) {
    const std::int32_t columns = batch.batch * batch.width;
    require(geometry.index_head_dim == kIndexDim && geometry.index_heads >= 1 &&
                geometry.index_heads <= kScoreHeadSlots && geometry.ratio > 1 && kIndexDim % geometry.ratio == 0 &&
                geometry.budget > 0 && geometry.budget % geometry.ratio == 0,
            "unsupported geometry");
    require(columns > 0 && index_q.data != nullptr && index_q.dtype == DType::BF16 && index_q.is_contiguous() &&
                index_q.ne[0] == kIndexDim && index_q.ne[1] == geometry.index_heads && index_q.ne[2] == columns,
            "index queries disagree with the batch");
    require(pooled_pages.data != nullptr && pooled_pages.dtype == DType::BF16 && pooled_pages.is_contiguous() &&
                pooled_pages.ne[0] == kIndexDim / geometry.ratio,
            "pooled pages must be the layer's BF16 plane");
    require(output.selected != nullptr && output.counts != nullptr, "selection outputs are missing");
    const Layout l = layout(geometry, columns, max_context);
    require(scratch != nullptr && scratch_bytes >= l.total, "scratch is too small");

    const std::int32_t top_blocks = geometry.budget / geometry.ratio;
    const std::int32_t max_blocks = max_context / geometry.ratio;
    auto* base                    = static_cast<unsigned char*>(scratch);
    auto* scores                  = reinterpret_cast<float*>(base);
    auto* state                   = reinterpret_cast<std::uint32_t*>(base + l.scores);
    auto* arrivals                = state;
    auto* histograms              = state + l.group;

    // Dynamic shared memory: candidates (aliasing the histogram) and the column bitmap.
    const std::size_t smem = kCandidateBytes + sizeof(unsigned) * 4 * static_cast<std::size_t>(ceil_div(max_blocks + 1, 128));
    reserve_shared(qsa_select_global_kernel, smem, g_global_reserved);

    const auto* index      = static_cast<const bf16*>(index_q.data);
    const auto* pooled     = static_cast<const bf16*>(pooled_pages.data);
    const auto* tables     = static_cast<const std::int32_t*>(batch.block_tables.data);
    const int table_stride = static_cast<int>(batch.block_tables.ne[0]);
    const auto* rows       = static_cast<const std::int32_t*>(batch.table_rows.data);
    const auto* positions  = static_cast<const std::int32_t*>(batch.positions.data);
    const int sms          = multiprocessors();
    const int max_steps    = ceil_div(max_blocks + 1, kScoreStepBlocks);
    const bool mma = allow_mma_score && batch.width >= 4 && (kPagedKVPageSize % geometry.ratio) == 0 &&
                     (kIndexDim / geometry.ratio) % 8 == 0 &&
                     (reinterpret_cast<std::uintptr_t>(index) & 3U) == 0U;
    const bool mma_vector = (reinterpret_cast<std::uintptr_t>(pooled) & 15U) == 0U &&
                            (spaces.pooled == nullptr ||
                             (reinterpret_cast<std::uintptr_t>(spaces.pooled) & 15U) == 0U);
    // Wide rows take the register-tiled kernel when a block's key is 16-byte aligned in one page.
    const bool wide = !mma && batch.width >= kWideMinWidth && geometry.ratio <= kPagedKVPageSize &&
                      (kIndexDim / geometry.ratio) % 8 == 0;
    if (wide) { reserve_shared(qsa_score_wide_kernel, kWideSharedBytes, g_wide_reserved); }
    for (std::int32_t begin = 0; begin < columns; begin += l.group) {
        const std::int32_t n       = std::min(l.group, columns - begin);
        const int slices           = std::clamp(std::min(ceil_div(max_blocks, kSliceBlocks), ceil_div(std::int64_t{16} * sms, n)),
                                                1, kMaxSlices);
        const int pass_clear_words = (slices > 1 && max_blocks > top_blocks) ? (l.group + n * kScoreBins) : 0;
        std::uint32_t* pass_state  = pass_clear_words > 0 ? state : nullptr;
        if (mma) {
            const int eff_width   = std::min(batch.width, n);
            const int tile        = eff_width <= 4    ? 4
                                    : eff_width <= 8  ? 8
                                    : eff_width <= 16 ? 16
                                    : eff_width <= 32 ? 32
                                    : eff_width <= 64 ? 64
                                                      : 128;
            // launch_mma_score dispatches Tile <= 64 (max 512 threads/CTA); split wider passes into
            // 64-column tiles so every column in the group is scored.
            const ScoreTiles mma_tiles = score_tiles(begin, n, batch.width, std::min(tile, 64));
            if (mma_tiles.count <= kMmaMaxTiles) {
                const int mma_key_tiles = std::max(1, ceil_div(max_blocks + 1, kMmaKeyTile));
                const int mma_columns   = mma_key_tiles * kMmaKeyTile;
                const int partitions    = ceil_div(max_blocks + 1, kSliceBlocks);
                const int stages =
                    (tile >= 32 && mma_columns <= 8192) || (tile <= 16 && mma_columns >= 8192) || tile == 128 ? 2 : 1;
                const int resident = tile == 128                  ? 1
                                     : (stages == 2 || tile >= 32) ? 2
                                     : tile == 16                  ? 4
                                                                   : 5;
                const int tasks       = mma_tiles.count * mma_key_tiles;
                const int workers     = std::max(1, std::min(resident * sms, tasks));
                const bool retain     = tile == 128 && partitions >= 6 && partitions <= 9;
                if (mma_vector) {
                    launch_mma_score<true>(tile, stages, retain, workers, stream, index, geometry.index_heads,
                                           pooled, tables, table_stride, rows, positions, tree_rows, batch.width,
                                           geometry.ratio, top_blocks, begin, n, mma_tiles, scores, l.stride,
                                           pass_state, pass_clear_words, spaces);
                } else {
                    launch_mma_score<false>(tile, stages, false, workers, stream, index, geometry.index_heads,
                                            pooled, tables, table_stride, rows, positions, tree_rows, batch.width,
                                            geometry.ratio, top_blocks, begin, n, mma_tiles, scores, l.stride,
                                            pass_state, pass_clear_words, spaces);
                }
            } else {
                const ScoreTiles tiles = score_tiles(begin, n, batch.width);
                const int ctas         = std::clamp(ceil_div(std::int64_t{kScoreCtasPerSm} * sms, tiles.count), 1,
                                                    std::max(1, ceil_div(max_steps, kScoreWarps)));
                CUDA_CHECK(pdl::launch_consumer({dim3(static_cast<unsigned>(ctas), static_cast<unsigned>(tiles.count)),
                                                 dim3(kScoreThreads), 0, stream},
                                                qsa_score_kernel, index, geometry.index_heads, pooled, tables,
                                                table_stride, rows, positions, tree_rows, batch.width, geometry.ratio,
                                                top_blocks, begin, tiles, scores, l.stride, pass_state, pass_clear_words,
                                                spaces));
            }
        } else if (wide) {
            const ScoreTiles tiles = score_tiles(begin, n, batch.width);
            const int ctas = std::clamp(ceil_div(std::int64_t{kWideCtasPerSm} * sms, tiles.count), 1,
                                        std::max(1, ceil_div(max_blocks + 1, kWideBlocks)));
            CUDA_CHECK(pdl::launch_consumer(
                {dim3(static_cast<unsigned>(ctas), static_cast<unsigned>(tiles.count)), dim3(kWideThreads),
                 kWideSharedBytes, stream},
                qsa_score_wide_kernel, index, geometry.index_heads, pooled, tables, table_stride, rows, positions,
                tree_rows, batch.width, geometry.ratio, top_blocks, begin, tiles, scores, l.stride, pass_state,
                pass_clear_words, spaces));
        } else {
            const ScoreTiles tiles = score_tiles(begin, n, batch.width);
            const int ctas = std::clamp(ceil_div(std::int64_t{kScoreCtasPerSm} * sms, tiles.count), 1,
                                        std::max(1, ceil_div(max_steps, kScoreWarps)));
            CUDA_CHECK(pdl::launch_consumer({dim3(static_cast<unsigned>(ctas), static_cast<unsigned>(tiles.count)),
                                             dim3(kScoreThreads), 0, stream},
                                            qsa_score_kernel, index, geometry.index_heads, pooled, tables,
                                            table_stride, rows, positions, tree_rows, batch.width, geometry.ratio,
                                            top_blocks, begin, tiles, scores, l.stride, pass_state, pass_clear_words,
                                            spaces));
        }
        CUDA_CHECK(pdl::launch_consumer({dim3(static_cast<unsigned>(slices), static_cast<unsigned>(n)),
                                         dim3(kSelectThreads), smem, stream},
                                        qsa_select_global_kernel, positions, tree_rows, batch.width, geometry.ratio,
                                        top_blocks, begin, static_cast<const float*>(scores), l.stride, histograms,
                                        arrivals, output.selected, output.counts));
    }
}

} // namespace infernix::ops::detail
