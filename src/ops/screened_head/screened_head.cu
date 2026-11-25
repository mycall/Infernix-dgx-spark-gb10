// Screened heads (include/infernix/ops/screened_head.h): every row of a vocabulary head scored from its
// converter-written 2-bit copy (q2_g64_fp16), and the rows near each column's best rescored exactly
// with the exact head's own arithmetic (B-03; Basalt's two-stage head).
//
// Cost model (RTX 5090, K = 2560): the screen streams 720 bytes per row (2.25 bits a weight) against
// the exact head's 1,360 (q4_g64_fp16 proposal head, 131,072 rows: 94 MB against 178 MB) or 5,120 (a
// BF16 vocabulary head, 248,320 rows: 179 MB against 1.27 GB), at the DRAM roofline. Its arithmetic is
// int8 tensor-core products: each 2-bit code is expanded with two integer instructions per four codes
// (no runtime repacking: the row-split layout's four codes per byte line up with the MMA fragment once
// the K order inside each 16-weight word is permuted, which the activation copy absorbs). The rescore
// reads at most `rescored` rows per column.
//
// Kernels, in launch order:
//   screen     persistent CTAs (as many per SM as the ring's shared memory admits: two at 1..8
//              columns, one at 9..16), four warps splitting K per 16-row item, a cp.async ring of
//              items (11.5 KB each, two deep at 1..8 columns and four at 9..16), the column
//              quantized to int8 per 64-group in shared memory. Per column it keeps the CTA's best
//              kListLength rows and (top form) the online softmax partial of its rows; the logits
//              form writes every screen score.
//   select     one CTA per column: the best screen score M, the rows within the margin of it (at most
//              `rescored`, the best first, at least the ranks), by radix selection on (score, row)
//              keys when the count is out of bounds.
//   rescore    the exact head's arithmetic for each selected row (the top form inside the selection
//              CTA; the logits form in its own launch: q8 rows by warps, BF16 rows by the tensor-core
//              tile over the listed rows).

#include "infernix/ops/screened_head.h"
#include "ops/screened_head/screened_head_launch.h"

#include "core/device.h"
#include "core/layout.h"
#include "core/pdl.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/score_id_order.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>

namespace infernix::ops {
namespace {

using namespace detail::screened_head;
using bf16 = __nv_bfloat16;

constexpr int kGroups        = kHidden / 64;     // 40 weight groups of 64 per row
constexpr int kScreenWarps   = 4;                // K split of an item: warp w takes groups [10w, 10w + 10)
constexpr int kScreenThreads = kScreenWarps * 32;
constexpr int kItemRows      = 16;               // one m16 row tile
// The ring depth pairs with the co-residency the shared memory admits (select_ring below): the deep
// ring prefetches three items and runs one 71 KiB CTA per SM; the narrow ring prefetches one and
// fits two 48 KiB CTAs, whose mutual barrier coverage measured faster on GB10's LPDDR5X UMA
// (2026-10-29: +1.1..2.8% at T=1..8 against the deep ring; a wider K split measured no gain and
// would break the column's bit-exact width invariance, so every form keeps the four-warp group
// spans). The 9..16-column form runs the MMA deeper than it streams and always takes the deep ring.
template <int Tiles, bool NarrowRing>
constexpr int kRingStages = NarrowRing ? 2 : 4;
constexpr int kCodeRowBytes  = kHidden / 4;      // 640: four 2-bit codes per byte
constexpr int kCodePitch     = kCodeRowBytes + 16; // rows g and g + 8 of a quad's word on distinct banks
constexpr int kScaleRowBytes = kGroups * 2;      // 80
constexpr int kStageCodeBytes = kItemRows * kCodePitch;
constexpr int kStageBytes     = kStageCodeBytes + kItemRows * kScaleRowBytes;
constexpr int kColumnPitch    = kHidden + 64;    // a column's int8 copy; 16 banks apart from the next
constexpr int kSelectThreads  = 512;
constexpr int kSelectWarps    = kSelectThreads / 32;
constexpr int kMaxRescored    = 256;
constexpr int kEntriesPerThread = kMaxGrid * kListLength / kSelectThreads; // candidates of one column
static_assert(kGroups % kScreenWarps == 0);
static_assert(kMaxGrid * kListLength % kSelectThreads == 0);
static_assert(kScreenThreads == 128 && kStageBytes % 16 == 0 && kStageCodeBytes % 16 == 0);

__device__ __forceinline__ float negative_infinity() { return __int_as_float(0xff800000); }

template <int Tiles>
__host__ __device__ constexpr int screen_columns() {
    return Tiles * 8;
}

// Dynamic shared memory of the screen kernel: the item ring, the int8 columns, each (column, group)'s
// FP32 scale and twice its code sum, and the warps' partial sums (warp 0's slot then holds the
// item's scores).
template <int Tiles, bool NarrowRing>
constexpr int screen_shared_bytes() {
    constexpr int C = screen_columns<Tiles>();
    return kRingStages<Tiles, NarrowRing> * kStageBytes + C * kColumnPitch + C * kGroups * 8 +
           kScreenWarps * kItemRows * C * 4;
}
static_assert(screen_shared_bytes<1, true>() * 2 <= 102400, "two narrow-ring screen CTAs fit one SM's budget");
static_assert(screen_shared_bytes<1, false>() <= 101376, "the deep-ring screen CTA fits one SM's opt-in");
static_assert(screen_shared_bytes<2, false>() <= 101376, "the screen kernel's ring and columns fit one CTA");
static_assert(screen_shared_bytes<3, true>() <= 101376, "the 24-column narrow-ring screen CTA fits one SM's opt-in");

constexpr int kGmemScreenTiles = 4;
constexpr int kGmemScreenRing  = 4;
constexpr int kGmemScreenCols  = screen_columns<kGmemScreenTiles>();
constexpr int screen_gmem_shared_bytes() {
    return kGmemScreenRing * kStageBytes + kScreenWarps * kItemRows * kGmemScreenCols * 4;
}
static_assert(screen_gmem_shared_bytes() <= 101376, "the 32-column gmem-x deep-ring screen CTA fits one SM's opt-in");

struct ScreenParams {
    const bf16* x;              // [K, T]
    std::uint8_t* xq;           // [T, kColumnPitch] pre-quantized DP4A copy (columns > 16)
    float2* xs;                 // [T, kGroups] per-group (scale, 2 * code sum)
    const std::uint8_t* codes;  // [R, 640]
    const std::uint8_t* scales; // [R, 40] binary16
    std::int32_t rows;          // R, a multiple of 16
    std::int32_t valid_rows;
    std::int32_t columns;       // T
    float* logits;              // [R, T] (the logits form)
    float2* lists;              // [T][grid][kListLength]: (score, row bits)
    float2* partials;           // [T][grid]: (max, sum of exp(score - max)) (the top form)
};

// (v, r) ranks above (w, s): the higher score, on equal scores the lower row.
__device__ __forceinline__ bool ranks_above(float v, int r, float w, int s) {
    return v > w || (v == w && r < s);
}

// The column's quantized int8 copy in the MMA's K order. Inside each 16-weight word of a code row,
// field f of byte b is weight 4b + f (row-split q2 packing); the A fragment takes field f of the four
// bytes as one register (k slots 4t..4t+3, t the thread's index in its quad). So the activation chunk
// of 16 values is stored transposed: byte 4f + b holds x[4b + f]. Columns past T are zero.
template <int Tiles>
__device__ void quantize_columns(const ScreenParams& p, std::uint8_t* xq, float2* xs) {
    constexpr int C = screen_columns<Tiles>();
    // Eight threads per (column, group), eight values each; every pass is uniform, so the
    // eight-lane reductions never diverge.
    for (int item = static_cast<int>(threadIdx.x); item < C * kGroups * 8; item += kScreenThreads) {
        const int column = item / (kGroups * 8), rest = item % (kGroups * 8);
        const int group = rest / 8, octet = rest % 8;
        float v[8];
        if (column < p.columns) {
            const uint4 raw = __ldcg(reinterpret_cast<const uint4*>(p.x + static_cast<std::size_t>(column) * kHidden +
                                                                    group * 64 + octet * 8));
            const bf16* values = reinterpret_cast<const bf16*>(&raw);
#pragma unroll
            for (int j = 0; j < 8; ++j) { v[j] = __bfloat162float(values[j]); }
        } else {
#pragma unroll
            for (int j = 0; j < 8; ++j) { v[j] = 0.0f; }
        }
        float amax = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) { amax = fmaxf(amax, fabsf(v[j])); }
#pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) { amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset)); }
        const float inverse = amax > 0.0f ? 127.0f / amax : 0.0f;
        int sum = 0;
        std::uint8_t* chunk = xq + column * kColumnPitch + group * 64 + (octet / 2) * 16;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int q = max(-127, min(127, __float2int_rn(v[j] * inverse)));
            sum += q;
            const int position = (octet & 1) * 8 + j; // the weight within its 16-value chunk
            chunk[4 * (position % 4) + position / 4] = static_cast<std::uint8_t>(static_cast<std::int8_t>(q));
        }
#pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) { sum += __shfl_xor_sync(0xffffffffu, sum, offset); }
        if (octet == 0) { xs[column * kGroups + group] = make_float2(amax / 127.0f, static_cast<float>(2 * sum)); }
    }
}

__global__ void __launch_bounds__(kScreenThreads) quantize_columns_gmem_kernel(ScreenParams p) {
    const int column = static_cast<int>(blockIdx.x);
    if (column >= p.columns) { return; }
    pdl::enter();
    const bf16* xcol         = p.x + static_cast<std::size_t>(column) * kHidden;
    std::uint8_t* xq_col     = p.xq + static_cast<std::size_t>(column) * kColumnPitch;
    float2* xs_col           = p.xs + static_cast<std::size_t>(column) * kGroups;
    for (int item = static_cast<int>(threadIdx.x); item < kGroups * 8; item += kScreenThreads) {
        const int group = item / 8, octet = item % 8;
        const uint4 raw = __ldcg(reinterpret_cast<const uint4*>(xcol + group * 64 + octet * 8));
        const bf16* values = reinterpret_cast<const bf16*>(&raw);
        float v[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) { v[j] = __bfloat162float(values[j]); }
        float amax = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) { amax = fmaxf(amax, fabsf(v[j])); }
#pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) { amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset)); }
        const float inverse = amax > 0.0f ? 127.0f / amax : 0.0f;
        int sum             = 0;
        std::uint8_t* chunk = xq_col + group * 64 + (octet / 2) * 16;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int q        = max(-127, min(127, __float2int_rn(v[j] * inverse)));
            sum               += q;
            const int position = (octet & 1) * 8 + j;
            chunk[4 * (position % 4) + position / 4] = static_cast<std::uint8_t>(static_cast<std::int8_t>(q));
        }
#pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) { sum += __shfl_xor_sync(0xffffffffu, sum, offset); }
        if (octet == 0) { xs_col[group] = make_float2(amax / 127.0f, static_cast<float>(2 * sum)); }
    }
    pdl::trigger_dependents();
}

// Persistent CTAs over 16-row items i = blockIdx.x + k * gridDim.x. A column's screen
// scores, lists and partials depend on that column alone (its own int8 copy, a fixed grid).
template <int Tiles, bool NarrowRing, bool Logits, bool Softmax>
__global__ void __launch_bounds__(kScreenThreads, 1) screen_kernel(ScreenParams p) {
    extern __shared__ __align__(16) std::uint8_t shared[];
    constexpr int C         = screen_columns<Tiles>();
    constexpr int kRing     = kRingStages<Tiles, NarrowRing>;
    constexpr int kWarps    = kScreenWarps;
    constexpr int kThreads  = kScreenThreads;
    constexpr int kGroupsPerWarp = kGroups / kWarps;
    std::uint8_t* stages   = shared;
    std::uint8_t* xq       = shared + kRing * kStageBytes;
    float2* xs             = reinterpret_cast<float2*>(xq + C * kColumnPitch);
    float* partial         = reinterpret_cast<float*>(xs + C * kGroups);
    const int tid = static_cast<int>(threadIdx.x), warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, t = lane & 3;
    // Eight 16-thread half-warps handle one tile's 8 columns x 16 item rows in parallel after MMA.
    const int sub        = tid >> 4;
    const int r          = tid & 15;
    const int half_shift = (sub & 1) * 16;
    const unsigned hmask = 0xffffu << half_shift;
    const int items = p.rows / kItemRows;
    const int first = static_cast<int>(blockIdx.x), stride = static_cast<int>(gridDim.x);
    const int mine  = first < items ? (items - first + stride - 1) / stride : 0;

    // Item k of this CTA into stage k % kRing: 16 contiguous code rows (into padded rows) and their
    // contiguous scales.
    const auto issue = [&](int k) {
        const std::size_t item = static_cast<std::size_t>(first) + static_cast<std::size_t>(k) * stride;
        std::uint8_t* stage    = stages + (k % kRing) * kStageBytes;
        const std::uint8_t* code_source  = p.codes + item * (kItemRows * kCodeRowBytes);
        const std::uint8_t* scale_source = p.scales + item * (kItemRows * kScaleRowBytes);
        for (int c = tid; c < kItemRows * kCodeRowBytes / 16; c += kThreads) {
            const int row = c / (kCodeRowBytes / 16), part = c % (kCodeRowBytes / 16);
            cp_async<16, Cache::cg>(stage + row * kCodePitch + part * 16, code_source + c * 16);
        }
        for (int c = tid; c < kItemRows * kScaleRowBytes / 16; c += kThreads) {
            cp_async<16, Cache::cg>(stage + kStageCodeBytes + c * 16, scale_source + c * 16);
        }
    };

    // The codes and scales are immutable: the first items stream in before the wait.
#pragma unroll
    for (int k = 0; k < kRing - 1; ++k) {
        if (k < mine) { issue(k); }
        cp_commit();
    }
    pdl::enter_streaming();
    quantize_columns<Tiles>(p, xq, xs);

    // Half-warp sub keeps column (nt * 8 + sub)'s best rows (descending) and its softmax partial
    // replicated identically across its 16 lanes r = 0..15.
    float best[Tiles][kListLength];
    int best_row[Tiles][kListLength];
    float column_max[Tiles];
    float column_sum[Tiles];
#pragma unroll
    for (int nt = 0; nt < Tiles; ++nt) {
#pragma unroll
        for (int j = 0; j < kListLength; ++j) {
            best[nt][j]     = negative_infinity();
            best_row[nt][j] = INT_MAX;
        }
        column_max[nt] = negative_infinity();
        column_sum[nt] = 0.0f;
    }

    for (int k = 0; k < mine; ++k) {
        cp_wait<kRing - 2>();
        __syncthreads(); // item k landed; every thread is done with item k - 1's stage and scores
        if (k + kRing - 1 < mine) { issue(k + kRing - 1); }
        cp_commit();
        const std::uint8_t* stage = stages + (k % kRing) * kStageBytes;
        const auto* scales        = reinterpret_cast<const __half*>(stage + kStageCodeBytes);
        float acc[Tiles][4];
#pragma unroll
        for (int nt = 0; nt < Tiles; ++nt) { acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.0f; }
#pragma unroll 2
        for (int j = 0; j < kGroupsPerWarp; ++j) {
            const int group = warp * kGroupsPerWarp + j;
            // Word t of the group in rows g and g + 8, each 2-bit code moved to its level c + 2 in [0, 3].
            const std::uint32_t w0 =
                *reinterpret_cast<const std::uint32_t*>(stage + g * kCodePitch + group * 16 + t * 4) ^ 0xAAAAAAAAu;
            const std::uint32_t w1 =
                *reinterpret_cast<const std::uint32_t*>(stage + (g + 8) * kCodePitch + group * 16 + t * 4) ^
                0xAAAAAAAAu;
            const float s0 = __half2float(scales[g * kGroups + group]);
            const float s1 = __half2float(scales[(g + 8) * kGroups + group]);
            constexpr std::uint32_t kLevels = 0x03030303u;
#pragma unroll
            for (int nt = 0; nt < Tiles; ++nt) {
                const uint4 b = *reinterpret_cast<const uint4*>(xq + (nt * 8 + g) * kColumnPitch + group * 64 + t * 16);
                int c0, c1, c2, c3;
                mma_s8s8_k32_zero(c0, c1, c2, c3, w0 & kLevels, w1 & kLevels, (w0 >> 2) & kLevels, (w1 >> 2) & kLevels,
                                  b.x, b.y);
                mma_s8(c0, c1, c2, c3, (w0 >> 4) & kLevels, (w1 >> 4) & kLevels, (w0 >> 6) & kLevels,
                       (w1 >> 6) & kLevels, b.z, b.w);
                // sum c x = sum (c + 2) x - 2 sum x, all exact in int32 and float.
                const float2 x0 = xs[(nt * 8 + 2 * t) * kGroups + group];
                const float2 x1 = xs[(nt * 8 + 2 * t + 1) * kGroups + group];
                acc[nt][0] = fmaf(s0 * x0.x, static_cast<float>(c0) - x0.y, acc[nt][0]);
                acc[nt][1] = fmaf(s0 * x1.x, static_cast<float>(c1) - x1.y, acc[nt][1]);
                acc[nt][2] = fmaf(s1 * x0.x, static_cast<float>(c2) - x0.y, acc[nt][2]);
                acc[nt][3] = fmaf(s1 * x1.x, static_cast<float>(c3) - x1.y, acc[nt][3]);
            }
        }
        float* own = partial + warp * kItemRows * C;
#pragma unroll
        for (int nt = 0; nt < Tiles; ++nt) {
            own[g * C + nt * 8 + 2 * t]           = acc[nt][0];
            own[g * C + nt * 8 + 2 * t + 1]       = acc[nt][1];
            own[(g + 8) * C + nt * 8 + 2 * t]     = acc[nt][2];
            own[(g + 8) * C + nt * 8 + 2 * t + 1] = acc[nt][3];
        }
        __syncthreads();
        const int row0 = (first + k * stride) * kItemRows;
        const int row  = row0 + r;
#pragma unroll
        for (int nt = 0; nt < Tiles; ++nt) {
            const int col = nt * 8 + sub;
            const int e   = r * C + col;
            // The K split's fixed summation order: warp 0's slot first, ascending.
            float v = partial[e];
#pragma unroll
            for (int w = 1; w < kWarps; ++w) { v += partial[w * kItemRows * C + e]; }
            if constexpr (Logits) {
                if (col < p.columns) {
                    p.logits[static_cast<std::size_t>(col) * p.rows + row] = v;
                }
            }
            const bool active = col < p.columns && row < p.valid_rows;
            const float score = active ? v : negative_infinity();
            float item_max    = score;
#pragma unroll
            for (int offset = 8; offset > 0; offset >>= 1) {
                item_max = fmaxf(item_max, __shfl_xor_sync(0xffffffffu, item_max, offset));
            }
            const unsigned cand_ballot = __ballot_sync(
                0xffffffffu,
                active && ranks_above(score, row, best[nt][kListLength - 1], best_row[nt][kListLength - 1]));
            unsigned cand_mask = (cand_ballot >> half_shift) & 0xffffu;
            while (cand_mask != 0u) {
                const int src      = __ffs(cand_mask) - 1;
                cand_mask         &= cand_mask - 1u;
                const int src_lane = half_shift + src;
                const float cv     = __shfl_sync(hmask, score, src_lane);
                const int crow     = row0 + src;
                if (ranks_above(cv, crow, best[nt][kListLength - 1], best_row[nt][kListLength - 1])) {
                    bool placed = false;
#pragma unroll
                    for (int i = kListLength - 1; i > 0; --i) {
                        if (!placed) {
                            if (ranks_above(cv, crow, best[nt][i - 1], best_row[nt][i - 1])) {
                                best[nt][i]     = best[nt][i - 1];
                                best_row[nt][i] = best_row[nt][i - 1];
                            } else {
                                best[nt][i]     = cv;
                                best_row[nt][i] = crow;
                                placed          = true;
                            }
                        }
                    }
                    if (!placed) {
                        best[nt][0]     = cv;
                        best_row[nt][0] = crow;
                    }
                }
            }
            if constexpr (Softmax) {
                const float top = fmaxf(column_max[nt], item_max);
                float term      = (item_max > negative_infinity() && active) ? __expf(score - top) : 0.0f;
#pragma unroll
                for (int offset = 8; offset > 0; offset >>= 1) {
                    term += __shfl_xor_sync(0xffffffffu, term, offset);
                }
                if (col < p.columns && item_max > negative_infinity()) {
                    column_sum[nt] = column_sum[nt] * __expf(column_max[nt] - top) + term;
                    column_max[nt] = top;
                }
            }
        }
    }
    pdl::trigger_dependents();
    __syncthreads();
    auto* out_lists = reinterpret_cast<float2*>(partial);
#pragma unroll
    for (int nt = 0; nt < Tiles; ++nt) {
        const int col = nt * 8 + sub;
        if (r == 0 && col < p.columns) {
#pragma unroll
            for (int j = 0; j < kListLength; ++j) {
                out_lists[sub * kListLength + j] = make_float2(best[nt][j], __int_as_float(best_row[nt][j]));
            }
        }
        __syncwarp();
        if (col < p.columns) {
            if (r < kListLength) {
                float2* list = p.lists + (static_cast<std::size_t>(col) * gridDim.x + blockIdx.x) * kListLength;
                list[r]      = out_lists[sub * kListLength + r];
            }
            if constexpr (Softmax) {
                if (r == 0) {
                    p.partials[static_cast<std::size_t>(col) * gridDim.x + blockIdx.x] =
                        make_float2(column_max[nt], column_sum[nt]);
                }
            }
        }
        __syncwarp();
    }
}

// 4-stage deep-ring 32-column screen kernel for T = 17..32 (and 32-col chunks of T > 32): holds
// only the 4-stage 2-bit weight+scale ring and warp partials in SMEM while streaming pre-quantized
// int8 activations from L2 (pinned by hardware access-policy window).
template <bool Logits, bool Softmax>
__global__ void __launch_bounds__(kScreenThreads, 1) screen_kernel_gmem_x(ScreenParams p) {
    extern __shared__ __align__(16) std::uint8_t shared[];
    constexpr int Tiles          = kGmemScreenTiles;
    constexpr int C              = kGmemScreenCols;
    constexpr int kRing          = kGmemScreenRing;
    constexpr int kWarps         = kScreenWarps;
    constexpr int kThreads       = kScreenThreads;
    constexpr int kGroupsPerWarp = kGroups / kWarps;
    std::uint8_t* stages         = shared;
    float* partial               = reinterpret_cast<float*>(stages + kRing * kStageBytes);
    const int tid = static_cast<int>(threadIdx.x), warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, t = lane & 3;
    const int sub        = tid >> 4;
    const int r          = tid & 15;
    const int half_shift = (sub & 1) * 16;
    const unsigned hmask = 0xffffu << half_shift;
    const int items      = p.rows / kItemRows;
    const int first      = static_cast<int>(blockIdx.x), stride = static_cast<int>(gridDim.x);
    const int mine       = first < items ? (items - first + stride - 1) / stride : 0;

    const auto issue = [&](int k) {
        const std::size_t item         = static_cast<std::size_t>(first) + static_cast<std::size_t>(k) * stride;
        std::uint8_t* stage            = stages + (k % kRing) * kStageBytes;
        const std::uint8_t* code_src   = p.codes + item * (kItemRows * kCodeRowBytes);
        const std::uint8_t* scale_src  = p.scales + item * (kItemRows * kScaleRowBytes);
        for (int c = tid; c < kItemRows * kCodeRowBytes / 16; c += kThreads) {
            const int row = c / (kCodeRowBytes / 16), part = c % (kCodeRowBytes / 16);
            cp_async<16, Cache::cg>(stage + row * kCodePitch + part * 16, code_src + c * 16);
        }
        for (int c = tid; c < kItemRows * kScaleRowBytes / 16; c += kThreads) {
            cp_async<16, Cache::cg>(stage + kStageCodeBytes + c * 16, scale_src + c * 16);
        }
    };

#pragma unroll
    for (int k = 0; k < kRing - 1; ++k) {
        if (k < mine) { issue(k); }
        cp_commit();
    }
    pdl::enter_streaming();

    float best[Tiles][kListLength];
    int best_row[Tiles][kListLength];
    float column_max[Tiles];
    float column_sum[Tiles];
#pragma unroll
    for (int nt = 0; nt < Tiles; ++nt) {
#pragma unroll
        for (int j = 0; j < kListLength; ++j) {
            best[nt][j]     = negative_infinity();
            best_row[nt][j] = INT_MAX;
        }
        column_max[nt] = negative_infinity();
        column_sum[nt] = 0.0f;
    }

    for (int k = 0; k < mine; ++k) {
        cp_wait<kRing - 2>();
        __syncthreads();
        if (k + kRing - 1 < mine) { issue(k + kRing - 1); }
        cp_commit();
        const std::uint8_t* stage = stages + (k % kRing) * kStageBytes;
        const auto* scales        = reinterpret_cast<const __half*>(stage + kStageCodeBytes);
        float acc[Tiles][4];
#pragma unroll
        for (int nt = 0; nt < Tiles; ++nt) { acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.0f; }
#pragma unroll 2
        for (int j = 0; j < kGroupsPerWarp; ++j) {
            const int group = warp * kGroupsPerWarp + j;
            const std::uint32_t w0 =
                *reinterpret_cast<const std::uint32_t*>(stage + g * kCodePitch + group * 16 + t * 4) ^ 0xAAAAAAAAu;
            const std::uint32_t w1 =
                *reinterpret_cast<const std::uint32_t*>(stage + (g + 8) * kCodePitch + group * 16 + t * 4) ^
                0xAAAAAAAAu;
            const float s0 = __half2float(scales[g * kGroups + group]);
            const float s1 = __half2float(scales[(g + 8) * kGroups + group]);
            constexpr std::uint32_t kLevels = 0x03030303u;
#pragma unroll
            for (int nt = 0; nt < Tiles; ++nt) {
                const int xcol = nt * 8 + g;
                const uint4 b =
                    xcol < p.columns
                        ? __ldg(reinterpret_cast<const uint4*>(p.xq + static_cast<std::size_t>(xcol) * kColumnPitch +
                                                               group * 64 + t * 16))
                        : make_uint4(0, 0, 0, 0);
                int c0, c1, c2, c3;
                mma_s8s8_k32_zero(c0, c1, c2, c3, w0 & kLevels, w1 & kLevels, (w0 >> 2) & kLevels, (w1 >> 2) & kLevels,
                                  b.x, b.y);
                mma_s8(c0, c1, c2, c3, (w0 >> 4) & kLevels, (w1 >> 4) & kLevels, (w0 >> 6) & kLevels,
                       (w1 >> 6) & kLevels, b.z, b.w);
                const int col0 = nt * 8 + 2 * t;
                const int col1 = col0 + 1;
                const float2 x0 =
                    col0 < p.columns ? __ldg(p.xs + static_cast<std::size_t>(col0) * kGroups + group)
                                     : make_float2(0.0f, 0.0f);
                const float2 x1 =
                    col1 < p.columns ? __ldg(p.xs + static_cast<std::size_t>(col1) * kGroups + group)
                                     : make_float2(0.0f, 0.0f);
                acc[nt][0] = fmaf(s0 * x0.x, static_cast<float>(c0) - x0.y, acc[nt][0]);
                acc[nt][1] = fmaf(s0 * x1.x, static_cast<float>(c1) - x1.y, acc[nt][1]);
                acc[nt][2] = fmaf(s1 * x0.x, static_cast<float>(c2) - x0.y, acc[nt][2]);
                acc[nt][3] = fmaf(s1 * x1.x, static_cast<float>(c3) - x1.y, acc[nt][3]);
            }
        }
        float* own = partial + warp * kItemRows * C;
#pragma unroll
        for (int nt = 0; nt < Tiles; ++nt) {
            own[g * C + nt * 8 + 2 * t]           = acc[nt][0];
            own[g * C + nt * 8 + 2 * t + 1]       = acc[nt][1];
            own[(g + 8) * C + nt * 8 + 2 * t]     = acc[nt][2];
            own[(g + 8) * C + nt * 8 + 2 * t + 1] = acc[nt][3];
        }
        __syncthreads();
        const int row0 = (first + k * stride) * kItemRows;
        const int row  = row0 + r;
#pragma unroll
        for (int nt = 0; nt < Tiles; ++nt) {
            const int col = nt * 8 + sub;
            const int e   = r * C + col;
            float v       = partial[e];
#pragma unroll
            for (int w = 1; w < kWarps; ++w) { v += partial[w * kItemRows * C + e]; }
            if constexpr (Logits) {
                if (col < p.columns) {
                    p.logits[static_cast<std::size_t>(col) * p.rows + row] = v;
                }
            }
            const bool active = col < p.columns && row < p.valid_rows;
            const float score = active ? v : negative_infinity();
            float item_max    = score;
#pragma unroll
            for (int offset = 8; offset > 0; offset >>= 1) {
                item_max = fmaxf(item_max, __shfl_xor_sync(0xffffffffu, item_max, offset));
            }
            const unsigned cand_ballot = __ballot_sync(
                0xffffffffu,
                active && ranks_above(score, row, best[nt][kListLength - 1], best_row[nt][kListLength - 1]));
            unsigned cand_mask = (cand_ballot >> half_shift) & 0xffffu;
            while (cand_mask != 0u) {
                const int src      = __ffs(cand_mask) - 1;
                cand_mask         &= cand_mask - 1u;
                const int src_lane = half_shift + src;
                const float cv     = __shfl_sync(hmask, score, src_lane);
                const int crow     = row0 + src;
                if (ranks_above(cv, crow, best[nt][kListLength - 1], best_row[nt][kListLength - 1])) {
                    bool placed = false;
#pragma unroll
                    for (int i = kListLength - 1; i > 0; --i) {
                        if (!placed) {
                            if (ranks_above(cv, crow, best[nt][i - 1], best_row[nt][i - 1])) {
                                best[nt][i]     = best[nt][i - 1];
                                best_row[nt][i] = best_row[nt][i - 1];
                            } else {
                                best[nt][i]     = cv;
                                best_row[nt][i] = crow;
                                placed          = true;
                            }
                        }
                    }
                    if (!placed) {
                        best[nt][0]     = cv;
                        best_row[nt][0] = crow;
                    }
                }
            }
            if constexpr (Softmax) {
                const float top = fmaxf(column_max[nt], item_max);
                float term      = (item_max > negative_infinity() && active) ? __expf(score - top) : 0.0f;
#pragma unroll
                for (int offset = 8; offset > 0; offset >>= 1) {
                    term += __shfl_xor_sync(0xffffffffu, term, offset);
                }
                if (col < p.columns && item_max > negative_infinity()) {
                    column_sum[nt] = column_sum[nt] * __expf(column_max[nt] - top) + term;
                    column_max[nt] = top;
                }
            }
        }
    }
    pdl::trigger_dependents();
    __syncthreads();
    auto* out_lists = reinterpret_cast<float2*>(partial);
#pragma unroll
    for (int nt = 0; nt < Tiles; ++nt) {
        const int col = nt * 8 + sub;
        if (r == 0 && col < p.columns) {
#pragma unroll
            for (int j = 0; j < kListLength; ++j) {
                out_lists[sub * kListLength + j] = make_float2(best[nt][j], __int_as_float(best_row[nt][j]));
            }
        }
        __syncwarp();
        if (col < p.columns) {
            if (r < kListLength) {
                float2* list = p.lists + (static_cast<std::size_t>(col) * gridDim.x + blockIdx.x) * kListLength;
                list[r]      = out_lists[sub * kListLength + r];
            }
            if constexpr (Softmax) {
                if (r == 0) {
                    p.partials[static_cast<std::size_t>(col) * gridDim.x + blockIdx.x] =
                        make_float2(column_max[nt], column_sum[nt]);
                }
            }
        }
        __syncwarp();
    }
}

// ---- Selection (one CTA of kSelectThreads per column)

struct SelectShared {
    std::uint32_t histogram[256];
    std::uint64_t reduce64[kSelectWarps];
    int reduce32[kSelectWarps];
    std::uint64_t prefix;
    int need;
    int digit_above; // keys above the chosen digit, in this pass
    int digit;
    int rows[kMaxRescored];
    float screen[kMaxRescored];
};

__device__ __forceinline__ std::uint64_t block_max(std::uint64_t value, SelectShared& s) {
    const int lane = static_cast<int>(threadIdx.x) & 31, warp = static_cast<int>(threadIdx.x) >> 5;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const std::uint64_t other = __shfl_xor_sync(0xffffffffu, value, offset);
        value                     = other > value ? other : value;
    }
    if (lane == 0) { s.reduce64[warp] = value; }
    __syncthreads();
    std::uint64_t out = s.reduce64[0];
    for (int w = 1; w < kSelectWarps; ++w) { out = s.reduce64[w] > out ? s.reduce64[w] : out; }
    __syncthreads();
    return out;
}

__device__ __forceinline__ int block_sum(int value, SelectShared& s) {
    const int lane = static_cast<int>(threadIdx.x) & 31, warp = static_cast<int>(threadIdx.x) >> 5;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) { value += __shfl_xor_sync(0xffffffffu, value, offset); }
    if (lane == 0) { s.reduce32[warp] = value; }
    __syncthreads();
    int out = 0;
    for (int w = 0; w < kSelectWarps; ++w) { out += s.reduce32[w]; }
    __syncthreads();
    return out;
}

// The `target`-th largest of the block's distinct nonzero keys (the keys at or above it are exactly
// `target`), by eight passes of 8-bit radix selection from the top byte.
__device__ std::uint64_t select_key(const std::uint64_t (&keys)[kEntriesPerThread], int target, SelectShared& s) {
    const int tid = static_cast<int>(threadIdx.x), lane = tid & 31, warp = tid >> 5;
    if (tid == 0) {
        s.prefix = 0;
        s.need   = target;
    }
    for (int shift = 56; shift >= 0; shift -= 8) {
        if (tid < 256) { s.histogram[tid] = 0; }
        __syncthreads();
        const std::uint64_t prefix = s.prefix;
        const std::uint64_t mask   = shift == 56 ? 0 : ~((std::uint64_t{1} << (shift + 8)) - 1);
#pragma unroll
        for (int i = 0; i < kEntriesPerThread; ++i) {
            if (keys[i] != 0 && (keys[i] & mask) == prefix) {
                atomicAdd(&s.histogram[(keys[i] >> shift) & 0xFF], 1u);
            }
        }
        __syncthreads();
        if (warp == 0) {
            // Lane l owns digits 255 - 8l down to 248 - 8l; scan from the top.
            int counts[8];
            int own = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                counts[i] = static_cast<int>(s.histogram[255 - 8 * lane - i]);
                own += counts[i];
            }
            int above = own; // inclusive scan over lanes (lane 0 = the highest digits)
#pragma unroll
            for (int offset = 1; offset < 32; offset <<= 1) {
                const int other = __shfl_up_sync(0xffffffffu, above, offset);
                if (lane >= offset) { above += other; }
            }
            const int before = above - own;
            const int need   = s.need;
            const unsigned holder = __ballot_sync(0xffffffffu, before < need && above >= need);
            if (lane == __ffs(holder) - 1) {
                int running = before;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    if (running + counts[i] >= need) {
                        s.digit       = 255 - 8 * lane - i;
                        s.digit_above = running;
                        break;
                    }
                    running += counts[i];
                }
            }
        }
        __syncthreads();
        if (tid == 0) {
            s.prefix |= static_cast<std::uint64_t>(s.digit) << shift;
            s.need -= s.digit_above;
        }
        __syncthreads();
    }
    return s.prefix;
}

// Selects column `column`'s rows to rescore from its lists ([grid][kListLength]) into s.rows / s.screen
// in list order; returns their count.
__device__ int select_rows(const float2* lists, int grid, float margin, int ranks, int cap, SelectShared& s) {
    const int tid     = static_cast<int>(threadIdx.x);
    const int entries = grid * kListLength;
    std::uint64_t keys[kEntriesPerThread];
    std::uint64_t top = 0;
#pragma unroll
    for (int i = 0; i < kEntriesPerThread; ++i) {
        const int e = tid + i * kSelectThreads;
        keys[i]     = 0;
        if (e < entries) {
            const float2 entry = __ldcg(lists + e);
            const int row      = __float_as_int(entry.y);
            if (row != INT_MAX) { keys[i] = score_id_order_key(entry.x, row); }
        }
        top = keys[i] > top ? keys[i] : top;
    }
    top = block_max(top, s);
    const float threshold = score_from_order_key(top) - margin;
    int within = 0, valid = 0;
#pragma unroll
    for (int i = 0; i < kEntriesPerThread; ++i) {
        valid += keys[i] != 0;
        within += keys[i] != 0 && score_from_order_key(keys[i]) >= threshold;
    }
    within = block_sum(within, s);
    valid  = block_sum(valid, s);
    const int target = min(min(max(within, ranks), cap), valid);
    // Within the bounds every row within the margin is taken; otherwise the `target` largest keys (the
    // keys at or above the target-th, which is the cut).
    const bool by_margin    = target == within;
    const std::uint64_t cut = by_margin ? 0 : select_key(keys, target, s);
    // Compaction in entry order (entry e = tid + i * kSelectThreads, ranked by (i, tid)).
    int offset = 0;
#pragma unroll
    for (int i = 0; i < kEntriesPerThread; ++i) {
        const bool chosen =
            keys[i] != 0 && (by_margin ? score_from_order_key(keys[i]) >= threshold : keys[i] >= cut);
        const int lane = tid & 31, warp = tid >> 5;
        const unsigned ballot = __ballot_sync(0xffffffffu, chosen);
        if (lane == 0) { s.reduce32[warp] = __popc(ballot); }
        __syncthreads();
        int base = offset;
        for (int w = 0; w < warp; ++w) { base += s.reduce32[w]; }
        int total = 0;
        for (int w = 0; w < kSelectWarps; ++w) { total += s.reduce32[w]; }
        if (chosen) {
            const int position = base + __popc(ballot & ((1u << lane) - 1u));
            if (position < kMaxRescored) {
                s.rows[position]   = id_from_order_key(keys[i]);
                s.screen[position] = score_from_order_key(keys[i]);
            }
        }
        offset += total;
        __syncthreads();
    }
    return min(target, kMaxRescored);
}

// ---- Exact rows (the narrow mapping of projection_fp32: lane l sums chunks l, l + 32, ... of eight
// weights in order with fmaf, then the xor butterfly), for one (row, column).

__device__ __forceinline__ float warp_butterfly(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) { v += __shfl_xor_sync(0xffffffffu, v, offset); }
    return v;
}

// q4_g64_fp16: row r's nibbles at r * padded_k / 2 (low nibble first), its scales at r * padded_k / 64.
__device__ __forceinline__ float q4_row(const std::uint8_t* codes, const __half* scales, std::int64_t row, int padded_k,
                                        const bf16* x, int lane) {
    const auto* w      = reinterpret_cast<const std::uint32_t*>(codes + row * (padded_k / 2));
    const __half* srow = scales + row * (padded_k / 64);
    float sum          = 0.0f;
#pragma unroll
    for (int i = 0; i < kHidden / 8 / 32; ++i) {
        const int chunk            = lane + 32 * i;
        const std::uint32_t packed = __ldg(w + chunk);
        const float scale          = __half2float(srow[chunk / 8]);
        const uint4 xp             = *reinterpret_cast<const uint4*>(x + chunk * 8);
        const bf16* xv             = reinterpret_cast<const bf16*>(&xp);
        float wf[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int u = static_cast<int>((packed >> (4 * j)) & 0xFU);
            wf[j]       = static_cast<float>(u >= 8 ? u - 16 : u) * scale;
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) { sum = fmaf(wf[j], __bfloat162float(xv[j]), sum); }
    }
    return warp_butterfly(sum);
}

// q8_g32_fp16: row r's codes at r * padded_k, its scales at r * padded_k / 32.
__device__ __forceinline__ float q8_row(const std::int8_t* codes, const __half* scales, std::int64_t row, int padded_k,
                                        const bf16* x, int lane) {
    const auto* w      = reinterpret_cast<const uint2*>(codes + row * padded_k);
    const __half* srow = scales + row * (padded_k / 32);
    float sum          = 0.0f;
#pragma unroll
    for (int i = 0; i < kHidden / 8 / 32; ++i) {
        const int chunk    = lane + 32 * i;
        const uint2 packed = __ldg(w + chunk);
        const float scale  = __half2float(srow[chunk / 4]);
        const uint4 xp     = __ldcg(reinterpret_cast<const uint4*>(x + chunk * 8));
        const bf16* xv     = reinterpret_cast<const bf16*>(&xp);
        const auto* wv     = reinterpret_cast<const std::int8_t*>(&packed);
        float wf[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) { wf[j] = static_cast<float>(wv[j]) * scale; }
#pragma unroll
        for (int j = 0; j < 8; ++j) { sum = fmaf(wf[j], __bfloat162float(xv[j]), sum); }
    }
    return warp_butterfly(sum);
}

// Same arithmetic as q8_row with x cached in __shared__ (must use normal load, never __ldcg).
__device__ __forceinline__ float q8_row_smem(const std::int8_t* codes, const __half* scales, std::int64_t row,
                                             int padded_k, const bf16* x_smem, int lane) {
    const auto* w      = reinterpret_cast<const uint2*>(codes + row * padded_k);
    const __half* srow = scales + row * (padded_k / 32);
    float sum          = 0.0f;
#pragma unroll
    for (int i = 0; i < kHidden / 8 / 32; ++i) {
        const int chunk    = lane + 32 * i;
        const uint2 packed = __ldg(w + chunk);
        const float scale  = __half2float(srow[chunk / 4]);
        const uint4 xp     = *reinterpret_cast<const uint4*>(x_smem + chunk * 8);
        const bf16* xv     = reinterpret_cast<const bf16*>(&xp);
        const auto* wv     = reinterpret_cast<const std::int8_t*>(&packed);
        float wf[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) { wf[j] = static_cast<float>(wv[j]) * scale; }
#pragma unroll
        for (int j = 0; j < 8; ++j) { sum = fmaf(wf[j], __bfloat162float(xv[j]), sum); }
    }
    return warp_butterfly(sum);
}

// ---- The top form's selection, rescore and ranks (one CTA per column)

struct TopParams {
    const bf16* x;
    const std::uint8_t* codes; // the q4 head
    const __half* scales;
    std::int32_t padded_k;
    const float2* lists;
    const float2* partials;
    std::int32_t grid;
    std::int32_t columns;
    float margin;
    std::int32_t cap;
    std::int32_t ranks;
    std::int32_t* rows_out;  // [ranks, T]
    float* logprobs_out;     // [ranks, T]
    const std::int32_t* row_ids;
    std::int32_t* tokens;
};

// The normalizer's tail rows, those at their screen scores, stand for their exact logits at s + v / 2:
// the converter's MSE screen leaves e = s + eps with eps about independent of s (the fit's residual is
// orthogonal to its copy), so E[exp(e) | s] = exp(s + v / 2) for a residual of variance v, and exp(s)
// alone undercounts the tail (by 0.3-0.6 nats on flat columns of the 131K-row proposal head). v is the
// mean square of e - s over the rescored rows outside the four best exact logits (the rows a column
// boosts carry residuals the screen attenuates), from at least kTailResidualRows rows, else 0.
constexpr int kTailExcludedRanks  = 4;
constexpr int kTailResidualRows   = 8;
constexpr float kTailResidualMax  = 4.0f;
constexpr int kNormalizerTerms    = 5; // tail partials, rescored exact, rescored screen, residual squares, rows

struct TopShared {
    SelectShared select;
    __align__(16) bf16 x[kHidden];
    float rounded[kMaxRescored];
    float reduce[kNormalizerTerms][kSelectWarps];
    int taken[kMaxRescored];
    int picked[kScreenedHeadMaxRanks];
    float picked_value[kScreenedHeadMaxRanks];
};

__global__ void __launch_bounds__(kSelectThreads) screened_top_kernel(TopParams p) {
    __shared__ TopShared s;
    const int tid = static_cast<int>(threadIdx.x), lane = tid & 31, warp = tid >> 5;
    const int column = static_cast<int>(blockIdx.x);
    pdl::enter();
    for (int i = tid; i < kHidden / 8; i += kSelectThreads) {
        reinterpret_cast<uint4*>(s.x)[i] =
            __ldcg(reinterpret_cast<const uint4*>(p.x + static_cast<std::size_t>(column) * kHidden) + i);
    }
    const int n = select_rows(p.lists + static_cast<std::size_t>(column) * p.grid * kListLength, p.grid, p.margin,
                              p.ranks, p.cap, s.select);
    __syncthreads();
    for (int e = warp; e < n; e += kSelectWarps) {
        const float v = q4_row(p.codes, p.scales, s.select.rows[e], p.padded_k, s.x, lane);
        if (lane == 0) {
            s.rounded[e] = __bfloat162float(__float2bfloat16_rn(v)); // the logit argmax and top_logprobs read
            s.taken[e]   = 0;
        }
    }
    __syncthreads();
    if (warp == 0) {
        // The ranks among the rescored rows: the larger BF16 logit, on ties the lower row. At least
        // kTailExcludedRanks (4) are picked whatever p.ranks, so the normalizer's residual rows (and
        // with it every log-probability) do not depend on the ranks a call asks for.
        const int pick_count = max(kTailExcludedRanks, p.ranks);
        for (int j = 0; j < pick_count; ++j) {
            float v = negative_infinity();
            int row = INT_MAX, at = -1;
            for (int e = lane; e < n; e += 32) {
                if (!s.taken[e] && ranks_above(s.rounded[e], s.select.rows[e], v, row)) {
                    v   = s.rounded[e];
                    row = s.select.rows[e];
                    at  = e;
                }
            }
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, v, offset);
                const int orow = __shfl_xor_sync(0xffffffffu, row, offset);
                const int oat  = __shfl_xor_sync(0xffffffffu, at, offset);
                if (ranks_above(ov, orow, v, row)) {
                    v   = ov;
                    row = orow;
                    at  = oat;
                }
            }
            if (lane == 0) {
                s.picked[j]       = at >= 0 ? row : 0;
                s.picked_value[j] = v;
                if (at >= 0) { s.taken[at] = j < kTailExcludedRanks ? 2 : 1; }
            }
            __syncwarp();
        }
    }
    __syncthreads();
    // The log-normalizer: the CTAs' partials over every row at its screen score, with the rescored
    // rows' screen terms swapped for their exact BF16 logits and the rest raised by exp(v / 2).
    float top = s.picked_value[0];
    for (int c = tid; c < p.grid; c += kSelectThreads) {
        top = fmaxf(top, __ldcg(p.partials + static_cast<std::size_t>(column) * p.grid + c).x);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) { top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, offset)); }
    if (lane == 0) { s.reduce[0][warp] = top; }
    __syncthreads();
    top = s.reduce[0][0];
    for (int w = 1; w < kSelectWarps; ++w) { top = fmaxf(top, s.reduce[0][w]); }
    __syncthreads();
    float terms[kNormalizerTerms] = {};
    for (int c = tid; c < p.grid; c += kSelectThreads) {
        const float2 part = __ldcg(p.partials + static_cast<std::size_t>(column) * p.grid + c);
        if (part.y > 0.0f) { terms[0] += part.y * expf(part.x - top); }
    }
    for (int e = tid; e < n; e += kSelectThreads) {
        terms[1] += expf(s.rounded[e] - top);
        terms[2] += expf(s.select.screen[e] - top);
        if (s.taken[e] != 2) {
            const float residual = s.rounded[e] - s.select.screen[e];
            terms[3] += residual * residual;
            terms[4] += 1.0f;
        }
    }
#pragma unroll
    for (int i = 0; i < kNormalizerTerms; ++i) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) { terms[i] += __shfl_xor_sync(0xffffffffu, terms[i], offset); }
        if (lane == 0) { s.reduce[i][warp] = terms[i]; }
    }
    __syncthreads();
    if (tid == 0) {
        float sums[kNormalizerTerms] = {};
        for (int i = 0; i < kNormalizerTerms; ++i) {
            for (int w = 0; w < kSelectWarps; ++w) { sums[i] += s.reduce[i][w]; }
        }
        const float residual_rows = sums[4];
        const float v = residual_rows >= static_cast<float>(kTailResidualRows)
                            ? fminf(sums[3] / residual_rows, kTailResidualMax)
                            : 0.0f;
        // The tail: every row's screen term less the rescored rows' (sums[0] - sums[2] >= 0 up to rounding).
        const float tail = fmaxf(sums[0] - sums[2], 0.0f) * expf(0.5f * v);
        float total      = sums[1] + tail;
        total                = fmaxf(total, expf(s.picked_value[0] - top)); // at least the pick's own term
        const float normalizer = top + logf(total);
        for (int j = 0; j < p.ranks; ++j) {
            p.rows_out[static_cast<std::size_t>(column) * p.ranks + j]     = s.picked[j];
            p.logprobs_out[static_cast<std::size_t>(column) * p.ranks + j] = s.picked_value[j] - normalizer;
        }
        if (p.tokens != nullptr) { p.tokens[column] = p.row_ids[s.picked[0]]; }
    }
}

// ---- The logits form's selection (one CTA per column) and q8 rescore (a warp per listed row)

struct ListParams {
    const float2* lists;
    std::int32_t grid;
    float margin;
    std::int32_t ranks;
    std::int32_t cap;
    std::int32_t* entries; // [T][cap]
    std::int32_t* counts;  // [T]
};

__global__ void __launch_bounds__(kSelectThreads) screened_list_kernel(ListParams p) {
    __shared__ SelectShared s;
    const int tid = static_cast<int>(threadIdx.x), column = static_cast<int>(blockIdx.x);
    pdl::enter();
    const int n = select_rows(p.lists + static_cast<std::size_t>(column) * p.grid * kListLength, p.grid, p.margin,
                              p.ranks, p.cap, s);
    __syncthreads();
    // Every slot holds a row (the tail repeats the first), so a gathered tile reads only valid rows.
    for (int i = tid; i < p.cap; i += kSelectThreads) {
        p.entries[static_cast<std::size_t>(column) * p.cap + i] = s.rows[i < n ? i : 0];
    }
    if (tid == 0) { p.counts[column] = n; }
}

struct CompactQ8Params {
    const bf16* x;
    const std::int8_t* codes;
    const __half* scales;
    std::int32_t padded_k;
    const float2* lists;
    std::int32_t grid;
    float margin;
    std::int32_t ranks;
    std::int32_t cap;
    std::int32_t* compact_ids; // [T][cap]
    float* compact_logits;     // [T][cap]
    std::int32_t* counts;      // [T]
};

struct CompactQ8Shared {
    SelectShared select;
    __align__(16) bf16 x[kHidden];
};

__global__ void __launch_bounds__(kSelectThreads) screened_compact_q8_kernel(CompactQ8Params p) {
    __shared__ CompactQ8Shared s;
    const int tid = static_cast<int>(threadIdx.x), lane = tid & 31, warp = tid >> 5;
    const int column = static_cast<int>(blockIdx.x);
    pdl::enter();
    for (int i = tid; i < kHidden / 8; i += kSelectThreads) {
        reinterpret_cast<uint4*>(s.x)[i] =
            __ldcg(reinterpret_cast<const uint4*>(p.x + static_cast<std::size_t>(column) * kHidden) + i);
    }
    const int n = select_rows(p.lists + static_cast<std::size_t>(column) * p.grid * kListLength, p.grid, p.margin,
                              p.ranks, p.cap, s.select);
    __syncthreads();
    const std::size_t base = static_cast<std::size_t>(column) * p.cap;
    for (int e = warp; e < n; e += kSelectWarps) {
        const int row = s.select.rows[e];
        const float v = q8_row_smem(p.codes, p.scales, row, p.padded_k, s.x, lane);
        if (lane == 0) {
            p.compact_ids[base + e]    = row;
            p.compact_logits[base + e] = v;
        }
    }
    if (tid == 0) { p.counts[column] = n; }
}

struct Q8RescoreParams {
    const bf16* x;
    const std::int8_t* codes;
    const __half* scales;
    std::int32_t padded_k;
    std::int32_t vocab_rows;
    std::int32_t columns;
    std::int32_t cap;
    const std::int32_t* entries;
    const std::int32_t* counts;
    float* logits;
};

__global__ void __launch_bounds__(kSelectThreads) q8_rescore_kernel(Q8RescoreParams p) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int e    = static_cast<int>(blockIdx.x) * kSelectWarps + (static_cast<int>(threadIdx.x) >> 5);
    pdl::enter();
    const int column = e / p.cap, i = e % p.cap;
    if (column >= p.columns || i >= __ldcg(p.counts + column)) { return; }
    const int row = __ldcg(p.entries + e);
    const float v = q8_row(p.codes, p.scales, row, p.padded_k, p.x + static_cast<std::size_t>(column) * kHidden, lane);
    if (lane == 0) { p.logits[static_cast<std::size_t>(column) * p.vocab_rows + row] = v; }
}

// ---- Host side

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::invalid_argument("screened_head: " + message); }
}

bool aligned16(const void* p) { return reinterpret_cast<std::uintptr_t>(p) % 16 == 0; }

void validate_policy(const ScreenedHeadPolicy& policy) {
    require(std::isfinite(policy.margin) && policy.margin >= 0.0F, "the margin must be finite and non-negative");
    require(policy.rescored >= 32 && policy.rescored <= kMaxRescored && policy.rescored % 32 == 0,
            "rescored rows must be a multiple of 32 in [32, 256]");
}

void validate_x(const Tensor& x) {
    require(x.data != nullptr && x.dtype == DType::BF16 && x.is_contiguous() && x.ne[0] == kHidden && x.ne[2] == 1 &&
                x.ne[3] == 1 && x.ne[1] >= 1 && x.ne[1] <= kScreenedHeadMaxColumns && aligned16(x.data),
            "x must be contiguous, aligned BF16 [2560, T] with T in [1, 64]");
}

void validate_screen(const Weight& screen) {
    require(screen.qtype == QType::Q2_G64_FP16 && screen.layout == QuantLayout::RowSplit && screen.k == kHidden &&
                screen.padded_shape[1] == kHidden && screen.n > 0 && screen.n % kItemRows == 0 &&
                screen.qdata != nullptr && screen.scales != nullptr && aligned16(screen.qdata) &&
                aligned16(screen.scales),
            "the screen must be a row-split q2_g64_fp16 [R, 2560] with R a multiple of 16");
}

// The 1..8-column narrow ring (two co-resident CTAs) is taken on sm_121-class unified-memory parts,
// where it measured faster (2026-10-29, GB10); every other device keeps the deep-ring CTA it was
// tuned with. Fixed per device, so a column's lists do not depend on the call.
bool narrow_ring_device() {
    thread_local int cached = -1;
    if (cached != -1) return cached == 1;
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    int major = 0, minor = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
    cached = (major == 12 && minor == 1) ? 1 : 0;
    return cached == 1;
}

struct ScreenL2PolicyProps {
    std::size_t max_window_bytes = 0;
};

ScreenL2PolicyProps screen_l2_policy_props() {
    thread_local int cached_device = -1;
    thread_local ScreenL2PolicyProps cached{};
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    if (device != cached_device) {
        int max_win = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&max_win, cudaDevAttrMaxAccessPolicyWindowSize, device));
        cached.max_window_bytes = max_win > 0 ? static_cast<std::size_t>(max_win) : 0;
        cached_device           = device;
    }
    return cached;
}

// The screen grid for one (Tiles, NarrowRing) shape: the co-resident CTAs per SM the occupancy
// admits times the SM count, at most kMaxGrid, then down to the item count. The occupancy is judged
// on a representative instantiation (the forms share the shared memory and register footprint), so
// the launch grid the round-robin candidate lists depend on is form-independent.
template <int Tiles, bool NarrowRing>
int screen_grid(int items) {
    thread_local int cached_grid = 0;
    if (cached_grid == 0) {
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        int sms = 0, per_sm = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, screen_kernel<Tiles, NarrowRing, false, true>,
                                                                kScreenThreads, screen_shared_bytes<Tiles, NarrowRing>()));
        cached_grid = std::clamp(sms * std::max(per_sm, 1), 1, kMaxGrid);
    }
    return std::min(cached_grid, items);
}

template <int Tiles, bool NarrowRing, bool Logits, bool Softmax>
void launch_screen(const ScreenParams& p, int grid, cudaStream_t stream) {
    constexpr auto kernel = screen_kernel<Tiles, NarrowRing, Logits, Softmax>;
    constexpr int bytes   = screen_shared_bytes<Tiles, NarrowRing>();
    static const cudaError_t configured =
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    CUDA_CHECK(configured);

    cudaLaunchAttribute attrs[2]{};
    int num_attrs = 0;
    if (pdl::graph_dependent(stream)) {
        attrs[num_attrs].id                                           = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[num_attrs].val.programmaticStreamSerializationAllowed   = 1;
        ++num_attrs;
    }

    // Isolate the hot input/reduction span in L2 so streaming 94-179 MiB of 2-bit codes and
    // scales evicts cleanly without trashing hot decode activations/workspace. (CUDA 13.0 ptxas on
    // sm_121a miscompiles per-instruction createpolicy + extern __shared__ into an invalid odd
    // LDGSTS desc[UR1], so hardware access-policy windows do the streaming isolation cleanly.)
    const ScreenL2PolicyProps l2 = screen_l2_policy_props();
    if (l2.max_window_bytes > 0 && p.x != nullptr && p.columns > 0) {
        const auto addr     = [](const void* ptr) { return reinterpret_cast<std::uintptr_t>(ptr); };
        const auto overlaps = [](std::uintptr_t a0, std::uintptr_t a1, std::uintptr_t b0, std::uintptr_t b1) {
            return a0 < b1 && b0 < a1;
        };
        const std::uintptr_t codes_lo  = addr(p.codes);
        const std::uintptr_t codes_hi  = codes_lo + static_cast<std::size_t>(p.rows) * kCodeRowBytes;
        const std::uintptr_t scales_lo = addr(p.scales);
        const std::uintptr_t scales_hi = scales_lo + static_cast<std::size_t>(p.rows) * kScaleRowBytes;
        const std::size_t max_pin      = std::min<std::size_t>(l2.max_window_bytes, 8ULL << 20);

        std::uintptr_t win_lo = addr(p.x);
        std::uintptr_t win_hi = win_lo + static_cast<std::size_t>(p.columns) * kHidden * sizeof(bf16);
        bool has_win = win_hi > win_lo && (win_hi - win_lo) <= max_pin &&
                       !overlaps(win_lo, win_hi, codes_lo, codes_hi) &&
                       !overlaps(win_lo, win_hi, scales_lo, scales_hi);
        const auto try_merge = [&](const void* ptr, std::size_t span_bytes) {
            if (!has_win || ptr == nullptr || span_bytes == 0) { return; }
            const std::uintptr_t lo     = addr(ptr);
            const std::uintptr_t hi     = lo + span_bytes;
            const std::uintptr_t new_lo = std::min(win_lo, lo);
            const std::uintptr_t new_hi = std::max(win_hi, hi);
            if (new_hi > new_lo && (new_hi - new_lo) <= max_pin && !overlaps(new_lo, new_hi, codes_lo, codes_hi) &&
                !overlaps(new_lo, new_hi, scales_lo, scales_hi)) {
                win_lo = new_lo;
                win_hi = new_hi;
            }
        };
        try_merge(p.lists, static_cast<std::size_t>(p.columns) * grid * kListLength * sizeof(float2));
        if constexpr (Softmax) {
            if (p.partials != nullptr) {
                try_merge(p.partials, static_cast<std::size_t>(p.columns) * grid * sizeof(float2));
            }
        }
        if (has_win && (win_hi - win_lo) <= l2.max_window_bytes) {
            attrs[num_attrs].id                              = cudaLaunchAttributeAccessPolicyWindow;
            attrs[num_attrs].val.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(win_lo);
            attrs[num_attrs].val.accessPolicyWindow.num_bytes = static_cast<std::size_t>(win_hi - win_lo);
            attrs[num_attrs].val.accessPolicyWindow.hitRatio = 1.0F;
            attrs[num_attrs].val.accessPolicyWindow.hitProp  = cudaAccessPropertyNormal;
            attrs[num_attrs].val.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
            ++num_attrs;
        }
    }

    cudaLaunchConfig_t config{};
    config.gridDim          = dim3(grid);
    config.blockDim         = dim3(kScreenThreads);
    config.dynamicSmemBytes = static_cast<std::size_t>(bytes);
    config.stream           = stream;
    config.attrs            = num_attrs > 0 ? attrs : nullptr;
    config.numAttrs         = static_cast<unsigned int>(num_attrs);
    CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, p));
}

template <bool Logits, bool Softmax>
void launch_screen_gmem_x(const ScreenParams& p, int grid, cudaStream_t stream) {
    constexpr auto kernel = screen_kernel_gmem_x<Logits, Softmax>;
    constexpr int bytes   = screen_gmem_shared_bytes();
    static const cudaError_t configured =
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    CUDA_CHECK(configured);

    cudaLaunchAttribute attrs[2]{};
    int num_attrs = 0;
    if (pdl::graph_dependent(stream)) {
        attrs[num_attrs].id                                           = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[num_attrs].val.programmaticStreamSerializationAllowed   = 1;
        ++num_attrs;
    }

    const ScreenL2PolicyProps l2 = screen_l2_policy_props();
    if (l2.max_window_bytes > 0 && p.xq != nullptr && p.columns > 0) {
        const auto addr     = [](const void* ptr) { return reinterpret_cast<std::uintptr_t>(ptr); };
        const auto overlaps = [](std::uintptr_t a0, std::uintptr_t a1, std::uintptr_t b0, std::uintptr_t b1) {
            return a0 < b1 && b0 < a1;
        };
        const std::uintptr_t codes_lo  = addr(p.codes);
        const std::uintptr_t codes_hi  = codes_lo + static_cast<std::size_t>(p.rows) * kCodeRowBytes;
        const std::uintptr_t scales_lo = addr(p.scales);
        const std::uintptr_t scales_hi = scales_lo + static_cast<std::size_t>(p.rows) * kScaleRowBytes;
        const std::size_t max_pin      = std::min<std::size_t>(l2.max_window_bytes, 8ULL << 20);

        std::uintptr_t win_lo = addr(p.xq);
        std::uintptr_t win_hi = win_lo + static_cast<std::size_t>(p.columns) * kColumnPitch;
        bool has_win          = win_hi > win_lo && (win_hi - win_lo) <= max_pin &&
                       !overlaps(win_lo, win_hi, codes_lo, codes_hi) &&
                       !overlaps(win_lo, win_hi, scales_lo, scales_hi);
        const auto try_merge = [&](const void* ptr, std::size_t span_bytes) {
            if (!has_win || ptr == nullptr || span_bytes == 0) { return; }
            const std::uintptr_t lo     = addr(ptr);
            const std::uintptr_t hi     = lo + span_bytes;
            const std::uintptr_t new_lo = std::min(win_lo, lo);
            const std::uintptr_t new_hi = std::max(win_hi, hi);
            if (new_hi > new_lo && (new_hi - new_lo) <= max_pin && !overlaps(new_lo, new_hi, codes_lo, codes_hi) &&
                !overlaps(new_lo, new_hi, scales_lo, scales_hi)) {
                win_lo = new_lo;
                win_hi = new_hi;
            }
        };
        try_merge(p.xs, static_cast<std::size_t>(p.columns) * kGroups * sizeof(float2));
        try_merge(p.lists, static_cast<std::size_t>(p.columns) * grid * kListLength * sizeof(float2));
        if constexpr (Softmax) {
            if (p.partials != nullptr) {
                try_merge(p.partials, static_cast<std::size_t>(p.columns) * grid * sizeof(float2));
            }
        }
        if (has_win && (win_hi - win_lo) <= l2.max_window_bytes) {
            attrs[num_attrs].id                              = cudaLaunchAttributeAccessPolicyWindow;
            attrs[num_attrs].val.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(win_lo);
            attrs[num_attrs].val.accessPolicyWindow.num_bytes = static_cast<std::size_t>(win_hi - win_lo);
            attrs[num_attrs].val.accessPolicyWindow.hitRatio = 1.0F;
            attrs[num_attrs].val.accessPolicyWindow.hitProp  = cudaAccessPropertyNormal;
            attrs[num_attrs].val.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
            ++num_attrs;
        }
    }

    cudaLaunchConfig_t config{};
    config.gridDim          = dim3(grid);
    config.blockDim         = dim3(kScreenThreads);
    config.dynamicSmemBytes = static_cast<std::size_t>(bytes);
    config.stream           = stream;
    config.attrs            = num_attrs > 0 ? attrs : nullptr;
    config.numAttrs         = static_cast<unsigned int>(num_attrs);
    CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, p));
}

int screen_grid_gmem_x(int items) {
    thread_local int cached_grid = 0;
    if (cached_grid == 0) {
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        int sms = 0, per_sm = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, screen_kernel_gmem_x<false, true>,
                                                                kScreenThreads, screen_gmem_shared_bytes()));
        cached_grid = std::clamp(sms * std::max(per_sm, 1), 1, kMaxGrid);
    }
    return std::min(cached_grid, items);
}

template <bool Logits, bool Softmax>
int run_screen(const ScreenParams& p, cudaStream_t stream) {
    const int items = p.rows / kItemRows;
    if (p.columns <= 8) {
        if (narrow_ring_device()) {
            const int grid = screen_grid<1, true>(items);
            launch_screen<1, true, Logits, Softmax>(p, grid, stream);
            return grid;
        }
        const int grid = screen_grid<1, false>(items);
        launch_screen<1, false, Logits, Softmax>(p, grid, stream);
        return grid;
    }
    if (p.columns <= 16) {
        const int grid = screen_grid<2, false>(items);
        launch_screen<2, false, Logits, Softmax>(p, grid, stream);
        return grid;
    }
    if (p.columns <= 24) {
        const int grid = screen_grid<3, true>(items);
        launch_screen<3, true, Logits, Softmax>(p, grid, stream);
        return grid;
    }
    CUDA_CHECK(pdl::launch_consumer({dim3(p.columns), dim3(kScreenThreads), 0, stream},
                                    quantize_columns_gmem_kernel, p));
    const int grid = screen_grid_gmem_x(items);
    for (int col = 0; col < p.columns; col += 32) {
        ScreenParams chunk = p;
        chunk.x            = p.x + static_cast<std::size_t>(col) * kHidden;
        chunk.xq           = p.xq + static_cast<std::size_t>(col) * kColumnPitch;
        chunk.xs           = p.xs + static_cast<std::size_t>(col) * kGroups;
        chunk.columns      = std::min(32, p.columns - col);
        chunk.logits       = p.logits != nullptr ? p.logits + static_cast<std::size_t>(col) * p.rows : nullptr;
        chunk.lists        = p.lists + static_cast<std::size_t>(col) * grid * kListLength;
        chunk.partials     = p.partials != nullptr ? p.partials + static_cast<std::size_t>(col) * grid : nullptr;
        launch_screen_gmem_x<Logits, Softmax>(chunk, grid, stream);
    }
    return grid;
}

// The workspace recipe, shared by the query and the calls.
struct Scratch {
    Tensor xq, xs, lists, partials, entries, counts;
};

template <class Allocator>
Scratch allocate_scratch(Allocator& allocator, std::int32_t columns, std::int32_t cap) {
    Scratch s;
    s.xq       = allocator.alloc(DType::I32, {kColumnPitch / 4, columns}, 256);
    s.xs       = allocator.alloc(DType::FP32, {2 * kGroups, columns}, 256);
    s.lists    = allocator.alloc(DType::FP32, {2 * kListLength, kMaxGrid, columns}, 256);
    s.partials = allocator.alloc(DType::FP32, {2, kMaxGrid, columns}, 256);
    s.entries  = allocator.alloc(DType::I32, {cap, columns}, 256);
    s.counts   = allocator.alloc(DType::I32, {columns}, 256);
    return s;
}

ScreenParams screen_params(const Tensor& x, const Weight& screen, std::int32_t valid_rows, const Scratch& scratch) {
    return {.x          = static_cast<const bf16*>(x.data),
            .xq         = static_cast<std::uint8_t*>(scratch.xq.data),
            .xs         = static_cast<float2*>(scratch.xs.data),
            .codes      = static_cast<const std::uint8_t*>(screen.qdata),
            .scales     = static_cast<const std::uint8_t*>(screen.scales),
            .rows       = screen.n,
            .valid_rows = valid_rows,
            .columns    = x.ne[1],
            .logits     = nullptr,
            .lists      = static_cast<float2*>(scratch.lists.data),
            .partials   = static_cast<float2*>(scratch.partials.data)};
}

bool validate_rescore_head(const Tensor& x, const Tensor& bf16_head, const Weight& head, const Weight& screen,
                           std::int32_t valid_rows, const ScreenedHeadPolicy& policy) {
    validate_policy(policy);
    validate_x(x);
    validate_screen(screen);
    const std::int32_t R = screen.n;
    const bool rows_bf16 = bf16_head.data != nullptr;
    if (rows_bf16) {
        require(head.qdata == nullptr && bf16_head.dtype == DType::BF16 && bf16_head.is_contiguous() &&
                    bf16_head.ne[0] == kHidden && bf16_head.ne[1] == R && bf16_head.ne[2] == 1 && aligned16(bf16_head.data),
                "a BF16 head must be contiguous, aligned rows [R, 2560] (a BF16 tensor [2560, R]) matching the screen");
    } else {
        require(head.qtype == QType::Q8_G32_FP16 && head.layout == QuantLayout::RowSplit && head.n == R &&
                    head.k == kHidden && head.padded_shape[1] % 32 == 0 && head.padded_shape[1] >= kHidden &&
                    head.qhigh == nullptr && head.qdata != nullptr && head.scales != nullptr && aligned16(head.qdata),
                "the head must be BF16 rows or a row-split q8_g32_fp16 [R, 2560] matching the screen");
    }
    require(valid_rows >= 1 && valid_rows <= R, "valid_rows must be in [1, R]");
    return rows_bf16;
}

} // namespace

std::size_t screened_head_workspace_bytes(std::int32_t head_rows, std::int32_t max_columns,
                                          const ScreenedHeadPolicy& policy) {
    validate_policy(policy);
    if (head_rows <= 0 || max_columns < 1 || max_columns > kScreenedHeadMaxColumns) {
        throw std::invalid_argument("screened_head workspace: invalid rows or column interval");
    }
    WorkspaceLayoutBuilder layout;
    (void)allocate_scratch(layout, max_columns, policy.rescored);
    return layout.peak_bytes();
}

int screened_head_screen_grid(std::int32_t head_rows, std::int32_t columns) {
    if (head_rows <= 0 || head_rows % kItemRows != 0 || columns < 1 || columns > kScreenedHeadMaxColumns) {
        throw std::invalid_argument("screened_head screen grid: invalid rows or column count");
    }
    const int items = head_rows / kItemRows;
    if (columns > 24) { return screen_grid_gmem_x(items); }
    if (columns > 16) { return screen_grid<3, true>(items); }
    if (columns > 8) { return screen_grid<2, false>(items); }
    return narrow_ring_device() ? screen_grid<1, true>(items) : screen_grid<1, false>(items);
}

void screened_head_top(const Tensor& x, const Weight& head, const Weight& screen, std::int32_t valid_rows,
                       const ScreenedHeadPolicy& policy, Tensor& rows, Tensor& logprobs, const Tensor& row_ids,
                       Tensor& tokens, WorkspaceArena& workspace, cudaStream_t stream) {
    validate_policy(policy);
    validate_x(x);
    validate_screen(screen);
    const std::int32_t T = x.ne[1], R = screen.n;
    require(head.qtype == QType::Q4_G64_FP16 && head.layout == QuantLayout::RowSplit && head.n == R &&
                head.k == kHidden && head.padded_shape[1] % 64 == 0 && head.padded_shape[1] >= kHidden &&
                head.qhigh == nullptr && head.qdata != nullptr && head.scales != nullptr && aligned16(head.qdata),
            "the head must be a row-split q4_g64_fp16 [R, 2560] matching the screen");
    require(valid_rows >= 1 && valid_rows <= R, "valid_rows must be in [1, R]");
    const std::int32_t ranks = rows.ne[0];
    require(ranks >= 1 && ranks <= kScreenedHeadMaxRanks && rows.data != nullptr && rows.dtype == DType::I32 &&
                rows.is_contiguous() && rows.ne[1] == T && logprobs.data != nullptr && logprobs.dtype == DType::FP32 &&
                logprobs.is_contiguous() && logprobs.ne[0] == ranks && logprobs.ne[1] == T,
            "rows and logprobs must be contiguous I32 / FP32 [ranks, T] with ranks in [1, 16]");
    require(ranks <= policy.rescored, "the policy must rescore at least the ranks");
    const bool mapped = row_ids.data != nullptr;
    require(mapped == (tokens.data != nullptr), "row_ids and tokens are both given or both empty");
    if (mapped) {
        require(row_ids.dtype == DType::I32 && row_ids.numel() == R && tokens.dtype == DType::I32 && tokens.numel() == T,
                "row_ids must be I32 [R] and tokens I32 [T]");
    }
    auto scope          = workspace.scope();
    const Scratch scratch = allocate_scratch(workspace, T, policy.rescored);
    const int grid      = run_screen<false, true>(screen_params(x, screen, valid_rows, scratch), stream);
    const TopParams top{.x            = static_cast<const bf16*>(x.data),
                        .codes        = static_cast<const std::uint8_t*>(head.qdata),
                        .scales       = static_cast<const __half*>(head.scales),
                        .padded_k     = head.padded_shape[1],
                        .lists        = static_cast<const float2*>(scratch.lists.data),
                        .partials     = static_cast<const float2*>(scratch.partials.data),
                        .grid         = grid,
                        .columns      = T,
                        .margin       = policy.margin,
                        .cap          = policy.rescored,
                        .ranks        = ranks,
                        .rows_out     = static_cast<std::int32_t*>(rows.data),
                        .logprobs_out = static_cast<float*>(logprobs.data),
                        .row_ids      = mapped ? static_cast<const std::int32_t*>(row_ids.data) : nullptr,
                        .tokens       = mapped ? static_cast<std::int32_t*>(tokens.data) : nullptr};
    CUDA_CHECK(pdl::launch_consumer({dim3(T), dim3(kSelectThreads), 0, stream}, screened_top_kernel, top));
}

void screened_head_logits(const Tensor& x, const Tensor& bf16_head, const Weight& head, const Weight& screen,
                          std::int32_t valid_rows, const ScreenedHeadPolicy& policy, Tensor& logits,
                          WorkspaceArena& workspace, cudaStream_t stream) {
    const bool rows_bf16 = validate_rescore_head(x, bf16_head, head, screen, valid_rows, policy);
    const std::int32_t T = x.ne[1], R = screen.n;
    require(logits.data != nullptr && logits.dtype == DType::FP32 && logits.is_contiguous() && logits.ne[0] == R &&
                logits.ne[1] == T && logits.ne[2] == 1 && logits.ne[3] == 1,
            "logits must be contiguous FP32 [R, T]");
    auto scope            = workspace.scope();
    const Scratch scratch = allocate_scratch(workspace, T, policy.rescored);
    ScreenParams params   = screen_params(x, screen, valid_rows, scratch);
    params.logits         = static_cast<float*>(logits.data);
    params.partials       = nullptr;
    const int grid        = run_screen<true, false>(params, stream);
    const ListParams list{.lists   = static_cast<const float2*>(scratch.lists.data),
                          .grid    = grid,
                          .margin  = policy.margin,
                          .ranks   = 1,
                          .cap     = policy.rescored,
                          .entries = static_cast<std::int32_t*>(scratch.entries.data),
                          .counts  = static_cast<std::int32_t*>(scratch.counts.data)};
    CUDA_CHECK(pdl::launch_consumer({dim3(T), dim3(kSelectThreads), 0, stream}, screened_list_kernel, list));
    if (rows_bf16) {
        detail::screened_head::bf16_rescore(x, bf16_head.data, R, T, list.entries, list.counts, policy.rescored,
                                            static_cast<float*>(logits.data), stream);
    } else {
        const Q8RescoreParams q8{.x          = static_cast<const bf16*>(x.data),
                                 .codes      = static_cast<const std::int8_t*>(head.qdata),
                                 .scales     = static_cast<const __half*>(head.scales),
                                 .padded_k   = head.padded_shape[1],
                                 .vocab_rows = R,
                                 .columns    = T,
                                 .cap        = policy.rescored,
                                 .entries    = list.entries,
                                 .counts     = list.counts,
                                 .logits     = static_cast<float*>(logits.data)};
        const int warps = T * policy.rescored;
        CUDA_CHECK(pdl::launch_consumer({dim3((warps + kSelectWarps - 1) / kSelectWarps), dim3(kSelectThreads), 0, stream},
                                        q8_rescore_kernel, q8));
    }
}

void screened_head_compact(const Tensor& x, const Tensor& bf16_head, const Weight& head, const Weight& screen,
                           std::int32_t valid_rows, const ScreenedHeadPolicy& policy, Tensor& compact_ids,
                           Tensor& compact_logits, Tensor& compact_counts, WorkspaceArena& workspace,
                           cudaStream_t stream) {
    const bool rows_bf16 = validate_rescore_head(x, bf16_head, head, screen, valid_rows, policy);
    const std::int32_t T = x.ne[1];
    require(compact_ids.data != nullptr && compact_ids.dtype == DType::I32 && compact_ids.is_contiguous() &&
                compact_ids.ne[0] == policy.rescored && compact_ids.ne[1] == T && compact_ids.ne[2] == 1 &&
                compact_ids.ne[3] == 1,
            "compact_ids must be contiguous I32 [rescored, T]");
    require(compact_logits.data != nullptr && compact_logits.dtype == DType::FP32 && compact_logits.is_contiguous() &&
                compact_logits.ne[0] == policy.rescored && compact_logits.ne[1] == T && compact_logits.ne[2] == 1 &&
                compact_logits.ne[3] == 1,
            "compact_logits must be contiguous FP32 [rescored, T]");
    require(compact_counts.data != nullptr && compact_counts.dtype == DType::I32 && compact_counts.is_contiguous() &&
                compact_counts.ne[0] == T && compact_counts.ne[1] == 1 && compact_counts.ne[2] == 1 &&
                compact_counts.ne[3] == 1,
            "compact_counts must be contiguous I32 [T]");
    auto scope                   = workspace.scope();
    const Scratch scratch        = allocate_scratch(workspace, T, policy.rescored);
    ScreenParams params          = screen_params(x, screen, valid_rows, scratch);
    params.logits                = nullptr;
    params.partials              = nullptr;
    const int grid               = run_screen<false, false>(params, stream);
    const std::int32_t min_ranks = std::min<std::int32_t>(2, valid_rows);
    if (rows_bf16) {
        const ListParams list{.lists   = static_cast<const float2*>(scratch.lists.data),
                              .grid    = grid,
                              .margin  = policy.margin,
                              .ranks   = min_ranks,
                              .cap     = policy.rescored,
                              .entries = static_cast<std::int32_t*>(compact_ids.data),
                              .counts  = static_cast<std::int32_t*>(compact_counts.data)};
        CUDA_CHECK(pdl::launch_consumer({dim3(T), dim3(kSelectThreads), 0, stream}, screened_list_kernel, list));
        detail::screened_head::bf16_rescore_compact(x, bf16_head.data, T, list.entries, list.counts, policy.rescored,
                                                    static_cast<float*>(compact_logits.data), stream);
    } else {
        const CompactQ8Params q8{.x              = static_cast<const bf16*>(x.data),
                                 .codes          = static_cast<const std::int8_t*>(head.qdata),
                                 .scales         = static_cast<const __half*>(head.scales),
                                 .padded_k       = head.padded_shape[1],
                                 .lists          = static_cast<const float2*>(scratch.lists.data),
                                 .grid           = grid,
                                 .margin         = policy.margin,
                                 .ranks          = min_ranks,
                                 .cap            = policy.rescored,
                                 .compact_ids    = static_cast<std::int32_t*>(compact_ids.data),
                                 .compact_logits = static_cast<float*>(compact_logits.data),
                                 .counts         = static_cast<std::int32_t*>(compact_counts.data)};
        CUDA_CHECK(pdl::launch_consumer({dim3(T), dim3(kSelectThreads), 0, stream}, screened_compact_q8_kernel, q8));
    }
}

} // namespace infernix::ops
