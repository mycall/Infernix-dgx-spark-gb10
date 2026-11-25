// Exact projections over GGUF block weights (include/infernix/ops/ggml_linear.h).
//
// encode_kernel writes each column once as A16 limbs (ops/ggml/ggml_dp4a.cuh: the column's
// exponent, then per 32-element slice its three limb planes and its X sum, with the optional input
// block permutation). rows_kernel gives each output row a group of `Lanes` lanes (4 to 32, so
// that every lane has several of the row's slices whatever K); a lane decodes its slices once per
// pass of up to kColumns columns and adds their exact terms into one int128 per column; the
// group's partial sums are added exactly (order cannot matter) and its first lane rounds. (Every
// rows CTA encoding a decode call's column into shared memory instead of encode_kernel measured
// 4 % slower end to end: 98.0 -> 94.0 tok/s, design §19.3.25.)
//
// Calls of kMmaMinColumns..kMmaMaxColumns columns (MTP verification) run mma_rows_kernel on tensor
// cores; calls of kTileMinColumns or more (prefill) first decode each segment's weights once into
// a plane (plane_kernel, workspace kPlaneBytes) and run the tensor-core tiles over it. Every route
// returns the same integers.

#include "ops/ggml/ggml_bf16_tile.cuh"
#include "ops/ggml/ggml_linear_a8.h"
#include "ops/ggml/ggml_linear_internal.cuh"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace infernix::ops {
namespace {

constexpr int kThreads        = kGgmlThreads;
constexpr int kWarps          = kGgmlWarps;
constexpr int kEncodeThreads  = 256;
constexpr int kColumns        = kGgmlColumns;
constexpr int kTileMinColumns = 32;
// The narrowest call of the BF16 wide route. GB10 (48 SMs) retune 2026-10-29: like the 8-bit wide route
// it is slower than the exact route's wide A16 tiles until the 128-column tile is filled (gdn_qkvz
// T=32: 774.0 us against exact's 343.9; T=64: 771.9 against 454.6), and from 128 columns it wins
// (861.0 against 1037.2), so it starts at its tile width rather than at the exact route's.
constexpr int kBf16WideMinColumns = 128;
constexpr int kEncodeColumns  = 2048;
// GB10 (24 MiB L2): 16 MiB decoded weight plane keeps each pass's plane L2-resident alongside the
// encoded activation while reducing workspace by 48 MiB compared to the 64 MiB RTX 5090 setting.
constexpr std::size_t kPlaneBytes = 16ULL << 20;

// Encodes one column with the whole CTA (Threads threads): its exponent (returned to every thread),
// then its limbs and slice sums into `column`. `warp_key` and `broadcast` are shared scratch.
template <int Threads>
__device__ int encode_column(const std::uint16_t* __restrict__ v, int k, const std::int32_t* __restrict__ perm,
                             int perm_block, std::int32_t* __restrict__ column, unsigned* warp_key, int* broadcast) {
    constexpr int kEncodeWarps = Threads / 32;
    const auto source = [&](int j) { return perm == nullptr ? j : perm[j / perm_block] * perm_block + j % perm_block; };
    unsigned key = 0;
    for (int j = threadIdx.x; j < k; j += Threads) { key = max(key, static_cast<unsigned>(v[j] & 0x7FFFU)); }
    key = __reduce_max_sync(0xFFFFFFFFU, key);
    if (threadIdx.x % 32 == 0) { warp_key[threadIdx.x / 32] = key; }
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned m = 0;
        for (int w = 0; w < kEncodeWarps; ++w) { m = max(m, warp_key[w]); }
        *broadcast = m >= 0x7F80U ? canon::kA16NonFinite : max(static_cast<int>(m >> 7), 1);
    }
    __syncthreads();
    const int e = *broadcast;
    const int groups = k / 4, slices = k / 32;
    const auto value = [&](int j) { return e == canon::kA16NonFinite ? 0 : canon::a16_value(v[source(j)], e); };
    for (int base = 0; base < groups; base += Threads) {
        const int g = base + static_cast<int>(threadIdx.x);
        int sum     = 0;
        if (g < groups) { sum = fast::store_group(column, g, value(4 * g), value(4 * g + 1), value(4 * g + 2), value(4 * g + 3)); }
        sum += __shfl_xor_sync(0xFFFFFFFFU, sum, 1);
        sum += __shfl_xor_sync(0xFFFFFFFFU, sum, 2);
        sum += __shfl_xor_sync(0xFFFFFFFFU, sum, 4);
        if (g < groups && g % 8 == 0) { column[slices * fast::kSliceWords + g / 8] = sum; }
    }
    return e;
}

template <int Threads>
__device__ int encode_column_vec(const std::uint16_t* __restrict__ v, int k, const std::int32_t* __restrict__ perm,
                                 int perm_block, std::int32_t* __restrict__ column, unsigned* warp_key, int* broadcast) {
    constexpr int kEncodeWarps = Threads / 32;
    constexpr int kKeep        = 4;
    const auto source = [&](int j) { return perm == nullptr ? j : perm[j / perm_block] * perm_block + j % perm_block; };
    const int chunks  = k / 8;
    const auto load   = [&](int c) { return *reinterpret_cast<const uint4*>(v + source(8 * c)); };
    const auto max8   = [](const uint4& u) {
        const unsigned m0 = max(u.x & 0x7FFFU, (u.x >> 16) & 0x7FFFU), m1 = max(u.y & 0x7FFFU, (u.y >> 16) & 0x7FFFU);
        const unsigned m2 = max(u.z & 0x7FFFU, (u.z >> 16) & 0x7FFFU), m3 = max(u.w & 0x7FFFU, (u.w >> 16) & 0x7FFFU);
        return max(max(m0, m1), max(m2, m3));
    };
    uint4 raw[kKeep];
    unsigned key = 0;
#pragma unroll
    for (int m = 0; m < kKeep; ++m) {
        const int c = m * Threads + static_cast<int>(threadIdx.x);
        raw[m]      = c < chunks ? load(c) : make_uint4(0, 0, 0, 0);
        key         = max(key, max8(raw[m]));
    }
    for (int c = kKeep * Threads + static_cast<int>(threadIdx.x); c < chunks; c += Threads) { key = max(key, max8(load(c))); }
    key = __reduce_max_sync(0xFFFFFFFFU, key);
    if (threadIdx.x % 32 == 0) { warp_key[threadIdx.x / 32] = key; }
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned m = 0;
        for (int w = 0; w < kEncodeWarps; ++w) { m = max(m, warp_key[w]); }
        *broadcast = m >= 0x7F80U ? canon::kA16NonFinite : max(static_cast<int>(m >> 7), 1);
    }
    __syncthreads();
    const int e = *broadcast;
    const int slices = k / 32;
    const auto value = [&](unsigned bits) {
        return e == canon::kA16NonFinite ? 0 : canon::a16_value(static_cast<std::uint16_t>(bits), e);
    };
    const auto encode = [&](int c, const uint4& u) {
        int sum = fast::store_group(column, 2 * c, value(u.x & 0xFFFFU), value(u.x >> 16), value(u.y & 0xFFFFU),
                                    value(u.y >> 16));
        sum += fast::store_group(column, 2 * c + 1, value(u.z & 0xFFFFU), value(u.z >> 16), value(u.w & 0xFFFFU),
                                 value(u.w >> 16));
        return sum;
    };
    const auto finish = [&](int c, int sum) {
        sum += __shfl_xor_sync(0xFFFFFFFFU, sum, 1);
        sum += __shfl_xor_sync(0xFFFFFFFFU, sum, 2);
        if (c < chunks && c % 4 == 0) { column[slices * fast::kSliceWords + c / 4] = sum; }
    };
#pragma unroll
    for (int m = 0; m < kKeep; ++m) {
        if (m * Threads >= chunks) { break; }
        const int c = m * Threads + static_cast<int>(threadIdx.x);
        finish(c, c < chunks ? encode(c, raw[m]) : 0);
    }
    for (int base = kKeep * Threads; base < chunks; base += Threads) {
        const int c = base + static_cast<int>(threadIdx.x);
        finish(c, c < chunks ? encode(c, load(c)) : 0);
    }
    return e;
}

__global__ void __launch_bounds__(kEncodeThreads)
    encode_kernel(const std::uint16_t* x, int k, const std::int32_t* __restrict__ perm, int perm_block,
                  std::int32_t* __restrict__ encoded, std::int32_t* __restrict__ emax, int vec) {
    __shared__ unsigned warp_key[kEncodeThreads / 32];
    __shared__ int broadcast;
    pdl::enter();
    const int t = blockIdx.x;
    const std::uint16_t* v  = x + static_cast<std::size_t>(t) * k;
    std::int32_t* column    = encoded + static_cast<std::size_t>(t) * fast::encoded_column_words(k);
    const int e = vec != 0 ? encode_column_vec<kEncodeThreads>(v, k, perm, perm_block, column, warp_key, &broadcast)
                           : encode_column<kEncodeThreads>(v, k, perm, perm_block, column, warp_key, &broadcast);
    if (threadIdx.x == 0) { emax[t] = e; }
}

constexpr int kMmaRows       = kGgmlMmaRows;
constexpr int kMmaWarps      = kGgmlMmaWarps;
constexpr int kMmaMinColumns = kGgmlMmaMinColumns;
constexpr int kMmaMaxColumns = kGgmlMmaMaxColumns;

static_assert(kMmaWarps > 1, "the mma rows' warps reduce through shared memory");
using MmaPartial = unsigned long long[kMmaWarps - 1][4][2][32];

enum : int { kPlain = 0, kStaged = 1 };

template <QType F, bool OutF32, int Stage>
__device__ __forceinline__ void mma_rows_body(const GroupSlot& seg, int row_bytes, int slices,
                                              const std::int32_t* encoded, const std::int32_t* emax,
                                              int columns, MmaPartial& partial,
                                              ggml::mma_rows::StagedRows<F>* staged = nullptr) {
    constexpr int Warps        = kMmaWarps;
    constexpr int kBlockSlices = ggml::tile::kBlockSlices<F>;
    const std::uint8_t* weights = seg.weights;
    const int rows = seg.rows, n = seg.n, out_offset = seg.out_offset;
    void* const out = seg.out;
    const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
    const int g = lane / 4, t = lane % 4;
    const int row0 = seg.block * kMmaRows;
    const std::uint8_t* wg  = weights + static_cast<std::size_t>(min(row0 + g, rows - 1)) * row_bytes;
    const std::uint8_t* wg8 = weights + static_cast<std::size_t>(min(row0 + g + 8, rows - 1)) * row_bytes;
    const std::size_t column_words = static_cast<std::size_t>(slices) * (fast::kSliceWords + 1);
    const std::int32_t* bcol = encoded + static_cast<std::size_t>(min(g, columns - 1)) * column_words + 2 * t;
    const std::int32_t* xs0 = encoded + static_cast<std::size_t>(min(2 * t, columns - 1)) * column_words +
                              static_cast<std::size_t>(slices) * fast::kSliceWords;
    const std::int32_t* xs1 = encoded + static_cast<std::size_t>(min(2 * t + 1, columns - 1)) * column_words +
                              static_cast<std::size_t>(slices) * fast::kSliceWords;
    ggml::I128 acc[4];
    if constexpr (Stage == kStaged) {
        const auto row = [&](int r) { return weights + static_cast<std::size_t>(min(row0 + r, rows - 1)) * row_bytes; };
        ggml::mma_rows::issue_first_unit<F>(staged[warp], row, lane, warp, slices);
        pdl::wait_for_dependencies();
        ggml::mma_rows::accumulate_staged<F>(acc, staged[warp], row, bcol, xs0, xs1, t, g, lane, warp, slices, Warps, true);
    } else {
        pdl::wait_for_dependencies();
        ggml::mma_rows::accumulate<F>(acc, wg, wg8, bcol, xs0, xs1, t, warp, slices / kBlockSlices, Warps);
    }
    ggml::mma_rows::reduce<Warps>(acc, partial, warp, lane);
    if (warp > 0) { return; }
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int row = row0 + g + 8 * (q >> 1), c = 2 * t + (q & 1);
        if (row < rows && c < columns) {
            const std::size_t at = static_cast<std::size_t>(c) * n + out_offset + row;
            if constexpr (OutF32) {
                static_cast<float*>(out)[at] = ggml::output_f32(acc[q], emax[c], F);
            } else {
                static_cast<std::uint16_t*>(out)[at] = ggml::output_bf16(acc[q], emax[c], F);
            }
        }
    }
}

// Four resident CTAs (128 threads) are intentional where the staged-row arena fits: on the 23.5 KiB
// formats, four blocks fit in 102,400 B (1,024 B driver-reserved per block) and the register cap
// 65536 / (128 * 4) = 128 binds at the same count, so ptxas keeps an 8-32 B live-set frame instead
// of dropping to three CTAs (the sibling int8 grouped-attention schedule measured a +15% regression
// from that relaxation). Q6_K stages 37.9 KiB per CTA, capping physical residency at two CTAs/SM;
// clamping the hint to the shared-memory fit raises Q6_K's register budget to 255 and empties its
// frame without losing occupancy.
constexpr std::size_t kSmSharedBytes        = 102400;
constexpr std::size_t kDriverSharedPerBlock = 1024;

constexpr int mma_rows_min_blocks(std::size_t staged_bytes) {
    const std::size_t block_smem =
        ((sizeof(MmaPartial) + staged_bytes + 127) & ~std::size_t{127}) + kDriverSharedPerBlock;
    const int max_by_smem = static_cast<int>(kSmSharedBytes / block_smem);
    return max_by_smem < 4 ? max_by_smem : 4;
}

template <QType F, bool OutF32, int Stage = kStaged>
__global__ void __launch_bounds__(
    kMmaWarps * 32,
    mma_rows_min_blocks(Stage == kStaged ? sizeof(ggml::mma_rows::StagedRows<F>) * kMmaWarps : 0))
    mma_rows_kernel(RowsGroup grp, int row_bytes, int slices, const std::int32_t* encoded,
                    const std::int32_t* emax, int columns) {
    __shared__ MmaPartial partial;
    __shared__ ggml::mma_rows::StagedRows<F> staged[kMmaWarps];
    mma_rows_body<F, OutF32, Stage>(group_slot(grp), row_bytes, slices, encoded, emax, columns, partial, staged);
}

template <QType F0, QType F1, bool OutF32>
__global__ void __launch_bounds__(
    kMmaWarps * 32,
    mma_rows_min_blocks(std::max(sizeof(ggml::mma_rows::StagedRows<F0>),
                                 sizeof(ggml::mma_rows::StagedRows<F1>)) *
                        kMmaWarps))
    mma_rows_mixed_kernel(RowsGroup grp, int slices, const std::int32_t* encoded,
                          const std::int32_t* emax, int columns) {
    __shared__ MmaPartial partial;
    __shared__ union {
        ggml::mma_rows::StagedRows<F0> s0[kMmaWarps];
        ggml::mma_rows::StagedRows<F1> s1[kMmaWarps];
    } staged;
    const GroupSlot seg = group_slot(grp);
    if (seg.format == 0) {
        mma_rows_body<F0, OutF32, kStaged>(seg, seg.row_bytes, slices, encoded, emax, columns, partial, staged.s0);
    } else {
        mma_rows_body<F1, OutF32, kStaged>(seg, seg.row_bytes, slices, encoded, emax, columns, partial, staged.s1);
    }
}

template <QType F>
inline constexpr bool kMmaBatched = F == QType::GGML_IQ3_S || F == QType::GGML_IQ3_XXS || F == QType::GGML_IQ2_S;

inline bool mma_batched_format(QType f) {
    return f == QType::GGML_IQ3_S || f == QType::GGML_IQ3_XXS || f == QType::GGML_IQ2_S;
}

constexpr int kMmaBatchColumns = 64;

template <QType F, bool OutF32>
void launch_narrow_format(RowsGroup g, int k, const Columns& in, int columns, cudaStream_t stream) {
    const int row_bytes = static_cast<int>(k / ggml_block_elems(F) * ggml_block_bytes(F));
    const int slices    = k / 32;
    int total_rows      = 0;
    for (int i = 0; i < g.count; ++i) { total_rows += g.rows[i]; }
    const int min_mma_cols = (total_rows >= 8192 && (kMmaBatched<F> || F == QType::GGML_Q6_K)) ? 1 : kMmaMinColumns;
    const bool mma       = columns >= min_mma_cols && columns <= kMmaMaxColumns && slices >= 8;
    const bool mma_batch = !mma && columns > kMmaMaxColumns && columns <= kMmaBatchColumns && slices >= 8;
    const int lanes      = ggml_lanes_for(slices);
    const int per_cta    = mma || mma_batch ? kMmaRows : kWarps * (32 / lanes);
    g.first[0]           = 0;
    for (int i = 0; i < g.count; ++i) { g.first[i + 1] = g.first[i] + (g.rows[i] + per_cta - 1) / per_cta; }
    const dim3 grid(g.first[g.count]);
    if (mma || mma_batch) {
        const int batches              = mma_batch ? (columns + kMmaMaxColumns - 1) / kMmaMaxColumns : 1;
        const int span                 = (columns + batches - 1) / batches;
        const std::size_t column_words = static_cast<std::size_t>(slices) * (fast::kSliceWords + 1);
        const std::size_t out_bytes    = OutF32 ? 4 : 2;
        for (int first = 0; first < columns; first += span) {
            const int count = std::min(span, columns - first);
            if (count == columns) {
                CUDA_CHECK(pdl::launch_consumer({grid, dim3(kMmaWarps * 32), 0, stream},
                                                mma_rows_kernel<F, OutF32, kStaged>, g, row_bytes, slices,
                                                in.encoded, in.emax, columns));
                continue;
            }
            RowsGroup part = g;
            for (int i = 0; i < part.count; ++i) {
                part.out[i] = static_cast<std::byte*>(g.out[i]) + static_cast<std::size_t>(first) * g.n[i] * out_bytes;
            }
            CUDA_CHECK(pdl::launch_consumer({grid, dim3(kMmaWarps * 32), 0, stream},
                                            mma_rows_kernel<F, OutF32, kStaged>, part, row_bytes, slices,
                                            in.encoded + static_cast<std::size_t>(first) * column_words,
                                            in.emax + first, count));
        }
        ggml_check_launch("mma rows");
        return;
    }
#define INFERNIX_GGML_LANES(LN)                                                                                        \
    case LN:                                                                                                           \
        if (columns == 1) {                                                                                            \
            CUDA_CHECK(pdl::launch_consumer({grid, dim3(kThreads), 0, stream}, rows_kernel<F, OutF32, LN, 1>,          \
                                            g, row_bytes, slices, in.encoded, in.emax, columns));                      \
        } else if (columns > kColumns && columns <= 2 * kColumns) {                                                    \
            CUDA_CHECK(pdl::launch_consumer({grid, dim3(kThreads), 0, stream},                                         \
                                            rows_kernel<F, OutF32, LN, 2 * kColumns>, g, row_bytes, slices,             \
                                            in.encoded, in.emax, columns));                                             \
        } else {                                                                                                       \
            CUDA_CHECK(pdl::launch_consumer({grid, dim3(kThreads), 0, stream}, rows_kernel<F, OutF32, LN, kColumns>,    \
                                            g, row_bytes, slices, in.encoded, in.emax, columns));                       \
        }                                                                                                              \
        break;
    switch (lanes) {
        INFERNIX_GGML_LANES(4)
        INFERNIX_GGML_LANES(8)
        INFERNIX_GGML_LANES(16)
        INFERNIX_GGML_LANES(32)
    default:
        throw std::logic_error("ggml_linear: lanes");
    }
#undef INFERNIX_GGML_LANES
    ggml_check_launch("rows");
}

bool mixed_format(QType f) {
    return f == QType::GGML_Q4_K || f == QType::GGML_Q5_K || f == QType::GGML_Q6_K || f == QType::GGML_IQ4_XS;
}

template <QType F0, QType F1>
void launch_mixed_pair(RowsGroup g, int k, const Columns& in, int columns, cudaStream_t stream) {
    g.first[0] = 0;
    for (int i = 0; i < g.count; ++i) { g.first[i + 1] = g.first[i] + (g.rows[i] + kMmaRows - 1) / kMmaRows; }
    const dim3 grid(g.first[g.count]);
    const int slices               = k / 32;
    const int batches              = (columns + kMmaMaxColumns - 1) / kMmaMaxColumns;
    const int span                 = (columns + batches - 1) / batches;
    const std::size_t column_words = static_cast<std::size_t>(slices) * (fast::kSliceWords + 1);
    for (int first = 0; first < columns; first += span) {
        const int count = std::min(span, columns - first);
        if (count == columns) {
            CUDA_CHECK(pdl::launch_consumer({grid, dim3(kMmaWarps * 32), 0, stream},
                                            mma_rows_mixed_kernel<F0, F1, false>, g, slices, in.encoded, in.emax, columns));
            continue;
        }
        RowsGroup part = g;
        for (int i = 0; i < part.count; ++i) {
            part.out[i] = static_cast<std::byte*>(g.out[i]) + static_cast<std::size_t>(first) * g.n[i] * 2;
        }
        CUDA_CHECK(pdl::launch_consumer({grid, dim3(kMmaWarps * 32), 0, stream},
                                        mma_rows_mixed_kernel<F0, F1, false>, part, slices,
                                        in.encoded + static_cast<std::size_t>(first) * column_words,
                                        in.emax + first, count));
    }
    ggml_check_launch("mma rows mixed");
}

bool launch_mixed(QType f0, QType f1, const RowsGroup& g, int k, const Columns& in, int columns, cudaStream_t stream) {
#define INFERNIX_GGML_PAIR(A, B)                                                                                       \
    if (f0 == QType::A && f1 == QType::B) {                                                                            \
        launch_mixed_pair<QType::A, QType::B>(g, k, in, columns, stream);                                              \
        return true;                                                                                                   \
    }
    INFERNIX_GGML_PAIR(GGML_Q4_K, GGML_Q5_K)
    INFERNIX_GGML_PAIR(GGML_Q4_K, GGML_Q6_K)
    INFERNIX_GGML_PAIR(GGML_Q4_K, GGML_IQ4_XS)
    INFERNIX_GGML_PAIR(GGML_Q5_K, GGML_Q6_K)
    INFERNIX_GGML_PAIR(GGML_Q5_K, GGML_IQ4_XS)
    INFERNIX_GGML_PAIR(GGML_Q6_K, GGML_IQ4_XS)
#undef INFERNIX_GGML_PAIR
    return false;
}

template <bool OutF32>
void launch_narrow(QType format, GgmlLayout layout, const RowsGroup& g, int k, const Columns& in, int columns,
                   cudaStream_t stream) {
    if (layout == GgmlLayout::Tiles16) {
        ggml_launch_tiled(format, OutF32, g, k, in, columns, stream);
        return;
    }
#define INFERNIX_GGML_NARROW(Q)                                                                                        \
    case Q:                                                                                                            \
        launch_narrow_format<Q, OutF32>(g, k, in, columns, stream);                                                    \
        break;
    switch (format) {
        INFERNIX_GGML_FORMATS(INFERNIX_GGML_NARROW)
    default:
        throw std::invalid_argument("ggml_linear: segment format is not a ggml block format");
    }
#undef INFERNIX_GGML_NARROW
}

// Column c's sum of x over each 32-element slice (input element j at x[perm(j)]), in element order:
// the K-quants' min term for the BF16 tiles. One CTA per column.
__global__ void __launch_bounds__(128)
    bf16_xsum_kernel(const std::uint16_t* x, int k, const std::int32_t* __restrict__ perm,
                     int perm_block, float* __restrict__ xsum) {
    const int c = static_cast<int>(blockIdx.x), slices = k / 32;
    const std::uint16_t* column = x + static_cast<std::size_t>(c) * k;
    for (int s = static_cast<int>(threadIdx.x); s < slices; s += static_cast<int>(blockDim.x)) {
        float sum = 0.0F;
        for (int e = 0; e < 32; ++e) {
            const int j   = 32 * s + e;
            const int src = perm == nullptr ? j : perm[j / perm_block] * perm_block + j % perm_block;
            sum += __uint_as_float(static_cast<unsigned>(column[src]) << 16);
        }
        xsum[static_cast<std::size_t>(c) * slices + s] = sum;
    }
}

// The ggml-bf16-prefill tiles over one slab's plane (ggml_bf16_tile.cuh), every column at once.
struct Bf16Input {
    const std::uint16_t* x  = nullptr;
    int k                   = 0;
    const std::int32_t* perm = nullptr;
    int perm_block          = 0;
    const float* xsum       = nullptr; // the K-quants' slice sums
};

template <ggml::bf16::Scaling S, bool OutF32>
void launch_bf16_scaling(const GgmlSegment& s, const Bf16Input& in, int columns, void* out, int n,
                         const std::uint8_t* plane, cudaStream_t stream) {
    namespace b = ggml::bf16;
    constexpr int kShared = static_cast<int>(sizeof(b::Shared));
    static const bool configured = [] {
        return cudaFuncSetAttribute(b::tile_kernel<S, OutF32>, cudaFuncAttributeMaxDynamicSharedMemorySize, kShared) ==
               cudaSuccess;
    }();
    ggml_require(configured, "the BF16 tile kernel's shared memory exceeds the device's");
    const int tiles = (s.rows + ggml::tile::kTileRows - 1) / ggml::tile::kTileRows;
    const dim3 grid((s.rows + b::kRows - 1) / b::kRows, (columns + b::kColumns - 1) / b::kColumns);
    const b::DenseColumns cols{.x = in.x, .k = in.k, .perm = in.perm, .perm_block = in.perm_block, .xsum = in.xsum,
                               .col0 = 0, .columns = columns};
    b::tile_kernel<S, OutF32><<<grid, b::kThreads, kShared, stream>>>(plane, tiles, s.rows, in.k / 32, cols, out, n,
                                                                     s.out_offset, ggml::kappa_log2(s.format));
    ggml_check_launch("bf16 tile");
}

template <bool OutF32>
void launch_bf16(const GgmlSegment& s, const Bf16Input& in, int columns, void* out, int n, const std::uint8_t* plane,
                 cudaStream_t stream) {
#define INFERNIX_GGML_BF16(Q)                                                                                          \
    case Q:                                                                                                            \
        launch_bf16_scaling<ggml::bf16::kScaling<Q>, OutF32>(s, in, columns, out, n, plane, stream);                   \
        break;
    switch (s.format) {
        INFERNIX_GGML_FORMATS(INFERNIX_GGML_BF16)
    default:
        throw std::invalid_argument("ggml_linear: segment format is not a ggml block format");
    }
#undef INFERNIX_GGML_BF16
}

std::size_t plane_bytes(int rows, int slices) {
    const std::size_t tiles = static_cast<std::size_t>((rows + ggml::tile::kTileRows - 1) / ggml::tile::kTileRows);
    return tiles * static_cast<std::size_t>(slices / ggml::tile::kStageSlices) * sizeof(ggml::tile::CodeStage);
}

struct Slab {
    GgmlSegment segment;
    std::size_t offset;
};

std::vector<std::vector<Slab>> plan_slabs(std::span<const GgmlSegment> segments, int k) {
    const int slices           = k / 32;
    const std::size_t per_tile = plane_bytes(ggml::tile::kTileRows, slices);
    const int max_rows = static_cast<int>(std::max<std::size_t>(kPlaneBytes / per_tile, 1)) * ggml::tile::kTileRows;
    std::vector<std::vector<Slab>> passes(1);
    std::size_t used = 0;
    for (const auto& s : segments) {
        const int row_bytes = static_cast<int>(k / ggml_block_elems(s.format) * ggml_block_bytes(s.format));
        for (int r = 0; r < s.rows; r += max_rows) {
            GgmlSegment slab = s;
            slab.rows        = std::min(max_rows, s.rows - r);
            slab.data        = s.data + static_cast<std::size_t>(r) * row_bytes;
            slab.out_offset  = s.out_offset + r;
            const std::size_t bytes = plane_bytes(slab.rows, slices);
            if (used + bytes > kPlaneBytes && !passes.back().empty()) {
                passes.emplace_back();
                used = 0;
            }
            passes.back().push_back(Slab{slab, used});
            used += (bytes + 255) / 256 * 256;
        }
    }
    return passes;
}

bool wide_call(int k, int columns) { return columns >= kTileMinColumns && (k / 32) % ggml::tile::kStageSlices == 0; }

void linear_bf16(const Tensor& x, std::span<const GgmlSegment> segments, std::span<Tensor* const> outs,
                 const std::int32_t* input_perm, std::int32_t perm_block, WorkspaceArena& workspace, cudaStream_t stream) {
    const int k = x.ne[0], columns = x.ne[1];
    ggml_require(input_perm == nullptr || perm_block % 4 == 0,
                 "the BF16 route reads x in four-element units of the permutation");
    ggml_require(reinterpret_cast<std::uintptr_t>(x.data) % 16 == 0, "the BF16 route reads x in 8-byte units");
    const DType dtype = outs.front()->dtype;
    const auto passes = plan_slabs(segments, k);
    auto* planes      = static_cast<std::uint8_t*>(workspace.alloc(DType::U8, {static_cast<int>(kPlaneBytes)}).data);
    Bf16Input in{.x = static_cast<const std::uint16_t*>(x.data), .k = k, .perm = input_perm, .perm_block = perm_block};
    const bool min_terms = std::any_of(segments.begin(), segments.end(), [](const GgmlSegment& s) {
        return s.format == QType::GGML_Q4_K || s.format == QType::GGML_Q5_K;
    });
    if (min_terms) {
        float* xsum = static_cast<float*>(workspace.alloc(DType::FP32, {k / 32, columns}).data);
        bf16_xsum_kernel<<<columns, 128, 0, stream>>>(in.x, k, input_perm, perm_block, xsum);
        ggml_check_launch("bf16 xsum");
        in.xsum = xsum;
    }
    for (const auto& pass : passes) {
        for (const auto& slab : pass) { ggml_decode_plane(slab.segment, k, planes + slab.offset, stream); }
        for (const auto& slab : pass) {
            const Tensor& out = *outs[static_cast<std::size_t>(slab.segment.output)];
            if (dtype == DType::FP32) {
                launch_bf16<true>(slab.segment, in, columns, out.data, out.ne[0], planes + slab.offset, stream);
            } else {
                launch_bf16<false>(slab.segment, in, columns, out.data, out.ne[0], planes + slab.offset, stream);
            }
        }
    }
}

// A narrow call's launches for its columns [first, first + count), encoded in `in`: the exact
// route's narrow dispatch, shared with ggml_linear_encoded (the producer-encoded input skips the
// encoding launch; the launches and the bits are the same).
void launch_narrow_call(std::span<const GgmlSegment> segments, int k, const Columns& in, int first, int count,
                        std::span<Tensor* const> outs, cudaStream_t stream) {
    const DType dtype           = outs.front()->dtype;
    const std::size_t out_bytes = dtype == DType::FP32 ? 4 : 2;
    const auto column_out       = [&](const GgmlSegment& s) {
        const Tensor& out = *outs[static_cast<std::size_t>(s.output)];
        return static_cast<void*>(static_cast<std::byte*>(out.data) + static_cast<std::size_t>(first) * out.ne[0] *
                                                                     out_bytes);
    };
    const GgmlLayout layout0 = segments.front().layout;
    const bool tiled_pass    = layout0 == GgmlLayout::Tiles16;
    if (dtype == DType::BF16 && count >= (tiled_pass ? 1 : 2) &&
        segments.size() <= static_cast<std::size_t>(RowsGroup::kMax)) {
        QType f0 = segments.front().format, f1 = f0;
        bool ok = true;
        for (const auto& s : segments) {
            const QType f = s.format;
            ok            = ok && mixed_format(f) && s.layout == layout0;
            if (f != f0 && f1 == f0) {
                f1 = f;
            } else if (f != f0 && f != f1) {
                ok = false;
            }
        }
        if (ok && f1 != f0) {
            if (static_cast<int>(f1) < static_cast<int>(f0)) { std::swap(f0, f1); }
            RowsGroup g;
            for (const auto& s : segments) {
                g.data[g.count]       = s.data;
                g.out[g.count]        = column_out(s);
                g.rows[g.count]       = s.rows;
                g.n[g.count]          = outs[static_cast<std::size_t>(s.output)]->ne[0];
                g.out_offset[g.count] = s.out_offset;
                g.row_bytes[g.count]  = static_cast<int>(ggml_row_bytes(s.format, k));
                g.format[g.count]     = s.format == f0 ? 0 : 1;
                ++g.count;
            }
            if (tiled_pass ? ggml_launch_tiled_mixed(f0, f1, g, k, in, count, stream)
                           : launch_mixed(f0, f1, g, k, in, count, stream)) {
                return;
            }
        }
    }
    std::vector<bool> taken(segments.size(), false);
    for (std::size_t i = 0; i < segments.size(); ++i) {
        if (taken[i]) { continue; }
        // The segments of each format and layout, in order of first appearance, share
        // launches of up to RowsGroup::kMax.
        const QType format      = segments[i].format;
        const GgmlLayout layout = segments[i].layout;
        RowsGroup g;
        for (std::size_t j = i; j < segments.size() && g.count < RowsGroup::kMax; ++j) {
            const GgmlSegment& s = segments[j];
            if (taken[j] || s.format != format || s.layout != layout) { continue; }
            taken[j]              = true;
            g.data[g.count]       = s.data;
            g.out[g.count]        = column_out(s);
            g.rows[g.count]       = s.rows;
            g.n[g.count]          = outs[static_cast<std::size_t>(s.output)]->ne[0];
            g.out_offset[g.count] = s.out_offset;
            ++g.count;
        }
        if (dtype == DType::FP32) {
            launch_narrow<true>(format, layout, g, k, in, count, stream);
        } else {
            launch_narrow<false>(format, layout, g, k, in, count, stream);
        }
    }
}

} // namespace

std::size_t ggml_linear_workspace_bytes(std::int32_t k, std::int32_t columns) {
    const auto align    = [](std::size_t v) { return (v + 255) / 256 * 256; };
    const std::size_t t = static_cast<std::size_t>(std::min(columns, kEncodeColumns));
    if (!wide_call(k, columns)) {
        const std::size_t exact = align(fast::encoded_column_words(k) * t * 4) + align(t * 4);
        return columns <= kA8MaxColumns ? std::max(exact, a8_linear_workspace_bytes(k, columns, true)) : exact;
    }
    const std::size_t exact = align(fast::encoded_column_words(k) * t * 4) + align(t * 4);
    const std::size_t bf16  = align(static_cast<std::size_t>(k / 32) * static_cast<std::size_t>(columns) * 4);
    return std::max(std::max(exact, bf16) + kPlaneBytes, a8_wide_linear_workspace_bytes(k, columns));
}

std::size_t ggml_row_bytes(QType format, std::int32_t k) {
    return static_cast<std::size_t>(k) / ggml_block_elems(format) * ggml_block_bytes(format);
}

std::vector<GgmlSegment> ggml_split_segments(std::span<const GgmlSegment> segments, std::int32_t k,
                                             std::span<const std::int32_t> bounds) {
    ggml_require(bounds.size() >= 2 && bounds.front() == 0 && std::is_sorted(bounds.begin(), bounds.end()),
                 "split bounds must ascend from 0");
    std::vector<GgmlSegment> out;
    for (const auto& s : segments) {
        ggml_require(s.out_offset >= 0 && s.out_offset + s.rows <= bounds.back(), "segment exceeds the split rows");
        const std::size_t row_bytes = ggml_row_bytes(s.format, k);
        for (std::size_t i = 0; i + 1 < bounds.size(); ++i) {
            const int lo = std::max(s.out_offset, bounds[i]), hi = std::min(s.out_offset + s.rows, bounds[i + 1]);
            if (lo >= hi) { continue; }
            ggml_require(s.layout != GgmlLayout::Tiles16 || (lo - s.out_offset) % 16 == 0,
                         "a ggml_tiles16_v1 segment splits only at multiples of 16 rows");
            GgmlSegment part = s;
            part.data        = s.data + static_cast<std::size_t>(lo - s.out_offset) * row_bytes;
            part.rows        = hi - lo;
            part.out_offset  = lo - bounds[i];
            part.output      = static_cast<std::int32_t>(i);
            out.push_back(part);
        }
    }
    return out;
}

std::int32_t ggml_encoded_column_words(std::int32_t k) {
    return static_cast<std::int32_t>(fast::encoded_column_words(k));
}

bool ggml_linear_takes_encoded(std::int32_t k, std::int32_t columns, LinearRoutes routes) {
    return columns > 0 && columns <= kEncodeColumns && k % 32 == 0 && !wide_call(k, columns) &&
           !(columns <= kA8MaxColumns && has_route(routes, LinearRoutes::GgmlA8));
}

void ggml_linear_encoded(const GgmlEncodedColumns& x, std::span<const GgmlSegment> segments,
                         std::span<Tensor* const> outs, cudaStream_t stream) {
    const int k = x.k, columns = x.emax.ne[0];
    ggml_require(x.words.data != nullptr && x.words.dtype == DType::I32 && x.words.is_contiguous() &&
                     x.emax.data != nullptr && x.emax.dtype == DType::I32 && x.emax.is_contiguous() &&
                     x.emax.ne[1] == 1 && k > 0 && k % 32 == 0 &&
                     x.words.ne[0] == ggml_encoded_column_words(k) && x.words.ne[1] == columns && x.words.ne[2] == 1,
                 "encoded x must be I32 words [ggml_encoded_column_words(K), T] and exponents [T]");
    ggml_require(!outs.empty() && outs.front() != nullptr, "no outputs");
    const DType dtype = outs.front()->dtype;
    for (const Tensor* o : outs) {
        ggml_require(o != nullptr && o->data != nullptr && (o->dtype == DType::FP32 || o->dtype == DType::BF16) &&
                         o->dtype == dtype && o->is_contiguous() && o->ne[1] == columns,
                     "outputs must be contiguous [N, T] of one dtype, BF16 or FP32");
    }
    ggml_require(columns > 0 && !segments.empty(), "shapes disagree");
    for (const auto& s : segments) {
        ggml_require(s.output >= 0 && static_cast<std::size_t>(s.output) < outs.size(), "segment names no output");
        ggml_require(is_ggml(s.format) && s.data != nullptr && s.rows > 0 && s.out_offset >= 0 &&
                         s.out_offset + s.rows <= outs[static_cast<std::size_t>(s.output)]->ne[0],
                     "segment exceeds its output");
        ggml_require(k % static_cast<int>(ggml_block_elems(s.format)) == 0,
                     "K is not whole blocks of a segment's format");
        ggml_require(s.layout == GgmlLayout::Rows || ggml_tiles16_supported(s.format),
                     "ggml_tiles16_v1 holds Q4_K, Q5_K, Q6_K and IQ4_XS matrices only");
    }
    ggml_require(columns <= kEncodeColumns && !wide_call(k, columns), "encoded x is for the narrow route's widths");
    const Columns in{.encoded = static_cast<const std::int32_t*>(x.words.data),
                     .emax    = static_cast<const std::int32_t*>(x.emax.data)};
    launch_narrow_call(segments, k, in, 0, columns, outs, stream);
}

void ggml_linear(const Tensor& x, std::span<const GgmlSegment> segments, Tensor& out, const std::int32_t* input_perm,
                 std::int32_t perm_block, WorkspaceArena& workspace, cudaStream_t stream, LinearRoutes routes) {
    for (const auto& s : segments) { ggml_require(s.output == 0, "a one-output call's segments name output 0"); }
    Tensor* const outs[] = {&out};
    ggml_linear(x, segments, outs, input_perm, perm_block, workspace, stream, routes);
}

void ggml_linear(const Tensor& x, std::span<const GgmlSegment> segments, std::span<Tensor* const> outs,
                 const std::int32_t* input_perm, std::int32_t perm_block, WorkspaceArena& workspace, cudaStream_t stream,
                 LinearRoutes routes) {
    ggml_require(valid_linear_routes(routes), "unknown linear route flags");
    ggml_require(x.data != nullptr && x.dtype == DType::BF16 && x.is_contiguous(), "x must be contiguous BF16 [K, T]");
    ggml_require(!outs.empty() && outs.front() != nullptr, "no outputs");
    const DType dtype = outs.front()->dtype;
    const int k = x.ne[0], columns = x.ne[1];
    for (const Tensor* o : outs) {
        ggml_require(o != nullptr && o->data != nullptr && (o->dtype == DType::FP32 || o->dtype == DType::BF16) &&
                         o->dtype == dtype && o->is_contiguous() && o->ne[1] == columns,
                     "outputs must be contiguous [N, T] of one dtype, BF16 or FP32");
    }
    ggml_require(columns > 0 && k % 32 == 0 && !segments.empty(), "shapes disagree");
    for (const auto& s : segments) {
        ggml_require(s.output >= 0 && static_cast<std::size_t>(s.output) < outs.size(), "segment names no output");
        ggml_require(is_ggml(s.format) && s.data != nullptr && s.rows > 0 && s.out_offset >= 0 &&
                         s.out_offset + s.rows <= outs[static_cast<std::size_t>(s.output)]->ne[0],
                     "segment exceeds its output");
        ggml_require(k % static_cast<int>(ggml_block_elems(s.format)) == 0, "K is not whole blocks of a segment's format");
        ggml_require(s.layout == GgmlLayout::Rows || ggml_tiles16_supported(s.format),
                     "ggml_tiles16_v1 holds Q4_K, Q5_K, Q6_K and IQ4_XS matrices only");
    }
    ggml_require(input_perm == nullptr || (perm_block > 0 && k % perm_block == 0), "input permutation needs whole blocks");
    const std::size_t out_bytes = dtype == DType::FP32 ? 4 : 2;
    auto scope                  = workspace.scope();
    const bool wide_route       = wide_call(k, columns);
    if (wide_route && columns >= kA8WideMinColumns && has_route(routes, LinearRoutes::GgmlA8Wide)) {
        ggml_linear_a8_wide(x, segments, outs, input_perm, perm_block, workspace, stream);
        return;
    }
    if (wide_route && columns >= kBf16WideMinColumns && has_route(routes, LinearRoutes::GgmlBf16Wide)) {
        linear_bf16(x, segments, outs, input_perm, perm_block, workspace, stream);
        return;
    }
    if (columns <= kA8MaxColumns && has_route(routes, LinearRoutes::GgmlA8)) {
        ggml_linear_a8(x, segments, outs, input_perm, perm_block, workspace, stream);
        return;
    }
    const int batch             = std::min(columns, kEncodeColumns);
    Tensor encoded              = workspace.alloc(DType::I32, {static_cast<int>(fast::encoded_column_words(k)), batch});
    Tensor emax                 = workspace.alloc(DType::I32, {batch});
    const Columns in{.encoded = static_cast<const std::int32_t*>(encoded.data),
                     .emax    = static_cast<const std::int32_t*>(emax.data)};
    int total_rows = 0;
    for (const auto& s : segments) { total_rows += s.rows; }
    const int mma_batch_limit = total_rows >= 2560 ? kTileMinColumns : (kMmaBatchColumns + 1);
    const bool mma_batched_call =
        columns < mma_batch_limit && std::all_of(segments.begin(), segments.end(), [](const GgmlSegment& s) {
            return s.layout == GgmlLayout::Rows && mma_batched_format(s.format);
        });
    const int wide_min_cols = total_rows >= 2560 ? kTileMinColumns : (ggml::tile::kTileColumns);
    const bool wide =
        columns >= wide_min_cols && (k / 32) % ggml::tile::kStageSlices == 0 && !mma_batched_call;
    const bool use_planes = wide && (columns > ggml::tile::kTileColumns || total_rows < 2560);
    std::vector<std::vector<Slab>> passes;
    std::uint8_t* planes = nullptr;
    if (use_planes) {
        passes = plan_slabs(segments, k);
        planes = static_cast<std::uint8_t*>(workspace.alloc(DType::U8, {static_cast<int>(kPlaneBytes)}).data);
    } else {
        passes.emplace_back();
        for (const auto& s : segments) { passes.back().push_back(Slab{s, 0}); }
    }
    const int vec =
        (input_perm == nullptr || perm_block % 8 == 0) && reinterpret_cast<std::uintptr_t>(x.data) % 16 == 0 ? 1 : 0;
    const auto encode_columns = [&](int first, int count) {
        CUDA_CHECK(pdl::launch_consumer({dim3(count), dim3(kEncodeThreads), 0, stream}, encode_kernel,
                                        static_cast<const std::uint16_t*>(x.data) + static_cast<std::size_t>(first) * k,
                                        k, input_perm, perm_block, static_cast<std::int32_t*>(encoded.data),
                                        static_cast<std::int32_t*>(emax.data), vec));
    };
    if (wide && columns <= batch) {
        encode_columns(0, columns);
        for (const auto& pass : passes) {
            if (use_planes) {
                for (const auto& slab : pass) { ggml_decode_plane(slab.segment, k, planes + slab.offset, stream); }
            }
            for (const auto& slab : pass) {
                const Tensor& out          = *outs[static_cast<std::size_t>(slab.segment.output)];
                const int n                = out.ne[0];
                void* const dst            = out.data;
                const std::uint8_t* splane = use_planes ? planes + slab.offset : nullptr;
                ggml_launch_tiles(slab.segment, k, dtype == DType::FP32, in, columns, dst, n, splane, stream);
            }
        }
        return;
    }
    for (const auto& pass : passes) {
        if (use_planes) {
            for (const auto& slab : pass) { ggml_decode_plane(slab.segment, k, planes + slab.offset, stream); }
        }
        for (int first = 0; first < columns; first += batch) {
            const int count = std::min(batch, columns - first);
            encode_columns(first, count);
            const auto column_out = [&](const GgmlSegment& s) {
                const Tensor& out = *outs[static_cast<std::size_t>(s.output)];
                return static_cast<void*>(static_cast<std::byte*>(out.data) +
                                          static_cast<std::size_t>(first) * out.ne[0] * out_bytes);
            };
            if (wide) {
                for (const auto& slab : pass) {
                    const int n                = outs[static_cast<std::size_t>(slab.segment.output)]->ne[0];
                    const std::uint8_t* splane = use_planes ? planes + slab.offset : nullptr;
                    ggml_launch_tiles(slab.segment, k, dtype == DType::FP32, in, count, column_out(slab.segment), n,
                                      splane, stream);
                }
                continue;
            }
            const GgmlLayout layout0 = pass.front().segment.layout;
            const bool tiled_pass    = layout0 == GgmlLayout::Tiles16;
            if (dtype == DType::BF16 && count >= (tiled_pass ? 1 : 2) &&
                pass.size() <= static_cast<std::size_t>(RowsGroup::kMax)) {
                QType f0 = pass.front().segment.format, f1 = f0;
                bool ok = true;
                for (const auto& slab : pass) {
                    const QType f = slab.segment.format;
                    ok = ok && mixed_format(f) && slab.segment.layout == layout0;
                    if (f != f0 && f1 == f0) {
                        f1 = f;
                    } else if (f != f0 && f != f1) {
                        ok = false;
                    }
                }
                if (ok && f1 != f0) {
                    if (static_cast<int>(f1) < static_cast<int>(f0)) { std::swap(f0, f1); }
                    RowsGroup g;
                    for (const auto& slab : pass) {
                        const GgmlSegment& s  = slab.segment;
                        g.data[g.count]       = s.data;
                        g.out[g.count]        = column_out(s);
                        g.rows[g.count]       = s.rows;
                        g.n[g.count]          = outs[static_cast<std::size_t>(s.output)]->ne[0];
                        g.out_offset[g.count] = s.out_offset;
                        g.row_bytes[g.count]  = static_cast<int>(ggml_row_bytes(s.format, k));
                        g.format[g.count]     = s.format == f0 ? 0 : 1;
                        ++g.count;
                    }
                    if (tiled_pass ? ggml_launch_tiled_mixed(f0, f1, g, k, in, count, stream)
                                   : launch_mixed(f0, f1, g, k, in, count, stream)) {
                        continue;
                    }
                }
            }
            std::vector<bool> taken(pass.size(), false);
            for (std::size_t i = 0; i < pass.size(); ++i) {
                if (taken[i]) { continue; }
                const QType format      = pass[i].segment.format;
                const GgmlLayout layout = pass[i].segment.layout;
                RowsGroup g;
                for (std::size_t j = i; j < pass.size() && g.count < RowsGroup::kMax; ++j) {
                    const GgmlSegment& s = pass[j].segment;
                    if (taken[j] || s.format != format || s.layout != layout) { continue; }
                    taken[j]              = true;
                    g.data[g.count]       = s.data;
                    g.out[g.count]        = column_out(s);
                    g.rows[g.count]       = s.rows;
                    g.n[g.count]          = outs[static_cast<std::size_t>(s.output)]->ne[0];
                    g.out_offset[g.count] = s.out_offset;
                    ++g.count;
                }
                if (dtype == DType::FP32) {
                    launch_narrow<true>(format, layout, g, k, in, count, stream);
                } else {
                    launch_narrow<false>(format, layout, g, k, in, count, stream);
                }
            }
        }
    }
}

} // namespace infernix::ops
