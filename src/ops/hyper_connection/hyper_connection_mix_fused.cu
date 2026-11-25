// The fused route of hyper_connection_mix: two kernels instead of norm, the down projection,
// gates, the up projection and collapse (docs/maintainer/qwen3_8-flash-next-design.md §8.3 K1a/K1b,
// §19.3.3 Phase 1b), for Q8_G32_FP16 RowSplit weights (Dense8) and BF16 contiguous weights (the
// bit-exact artifact). The weight codec only changes how a lane's 8 weights are read; the order of
// every product and sum is the codec-independent schedule below.
//
// K1a hc_down_partials: one CTA per (stream s, 8 rows of W_down), a row per warp. The CTA stages its
// stream's slice of R for up to 8 columns, reduces each column's sum of squares in a fixed order
// (per-thread sums over its elements in index order, a warp butterfly, warps in order), forms
// Rn = R * inv * (1 + w) in FP32 shared memory and computes the slice's dot products in the
// lane-chunk order (lane l owns weights k = 8l + 256m, fmaf in k order, a warp butterfly). Outputs:
// FP32 partial[s][row][t]; the row-group-0 CTA of each stream writes inv[s][t].
//
// K1b hc_up_collapse: one CTA per 16 features d, a warp per d. Every CTA sums the partials in the
// fixed order s = 0..S-1 into z, forms m = SiLU(z / S) in FP32 shared memory (CTA 0 also writes
// inject = 2 sigmoid(z[rank + s] / S)), computes the S rows (s*H + d) of W_up against m in the same
// lane-chunk order, and writes x[d] = bf16((1/S) * sum_s sigmoid(u_s) * Rn_s) with Rn_s formed in
// FP32 from R, inv and w.
//
// No intermediate is rounded to BF16 (op development §6.1): Rn, z, m and u stay FP32. Each column is
// computed independently in a fixed order, so a column's result does not depend on how many
// columns share the call. Calls of two or more columns run the same arithmetic with Strata v4's read
// scheduling (hc_down_partials_v4 / hc_up_collapse_v4 below).

#include "ops/hyper_connection/hyper_connection_mix_fused.h"

#include "ops/hyper_connection/hyper_connection_inject.cuh"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"

#include "core/pdl.cuh"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace infernix::ops::detail {
namespace {

using bf16 = __nv_bfloat16;

constexpr int kStreams      = 4;
constexpr int kHidden       = 2560;
constexpr int kChunks       = kHidden / 256; // a lane's 8-weight chunks per stream slice
constexpr int kMaxRank      = 512;
constexpr int kDownRows     = 8; // rows of W_down per CTA (a row per warp)
constexpr int kDownThreads  = kDownRows * 32;
// Columns per pass. Decode verifies at most 4 columns (draft-tokens 3 + 1), so 8 wasted half the
// dot-product/norm iterations behind `if (c < pass)` guards and kept acc[] large. Halving it drops
// the Q8 kernel from 124 to 96 registers/thread (zero spills), letting more blocks/SM fit at narrow
// widths without changing the grid (no extra weight streams contending the unified fabric). Bit-exact
// by the fixed-order guarantee above. Model A/B (interleaved, best config): +0.655% mean, 3/4 rounds.
constexpr int kPassColumns  = 4;
constexpr int kSimtMaxColumns = 4;
constexpr int kUpFeatures   = 16; // features d per CTA (a d per warp)
constexpr int kUpThreads    = kUpFeatures * 32;
constexpr std::size_t kDownSharedBytes = static_cast<std::size_t>(kPassColumns) * kHidden * sizeof(float);

void check_launch(cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("hyper_connection_mix ") + what + ": " + cudaGetErrorString(error));
    }
}

__device__ __forceinline__ float sigmoidf(float x) { return 1.0F / (1.0F + expf(-x)); }

// The norm weight w is BF16, or FP32 (NVIDIA FP4-Local's trained norms); every kernel forms 1 + w in
// FP32 from the represented value.
__device__ __forceinline__ float norm_value(bf16 w) { return __bfloat162float(w); }
__device__ __forceinline__ float norm_value(float w) { return w; }

__device__ __forceinline__ void bf16x8_bits_to_floats(const uint4& bits, float (&v)[8]) {
    const float2 a = bf16x2_bits_to_float2(bits.x), b = bf16x2_bits_to_float2(bits.y);
    const float2 c = bf16x2_bits_to_float2(bits.z), d = bf16x2_bits_to_float2(bits.w);
    v[0] = a.x, v[1] = a.y, v[2] = b.x, v[3] = b.y, v[4] = c.x, v[5] = c.y, v[6] = d.x, v[7] = d.y;
}

// Eight consecutive norm weights as the v4 kernels stage them: one 16-byte vector of BF16 words, or
// two of FP32 words.
template <class Norm>
struct NormGroup;

template <>
struct NormGroup<bf16> {
    uint4 bits;
    __device__ static NormGroup load(const bf16* p) { return {load_ldg<uint4>(p)}; }
    __device__ static NormGroup zero() { return {make_uint4(0U, 0U, 0U, 0U)}; }
    __device__ void floats(float (&w)[8]) const { bf16x8_bits_to_floats(bits, w); }
};

template <>
struct NormGroup<float> {
    float4 low, high;
    __device__ static NormGroup load(const float* p) { return {load_ldg<float4>(p), load_ldg<float4>(p + 4)}; }
    __device__ static NormGroup zero() { return {make_float4(0.F, 0.F, 0.F, 0.F), make_float4(0.F, 0.F, 0.F, 0.F)}; }
    __device__ void floats(float (&w)[8]) const {
        w[0] = low.x, w[1] = low.y, w[2] = low.z, w[3] = low.w, w[4] = high.x, w[5] = high.y, w[6] = high.z,
        w[7] = high.w;
    }
};

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) { value += __shfl_xor_sync(0xFFFFFFFFU, value, offset); }
    return value;
}

__device__ __forceinline__ float half_bits_to_float(std::uint16_t bits) {
    return __half2float(__ushort_as_half(static_cast<unsigned short>(bits)));
}

// Eight residual values with the pending inject applied: hc_injected(R, inject, y) element-wise, the
// value hyper_connection_inject would have stored (hyper_connection_inject_mix's fold).
__device__ __forceinline__ uint4 injected8(const uint4& r_bits, const uint4& y_bits, float inject) {
    float r[8], y[8];
    bf16x8_bits_to_floats(r_bits, r);
    bf16x8_bits_to_floats(y_bits, y);
    std::uint32_t out[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const bf16 lo = hc_injected(r[2 * j], inject, y[2 * j]);
        const bf16 hi = hc_injected(r[2 * j + 1], inject, y[2 * j + 1]);
        out[j] = static_cast<std::uint32_t>(__bfloat16_as_ushort(lo)) |
                 (static_cast<std::uint32_t>(__bfloat16_as_ushort(hi)) << 16);
    }
    return make_uint4(out[0], out[1], out[2], out[3]);
}

// q8_g32_fp16 in row_split_k128_v1: int8 codes [rows, padded_k], FP16 scales per 32 codes.
struct Q8Codec {
    struct Params {
        const std::uint8_t* codes;
        const std::uint8_t* scales;
        int padded_k;
    };
    struct Chunk {
        uint2 code;
        float scale;
    };
    __device__ static Chunk load(const Params& p, std::int64_t row, int k, std::uint64_t policy) {
        return {ld_nc_na_evict_first_vec<uint2>(p.codes + row * p.padded_k + k, policy),
                half_bits_to_float(ld_nc_na<std::uint16_t>(p.scales + row * (p.padded_k / 32) * 2 + (k / 32) * 2))};
    }
    __device__ static void decode(const Chunk& c, float (&w)[8]) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            w[i] = static_cast<float>(static_cast<std::int8_t>((i < 4 ? c.code.x : c.code.y) >> ((i & 3) * 8))) *
                   c.scale;
        }
    }
};

// q4_g64_fp16 in row_split_k128_v1: 4-bit signed-offset codes [rows, padded_k / 2], FP16 scales per 64 codes.
struct Q4Codec {
    struct Params {
        const std::uint8_t* codes;
        const std::uint8_t* scales;
        int padded_k;
    };
    struct Chunk {
        std::uint32_t code;
        std::uint16_t scale_bits;
    };
    __device__ static Chunk load(const Params& p, std::int64_t row, int k, std::uint64_t policy) {
        return {ld_nc_na_evict_first_vec<std::uint32_t>(p.codes + row * (p.padded_k / 2) + (k / 2), policy),
                ld_nc_na<std::uint16_t>(p.scales + row * (p.padded_k / 64) * 2 + (k / 64) * 2)};
    }
    __device__ static void decode(const Chunk& c, float (&w)[8]) {
        Q4SimtDecodeAtom::decode_eight(c.code, c.scale_bits, w);
    }
};

// BF16 contiguous: rows of `stride` elements.
struct Bf16Codec {
    struct Params {
        const bf16* data;
        int stride;
    };
    struct Chunk {
        uint4 bits;
    };
    __device__ static Chunk load(const Params& p, std::int64_t row, int k, std::uint64_t policy) {
        return {ld_nc_na_evict_first_vec<uint4>(p.data + row * p.stride + k, policy)};
    }
    __device__ static void decode(const Chunk& c, float (&w)[8]) {
        const float2 a = bf16x2_bits_to_float2(c.bits.x), b = bf16x2_bits_to_float2(c.bits.y);
        const float2 d = bf16x2_bits_to_float2(c.bits.z), e = bf16x2_bits_to_float2(c.bits.w);
        w[0] = a.x, w[1] = a.y, w[2] = b.x, w[3] = b.y, w[4] = d.x, w[5] = d.y, w[6] = e.x, w[7] = e.y;
    }
};

// Pending: the attention half's inject folds into the residual reads (hyper_connection_inject_mix):
// every value read is hc_injected(R, pending_inject[t][s], pending_y[t][d]).
template <class Codec, class Norm, bool Pending, int PassColumns = 1, int MinBlocks = 2>
__global__ __launch_bounds__(kDownThreads, MinBlocks) void hc_down_partials_kernel(
    const bf16* residual, const Norm* __restrict__ norm_weight, typename Codec::Params down, int rows,
    int columns, float eps, float* __restrict__ partial, float* __restrict__ inv_out,
    const bf16* pending_y, const float* pending_inject) {
    extern __shared__ __align__(16) float rn[]; // [PassColumns][kHidden]
    __shared__ float scratch[kDownThreads / 32][PassColumns];
    __shared__ float inv[PassColumns];
    const int lane = static_cast<int>(threadIdx.x) & 31, warp = static_cast<int>(threadIdx.x) >> 5;
    const int s    = static_cast<int>(blockIdx.y);
    const int row  = static_cast<int>(blockIdx.x) * kDownRows + warp;
    const bool live = row < rows;
    constexpr int width = kStreams * kHidden;
    constexpr int kOwned = kHidden / kDownThreads; // features d = threadIdx.x + kDownThreads * j a thread scales
    static_assert(kHidden % kDownThreads == 0);

    // 1. Weights first: this warp's row of the stream's K slice and the thread's norm weights (1 + w),
    // before any activation.
    const std::uint64_t weight_policy = l2_evict_first_policy();
    const std::int64_t source = live ? row : 0;
    typename Codec::Chunk chunk[kChunks];
#pragma unroll
    for (int m = 0; m < kChunks; ++m) { chunk[m] = Codec::load(down, source, s * kHidden + m * 256 + lane * 8, weight_policy); }
    const Norm* w = norm_weight + static_cast<std::int64_t>(s) * kHidden;
    float w_bias[kOwned];
#pragma unroll
    for (int j = 0; j < kOwned; ++j) { w_bias[j] = 1.0F + norm_value(w[threadIdx.x + kDownThreads * j]); }
    // Launched as a programmatic dependent: the weights load while the producer finishes. The up
    // collapse may launch beside this grid at once and fetch its own weights meanwhile; it reads the
    // partials only after this grid completes.
    pdl::trigger_dependents();
    pdl::wait_for_dependencies();
    // Lanes 4-7 of each quarter warp read a chunk's two float4 halves of Rn in the other order, so a
    // quarter warp's eight 16-byte reads cover all 32 banks.
    const int first_half = (lane >> 2) & 1;

    if constexpr (PassColumns == 1) {
        // Single-column decode specialization (columns == 1; columns >= 2 routes to v4).
        constexpr int kSlices = (kHidden + kDownThreads * 8 - 1) / (kDownThreads * 8);
        const bf16* r         = residual + static_cast<std::int64_t>(s) * kHidden;
        uint4 values[kSlices];
        uint4 ys[Pending ? kSlices : 1];
#pragma unroll
        for (int h = 0; h < kSlices; ++h) {
            const int i = static_cast<int>(threadIdx.x) * 8 + kDownThreads * 8 * h;
            if (i < kHidden) {
                values[h] = load_ldg<uint4>(r + i);
                if constexpr (Pending) { ys[h] = load_ldg<uint4>(pending_y + i); }
            }
        }
        if constexpr (Pending) {
            const float inject = pending_inject[s];
#pragma unroll
            for (int h = 0; h < kSlices; ++h) {
                const int i = static_cast<int>(threadIdx.x) * 8 + kDownThreads * 8 * h;
                if (i < kHidden) { values[h] = injected8(values[h], ys[h], inject); }
            }
        }
        float sum_sq = 0.0F;
#pragma unroll
        for (int h = 0; h < kSlices; ++h) {
            const int i = static_cast<int>(threadIdx.x) * 8 + kDownThreads * 8 * h;
            if (i < kHidden) {
                const uint4 u   = values[h];
                const float2 v0 = bf16x2_bits_to_float2(u.x), v1 = bf16x2_bits_to_float2(u.y);
                const float2 v2 = bf16x2_bits_to_float2(u.z), v3 = bf16x2_bits_to_float2(u.w);
                const float v[8] = {v0.x, v0.y, v1.x, v1.y, v2.x, v2.y, v3.x, v3.y};
                float4* out      = reinterpret_cast<float4*>(rn + i);
                out[0]           = make_float4(v[0], v[1], v[2], v[3]);
                out[1]           = make_float4(v[4], v[5], v[6], v[7]);
#pragma unroll
                for (int j = 0; j < 8; ++j) { sum_sq += v[j] * v[j]; }
            }
        }
        const float warp_total = warp_sum(sum_sq);
        if (lane == 0) { scratch[warp][0] = warp_total; }
        __syncthreads();
        if (threadIdx.x == 0) {
            float total = 0.0F;
#pragma unroll
            for (int wp = 0; wp < kDownThreads / 32; ++wp) { total += scratch[wp][0]; }
            inv[0] = rsqrtf(total / static_cast<float>(kHidden) + eps);
        }
        __syncthreads();
        const float inv0 = inv[0];
#pragma unroll
        for (int j = 0; j < kOwned; ++j) {
            const int i = static_cast<int>(threadIdx.x) + kDownThreads * j;
            rn[i]       = rn[i] * inv0 * w_bias[j];
        }
        if (blockIdx.x == 0 && threadIdx.x == 0) { inv_out[s] = inv0; }
        __syncthreads();
        float acc = 0.0F;
#pragma unroll
        for (int m = 0; m < kChunks; ++m) {
            const int kk = m * 256 + lane * 8;
            float wv[8];
            Codec::decode(chunk[m], wv);
            const float4* half = reinterpret_cast<const float4*>(rn + kk);
            const float4 p = half[first_half], q = half[first_half ^ 1];
            const float4 a = first_half ? q : p, b = first_half ? p : q;
            acc = fmaf(wv[0], a.x, acc);
            acc = fmaf(wv[1], a.y, acc);
            acc = fmaf(wv[2], a.z, acc);
            acc = fmaf(wv[3], a.w, acc);
            acc = fmaf(wv[4], b.x, acc);
            acc = fmaf(wv[5], b.y, acc);
            acc = fmaf(wv[6], b.z, acc);
            acc = fmaf(wv[7], b.w, acc);
        }
        const float total = warp_sum(acc);
        if (lane == 0 && live) { partial[static_cast<std::int64_t>(s) * rows + row] = total; }
    } else {
#pragma unroll 1
        for (int c0 = 0; c0 < columns; c0 += PassColumns) {
            const int pass = min(PassColumns, columns - c0);
            // 2. Stage the slice in FP32 with each column's per-thread sum of squares (its elements in
            // index order), then every column's warp butterfly and its warps' sums in warp order.
            // Every column's loads are issued before any is used (one CTA per SM: registers are free).
            constexpr int kSlices = (kHidden + kDownThreads * 8 - 1) / (kDownThreads * 8);
            uint4 values[PassColumns][kSlices];
            uint4 ys[Pending ? PassColumns : 1][kSlices];
#pragma unroll
            for (int c = 0; c < PassColumns; ++c) {
                const bf16* r = residual + static_cast<std::int64_t>(c0 + c) * width + static_cast<std::int64_t>(s) * kHidden;
#pragma unroll
                for (int h = 0; h < kSlices; ++h) {
                    const int i = static_cast<int>(threadIdx.x) * 8 + kDownThreads * 8 * h;
                    if (c < pass && i < kHidden) {
                        values[c][h] = load_ldg<uint4>(r + i);
                        if constexpr (Pending) {
                            ys[c][h] = load_ldg<uint4>(pending_y + static_cast<std::int64_t>(c0 + c) * kHidden + i);
                        }
                    }
                }
            }
            if constexpr (Pending) {
#pragma unroll
                for (int c = 0; c < PassColumns; ++c) {
                    if (c < pass) {
                        const float inject = pending_inject[static_cast<std::int64_t>(c0 + c) * kStreams + s];
#pragma unroll
                        for (int h = 0; h < kSlices; ++h) {
                            const int i = static_cast<int>(threadIdx.x) * 8 + kDownThreads * 8 * h;
                            if (i < kHidden) { values[c][h] = injected8(values[c][h], ys[c][h], inject); }
                        }
                    }
                }
            }
            float sums[PassColumns];
#pragma unroll
            for (int c = 0; c < PassColumns; ++c) {
                sums[c] = 0.0F;
#pragma unroll
                for (int h = 0; h < kSlices; ++h) {
                    const int i = static_cast<int>(threadIdx.x) * 8 + kDownThreads * 8 * h;
                    if (c < pass && i < kHidden) {
                        const uint4 u   = values[c][h];
                        const float2 v0 = bf16x2_bits_to_float2(u.x), v1 = bf16x2_bits_to_float2(u.y);
                        const float2 v2 = bf16x2_bits_to_float2(u.z), v3 = bf16x2_bits_to_float2(u.w);
                        const float v[8] = {v0.x, v0.y, v1.x, v1.y, v2.x, v2.y, v3.x, v3.y};
                        float4* out = reinterpret_cast<float4*>(rn + c * kHidden + i);
                        out[0] = make_float4(v[0], v[1], v[2], v[3]);
                        out[1] = make_float4(v[4], v[5], v[6], v[7]);
#pragma unroll
                        for (int j = 0; j < 8; ++j) { sums[c] += v[j] * v[j]; }
                    }
                }
            }
#pragma unroll
            for (int c = 0; c < PassColumns; ++c) {
                if (c < pass) {
                    const float total = warp_sum(sums[c]);
                    if (lane == 0) { scratch[warp][c] = total; }
                }
            }
            __syncthreads();
            if (static_cast<int>(threadIdx.x) < pass) {
                float total = 0.0F;
#pragma unroll
                for (int wp = 0; wp < kDownThreads / 32; ++wp) { total += scratch[wp][threadIdx.x]; }
                inv[threadIdx.x] = rsqrtf(total / static_cast<float>(kHidden) + eps);
            }
            __syncthreads();
            // Rn = R * inv * (1 + w), FP32.
            for (int c = 0; c < pass; ++c) {
#pragma unroll
                for (int j = 0; j < kOwned; ++j) {
                    const int i = c * kHidden + static_cast<int>(threadIdx.x) + kDownThreads * j;
                    rn[i]       = rn[i] * inv[c] * w_bias[j];
                }
            }
            if (blockIdx.x == 0 && static_cast<int>(threadIdx.x) < pass) {
                inv_out[static_cast<std::int64_t>(s) * columns + c0 + static_cast<int>(threadIdx.x)] = inv[threadIdx.x];
            }
            __syncthreads();
            // 3. The row's dot products: the lane-chunk order, then a warp butterfly.
            float acc[PassColumns];
#pragma unroll
            for (int c = 0; c < PassColumns; ++c) { acc[c] = 0.0F; }
#pragma unroll
            for (int m = 0; m < kChunks; ++m) {
                const int kk = m * 256 + lane * 8;
                float wv[8];
                Codec::decode(chunk[m], wv);
#pragma unroll
                for (int c = 0; c < PassColumns; ++c) {
                    if (c < pass) {
                        const float4* half = reinterpret_cast<const float4*>(rn + c * kHidden + kk);
                        const float4 p = half[first_half], q = half[first_half ^ 1];
                        const float4 a = first_half ? q : p, b = first_half ? p : q;
                        acc[c] = fmaf(wv[0], a.x, acc[c]);
                        acc[c] = fmaf(wv[1], a.y, acc[c]);
                        acc[c] = fmaf(wv[2], a.z, acc[c]);
                        acc[c] = fmaf(wv[3], a.w, acc[c]);
                        acc[c] = fmaf(wv[4], b.x, acc[c]);
                        acc[c] = fmaf(wv[5], b.y, acc[c]);
                        acc[c] = fmaf(wv[6], b.z, acc[c]);
                        acc[c] = fmaf(wv[7], b.w, acc[c]);
                    }
                }
            }
#pragma unroll
            for (int c = 0; c < PassColumns; ++c) {
                if (c < pass) {
                    const float total = warp_sum(acc[c]);
                    if (lane == 0 && live) {
                        partial[(static_cast<std::int64_t>(s) * rows + row) * columns + c0 + c] = total;
                    }
                }
            }
            if (PassColumns > 1 || c0 + PassColumns < columns) {
                __syncthreads(); // the next pass overwrites rn
            }
        }
    }
}

// Pending: as K1a's, and each residual value it reads (every one of the call, once) is written back with
// the inject applied, as hyper_connection_inject would have left it. The residual is written only then.
template <class Codec, class Norm, bool Pending, int Columns = 0, int MinBlocks = 1>
__global__ __launch_bounds__(kUpThreads, MinBlocks) void hc_up_collapse_kernel(
    const float* partial, const float* inv_in, bf16* __restrict__ residual,
    const Norm* __restrict__ norm_weight, typename Codec::Params up, int rank, int down_rows, int columns,
    float* __restrict__ inject, bf16* __restrict__ x, const bf16* pending_y,
    const float* pending_inject) {
    // Dynamic shared memory: sized to actual rank x columns at launch time rather than the
    // worst-case compile-time bounds. For decode (rank=320, columns=1) this drops from 128 KB
    // to ~5 KB, raising CTAs-per-SM from 1 to ~18 and exposing enough warps to hide DRAM latency.
    extern __shared__ __align__(16) float dyn_smem[];
    auto* m   = dyn_smem;                                            // [columns][rank]
    auto* inv = dyn_smem + static_cast<std::size_t>(columns) * rank; // [streams][columns]
    const int lane = static_cast<int>(threadIdx.x) & 31, warp = static_cast<int>(threadIdx.x) >> 5;
    const int d    = static_cast<int>(blockIdx.x) * kUpFeatures + warp;
    const bool live = d < kHidden;
    constexpr int width   = kStreams * kHidden;
    const int chunks    = rank / 8;
    const std::uint64_t weight_policy = l2_evict_first_policy();

    if constexpr (Columns == 1) {
        constexpr int kRowsPerWarp = 2;
        __shared__ float rs_smem[kUpFeatures][kRowsPerWarp][kStreams];
        const int d0    = (static_cast<int>(blockIdx.x) * kUpFeatures + warp) * kRowsPerWarp;
        const int slane = lane < kStreams ? lane : 0;
        constexpr int kPreloadStreams = kStreams / 2;
        typename Codec::Chunk chunk0[kPreloadStreams];
#pragma unroll
        for (int s = 0; s < kPreloadStreams; ++s) {
            const std::int64_t row = static_cast<std::int64_t>(s) * kHidden + (d0 < kHidden ? d0 : 0);
            chunk0[s]              = lane < chunks ? Codec::load(up, row, lane * 8, weight_policy) : typename Codec::Chunk{};
        }
        pdl::trigger_dependents();
        pdl::wait_for_dependencies();
        if (lane < kStreams) {
#pragma unroll
            for (int r = 0; r < kRowsPerWarp; ++r) {
                const int dr  = d0 + r;
                const bool ok = dr < kHidden;
                bf16* at      = residual + static_cast<std::int64_t>(slane) * kHidden + (ok ? dr : 0);
                float rv_s    = 0.0F;
                if constexpr (Pending) {
                    const bf16 value = hc_injected(__bfloat162float(*at), pending_inject[slane],
                                                   __bfloat162float(pending_y[ok ? dr : 0]));
                    if (ok) { *at = value; }
                    rv_s = __bfloat162float(value);
                } else {
                    rv_s = ok ? __bfloat162float(*at) : 0.0F;
                }
                rs_smem[warp][r][lane] = rv_s;
            }
        }
        const int first_half    = (lane >> 2) & 1;
        const float inv_streams = 1.0F / static_cast<float>(kStreams);
        for (int k = static_cast<int>(threadIdx.x); k < down_rows; k += kUpThreads) {
            float z = 0.0F;
#pragma unroll
            for (int s = 0; s < kStreams; ++s) { z += partial[static_cast<std::int64_t>(s) * down_rows + k]; }
            const float v = z * inv_streams;
            if (k < rank) {
                m[k] = v * sigmoidf(v);
            } else if (inject != nullptr && blockIdx.x == 0) {
                inject[k - rank] = 2.0F * sigmoidf(v);
            }
        }
        if (threadIdx.x < kStreams) { inv[threadIdx.x] = inv_in[threadIdx.x]; }
        __syncthreads();
        if (d0 >= kHidden) { return; }

#pragma unroll 1
        for (int r = 0; r < kRowsPerWarp; ++r) {
            const int dr = d0 + r;
            if (dr >= kHidden) { continue; }
            float acc[kStreams];
#pragma unroll
            for (int s = 0; s < kStreams; ++s) { acc[s] = 0.0F; }
            if (lane < chunks) {
                const float4* half = reinterpret_cast<const float4*>(m + lane * 8);
                const float4 p = half[first_half], q = half[first_half ^ 1];
                const float4 a = first_half ? q : p, b = first_half ? p : q;
#pragma unroll
                for (int s = 0; s < kStreams; ++s) {
                    const std::int64_t row = static_cast<std::int64_t>(s) * kHidden + dr;
                    const auto ch =
                        (r == 0 && s < kPreloadStreams) ? chunk0[s] : Codec::load(up, row, lane * 8, weight_policy);
                    float wv[8];
                    Codec::decode(ch, wv);
                    acc[s] = fmaf(wv[0], a.x, acc[s]);
                    acc[s] = fmaf(wv[1], a.y, acc[s]);
                    acc[s] = fmaf(wv[2], a.z, acc[s]);
                    acc[s] = fmaf(wv[3], a.w, acc[s]);
                    acc[s] = fmaf(wv[4], b.x, acc[s]);
                    acc[s] = fmaf(wv[5], b.y, acc[s]);
                    acc[s] = fmaf(wv[6], b.z, acc[s]);
                    acc[s] = fmaf(wv[7], b.w, acc[s]);
                }
            }
            const int c1 = lane + 32;
            if (c1 < chunks) {
                const float4* half = reinterpret_cast<const float4*>(m + c1 * 8);
                const float4 p = half[first_half], q = half[first_half ^ 1];
                const float4 a = first_half ? q : p, b = first_half ? p : q;
#pragma unroll
                for (int s = 0; s < kStreams; ++s) {
                    const std::int64_t row = static_cast<std::int64_t>(s) * kHidden + dr;
                    const auto chunk1      = Codec::load(up, row, c1 * 8, weight_policy);
                    float wv[8];
                    Codec::decode(chunk1, wv);
                    acc[s] = fmaf(wv[0], a.x, acc[s]);
                    acc[s] = fmaf(wv[1], a.y, acc[s]);
                    acc[s] = fmaf(wv[2], a.z, acc[s]);
                    acc[s] = fmaf(wv[3], a.w, acc[s]);
                    acc[s] = fmaf(wv[4], b.x, acc[s]);
                    acc[s] = fmaf(wv[5], b.y, acc[s]);
                    acc[s] = fmaf(wv[6], b.z, acc[s]);
                    acc[s] = fmaf(wv[7], b.w, acc[s]);
                }
            }
#pragma unroll
            for (int s = 0; s < kStreams; ++s) { acc[s] = warp_sum(acc[s]); }
            if (lane == 0) {
                float sum = 0.0F;
#pragma unroll
                for (int s = 0; s < kStreams; ++s) {
                    const float wb_s =
                        1.0F + norm_value(norm_weight[static_cast<std::int64_t>(s) * kHidden + dr]);
                    const float rn = rs_smem[warp][r][s] * inv[s] * wb_s;
                    sum += sigmoidf(acc[s]) * rn;
                }
                x[dr] = __float2bfloat16_rn(sum * inv_streams);
            }
        }
    } else {
        // 1. Weights first: the S rows (s*H + d) of W_up, chunks lane and lane + 32.
        typename Codec::Chunk chunk[kStreams][2];
#pragma unroll
        for (int s = 0; s < kStreams; ++s) {
            const std::int64_t row = static_cast<std::int64_t>(s) * kHidden + (live ? d : 0);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int c = lane + 32 * h;
                chunk[s][h] = c < chunks ? Codec::load(up, row, c * 8, weight_policy) : typename Codec::Chunk{};
            }
        }
        // The norm weights (1 + w) of the warp's S rows, and the residual R[s*H + d, t] for every (t, s):
        // entry e = t*S + s is held by lane e % 32 in r_held[e / 32], for lane 0's collapse.
        static_assert(kStreams * kSimtMaxColumns <= 64);
        const int dl = live ? d : 0;
        float w_bias[kStreams];
#pragma unroll
        for (int s = 0; s < kStreams; ++s) {
            w_bias[s] = 1.0F + norm_value(norm_weight[static_cast<std::int64_t>(s) * kHidden + dl]);
        }
        // Launched as a programmatic dependent: the weights load while the down partials finish. The next
        // projection may launch beside this grid at once and fetch its weights meanwhile.
        pdl::trigger_dependents();
        pdl::wait_for_dependencies();
        float r_held[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int e = lane + 32 * h, t = e / kStreams, s = e - t * kStreams;
            bf16* at    = residual + static_cast<std::int64_t>(t) * width + static_cast<std::int64_t>(s) * kHidden + dl;
            if constexpr (Pending) {
                r_held[h] = 0.0F;
                if (t < columns) {
                    const bf16 value = hc_injected(__bfloat162float(*at), pending_inject[static_cast<std::int64_t>(t) * kStreams + s],
                                                   __bfloat162float(pending_y[static_cast<std::int64_t>(t) * kHidden + dl]));
                    if (live) { *at = value; }
                    r_held[h] = __bfloat162float(value);
                }
            } else {
                r_held[h] = t < columns ? __bfloat162float(*at) : 0.0F;
            }
        }
        // Lanes 4-7 of each quarter warp read a chunk's two float4 halves of m in the other order, so a
        // quarter warp's eight 16-byte reads cover all 32 banks.
        const int first_half = (lane >> 2) & 1;

        // 2. z = sum_s partial[s] in order s = 0..S-1; m = SiLU(z / S); CTA 0 writes the injects.
        const float inv_streams = 1.0F / static_cast<float>(kStreams);
#pragma unroll 4
        for (int i = static_cast<int>(threadIdx.x); i < down_rows * columns; i += kUpThreads) {
            const int k = i / columns, t = i - k * columns;
            float z     = 0.0F;
#pragma unroll
            for (int s = 0; s < kStreams; ++s) { z += partial[(static_cast<std::int64_t>(s) * down_rows + k) * columns + t]; }
            const float v = z * inv_streams;
            if (k < rank) {
                m[t * rank + k] = v * sigmoidf(v);
            } else if (inject != nullptr && blockIdx.x == 0) {
                inject[static_cast<std::int64_t>(t) * kStreams + (k - rank)] = 2.0F * sigmoidf(v);
            }
        }
        for (int i = static_cast<int>(threadIdx.x); i < kStreams * columns; i += kUpThreads) { inv[i] = inv_in[i]; }
        __syncthreads();
        if (!live) { return; }

        // 3. u_s = W_up[s*H + d] . m in the lane-chunk order, then the collapse in lane 0.
#pragma unroll 1
        for (int t = 0; t < columns; ++t) {
            const float* mt = m + t * rank;
            float acc[kStreams];
#pragma unroll
            for (int s = 0; s < kStreams; ++s) { acc[s] = 0.0F; }
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int c = lane + 32 * h;
                if (c < chunks) {
                    const float4* half = reinterpret_cast<const float4*>(mt + c * 8);
                    const float4 p = half[first_half], q = half[first_half ^ 1];
                    const float4 a = first_half ? q : p, b = first_half ? p : q;
#pragma unroll
                    for (int s = 0; s < kStreams; ++s) {
                        float wv[8];
                        Codec::decode(chunk[s][h], wv);
                        acc[s] = fmaf(wv[0], a.x, acc[s]);
                        acc[s] = fmaf(wv[1], a.y, acc[s]);
                        acc[s] = fmaf(wv[2], a.z, acc[s]);
                        acc[s] = fmaf(wv[3], a.w, acc[s]);
                        acc[s] = fmaf(wv[4], b.x, acc[s]);
                        acc[s] = fmaf(wv[5], b.y, acc[s]);
                        acc[s] = fmaf(wv[6], b.z, acc[s]);
                        acc[s] = fmaf(wv[7], b.w, acc[s]);
                    }
                }
            }
#pragma unroll
            for (int s = 0; s < kStreams; ++s) { acc[s] = warp_sum(acc[s]); }
            float r[kStreams];
#pragma unroll
            for (int s = 0; s < kStreams; ++s) {
                const int e    = t * kStreams + s;
                const float lo = __shfl_sync(0xFFFFFFFFU, r_held[0], e & 31);
                const float hi = __shfl_sync(0xFFFFFFFFU, r_held[1], e & 31);
                r[s]           = e < 32 ? lo : hi;
            }
            if (lane == 0) {
                float sum = 0.0F;
#pragma unroll
                for (int s = 0; s < kStreams; ++s) {
                    const float rn = r[s] * inv[s * columns + t] * w_bias[s];
                    sum += sigmoidf(acc[s]) * rn;
                }
                x[static_cast<std::int64_t>(t) * kHidden + d] = __float2bfloat16_rn(sum * inv_streams);
            }
        }
    }
}

// Strata v4's hyper-connection read scheduling (strata-nvfp4 src/kernels/cuda/fused_gr.cu,
// gr_down_v4_kernel and gr_up_v4_kernel) for calls of two or more columns. Every product and sum
// below is the one K1a/K1b compute, in the same order, so the outputs are bit-identical. RTX 5090,
// Q8 mixer, 2026-10-16: per call -7.6 % (T 2) to -17.8 % (T 5), +2.2 % at T 1 (which keeps
// K1a/K1b); Flash-Next Dense8 MTP 4 decode +0.9 %, plain decode +0.6 %. Two rows per warp in K1a
// (v4's other change) spilled 120-160 bytes of stack and was slower (-0 to +20 % per call).
//
// K1a/K1b already had v4's weights-first loads (every weight chunk in registers before the PDL
// wait), the residual read through __ldg, its lane-chunk K mapping (k = 8 lane + 256 m) and, in
// K1b, tokens outer with each token's m chunk read once for all of a warp's rows. What these
// kernels add:
//   K1a v4: Rn is staged once, straight from the registers that summed its squares, in two planes
//     (floats 0-3 of every 8 in plane 0, floats 4-7 in plane 1). The 16-byte stores of a quarter
//     warp and the dot product's reads are then conflict-free, and the raw store plus in-place
//     rescale pass of K1a (a two-way conflicted store, then a read and a write) is gone. The CTA
//     keeps 8 warps of one row each, so the norm's per-thread sums and warp order are unchanged.
//   K1b v4: 8 warps of 8 rows (2 features x 4 streams, v4's row map) instead of 16 warps of 4; a
//     warp's 8 row sums of a token reduce by one transposed butterfly of 9 shuffles, so a CTA issues
//     8 x 9 = 72 shuffles per token instead of 16 x (4 warp_sums' 20 + 8 for lane 0's residual
//     gather) = 448, and each m read feeds 8 rows instead of 4; m is staged in two planes; partials,
//     inv and R are read through __ldg. The collapse runs after the token loop, one thread per
//     (token, feature), from the row sums in shared memory, with K1b's own expressions in its order.
// Not adopted: v4's deferred RMS scale (Strata multiplies unscaled partials by inv in up, another
// rounding) and its collapse (it rounds each sigmoid(u) Rn product before the stream sum, where
// K1b's chain may fuse each product into the sum).
//
// Why the transposed butterfly is exact. warp_sum adds own + partner at xor offsets 16, 8, 4, 2,
// 1: level j turns lane l's value into v_l + v_(l ^ o_j). IEEE-754 addition is commutative (signed
// zeros included; the GPU returns the canonical NaN for any NaN operand), so lanes l and l ^ o_j
// hold identical bits after every level, and every lane ends with the same tree: the pairs added
// at each level are fixed by the xor offsets alone. warp_sum8_transposed reduces eight values per
// lane at once. At offset 16 lane l keeps the four values of its half (values 4 b4 .. 4 b4 + 3,
// b4 = bit 4 of l) and adds the partner's same four: for each such value r it computes
// v_l[r] + v_(l^16)[r], exactly warp_sum's first level for value r in lane l. At offset 8 the
// partner l ^ 8 has the same bit 4, so it holds the same four values' level-1 sums; lane l keeps
// two (by bit 3) and adds the partner's: warp_sum's second level. Offset 4 does the same for one
// value (by bit 2), and offsets 2 and 1 are ordinary butterfly levels between lanes holding the same
// value. Lane l ends with value l >> 2, every level formed from warp_sum's own pairs: bit-identical
// to warp_sum of that value, in every lane of the group 4 (l >> 2) .. 4 (l >> 2) + 3.

// Eight per-lane values reduced across the warp: lane l returns the warp_sum of v[l >> 2].
__device__ __forceinline__ float warp_sum8_transposed(const float (&v)[8], int lane) {
    const bool b4 = (lane & 16) != 0, b3 = (lane & 8) != 0, b2 = (lane & 4) != 0;
    float h4[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float keep = b4 ? v[4 + i] : v[i];
        const float send = b4 ? v[i] : v[4 + i];
        h4[i]            = keep + __shfl_xor_sync(0xFFFFFFFFU, send, 16);
    }
    float h2[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const float keep = b3 ? h4[2 + i] : h4[i];
        const float send = b3 ? h4[i] : h4[2 + i];
        h2[i]            = keep + __shfl_xor_sync(0xFFFFFFFFU, send, 8);
    }
    const float keep = b2 ? h2[1] : h2[0];
    const float send = b2 ? h2[0] : h2[1];
    float total      = keep + __shfl_xor_sync(0xFFFFFFFFU, send, 4);
    total += __shfl_xor_sync(0xFFFFFFFFU, total, 2);
    total += __shfl_xor_sync(0xFFFFFFFFU, total, 1);
    return total;
}

// K1a with v4's staging (see above): one row of W_down per warp, 8 warps per CTA. Shared Rn:
// [pass column][plane][kHidden / 2]. A per-column CTA grid (one CTA per row group, stream and
// column, GB10 candidate 2026-10-30) measured 3..8 % slower at the widths this branch now serves
// (T = 2..3) and neutral at 4..8, so the pass loop stays.
template <class Codec, class Norm, bool Pending, int ExactColumns = 0>
__global__ __launch_bounds__(kDownThreads, (ExactColumns >= 2 && ExactColumns <= 4) ? 2 : 1) void hc_down_partials_v4_kernel(
    const bf16* residual, const Norm* __restrict__ norm_weight, typename Codec::Params down, int rows,
    int columns, float eps, float* __restrict__ partial, float* __restrict__ inv_out,
    const bf16* pending_y, const float* pending_inject) {
    constexpr int RowsPerWarp = 2;
    constexpr int kColumns    = ExactColumns > 0 ? ExactColumns : kPassColumns; // columns per pass
    extern __shared__ __align__(16) float rn[]; // [kColumns][2][kHidden / 2]
    __shared__ float scratch[kDownThreads / 32][kColumns];
    __shared__ float inv[kColumns];
    const int lane = static_cast<int>(threadIdx.x) & 31, warp = static_cast<int>(threadIdx.x) >> 5;
    const int s    = static_cast<int>(blockIdx.y);
    const int row0 = (static_cast<int>(blockIdx.x) * (kDownThreads / 32) + warp) * RowsPerWarp;
    constexpr int width   = kStreams * kHidden;
    constexpr int kSlices = (kHidden + kDownThreads * 8 - 1) / (kDownThreads * 8);
    constexpr int kPlane  = kHidden / 8; // float4s per plane of one column
    static_assert(kHidden % 8 == 0);

    // 1. Weights first: the warp's rows of the stream's K slice, and the norm weights of the 8-element
    // groups this thread stages (i = 8 tid + 8 kDownThreads h, the same elements K1a's thread sums).
    constexpr int kPreloadChunks = ExactColumns == 4 ? 1 : (ExactColumns == 3 ? 2 : (kChunks / 2));
    const std::uint64_t weight_policy = l2_evict_first_policy();
    typename Codec::Chunk chunk0[kPreloadChunks];
    {
        const std::int64_t source = row0 < rows ? row0 : 0;
#pragma unroll
        for (int m = 0; m < kPreloadChunks; ++m) {
            chunk0[m] = Codec::load(down, source, s * kHidden + m * 256 + lane * 8, weight_policy);
        }
    }
    const Norm* w = norm_weight + static_cast<std::int64_t>(s) * kHidden;
    NormGroup<Norm> w_bits[kSlices];
#pragma unroll
    for (int h = 0; h < kSlices; ++h) {
        const int i = static_cast<int>(threadIdx.x) * 8 + kDownThreads * 8 * h;
        w_bits[h]   = i < kHidden ? NormGroup<Norm>::load(w + i) : NormGroup<Norm>::zero();
    }
    pdl::trigger_dependents();
    pdl::wait_for_dependencies();

    const int n_cols = ExactColumns > 0 ? ExactColumns : columns;
#pragma unroll 1
    for (int c0 = 0; c0 < n_cols; c0 += kColumns) {
        const int pass = ExactColumns > 0 ? ExactColumns : min(kColumns, columns - c0);
        // 2. K1a's loads (and pending inject) and per-thread sums of squares (its elements in index
        // order), its warp butterflies and its warps' sums in warp order.
        uint4 values[kColumns][kSlices];
#pragma unroll
        for (int c = 0; c < kColumns; ++c) {
            const bf16* r = residual + static_cast<std::int64_t>(c0 + c) * width + static_cast<std::int64_t>(s) * kHidden;
            float inject  = 0.0F;
            if constexpr (Pending) {
                if (ExactColumns > 0 || c < pass) {
                    inject = pending_inject[static_cast<std::int64_t>(c0 + c) * kStreams + s];
                }
            }
#pragma unroll
            for (int h = 0; h < kSlices; ++h) {
                const int i = static_cast<int>(threadIdx.x) * 8 + kDownThreads * 8 * h;
                if ((ExactColumns > 0 || c < pass) && i < kHidden) {
                    uint4 v = load_ldg<uint4>(r + i);
                    if constexpr (Pending) {
                        const uint4 y = load_ldg<uint4>(pending_y + static_cast<std::int64_t>(c0 + c) * kHidden + i);
                        v             = injected8(v, y, inject);
                    }
                    values[c][h] = v;
                }
            }
        }
        float sums[kColumns];
#pragma unroll
        for (int c = 0; c < kColumns; ++c) {
            sums[c] = 0.0F;
#pragma unroll
            for (int h = 0; h < kSlices; ++h) {
                const int i = static_cast<int>(threadIdx.x) * 8 + kDownThreads * 8 * h;
                if ((ExactColumns > 0 || c < pass) && i < kHidden) {
                    float v[8];
                    bf16x8_bits_to_floats(values[c][h], v);
#pragma unroll
                    for (int j = 0; j < 8; ++j) { sums[c] += v[j] * v[j]; }
                }
            }
        }
#pragma unroll
        for (int c = 0; c < kColumns; ++c) {
            if (ExactColumns > 0 || c < pass) {
                const float total = warp_sum(sums[c]);
                if (lane == 0) { scratch[warp][c] = total; }
            }
        }
        __syncthreads();
        if (static_cast<int>(threadIdx.x) < pass) {
            float total = 0.0F;
#pragma unroll
            for (int wp = 0; wp < kDownThreads / 32; ++wp) { total += scratch[wp][threadIdx.x]; }
            inv[threadIdx.x] = rsqrtf(total / static_cast<float>(kHidden) + eps);
        }
        __syncthreads();
        // Rn = R * inv * (1 + w), FP32, K1a's expression, stored once from the registers: group g of 8
        // elements goes to float4 g of plane 0 (elements 0-3) and of plane 1 (elements 4-7).
#pragma unroll
        for (int c = 0; c < kColumns; ++c) {
            if (ExactColumns > 0 || c < pass) {
                const float ic = inv[c];
                float4* plane  = reinterpret_cast<float4*>(rn) + c * 2 * kPlane;
#pragma unroll
                for (int h = 0; h < kSlices; ++h) {
                    const int i = static_cast<int>(threadIdx.x) * 8 + kDownThreads * 8 * h;
                    if (i < kHidden) {
                        float v[8], b[8];
                        bf16x8_bits_to_floats(values[c][h], v);
                        w_bits[h].floats(b);
#pragma unroll
                        for (int j = 0; j < 8; ++j) { b[j] = 1.0F + b[j]; }
                        const int g       = static_cast<int>(threadIdx.x) + kDownThreads * h;
                        plane[g] = make_float4(v[0] * ic * b[0], v[1] * ic * b[1], v[2] * ic * b[2],
                                               v[3] * ic * b[3]);
                        plane[kPlane + g] = make_float4(v[4] * ic * b[4], v[5] * ic * b[5], v[6] * ic * b[6],
                                                        v[7] * ic * b[7]);
                    }
                }
            }
        }
        if (blockIdx.x == 0 && static_cast<int>(threadIdx.x) < pass) {
            inv_out[static_cast<std::int64_t>(s) * n_cols + c0 + static_cast<int>(threadIdx.x)] = inv[threadIdx.x];
        }
        __syncthreads();
        // 3. Each row's dot products: the lane-chunk order, then a warp butterfly per (row, column).
        float acc[RowsPerWarp][kColumns];
#pragma unroll
        for (int r = 0; r < RowsPerWarp; ++r) {
#pragma unroll
            for (int c = 0; c < kColumns; ++c) { acc[r][c] = 0.0F; }
        }
#pragma unroll
        for (int m = 0; m < kChunks; ++m) {
            const int g = m * 32 + lane;
            float wv[RowsPerWarp][8];
            if (m < kPreloadChunks) {
                Codec::decode(chunk0[m], wv[0]);
            } else {
                const std::int64_t source = row0 < rows ? row0 : 0;
                Codec::decode(Codec::load(down, source, s * kHidden + m * 256 + lane * 8, weight_policy), wv[0]);
            }
            if constexpr (RowsPerWarp > 1) {
#pragma unroll
                for (int r = 1; r < RowsPerWarp; ++r) {
                    const std::int64_t source = row0 + r < rows ? row0 + r : 0;
                    Codec::decode(Codec::load(down, source, s * kHidden + m * 256 + lane * 8, weight_policy), wv[r]);
                }
            }
#pragma unroll
            for (int c = 0; c < kColumns; ++c) {
                if (ExactColumns > 0 || c < pass) {
                    const float4* plane = reinterpret_cast<const float4*>(rn) + c * 2 * kPlane;
                    const float4 a = plane[g], b = plane[kPlane + g];
#pragma unroll
                    for (int r = 0; r < RowsPerWarp; ++r) {
                        acc[r][c] = fmaf(wv[r][0], a.x, acc[r][c]);
                        acc[r][c] = fmaf(wv[r][1], a.y, acc[r][c]);
                        acc[r][c] = fmaf(wv[r][2], a.z, acc[r][c]);
                        acc[r][c] = fmaf(wv[r][3], a.w, acc[r][c]);
                        acc[r][c] = fmaf(wv[r][4], b.x, acc[r][c]);
                        acc[r][c] = fmaf(wv[r][5], b.y, acc[r][c]);
                        acc[r][c] = fmaf(wv[r][6], b.z, acc[r][c]);
                        acc[r][c] = fmaf(wv[r][7], b.w, acc[r][c]);
                    }
                }
            }
        }
#pragma unroll
        for (int r = 0; r < RowsPerWarp; ++r) {
#pragma unroll
            for (int c = 0; c < kColumns; ++c) {
                if (ExactColumns > 0 || c < pass) {
                    const float total = warp_sum(acc[r][c]);
                    if (lane == 0 && row0 + r < rows) {
                        partial[(static_cast<std::int64_t>(s) * rows + row0 + r) * n_cols + c0 + c] = total;
                    }
                }
            }
        }
        if constexpr (ExactColumns == 0) {
            __syncthreads(); // the next pass overwrites rn
        }
    }
}

// K1b with v4's row map and transposed butterfly (see above).
// The v4 up collapse keeps __launch_bounds__ minBlocks = 1: its natural footprint is ~168 registers,
// so the SM already caps at one block per SM and forcing the 2-block 128-register ceiling spilled
// 128 bytes of stack into every instantiation when measured on GB10 (2026-10-30) with no bandwidth
// to gain. The call's own dynamic SMEM (below) shrinks the shared footprint without touching
// registers.
constexpr int kUpV4Warps   = 8;
constexpr int kUpV4Threads = kUpV4Warps * 32;
constexpr int kUpV4Rows    = 8; // a warp's rows: features warp and warp + 8, 4 streams each
// Worst-case dynamic SMEM for one v4 up collapse call: m [T][rank], u [T][S][features], inv [S][T].
constexpr int kUpV4MaxSmem = static_cast<int>(
    (static_cast<std::size_t>(kSimtMaxColumns) * (kMaxRank + kStreams * kUpFeatures) +
     static_cast<std::size_t>(kStreams) * kSimtMaxColumns) *
    sizeof(float));

template <class Codec, class Norm, bool Pending, int ExactColumns = 0>
__global__ __launch_bounds__(
    kUpV4Threads,
    (std::is_same_v<Codec, Q8Codec> || std::is_same_v<Codec, Q4Codec>) && (ExactColumns >= 2 && ExactColumns <= 4) ? 2 : 1) void hc_up_collapse_v4_kernel(
    const float* partial, const float* inv_in, bf16* __restrict__ residual,
    const Norm* __restrict__ norm_weight, typename Codec::Params up, int rank, int down_rows, int columns,
    float* __restrict__ inject, bf16* __restrict__ x, const bf16* pending_y,
    const float* pending_inject) {
    static_assert(kUpFeatures == 2 * kUpV4Warps && kUpV4Rows == 2 * kStreams);
    static_assert(kUpFeatures * kSimtMaxColumns <= kUpV4Threads);
    const int n_cols = ExactColumns > 0 ? ExactColumns : columns;
    // The three arrays move through dynamic shared memory sized to this call (as in
    // hc_up_collapse_kernel), so a narrow call stages its own bytes instead of the old worst-case
    // static kHcMixFusedMaxColumns x kMaxRank reservation. The layouts and every index are the ones
    // the static arrays had; rank % 8 == 0 keeps each m row 16-byte aligned for the float4 reads.
    extern __shared__ __align__(16) float dyn_smem[];
    float* m   = dyn_smem;                                                        // [T][2][rank / 2]
    float* u   = m + static_cast<std::size_t>(n_cols) * rank;                     // [T][S][feature]
    float* inv = u + static_cast<std::size_t>(n_cols) * kStreams * kUpFeatures;   // [S][T]
    const int tid  = static_cast<int>(threadIdx.x);
    const int lane = tid & 31, warp = tid >> 5;
    const int d0   = static_cast<int>(blockIdx.x) * kUpFeatures;
    constexpr int width = kStreams * kHidden;
    const int chunks    = rank / 8;

    // 1. Weights first: the warp's 8 rows (s/2 = stream, d0 + warp + 8*(s%2)), chunks lane and lane + 32.
    const std::uint64_t weight_policy = l2_evict_first_policy();
    typename Codec::Chunk chunk[kUpV4Rows][2];
#pragma unroll
    for (int q = 0; q < kUpV4Rows; ++q) {
        const int d            = d0 + warp + kUpV4Warps * (q & 1);
        const std::int64_t row = static_cast<std::int64_t>(q >> 1) * kHidden + (d < kHidden ? d : 0);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int c = lane + 32 * h;
            chunk[q][h] = c < chunks ? Codec::load(up, row, c * 8, weight_policy) : typename Codec::Chunk{};
        }
    }
    // The collapse thread of (token te, feature de) and its norm weights (1 + w).
    const int fe = tid % kUpFeatures, te = tid / kUpFeatures;
    const int de = d0 + fe;
    const int dl = de < kHidden ? de : 0;
    float w_bias[kStreams];
#pragma unroll
    for (int s = 0; s < kStreams; ++s) {
        w_bias[s] = 1.0F + norm_value(norm_weight[static_cast<std::int64_t>(s) * kHidden + dl]);
    }
    pdl::trigger_dependents();
    pdl::wait_for_dependencies();
    const bool collapses = de < kHidden && te < n_cols;
    float r[kStreams];
#pragma unroll
    for (int s = 0; s < kStreams; ++s) {
        bf16* at = residual + static_cast<std::int64_t>(te) * width + static_cast<std::int64_t>(s) * kHidden + de;
        if constexpr (Pending) {
            // Every (column, stream, feature) belongs to one collapse thread: it reads and writes it back.
            r[s] = 0.0F;
            if (collapses) {
                const bf16 value = hc_injected(__bfloat162float(*at), pending_inject[static_cast<std::int64_t>(te) * kStreams + s],
                                               __bfloat162float(pending_y[static_cast<std::int64_t>(te) * kHidden + de]));
                *at  = value;
                r[s] = __bfloat162float(value);
            }
        } else {
            r[s] = collapses ? __bfloat162float(__ldg(at)) : 0.0F;
        }
    }

    // 2. z = sum_s partial[s] in order s = 0..S-1; m = SiLU(z / S) into plane (k / 4) % 2 at
    // 4 (k / 8) + k % 4; CTA 0 writes the injects.
    const float inv_streams = 1.0F / static_cast<float>(kStreams);
    const int half          = rank / 2;
#pragma unroll 4
    for (int i = tid; i < down_rows * n_cols; i += kUpV4Threads) {
        const int k = i / n_cols, t = i - k * n_cols;
        float z     = 0.0F;
#pragma unroll
        for (int s = 0; s < kStreams; ++s) {
            z += __ldg(partial + (static_cast<std::int64_t>(s) * down_rows + k) * n_cols + t);
        }
        const float v = z * inv_streams;
        if (k < rank) {
            m[t * rank + ((k >> 2) & 1) * half + (k >> 3) * 4 + (k & 3)] = v * sigmoidf(v);
        } else if (inject != nullptr && blockIdx.x == 0) {
            inject[static_cast<std::int64_t>(t) * kStreams + (k - rank)] = 2.0F * sigmoidf(v);
        }
    }
    for (int i = tid; i < kStreams * n_cols; i += kUpV4Threads) { inv[i] = __ldg(inv_in + i); }
    __syncthreads();

    // 3. Per token, the warp's 8 rows in the lane-chunk order, one transposed butterfly, and lane
    // 4q stores row q's sum.
    if constexpr (ExactColumns > 0) {
        float acc[ExactColumns][kUpV4Rows];
#pragma unroll
        for (int t = 0; t < ExactColumns; ++t) {
#pragma unroll
            for (int q = 0; q < kUpV4Rows; ++q) { acc[t][q] = 0.0F; }
        }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int c = lane + 32 * h;
            if (c < chunks) {
                float4 a[ExactColumns], b[ExactColumns];
#pragma unroll
                for (int t = 0; t < ExactColumns; ++t) {
                    const float4* mt = reinterpret_cast<const float4*>(m + t * rank);
                    a[t]             = mt[c];
                    b[t]             = mt[chunks + c];
                }
#pragma unroll
                for (int q = 0; q < kUpV4Rows; ++q) {
                    float wv[8];
                    Codec::decode(chunk[q][h], wv);
#pragma unroll
                    for (int t = 0; t < ExactColumns; ++t) {
                        acc[t][q] = fmaf(wv[0], a[t].x, acc[t][q]);
                        acc[t][q] = fmaf(wv[1], a[t].y, acc[t][q]);
                        acc[t][q] = fmaf(wv[2], a[t].z, acc[t][q]);
                        acc[t][q] = fmaf(wv[3], a[t].w, acc[t][q]);
                        acc[t][q] = fmaf(wv[4], b[t].x, acc[t][q]);
                        acc[t][q] = fmaf(wv[5], b[t].y, acc[t][q]);
                        acc[t][q] = fmaf(wv[6], b[t].z, acc[t][q]);
                        acc[t][q] = fmaf(wv[7], b[t].w, acc[t][q]);
                    }
                }
            }
        }
#pragma unroll
        for (int t = 0; t < ExactColumns; ++t) {
            const float total = warp_sum8_transposed(acc[t], lane);
            if ((lane & 3) == 0) {
                const int q = lane >> 2;
                u[(t * kStreams + (q >> 1)) * kUpFeatures + warp + kUpV4Warps * (q & 1)] = total;
            }
        }
    } else {
        for (int t = 0; t < n_cols; ++t) {
            const float4* mt = reinterpret_cast<const float4*>(m + t * rank);
            float acc[kUpV4Rows];
#pragma unroll
            for (int q = 0; q < kUpV4Rows; ++q) { acc[q] = 0.0F; }
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int c = lane + 32 * h;
                if (c < chunks) {
                    const float4 a = mt[c], b = mt[chunks + c];
#pragma unroll
                    for (int q = 0; q < kUpV4Rows; ++q) {
                        float wv[8];
                        Codec::decode(chunk[q][h], wv);
                        acc[q] = fmaf(wv[0], a.x, acc[q]);
                        acc[q] = fmaf(wv[1], a.y, acc[q]);
                        acc[q] = fmaf(wv[2], a.z, acc[q]);
                        acc[q] = fmaf(wv[3], a.w, acc[q]);
                        acc[q] = fmaf(wv[4], b.x, acc[q]);
                        acc[q] = fmaf(wv[5], b.y, acc[q]);
                        acc[q] = fmaf(wv[6], b.z, acc[q]);
                        acc[q] = fmaf(wv[7], b.w, acc[q]);
                    }
                }
            }
            const float total = warp_sum8_transposed(acc, lane);
            if ((lane & 3) == 0) {
                const int q = lane >> 2;
                u[(t * kStreams + (q >> 1)) * kUpFeatures + warp + kUpV4Warps * (q & 1)] = total;
            }
        }
    }
    __syncthreads();

    // 4. The collapse, K1b's expressions in its order.
    if (collapses) {
        float sum = 0.0F;
#pragma unroll
        for (int s = 0; s < kStreams; ++s) {
            const float rn = r[s] * inv[s * n_cols + te] * w_bias[s];
            sum += sigmoidf(u[(te * kStreams + s) * kUpFeatures + fe]) * rn;
        }
        x[static_cast<std::int64_t>(te) * kHidden + de] = __float2bfloat16_rn(sum * inv_streams);
    }
}

bool q8_rowsplit(const Weight& w) noexcept {
    return w.qtype == QType::Q8_G32_FP16 && w.layout == QuantLayout::RowSplit && w.qdata != nullptr &&
           w.scales != nullptr && w.padded_shape[1] % 128 == 0 && w.padded_shape[1] >= w.k;
}

bool q4_rowsplit(const Weight& w) noexcept {
    return w.qtype == QType::Q4_G64_FP16 && w.layout == QuantLayout::RowSplit && w.qdata != nullptr &&
           w.scales != nullptr && w.padded_shape[1] % 128 == 0 && w.padded_shape[1] >= w.k;
}

int bf16_stride(const Weight& w) noexcept { return w.padded_shape[1] >= w.k ? w.padded_shape[1] : w.k; }

bool bf16_contiguous(const Weight& w) noexcept {
    return w.qtype == QType::BF16 && w.layout == QuantLayout::Contiguous && w.qdata != nullptr &&
           reinterpret_cast<std::uintptr_t>(w.qdata) % 16 == 0 && bf16_stride(w) % 8 == 0;
}

Q8Codec::Params q8_params(const Weight& w) {
    return {static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint8_t*>(w.scales), w.padded_shape[1]};
}

Q4Codec::Params q4_params(const Weight& w) {
    return {static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint8_t*>(w.scales), w.padded_shape[1]};
}

Bf16Codec::Params bf16_params(const Weight& w) { return {static_cast<const bf16*>(w.qdata), bf16_stride(w)}; }

// The attention half's pending inject of hyper_connection_inject_mix (null y: none).
struct PendingInject {
    const bf16* y        = nullptr; // BF16 [H, T]
    const float* inject  = nullptr; // FP32 [S, T]
};

// Launches hc_down_partials_v4_kernel in K1a's place.
template <class Codec, class Norm, bool Fold>
void launch_down_v4(const Tensor& residual, const Tensor& norm_weight, typename Codec::Params down, int down_rows,
                    int streams, float eps, float* partial, float* inv, std::size_t rn_bytes, const PendingInject& pending,
                    cudaStream_t stream) {
    static const bool configured = [] {
        const cudaError_t e0 = cudaFuncSetAttribute(hc_down_partials_v4_kernel<Codec, Norm, Fold, 0>,
                                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                    static_cast<int>(kDownSharedBytes));
        const cudaError_t e2 = cudaFuncSetAttribute(hc_down_partials_v4_kernel<Codec, Norm, Fold, 2>,
                                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                    static_cast<int>(2 * kHidden * sizeof(float)));
        const cudaError_t e3 = cudaFuncSetAttribute(hc_down_partials_v4_kernel<Codec, Norm, Fold, 3>,
                                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                    static_cast<int>(3 * kHidden * sizeof(float)));
        const cudaError_t e4 = cudaFuncSetAttribute(hc_down_partials_v4_kernel<Codec, Norm, Fold, 4>,
                                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                    static_cast<int>(kDownSharedBytes));
        const cudaError_t error = e0 != cudaSuccess ? e0 : (e2 != cudaSuccess ? e2 : (e3 != cudaSuccess ? e3 : e4));
        if (error != cudaSuccess) {
            throw std::runtime_error(std::string("hyper_connection_mix: shared memory: ") + cudaGetErrorString(error));
        }
        return true;
    }();
    (void)configured;
    const int columns = residual.ne[1];
    const auto launch_exact = [&]<int ExactCols>() {
        constexpr int kDownV4Rows = kDownRows * 2;
        check_launch(pdl::launch_consumer({dim3((down_rows + kDownV4Rows - 1) / kDownV4Rows, streams), dim3(kDownThreads),
                                           rn_bytes, stream},
                                          hc_down_partials_v4_kernel<Codec, Norm, Fold, ExactCols>,
                                          static_cast<const bf16*>(residual.data), static_cast<const Norm*>(norm_weight.data),
                                          down, down_rows, columns, eps, partial, inv, pending.y, pending.inject),
                     "down partials v4");
    };
    if (columns == 2) {
        launch_exact.template operator()<2>();
    } else if (columns == 3) {
        launch_exact.template operator()<3>();
    } else if (columns == 4) {
        launch_exact.template operator()<4>();
    } else {
        launch_exact.template operator()<0>();
    }
}

// --- Native HMMA.16816 (mma_bf16) + Cluster-DSMEM Fused Q4 route (T = 5..16) ---

constexpr int kDownMmaRows     = 16;
constexpr int kDownMmaThreads  = 128; // 4 K-warps per 16-row CTA
constexpr int kDownMmaGroups   = kHidden / 64; // 40 groups per stream
constexpr int kDownMmaTiles    = kDownMmaGroups / 4; // 10 tiles across 4 K-warps
constexpr int kUpMmaFeatures   = 16;
constexpr int kUpMmaThreads    = 128; // 4 warps = 4 streams x 16 features
constexpr int kUpMmaMaxGroups  = kMaxRank / 64; // 8 groups max (rank <= 512)

__device__ __forceinline__ int hc_mma_swizzle64(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

__device__ __forceinline__ void hc_q4_integer_quad(std::uint8_t c0, std::uint8_t c1, unsigned& out0, unsigned& out1) {
    constexpr unsigned kBias  = 0x43084308U; // bf16 136.0 in both halves
    constexpr unsigned kMagic = 0x43004300U; // bf16 128.0 in both halves
    const unsigned packed     = (static_cast<unsigned>(c0) | (static_cast<unsigned>(c1) << 8)) ^ 0x00008888U;
    const unsigned lo         = packed & 0x00000F0FU;
    const unsigned hi         = (packed >> 4) & 0x00000F0FU;
    const unsigned pairs      = __byte_perm(lo, hi, 0x5140U);
    const unsigned b0         = __byte_perm(pairs, kMagic, 0x7150U);
    const unsigned b1         = __byte_perm(pairs, kMagic, 0x7372U);
    const __nv_bfloat162 v0   = __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&b0),
                                        *reinterpret_cast<const __nv_bfloat162*>(&kBias));
    const __nv_bfloat162 v1   = __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&b1),
                                        *reinterpret_cast<const __nv_bfloat162*>(&kBias));
    out0                      = *reinterpret_cast<const unsigned*>(&v0);
    out1                      = *reinterpret_cast<const unsigned*>(&v1);
}

__device__ __forceinline__ uint4 pack_bf16x8(const float (&v)[8]) {
    const __nv_bfloat162 p0 = __floats2bfloat162_rn(v[0], v[1]);
    const __nv_bfloat162 p1 = __floats2bfloat162_rn(v[2], v[3]);
    const __nv_bfloat162 p2 = __floats2bfloat162_rn(v[4], v[5]);
    const __nv_bfloat162 p3 = __floats2bfloat162_rn(v[6], v[7]);
    return make_uint4(*reinterpret_cast<const unsigned*>(&p0), *reinterpret_cast<const unsigned*>(&p1),
                      *reinterpret_cast<const unsigned*>(&p2), *reinterpret_cast<const unsigned*>(&p3));
}

template <int TCap>
struct DownMmaShared {
    static constexpr int kNT = TCap / 8;
    alignas(16) bf16 rn[kDownMmaGroups][TCap][64];
    union {
        struct {
            alignas(16) std::uint8_t codes[2][kDownMmaRows][4][32];
            alignas(16) std::uint16_t scales[2][kDownMmaRows][4];
        } w;
        alignas(16) float warp_partial[4][kNT][32][4];
    } u;
    alignas(16) float cta_partial[kDownMmaRows][TCap];
    alignas(16) float norm_scratch[4][TCap];
    alignas(16) float inv[TCap];
};

// K1a (HMMA.16816 + Cluster DSMEM): stages BF16 Rn once into swizzled SMEM, multiplies against
// Q4_G64_FP16 down weights via mma_bf16 across 4 K-warps (scaling FP32 per 64-K group), reduces
// across the 4 streams in DSMEM on cluster rank 0, and writes BF16 m = SiLU(z / 4) and FP32 inject.
template <class Norm, bool Pending, int TCap>
__global__ __launch_bounds__(kDownMmaThreads, TCap == 8 ? 2 : 1) void hc_down_partials_q4_mma_kernel(
    const bf16* residual, const Norm* __restrict__ norm_weight, Q4Codec::Params down, int rows, int rank,
    int columns, float eps, bf16* __restrict__ m_out, float* __restrict__ inject, float* __restrict__ inv_out,
    const bf16* pending_y, const float* pending_inject) {
    static_assert(TCap == 8 || TCap == 16);
    static_assert(kDownMmaGroups * 64 == kHidden && kDownMmaTiles * 4 == kDownMmaGroups);
    constexpr int kNT        = TCap / 8;
    constexpr int kNormOwned = 3; // (2560 / 8) = 320 chunks covered by 128 * 3 = 384 threads
    constexpr int width      = kStreams * kHidden;

    extern __shared__ __align__(16) unsigned char mma_down_smem[];
    auto* shared = reinterpret_cast<DownMmaShared<TCap>*>(mma_down_smem);

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;
    const int s    = static_cast<int>(blockIdx.y);
    const int row0 = static_cast<int>(blockIdx.x) * kDownMmaRows;
    const int row_groups = down.padded_k >> 6;
    const std::uint64_t weight_policy = l2_evict_first_policy();

    const auto stage_weights = [&](int slot, int tile) {
        const int first_group = s * kDownMmaGroups + tile * 4;
        const int local_row   = tid >> 3; // 0..15
        const int chunk       = tid & 7;  // 0..7 -> 4 groups x 2 16B halves
        const int g_row       = row0 + local_row;
        void* cdst            = &shared->u.w.codes[slot][local_row][chunk >> 1][(chunk & 1) * 16];
        if (g_row < rows) {
            cp_async_hinted<16, Cache::cg_evict_first>(
                cdst,
                down.codes + (static_cast<std::int64_t>(g_row) * row_groups + first_group) * 32 + chunk * 16,
                weight_policy);
        } else {
            store_vec(cdst, make_uint4(0U, 0U, 0U, 0U));
        }
        if (tid < 32) {
            const int srow  = tid >> 1; // 0..15
            const int pair  = tid & 1;  // 0..1
            const int sg_row = row0 + srow;
            void* sdst       = &shared->u.w.scales[slot][srow][pair * 2];
            if (sg_row < rows) {
                cp_async<4>(
                    sdst,
                    down.scales + (static_cast<std::int64_t>(sg_row) * row_groups + first_group + pair * 2) * 2);
            } else {
                *reinterpret_cast<std::uint32_t*>(sdst) = 0U;
            }
        }
    };

    // 1. Weights & norm bias first, before blocking on producer dependency.
    stage_weights(0, 0);
    cp_commit();
    const Norm* w_ptr       = norm_weight + static_cast<std::int64_t>(s) * kHidden;
    const bool aligned_norm = (reinterpret_cast<std::uintptr_t>(w_ptr) & 15U) == 0U;
    float w_bias[kNormOwned][8];
    if (aligned_norm) {
#pragma unroll
        for (int h = 0; h < kNormOwned; ++h) {
            const int i            = (tid + kDownMmaThreads * h) * 8;
            const NormGroup<Norm> ng = i < kHidden ? NormGroup<Norm>::load(w_ptr + i) : NormGroup<Norm>::zero();
            ng.floats(w_bias[h]);
#pragma unroll
            for (int j = 0; j < 8; ++j) { w_bias[h][j] = 1.0F + w_bias[h][j]; }
        }
    } else {
#pragma unroll
        for (int h = 0; h < kNormOwned; ++h) {
            const int i = (tid + kDownMmaThreads * h) * 8;
#pragma unroll
            for (int j = 0; j < 8; ++j) { w_bias[h][j] = i < kHidden ? (1.0F + norm_value(w_ptr[i + j])) : 1.0F; }
        }
    }
    pdl::wait_for_dependencies();

    // 2. Stage raw BF16 residual (or injected residual) into swizzled SMEM while computing sum_sq.
#pragma unroll 1
    for (int c = 0; c < columns; ++c) {
        const bf16* r  = residual + static_cast<std::int64_t>(c) * width + static_cast<std::int64_t>(s) * kHidden;
        float sum_sq   = 0.0F;
        float inject_c = 0.0F;
        if constexpr (Pending) { inject_c = pending_inject[static_cast<std::int64_t>(c) * kStreams + s]; }
#pragma unroll
        for (int h = 0; h < kNormOwned; ++h) {
            const int i = (tid + kDownMmaThreads * h) * 8;
            if (i < kHidden) {
                uint4 val = load_ldg<uint4>(r + i);
                if constexpr (Pending) {
                    const uint4 yv = load_ldg<uint4>(pending_y + static_cast<std::int64_t>(c) * kHidden + i);
                    val            = injected8(val, yv, inject_c);
                }
                const int grp = i >> 6;
                const int off = i & 63;
                store_vec(&shared->rn[grp][c][hc_mma_swizzle64(c, off)], val);
                float vf[8];
                bf16x8_bits_to_floats(val, vf);
#pragma unroll
                for (int j = 0; j < 8; ++j) { sum_sq += vf[j] * vf[j]; }
            }
        }
        const float ws = warp_sum(sum_sq);
        if (lane == 0) { shared->norm_scratch[warp][c] = ws; }
    }
    __syncthreads();
    if (tid < columns) {
        const float total = shared->norm_scratch[0][tid] + shared->norm_scratch[1][tid] +
                            shared->norm_scratch[2][tid] + shared->norm_scratch[3][tid];
        const float iv    = rsqrtf(total / static_cast<float>(kHidden) + eps);
        shared->inv[tid]  = iv;
        if (blockIdx.x == 0) { inv_out[static_cast<std::int64_t>(s) * columns + tid] = iv; }
    }
    __syncthreads();

    // Rescale Rn -> BF16 in-place and zero unused token columns [columns .. TCap - 1].
#pragma unroll 1
    for (int c = 0; c < TCap; ++c) {
        if (c < columns) {
            const float ic = shared->inv[c];
#pragma unroll
            for (int h = 0; h < kNormOwned; ++h) {
                const int i = (tid + kDownMmaThreads * h) * 8;
                if (i < kHidden) {
                    const int grp   = i >> 6;
                    const int off   = i & 63;
                    bf16* ptr       = &shared->rn[grp][c][hc_mma_swizzle64(c, off)];
                    const uint4 val = load_vec<uint4>(ptr);
                    float vf[8];
                    bf16x8_bits_to_floats(val, vf);
#pragma unroll
                    for (int j = 0; j < 8; ++j) { vf[j] = vf[j] * ic * w_bias[h][j]; }
                    store_vec(ptr, pack_bf16x8(vf));
                }
            }
        } else {
#pragma unroll
            for (int h = 0; h < kNormOwned; ++h) {
                const int i = (tid + kDownMmaThreads * h) * 8;
                if (i < kHidden) {
                    const int grp = i >> 6;
                    const int off = i & 63;
                    store_vec(&shared->rn[grp][c][hc_mma_swizzle64(c, off)], make_uint4(0U, 0U, 0U, 0U));
                }
            }
        }
    }
    stage_weights(1, 1);
    cp_commit();

    // 3. Double-buffered K-loop over all 40 groups of 64 (10 tiles x 4 K-warps).
    float acc[kNT][4];
#pragma unroll
    for (int ni = 0; ni < kNT; ++ni) { acc[ni][0] = acc[ni][1] = acc[ni][2] = acc[ni][3] = 0.0F; }
    for (int tile = 0; tile < kDownMmaTiles; ++tile) {
        const int slot = tile & 1;
        cp_wait<1>();
        __syncthreads();
        const int group           = tile * 4 + warp;
        float group_acc[kNT][4]   = {};
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            const int byte0 = ks * 8 + lid;
            unsigned a0, a1, a2, a3;
            hc_q4_integer_quad(shared->u.w.codes[slot][gid][warp][byte0],
                               shared->u.w.codes[slot][gid][warp][byte0 + 4], a0, a2);
            hc_q4_integer_quad(shared->u.w.codes[slot][gid + 8][warp][byte0],
                               shared->u.w.codes[slot][gid + 8][warp][byte0 + 4], a1, a3);
            const int bk = ks * 16 + (((lane >> 3) & 1) * 8);
#pragma unroll
            for (int ni = 0; ni < kNT; ++ni) {
                const int br = ni * 8 + (lane & 7);
                unsigned b0, b1;
                ldmatrix_x2(b0, b1, smem_addr(&shared->rn[group][br][hc_mma_swizzle64(br, bk)]));
                mma_bf16(group_acc[ni][0], group_acc[ni][1], group_acc[ni][2], group_acc[ni][3], a0, a1, a2, a3, b0,
                         b1);
            }
        }
        const float top = __half2float(__ushort_as_half(shared->u.w.scales[slot][gid][warp]));
        const float bot = __half2float(__ushort_as_half(shared->u.w.scales[slot][gid + 8][warp]));
#pragma unroll
        for (int ni = 0; ni < kNT; ++ni) {
            acc[ni][0] = fmaf(group_acc[ni][0], top, acc[ni][0]);
            acc[ni][1] = fmaf(group_acc[ni][1], top, acc[ni][1]);
            acc[ni][2] = fmaf(group_acc[ni][2], bot, acc[ni][2]);
            acc[ni][3] = fmaf(group_acc[ni][3], bot, acc[ni][3]);
        }
        if (tile + 2 < kDownMmaTiles) {
            __syncthreads();
            stage_weights(slot, tile + 2);
            cp_commit();
        } else {
            cp_commit();
        }
    }

    // 4. Reduce K-warps in fixed order 0..3, then DSMEM-reduce streams 0..3 on rank 0 -> BF16 m / FP32 inject.
    pdl::trigger_dependents();
    cp_wait<0>();
    __syncthreads();
    if (warp != 0) {
#pragma unroll
        for (int ni = 0; ni < kNT; ++ni) {
            store_vec(shared->u.warp_partial[warp][ni][lane],
                      make_float4(acc[ni][0], acc[ni][1], acc[ni][2], acc[ni][3]));
        }
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int ni = 0; ni < kNT; ++ni) {
            float4 sum = make_float4(acc[ni][0], acc[ni][1], acc[ni][2], acc[ni][3]);
#pragma unroll
            for (int part = 1; part < 4; ++part) {
                const float4 p = load_vec<float4>(shared->u.warp_partial[part][ni][lane]);
                sum.x += p.x;
                sum.y += p.y;
                sum.z += p.z;
                sum.w += p.w;
            }
            const int c0                     = ni * 8 + 2 * lid;
            shared->cta_partial[gid][c0]     = sum.x;
            shared->cta_partial[gid][c0 + 1] = sum.y;
            shared->cta_partial[gid + 8][c0]     = sum.z;
            shared->cta_partial[gid + 8][c0 + 1] = sum.w;
        }
    }
    auto cluster = cooperative_groups::this_cluster();
    cluster.sync();
    if (cluster.block_rank() == 0) {
        for (int item = tid; item < kDownMmaRows * TCap; item += kDownMmaThreads) {
            const int local_row = item / TCap;
            const int t         = item - local_row * TCap;
            const int row       = row0 + local_row;
            if (row < rows && t < columns) {
                float z = shared->cta_partial[local_row][t];
#pragma unroll
                for (int src = 1; src < kStreams; ++src) {
                    const auto* remote = cluster.map_shared_rank(shared, src);
                    z += remote->cta_partial[local_row][t];
                }
                const float v = z * 0.25F;
                if (row < rank) {
                    m_out[static_cast<std::int64_t>(t) * rank + row] = __float2bfloat16_rn(v * sigmoidf(v));
                } else if (inject != nullptr) {
                    inject[static_cast<std::int64_t>(t) * kStreams + (row - rank)] = 2.0F * sigmoidf(v);
                }
            }
        }
    }
    cluster.sync();
}

// K1b (HMMA.16816): each of 160 CTAs covers 16 features x 4 streams (1 stream per warp), reads
// W_up Q4 weights once into SMEM, computes u = W_up m_bf16 via mma_bf16, and collapses residual.
template <class Norm, bool Pending, int TCap>
__global__ __launch_bounds__(kUpMmaThreads, 4) void hc_up_collapse_q4_mma_kernel(
    const bf16* m_in, const float* inv_in, bf16* __restrict__ residual,
    const Norm* __restrict__ norm_weight, Q4Codec::Params up, int rank, int columns, bf16* __restrict__ x,
    const bf16* pending_y, const float* pending_inject) {
    static_assert(TCap == 8 || TCap == 16);
    static_assert(kUpMmaMaxGroups == kMaxRank / 64);
    constexpr int kNT   = TCap / 8;
    constexpr int width = kStreams * kHidden;

    extern __shared__ __align__(16) unsigned char mma_up_smem[];
    __shared__ std::uint16_t scales_smem[64][kUpMmaMaxGroups];
    __shared__ float inv_smem[kStreams * TCap];

    const int tid        = static_cast<int>(threadIdx.x);
    const int warp       = tid >> 5;
    const int lane       = tid & 31;
    const int gid        = lane >> 2;
    const int lid        = lane & 3;
    const int d0         = static_cast<int>(blockIdx.x) * kUpMmaFeatures;
    const int groups     = rank >> 6;
    const int row_groups = up.padded_k >> 6;
    const int chunks_per_row = groups * 2;
    const std::size_t codes_row_bytes = static_cast<std::size_t>(chunks_per_row) * 16U;
    const std::size_t raw_codes_bytes = static_cast<std::size_t>(64) * codes_row_bytes;
    const std::size_t u_smem_bytes    = static_cast<std::size_t>(TCap) * (kStreams * kUpMmaFeatures) * sizeof(float);
    const std::size_t codes_bytes     = raw_codes_bytes > u_smem_bytes ? raw_codes_bytes : u_smem_bytes;

    std::uint8_t* codes_smem = mma_up_smem;
    auto* m_smem             = reinterpret_cast<bf16*>(mma_up_smem + codes_bytes);
    const std::uint64_t weight_policy = l2_evict_first_policy();

    // 1. Weights & norm weights first, before wait_for_dependencies().
    for (int item = tid; item < 64 * chunks_per_row; item += kUpMmaThreads) {
        const int local_row      = item / chunks_per_row;
        const int chunk          = item - local_row * chunks_per_row;
        const int s_idx          = local_row >> 4;
        const int f_idx          = local_row & 15;
        const std::int64_t g_row = static_cast<std::int64_t>(s_idx) * kHidden + d0 + f_idx;
        cp_async_hinted<16, Cache::cg_evict_first>(
            codes_smem + static_cast<std::size_t>(local_row) * codes_row_bytes + chunk * 16,
            up.codes + (g_row * row_groups * 2 + chunk) * 16, weight_policy);
    }
    for (int item = tid; item < 64 * groups; item += kUpMmaThreads) {
        const int local_row      = item / groups;
        const int g              = item - local_row * groups;
        const int s_idx          = local_row >> 4;
        const int f_idx          = local_row & 15;
        const std::int64_t g_row = static_cast<std::int64_t>(s_idx) * kHidden + d0 + f_idx;
        scales_smem[local_row][g] =
            ld_nc_na<std::uint16_t>(up.scales + (g_row * row_groups + g) * 2);
    }
    const int fe = tid & 15;
    const int de = d0 + fe;
    float w_bias[kStreams];
#pragma unroll
    for (int s = 0; s < kStreams; ++s) {
        w_bias[s] = 1.0F + norm_value(norm_weight[static_cast<std::int64_t>(s) * kHidden + de]);
    }
    cp_commit();
    pdl::trigger_dependents();
    pdl::wait_for_dependencies();

    // 2. Stage BF16 m and inv.
    const int total_m_chunks = groups * TCap * 8;
    for (int item = tid; item < total_m_chunks; item += kUpMmaThreads) {
        const int k8 = item & 7;
        const int ct = item >> 3;
        const int c  = ct % TCap;
        const int g  = ct / TCap;
        bf16* dst    = m_smem + (static_cast<std::size_t>(g) * TCap + c) * 64 + hc_mma_swizzle64(c, k8 * 8);
        if (c < columns) {
            cp_async<16, Cache::ca>(
                dst, m_in + static_cast<std::int64_t>(c) * rank + static_cast<std::int64_t>(g) * 64 + k8 * 8);
        } else {
            store_vec(dst, make_uint4(0U, 0U, 0U, 0U));
        }
    }
    if (tid < kStreams * columns) { inv_smem[tid] = __ldg(inv_in + tid); }
    cp_commit();
    cp_wait<0>();
    __syncthreads();

    // 3. Single pass MMA over rank (4 warps = 4 streams x 16 features).
    const int row_base = warp * 16;
    float acc[kNT][4];
#pragma unroll
    for (int ni = 0; ni < kNT; ++ni) { acc[ni][0] = acc[ni][1] = acc[ni][2] = acc[ni][3] = 0.0F; }
    const std::uint8_t* rtop = codes_smem + static_cast<std::size_t>(row_base + gid) * codes_row_bytes;
    const std::uint8_t* rbot = codes_smem + static_cast<std::size_t>(row_base + gid + 8) * codes_row_bytes;
    for (int g = 0; g < groups; ++g) {
        float group_acc[kNT][4] = {};
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            const int byte0 = g * 32 + ks * 8 + lid;
            unsigned a0, a1, a2, a3;
            hc_q4_integer_quad(rtop[byte0], rtop[byte0 + 4], a0, a2);
            hc_q4_integer_quad(rbot[byte0], rbot[byte0 + 4], a1, a3);
            const int bk = ks * 16 + (((lane >> 3) & 1) * 8);
#pragma unroll
            for (int ni = 0; ni < kNT; ++ni) {
                const int br = ni * 8 + (lane & 7);
                unsigned b0, b1;
                ldmatrix_x2(b0, b1, smem_addr(&m_smem[(static_cast<std::size_t>(g) * TCap + br) * 64 +
                                                      hc_mma_swizzle64(br, bk)]));
                mma_bf16(group_acc[ni][0], group_acc[ni][1], group_acc[ni][2], group_acc[ni][3], a0, a1, a2, a3, b0,
                         b1);
            }
        }
        const float top = __half2float(__ushort_as_half(scales_smem[row_base + gid][g]));
        const float bot = __half2float(__ushort_as_half(scales_smem[row_base + gid + 8][g]));
#pragma unroll
        for (int ni = 0; ni < kNT; ++ni) {
            acc[ni][0] = fmaf(group_acc[ni][0], top, acc[ni][0]);
            acc[ni][1] = fmaf(group_acc[ni][1], top, acc[ni][1]);
            acc[ni][2] = fmaf(group_acc[ni][2], bot, acc[ni][2]);
            acc[ni][3] = fmaf(group_acc[ni][3], bot, acc[ni][3]);
        }
    }

    // 4. Write u [TCap][4 * 16] over codes_smem and collapse into x.
    __syncthreads();
    auto* u_smem = reinterpret_cast<float*>(mma_up_smem);
    constexpr int u_stride = kStreams * kUpMmaFeatures;
#pragma unroll
    for (int ni = 0; ni < kNT; ++ni) {
        const int c0 = ni * 8 + 2 * lid;
        u_smem[c0 * u_stride + row_base + gid]           = acc[ni][0];
        u_smem[(c0 + 1) * u_stride + row_base + gid]     = acc[ni][1];
        u_smem[c0 * u_stride + row_base + gid + 8]       = acc[ni][2];
        u_smem[(c0 + 1) * u_stride + row_base + gid + 8] = acc[ni][3];
    }
    __syncthreads();

    constexpr float inv_streams = 1.0F / static_cast<float>(kStreams);
#pragma unroll
    for (int p = 0; p < TCap / 8; ++p) {
        const int te = p * 8 + (tid >> 4);
        if (te < columns) {
            float sum = 0.0F;
#pragma unroll
            for (int s = 0; s < kStreams; ++s) {
                bf16* at = residual + static_cast<std::int64_t>(te) * width + static_cast<std::int64_t>(s) * kHidden + de;
                float rv;
                if constexpr (Pending) {
                    const bf16 value = hc_injected(
                        __bfloat162float(*at), pending_inject[static_cast<std::int64_t>(te) * kStreams + s],
                        __bfloat162float(pending_y[static_cast<std::int64_t>(te) * kHidden + de]));
                    *at = value;
                    rv  = __bfloat162float(value);
                } else {
                    rv = __bfloat162float(__ldg(at));
                }
                const float rn = rv * inv_smem[s * columns + te] * w_bias[s];
                sum += sigmoidf(u_smem[te * u_stride + s * kUpMmaFeatures + fe]) * rn;
            }
            x[static_cast<std::int64_t>(te) * kHidden + de] = __float2bfloat16_rn(sum * inv_streams);
        }
    }
}

template <class Norm, bool Fold, int TCap>
void launch_q4_mma_slice(const bf16* residual_ptr, const Norm* norm_ptr, Q4Codec::Params down, Q4Codec::Params up,
                         int down_rows, int rank, int streams, float eps, int slice_cols, bf16* x_ptr,
                         float* inject_ptr, void* slice_partials, const bf16* pending_y_ptr,
                         const float* pending_inject_ptr, cudaStream_t stream) {
    static const bool configured = [] {
        const cudaError_t error = cudaFuncSetAttribute(
            hc_down_partials_q4_mma_kernel<Norm, Fold, TCap>, cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(sizeof(DownMmaShared<TCap>)));
        if (error != cudaSuccess) {
            throw std::runtime_error(std::string("hyper_connection_mix: q4 mma shared memory: ") +
                                     cudaGetErrorString(error));
        }
        return true;
    }();
    (void)configured;
    auto* m_buf                   = static_cast<bf16*>(slice_partials);
    const std::size_t m_buf_bytes =
        (static_cast<std::size_t>(rank) * slice_cols * sizeof(bf16) + 255) / 256 * 256;
    auto* inv = reinterpret_cast<float*>(static_cast<std::byte*>(slice_partials) + m_buf_bytes);

    cudaLaunchAttribute attributes[2]{};
    attributes[0].id               = cudaLaunchAttributeClusterDimension;
    attributes[0].val.clusterDim.x = 1;
    attributes[0].val.clusterDim.y = static_cast<unsigned>(streams);
    attributes[0].val.clusterDim.z = 1;
    unsigned attr_count            = 1;
    if (pdl::graph_dependent(stream)) {
        attributes[1].id                                         = cudaLaunchAttributeProgrammaticStreamSerialization;
        attributes[1].val.programmaticStreamSerializationAllowed = 1;
        attr_count                                               = 2;
    }
    cudaLaunchConfig_t config{};
    config.gridDim          = dim3((down_rows + kDownMmaRows - 1) / kDownMmaRows, streams);
    config.blockDim         = dim3(kDownMmaThreads);
    config.dynamicSmemBytes = sizeof(DownMmaShared<TCap>);
    config.stream           = stream;
    config.attrs            = attributes;
    config.numAttrs         = attr_count;
    check_launch(cudaLaunchKernelEx(&config, hc_down_partials_q4_mma_kernel<Norm, Fold, TCap>, residual_ptr, norm_ptr,
                                    down, down_rows, rank, slice_cols, eps, m_buf, inject_ptr, inv, pending_y_ptr,
                                    pending_inject_ptr),
                 "down partials q4 mma");

    const int groups                = rank >> 6;
    const std::size_t codes_bytes   = std::max(
        static_cast<std::size_t>(64) * static_cast<std::size_t>(groups) * 32U,
        static_cast<std::size_t>(TCap) * (kStreams * kUpMmaFeatures) * sizeof(float));
    const std::size_t m_bytes       =
        static_cast<std::size_t>(groups) * static_cast<std::size_t>(TCap) * 64U * sizeof(bf16);
    const std::size_t up_smem_bytes = codes_bytes + m_bytes;
    check_launch(pdl::launch_consumer({dim3(kHidden / kUpMmaFeatures), dim3(kUpMmaThreads), up_smem_bytes, stream},
                                      hc_up_collapse_q4_mma_kernel<Norm, Fold, TCap>, static_cast<const bf16*>(m_buf),
                                      static_cast<const float*>(inv), const_cast<bf16*>(residual_ptr), norm_ptr, up,
                                      rank, slice_cols, x_ptr, pending_y_ptr, pending_inject_ptr),
                 "up collapse q4 mma");
}

template <class Codec, class Norm, bool Fold>
void launch(const Tensor& residual, const Tensor& norm_weight, typename Codec::Params down,
            typename Codec::Params up, int down_rows, int rank, int streams, float eps, Tensor& x, Tensor* inject,
            void* partials, const PendingInject& pending, cudaStream_t stream) {
    const int columns = residual.ne[1];
    if constexpr (std::is_same_v<Codec, Q4Codec>) {
        if (columns >= 5) {
            const auto* norm_ptr = static_cast<const Norm*>(norm_weight.data);
            const auto* res_ptr  = static_cast<const bf16*>(residual.data);
            auto* x_ptr          = static_cast<bf16*>(x.data);
            float* inject_ptr    = inject != nullptr ? static_cast<float*>(inject->data) : nullptr;
            if (columns <= 8) {
                launch_q4_mma_slice<Norm, Fold, 8>(res_ptr, norm_ptr, down, up, down_rows, rank, streams, eps,
                                                   columns, x_ptr, inject_ptr, partials, pending.y, pending.inject,
                                                   stream);
            } else {
                launch_q4_mma_slice<Norm, Fold, 16>(res_ptr, norm_ptr, down, up, down_rows, rank, streams, eps,
                                                    columns, x_ptr, inject_ptr, partials, pending.y, pending.inject,
                                                    stream);
            }
            return;
        }
    }
    static const bool configured = [] {
        const cudaError_t e1 = cudaFuncSetAttribute(hc_down_partials_kernel<Codec, Norm, Fold, 1, 2>,
                                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                    static_cast<int>(kHidden * sizeof(float)));
        const cudaError_t e1_wide = cudaFuncSetAttribute(hc_down_partials_kernel<Codec, Norm, Fold, kPassColumns, 1>,
                                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                         static_cast<int>(kDownSharedBytes));
        // The up collapse's arrays shrink to the call's own rank x columns; the attribute states the
        // worst case it may ever be given, not what a decode call allocates.
        constexpr int kUpMaxSmem = static_cast<int>(
            (static_cast<std::size_t>(kSimtMaxColumns) * kMaxRank +
             static_cast<std::size_t>(kStreams) * kSimtMaxColumns) *
            sizeof(float));
        constexpr int kUp1Smem = static_cast<int>((kMaxRank + kStreams) * sizeof(float));
        const cudaError_t e2_1 = cudaFuncSetAttribute(hc_up_collapse_kernel<Codec, Norm, Fold, 1, 2>,
                                                      cudaFuncAttributeMaxDynamicSharedMemorySize, kUp1Smem);
        const cudaError_t e2 = cudaFuncSetAttribute(hc_up_collapse_kernel<Codec, Norm, Fold>,
                                                    cudaFuncAttributeMaxDynamicSharedMemorySize, kUpMaxSmem);
        const cudaError_t e3_0 = cudaFuncSetAttribute(hc_up_collapse_v4_kernel<Codec, Norm, Fold, 0>,
                                                      cudaFuncAttributeMaxDynamicSharedMemorySize, kUpV4MaxSmem);
        const cudaError_t e3_2 = cudaFuncSetAttribute(hc_up_collapse_v4_kernel<Codec, Norm, Fold, 2>,
                                                      cudaFuncAttributeMaxDynamicSharedMemorySize, kUpV4MaxSmem);
        const cudaError_t e3_3 = cudaFuncSetAttribute(hc_up_collapse_v4_kernel<Codec, Norm, Fold, 3>,
                                                      cudaFuncAttributeMaxDynamicSharedMemorySize, kUpV4MaxSmem);
        const cudaError_t e3_4 = cudaFuncSetAttribute(hc_up_collapse_v4_kernel<Codec, Norm, Fold, 4>,
                                                      cudaFuncAttributeMaxDynamicSharedMemorySize, kUpV4MaxSmem);
        const cudaError_t e3 =
            e3_0 != cudaSuccess ? e3_0 : (e3_2 != cudaSuccess ? e3_2 : (e3_3 != cudaSuccess ? e3_3 : e3_4));
        if (e1 != cudaSuccess || e1_wide != cudaSuccess || e2_1 != cudaSuccess || e2 != cudaSuccess ||
            e3 != cudaSuccess) {
            throw std::runtime_error(
                std::string("hyper_connection_mix: shared memory: ") +
                cudaGetErrorString(e1 != cudaSuccess
                                       ? e1
                                       : (e1_wide != cudaSuccess
                                              ? e1_wide
                                              : (e2_1 != cudaSuccess
                                                     ? e2_1
                                                     : (e2 != cudaSuccess ? e2 : e3)))));
        }
        return true;
    }();
    (void)configured;
    // The fold writes the injected residual back (K1b); the plain mix only reads it.
    auto* residual_data = static_cast<bf16*>(residual.data);
    auto* partial       = static_cast<float*>(partials);
    const std::size_t partial_bytes =
        (static_cast<std::size_t>(streams) * down_rows * columns * sizeof(float) + 255) / 256 * 256;
    auto* inv = reinterpret_cast<float*>(static_cast<std::byte*>(partials) + partial_bytes);
    // Both kernels read their weights before pdl::enter(), so as programmatic dependents in a captured
    // graph they fetch them while the kernel before them finishes.
    // Rn for the columns of one pass only, so narrow decode fits beside another CTA on its SM.
    const std::size_t rn_bytes =
        static_cast<std::size_t>(std::min(columns, kPassColumns)) * kHidden * sizeof(float);
    // Two or more columns take v4's read scheduling (above); the v4 K1a reads the norm weights as
    // 16-byte vectors.
    const bool v4 = columns >= 2 && reinterpret_cast<std::uintptr_t>(norm_weight.data) % 16 == 0;
    if (v4) {
        launch_down_v4<Codec, Norm, Fold>(residual, norm_weight, down, down_rows, streams, eps, partial, inv, rn_bytes,
                                          pending, stream);
    } else if (columns == 1) {
        check_launch(pdl::launch_consumer({dim3((down_rows + kDownRows - 1) / kDownRows, streams), dim3(kDownThreads),
                                           rn_bytes, stream},
                                          hc_down_partials_kernel<Codec, Norm, Fold, 1, 2>,
                                          static_cast<const bf16*>(residual.data),
                                          static_cast<const Norm*>(norm_weight.data), down, down_rows, columns, eps,
                                          partial, inv, pending.y, pending.inject),
                     "down partials");
    } else {
        check_launch(pdl::launch_consumer({dim3((down_rows + kDownRows - 1) / kDownRows, streams), dim3(kDownThreads),
                                           rn_bytes, stream},
                                          hc_down_partials_kernel<Codec, Norm, Fold, kPassColumns, 1>,
                                          static_cast<const bf16*>(residual.data),
                                          static_cast<const Norm*>(norm_weight.data), down, down_rows, columns, eps,
                                          partial, inv, pending.y, pending.inject),
                     "down partials");
    }
    if (v4) {
        // The v4 collapse's arrays are sized to this call: m [T][rank], u [T][S][features], inv [S][T].
        const std::size_t up_v4_smem =
            (static_cast<std::size_t>(columns) * (static_cast<std::size_t>(rank) + kStreams * kUpFeatures) +
             static_cast<std::size_t>(kStreams) * static_cast<std::size_t>(columns)) *
            sizeof(float);
        const auto launch_up_v4 = [&]<int ExactCols>() {
            check_launch(pdl::launch_consumer({dim3((kHidden + kUpFeatures - 1) / kUpFeatures), dim3(kUpV4Threads),
                                               up_v4_smem, stream},
                                              hc_up_collapse_v4_kernel<Codec, Norm, Fold, ExactCols>,
                                              static_cast<const float*>(partial),
                                              static_cast<const float*>(inv), residual_data,
                                              static_cast<const Norm*>(norm_weight.data), up, rank, down_rows, columns,
                                              inject != nullptr ? static_cast<float*>(inject->data) : nullptr,
                                              static_cast<bf16*>(x.data), pending.y, pending.inject),
                         "up collapse v4");
        };
        if (columns == 2) {
            launch_up_v4.template operator()<2>();
        } else if (columns == 3) {
            launch_up_v4.template operator()<3>();
        } else if (columns == 4) {
            launch_up_v4.template operator()<4>();
        } else {
            launch_up_v4.template operator()<0>();
        }
        return;
    }
    // The up collapse's shared arrays are m[columns][rank] and inv[streams][columns]; sizing them
    // from the call's own values (rather than kHcMixFusedMaxColumns * kMaxRank) is what lets
    // decode's ~5 KB sit beside a down CTA on its SM instead of reserving 128 KB.
    const std::size_t up_smem =
        (static_cast<std::size_t>(columns) * static_cast<std::size_t>(rank) +
         static_cast<std::size_t>(streams) * static_cast<std::size_t>(columns)) *
        sizeof(float);
    if (columns == 1) {
        check_launch(pdl::launch_consumer({dim3((kHidden + kUpFeatures * 2 - 1) / (kUpFeatures * 2)), dim3(kUpThreads), up_smem, stream},
                                          hc_up_collapse_kernel<Codec, Norm, Fold, 1, 2>, static_cast<const float*>(partial),
                                          static_cast<const float*>(inv), residual_data,
                                          static_cast<const Norm*>(norm_weight.data), up, rank, down_rows, columns,
                                          inject != nullptr ? static_cast<float*>(inject->data) : nullptr,
                                          static_cast<bf16*>(x.data), pending.y, pending.inject),
                     "up collapse");
    } else {
        check_launch(pdl::launch_consumer({dim3((kHidden + kUpFeatures - 1) / kUpFeatures), dim3(kUpThreads), up_smem, stream},
                                          hc_up_collapse_kernel<Codec, Norm, Fold>, static_cast<const float*>(partial),
                                          static_cast<const float*>(inv), residual_data,
                                          static_cast<const Norm*>(norm_weight.data), up, rank, down_rows, columns,
                                          inject != nullptr ? static_cast<float*>(inject->data) : nullptr,
                                          static_cast<bf16*>(x.data), pending.y, pending.inject),
                     "up collapse");
    }
}

template <class Codec>
void launch_norm(const Tensor& residual, const Tensor& norm_weight, typename Codec::Params down,
                 typename Codec::Params up, int down_rows, int rank, int streams, float eps, Tensor& x, Tensor* inject,
                 void* partials, const PendingInject& pending, cudaStream_t stream) {
    const auto with = [&]<class Norm>() {
        if (pending.y != nullptr) {
            launch<Codec, Norm, true>(residual, norm_weight, down, up, down_rows, rank, streams, eps, x, inject, partials,
                                      pending, stream);
        } else {
            launch<Codec, Norm, false>(residual, norm_weight, down, up, down_rows, rank, streams, eps, x, inject,
                                       partials, pending, stream);
        }
    };
    if (norm_weight.dtype == DType::FP32) {
        with.template operator()<float>();
    } else {
        with.template operator()<bf16>();
    }
}

void mix_fused(const Tensor& residual, const Tensor& norm_weight, const Weight& down, const Weight& up,
               std::int32_t streams, std::int32_t rank, float eps, Tensor& x, Tensor* inject, void* partials,
               const PendingInject& pending, cudaStream_t stream) {
    if (!hc_mix_fused_supported(down, up, x.ne[0], streams, rank, residual.ne[1])) {
        throw std::invalid_argument("hyper_connection_mix: the fused route does not serve this problem");
    }
    if (inject != nullptr && down.n != rank + streams) {
        throw std::invalid_argument("hyper_connection_mix: injects need the down projection's injection rows");
    }
    if (down.qtype == QType::Q8_G32_FP16) {
        launch_norm<Q8Codec>(residual, norm_weight, q8_params(down), q8_params(up), down.n, rank, streams, eps, x,
                             inject, partials, pending, stream);
    } else if (down.qtype == QType::Q4_G64_FP16) {
        launch_norm<Q4Codec>(residual, norm_weight, q4_params(down), q4_params(up), down.n, rank, streams, eps, x,
                             inject, partials, pending, stream);
    } else {
        launch_norm<Bf16Codec>(residual, norm_weight, bf16_params(down), bf16_params(up), down.n, rank, streams, eps,
                               x, inject, partials, pending, stream);
    }
}

} // namespace

bool hc_mix_fused_supported(const Weight& down, const Weight& up, std::int32_t hidden, std::int32_t streams,
                            std::int32_t rank, std::int32_t columns) noexcept {
    const bool q8   = q8_rowsplit(down) && q8_rowsplit(up);
    const bool q4   = q4_rowsplit(down) && q4_rowsplit(up);
    const bool bf16 = bf16_contiguous(down) && bf16_contiguous(up);
    const bool q4_mma_aligned =
        q4 && (rank % 64 == 0) &&
        (reinterpret_cast<std::uintptr_t>(down.qdata) & 15U) == 0U &&
        (reinterpret_cast<std::uintptr_t>(down.scales) & 3U) == 0U &&
        (reinterpret_cast<std::uintptr_t>(up.qdata) & 15U) == 0U;
    const std::int32_t max_columns =
        q4 ? (q4_mma_aligned ? kHcMixFusedMaxColumnsQ4 : kSimtMaxColumns)
           : (q8 ? kHcMixFusedMaxColumnsQ8 : kHcMixFusedMaxColumnsBf16);
    return (q8 || q4 || bf16) && columns >= 1 && columns <= max_columns && hidden == kHidden && streams == kStreams &&
           rank > 0 && rank % 8 == 0 && rank <= kMaxRank && down.k == kStreams * kHidden &&
           (down.n == rank || down.n == rank + kStreams) && up.n == kStreams * kHidden && up.k == rank;
}

std::size_t hc_mix_fused_workspace_bytes(std::int32_t down_rows, std::int32_t streams, std::int32_t columns) noexcept {
    const auto align = [](std::size_t bytes) { return (bytes + 255) / 256 * 256; };
    return align(static_cast<std::size_t>(streams) * down_rows * columns * sizeof(float)) +
           align(static_cast<std::size_t>(streams) * columns * sizeof(float));
}

void hc_mix_fused(const Tensor& residual, const Tensor& norm_weight, const Weight& down, const Weight& up,
                  std::int32_t streams, std::int32_t rank, float eps, Tensor& x, Tensor* inject, void* partials,
                  cudaStream_t stream) {
    mix_fused(residual, norm_weight, down, up, streams, rank, eps, x, inject, partials, PendingInject{}, stream);
}

void hc_inject_mix_fused(const Tensor& y, const Tensor& pending_inject, Tensor& residual, const Tensor& norm_weight,
                         const Weight& down, const Weight& up, std::int32_t streams, std::int32_t rank, float eps,
                         Tensor& x, Tensor* inject, void* partials, cudaStream_t stream) {
    const std::int32_t T = residual.ne[1];
    if (y.data == nullptr || y.dtype != DType::BF16 || !y.is_contiguous() || y.ne[0] != kHidden || y.ne[1] != T ||
        pending_inject.data == nullptr || pending_inject.dtype != DType::FP32 || !pending_inject.is_contiguous() ||
        pending_inject.ne[0] != streams || pending_inject.ne[1] != T) {
        throw std::invalid_argument("hyper_connection_inject_mix: y or the pending inject disagree with the residual");
    }
    if (inject != nullptr && inject->data == pending_inject.data) {
        throw std::invalid_argument("hyper_connection_inject_mix: the mix's injects would overwrite the pending ones");
    }
    mix_fused(residual, norm_weight, down, up, streams, rank, eps, x, inject, partials,
              PendingInject{static_cast<const bf16*>(y.data), static_cast<const float*>(pending_inject.data)}, stream);
}

} // namespace infernix::ops::detail
