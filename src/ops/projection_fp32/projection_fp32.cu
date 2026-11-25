#include "infernix/ops/projection_fp32.h"
#include "ops/projection_fp32/projection_fp32_route.h"
#include "ops/projection_fp32/projection_fp32_mma.h"
#include "ops/common/memory.cuh"
#include "core/device.h"
#include "core/pdl.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace infernix::ops {
namespace {

using bf16 = __nv_bfloat16;

constexpr int kThreads     = 256;
constexpr int kWarps       = kThreads / 32;
constexpr int kRowsPerWarp = 4;
constexpr int kRowsPerCta  = kWarps * kRowsPerWarp;
constexpr int kColumns     = 8; // columns staged per CTA
constexpr int kMaxSegments = 4;
constexpr int kMaxK        = 3072;
constexpr int kMaxLaneChunks = kMaxK / 8 / 32; // 8-element chunks one lane reads per row
// R4 experiment switch (P1-K1): hold the narrow q8 kernel's preloaded scales as packed FP16 pairs.
constexpr bool kQ8PackScales = true;
// The same preload/packed-scale structure for the q4_g64_fp16 head (the proposal head): its codes are
// 4-byte words, so the streaming load is the u32 form of the evict-first helper.
constexpr bool kQ4PackScales = true;

struct Segments {
    const bf16* data[kMaxSegments];
    int begin[kMaxSegments]; // first output row of each segment
    int count;
};

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("projection_fp32: ") + message); }
}

bool aligned16(const void* p) { return reinterpret_cast<std::uintptr_t>(p) % 16 == 0; }

// Issues lane `lane`'s chunk loads of `row` (chunks lane, lane + 32, ...) into `loaded`.
__device__ __forceinline__ void load_row(const Segments& segments, int row, int k, int chunks, int lane,
                                         uint4 (&loaded)[kMaxLaneChunks]) {
    const bf16* data = segments.data[0];
    int begin        = 0;
    if (segments.count > 1) {
        if (row >= segments.begin[1]) {
            data  = segments.data[1];
            begin = segments.begin[1];
        }
        if (segments.count > 2 && row >= segments.begin[2]) {
            data  = segments.data[2];
            begin = segments.begin[2];
        }
        if (segments.count > 3 && row >= segments.begin[3]) {
            data  = segments.data[3];
            begin = segments.begin[3];
        }
    }
    const auto* w = reinterpret_cast<const uint4*>(data + static_cast<std::size_t>(row - begin) * k);
#pragma unroll
    for (int i = 0; i < kMaxLaneChunks; ++i) {
        const int chunk = lane + 32 * i;
        if (chunk < chunks) { loaded[i] = w[chunk]; }
    }
}

// Warp w of a CTA owns rows first + w, first + w + Warps, ...; lane l accumulates the 8-element
// chunks l, l + 32, ... in order and the warp reduces by a fixed butterfly, so an output's bits
// depend only on K. A lane issues all of a row's chunk loads before its FMAs, and the first row's
// loads before x is staged, so the weights' DRAM trip overlaps the staging instead of following
// it. Decode-width calls give each warp one row (Threads = 128, RowsPerWarp = 1: 129 CTAs for a
// 513-row router instead of 17), so a router's latency is one DRAM round trip instead of ~40; wide
// calls keep four rows per warp, which stages x in fewer CTAs. Both mappings produce the same bits.
// One or four tail rows (e.g. Qwen4Exp's 1-row shared_score after the 512-row router in router-mma).
// Stages the tail weight rows once into shared memory (5 KiB per row at K = 2560), then each warp
// computes one output column's dot product in registers with the narrow kernels' exact lane-chunk order
// and warp butterfly, writing to out[col * out_stride + row].
constexpr int kTailRowWarps = 8;
constexpr int kTailRowThreads = kTailRowWarps * 32;

__global__ void __launch_bounds__(kTailRowThreads, 4)
    projection_tail_row_kernel(const bf16* __restrict__ x, int k, int columns, const bf16* __restrict__ w,
                               float* __restrict__ out, int out_stride) {
    extern __shared__ uint4 w_staged[]; // [chunks]
    const int chunks = k / 8;
    const int tid    = static_cast<int>(threadIdx.x);
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int column = static_cast<int>(blockIdx.x) * kTailRowWarps + warp;
    const auto* ws   = reinterpret_cast<const uint4*>(w);
    for (int c = tid; c < chunks; c += kTailRowThreads) { w_staged[c] = ws[c]; }
    __syncthreads();
    if (column >= columns) { return; }
    const auto* xs = reinterpret_cast<const uint4*>(x + static_cast<std::size_t>(column) * k);
    uint4 x_loaded[kMaxLaneChunks];
#pragma unroll
    for (int i = 0; i < kMaxLaneChunks; ++i) {
        const int chunk = lane + 32 * i;
        if (chunk < chunks) { x_loaded[i] = xs[chunk]; }
    }
    float sum = 0.0F;
#pragma unroll
    for (int i = 0; i < kMaxLaneChunks; ++i) {
        const int chunk = lane + 32 * i;
        if (chunk >= chunks) { break; }
        const uint4 wp = w_staged[chunk];
        const uint4 xp = x_loaded[i];
        const bf16* wv = reinterpret_cast<const bf16*>(&wp);
        const bf16* xv = reinterpret_cast<const bf16*>(&xp);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            sum = fmaf(__bfloat162float(wv[j]), __bfloat162float(xv[j]), sum);
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) { sum += __shfl_xor_sync(0xFFFFFFFFU, sum, offset); }
    if (lane == 0) { out[static_cast<std::size_t>(column) * out_stride] = sum; }
}

template <int Threads, int RowsPerWarp, int MaxCols = kColumns>
__global__ void __launch_bounds__(Threads, (Threads == 128 && MaxCols == 1) ? 8 : (Threads == 128 ? 4 : 2))
    projection_kernel(const bf16* __restrict__ x, int k, int columns, int rows, Segments segments, float* __restrict__ out) {
    constexpr int Warps = Threads / 32;
    constexpr bool kExactCols = MaxCols < kColumns;
    // 16-byte x chunks of the CTA's columns, as one unrolled batch of at most this many per thread.
    constexpr int kStagePerThread = (MaxCols * kMaxK / 8 + Threads - 1) / Threads;
    extern __shared__ uint4 staged[];
    const int first_column = kExactCols ? 0 : (blockIdx.y * kColumns);
    const int count        = kExactCols ? MaxCols : min(MaxCols, columns - first_column);
    const int chunks       = k / 8;
    const int lane         = threadIdx.x % 32;
    const int first_row    = blockIdx.x * (Warps * RowsPerWarp) + threadIdx.x / 32;
    uint4 loaded[kMaxLaneChunks];
    if (first_row < rows) { load_row(segments, first_row, k, chunks, lane, loaded); }
    // Launched as a programmatic dependent: the weights are immutable, so their loads precede the wait.
    pdl::enter();
    // The CTA's columns of x are contiguous, so chunk i of the staging is chunk i of x from there.
    const auto* xs = reinterpret_cast<const uint4*>(x + static_cast<std::size_t>(first_column) * k);
#pragma unroll
    for (int u = 0; u < kStagePerThread; ++u) {
        const int i = static_cast<int>(threadIdx.x) + u * Threads;
        if (i < count * chunks) {
            const auto dst = static_cast<unsigned>(__cvta_generic_to_shared(staged + i));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(xs + i) : "memory");
        }
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();
    for (int r = 0; r < RowsPerWarp; ++r) {
        const int row = first_row + r * Warps;
        if (row >= rows) { return; }
        if (r > 0) { load_row(segments, row, k, chunks, lane, loaded); }
        float sums[MaxCols] = {};
#pragma unroll
        for (int i = 0; i < kMaxLaneChunks; ++i) {
            const int chunk = lane + 32 * i;
            if (chunk >= chunks) { break; }
            const uint4 packed = loaded[i];
            const bf16* wv     = reinterpret_cast<const bf16*>(&packed);
            float wf[8];
            for (int i = 0; i < 8; ++i) { wf[i] = __bfloat162float(wv[i]); }
            for (int column = 0; column < MaxCols; ++column) {
                if (kExactCols || column < count) {
                    const uint4 xp = staged[column * chunks + chunk];
                    const bf16* xv = reinterpret_cast<const bf16*>(&xp);
                    for (int i = 0; i < 8; ++i) { sums[column] = fmaf(wf[i], __bfloat162float(xv[i]), sums[column]); }
                }
            }
        }
        for (int column = 0; column < MaxCols; ++column) {
            float v = sums[column];
            for (int offset = 16; offset > 0; offset >>= 1) { v += __shfl_xor_sync(0xFFFFFFFFU, v, offset); }
            if (lane == 0 && (kExactCols || column < count)) {
                out[static_cast<std::size_t>(first_column + column) * rows + row] = v;
            }
        }
    }
}

// Wide calls (prefill chunks). The narrow mapping re-reads a CTA's weight rows for every eight
// columns and reads nine 16-byte words per 64 FMAs: a 4K-token router call takes ~0.75 ms at
// ~14 TFLOP/s. Here a CTA owns a kWideRows x kWideColumns output tile and walks K in slices of 256
// elements, which is exactly one 8-element chunk per lane: in slice s lane l handles chunk
// 32 s + l, the narrow kernels' l-th lane partial in the same order. Each slice's weight rows and
// x columns are staged with cp.async (double-buffered); a warp owns a 4-row x 8-column sub-tile, so
// a lane loads 12 16-byte words per 256 FMAs. After the last slice each warp reduces its partials
// by the same butterfly, so the bits equal the narrow kernels' (an output's summation still
// depends only on K).
// 512 threads: a lane's 32 accumulators and 32 widened weights need more than the 64 registers a
// 1,024-thread CTA allows (that version spilled and ran at 1.6 ms).
constexpr int kWideRows        = 32; // output rows per CTA
constexpr int kWideColumns     = 16; // output columns per CTA
constexpr int kWideTileRows    = 4;  // per warp
constexpr int kWideTileColumns = 8;
constexpr int kWideRowWarps    = kWideRows / kWideTileRows;       // 8
constexpr int kWideColumnWarps = kWideColumns / kWideTileColumns; // 2
constexpr int kWideThreads     = kWideRowWarps * kWideColumnWarps * 32;
constexpr int kSliceChunks     = 32; // chunks per K slice: one per lane
constexpr int kStageWords      = (kWideRows + kWideColumns) * kSliceChunks; // 16-byte words per stage
constexpr int kWideSharedBytes = 2 * kStageWords * 16;
constexpr int kWideMinColumns  = 64; // narrower calls keep the narrow mapping (same bits)
static_assert(kWideThreads == 512 && kStageWords % kWideThreads == 0);

// The address of chunk `chunk` (8 elements) of output row `row`.
__device__ __forceinline__ const uint4* row_chunk(const Segments& segments, int row, int k, int chunk) {
    const bf16* data = segments.data[0];
    int begin        = 0;
    if (segments.count > 1) {
        if (row >= segments.begin[1]) {
            data  = segments.data[1];
            begin = segments.begin[1];
        }
        if (segments.count > 2 && row >= segments.begin[2]) {
            data  = segments.data[2];
            begin = segments.begin[2];
        }
        if (segments.count > 3 && row >= segments.begin[3]) {
            data  = segments.data[3];
            begin = segments.begin[3];
        }
    }
    return reinterpret_cast<const uint4*>(data + static_cast<std::size_t>(row - begin) * k) + chunk;
}

__device__ __forceinline__ void widen8(const uint4& packed, float (&out)[8]) {
    const auto* v = reinterpret_cast<const bf16*>(&packed);
#pragma unroll
    for (int i = 0; i < 8; ++i) { out[i] = __bfloat162float(v[i]); }
}

// 16 bytes from `src` to shared `dst`, or zeros when `valid` is false (src is then not read).
__device__ __forceinline__ void stage16(uint4* dst, const void* src, bool valid) {
    const auto d = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(valid ? 16 : 0) : "memory");
}

// row_block_offset: the first row block (router-mma computes the rows before it on tensor cores).
__global__ void __launch_bounds__(kWideThreads, 1)
    projection_wide_kernel(const bf16* __restrict__ x, int k, int columns, int rows, Segments segments, float* __restrict__ out,
                           int row_block_offset = 0) {
    extern __shared__ uint4 stage[]; // [2][rows then columns][kSliceChunks]
    const int chunks       = k / 8;
    const int slices       = (chunks + kSliceChunks - 1) / kSliceChunks;
    const int first_row    = (static_cast<int>(blockIdx.x) + row_block_offset) * kWideRows;
    const int first_column = static_cast<int>(blockIdx.y) * kWideColumns;
    const auto issue = [&](int slice, int buffer) {
        uint4* base = stage + buffer * kStageWords;
#pragma unroll
        for (int u = 0; u < kStageWords / kWideThreads; ++u) {
            const int i     = static_cast<int>(threadIdx.x) + u * kWideThreads;
            const int line  = i / kSliceChunks;
            const int chunk = slice * kSliceChunks + i % kSliceChunks;
            if (line < kWideRows) {
                const int row    = first_row + line;
                const bool valid = row < rows && chunk < chunks;
                stage16(base + i, valid ? static_cast<const void*>(row_chunk(segments, row, k, chunk)) : x, valid);
            } else {
                const int column = first_column + line - kWideRows;
                const bool valid = column < columns && chunk < chunks;
                stage16(base + i, valid ? static_cast<const void*>(x + static_cast<std::size_t>(column) * k + 8 * chunk) : x,
                        valid);
            }
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    };
    const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
    const int r0 = (warp % kWideRowWarps) * kWideTileRows;
    const int c0 = (warp / kWideRowWarps) * kWideTileColumns;
    float sums[kWideTileRows][kWideTileColumns] = {};
    issue(0, 0);
    for (int s = 0; s < slices; ++s) {
        if (s + 1 < slices) {
            issue(s + 1, (s + 1) & 1);
            asm volatile("cp.async.wait_group 1;\n" ::: "memory");
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        }
        __syncthreads();
        if (s * kSliceChunks + lane < chunks) { // the narrow kernels' loop ends at the last chunk too
            const uint4* base = stage + (s & 1) * kStageWords;
            float wf[kWideTileRows][8];
#pragma unroll
            for (int r = 0; r < kWideTileRows; ++r) { widen8(base[(r0 + r) * kSliceChunks + lane], wf[r]); }
#pragma unroll
            for (int c = 0; c < kWideTileColumns; ++c) {
                float xf[8];
                widen8(base[(kWideRows + c0 + c) * kSliceChunks + lane], xf);
#pragma unroll
                for (int r = 0; r < kWideTileRows; ++r) {
#pragma unroll
                    for (int i = 0; i < 8; ++i) { sums[r][c] = fmaf(wf[r][i], xf[i], sums[r][c]); }
                }
            }
        }
        __syncthreads(); // the next iteration's issue overwrites this buffer
    }
#pragma unroll
    for (int r = 0; r < kWideTileRows; ++r) {
        const int row = first_row + r0 + r;
#pragma unroll
        for (int c = 0; c < kWideTileColumns; ++c) {
            const int column = first_column + c0 + c;
            float v          = sums[r][c];
            for (int offset = 16; offset > 0; offset >>= 1) { v += __shfl_xor_sync(0xFFFFFFFFU, v, offset); }
            if (lane == 0 && row < rows && column < columns) { out[static_cast<std::size_t>(column) * rows + row] = v; }
        }
    }
}

// The q8_g32_fp16 form: row r's codes start at r * padded_k, its FP16 scales at r * padded_k / 32.
// Lane l reads the same 8-element chunks as the BF16 form, decodes each weight exactly and
// accumulates in the same order.
template <int MaxCols = kColumns>
__global__ void __launch_bounds__(kThreads, MaxCols == 1 ? 3 : 2)
    projection_q8_kernel(const bf16* __restrict__ x, int k, int padded_k, int columns, int rows,
                         const std::int8_t* __restrict__ codes, const __half* __restrict__ scales,
                         float* __restrict__ out) {
    extern __shared__ uint4 staged[];
    const int first_column = blockIdx.y * kColumns;
    const int count        = min(MaxCols, columns - first_column);
    const int chunks       = k / 8;
    pdl::enter();
    for (int i = threadIdx.x; i < count * chunks; i += blockDim.x) {
        const int column = i / chunks, chunk = i % chunks;
        staged[column * chunks + chunk] =
            reinterpret_cast<const uint4*>(x + static_cast<std::size_t>(first_column + column) * k)[chunk];
    }
    __syncthreads();
    const int lane = threadIdx.x % 32;
    for (int r = 0; r < kRowsPerWarp; ++r) {
        const int row = blockIdx.x * kRowsPerCta + r * kWarps + threadIdx.x / 32;
        if (row >= rows) { return; }
        const auto* w      = reinterpret_cast<const uint2*>(codes + static_cast<std::size_t>(row) * padded_k);
        const __half* srow = scales + static_cast<std::size_t>(row) * (padded_k / 32);
        // Expose independent row loads before the dependent FMA chain. Compile-time slots keep
        // these arrays in registers; consume chunks in the original lane + 32*i order.
        uint2 loaded[kMaxLaneChunks];
        float sums[MaxCols] = {};
        if constexpr (kQ8PackScales) {
            // A lane's slot i and slot i + 1 read scale indices floor(l/4) + 8i and + 8(i + 1): eight
            // half-entries apart, never neighbours. Each pair of live scales is therefore held as two
            // independently loaded FP16 values packed into one 32-bit register and widened at use.
            __half2 loaded_scale[kMaxLaneChunks / 2];
#pragma unroll
            for (int i = 0; i < kMaxLaneChunks; i += 2) {
                __half s[2] = {__ushort_as_half(0), __ushort_as_half(0)};
#pragma unroll
                for (int u = 0; u < 2; ++u) {
                    const int chunk = lane + 32 * (i + u);
                    if (chunk < chunks) {
                        // Read once per call (a full-vocabulary head is 636 MB): it leaves L2 first.
                        loaded[i + u] = ld_nc_na_evict_first(w + chunk);
                        s[u]          = srow[chunk / 4];
                    }
                }
                loaded_scale[i / 2] = __halves2half2(s[0], s[1]);
            }
#pragma unroll
            for (int i = 0; i < kMaxLaneChunks; ++i) {
                const int chunk = lane + 32 * i;
                if (chunk >= chunks) { break; }
                const __half2 pair  = loaded_scale[i / 2];
                const float   scale = __half2float((i & 1) ? pair.y : pair.x);
                const auto* wv      = reinterpret_cast<const std::int8_t*>(&loaded[i]);
                float wf[8];
                for (int j = 0; j < 8; ++j) { wf[j] = static_cast<float>(wv[j]) * scale; }
                for (int column = 0; column < MaxCols; ++column) {
                    if (MaxCols == 1 || column < count) {
                        const uint4 xp = staged[column * chunks + chunk];
                        const bf16* xv = reinterpret_cast<const bf16*>(&xp);
                        for (int j = 0; j < 8; ++j) {
                            sums[column] = fmaf(wf[j], __bfloat162float(xv[j]), sums[column]);
                        }
                    }
                }
            }
        } else {
            float loaded_scale[kMaxLaneChunks];
#pragma unroll
            for (int i = 0; i < kMaxLaneChunks; ++i) {
                const int chunk = lane + 32 * i;
                if (chunk < chunks) {
                    // Read once per call (a full-vocabulary head is 636 MB): it leaves L2 first.
                    loaded[i] = ld_nc_na_evict_first(w + chunk);
                    loaded_scale[i] = __half2float(srow[chunk / 4]);
                }
            }
#pragma unroll
            for (int i = 0; i < kMaxLaneChunks; ++i) {
                const int chunk = lane + 32 * i;
                if (chunk >= chunks) { break; }
                const uint2 packed = loaded[i];
                const float scale  = loaded_scale[i];
                const auto* wv     = reinterpret_cast<const std::int8_t*>(&packed);
                float wf[8];
                for (int j = 0; j < 8; ++j) { wf[j] = static_cast<float>(wv[j]) * scale; }
                for (int column = 0; column < MaxCols; ++column) {
                    if (MaxCols == 1 || column < count) {
                        const uint4 xp = staged[column * chunks + chunk];
                        const bf16* xv = reinterpret_cast<const bf16*>(&xp);
                        for (int j = 0; j < 8; ++j) {
                            sums[column] = fmaf(wf[j], __bfloat162float(xv[j]), sums[column]);
                        }
                    }
                }
            }
        }
        for (int column = 0; column < MaxCols; ++column) {
            float v = sums[column];
            for (int offset = 16; offset > 0; offset >>= 1) { v += __shfl_xor_sync(0xFFFFFFFFU, v, offset); }
            if (lane == 0 && (MaxCols == 1 || column < count)) {
                out[static_cast<std::size_t>(first_column + column) * rows + row] = v;
            }
        }
    }
}

// The q4_g64_fp16 form: row r's two's-complement nibbles start at r * padded_k / 2 (low nibble
// first), its FP16 scales at r * padded_k / 64. Same chunks and order as the other forms.
template <int MaxCols = kColumns>
__global__ void __launch_bounds__(kThreads, 3)
    projection_q4_kernel(const bf16* __restrict__ x, int k, int padded_k, int columns, int rows,
                         const std::uint8_t* __restrict__ codes, const __half* __restrict__ scales,
                         float* __restrict__ out) {
    constexpr bool kExactCols = MaxCols < kColumns;
    extern __shared__ uint4 staged[];
    const int first_column = kExactCols ? 0 : (blockIdx.y * kColumns);
    const int count        = kExactCols ? MaxCols : min(MaxCols, columns - first_column);
    const int chunks       = k / 8;
    pdl::enter();
    for (int i = threadIdx.x; i < count * chunks; i += blockDim.x) {
        const int column = i / chunks, chunk = i % chunks;
        staged[column * chunks + chunk] =
            reinterpret_cast<const uint4*>(x + static_cast<std::size_t>(first_column + column) * k)[chunk];
    }
    __syncthreads();
    const int lane = threadIdx.x % 32;
    for (int r = 0; r < kRowsPerWarp; ++r) {
        const int row = blockIdx.x * kRowsPerCta + r * kWarps + threadIdx.x / 32;
        if (row >= rows) { return; }
        const auto* w      = reinterpret_cast<const std::uint32_t*>(codes + static_cast<std::size_t>(row) * (padded_k / 2));
        const __half* srow = scales + static_cast<std::size_t>(row) * (padded_k / 64);
        // Expose the independent row loads before the dependent FMA chain, in the same slot order as
        // the q8 form: a lane's slot i and slot i + 1 read scale indices floor(l/8) + 4i and + 4(i + 1),
        // four half-entries apart (never neighbours, so a half2 load is not interchangeable). Hold the
        // pair as two independently loaded FP16 values packed into one 32-bit register and widen at use.
        // Codes are 4-byte words, read with the streaming hint (a full-vocabulary head is 341 MB and
        // leaves L2 first).
        float sums[MaxCols] = {};
        if constexpr (kQ4PackScales) {
            std::uint32_t loaded[kMaxLaneChunks];
            __half2 loaded_scale[kMaxLaneChunks / 2];
#pragma unroll
            for (int i = 0; i < kMaxLaneChunks; i += 2) {
                __half s[2] = {__ushort_as_half(0), __ushort_as_half(0)};
#pragma unroll
                for (int u = 0; u < 2; ++u) {
                    const int chunk = lane + 32 * (i + u);
                    if (chunk < chunks) {
                        loaded[i + u] = ld_nc_na_evict_first_u32(w + chunk);
                        s[u]          = srow[chunk / 8];
                    }
                }
                loaded_scale[i / 2] = __halves2half2(s[0], s[1]);
            }
#pragma unroll
            for (int i = 0; i < kMaxLaneChunks; ++i) {
                const int chunk = lane + 32 * i;
                if (chunk >= chunks) { break; }
                const __half2 pair  = loaded_scale[i / 2];
                const float   scale = __half2float((i & 1) ? pair.y : pair.x);
                const std::uint32_t packed = loaded[i];
                float wf[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int u = static_cast<int>((packed >> (4 * j)) & 0xFU);
                    wf[j]       = static_cast<float>(u >= 8 ? u - 16 : u) * scale;
                }
                for (int column = 0; column < MaxCols; ++column) {
                    if (kExactCols || column < count) {
                        const uint4 xp = staged[column * chunks + chunk];
                        const bf16* xv = reinterpret_cast<const bf16*>(&xp);
                        for (int j = 0; j < 8; ++j) {
                            sums[column] = fmaf(wf[j], __bfloat162float(xv[j]), sums[column]);
                        }
                    }
                }
            }
        } else {
            for (int chunk = lane; chunk < chunks; chunk += 32) {
                const std::uint32_t packed = w[chunk];
                const float scale          = __half2float(srow[chunk / 8]);
                float wf[8];
                for (int i = 0; i < 8; ++i) {
                    const int u = static_cast<int>((packed >> (4 * i)) & 0xFU);
                    wf[i]       = static_cast<float>(u >= 8 ? u - 16 : u) * scale;
                }
                for (int column = 0; column < MaxCols; ++column) {
                    if (kExactCols || column < count) {
                        const uint4 xp = staged[column * chunks + chunk];
                        const bf16* xv = reinterpret_cast<const bf16*>(&xp);
                        for (int i = 0; i < 8; ++i) {
                            sums[column] = fmaf(wf[i], __bfloat162float(xv[i]), sums[column]);
                        }
                    }
                }
            }
        }
        for (int column = 0; column < MaxCols; ++column) {
            float v = sums[column];
            for (int offset = 16; offset > 0; offset >>= 1) { v += __shfl_xor_sync(0xFFFFFFFFU, v, offset); }
            if (lane == 0 && (kExactCols || column < count)) {
                out[static_cast<std::size_t>(first_column + column) * rows + row] = v;
            }
        }
    }
}

// Tall projections (vocabulary heads, ~250K rows) at verification widths. The narrow mapping stages
// eight columns per CTA, so 9..16 columns read every weight twice, and it gives each warp one row at a
// time, so each staged x chunk is widened again for every row: the BF16 head [248320, 2560] took
// 1.04 ms at 8 columns and 1.5 ms at 9-16 against a 0.71 ms DRAM floor. Here a CTA stages all of a
// call's columns (up to 16) once, each warp preloads all of its two rows' chunks before the first FMA,
// and each x chunk is widened once for both rows. Lane l still accumulates chunks l, l + 32, ... in
// order per (row, column) and the warp reduces by the same butterfly, so the bits equal the narrow
// kernels' at every width.
constexpr int kTallThreads        = 256;
constexpr int kTallWarps          = kTallThreads / 32;
constexpr int kTallRowsPerWarp    = 2;
constexpr int kTall2CtaColumns    = 9;
constexpr int kTallMaxColumns     = 32;
constexpr int kTallPassLaneChunks = 3;
constexpr int kTallPassChunks     = 32 * kTallPassLaneChunks;
constexpr int kTallMinRows        = 32768;
// From these widths the tall mapping is faster (RTX 5090, [248320, 2560], 2026-10-08): BF16 840 vs
// 1043 us at 8 columns and 869 vs 1500 at 9; q8_g32_fp16 841 vs 1209 at 9 but slower up to 8
// (799 vs 762), where the narrow mapping stays near the DRAM floor.
constexpr int kTallMinColumnsBf16 = 8;
constexpr int kTallMinColumnsQ8   = 1;
constexpr int kTallMinColumnsQ4   = 1;
constexpr std::size_t kTallMaxSharedBytes =
    static_cast<std::size_t>(kTallMaxColumns) * kTallPassChunks * sizeof(uint4);

struct TallBf16 {
    Segments segments;
    int k;
    using Chunk = uint4;
    __device__ Chunk load(int row, int chunk) const { return ld_nc_na<uint4>(row_chunk(segments, row, k, chunk)); }
    __device__ static void decode(const Chunk& c, float (&w)[8]) { widen8(c, w); }

    struct RowLoader {
        const TallBf16& parent;
        int r;
        __device__ __forceinline__ Chunk load(int chunk) const { return parent.load(r, chunk); }
    };
    __device__ __forceinline__ RowLoader row(int r) const { return {*this, r}; }
};

struct TallQ8 {
    const std::int8_t* codes;
    const __half* scales;
    int padded_k;
    struct Chunk {
        uint2 code;
        __half scale;
    };
    __device__ Chunk load(int row, int chunk) const {
        return {ld_nc_na_evict_first(
                    reinterpret_cast<const uint2*>(codes + static_cast<std::size_t>(row) * padded_k) + chunk),
                __ldcg(scales + static_cast<std::size_t>(row) * (padded_k / 32) + (chunk >> 2))};
    }
    __device__ static void decode(const Chunk& c, float (&w)[8]) {
        const auto* v     = reinterpret_cast<const std::int8_t*>(&c.code);
        const float scale = __half2float(c.scale);
#pragma unroll
        for (int i = 0; i < 8; ++i) { w[i] = static_cast<float>(v[i]) * scale; }
    }

    struct RowLoader {
        const uint2* row_codes;
        const __half* row_scales;
        __device__ __forceinline__ Chunk load(int chunk) const {
            return {ld_nc_na_evict_first(row_codes + chunk), __ldcg(row_scales + (chunk >> 2))};
        }
    };
    __device__ __forceinline__ RowLoader row(int r) const {
        return {reinterpret_cast<const uint2*>(codes + static_cast<std::size_t>(r) * padded_k),
                scales + static_cast<std::size_t>(r) * (padded_k / 32)};
    }
};

struct TallQ4 {
    const std::uint8_t* codes;
    const __half* scales;
    int padded_k;
    struct Chunk {
        std::uint32_t code;
        __half scale;
    };
    __device__ Chunk load(int row, int chunk) const {
        return {ld_nc_na_evict_first_u32(
                    reinterpret_cast<const std::uint32_t*>(codes + static_cast<std::size_t>(row) * (padded_k / 2)) +
                    chunk),
                __ldcg(scales + static_cast<std::size_t>(row) * (padded_k / 64) + (chunk >> 3))};
    }
    __device__ static void decode(const Chunk& c, float (&w)[8]) {
        const float scale = __half2float(c.scale);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int u = static_cast<int>((c.code >> (4 * i)) & 0xFU);
            w[i]        = static_cast<float>(u >= 8 ? u - 16 : u) * scale;
        }
    }

    struct RowLoader {
        const std::uint32_t* row_codes;
        const __half* row_scales;
        __device__ __forceinline__ Chunk load(int chunk) const {
            return {ld_nc_na_evict_first_u32(row_codes + chunk), __ldcg(row_scales + (chunk >> 3))};
        }
    };
    __device__ __forceinline__ RowLoader row(int r) const {
        return {reinterpret_cast<const std::uint32_t*>(codes + static_cast<std::size_t>(r) * (padded_k / 2)),
                scales + static_cast<std::size_t>(r) * (padded_k / 64)};
    }
};

template <class Codec, int MaxCols>
__global__ void __launch_bounds__(
    kTallThreads, (std::is_same_v<Codec, TallQ8> || std::is_same_v<Codec, TallQ4>) ? 2 : 1)
    projection_tall_exact_kernel(const bf16* __restrict__ x, int k, int rows, Codec codec,
                                 float* __restrict__ out) {
    extern __shared__ uint4 staged[]; // [MaxCols][chunks]
    const int chunks    = k / 8;
    const int lane      = static_cast<int>(threadIdx.x) & 31;
    const int warp      = static_cast<int>(threadIdx.x) >> 5;
    const int first_row = (static_cast<int>(blockIdx.x) * kTallWarps + warp) * kTallRowsPerWarp;
    // 1. Every chunk of both rows, before x is staged.
    typename Codec::Chunk w[kTallRowsPerWarp][kMaxLaneChunks];
#pragma unroll
    for (int r = 0; r < kTallRowsPerWarp; ++r) {
        const int row = min(first_row + r, rows - 1);
        const auto row_loader = codec.row(row);
#pragma unroll
        for (int i = 0; i < kMaxLaneChunks; ++i) {
            const int chunk = lane + 32 * i;
            if (chunk < chunks) { w[r][i] = row_loader.load(chunk); }
        }
    }
    for (int i = static_cast<int>(threadIdx.x); i < MaxCols * chunks; i += kTallThreads) {
        const auto dst = static_cast<unsigned>(__cvta_generic_to_shared(staged + i));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(reinterpret_cast<const uint4*>(x) + i)
                     : "memory");
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();
    // 2. Per chunk in lane order: widen both rows' weights, then each column's x once.
    float sums[kTallRowsPerWarp][MaxCols] = {};
#pragma unroll
    for (int i = 0; i < kMaxLaneChunks; ++i) {
        const int chunk = lane + 32 * i;
        if (chunk >= chunks) { break; }
        float wf[kTallRowsPerWarp][8];
#pragma unroll
        for (int r = 0; r < kTallRowsPerWarp; ++r) { Codec::decode(w[r][i], wf[r]); }
#pragma unroll
        for (int column = 0; column < MaxCols; ++column) {
            float xf[8];
            widen8(staged[column * chunks + chunk], xf);
#pragma unroll
            for (int r = 0; r < kTallRowsPerWarp; ++r) {
#pragma unroll
                for (int j = 0; j < 8; ++j) { sums[r][column] = fmaf(wf[r][j], xf[j], sums[r][column]); }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < kTallRowsPerWarp; ++r) {
        const int row = first_row + r;
#pragma unroll
        for (int column = 0; column < MaxCols; ++column) {
            float v = sums[r][column];
            for (int offset = 16; offset > 0; offset >>= 1) { v += __shfl_xor_sync(0xFFFFFFFFU, v, offset); }
            if (lane == 0 && row < rows) { out[static_cast<std::size_t>(column) * rows + row] = v; }
        }
    }
}

template <class Codec, int MaxCols = kTallMaxColumns, bool SingleStage = false>
__global__ void __launch_bounds__(
    kTallThreads, (std::is_same_v<Codec, TallQ8> || std::is_same_v<Codec, TallQ4>) ? 2 : 1)
    projection_tall_kernel(const bf16* __restrict__ x, int k, int columns, int rows, Codec codec,
                           float* __restrict__ out) {
    extern __shared__ uint4 staged[]; // [columns][SingleStage ? chunks : kTallPassChunks]
    const int chunks    = k / 8;
    const int lane      = static_cast<int>(threadIdx.x) % 32;
    const int first_row = (static_cast<int>(blockIdx.x) * kTallWarps + static_cast<int>(threadIdx.x) / 32) *
                          kTallRowsPerWarp;
    if constexpr (SingleStage) {
        for (int i = static_cast<int>(threadIdx.x); i < columns * chunks; i += kTallThreads) {
            const auto dst  = static_cast<unsigned>(__cvta_generic_to_shared(staged + i));
            const auto* src = reinterpret_cast<const uint4*>(x) + i;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src) : "memory");
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();
    }
    float sums[kTallRowsPerWarp][MaxCols] = {};
    for (int pass_begin = 0; pass_begin < chunks; pass_begin += kTallPassChunks) {
        const int pass_chunks = min(kTallPassChunks, chunks - pass_begin);
        typename Codec::Chunk w[kTallRowsPerWarp][kTallPassLaneChunks];
#pragma unroll
        for (int r = 0; r < kTallRowsPerWarp; ++r) {
            const int row = min(first_row + r, rows - 1);
#pragma unroll
            for (int i = 0; i < kTallPassLaneChunks; ++i) {
                const int chunk = pass_begin + lane + 32 * i;
                if (chunk < chunks) { w[r][i] = codec.load(row, chunk); }
            }
        }
        if constexpr (!SingleStage) {
            if (pass_begin > 0) { __syncthreads(); }
            for (int i = static_cast<int>(threadIdx.x); i < columns * kTallPassChunks; i += kTallThreads) {
                const int column = i / kTallPassChunks;
                const int c      = i % kTallPassChunks;
                if (c < pass_chunks) {
                    const auto dst = static_cast<unsigned>(__cvta_generic_to_shared(staged + i));
                    const auto* src =
                        reinterpret_cast<const uint4*>(x) + static_cast<std::size_t>(column) * chunks + pass_begin + c;
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src) : "memory");
                }
            }
            asm volatile("cp.async.commit_group;\n" ::: "memory");
            asm volatile("cp.async.wait_group 0;\n" ::: "memory");
            __syncthreads();
        }
#pragma unroll
        for (int i = 0; i < kTallPassLaneChunks; ++i) {
            const int c     = lane + 32 * i;
            const int chunk = pass_begin + c;
            if (chunk >= chunks) { break; }
            float wf[kTallRowsPerWarp][8];
#pragma unroll
            for (int r = 0; r < kTallRowsPerWarp; ++r) { Codec::decode(w[r][i], wf[r]); }
#pragma unroll
            for (int column = 0; column < MaxCols; ++column) {
                if (column < columns) {
                    float xf[8];
                    const int staged_idx = SingleStage ? (column * chunks + chunk) : (column * kTallPassChunks + c);
                    widen8(staged[staged_idx], xf);
#pragma unroll
                    for (int r = 0; r < kTallRowsPerWarp; ++r) {
#pragma unroll
                        for (int j = 0; j < 8; ++j) { sums[r][column] = fmaf(wf[r][j], xf[j], sums[r][column]); }
                    }
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < kTallRowsPerWarp; ++r) {
        const int row = first_row + r;
#pragma unroll
        for (int column = 0; column < MaxCols; ++column) {
            if (column < columns) {
                float v = sums[r][column];
                for (int offset = 16; offset > 0; offset >>= 1) { v += __shfl_xor_sync(0xFFFFFFFFU, v, offset); }
                if (lane == 0 && row < rows) { out[static_cast<std::size_t>(column) * rows + row] = v; }
            }
        }
    }
}

template <class Codec>
void launch_tall(const Tensor& x, int k, int columns, int rows, const Codec& codec, Tensor& out, cudaStream_t stream) {
    const int rows_per_cta         = kTallWarps * kTallRowsPerWarp;
    const int blocks               = (rows + rows_per_cta - 1) / rows_per_cta;
    const std::size_t staged_exact = static_cast<std::size_t>(columns) * k * sizeof(bf16);
    const std::size_t staged_pass  = static_cast<std::size_t>(columns) * kTallPassChunks * sizeof(uint4);
    const auto launch_exact = [&]<int MaxCols>() {
        projection_tall_exact_kernel<Codec, MaxCols><<<blocks, kTallThreads, staged_exact, stream>>>(
            static_cast<const bf16*>(x.data), k, rows, codec, static_cast<float*>(out.data));
    };
    const auto launch_pass = [&]<int MaxCols>() {
        if (staged_exact <= 49152U) {
            static const bool configured_single = [] {
                return cudaFuncSetAttribute(projection_tall_kernel<Codec, MaxCols, true>,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize, 49152) == cudaSuccess;
            }();
            require(configured_single, "the single-stage tall kernel's shared memory could not be configured");
            projection_tall_kernel<Codec, MaxCols, true><<<blocks, kTallThreads, staged_exact, stream>>>(
                static_cast<const bf16*>(x.data), k, columns, rows, codec, static_cast<float*>(out.data));
            return;
        }
        static const bool configured = [] {
            return cudaFuncSetAttribute(projection_tall_kernel<Codec, MaxCols, false>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(kTallMaxSharedBytes)) == cudaSuccess;
        }();
        require(configured, "the tall kernel's shared memory could not be configured");
        projection_tall_kernel<Codec, MaxCols, false><<<blocks, kTallThreads, staged_pass, stream>>>(
            static_cast<const bf16*>(x.data), k, columns, rows, codec, static_cast<float*>(out.data));
    };
    switch (columns) {
    case 1:
        launch_exact.template operator()<1>();
        return;
    case 2:
        launch_exact.template operator()<2>();
        return;
    case 3:
        launch_exact.template operator()<3>();
        return;
    case 4:
        launch_exact.template operator()<4>();
        return;
    case 5:
        launch_exact.template operator()<5>();
        return;
    default:
        if (columns <= 8) {
            launch_pass.template operator()<8>();
        } else if (columns <= 12) {
            launch_pass.template operator()<12>();
        } else if (columns <= 16) {
            launch_pass.template operator()<16>();
        } else if (columns <= 20) {
            launch_pass.template operator()<20>();
        } else if (columns <= 24) {
            launch_pass.template operator()<24>();
        } else {
            launch_pass.template operator()<kTallMaxColumns>();
        }
        return;
    }
}

// BF16 vocabulary heads at 1..16 columns: the tensor-core mapping, one route for every width, at the
// DRAM floor (701-708 us for [248320, 2560] at 1..16 columns, 2026-10-08).
bool mma_route(int rows, int columns, bool supported, detail::ProjectionRoute route) {
    if (route == detail::ProjectionRoute::Mma) {
        require(supported, "the tensor-core mapping does not serve this problem");
        return true;
    }
    return route == detail::ProjectionRoute::Automatic && supported && rows >= kTallMinRows &&
           columns <= detail::kProjectionMmaMaxColumns;
}

bool tall_route(int rows, int columns, int min_columns, detail::ProjectionRoute route) {
    if (route == detail::ProjectionRoute::Tall) { return true; }
    if (route == detail::ProjectionRoute::Narrow) { return false; }
    return (rows >= kTallMinRows || (columns > 16 && rows >= 512)) && columns >= min_columns &&
           columns <= kTallMaxColumns;
}

} // namespace

void projection_fp32(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    detail::projection_fp32_route(x, weight, out, detail::ProjectionRoute::Automatic, stream);
}

void projection_fp32(const Tensor& x, std::span<const Tensor* const> weights, Tensor& out,
                     cudaStream_t stream) {
    detail::projection_fp32_route(x, weights, out, detail::ProjectionRoute::Automatic, stream);
}

void projection_fp32_router_mma(const Tensor& x, std::span<const Tensor* const> weights, Tensor& out,
                                cudaStream_t stream) {
    require(x.data != nullptr && x.dtype == DType::BF16 && x.is_contiguous() && x.ne[2] == 1 && x.ne[3] == 1 &&
                aligned16(x.data),
            "x must be contiguous, 16-byte aligned BF16 [K, T]");
    require(out.data != nullptr && out.dtype == DType::FP32 && out.is_contiguous(), "out must be contiguous FP32");
    const int k = x.ne[0], columns = x.ne[1];
    require(k > 0 && k <= kMaxK && columns >= 1, "router-mma: K up to 3072 and at least 1 column");
    require(!weights.empty() && weights.size() <= kMaxSegments, "between one and four weights");
    Segments segments{};
    segments.count = static_cast<int>(weights.size());
    int rows       = 0;
    for (std::size_t i = 0; i < weights.size(); ++i) {
        const Tensor& w = *weights[i];
        require(w.data != nullptr && w.dtype == DType::BF16 && w.is_contiguous() && w.ne[0] == k && w.ne[1] > 0 &&
                    w.ne[2] == 1 && w.ne[3] == 1 && aligned16(w.data),
                "each weight must be contiguous, 16-byte aligned BF16 [K, N]");
        segments.data[i]  = static_cast<const bf16*>(w.data);
        segments.begin[i] = rows;
        rows += w.ne[1];
    }
    require(out.ne[0] == rows && out.ne[1] == columns && out.ne[2] == 1 && out.ne[3] == 1,
            "out must be FP32 [sum N, T]");
    const int first_rows = weights[0]->ne[1];
    require(detail::projection_fp32_bf16_wide_mma_supported(first_rows, k) && first_rows % kWideRows == 0,
            "router-mma: the first weight's rows and K must be multiples of 64");
    detail::projection_fp32_bf16_wide_mma(x, segments.data[0], first_rows, rows, static_cast<float*>(out.data), stream);
    if (rows > first_rows) {
        if (rows - first_rows == 1 && weights.size() == 2) {
            // Single trailing row (Qwen4Exp's shared_score row 512): stage the 1 weight row into 5 KiB SMEM
            // and compute 8 token columns per CTA without staging 31 dummy weight rows across 32 KiB SMEM.
            const std::size_t w_bytes = static_cast<std::size_t>(k) * sizeof(bf16);
            projection_tail_row_kernel<<<(columns + kTailRowWarps - 1) / kTailRowWarps, kTailRowThreads, w_bytes, stream>>>(
                static_cast<const bf16*>(x.data), k, columns, segments.data[1],
                static_cast<float*>(out.data) + first_rows, rows);
        } else {
            // The other weights' rows: the wide SIMT kernel's own bits, from the row block after the first weight.
            static const bool configured = [] {
                return cudaFuncSetAttribute(projection_wide_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            kWideSharedBytes) == cudaSuccess;
            }();
            require(configured, "the wide kernel's shared memory could not be configured");
            const int offset = first_rows / kWideRows;
            projection_wide_kernel<<<dim3((rows + kWideRows - 1) / kWideRows - offset, (columns + kWideColumns - 1) / kWideColumns),
                                     kWideThreads, kWideSharedBytes, stream>>>(
                static_cast<const bf16*>(x.data), k, columns, rows, segments, static_cast<float*>(out.data), offset);
        }
    }
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("projection_fp32_router_mma: ") + cudaGetErrorString(error));
    }
}

void detail::projection_fp32_route(const Tensor& x, const Weight& weight, Tensor& out, ProjectionRoute route,
                                   cudaStream_t stream) {
    require(x.data != nullptr && x.dtype == DType::BF16 && x.is_contiguous() && x.ne[2] == 1 && x.ne[3] == 1,
            "x must be contiguous BF16 [K, T]");
    require(out.data != nullptr && out.dtype == DType::FP32 && out.is_contiguous(), "out must be contiguous FP32");
    const int k = x.ne[0], columns = x.ne[1], rows = weight.n, padded_k = weight.padded_shape[1];
    require(k > 0 && k % 8 == 0 && k <= kMaxK && columns > 0, "K must be a multiple of 8 up to 3072");
    const bool q4 = weight.qtype == QType::Q4_G64_FP16;
    require((weight.qtype == QType::Q8_G32_FP16 || q4) && weight.layout == QuantLayout::RowSplit &&
                weight.qhigh == nullptr && weight.k == k && rows > 0 && padded_k >= k &&
                padded_k % (q4 ? 64 : 32) == 0 && weight.qdata != nullptr && weight.scales != nullptr &&
                aligned16(weight.qdata) && aligned16(x.data),
            "weight must be an aligned row-split q8_g32_fp16 or q4_g64_fp16 [N, K]");
    require(out.ne[0] == rows && out.ne[1] == columns && out.ne[2] == 1 && out.ne[3] == 1,
            "out must be FP32 [N, T]");
    const dim3 grid((rows + kRowsPerCta - 1) / kRowsPerCta, (columns + kColumns - 1) / kColumns);
    const std::size_t staged = static_cast<std::size_t>(std::min(kColumns, columns)) * k * sizeof(bf16);
    const int min_auto_tall = q4 ? kTallMinColumnsQ4 : kTallMinColumnsQ8;
    const int max_auto_tall = q4 ? kTall2CtaColumns : kTallMaxColumns;
    if (columns <= kTallMaxColumns && tall_route(rows, columns, min_auto_tall, route) &&
        (route == ProjectionRoute::Tall || columns <= max_auto_tall)) {
        if (q4) {
            launch_tall(x, k, columns, rows,
                        TallQ4{static_cast<const std::uint8_t*>(weight.qdata),
                               static_cast<const __half*>(weight.scales), padded_k},
                        out, stream);
        } else {
            launch_tall(x, k, columns, rows,
                        TallQ8{static_cast<const std::int8_t*>(weight.qdata),
                               static_cast<const __half*>(weight.scales), padded_k},
                        out, stream);
        }
    } else if (q4) {
        const auto kernel = columns == 1   ? projection_q4_kernel<1>
                            : columns == 2 ? projection_q4_kernel<2>
                            : columns == 3 ? projection_q4_kernel<3>
                            : columns == 4 ? projection_q4_kernel<4>
                            : columns == 5 ? projection_q4_kernel<5>
                                           : projection_q4_kernel<kColumns>;
        CUDA_CHECK(pdl::launch_consumer({grid, dim3(kThreads), staged, stream}, kernel,
                                        static_cast<const bf16*>(x.data), k, padded_k, columns, rows,
                                        static_cast<const std::uint8_t*>(weight.qdata),
                                        static_cast<const __half*>(weight.scales), static_cast<float*>(out.data)));
    } else {
        const auto kernel = columns == 1 ? projection_q8_kernel<1> : projection_q8_kernel<kColumns>;
        CUDA_CHECK(pdl::launch_consumer({grid, dim3(kThreads), staged, stream}, kernel,
                                        static_cast<const bf16*>(x.data), k, padded_k, columns, rows,
                                        static_cast<const std::int8_t*>(weight.qdata),
                                        static_cast<const __half*>(weight.scales), static_cast<float*>(out.data)));
    }
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("projection_fp32 quantized: ") + cudaGetErrorString(error));
    }
}

void detail::projection_fp32_route(const Tensor& x, std::span<const Tensor* const> weights, Tensor& out,
                                   ProjectionRoute route, cudaStream_t stream) {
    require(x.data != nullptr && x.dtype == DType::BF16 && x.is_contiguous() && x.ne[2] == 1 && x.ne[3] == 1,
            "x must be contiguous BF16 [K, T]");
    require(out.data != nullptr && out.dtype == DType::FP32 && out.is_contiguous(), "out must be contiguous FP32");
    const int k = x.ne[0], columns = x.ne[1];
    require(k > 0 && k % 8 == 0 && k <= kMaxK && columns > 0, "K must be a multiple of 8 up to 3072");
    require(!weights.empty() && weights.size() <= kMaxSegments, "between one and four weights");
    require(aligned16(x.data), "x must be 16-byte aligned");
    Segments segments{};
    segments.count = static_cast<int>(weights.size());
    int rows       = 0;
    for (std::size_t i = 0; i < weights.size(); ++i) {
        const Tensor& w = *weights[i];
        require(w.data != nullptr && w.dtype == DType::BF16 && w.is_contiguous() && w.ne[0] == k && w.ne[1] > 0 &&
                    w.ne[2] == 1 && w.ne[3] == 1 && aligned16(w.data),
                "each weight must be contiguous, 16-byte aligned BF16 [K, N]");
        segments.data[i]  = static_cast<const bf16*>(w.data);
        segments.begin[i] = rows;
        rows += w.ne[1];
    }
    require(out.ne[0] == rows && out.ne[1] == columns && out.ne[2] == 1 && out.ne[3] == 1,
            "out must be FP32 [sum N, T]");
    const std::size_t staged = static_cast<std::size_t>(std::min(kColumns, columns)) * k * sizeof(bf16);
    const dim3 column_groups(1, (columns + kColumns - 1) / kColumns);
    if (segments.count == 1 &&
        mma_route(rows, columns, detail::projection_fp32_bf16_mma_supported(rows, k, columns), route)) {
        detail::projection_fp32_bf16_mma(x, segments.data[0], rows, out, stream);
    } else if (columns <= kTallMaxColumns && tall_route(rows, columns, kTallMinColumnsBf16, route)) {
        launch_tall(x, k, columns, rows, TallBf16{segments, k}, out, stream);
    } else if (columns >= kWideMinColumns) {
        static const bool configured = [] {
            return cudaFuncSetAttribute(projection_wide_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        kWideSharedBytes) == cudaSuccess;
        }();
        require(configured, "the wide kernel's shared memory could not be configured");
        projection_wide_kernel<<<dim3((rows + kWideRows - 1) / kWideRows, (columns + kWideColumns - 1) / kWideColumns),
                                 kWideThreads, kWideSharedBytes, stream>>>(
            static_cast<const bf16*>(x.data), k, columns, rows, segments, static_cast<float*>(out.data));
    } else {
        constexpr int threads = 128, rows_per_warp = 1, rows_per_cta = threads / 32 * rows_per_warp;
        const auto kernel = columns == 1   ? projection_kernel<threads, rows_per_warp, 1>
                            : columns == 2 ? projection_kernel<threads, rows_per_warp, 2>
                            : columns == 3 ? projection_kernel<threads, rows_per_warp, 3>
                            : columns == 4 ? projection_kernel<threads, rows_per_warp, 4>
                            : columns == 5 ? projection_kernel<threads, rows_per_warp, 5>
                                           : projection_kernel<threads, rows_per_warp, kColumns>;
        CUDA_CHECK(pdl::launch_consumer({dim3((rows + rows_per_cta - 1) / rows_per_cta, column_groups.y), dim3(threads),
                                         staged, stream},
                                        kernel, static_cast<const bf16*>(x.data), k,
                                        columns, rows, segments, static_cast<float*>(out.data)));
    }
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("projection_fp32: ") + cudaGetErrorString(error));
    }
}

} // namespace infernix::ops
