#pragma once

#include "infernix/ops/ggml_linear.h"

#include "core/device.h"
#include "core/ggml_format.h"
#include "core/pdl.cuh"
#include "ops/ggml/ggml_dp4a.cuh"
#include "ops/ggml/ggml_exact.h"
#include "ops/ggml/ggml_mma_rows.cuh"
#include "ops/ggml/ggml_tile.cuh"

#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace infernix::ops {

namespace fast = ggml::fast;

inline constexpr int kGgmlThreads       = 128;
inline constexpr int kGgmlWarps         = kGgmlThreads / 32;
inline constexpr int kGgmlColumns       = 4;
inline constexpr int kGgmlMmaRows       = 16;
inline constexpr int kGgmlMmaWarps      = 4;
inline constexpr int kGgmlMmaMinColumns = 3;
inline constexpr int kGgmlMmaMaxColumns = 8;

template <QType F>
inline constexpr bool kGgmlTiledFormat = ggml_tiles16_supported(F);
template <QType F>
inline constexpr int kGgmlTiledRun = 16 * static_cast<int>(ggml_block_bytes(F));

struct RowsGroup {
    static constexpr int kMax = 4;
    const std::uint8_t* data[kMax] = {};
    void* out[kMax]                = {};
    int rows[kMax]                 = {};
    int n[kMax]                    = {};
    int out_offset[kMax]           = {};
    int row_bytes[kMax]            = {};
    int format[kMax]               = {};
    int first[kMax + 1]            = {};
    int count                      = 0;
};

struct GroupSlot {
    const std::uint8_t* weights;
    void* out;
    int rows, n, out_offset, block, row_bytes, format;
};

struct Columns {
    const std::int32_t* encoded = nullptr;
    const std::int32_t* emax    = nullptr;
};

__device__ __forceinline__ GroupSlot group_slot(const RowsGroup& g) {
    const int b = static_cast<int>(blockIdx.x);
    int i       = 0;
#pragma unroll
    for (int j = 1; j < RowsGroup::kMax; ++j) { i = j < g.count && b >= g.first[j] ? j : i; }
    GroupSlot s{g.data[0], g.out[0], g.rows[0], g.n[0], g.out_offset[0], b - g.first[0], g.row_bytes[0], g.format[0]};
#pragma unroll
    for (int j = 1; j < RowsGroup::kMax; ++j) {
        if (i == j) {
            s = GroupSlot{g.data[j], g.out[j], g.rows[j], g.n[j], g.out_offset[j], b - g.first[j], g.row_bytes[j], g.format[j]};
        }
    }
    return s;
}

inline void ggml_require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("ggml_linear: ") + message); }
}

inline void ggml_check_launch(const char* what) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("ggml_linear ") + what + ": " + cudaGetErrorString(error));
    }
}

inline int ggml_lanes_for(int slices) {
    int lanes = 4;
    while (lanes < 32 && slices / (2 * lanes) >= 4) { lanes *= 2; }
    return lanes;
}

__device__ __forceinline__ void shfl_add(ggml::I128& r, int offset) {
    const auto lo = __shfl_xor_sync(0xFFFFFFFFU, static_cast<unsigned long long>(r.lo), offset);
    const auto hi = __shfl_xor_sync(0xFFFFFFFFU, static_cast<long long>(r.hi), offset);
    ggml::add(r, ggml::I128{static_cast<std::uint64_t>(lo), static_cast<std::int64_t>(hi)});
}

template <QType F, bool OutF32, int Lanes, int C, bool Tiled = false>
__global__ void __launch_bounds__(kGgmlThreads)
    rows_kernel(RowsGroup g, int row_bytes, int slices, const std::int32_t* encoded,
                const std::int32_t* emax, int columns) {
    constexpr int kRowsPerWarp        = 32 / Lanes;
    constexpr std::size_t kBlockBytes = ggml_block_bytes(F);
    const GroupSlot seg               = group_slot(g);
    const std::uint8_t* weights       = seg.weights;
    const int rows = seg.rows, n = seg.n, out_offset = seg.out_offset;
    void* const out   = seg.out;
    const int warp    = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int part    = lane % Lanes;
    const int row     = (seg.block * kGgmlWarps + warp) * kRowsPerWarp + lane / Lanes;
    const bool active = row < rows;
    const int r       = active ? row : 0;
    const std::uint8_t* w = Tiled ? weights + static_cast<std::size_t>(r / 16) * 16 * row_bytes + (r % 16) * kBlockBytes
                                  : weights + static_cast<std::size_t>(r) * row_bytes;
    const auto row_of = [&](int s) {
        if constexpr (Tiled) {
            static_assert(ggml_block_elems(F) == 256, "tiled rows: 256-element blocks");
            return w + static_cast<std::size_t>(s / 8) * 15 * kBlockBytes;
        } else {
            return w;
        }
    };
    const std::size_t column_words = static_cast<std::size_t>(slices) * (fast::kSliceWords + 1);
    fast::Slice first{};
    if (active) { first = fast::slice<F>(row_of(part), part); }
    pdl::wait_for_dependencies();
    for (int base = 0; base < columns; base += C) {
        const int count = min(C, columns - base);
        ggml::I128 acc[C];
        if (active) {
            for (int s = part; s < slices; s += Lanes) {
                const fast::Slice ws      = base == 0 && s == part ? first : fast::slice<F>(row_of(s), s);
                const fast::SliceScale sc = fast::slice_scale(ws);
#pragma unroll
                for (int c = 0; c < C; ++c) {
                    if (c < count) {
                        const std::int32_t* column = encoded + static_cast<std::size_t>(base + c) * column_words;
                        fast::add_slice(acc[c], ws, sc, column + s * fast::kSliceWords,
                                        fast::kMin<F> ? column[slices * fast::kSliceWords + s] : 0);
                    }
                }
            }
        }
#pragma unroll
        for (int c = 0; c < C; ++c) {
            if (c < count) {
#pragma unroll
                for (int offset = Lanes / 2; offset >= 1; offset >>= 1) { shfl_add(acc[c], offset); }
                if (active && part == 0) {
                    const int t          = base + c;
                    const std::size_t at = static_cast<std::size_t>(t) * n + out_offset + row;
                    if constexpr (OutF32) {
                        static_cast<float*>(out)[at] = ggml::output_f32(acc[c], emax[t], F);
                    } else {
                        static_cast<std::uint16_t*>(out)[at] = ggml::output_bf16(acc[c], emax[t], F);
                    }
                }
            }
        }
    }
}

#define INFERNIX_GGML_FORMATS(X)                                                                                       \
    X(QType::GGML_Q2_0)                                                                                                \
    X(QType::GGML_Q4_K)                                                                                                \
    X(QType::GGML_Q5_K)                                                                                                \
    X(QType::GGML_Q6_K)                                                                                                \
    X(QType::GGML_IQ2_S)                                                                                               \
    X(QType::GGML_IQ3_XXS)                                                                                             \
    X(QType::GGML_IQ3_S)                                                                                               \
    X(QType::GGML_IQ4_XS)                                                                                              \
    X(QType::GGML_IQ4_NL)

void ggml_launch_tiled(QType format, bool out_f32, const RowsGroup& g, int k, const Columns& in, int columns,
                       cudaStream_t stream);
bool ggml_launch_tiled_mixed(QType f0, QType f1, const RowsGroup& g, int k, const Columns& in, int columns,
                             cudaStream_t stream);
void ggml_decode_plane(const GgmlSegment& s, int k, std::uint8_t* plane, cudaStream_t stream);
void ggml_launch_tiles(const GgmlSegment& s, int k, bool out_f32, const Columns& in, int columns, void* out, int n,
                       const std::uint8_t* plane, cudaStream_t stream);

} // namespace infernix::ops
