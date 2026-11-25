#include "ops/ggml/ggml_linear_internal.cuh"

namespace infernix::ops {
namespace {

template <QType F, bool Tiled>
__global__ void __launch_bounds__(ggml::tile::kThreads)
    plane_kernel(const std::uint8_t* __restrict__ weights, int rows, int row_bytes, std::uint8_t* __restrict__ plane) {
    __shared__ ggml::tile::CodeStage cs;
    const int tile = blockIdx.x, st = blockIdx.y, stages = gridDim.y;
    const auto row_of = [&](int r) {
        const int row = min(tile * ggml::tile::kTileRows + r, rows - 1);
        if constexpr (Tiled) {
            static_assert(ggml::tile::kBlockSlices<F> % ggml::tile::kStageSlices == 0, "a stage inside one block");
            const int b = st * ggml::tile::kStageSlices / ggml::tile::kBlockSlices<F>;
            return weights + static_cast<std::size_t>(row / 16) * 16 * row_bytes +
                   static_cast<std::size_t>(row % 16 + 15 * b) * ggml_block_bytes(F);
        } else {
            return weights + static_cast<std::size_t>(row) * row_bytes;
        }
    };
    ggml::tile::decode_stage<F>(cs, row_of, st);
    __syncthreads();
    auto* dst       = reinterpret_cast<int4*>(plane + (static_cast<std::size_t>(tile) * stages + st) * sizeof(cs));
    const auto* src = reinterpret_cast<const int4*>(&cs);
    for (int i = threadIdx.x; i < ggml::tile::kImageBytes<F> / 16; i += blockDim.x) { dst[i] = src[i]; }
}

template <QType F>
inline constexpr int kPlaneTileThreads = F == QType::GGML_IQ4_NL ? 256 : 512;
template <QType F>
inline constexpr int kPlaneTileBlocks = kPlaneTileThreads<F> == 512 ? 1 : 2;

template <QType F, bool OutF32, bool Tiled>
__global__ void __launch_bounds__(kPlaneTileThreads<F>, kPlaneTileBlocks<F>)
    tile_kernel(const std::uint8_t* __restrict__ weights, int rows, int row_bytes, int slices,
                const std::int32_t* encoded, const std::int32_t* emax, int columns,
                void* __restrict__ out, int n, int out_offset, const std::uint8_t* __restrict__ plane) {
    extern __shared__ __align__(16) unsigned char tile_shared[];
    auto& sm       = *reinterpret_cast<ggml::tile::PlaneShared*>(tile_shared);
    const int row0 = blockIdx.y * ggml::tile::kTileRows, col0 = blockIdx.x * ggml::tile::kTileColumns;
    const std::size_t column_words = static_cast<std::size_t>(slices) * (fast::kSliceWords + 1);
    const auto limbs = [&](int c) { return encoded + static_cast<std::size_t>(min(col0 + c, columns - 1)) * column_words; };
    ggml::tile::Accumulators<kPlaneTileThreads<F>> acc;
    const auto row_of = [&](int r) {
        const int row = min(row0 + r, rows - 1);
        if constexpr (Tiled) {
            return weights + static_cast<std::size_t>(row / 16) * 16 * row_bytes +
                   static_cast<std::size_t>(row % 16) * ggml_block_bytes(F);
        } else {
            return weights + static_cast<std::size_t>(row) * row_bytes;
        }
    };
    const int tile_columns   = min(ggml::tile::kTileColumns, columns - col0);
    const std::size_t stages = static_cast<std::size_t>(slices / ggml::tile::kStageSlices);
    ggml::tile::compute_plane<F, kPlaneTileThreads<F>, Tiled ? 16 : 1>(
        sm, row_of, plane + blockIdx.y * stages * sizeof(ggml::tile::CodeStage), slices, limbs,
        slices * fast::kSliceWords, tile_columns, acc);
    const ggml::tile::WarpTile wt = ggml::tile::warp_tile<kPlaneTileThreads<F>>(tile_columns);
    const int lane                = threadIdx.x % 32;
    constexpr int kMT             = ggml::tile::kMTilesOf<kPlaneTileThreads<F>>;
#pragma unroll
    for (int mi = 0; mi < kMT; ++mi) {
#pragma unroll
        for (int nt = 0; nt < ggml::tile::kNTiles; ++nt) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int row = row0 + ggml::tile::out_row(wt, lane, mi, q);
                const int t   = col0 + ggml::tile::out_column(wt, lane, nt, q);
                if (mi < wt.mtiles && nt < wt.ntiles && row < rows && t < columns) {
                    const std::size_t at = static_cast<std::size_t>(t) * n + out_offset + row;
                    if constexpr (OutF32) {
                        static_cast<float*>(out)[at] = ggml::output_f32(acc.r[mi][nt][q], emax[t], F);
                    } else {
                        static_cast<std::uint16_t*>(out)[at] = ggml::output_bf16(acc.r[mi][nt][q], emax[t], F);
                    }
                }
            }
        }
    }
}

template <QType F>
inline constexpr int kDecodeTileThreads = 512;

template <QType F, bool OutF32, bool Tiled>
__global__ void __launch_bounds__(kDecodeTileThreads<F>, 1)
    decode_tile_kernel(const std::uint8_t* __restrict__ weights, int rows, int row_bytes, int slices,
                       const std::int32_t* encoded, const std::int32_t* emax, int columns,
                       void* __restrict__ out, int n, int out_offset) {
    extern __shared__ __align__(16) unsigned char decode_tile_shared[];
    using Sm       = ggml::tile::SharedT<ggml::tile::kRawChunksOf<F>>;
    auto& sm       = *reinterpret_cast<Sm*>(decode_tile_shared);
    const int row0 = blockIdx.y * ggml::tile::kTileRows, col0 = blockIdx.x * ggml::tile::kTileColumns;
    const std::size_t column_words = static_cast<std::size_t>(slices) * (fast::kSliceWords + 1);
    const auto limbs = [&](int c) { return encoded + static_cast<std::size_t>(min(col0 + c, columns - 1)) * column_words; };
    ggml::tile::Accumulators<kDecodeTileThreads<F>> acc;
    const auto row_of = [&](int r) {
        const int row = min(row0 + r, rows - 1);
        if constexpr (Tiled) {
            return weights + static_cast<std::size_t>(row / 16) * 16 * row_bytes +
                   static_cast<std::size_t>(row % 16) * ggml_block_bytes(F);
        } else {
            return weights + static_cast<std::size_t>(row) * row_bytes;
        }
    };
    const int tile_columns = min(ggml::tile::kTileColumns, columns - col0);
    pdl::wait_for_dependencies();
    ggml::tile::compute<F, kDecodeTileThreads<F>, Tiled ? 16 : 1>(sm, row_of, slices, limbs, slices * fast::kSliceWords,
                                                                  tile_columns, acc);
    const ggml::tile::WarpTile wt = ggml::tile::warp_tile<kDecodeTileThreads<F>>(tile_columns);
    const int lane                = threadIdx.x % 32;
    constexpr int kMT             = ggml::tile::kMTilesOf<kDecodeTileThreads<F>>;
#pragma unroll
    for (int mi = 0; mi < kMT; ++mi) {
#pragma unroll
        for (int nt = 0; nt < ggml::tile::kNTiles; ++nt) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int row = row0 + ggml::tile::out_row(wt, lane, mi, q);
                const int t   = col0 + ggml::tile::out_column(wt, lane, nt, q);
                if (mi < wt.mtiles && nt < wt.ntiles && row < rows && t < columns) {
                    const std::size_t at = static_cast<std::size_t>(t) * n + out_offset + row;
                    if constexpr (OutF32) {
                        static_cast<float*>(out)[at] = ggml::output_f32(acc.r[mi][nt][q], emax[t], F);
                    } else {
                        static_cast<std::uint16_t*>(out)[at] = ggml::output_bf16(acc.r[mi][nt][q], emax[t], F);
                    }
                }
            }
        }
    }
}

template <QType F, bool OutF32, bool Tiled>
void launch_tile_format(const GgmlSegment& s, int k, const Columns& in, int columns, void* out, int n,
                        const std::uint8_t* plane, cudaStream_t stream) {
    const int row_bytes = static_cast<int>(k / ggml_block_elems(F) * ggml_block_bytes(F));
    const int slices    = k / 32;
    const dim3 grid((columns + ggml::tile::kTileColumns - 1) / ggml::tile::kTileColumns,
                    (s.rows + ggml::tile::kTileRows - 1) / ggml::tile::kTileRows);
    if (plane == nullptr) {
        using Sm                            = ggml::tile::SharedT<ggml::tile::kRawChunksOf<F>>;
        constexpr int kDecodeShared         = static_cast<int>(sizeof(Sm));
        static const bool decode_configured = [] {
            return cudaFuncSetAttribute(decode_tile_kernel<F, OutF32, Tiled>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        kDecodeShared) == cudaSuccess;
        }();
        ggml_require(decode_configured, "the decode tile kernel's shared memory exceeds the device's");
        CUDA_CHECK(pdl::launch_consumer({grid, dim3(kDecodeTileThreads<F>), static_cast<std::size_t>(kDecodeShared), stream},
                                        decode_tile_kernel<F, OutF32, Tiled>, s.data, s.rows, row_bytes, slices,
                                        in.encoded, in.emax, columns, out, n, s.out_offset));
        ggml_check_launch("decode tile");
        return;
    }
    constexpr int kShared        = static_cast<int>(sizeof(ggml::tile::PlaneShared));
    static const bool configured = [] {
        return cudaFuncSetAttribute(tile_kernel<F, OutF32, Tiled>, cudaFuncAttributeMaxDynamicSharedMemorySize, kShared) ==
               cudaSuccess;
    }();
    ggml_require(configured, "the tile kernel's shared memory exceeds the device's");
    tile_kernel<F, OutF32, Tiled><<<grid, kPlaneTileThreads<F>, kShared, stream>>>(
        s.data, s.rows, row_bytes, slices, in.encoded, in.emax, columns, out, n, s.out_offset, plane);
    ggml_check_launch("tile");
}

template <bool OutF32>
void launch_tiles_impl(const GgmlSegment& s, int k, const Columns& in, int columns, void* out, int n,
                       const std::uint8_t* plane, cudaStream_t stream) {
#define INFERNIX_GGML_TILES(Q)                                                                                         \
    case Q:                                                                                                            \
        if (s.layout == GgmlLayout::Tiles16) {                                                                         \
            if constexpr (kGgmlTiledFormat<Q>) {                                                                       \
                launch_tile_format<Q, OutF32, true>(s, k, in, columns, out, n, plane, stream);                         \
                break;                                                                                                 \
            }                                                                                                          \
            throw std::logic_error("ggml_linear: a tiled segment's format has no tiled kernels");                      \
        }                                                                                                              \
        launch_tile_format<Q, OutF32, false>(s, k, in, columns, out, n, plane, stream);                                \
        break;
    switch (s.format) {
        INFERNIX_GGML_FORMATS(INFERNIX_GGML_TILES)
    default:
        throw std::invalid_argument("ggml_linear: segment format is not a ggml block format");
    }
#undef INFERNIX_GGML_TILES
}

} // namespace

void ggml_launch_tiles(const GgmlSegment& s, int k, bool out_f32, const Columns& in, int columns, void* out, int n,
                       const std::uint8_t* plane, cudaStream_t stream) {
    if (out_f32) {
        launch_tiles_impl<true>(s, k, in, columns, out, n, plane, stream);
    } else {
        launch_tiles_impl<false>(s, k, in, columns, out, n, plane, stream);
    }
}

void ggml_decode_plane(const GgmlSegment& s, int k, std::uint8_t* plane, cudaStream_t stream) {
    const int slices = k / 32;
    const dim3 grid((s.rows + ggml::tile::kTileRows - 1) / ggml::tile::kTileRows, slices / ggml::tile::kStageSlices);
#define INFERNIX_GGML_PLANE(Q)                                                                                         \
    case Q: {                                                                                                          \
        const int row_bytes = static_cast<int>(k / ggml_block_elems(Q) * ggml_block_bytes(Q));                         \
        if (s.layout == GgmlLayout::Tiles16) {                                                                         \
            if constexpr (kGgmlTiledFormat<Q>) {                                                                       \
                plane_kernel<Q, true><<<grid, ggml::tile::kThreads, 0, stream>>>(s.data, s.rows, row_bytes, plane);   \
                break;                                                                                                 \
            }                                                                                                          \
            throw std::logic_error("ggml_linear: a tiled segment's format has no tiled kernels");                      \
        }                                                                                                              \
        plane_kernel<Q, false><<<grid, ggml::tile::kThreads, 0, stream>>>(s.data, s.rows, row_bytes, plane);          \
        break;                                                                                                         \
    }
    switch (s.format) {
        INFERNIX_GGML_FORMATS(INFERNIX_GGML_PLANE)
    default:
        throw std::invalid_argument("ggml_linear: segment format is not a ggml block format");
    }
#undef INFERNIX_GGML_PLANE
    ggml_check_launch("plane");
}

} // namespace infernix::ops
