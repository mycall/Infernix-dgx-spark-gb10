// Register-streamed Q8 LinearSwiGLU for small T (Qwen3.8-Flash-Next's shared expert, gate/up
// [1280, 2560] -> [640]; design §19.3.3 Phase 1b). A warp owns output row i: it loads every code and
// scale of gate row i and up row M + i into registers before reading x, then computes both dot
// products with the K1 lane-chunk order (lane l owns codes k = 8l + 256m, fmaf in k order, a warp
// butterfly) and writes bf16(SiLU(gate) * up) with gate and up in FP32. A column's result does not
// depend on how many columns share the call.

#include "core/device.h"
#include "core/pdl.cuh"
#include "core/weight.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear_swiglu/q8/q8_linear_swiglu_kernels.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <stdexcept>
#include <string>

namespace infernix::ops::detail {
namespace {

using bf16 = __nv_bfloat16;

constexpr int kK      = 2560;
constexpr int kChunks = kK / 256;
constexpr int kWarps  = 4; // output rows per CTA

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) { value += __shfl_xor_sync(0xFFFFFFFFU, value, offset); }
    return value;
}

__device__ __forceinline__ void decode8(uint2 code, float scale, float (&w)[8]) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        w[i] = static_cast<float>(static_cast<std::int8_t>((i < 4 ? code.x : code.y) >> ((i & 3) * 8))) * scale;
    }
}

template <int kPassCols, bool kExactSinglePass>
__global__ __launch_bounds__(kWarps * 32, 4) void q8_linear_swiglu_stream_pair_kernel(
    const bf16* x, const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ scales,
    int m_rows, int padded_k, int columns, bf16* __restrict__ out) {
    const int lane = static_cast<int>(threadIdx.x) & 31, warp = static_cast<int>(threadIdx.x) >> 5;
    const int i    = static_cast<int>(blockIdx.x) * kWarps + warp;
    if (i >= m_rows) { return; }

    // Weights first: gate row i and up row M + i.
    uint2 code[2][kChunks];
    std::uint32_t scale_bits[2][kChunks / 2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const std::int64_t row        = static_cast<std::int64_t>(h) * m_rows + i;
        const std::uint8_t* code_row  = codes + row * padded_k;
        const std::uint8_t* scale_row = scales + row * (padded_k / 32) * 2;
#pragma unroll
        for (int m = 0; m < kChunks; ++m) {
            const int kk             = m * 256 + lane * 8;
            code[h][m]               = ld_nc_na<uint2>(code_row + kk);
            const std::uint32_t bits = ld_nc_na<std::uint16_t>(scale_row + (kk / 32) * 2);
            if (m % 2 == 0) {
                scale_bits[h][m / 2] = bits;
            } else {
                scale_bits[h][m / 2] |= bits << 16;
            }
        }
    }
    pdl::wait_for_dependencies();

    float gate[kPassCols], up[kPassCols];
#pragma unroll
    for (int c = 0; c < kPassCols; ++c) {
        gate[c] = 0.0F;
        up[c]   = 0.0F;
    }
#pragma unroll
    for (int m = 0; m < kChunks; ++m) {
        const int kk          = m * 256 + lane * 8;
        const unsigned g_bits = (scale_bits[0][m / 2] >> ((m % 2) * 16)) & 0xFFFFU;
        const unsigned u_bits = (scale_bits[1][m / 2] >> ((m % 2) * 16)) & 0xFFFFU;
        float wg[8], wu[8];
        decode8(code[0][m], __half2float(__ushort_as_half(static_cast<unsigned short>(g_bits))), wg);
        decode8(code[1][m], __half2float(__ushort_as_half(static_cast<unsigned short>(u_bits))), wu);
#pragma unroll
        for (int c = 0; c < kPassCols; ++c) {
            const uint4 values = load_ldg<uint4>(x + static_cast<std::int64_t>(c) * kK + kk);
            const float2 x0 = bf16x2_bits_to_float2(values.x), x1 = bf16x2_bits_to_float2(values.y);
            const float2 x2 = bf16x2_bits_to_float2(values.z), x3 = bf16x2_bits_to_float2(values.w);
            const float xv[8] = {x0.x, x0.y, x1.x, x1.y, x2.x, x2.y, x3.x, x3.y};
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                gate[c] = fmaf(wg[j], xv[j], gate[c]);
                up[c]   = fmaf(wu[j], xv[j], up[c]);
            }
        }
    }
#pragma unroll
    for (int c = 0; c < kPassCols; ++c) {
        const float g = warp_sum(gate[c]);
        const float u = warp_sum(up[c]);
        if (lane == 0) {
            out[static_cast<std::int64_t>(c) * m_rows + i] =
                __float2bfloat16_rn(g / (1.0F + expf(-g)) * u);
        }
    }
}

} // namespace

void q8_linear_swiglu_stream_pair_launch(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const int m_rows = out.ne[0], columns = x.ne[1];
    if (x.ne[0] != kK || w.k != kK || w.n != 2 * m_rows || columns <= 0 || columns > 16) {
        throw std::invalid_argument("Q8 LinearSwiGLU stream pair: unsupported problem");
    }
    const dim3 grid((m_rows + kWarps - 1) / kWarps), block(kWarps * 32);
    const auto launch = [&]<int kPassCols>() {
        CUDA_CHECK(pdl::launch_consumer(
            {grid, block, 0, stream}, q8_linear_swiglu_stream_pair_kernel<kPassCols, true>,
            static_cast<const bf16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), m_rows, w.padded_shape[1], columns,
            static_cast<bf16*>(out.data)));
    };
    switch (columns) {
    case 1: launch.template operator()<1>(); break;
    case 2: launch.template operator()<2>(); break;
    case 3: launch.template operator()<3>(); break;
    case 4: launch.template operator()<4>(); break;
    case 5: launch.template operator()<5>(); break;
    case 6: launch.template operator()<6>(); break;
    case 7: launch.template operator()<7>(); break;
    case 8: launch.template operator()<8>(); break;
    case 9: launch.template operator()<9>(); break;
    case 10: launch.template operator()<10>(); break;
    case 11: launch.template operator()<11>(); break;
    case 12: launch.template operator()<12>(); break;
    case 13: launch.template operator()<13>(); break;
    case 14: launch.template operator()<14>(); break;
    case 15: launch.template operator()<15>(); break;
    default: launch.template operator()<16>(); break;
    }
}

} // namespace infernix::ops::detail
