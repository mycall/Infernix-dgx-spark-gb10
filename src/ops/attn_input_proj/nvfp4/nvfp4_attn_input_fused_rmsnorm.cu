#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"

#include "core/device.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_a4_tma_launch.h"
#include "ops/kernel/rmsnorm.cuh"
#include "ops/linear/nvfp4/nvfp4_a4_quantize.cuh"
#include "ops/linear/nvfp4/nvfp4_geometry.h"
#include "ops/linear/nvfp4/nvfp4_operands.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {

using Geometry              = Nvfp4N14336K5120;
constexpr int kBlock        = 320;
constexpr int kVecsPerRow   = Geometry::kInputRows / 8; // 640 uint4s per row
constexpr int kWarpsPerBlock = kBlock / kWarpSize;      // 10 warps
static_assert(kVecsPerRow == 2 * kBlock);
static_assert(Geometry::kGroupsPerRow == kBlock);

// The sum and Offset epilogue deliberately match the D=5120 prefill RMSNorm launcher's
// (rmsnorm_d5120_prefill_2row_vec16_kernel) uint4 ownership, accumulation order, 10-warp reduction,
// and BF16 store rounding. The shared row is the exact BF16 image which the standalone quantizer
// would read from the intermediate global allocation. The scales go to the tiled plane the TMA
// route reads; a token past the real count is a tile's padding and gets zero scales, as the
// standalone quantizer writes them.
template <Nvfp4ScaleLayout Layout>
__launch_bounds__(kBlock, 2) __global__
    void rmsnorm_nvfp4_quantize_kernel(const uint4* __restrict__ residual,
                                       const uint4* __restrict__ norm_weight,
                                       std::uint8_t* __restrict__ codes,
                                       std::uint8_t* __restrict__ scales, std::int32_t tokens,
                                       float eps, float input_scale_divisor) {
    static_assert(Layout != Nvfp4ScaleLayout::RowMajor);
    // Consumer protocol: the fused launch keeps this grid behind its residual producer in stream
    // order, so the wait is satisfied at launch there; the padding early-return below still makes
    // the protocol complete for every CTA of a programmatic launch.
    pdl::enter();
    const int token             = static_cast<int>(blockIdx.x);
    const int thread            = static_cast<int>(threadIdx.x);
    constexpr int kGroupsPerRow = Geometry::kGroupsPerRow;
    if (token >= tokens) {
        // Four consecutive groups of one token occupy one aligned 4-byte word of the tiled plane
        // (a 16-group tile row is 16 bytes), so the quad's first lane zeroes its word: one fourth
        // of the padding stores, each 4 bytes.
        if ((thread & 3) == 0) {
            store_vec(scales + nvfp4_tiled_scale_offset<Geometry, Layout>(token, thread),
                      std::uint32_t{0});
        }
        return;
    }

    const int lane              = thread & (kWarpSize - 1);
    const int warp              = thread / kWarpSize;
    const std::int64_t row_base = static_cast<std::int64_t>(token) * kVecsPerRow;
    const uint4 v0              = residual[row_base + thread];
    const uint4 v1              = residual[row_base + kBlock + thread];
    const uint4 w0              = norm_weight[thread];
    const uint4 w1              = norm_weight[kBlock + thread];

    float sum = rms_vec_sq(v0) + rms_vec_sq(v1);
    sum       = warp_reduce_sum(sum);

    __shared__ float warp_sums[kWarpsPerBlock];
    if (lane == 0) { warp_sums[warp] = sum; }
    __syncthreads();

    float total = warp_sums[0];
#pragma unroll
    for (int i = 1; i < kWarpsPerBlock; ++i) { total += warp_sums[i]; }
    const float inv = rsqrtf(total * (1.0f / static_cast<float>(Geometry::kInputRows)) + eps);

    __shared__ alignas(16) uint4 staged[kVecsPerRow];
    staged[thread]          = rms_vec_out<RmsEpilogue::Offset>(v0, w0, inv);
    staged[kBlock + thread] = rms_vec_out<RmsEpilogue::Offset>(v1, w1, inv);
    __syncthreads();

    const auto* staged_bf16 = reinterpret_cast<const __nv_bfloat16*>(staged);
    const Nvfp4QuantizedK16 quantized =
        quantize_nvfp4_k16(staged_bf16 + thread * 16, input_scale_divisor);
    auto* code_destination =
        codes + static_cast<std::int64_t>(token) * Geometry::kCodeBytesPerRow + thread * 8;
    store_vec(code_destination, make_uint2(quantized.codes_lo, quantized.codes_hi));
    // Four consecutive lanes hold four consecutive groups of this token, whose scale bytes are
    // one aligned word of the tiled plane: pack them and store 4 bytes once per quad.
    store_nvfp4_scale_quad(scales, nvfp4_tiled_scale_offset<Geometry, Layout>(token, thread),
                           quantized.scale, thread);
}

template <Nvfp4ScaleLayout Layout>
void launch_quantize(const Tensor& residual, const Tensor& norm_weight, float eps,
                     float input_scale_divisor, Nvfp4A4Workspace workspace, cudaStream_t stream) {
    const std::int32_t tokens = residual.ne[1];
    const int padded          = nvfp4_a4_padded_tokens(tokens, Layout);
    if (static_cast<std::size_t>(padded) * Geometry::kGroupsPerRow > workspace.scale_bytes) {
        throw std::logic_error("fused rmsnorm NVFP4 quantize: scale plane is too small");
    }
    // Serialized: the quantizer reads the residual its producer just wrote, so it keeps full
    // stream ordering; the downstream GEMM is the programmatic dependent of this grid.
    CUDA_CHECK(pdl::launch_with(pdl::Dependency::Serialized,
                                {dim3(padded), dim3(kBlock), 0, stream},
                                rmsnorm_nvfp4_quantize_kernel<Layout>,
                                static_cast<const uint4*>(residual.data),
                                static_cast<const uint4*>(norm_weight.data),
                                workspace.codes, workspace.scales, tokens, eps,
                                input_scale_divisor));
}

} // namespace

void launch_nvfp4_attn_input_fused_rmsnorm_quantize(const Tensor& residual,
                                                    const Tensor& norm_weight, float eps,
                                                    float input_scale_divisor,
                                                    Nvfp4A4Workspace workspace,
                                                    Nvfp4ScaleLayout layout, cudaStream_t stream) {
    switch (layout) {
    case Nvfp4ScaleLayout::Tiled128:
        launch_quantize<Nvfp4ScaleLayout::Tiled128>(residual, norm_weight, eps,
                                                    input_scale_divisor, workspace, stream);
        return;
    case Nvfp4ScaleLayout::Tiled256:
        launch_quantize<Nvfp4ScaleLayout::Tiled256>(residual, norm_weight, eps,
                                                    input_scale_divisor, workspace, stream);
        return;
    case Nvfp4ScaleLayout::RowMajor:
        break;
    }
    throw std::invalid_argument("fused rmsnorm NVFP4 quantize: the TMA route reads a tiled plane");
}

void nvfp4_attn_input_fused_rmsnorm_launch(const Tensor& residual, const Tensor& norm_weight,
                                           float eps, const Weight& weight, Tensor& q, Tensor& gate,
                                           Tensor& k, Tensor& v, WorkspaceArena& workspace,
                                           cudaStream_t stream) {
    const std::int32_t tokens      = residual.ne[1];
    const Nvfp4ScaleLayout layout  = nvfp4_attn_input_scale_layout(tokens);
    auto scope                     = workspace.scope();
    const Nvfp4A4Workspace scratch = allocate_nvfp4_a4_workspace(workspace, tokens, weight.k);
    launch_nvfp4_attn_input_fused_rmsnorm_quantize(residual, norm_weight, eps,
                                                   weight.input_scale_divisor, scratch, layout,
                                                   stream);
    // The GEMM reads only this quantizer's planes, so it launches as its programmatic dependent:
    // captured launches overlap its preamble with the quantizer's tail and wait inside for the
    // planes; eager launches keep full stream serialization.
    launch_nvfp4_a4_tma_attention(nvfp4_a4_operands(weight, scratch, tokens, layout),
                                  static_cast<__nv_bfloat16*>(q.data),
                                  static_cast<__nv_bfloat16*>(gate.data),
                                  static_cast<__nv_bfloat16*>(k.data),
                                  static_cast<__nv_bfloat16*>(v.data), stream,
                                  pdl::Dependency::Programmatic);
}

} // namespace infernix::ops::detail
