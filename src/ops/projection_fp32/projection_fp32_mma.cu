// The tensor-core route of projection_fp32 for BF16 vocabulary heads at 1..16 columns: the BF16 sliced-K
// MMA tile with an FP32 output. BF16 products are exact in FP32 and the MMA accumulates in FP32, so the
// Op's bound holds; one route serves every width, so a column's bits do not depend on the call's width
// (up to 16). The head's logits cost one read of its weights at every width: 701-708 us for
// [248320, 2560] against the narrow SIMT mapping's 1.05 ms at 8 columns and 1.5 ms at 9-16.

#include "ops/projection_fp32/projection_fp32_mma.h"

#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_launch.cuh"
#include "ops/linear/bf16/bf16_a16_tma_mma.cuh"
#include "ops/linear/common/epilogue.cuh"

namespace infernix::ops::detail {
namespace {

struct ProjectionFp32Output {
    float* data;
    std::int32_t rows;

    __device__ __forceinline__ void store(int row, int token, float value) const {
        data[static_cast<std::int64_t>(token) * rows + row] = value;
    }
};

using HeadSchedule   = Bf16A16SlicedKMmaSchedule<32, 16, 4, 64, 2, Cache::cg, Cache::ca, 2>;
using HeadSchedule32 = Bf16A16SlicedKMmaSchedule<32, 32, 4, 64, 2, Cache::cg, Cache::ca, 2>;

using RouterR16T16K256 = Bf16A16SlicedKMmaSchedule<16, 16, 8, 32, 2, Cache::cg, Cache::ca, 2>;
using RouterR16T32K256 = Bf16A16SlicedKMmaSchedule<16, 32, 8, 32, 2, Cache::cg, Cache::ca, 2>;
using RouterR16T16K64  = Bf16A16SlicedKMmaSchedule<16, 16, 4, 16, 2, Cache::cg, Cache::ca, 2>;
using RouterR16T32K64  = Bf16A16SlicedKMmaSchedule<16, 32, 4, 16, 2, Cache::cg, Cache::ca, 2>;

} // namespace

bool projection_fp32_bf16_wide_mma_supported(int rows, int k) noexcept {
    return rows > 0 && rows % Bf16A16TmaR64T64K64S3::kBlockRows == 0 && k % Bf16A16TmaR64T64K64S3::kBlockK == 0;
}

// Below 64 columns: the sliced-K BF16 tile on BlockRows = 16; 64..8,192 columns: 64-token TMA tiles;
// above 8,192 columns: 128-token TMA tiles.
void projection_fp32_bf16_wide_mma(const Tensor& x, const void* weight, int rows, int out_rows, float* out,
                                   cudaStream_t stream) {
    const Bf16A16Operands p{static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(weight),
                            rows, static_cast<int>(x.ne[0]), static_cast<int>(x.ne[1])};
    const ProjectionFp32Output output{out, out_rows};
    if (p.tokens < 64) {
        if (p.k % 256 == 0) {
            if (p.tokens <= 16) {
                launch_bf16_a16_sliced_k_mma<RouterR16T16K256>(p, output, LinearIdentityEpilogue{}, stream);
            } else {
                launch_bf16_a16_sliced_k_mma<RouterR16T32K256>(p, output, LinearIdentityEpilogue{}, stream);
            }
        } else {
            if (p.tokens <= 16) {
                launch_bf16_a16_sliced_k_mma<RouterR16T16K64>(p, output, LinearIdentityEpilogue{}, stream);
            } else {
                launch_bf16_a16_sliced_k_mma<RouterR16T32K64>(p, output, LinearIdentityEpilogue{}, stream);
            }
        }
        return;
    }
    if (p.tokens > 512 && rows <= 2048) {
        launch_bf16_a16_tma_mma<Bf16A16TmaR64T128K64S2Rows>(p, output, LinearIdentityEpilogue{}, stream);
    } else if (p.tokens <= 8192) {
        launch_bf16_a16_tma_mma<Bf16A16TmaR64T64K64S3>(p, output, LinearIdentityEpilogue{}, stream);
    } else {
        launch_bf16_a16_tma_mma<Bf16A16TmaR64T128K64S2>(p, output, LinearIdentityEpilogue{}, stream);
    }
}

bool projection_fp32_bf16_mma_supported(int rows, int k, int columns) noexcept {
    return columns >= 1 && columns <= kProjectionMmaMaxColumns && rows % HeadSchedule::kBlockRows == 0 &&
           k % HeadSchedule::kBlockK == 0;
}

void projection_fp32_bf16_mma(const Tensor& x, const void* weight, int rows, Tensor& out, cudaStream_t stream) {
    const Bf16A16Operands p{static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(weight),
                            rows, static_cast<int>(x.ne[0]), static_cast<int>(x.ne[1])};
    const ProjectionFp32Output output{static_cast<float*>(out.data), rows};
    if (p.tokens <= 16) {
        launch_bf16_a16_sliced_k_mma<HeadSchedule>(p, output, LinearIdentityEpilogue{}, stream);
    } else {
        launch_bf16_a16_sliced_k_mma<HeadSchedule32>(p, output, LinearIdentityEpilogue{}, stream);
    }
}

} // namespace infernix::ops::detail
