// Native sm_121a NVFP4 OMMA narrow/medium W4A4 MoE kernels (1 <= count <= moe::kMaxColumns,
// or split-input experts via count-8 chunking), included by moe_layer.cu inside its anonymous namespace.
//
// Matrix A = 16 weight rows x 64 K elements (`rg16_a_fragments_seq`, sequential units u0..u3);
// Matrix B = up to 8 token/activation columns x 64 K elements (`OmmaXChunk` in SMEM, zero bank conflicts);
// Output C = FP32 16 x 8 (`c0`=(g, 2t), `c1`=(g, 2t+1), `c2`=(g+8, 2t), `c3`=(g+8, 2t+1)).
// Both A and B use `plane_block` [0, 2, 1, 3] ordering within each 4-block K64 chunk for `scale_vec::4X`.
// Intermediate activations `h` are stored in `h_blocks` as `OmmaHRow` (368 B <= kHRowStride 480 B)
// so down_omma reads h directly with zero SMEM bank conflicts and zero cross-warp reduction.

struct OmmaXChunk {
    std::uint32_t codes[16][8][2]; // [pos = 4*k64 + plane_block(b & 3)][col 0..7][lo=0 / hi=1]
    std::uint32_t scales4[4][8];   // [k64 0..3][col 0..7] -> bytes [b0, b2, b1, b3] E4M3FN scales
};

template <class G>
struct OmmaShape {
    static constexpr int kThreads        = 128; // 4 warps
    static constexpr int kWarps          = 4;
    static constexpr int kGateUpSlices   = G::kGateUpRowGroups / 2; // 40 (16 intermediates = 1 h block each)
    static constexpr int kGateUpChunks   = G::kGateUpBlocks / 16;   // 10 chunks x 16 units
    static constexpr int kGateUpStageB   = 2 * 16 * static_cast<int>(moe::kUnitBytes); // 4,608 B
    static constexpr int kDownTiles      = G::kDownRowGroups / 4;   // 40 (4 row groups per tile)
    static constexpr int kDownChunks     = G::kDownBlocks / 8;      // 5 stages x 8 units
    static constexpr int kDownWarpStageB = 8 * static_cast<int>(moe::kUnitBytes);      // 1,152 B
    static constexpr int kDownRgBytes    = G::kDownBlocks * static_cast<int>(moe::kUnitBytes); // 5,760 B
    static constexpr int kHRowBytes      = 368; // 320 code B + 40 scale B, odd x 16 B -> 0 bank conflicts
    static_assert(G::kGateUpBlocks % 16 == 0 && G::kDownBlocks % 8 == 0 && G::kDownRowGroups % 4 == 0);
    static_assert(moe::wide::kHRowStride >= kHRowBytes && moe::wide::kHCodeBytes == 320);
};

__device__ __forceinline__ std::uint32_t omma_lds32(const std::uint8_t* p) {
    return *reinterpret_cast<const std::uint32_t*>(p);
}

__device__ __forceinline__ void rg16_a_fragments_seq(const std::uint8_t* units, int group, int t, std::uint32_t& a0,
                                                      std::uint32_t& a1, std::uint32_t& a2, std::uint32_t& a3,
                                                      std::uint32_t& sfa) {
    const int quad            = 64 * (t & 1) + 4 * group;
    const std::uint8_t* u02   = units + static_cast<std::size_t>(2 * (t >> 1)) * moe::kUnitBytes + quad;
    const std::uint8_t* u13   = units + static_cast<std::size_t>(1 + 2 * (t >> 1)) * moe::kUnitBytes + quad;
    const std::uint32_t w0    = omma_lds32(u02);
    const std::uint32_t w1    = omma_lds32(u02 + 32);
    const std::uint32_t v0    = omma_lds32(u13);
    const std::uint32_t v1    = omma_lds32(u13 + 32);
    a0                        = (w0 & 0x0F0F0F0FU) | ((w1 & 0x0F0F0F0FU) << 4);
    a1                        = ((w0 >> 4) & 0x0F0F0F0FU) | (w1 & 0xF0F0F0F0U);
    a2                        = (v0 & 0x0F0F0F0FU) | ((v1 & 0x0F0F0F0FU) << 4);
    a3                        = ((v0 >> 4) & 0x0F0F0F0FU) | (v1 & 0xF0F0F0F0U);
    sfa                       = 0U;
    if ((t & 2) == 0) {
        const int sw            = 128 + ((t & 1) << 3) + 4 * (group >> 2);
        const unsigned sel      = static_cast<unsigned>(group & 3) | (static_cast<unsigned>(4 + (group & 3)) << 4);
        const std::uint32_t s0  = omma_lds32(units + sw);
        const std::uint32_t s1  = omma_lds32(units + moe::kUnitBytes + sw);
        const std::uint32_t s2  = omma_lds32(units + 2 * moe::kUnitBytes + sw);
        const std::uint32_t s3  = omma_lds32(units + 3 * moe::kUnitBytes + sw);
        sfa                     = __byte_perm(__byte_perm(s0, s2, sel), __byte_perm(s1, s3, sel), 0x5410);
    }
}

template <class G>
__device__ __forceinline__ void omma_load_x_f32_amax(const bf16* x, int token_col, int chunk, int b_local,
                                                     float (&v)[16], float& amax) {
    const auto* src = reinterpret_cast<const uint4*>(
        reinterpret_cast<const std::uint16_t*>(x) + static_cast<std::size_t>(token_col) * G::kHidden +
        static_cast<std::size_t>(chunk * 16 + b_local) * 16);
    const uint4 s0 = src[0], s1 = src[1];
    const unsigned words[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
    amax = 0.0F;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float f0 = canon::bf16_to_f32(static_cast<std::uint16_t>(words[i] & 0xFFFFU));
        const float f1 = canon::bf16_to_f32(static_cast<std::uint16_t>(words[i] >> 16));
        v[2 * i]       = f0;
        v[2 * i + 1]   = f1;
        const float a0 = fabsf(f0);
        const float a1 = fabsf(f1);
        amax           = a0 > amax || a0 != a0 ? a0 : amax;
        amax           = a1 > amax || a1 != a1 ? a1 : amax;
    }
}

__device__ __forceinline__ std::uint32_t omma_pack_scales4(std::uint8_t sw, int lane) {
    const unsigned base_lane = static_cast<unsigned>(lane & ~3);
    const unsigned s0        = __shfl_sync(0xFFFFFFFFU, static_cast<unsigned>(sw), base_lane + 0U);
    const unsigned s1        = __shfl_sync(0xFFFFFFFFU, static_cast<unsigned>(sw), base_lane + 2U);
    const unsigned s2        = __shfl_sync(0xFFFFFFFFU, static_cast<unsigned>(sw), base_lane + 1U);
    const unsigned s3        = __shfl_sync(0xFFFFFFFFU, static_cast<unsigned>(sw), base_lane + 3U);
    return s0 | (s1 << 8) | (s2 << 16) | (s3 << 24);
}

__device__ __forceinline__ float omma_rcp_nr(float d) {
    float r;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(d));
    return fmaf(r, fmaf(-d, r, 1.0F), r);
}

__device__ __forceinline__ void omma_quantize_a4_plane_from_f32(const float (&v)[16], float amax, float input_scale,
                                                                unsigned& low, unsigned& high,
                                                                std::uint8_t& scale_word) {
    const float g6           = canon::mul_rn(6.0F, input_scale);
    const std::uint32_t g6_u = canon::f32_bits(g6) & 0x7FFFFFFFU;
    float ratio;
    if (amax == 0.0F) {
        ratio = 0.0F;
    } else if (g6_u >= 0x08000000U && g6_u < 0x70000000U) {
        ratio = canon::mul_rn(amax, omma_rcp_nr(g6));
    } else {
        ratio = __fdividef(amax, g6);
    }
    scale_word = canon::e4m3_rn_satfinite(ratio);
    if (canon::e4m3_scaled(scale_word) == 0) {
        low  = 0U;
        high = 0U;
        return;
    }
    const float d           = canon::mul_rn(canon::e4m3_value(scale_word), input_scale);
    const std::uint32_t d_u = canon::f32_bits(d) & 0x7FFFFFFFU;
    const float inv_d       = (d_u >= 0x08000000U && d_u < 0x70000000U) ? omma_rcp_nr(d) : __fdividef(1.0F, d);
    float q[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) { q[j] = canon::mul_rn(v[j], inv_d); }
    asm("{\n\t"
        ".reg .b8 b0, b1, b2, b3, b4, b5, b6, b7;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 b0, %6, %2;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 b1, %7, %3;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 b2, %8, %4;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 b3, %9, %5;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 b4, %14, %10;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 b5, %15, %11;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 b6, %16, %12;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 b7, %17, %13;\n\t"
        "mov.b32 %0, {b0, b1, b2, b3};\n\t"
        "mov.b32 %1, {b4, b5, b6, b7};\n\t"
        "}"
        : "=r"(low), "=r"(high)
        : "f"(q[0]), "f"(q[1]), "f"(q[2]), "f"(q[3]), "f"(q[4]), "f"(q[5]), "f"(q[6]), "f"(q[7]),
          "f"(q[8]), "f"(q[9]), "f"(q[10]), "f"(q[11]), "f"(q[12]), "f"(q[13]), "f"(q[14]), "f"(q[15]));
}

__device__ __forceinline__ std::uint16_t omma_swiglu_bf16(std::uint16_t gate_bf16, std::uint16_t up_bf16) {
    const float g     = canon::bf16_to_f32(gate_bf16);
    const float u     = canon::bf16_to_f32(up_bf16);
    const float abs_g = fabsf(g);
    const float e     = __expf(-abs_g);
    const float denom = 1.0F + e;
    const float inv   = omma_rcp_nr(denom);
    const float silu  = (g >= 0.0F ? g : canon::mul_rn(g, e)) * inv;
    return canon::f32_to_bf16_rn(canon::mul_rn(silu, u));
}

template <class G>
__device__ __forceinline__ void omma_quantize_x_chunk(const bf16* x, int token_col, bool live_col, int chunk,
                                                      float input_scale, OmmaXChunk& dst) {
    const int tid     = static_cast<int>(threadIdx.x);
    const int lane    = tid & 31;
    const int col     = tid >> 4;   // 0..7
    const int b_local = tid & 15;   // 0..15 blocks within the 256-elem chunk
    const int k64     = b_local >> 2;
    std::uint8_t sw   = 0U;
    if (live_col) {
        float v[16];
        float amax = 0.0F;
        omma_load_x_f32_amax<G>(x, token_col, chunk, b_local, v, amax);
        unsigned low = 0U, high = 0U;
        omma_quantize_a4_plane_from_f32(v, amax, input_scale, low, high, sw);
        const int pos = 4 * k64 + moe::wide::plane_block(b_local & 3);
        *reinterpret_cast<uint2*>(dst.codes[pos][col]) = make_uint2(low, high);
    }
    const std::uint32_t packed = omma_pack_scales4(sw, lane);
    if (live_col && (b_local & 3) == 0) { dst.scales4[k64][col] = packed; }
}

template <class G>
__device__ __forceinline__ void omma_quantize_x_chunk_split(const bf16* x, int token_col, bool live_col, int chunk,
                                                            float scale_gate, float scale_up, OmmaXChunk& dst_gate,
                                                            OmmaXChunk& dst_up) {
    const int tid     = static_cast<int>(threadIdx.x);
    const int lane    = tid & 31;
    const int col     = tid >> 4;
    const int b_local = tid & 15;
    const int k64     = b_local >> 2;
    std::uint8_t sw_g = 0U, sw_u = 0U;
    if (live_col) {
        float v[16];
        float amax = 0.0F;
        omma_load_x_f32_amax<G>(x, token_col, chunk, b_local, v, amax);
        const int pos = 4 * k64 + moe::wide::plane_block(b_local & 3);
        unsigned low = 0U, high = 0U;
        omma_quantize_a4_plane_from_f32(v, amax, scale_gate, low, high, sw_g);
        *reinterpret_cast<uint2*>(dst_gate.codes[pos][col]) = make_uint2(low, high);
        omma_quantize_a4_plane_from_f32(v, amax, scale_up, low, high, sw_u);
        *reinterpret_cast<uint2*>(dst_up.codes[pos][col]) = make_uint2(low, high);
    }
    const std::uint32_t pg = omma_pack_scales4(sw_g, lane);
    const std::uint32_t pu = omma_pack_scales4(sw_u, lane);
    if (live_col && (b_local & 3) == 0) {
        dst_gate.scales4[k64][col] = pg;
        dst_up.scales4[k64][col]   = pu;
    }
}

__device__ __forceinline__ void omma_bulk_load_hinted(void* destination, const void* source, std::uint32_t bytes,
                                                      std::uint64_t* barrier, std::uint64_t cache_policy) {
    asm volatile(
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
        :
        : "r"(smem_addr(destination)), "l"(source), "r"(bytes), "r"(smem_addr(barrier)), "l"(cache_policy)
        : "memory");
}

template <class G>
struct GateUpOmmaShared {
    union {
        struct {
            alignas(16) std::uint8_t w[2][2][16 * moe::kUnitBytes]; // 9,216 B
            OmmaXChunk x[2];                                         // 2,304 B
        } pipe;
        struct {
            float partial[2][16][8]; // 1,024 B
            std::uint16_t h[8][16];  //   256 B
        } epi;
    };
    alignas(8) std::uint64_t mbar[2];
};

template <class G>
struct DownOmmaShared {
    alignas(16) std::uint8_t w[2][OmmaShape<G>::kWarps][OmmaShape<G>::kDownWarpStageB]; // 9,216 B
    alignas(16) std::uint8_t h[8][OmmaShape<G>::kHRowBytes];                              // 2,944 B
    alignas(8) std::uint64_t mbar[2];
};

__device__ __forceinline__ void omma_stage_xplane_chunk(const std::uint8_t* x_plane, int first_entry, int n_cols,
                                                        int chunk, OmmaXChunk& dst) {
    const int tid = static_cast<int>(threadIdx.x);
    if (tid < 8 * n_cols) {
        const int col = tid >> 3;
        const int v   = tid & 7;
        const auto* src = reinterpret_cast<const uint4*>(
            x_plane + static_cast<std::size_t>(first_entry + col) * moe::wide::kXRowBytes + 128 * chunk + 16 * v);
        const uint4 q                = __ldg(src);
        dst.codes[2 * v][col][0]     = q.x;
        dst.codes[2 * v][col][1]     = q.y;
        dst.codes[2 * v + 1][col][0] = q.z;
        dst.codes[2 * v + 1][col][1] = q.w;
    } else if (tid >= 64 && tid < 64 + n_cols) {
        const int col = tid - 64;
        const auto* src = reinterpret_cast<const uint4*>(
            x_plane + static_cast<std::size_t>(first_entry + col) * moe::wide::kXRowBytes +
            moe::wide::kXCodeBytes + 16 * chunk);
        const uint4 sc      = __ldg(src);
        dst.scales4[0][col] = sc.x;
        dst.scales4[1][col] = sc.y;
        dst.scales4[2][col] = sc.z;
        dst.scales4[3][col] = sc.w;
    }
}

template <class G, bool UseXPlane = false>
__device__ __forceinline__ void gate_up_omma_slice(GateUpOmmaShared<G>& sm, const bf16* x,
                                                     const MoeDispatch& dispatch, int top_k,
                                                     const std::uint8_t* record, const moe::ExpertScales& scales,
                                                     int first, int count, int slice, const std::uint8_t* x_plane,
                                                     std::uint8_t* h_base, std::uint32_t& phase_bits) {
    using S              = OmmaShape<G>;
    const int tid        = static_cast<int>(threadIdx.x);
    const int warp       = tid >> 5;
    const int lane       = tid & 31;
    const int group      = lane >> 2;
    const int t          = lane & 3;
    const int rg_local   = warp & 1; // 0: intermediates 0..7, 1: intermediates 8..15
    const int ks         = warp >> 1; // 0: K64 steps 0,2 in chunk; 1: K64 steps 1,3 in chunk
    const bool split_in  = scales.input_gate != scales.input_up;
    const std::uint8_t* units_base =
        record + static_cast<std::size_t>(2 * slice) * G::kGateUpBlocks * moe::kUnitBytes;

    const auto issue_w = [&](int chunk, int buf) {
        if (tid == 0) {
            constexpr int kRgBytes            = 16 * static_cast<int>(moe::kUnitBytes); // 2,304 B
            const std::size_t k_off           = static_cast<std::size_t>(chunk) * kRgBytes;
            const std::uint64_t weight_policy = l2_evict_first_policy();
            cta_mbarrier_arrive_expect_tx(&sm.mbar[buf], static_cast<std::uint32_t>(2 * kRgBytes));
            omma_bulk_load_hinted(sm.pipe.w[buf][0], units_base + k_off, kRgBytes, &sm.mbar[buf], weight_policy);
            omma_bulk_load_hinted(sm.pipe.w[buf][1],
                                  units_base + static_cast<std::size_t>(G::kGateUpBlocks) * moe::kUnitBytes + k_off,
                                  kRgBytes, &sm.mbar[buf], weight_policy);
        }
    };

    for (int c_base = 0; c_base < count; c_base += 8) {
        const int n_cols = min(8, count - c_base);

        float acc[4];
        if (!split_in) {
            int tok       = 0;
            bool live_col = false;
            if constexpr (!UseXPlane) {
                const int col = tid >> 4;
                live_col      = col < n_cols;
                tok           = live_col
                                    ? moe::wide::entry_column(dispatch.entries[first + c_base + col], top_k,
                                                              dispatch.shared_first)
                                    : 0;
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) { acc[i] = 0.0F; }
            issue_w(0, 0);
            if constexpr (UseXPlane) {
                omma_stage_xplane_chunk(x_plane, first + c_base, n_cols, 0, sm.pipe.x[0]);
            } else {
                omma_quantize_x_chunk<G>(x, tok, live_col, 0, scales.input_gate, sm.pipe.x[0]);
            }
            issue_w(1, 1);
            cta_mbarrier_wait(&sm.mbar[0], phase_bits & 1U);
            phase_bits ^= 1U;
            __syncthreads();

#pragma unroll 1
            for (int ch = 0; ch < S::kGateUpChunks; ++ch) {
                const OmmaXChunk& xc = sm.pipe.x[ch & 1];
                const int b_col      = min(group, n_cols - 1);
                const std::uint8_t* wu = sm.pipe.w[ch & 1][rg_local];
#pragma unroll
                for (int k_step = 0; k_step < 2; ++k_step) {
                    const int w_idx = ks + 2 * k_step; // {0, 2} for ks=0, {1, 3} for ks=1
                    std::uint32_t a0, a1, a2, a3, sfa;
                    rg16_a_fragments_seq(wu + w_idx * 4 * moe::kUnitBytes, group, t, a0, a1, a2, a3, sfa);
                    const std::uint32_t b0  = xc.codes[4 * w_idx + (t >> 1)][b_col][t & 1];
                    const std::uint32_t b1  = xc.codes[4 * w_idx + 2 + (t >> 1)][b_col][t & 1];
                    const std::uint32_t sfb = xc.scales4[w_idx][b_col];
                    mma_nvfp4_e4m3(acc[0], acc[1], acc[2], acc[3], a0, a1, a2, a3, b0, b1, sfa, sfb);
                }
                if (ch + 1 < S::kGateUpChunks) {
                    if constexpr (UseXPlane) {
                        omma_stage_xplane_chunk(x_plane, first + c_base, n_cols, ch + 1, sm.pipe.x[(ch + 1) & 1]);
                    } else {
                        omma_quantize_x_chunk<G>(x, tok, live_col, ch + 1, scales.input_gate, sm.pipe.x[(ch + 1) & 1]);
                    }
                }
                fence_reads_before_async_refill();
                __syncthreads();
                if (ch + 2 < S::kGateUpChunks) { issue_w(ch + 2, ch & 1); }
                if (ch + 1 < S::kGateUpChunks) {
                    const int nxt = (ch + 1) & 1;
                    cta_mbarrier_wait(&sm.mbar[nxt], (phase_bits >> nxt) & 1U);
                    phase_bits ^= (1U << nxt);
                }
            }
        } else {
            const int col       = tid >> 4;
            const bool live_col = col < n_cols;
            const int tok =
                live_col
                    ? moe::wide::entry_column(dispatch.entries[first + c_base + col], top_k, dispatch.shared_first)
                    : 0;
            float acc_g[4] = {0.0F, 0.0F, 0.0F, 0.0F};
            float acc_u[4] = {0.0F, 0.0F, 0.0F, 0.0F};
            issue_w(0, 0);
            omma_quantize_x_chunk_split<G>(x, tok, live_col, 0, scales.input_gate, scales.input_up, sm.pipe.x[0],
                                           sm.pipe.x[1]);
            issue_w(1, 1);
            cta_mbarrier_wait(&sm.mbar[0], phase_bits & 1U);
            phase_bits ^= 1U;
            __syncthreads();

#pragma unroll 1
            for (int ch = 0; ch < S::kGateUpChunks; ++ch) {
                const int b_col      = min(group, n_cols - 1);
                const std::uint8_t* wu = sm.pipe.w[ch & 1][rg_local];
#pragma unroll
                for (int k_step = 0; k_step < 2; ++k_step) {
                    const int w_idx = ks + 2 * k_step;
                    std::uint32_t a0, a1, a2, a3, sfa;
                    rg16_a_fragments_seq(wu + w_idx * 4 * moe::kUnitBytes, group, t, a0, a1, a2, a3, sfa);
                    const std::uint32_t bg0 = sm.pipe.x[0].codes[4 * w_idx + (t >> 1)][b_col][t & 1];
                    const std::uint32_t bg1 = sm.pipe.x[0].codes[4 * w_idx + 2 + (t >> 1)][b_col][t & 1];
                    const std::uint32_t sg  = sm.pipe.x[0].scales4[w_idx][b_col];
                    mma_nvfp4_e4m3(acc_g[0], acc_g[1], acc_g[2], acc_g[3], a0, a1, a2, a3, bg0, bg1, sfa, sg);
                    const std::uint32_t bu0 = sm.pipe.x[1].codes[4 * w_idx + (t >> 1)][b_col][t & 1];
                    const std::uint32_t bu1 = sm.pipe.x[1].codes[4 * w_idx + 2 + (t >> 1)][b_col][t & 1];
                    const std::uint32_t su  = sm.pipe.x[1].scales4[w_idx][b_col];
                    mma_nvfp4_e4m3(acc_u[0], acc_u[1], acc_u[2], acc_u[3], a0, a1, a2, a3, bu0, bu1, sfa, su);
                }
                fence_reads_before_async_refill();
                __syncthreads();
                if (ch + 2 < S::kGateUpChunks) { issue_w(ch + 2, ch & 1); }
                if (ch + 1 < S::kGateUpChunks) {
                    omma_quantize_x_chunk_split<G>(x, tok, live_col, ch + 1, scales.input_gate, scales.input_up,
                                                   sm.pipe.x[0], sm.pipe.x[1]);
                    const int nxt = (ch + 1) & 1;
                    cta_mbarrier_wait(&sm.mbar[nxt], (phase_bits >> nxt) & 1U);
                    phase_bits ^= (1U << nxt);
                    __syncthreads();
                }
            }
            const bool is_gate = (group & 1) == 0;
#pragma unroll
            for (int i = 0; i < 4; ++i) { acc[i] = is_gate ? acc_g[i] : acc_u[i]; }
        }

        if (ks == 1) {
            sm.epi.partial[rg_local][group][2 * t]         = acc[0];
            sm.epi.partial[rg_local][group][2 * t + 1]     = acc[1];
            sm.epi.partial[rg_local][group + 8][2 * t]     = acc[2];
            sm.epi.partial[rg_local][group + 8][2 * t + 1] = acc[3];
        }
        __syncthreads();
        if (ks == 0) {
            const float s0     = acc[0] + sm.epi.partial[rg_local][group][2 * t];
            const float s1     = acc[1] + sm.epi.partial[rg_local][group][2 * t + 1];
            const float s2     = acc[2] + sm.epi.partial[rg_local][group + 8][2 * t];
            const float s3     = acc[3] + sm.epi.partial[rg_local][group + 8][2 * t + 1];
            const bool is_gate = (group & 1) == 0;
            const float alpha  = is_gate ? scales.alpha_gate : scales.alpha_up;
            const std::uint16_t y0 = canon::f32_to_bf16_rn(canon::mul_rn(s0, alpha));
            const std::uint16_t y1 = canon::f32_to_bf16_rn(canon::mul_rn(s1, alpha));
            const std::uint16_t y2 = canon::f32_to_bf16_rn(canon::mul_rn(s2, alpha));
            const std::uint16_t y3 = canon::f32_to_bf16_rn(canon::mul_rn(s3, alpha));
            const auto u0          = static_cast<std::uint16_t>(__shfl_down_sync(0xFFFFFFFFU, y0, 4));
            const auto u1          = static_cast<std::uint16_t>(__shfl_down_sync(0xFFFFFFFFU, y1, 4));
            const auto u2          = static_cast<std::uint16_t>(__shfl_down_sync(0xFFFFFFFFU, y2, 4));
            const auto u3          = static_cast<std::uint16_t>(__shfl_down_sync(0xFFFFFFFFU, y3, 4));
            if (is_gate) {
                const int i_lo = 8 * rg_local + (group >> 1);
                const int i_hi = i_lo + 4;
                const int c0 = 2 * t, c1 = 2 * t + 1;
                if (c0 < n_cols) {
                    sm.epi.h[c0][i_lo] = omma_swiglu_bf16(y0, u0);
                    sm.epi.h[c0][i_hi] = omma_swiglu_bf16(y2, u2);
                }
                if (c1 < n_cols) {
                    sm.epi.h[c1][i_lo] = omma_swiglu_bf16(y1, u1);
                    sm.epi.h[c1][i_hi] = omma_swiglu_bf16(y3, u3);
                }
            }
        }
        __syncthreads();
        if (tid < n_cols) {
            float hv[16];
            float amax = 0.0F;
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const float f = canon::bf16_to_f32(sm.epi.h[tid][j]);
                hv[j]         = f;
                const float a = fabsf(f);
                amax          = a > amax || a != a ? a : amax;
            }
            unsigned low = 0U, high = 0U;
            std::uint8_t sw = 0U;
            omma_quantize_a4_plane_from_f32(hv, amax, scales.input_down, low, high, sw);
            const int pos       = (slice & ~3) + moe::wide::plane_block(slice & 3);
            std::uint8_t* h_row = h_base + static_cast<std::size_t>(first + c_base + tid) * moe::wide::kHRowStride;
            *reinterpret_cast<uint2*>(h_row + 8 * pos) = make_uint2(low, high);
            h_row[moe::wide::kHCodeBytes + pos]        = sw;
        }
        if (c_base + 8 < count) {
            fence_reads_before_async_refill();
            __syncthreads();
        }
    }
}

template <class G>
__device__ __forceinline__ void down_omma_tile_compute(DownOmmaShared<G>& sm, const MoeDispatch& dispatch,
                                                       const std::uint8_t* record, const moe::ExpertScales& scales,
                                                       int first, int count, int tile, const std::uint8_t* h_base,
                                                       bf16* outputs, bool preloaded_stage0,
                                                       std::uint32_t& phase_bits) {
    using S        = OmmaShape<G>;
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int group = lane >> 2;
    const int t     = lane & 3;
    const int rg    = 4 * tile + warp;
    const std::uint8_t* rg_base =
        record + G::kGateUpBytes + static_cast<std::size_t>(4 * tile) * S::kDownRgBytes;

    const auto stage_w = [&](int st, int buf) {
        if (tid == 0) {
            cta_mbarrier_arrive_expect_tx(&sm.mbar[buf], static_cast<std::uint32_t>(4 * S::kDownWarpStageB));
            const std::size_t k_off           = static_cast<std::size_t>(st) * S::kDownWarpStageB;
            const std::uint64_t weight_policy = l2_evict_first_policy();
#pragma unroll
            for (int w = 0; w < S::kWarps; ++w) {
                const std::uint8_t* src = rg_base + static_cast<std::size_t>(w) * S::kDownRgBytes + k_off;
                omma_bulk_load_hinted(sm.w[buf][w], src, S::kDownWarpStageB, &sm.mbar[buf], weight_policy);
            }
        }
    };

    for (int c_base = 0; c_base < count; c_base += 8) {
        const int n_cols = min(8, count - c_base);
        if (!preloaded_stage0 || c_base > 0) { stage_w(0, 0); }
        constexpr int kHVecs = S::kHRowBytes / 16; // 23
        for (int i = tid; i < n_cols * kHVecs; i += S::kThreads) {
            const int c = i / kHVecs;
            const int k = i % kHVecs;
            ggml::tile::cp_async16(
                sm.h[c] + 16 * k,
                h_base + static_cast<std::size_t>(first + c_base + c) * moe::wide::kHRowStride + 16 * k, 16);
        }
        ggml::tile::cp_async_commit();
        stage_w(1, 1);
        ggml::tile::cp_async_wait<0>();
        cta_mbarrier_wait(&sm.mbar[0], phase_bits & 1U);
        phase_bits ^= 1U;
        __syncthreads();

        float acc[4]              = {0.0F, 0.0F, 0.0F, 0.0F};
        const std::uint8_t* h_row = sm.h[min(group, n_cols - 1)];
        for (int st = 0; st < S::kDownChunks; ++st) {
            const std::uint8_t* wu = sm.w[st & 1][warp];
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int k64 = 2 * st + half;
                std::uint32_t a0, a1, a2, a3, sfa;
                rg16_a_fragments_seq(wu + half * 4 * moe::kUnitBytes, group, t, a0, a1, a2, a3, sfa);
                const std::uint32_t b0  = omma_lds32(h_row + 32 * k64 + 4 * t);
                const std::uint32_t b1  = omma_lds32(h_row + 32 * k64 + 16 + 4 * t);
                const std::uint32_t sfb = omma_lds32(h_row + moe::wide::kHCodeBytes + 4 * k64);
                mma_nvfp4_e4m3(acc[0], acc[1], acc[2], acc[3], a0, a1, a2, a3, b0, b1, sfa, sfb);
            }
            fence_reads_before_async_refill();
            __syncthreads();
            if (st + 2 < S::kDownChunks) { stage_w(st + 2, st & 1); }
            if (st + 1 < S::kDownChunks) {
                const int nxt = (st + 1) & 1;
                cta_mbarrier_wait(&sm.mbar[nxt], (phase_bits >> nxt) & 1U);
                phase_bits ^= (1U << nxt);
            }
        }

        const float alpha = scales.alpha_down;
        const int row0    = 16 * rg + group;
        const int row1    = row0 + 8;
        auto* out_u16     = reinterpret_cast<std::uint16_t*>(outputs);
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int col_local = 2 * t + i;
            if (col_local < n_cols) {
                const int out_col = dispatch.entries[first + c_base + col_local];
                std::uint16_t* dst = out_u16 + static_cast<std::size_t>(out_col) * G::kHidden;
                dst[row0]          = canon::f32_to_bf16_rn(canon::mul_rn(acc[i], alpha));
                dst[row1]          = canon::f32_to_bf16_rn(canon::mul_rn(acc[2 + i], alpha));
            }
        }
        if (c_base + 8 < count) { __syncthreads(); }
    }
}

template <class G>
__global__ void __launch_bounds__(OmmaShape<G>::kThreads, 7)
    gate_up_omma_kernel(const bf16* x, MoeDispatch dispatch, MoeExpertSource source, int top_k,
                        const std::uint8_t* const* job_records, const std::int32_t* flags,
                        int base, std::uint8_t* __restrict__ h_base, PassPhase phase) {
    __shared__ GateUpOmmaShared<G> sm;
    if (threadIdx.x == 0) {
        cta_mbarrier_init(&sm.mbar[0], 1);
        cta_mbarrier_init(&sm.mbar[1], 1);
    }
    cta_mbarrier_fence_init();
    __syncthreads();
    std::uint32_t mbar_phase = 0U;
    pdl::wait_for_dependencies();
    const int job   = base + static_cast<int>(blockIdx.y);
    const int slice = static_cast<int>(blockIdx.x);
    if (job < *dispatch.job_count && (flags == nullptr || flags[job] == 0)) {
        const int expert           = dispatch.jobs[job];
        const std::uint8_t* record = pass_record(phase, source, expert, job_records, job);
        if (record != nullptr) {
            const moe::ExpertScales scales = source.scales[expert];
            const int first = dispatch.offsets[expert], count = dispatch.offsets[expert + 1] - first;
            if (count > 0 && !moe::wide_route(count, scales)) {
                gate_up_omma_slice<G, false>(sm, x, dispatch, top_k, record, scales, first, count, slice, nullptr,
                                             h_base, mbar_phase);
            }
        }
    }
    pdl::trigger_dependents();
}

template <class G>
__global__ void __launch_bounds__(OmmaShape<G>::kThreads, 7)
    down_omma_kernel(MoeDispatch dispatch, MoeExpertSource source, const std::uint8_t* const* job_records,
                     const std::int32_t* flags, int base, const std::uint8_t* h_base,
                     bf16* __restrict__ outputs, PassPhase phase) {
    using S = OmmaShape<G>;
    __shared__ DownOmmaShared<G> sm;
    if (threadIdx.x == 0) {
        cta_mbarrier_init(&sm.mbar[0], 1);
        cta_mbarrier_init(&sm.mbar[1], 1);
    }
    cta_mbarrier_fence_init();
    __syncthreads();
    std::uint32_t mbar_phase = 0U;

    const int job  = base + static_cast<int>(blockIdx.y);
    const int tile = static_cast<int>(blockIdx.x);
    const int tid  = static_cast<int>(threadIdx.x);

    int expert = 0, first = 0, count = 0;
    const std::uint8_t* record = nullptr;
    moe::ExpertScales scales{};
    bool preloaded = false;
    // dispatch.* is read-only across gate/up -> down, so prefetching immutable down weights stage 0
    // before the PDL wait hides latency without depending on h_base.
    if (job < *dispatch.job_count && (flags == nullptr || flags[job] == 0)) {
        expert = dispatch.jobs[job];
        record = pass_record(phase, source, expert, job_records, job);
        if (record != nullptr) {
            scales = source.scales[expert];
            first  = dispatch.offsets[expert];
            count  = dispatch.offsets[expert + 1] - first;
            if (count <= 0 || moe::wide_route(count, scales)) {
                record = nullptr;
            } else {
                if (tid == 0) {
                    const std::uint8_t* rg_base =
                        record + G::kGateUpBytes + static_cast<std::size_t>(4 * tile) * S::kDownRgBytes;
                    const std::uint64_t weight_policy = l2_evict_first_policy();
                    cta_mbarrier_arrive_expect_tx(&sm.mbar[0], static_cast<std::uint32_t>(4 * S::kDownWarpStageB));
#pragma unroll
                    for (int w = 0; w < S::kWarps; ++w) {
                        const std::uint8_t* src = rg_base + static_cast<std::size_t>(w) * S::kDownRgBytes;
                        omma_bulk_load_hinted(sm.w[0][w], src, S::kDownWarpStageB, &sm.mbar[0], weight_policy);
                    }
                }
                preloaded = true;
            }
        }
    }
    pdl::wait_for_dependencies();
    if (record == nullptr) { return; }
    down_omma_tile_compute<G>(sm, dispatch, record, scales, first, count, tile, h_base, outputs, preloaded, mbar_phase);
}

template <class G>
__global__ void __launch_bounds__(OmmaShape<G>::kThreads, 7)
    gate_up_omma_persistent_kernel(const bf16* x, MoeDispatch dispatch, MoeExpertSource source, int top_k,
                                   const std::uint8_t* const* job_records,
                                   const std::int32_t* narrow_jobs,
                                   const std::int32_t* pass_narrow, int pass_index,
                                   const std::uint8_t* x_plane, std::uint8_t* __restrict__ h_base) {
    using S = OmmaShape<G>;
    __shared__ GateUpOmmaShared<G> sm;
    if (threadIdx.x == 0) {
        cta_mbarrier_init(&sm.mbar[0], 1);
        cta_mbarrier_init(&sm.mbar[1], 1);
    }
    cta_mbarrier_fence_init();
    __syncthreads();
    std::uint32_t mbar_phase = 0U;
    pdl::wait_for_dependencies();
    const int n_begin     = pass_narrow[pass_index];
    const int n_count     = pass_narrow[pass_index + 1] - n_begin;
    const int total_tasks = n_count * S::kGateUpSlices;
    for (int task = static_cast<int>(blockIdx.x); task < total_tasks; task += static_cast<int>(gridDim.x)) {
        const int job              = narrow_jobs[n_begin + task / S::kGateUpSlices];
        const int slice            = task % S::kGateUpSlices;
        const int expert           = dispatch.jobs[job];
        const std::uint8_t* record = pass_record(PassPhase::All, source, expert, job_records, job);
        if (record == nullptr) { continue; }
        const moe::ExpertScales scales = source.scales[expert];
        const int first = dispatch.offsets[expert], count = dispatch.offsets[expert + 1] - first;
        if (count <= 0 || moe::wide_route(count, scales)) { continue; }
        gate_up_omma_slice<G, true>(sm, x, dispatch, top_k, record, scales, first, count, slice, x_plane, h_base,
                                    mbar_phase);
        fence_reads_before_async_refill();
        __syncthreads();
    }
    pdl::trigger_dependents();
}

template <class G>
__global__ void __launch_bounds__(OmmaShape<G>::kThreads, 7)
    down_omma_persistent_kernel(MoeDispatch dispatch, MoeExpertSource source,
                                const std::uint8_t* const* job_records,
                                const std::int32_t* narrow_jobs,
                                const std::int32_t* pass_narrow, int pass_index,
                                const std::uint8_t* h_base, bf16* __restrict__ outputs) {
    using S = OmmaShape<G>;
    __shared__ DownOmmaShared<G> sm;
    if (threadIdx.x == 0) {
        cta_mbarrier_init(&sm.mbar[0], 1);
        cta_mbarrier_init(&sm.mbar[1], 1);
    }
    cta_mbarrier_fence_init();
    __syncthreads();
    std::uint32_t mbar_phase = 0U;

    const int tid         = static_cast<int>(threadIdx.x);
    const int n_begin     = pass_narrow[pass_index];
    const int n_count     = pass_narrow[pass_index + 1] - n_begin;
    const int total_tasks = n_count * S::kDownTiles;
    const int task0       = static_cast<int>(blockIdx.x);

    bool preloaded0            = false;
    const std::uint8_t* rec0   = nullptr;
    int first0 = 0, count0 = 0, tile0 = 0;
    moe::ExpertScales scales0{};
    if (task0 < total_tasks) {
        const int job = narrow_jobs[n_begin + task0 / S::kDownTiles];
        tile0         = task0 % S::kDownTiles;
        const int exp = dispatch.jobs[job];
        rec0          = pass_record(PassPhase::All, source, exp, job_records, job);
        if (rec0 != nullptr) {
            scales0 = source.scales[exp];
            first0  = dispatch.offsets[exp];
            count0  = dispatch.offsets[exp + 1] - first0;
            if (count0 <= 0 || moe::wide_route(count0, scales0)) {
                rec0 = nullptr;
            } else {
                if (tid == 0) {
                    const std::uint8_t* rg_base =
                        rec0 + G::kGateUpBytes + static_cast<std::size_t>(4 * tile0) * S::kDownRgBytes;
                    const std::uint64_t weight_policy = l2_evict_first_policy();
                    cta_mbarrier_arrive_expect_tx(&sm.mbar[0], static_cast<std::uint32_t>(4 * S::kDownWarpStageB));
#pragma unroll
                    for (int w = 0; w < S::kWarps; ++w) {
                        const std::uint8_t* src = rg_base + static_cast<std::size_t>(w) * S::kDownRgBytes;
                        omma_bulk_load_hinted(sm.w[0][w], src, S::kDownWarpStageB, &sm.mbar[0], weight_policy);
                    }
                }
                preloaded0 = true;
            }
        }
    }
    pdl::wait_for_dependencies();
    if (task0 < total_tasks && rec0 != nullptr) {
        down_omma_tile_compute<G>(sm, dispatch, rec0, scales0, first0, count0, tile0, h_base, outputs, preloaded0,
                                  mbar_phase);
        fence_reads_before_async_refill();
        __syncthreads();
    }
    for (int task = task0 + static_cast<int>(gridDim.x); task < total_tasks; task += static_cast<int>(gridDim.x)) {
        const int job              = narrow_jobs[n_begin + task / S::kDownTiles];
        const int tile             = task % S::kDownTiles;
        const int expert           = dispatch.jobs[job];
        const std::uint8_t* record = pass_record(PassPhase::All, source, expert, job_records, job);
        if (record == nullptr) { continue; }
        const moe::ExpertScales scales = source.scales[expert];
        const int first = dispatch.offsets[expert], count = dispatch.offsets[expert + 1] - first;
        if (count <= 0 || moe::wide_route(count, scales)) { continue; }
        down_omma_tile_compute<G>(sm, dispatch, record, scales, first, count, tile, h_base, outputs, false, mbar_phase);
        fence_reads_before_async_refill();
        __syncthreads();
    }
}

int persistent_omma_ctas() {
    static const int ctas = [] {
        int dev = 0, sms = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
        return sms * 7;
    }();
    return ctas;
}

void launch_expert_pass_a4_omma(const bf16* x, const MoeDispatch& dispatch, const MoeExpertSource& source, int top_k,
                                const std::uint8_t* const* job_records, const std::int32_t* flags, int base, int jobs,
                                canon::A4Block* h_blocks, const moe::wide::ExpertsWorkspace& layout, int pass_index,
                                bool use_xplane, bf16* outputs, cudaStream_t stream, PassPhase phase, PassPart part) {
    using G                    = moe::FlashNextExperts;
    using S                    = OmmaShape<G>;
    auto* h_base               = reinterpret_cast<std::uint8_t*>(h_blocks);
    const bool unstaged_direct = source.staging_slots == 0 && source.fetch.request == nullptr && phase == PassPhase::All;
    if (use_xplane) {
        const int persistent = persistent_omma_ctas();
        if (part != PassPart::Down) {
            CUDA_CHECK(pdl::launch_with(unstaged_direct ? pdl::Dependency::Programmatic : pdl::Dependency::Serialized,
                                        {dim3(persistent), dim3(S::kThreads), 0, stream},
                                        gate_up_omma_persistent_kernel<G>, x, dispatch, source, top_k, job_records,
                                        layout.narrow_jobs, layout.pass_narrow, pass_index, layout.x_plane, h_base));
            check_launch("A4 OMMA gate/up xplane");
        }
        if (part == PassPart::GateUp) { return; }
        const auto dep = part == PassPart::Down ? pdl::Dependency::Serialized : pdl::Dependency::Programmatic;
        CUDA_CHECK(pdl::launch_with(dep, {dim3(persistent), dim3(S::kThreads), 0, stream},
                                    down_omma_persistent_kernel<G>, dispatch, source, job_records, layout.narrow_jobs,
                                    layout.pass_narrow, pass_index, h_base, outputs));
        check_launch("A4 OMMA down xplane");
        return;
    }
    if (part != PassPart::Down) {
        const auto gu_dep = unstaged_direct ? pdl::Dependency::Programmatic : pdl::Dependency::Serialized;
        CUDA_CHECK(pdl::launch_with(gu_dep, {dim3(S::kGateUpSlices, jobs), dim3(S::kThreads), 0, stream},
                                    gate_up_omma_kernel<G>, x, dispatch, source, top_k, job_records, flags, base,
                                    h_base, phase));
        check_launch("A4 OMMA gate/up");
    }
    if (part == PassPart::GateUp) { return; }
    const auto dep = part == PassPart::Down ? pdl::Dependency::Serialized : pdl::Dependency::Programmatic;
    CUDA_CHECK(pdl::launch_with(dep, {dim3(S::kDownTiles, jobs), dim3(S::kThreads), 0, stream}, down_omma_kernel<G>,
                                dispatch, source, job_records, flags, base, h_base, outputs, phase));
    check_launch("A4 OMMA down");
}
