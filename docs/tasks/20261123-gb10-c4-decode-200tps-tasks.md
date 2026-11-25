# GB10 (`sm_121a`) `c=4` Decode 200+ t/s Optimization Tasks (2026-11-23)

- **Date:** 2026-11-23
- **Target Architecture:** NVIDIA DGX Spark GB10 (`sm_121a`, CC `12.1`, 48 SMs, 99 KiB / `101,376 B` max dynamic SMEM/SM, 24 MiB L2, 128 GiB / `121.7 GiB` unified LPDDR5X UMA, `273 GB/s` peak / `~246 GB/s` sustained read)
- **Profile & Baseline Source:** [`profiles/nsys/benchy_c4_v2_nodes.sqlite`](../../profiles/nsys/benchy_c4_v2_nodes.sqlite) (Cluster 1 pure decode: `2,526.6 ms` wall across `38` rounds = **`66.5 ms/round`**); [`README.md`](../../README.md) (`f65e1a776`) baseline `c=4` `tg128`: **`167.3 / 166.0 / 161.1 t/s`** at `d0 / d4096 / d8192`
- **Primary Model Scope:** `Qwen3.8-Flash-Next` (`HC4-W4Dense`, `int8` KV default) on `infernix-serve`
- **Companion Roadmaps:** [`20261123-gb10-decode-concurrency-perf-tasks.md`](20261123-gb10-decode-concurrency-perf-tasks.md), [`20261122-gb10-hardware-utilization-tasks.md`](20261122-gb10-hardware-utilization-tasks.md)

---

## 1. Roofline Budget & Pure-Decode Profile Breakdown

### 1.1 Arithmetic Target for Sustained `200+ t/s` at `c=4`

Generating `4 × 128 = 512` tokens at **`≥ 200 t/s`** requires total pure-decode wall time **`≤ 2,560 ms`**:

- At typical `llama-benchy` MTP acceptance (`~12.0 tok/round` across 4 lanes → `~43` rounds): `43 × 66.5 ms = 2,860 ms` (`~179 t/s` peak; `~167–171 t/s` end-to-end with `d4096`/`d8192` KV + host overhead). Note that the `2,526.6 ms` Cluster 1 trace only reaches `~202 t/s` when a `38`-round seed hits unusually high MTP acceptance (`~13.5 tok/round`).
- Sustaining **`200+ t/s`** across normal acceptance variance requires **at least one** (ideally both):
  1. Cutting round latency from **`66.5 ms/round` → `≤ 59.5 ms/round`** (`-7.0 ms/round`, `-10.5%` at fixed MTP acceptance), without regressing `c=1` or prefill.
  2. Lifting accepted tokens per round by **`+12–15%`** (or removing multi-step MTP draft serialization) via **DFlash2 block drafting** and **compact-logit verification trees**, pushing sustained `c=4` decode toward **`215–240+ t/s`**.

### 1.2 Pure-Decode Kernel Profile (`benchy_c4_v2_nodes.sqlite`, Cluster 1, 38 Rounds)

Wall time: **`2,526.6 ms`** | Kernel sum: **`2,583.6 ms`** (minor stream overlap with shared-expert side stream) | Total idle gaps `> 100 us`: **`32.45 ms`** (`45` gaps):

| Category / Kernel | Calls (`n`) | Total Time | % Kernel Sum | `ms / round` | Typical Launch Shape / Notes |
|---|---|---|---|---|---|
| `gate_up_omma_persistent_kernel` | `1,825` | `873.42 ms` | `33.8%` | `22.97 ms` | `grid=(336,1,1) blk=128`; `REG=94`, re-quantizes BF16 `x` `40×`/expert |
| `down_omma_persistent_kernel` | `1,825` | `430.96 ms` | `16.7%` | `11.33 ms` | `grid=(336,1,1) blk=128`; drains pipeline between sequential tasks |
| `nvfp4_a16_sliced_k_mma_kernel` + `sliced_k_pipelined_kernel` | `3,880 + 3,840` | `294.28 + 45.54 ms` | `13.2%` | `8.94 ms` | Dense trunk NVFP4 projections (roofline-near DRAM stream) |
| `hc_down_partials_q4_mma_kernel` + `hc_up_collapse_q4_mma_kernel` (+ `*_v4`) | `3,880 + 3,880` | `154.38 + 84.58 + 23.99 ms` | `10.2%` | `6.92 ms` | `grid=(21,4,1)` / `(160,1,1) blk=128`; 2-slice `16+16` at `T=17..32` |
| `recurrent_fold_kernel` | `40` | `78.55 ms` | `3.0%` | `2.07 ms` | `grid=(192,4,36) blk=32`; `453 MB/rnd` R/W (`~219 GB/s`) |
| `recurrent_record_conv_kernel` | `1,368` | `67.12 ms` | `2.6%` | `1.77 ms` | `grid=(48,4,8) blk=32`; `226.5 MB/rnd` state read (`~128 GB/s`) |
| `screen_kernel` (MTP draft `grid=96` + verify `grid=48`) | `175` (`137 + 38`) | `93.32 ms` (`60.09 + 33.23`) | `3.6%` | `2.45 ms` | `~4.6` calls/round; `grid=48` double-passes at `T > 24` |
| `grouped_kernel` (`wide_expert`) | `3,650` | `78.43 ms` | `3.0%` | `2.06 ms` | `grid=(20,1,1)` / `(40,1,1) blk=288`; `9..24`-col experts starve 48 SMs |
| MTP layer SIMT MoE + QKV (`gate_up_kernel` + `down_kernel` + `q8_a16_stream_kernel`) | `134` each | `36.95 + 19.08 + 20.77 ms` | `3.0%` | `2.02 ms` | `~3.53` serial MTP steps/round (`resident_moe_experts` + MTP attention) |
| Router SIMT (`projection_kernel` + `projection_tall_kernel`) | `1,142 + 912` | `30.58 + 22.11 ms` | `2.0%` | `1.39 ms` | `63 MB/rnd` weights → `0.26 ms` DRAM floor (`5.3×` over roofline) |
| Non-persistent OMMA fallback (`gate_up_omma_kernel` + `down_omma_kernel`) | `96 + 96` | `29.25 + 13.78 ms` | `1.7%` | `1.13 ms` | Fires when `use_xplane == false` |
| Host-to-GPU idle gaps `> 100 us` + `wait_pinned_word_kernel` | `45` gaps / `40` waits | `32.45 + 18.84 ms` | — | `1.35 ms` | `cudaGraphLaunch_v10000` (`841–1,358 us` on CPU) after round sync |

---

## 2. Impact-Ranked Task Summary (`T1`–`T11`)

> **Execution Discipline:** Implement step by step (`T1` → `T11`). Validate each task individually via **Gate A** (oracle tests, CUDA stack-spill ratchet, SASS/register check, diff hygiene) and **Gate B** (fast Op microbenchmark and/or `~20–40 s` native `infernix_bench -pg 512,128 -r 2` smoke run — see Section 6). **Do NOT run full `llama-benchy` during intermediate tasks**; run the full `llama-benchy` matrix **once in Gate C** after all selected tasks have landed.

| Tier | Priority | Task ID | Goal | Primary Files / Symbols | Estimated `c=4` Impact | `c=1` / Prefill Risk | Status |
|---|---|---|---|---|---|---|---|
| **Tier 1** | **P0** | **T1** | Cap `kHcMixFusedMaxColumnsQ4 = 16` to eliminate `T=17..32` 2-slice Q4 MMA cliff | [`hyper_connection_mix_fused.h`](../../src/ops/hyper_connection/hyper_connection_mix_fused.h#L22), [`hyper_connection.cu`](../../src/ops/hyper_connection/hyper_connection.cu#L743-L785) | **`-1.1 to -1.3 ms/round`** (up to `-2.5 ms` at `T=28..32`) | None (`T<=16` & prefill `T>32` unchanged) | Done (Kept: `T=20/24/32` 42.42–42.86 us, `c=1` `+0.20%`) |
| **Tier 1** | **P0** | **T2** | Split verify CUDA graph into small head (layers `0..2`) + tail to overlap `cudaGraphLaunch` CPU cost | [`program_impl.h`](../../src/models/qwen4_exp/program/program_impl.h#L5226-L5240), [`decode_graph.cpp`](../../src/core/decode_graph.cpp#L234-L239) | **`-0.85 to -1.1 ms/round`** | Beneficial or neutral | Done (Kept: `upload->Head` 8.5–31.5 us, `c=1` `+1.0%`) |
| **Tier 1** | **P0** | **T3** | Add decode-width (`T=9..32`) Tensor-Core schedule to `projection_fp32_router_mma` and lower `kRouterMmaMinColumns` | [`fast_path_flags.h`](../../src/models/qwen4_exp/execution/fast_path_flags.h#L49), [`projection_fp32.cu`](../../src/ops/projection_fp32/projection_fp32.cu#L876-L926) | **`-0.8 to -1.0 ms/round`** | None (`T<=8` stays SIMT) | Done (Kept: `kRouterMmaMinColumns=9`, `N=513` -45% to -55%, `c=1` `+0.6%`) |
| **Tier 1** | **P1** | **T4** | Port `recurrent_record_conv_kernel` to 256-thread 16-lane `FoldWindow` subgroup layout | [`recurrent.cu`](../../src/ops/linear_attention/gated_delta_net/recurrent.cu#L270-L332), [`recurrent.cuh`](../../src/ops/linear_attention/gated_delta_net/recurrent.cuh#L1007) | **`-0.7 ms/round`** | Beneficial or neutral (verify `T=1..8` A/B) | Reverted (Rejected: v1 `-1.9%` at `B=4` `REG=90`, v2 `+13%` `REG=109`) |
| **Tier 1** | **P1** | **T5** | Single-pass `C=32` + 4-stage deep ring in `screen_kernel` for `T=17..32` | [`screened_head.cu`](../../src/ops/screened_head/screened_head.cu#L60-L105) | **`-0.85 ms/round`** at `T=25..32` (`-0.15 ms` at `T=17..24`) | None (`T<=16` keeps existing paths) | Done (Kept `T>24`: `T=17..24` `<3,true>` ~880 us, `T=25..32` `gmem_x` 1-pass ~1.22 ms) |
| **Tier 2** | **P1** | **T6** | Load pre-quantized `layout.x_plane` via `cp.async` inside `gate_up_omma_slice` instead of `40×` BF16→FP4 re-quant | [`narrow_a4_omma.cuh`](../../src/ops/offloaded_sparse_moe/cuda/narrow_a4_omma.cuh#L219-L245), [`wide_expert.cu`](../../src/ops/offloaded_sparse_moe/cuda/wide_expert.cu#L261-L296) | **`-0.8 to -1.5 ms/round`** | Neutral/faster if `REG <= 94` & 2-stage preserved | Done (Already in HEAD 098b2f4a3: omma_stage_xplane_chunk + UseXPlane) |
| **Tier 2** | **P1** | **T7** | Inter-task `stage 0` weight prefetch across persistent epilogues in `gate_up/down_omma_persistent_kernel` | [`narrow_a4_omma.cuh`](../../src/ops/offloaded_sparse_moe/cuda/narrow_a4_omma.cuh#L536-L644) | **`-0.8 to -1.3 ms/round`** | Neutral/faster at `c=1`; strict `mbarrier` phase proof | Reverted (Rejected: `c=4` 92.83–93.06 vs 93.83–93.94 baseline with `SHARED:12560`, 92.58–92.82 with `13840`; `c=1` 79.89 vs 80.11–80.17) |
| **Tier 2** | **P1** | **T8** | Dedicated `9..24`-col small-wide OMMA kernel (replacing `grouped_kernel`) + close `use_xplane == false` fallback | [`wide_expert.cuh`](../../src/ops/offloaded_sparse_moe/cuda/wide_expert.cuh#L260), [`moe_layer.cu`](../../src/ops/offloaded_sparse_moe/cuda/moe_layer.cu#L2589-L2599) | **`-1.0 to -1.5 ms/round`** (`grouped`) + **`-0.5 to -1.0 ms/round`** (fallback) | None if `1..8`-col persistent kernel untouched | Reverted (Rejected: Part A only `15.4 ms` in infernix_bench / `119.4 ms` in benchy & `c=4` `92.84`; Part B `T=2..8` `xplane` `+3%..+12%` slower on `decode_bench`, higher MoE GPU ms/call in nsys, `-2.5%` `c=1` via 1-ULP `spec acc` shift) |
| **Tier 3** | **P1** | **T9** | Switch `hc4` to DFlash2 block drafting + tune `kDFlashMultiRowMaxDrafts` (`6..7`) *(or port MTP `resident_moe_experts` to A4 OMMA)* | [`run_server.sh`](../../tools/run_server.sh#L50), [`program_impl.h`](../../src/models/qwen4_exp/program/program_impl.h#L7930-L7960), [`dflash2_program.cpp`](../../src/models/qwen4_exp/program/dflash2_program.cpp#L61-L115) | **`-3.0 to -4.0 ms/round`** draft + higher tok/round *(MTP fallback: `-0.8 to -1.0 ms`)* | Verify `c=1` DFlash2 accept on `hc4` | Rejected DFlash2 (`c=4` `89.05 vs 93.94`); MTP fallback Deferred (Bandwidth-Bound: `resident_moe_bench` `q4` `T=4` already `228 GB/s` / `457.7 us`) |
| **Tier 3** | **P2** | **T10** | Enable compact screened logits for verification trees (`D4` continuation) | [`program_impl.h`](../../src/models/qwen4_exp/program/program_impl.h#L5485-L5495), [`speculative_round.cuh`](../../src/ops/kernel/speculative_round.cuh#L1238-L1290) | **`+5–10%` tok/round** at `c=2..4` | Verify `c=1` tree regression vs greedy/compact | Done (Gate A 5/5 + `STACK:0 LOCAL:0 REG:79`; sampled `c=2` `73.81 ± 2.52` / `spec acc 0.6531` vs `NO_UNIVER_TREES=1` `76.77 ± 3.97` / `0.6286`; `c=4` `89.67 ± 4.98`) |
| **Tier 3** | **P2** | **T11** | Long-context & state BW: `KV_DTYPE=k8v4` A/B (`d4096`/`d8192`) + optional FP8 GDN state fast path | [`run_server.sh`](../../tools/run_server.sh#L52), [`recurrent.cu`](../../src/ops/linear_attention/gated_delta_net/recurrent.cu#L25-L33) | **`k8v4`:** halves V-KV at long ctx; **FP8 GDN:** `-2.0 ms/round` | Gate via [`model-quality.md`](../maintainer/model-quality.md) | Evaluated `k8v4` (`int8` kept default; `-pg 4096,128 -r 2` `c=4` `79.66±0.20` w/ `ngram 654` vs `int8` `75.92±0.74` w/ `ngram 522`; `c=1` `144.43±39.24` vs `97.47±0.03`; FP8 GDN deferred) |

---

## 3. Detailed Source-Level Task Specifications (`T1`–`T11`)

### T1 (`P0`): Cap `kHcMixFusedMaxColumnsQ4 = 16` for `T = 17..32`
- **Files & Symbols:** [`src/ops/hyper_connection/hyper_connection_mix_fused.h#L22`](../../src/ops/hyper_connection/hyper_connection_mix_fused.h#L22) (`kHcMixFusedMaxColumnsQ4`), [`src/ops/hyper_connection/hyper_connection.cu#L743-L785`](../../src/ops/hyper_connection/hyper_connection.cu#L743-L785) (`hyper_connection_mix`, `hyper_connection_inject_mix`).
- **Measured Evidence:**
  - `hc_down_partials_q4_mma_kernel` (`154.38 ms`, `3,880` calls) + `hc_up_collapse_q4_mma_kernel` (`84.58 ms`, `3,880` calls) cost **`6.29 ms/round`** in Cluster 1.
  - Although [`20261123-gb10-decode-concurrency-perf-tasks.md`](20261123-gb10-decode-concurrency-perf-tasks.md#L52-L57) §3 D3 recorded reverting the `TCap=32` / D3 experiment, `kHcMixFusedMaxColumnsQ4` remains **`32`** in tree (`16+16` slicing for `T = 17..32`).
  - Microbenchmarks recorded in D3 (`infernix_hyper_connection_bench --codec q4 --layers 48`) measured `HEAD`'s 2-slice fused Q4 MMA at **`51.36 / 53.26 / 67.23 us`** (and `52.83 / 54.75 / 68.62 us` in D3 table summary) for `T = 20 / 24 / 32`, whereas the unfused composed route (`hyper_connection_norm` + `hc_mix_from_normalized`) measured **`41.44 / 41.92 / 41.84 us`**.
- **Concrete Code Change:** Lower `kHcMixFusedMaxColumnsQ4` from `32` to `16` so `T = 1..16` retains the fused Q4 MMA path (`37.62 us` at `T=16`), while `T = 17..32` routes directly to the composed `hc_mix_from_normalized` path (`hyper_connection_inject_norm` / `hyper_connection_norm` + `linear`/`hc_up_collapse`).
- **Acceptance Criteria (Gate A + Quick Gate B):**
  - *Code Quality (Gate A):* `infernix_hyper_connection_test` passes; `infernix_cuda_stack_check` clean; `git diff --check` clean.
  - *Quick Benchmark (Gate B):* `./build-native/bench/infernix_hyper_connection_bench --columns 1,4,8,16,20,24,32 --codec q4 --layers 48` — require `T = 20, 24, 32 <= 43.5 us/call` (`>= 18%` faster than `51.4–68.6 us` baseline) and `T <= 16` within `±1.5%` (target `c=4` savings: `-1.1 to -1.3 ms/round`, up to `-2.5 ms/round` at `T=28..32`).

### T2 (`P0`): Split `verify_graph_body` to Overlap the `cudaGraphLaunch` CPU Bubble
- **Files & Symbols:** [`src/models/qwen4_exp/program/program_impl.h#L5226-L5240`](../../src/models/qwen4_exp/program/program_impl.h#L5226-L5240) (`verify_graph`, `verify_graph_body`), [`#L5307-L5312`](../../src/models/qwen4_exp/program/program_impl.h#L5307-L5312) (`replay`), [`#L5786-L5796`](../../src/models/qwen4_exp/program/program_impl.h#L5786-L5796), [`src/core/decode_graph.cpp#L234-L239`](../../src/core/decode_graph.cpp#L234-L239), [`src/models/qwen4_exp/program/graph_set.cpp#L142-L150`](../../src/models/qwen4_exp/program/graph_set.cpp#L142-L150).
- **Measured Evidence:** Cluster 1 exhibits `45` inter-kernel idle gaps `> 100 us` totaling **`32.45 ms`** (`0.85 ms/round`), dominated by `750–1,096 us` gaps between `upload_pinned_kernel` and `embed_gather_dense_kernel` where `CUPTI_ACTIVITY_KIND_RUNTIME` attributes **`841–1,358 us`** of host time to `cudaGraphLaunch_v10000` for the monolithic 48-layer verify graph immediately after host synchronization drains the stream.
- **Concrete Code Change:** Partition `verify_graph_body` (and corresponding `VerifyGraphPair` / startup graph capture entries in `graph_set.cpp`) into:
  1. A **head graph** covering embedding/initial prep + layers `0..2` (`~35 us` host `cudaGraphLaunch` overhead, `~3 ms` GPU execution).
  2. A **tail graph** covering layers `3..47` + final norm/LM head.
  Launching the small head graph wakes the GPU immediately after `upload_pinned`, letting the CPU pay the `~800 us` `cudaGraphLaunch` cost of the tail graph while the GPU executes layers `0..2`. Preserve exact PDL / side-stream (`shared_stream`) barrier isolation across the split boundary (complete pending side-stream events inside head before returning to tail).
- **Acceptance Criteria (Gate A + Quick Gate B):**
  - *Code Quality (Gate A):* `infernix_qwen4_exp_speculative_test` and `infernix_qwen4_exp_startup_capture_real_test` (when artifact env set) pass; `--reproducible` tokens match monolithic verify graph bit-for-bit; `infernix_cuda_stack_check` clean.
  - *Quick Benchmark (Gate B):* Quick `infernix_bench` `c=4` `-pg 512,128 -r 2 --warmup 1` + one `--profile-measured` `nsys` spot capture confirming `upload_pinned_kernel -> embed_gather_dense_kernel` GPU idle gap drops from `~850 us` to `< 50 us`, with net `c=4` speedup and `c=1` `-pg 512,128 -r 2` within `>= -1.5%`.

### T3 (`P0`): Enable Tensor-Core `projection_fp32_router_mma` at Decode Widths (`T = 9..32`)
- **Files & Symbols:** [`src/models/qwen4_exp/execution/fast_path_flags.h#L49`](../../src/models/qwen4_exp/execution/fast_path_flags.h#L49) (`kRouterMmaMinColumns = 256`), [`src/models/qwen4_exp/execution/forward.cpp#L1365-L1397`](../../src/models/qwen4_exp/execution/forward.cpp#L1365-L1397), [`src/ops/projection_fp32/projection_fp32.cu#L128`](../../src/ops/projection_fp32/projection_fp32.cu#L128) (`projection_kernel`), [`#L700`](../../src/ops/projection_fp32/projection_fp32.cu#L700) (`projection_tall_kernel`), [`#L876-L926`](../../src/ops/projection_fp32/projection_fp32.cu#L876-L926) (`projection_fp32_router_mma`).
- **Measured Evidence:** SIMT `projection_kernel` (`30.58 ms`, `n=1,142`) + `projection_tall_kernel` (`22.11 ms`, `n=912`) = **`52.69 ms` (`1.39 ms/round`)** across 48 MoE `[257, 2560]` BF16 routers (`~63 MB/round` weights = **`0.26 ms/round`** DRAM floor, currently running at `~19%` of bandwidth ceiling).
- **Concrete Code Change:** Implement a decode-width (`T = 9..32`) HMMA.16816 BF16×BF16→FP32 schedule for `projection_fp32_router_mma` (exploiting GB10's 1:1 FP32 accumulation parity, with single-row `shared_score` tail fused or kept in `projection_tail_row_kernel`), relax the `columns >= 64` guard inside `projection_fp32_router_mma`, and lower `kRouterMmaMinColumns` from `256` to the measured crossover (`~9` or `16`). Leave `T = 1..8` (`c=1` and narrow single-row) on `projection_kernel`.
- **Acceptance Criteria (Gate A + Quick Gate B):**
  - *Code Quality (Gate A):* `infernix_projection_fp32_test` passes naive FP32 oracle per [`op-development.md`](../maintainer/op-development.md) across `T = 9, 12, 16, 20, 24, 32`, `rows = {256, 257}`, `K = 2560`; `infernix_cuda_stack_check` clean (`STACK=0, LOCAL=0`); `ckt static --sass` confirms HMMA instructions and zero local spills.
  - *Quick Benchmark (Gate B):* Quick `infernix_bench` `c=4` `-pg 512,128 -r 2 --warmup 1` (with `--profile-measured` `nsys` spot check confirming router kernel total drops `> 60%`, saving `~0.8–1.0 ms/round`); `c=1` (`T <= 8`) within `>= -1.0%`.

### T4 (`P1`): Port `recurrent_record_conv_kernel` to the 256-Thread 16-Lane `FoldWindow` Layout
- **Files & Symbols:** [`src/ops/linear_attention/gated_delta_net/recurrent.cu#L270-L332`](../../src/ops/linear_attention/gated_delta_net/recurrent.cu#L270-L332) (`launch_recurrent_record_conv`), [`src/ops/linear_attention/gated_delta_net/recurrent.cuh#L1007`](../../src/ops/linear_attention/gated_delta_net/recurrent.cuh#L1007) (`recurrent_record_conv_kernel`), [`#L1290-L1418`](../../src/ops/linear_attention/gated_delta_net/recurrent.cuh#L1290-L1418) (`recurrent_fold_kernel`).
- **Measured Evidence:** In Cluster 1, `recurrent_record_conv_kernel` (`grid=(48,4,8) blk=32`, 128 threads total) reads `226.5 MB/round` of FP16 GDN state across 36 layers in **`1.77 ms/round` (`128 GB/s`, 52% of sustained DRAM BW)**, whereas `recurrent_fold_kernel` (`FoldWindow`, `grid=(192,4,36) blk=32` = 256 threads total, `kFoldDv = 32`, 4 tiles/head, `cp.async` staging, 16-lane subgroup) reads+writes `453 MB/round` in `2.07 ms/round` (**`219 GB/s`**).
- **Concrete Code Change:** Port `recurrent_record_conv_kernel` (for aligned widths `<= 16`) to the same 256-thread 16-lane subgroup tiling (`kFoldDv = 32`, halving CTA count per head×row, `cp.async` staging of QKV/conv records) while preserving the exact 32-lane warp FP32 reduction tree `((lo4) + (hi4) then xor 8,4,2,1)` for bit-exact equivalence on `--reproducible` / oracle checks.
- **Acceptance Criteria (Gate A + Quick Gate B):**
  - *Code Quality (Gate A):* `infernix_gated_delta_net_test`, `infernix_gated_delta_net_replay_record_test`, `infernix_gated_delta_net_replay_record_conv_test`, `infernix_gdn_replay_records_test`, and `infernix_gdn_replay_fold_test` pass (bit-exact FP32 and FP16 state); `infernix_cuda_stack_check` clean; `ckt static --sass` confirms zero spills.
  - *Quick Benchmark (Gate B):* `./build-native/bench/infernix_gdn_replay_bench --profile 35b --component recurrent --width 4 --batch 4` (plus `--width 1..8 --batch 1,2,4`) and `./build-native/bench/infernix_gated_delta_net_bench` — require `recurrent_record_conv >= 200 GB/s` (`>= 30%` latency reduction at `batch=4`, `~-0.7 ms/round`) and `batch=1` within `>= -1.5%`.

### T5 (`P1`): Single-Pass `C = 32` + 4-Stage Deep Ring in `screen_kernel` (`T = 17..32`)
- **Files & Symbols:** [`src/ops/screened_head/screened_head.cu#L60-L105`](../../src/ops/screened_head/screened_head.cu#L60-L105) (`kRingStages`, `screen_shared_bytes`), [`#L1019-L1048`](../../src/ops/screened_head/screened_head.cu#L1019-L1048) (`run_screen`), [`include/infernix/ops/screened_head.h`](../../include/infernix/ops/screened_head.h).
- **Measured Evidence:** At `T > 24` (`c=4` with `W=7..8` under DFlash2 or stitched n-gram), `run_screen` loops `col += 24` and launches `screen_kernel` **twice back-to-back** (seen in Cluster 0: `892 us + 851 us = 1,743 us`), reading all `178.8 MB` of 2-bit LM-head screen codes+scales twice (`178.8 MB / 246 GB/s = 727 us` DRAM floor). At `T = 20..24`, `Tiles=3, NarrowRing=true` uses only 2 ring stages and 1 CTA/SM (`874 us` measured).
- **Concrete Code Change:** Keep only the 4-stage 2-bit weight+scale ring (`4 × kStageBytes ≈ 47 KiB`) in SMEM and stream/cache the quantized int8 activations `xq` (`24..32 × 2,560 B = 60–80 KiB`) in L1/L2 (pinned via hardware access-policy window already used in [`screened_head.cu#L963-L1007`](../../src/ops/screened_head/screened_head.cu#L963-L1007)) or a small per-chunk activation buffer, enabling a **single-pass `Tiles = 4` (`C = 32`), `NarrowRing = false` (4-stage deep ring)** kernel for `T = 17..32`. Keep `T = 1..16` unchanged.
- **Acceptance Criteria (Gate A + Quick Gate B):**
  - *Code Quality (Gate A):* `infernix_screened_head_test` passes (compact + logits at `T = 20, 24, 28, 32`); `infernix_cuda_stack_check` clean; `ckt static --sass` confirms zero spills and dynamic SMEM `<= 101,376 B`.
  - *Quick Benchmark (Gate B):* Quick `infernix_bench` `c=4` `-pg 512,128 -r 2 --warmup 1` + one `--profile-measured` `nsys` spot capture verifying a single `screen_kernel` launch for `T = 25..32` at `~0.85–0.90 ms` (vs `~1.74 ms`) and `-0.10 to -0.15 ms` per call at `T = 17..24`; `c=1` (`T <= 8`) untouched.

### T6 (`P1`): Read Pre-Quantized `layout.x_plane` in `gate_up_omma_slice` (Eliminate `40×` Re-Quantization)
- **Files & Symbols:** [`src/ops/offloaded_sparse_moe/cuda/narrow_a4_omma.cuh#L118-L169`](../../src/ops/offloaded_sparse_moe/cuda/narrow_a4_omma.cuh#L118-L169), [`#L176-L260`](../../src/ops/offloaded_sparse_moe/cuda/narrow_a4_omma.cuh#L176-L260) (`gate_up_omma_slice`), [`src/ops/offloaded_sparse_moe/cuda/wide_expert.cu#L261-L296`](../../src/ops/offloaded_sparse_moe/cuda/wide_expert.cu#L261-L296) (`quantize_token_small_kernel`).
- **Measured Evidence:** `quantize_token_small_kernel` already runs at launch of each MoE layer (`7.92 ms` in Cluster 1) and populates `layout.x_plane`. Yet `gate_up_omma_slice` (`873.42 ms` in `gate_up_omma_persistent_kernel`, `33.8%` of decode) calls `omma_quantize_x_chunk` / `omma_quantize_x_chunk_split` inside **every one of the 40 slice CTAs (`kGateUpSlices = 40`) per expert** — repeating BF16 loads, `amax` reductions, and FP4 quantization ALU `40×` per expert (`~86,000×/round`).
- **Concrete Code Change:** Store OMMA-packed FP4 codes and packed E4M3 scales into `layout.x_plane` (or a dedicated narrow slice plane populated by `quantize_token_small_kernel`) and load `sm.pipe.x` in `gate_up_omma_slice` via 128-bit `cp.async` instead of recomputing FP4 quantization. **Hard constraint:** strictly retain the 2-stage pipeline and `REG <= 94` (`4 blk/SM`, `__launch_bounds__(128, 5)`).
- **Acceptance Criteria (Gate A + Quick Gate B):**
  - *Code Quality (Gate A):* `infernix_offloaded_moe_cuda_test`, `infernix_offloaded_moe_layer_test`, `infernix_offloaded_moe_wide_test`, and `infernix_qwen4_exp_expert_activations_test` pass; `infernix_cuda_stack_check` clean; `ckt static --sass` confirms `gate_up_omma_persistent_kernel` keeps `REG <= 94`, `STACK = 0`, `LOCAL = 0`, and 2-stage pipeline intact.
  - *Quick Benchmark (Gate B):* `./build-native/bench/infernix_offloaded_moe_decode_bench --repeat 20 --tokens 4 --omma` (`--tokens 1,4,8`) + `./build-native/bench/infernix_offloaded_moe_wide_bench --tokens 4,8,16,20,24 --placement frames --activation a4 --cold` — require `>= 3%` speedup on `gate_up`/total at `T = 16..24` (`~-0.8 to -1.5 ms/round`) and `T = 1, 4, 8` within `>= -1.5%`.

### T7 (`P1`): Inter-Task `stage 0` Weight Prefetch Across Persistent OMMA Epilogues
- **Files & Symbols:** [`src/ops/offloaded_sparse_moe/cuda/narrow_a4_omma.cuh#L536-L644`](../../src/ops/offloaded_sparse_moe/cuda/narrow_a4_omma.cuh#L536-L644) (`gate_up_omma_persistent_kernel`, `down_omma_persistent_kernel`).
- **Measured Evidence:** Each persistent CTA runs `~5` sequential slice/tile tasks across the 48 MoE layers. Currently only `task0` of `down_omma_persistent_kernel` preloads `stage 0` (`preloaded0`); every loop transition `task -> task + gridDim.x` drains the pipeline, runs epilogue SwiGLU/global stores + `__syncthreads()`, and starts the next task's weight fetch cold (`~400–600` cycles LPDDR5X latency exposed per task transition).
- **Concrete Code Change:** Resolve `nxt_task` metadata before the epilogue inside `gate_up_omma_persistent_kernel` and `down_omma_persistent_kernel` and asynchronously issue `nxt_task`'s `stage 0` weight `cp.async.bulk` into the free buffer behind an explicit per-task `mbarrier` phase bit, overlapping DRAM latency with epilogue compute and global stores. Keep `REG <= 94` and `SHARED` unchanged.
- **Acceptance Criteria (Gate A + Quick Gate B):**
  - *Code Quality (Gate A):* `infernix_offloaded_moe_cuda_test` and `infernix_offloaded_moe_layer_test` pass across `count = 1..8` and multi-pass stress; `infernix_cuda_stack_check` clean; `ckt static --sass` confirms `REG <= 94` and zero spills.
  - *Quick Benchmark (Gate B):* `./build-native/bench/infernix_offloaded_moe_decode_bench --repeat 20 --tokens 4 --omma` + `3` consecutive repetitions of quick `infernix_bench` (`--concurrency 4 --rows distinct -pg 512,128 -r 3 --warmup 1`) confirming net `c=4` gain (`~-0.8 to -1.3 ms/round`), `c=1` within `>= -1.5%`, and **zero `mbarrier` phase hangs/deadlocks**.

### T8 (`P1`): Dedicated Small-Wide OMMA Kernel (`9..24` Cols) + Close Non-Persistent OMMA Fallback
- **Files & Symbols:** [`src/ops/offloaded_sparse_moe/cuda/wide_expert.cu#L37`](../../src/ops/offloaded_sparse_moe/cuda/wide_expert.cu#L37) (`plan_kernel`), [`src/ops/offloaded_sparse_moe/cuda/wide_expert.cuh#L260`](../../src/ops/offloaded_sparse_moe/cuda/wide_expert.cuh#L260) (`grouped_kernel`), [`src/ops/offloaded_sparse_moe/cuda/wide_expert.h#L51`](../../src/ops/offloaded_sparse_moe/cuda/wide_expert.h#L51) (`wide_route`), [`src/ops/offloaded_sparse_moe/cuda/moe_layer.cu#L2589-L2599`](../../src/ops/offloaded_sparse_moe/cuda/moe_layer.cu#L2589-L2599), [`src/ops/offloaded_sparse_moe/cuda/narrow_a4_omma.cuh#L656-L693`](../../src/ops/offloaded_sparse_moe/cuda/narrow_a4_omma.cuh#L656-L693).
- **Measured Evidence:**
  - `grouped_kernel` (`78.43 ms`, `n=3,650`, `2.06 ms/round`) launches with `grid=(20,1,1)` / `(40,1,1)` on 48 SMs (leaving **58% of SMs idle**) for the ~1–3 experts per layer hitting `9..24` columns at `c=4`.
  - Separately, non-persistent `gate_up_omma_kernel` (`29.25 ms`, `n=96`) + `down_omma_kernel` (`13.78 ms`, `n=96`) = **`43.03 ms` (`1.13 ms/round`)** still fires when `use_xplane == false`.
- **Concrete Code Change:**
  1. Leave `gate_up_omma_persistent_kernel` / `down_omma_persistent_kernel` (`count <= 8`) **completely untouched**, and introduce a **separate** `9..24`-column small-wide OMMA kernel (`grid = 40 × wide_small_experts`, fresh local `mbarrier` per chunk/tile, up to `kBlockTokens = 8` × 3 slices/passes or register-accumulated column groups) dispatched only for experts with `9 <= count <= 24`, keeping all 48 SMs active while reading each expert's weights once.
  2. Audit the `use_xplane == false` path at `T = 20..24` in `moe_layer.cu` / `narrow_a4_omma.cuh` and route decode multi-token layers through `xplane` persistent dispatch where eligible.
- **Acceptance Criteria (Gate A + Quick Gate B):**
  - *Code Quality (Gate A):* `infernix_offloaded_moe_wide_test`, `infernix_offloaded_moe_layer_test`, and `infernix_offloaded_moe_cuda_test` pass at `count = 1..24`; `infernix_cuda_stack_check` clean; `ckt static --sass` verifies the `count <= 8` persistent kernel retains unchanged `REG`/spill counts.
  - *Quick Benchmark (Gate B):* `./build-native/bench/infernix_offloaded_moe_wide_bench --tokens 12,16,20,24 --columns-per-expert 12 --placement frames --activation a4 --cold` (plus Zipfian `--tokens 16,20,24`) showing clear win over `grouped_kernel` + `3` consecutive repetitions of quick `infernix_bench` (`--concurrency 4 --rows distinct -pg 512,128 -r 3 --warmup 1`) confirming net `c=4` gain, `c=1` within `>= -1.5%`, and **zero GPU hang** (avoiding full `llama-benchy` until Gate C).

### T9 (`P1`): Switch `hc4` to DFlash2 Block Drafting + Tune `kDFlashMultiRowMaxDrafts` (`6..7`)
- **Files & Symbols:** [`tools/run_server.sh#L50`](../../tools/run_server.sh#L50) (`DRAFT_PATH`), [`#L155`](../../tools/run_server.sh#L155) (`SPEC_BACKEND`), [`src/models/qwen4_exp/program/program_impl.h#L6527-L6590`](../../src/models/qwen4_exp/program/program_impl.h#L6527-L6590), [`#L7930-L7960`](../../src/models/qwen4_exp/program/program_impl.h#L7930-L7960) (`dflash2_rows_ = 4`, `kMultiRowMaxDrafts`), [`src/models/qwen4_exp/program/dflash2_program.cpp#L61-L115`](../../src/models/qwen4_exp/program/dflash2_program.cpp#L61-L115).
  *(MTP fallback path if keeping `SPEC_BACKEND=mtp`: [`src/models/qwen4_exp/execution/forward.cpp#L1633-L1638`](../../src/models/qwen4_exp/execution/forward.cpp#L1633-L1638) → [`src/ops/offloaded_sparse_moe/cuda/narrow_expert.cu#L53`](../../src/ops/offloaded_sparse_moe/cuda/narrow_expert.cu#L53).)*
- **Measured Evidence:** Under `SPEC_BACKEND=mtp`, `3.53` serial MTP steps/round burn **`~4.1–4.5 ms/round`** in Cluster 1: `screen_kernel` (`grid=96`, `60.09 ms` = `1.58 ms/round`), SIMT `resident_moe_experts` `gate_up_kernel` + `down_kernel` (`56.03 ms` = `1.47 ms/round`), `q8_a16_stream_kernel` (`20.77 ms` = `0.55 ms/round`), plus HC/QSA/sampling (`~24 ms` = `0.63 ms/round`). Meanwhile `/opt/etc/models/Qwen3.8-Flash-Next-DFlash2-Head/Qwen3.8-Flash-Next-DFlash2-Infernix.infernix` is already on disk and `dflash2_rows_ = 4` is already implemented in `program_impl.h`.
- **Concrete Code Change:**
  1. Benchmark and switch `hc4` default `SPEC_BACKEND` to `dflash2` in `run_server.sh` (one parallel draft block proposal instead of 3.5–5 serial autoregressive MTP steps + per-step `screen_kernel`).
  2. Decouple DFlash2's `c=3..4` max draft width (`kDFlashMultiRowMaxDrafts`, tuned to `6..7` paired with **T5** so `4 × 7 = 28..32` columns stay single-pass in `screen_kernel`) from MTP's `kMultiRowMaxDrafts = 5`.
  3. *(MTP fallback alternative)*: If MTP remains active for any workload, port `resident_moe_experts` (`gate_up_kernel`/`down_kernel` in `narrow_expert.cu`) to A4 OMMA (`-0.8 to -1.0 ms/round`).
- **Acceptance Criteria (Gate A + Quick Gate B):**
  - *Code Quality (Gate A):* `infernix_qwen4_exp_speculative_test` and `infernix_qwen4_exp_fast_paths_test` pass; if implementing the MTP `resident_moe_experts` fallback, `infernix_resident_moe_test` and `infernix_cuda_stack_check` pass.
  - *Quick Benchmark (Gate B):* Quick `infernix_bench` A/B (`--spec dflash2 --draft-model ...` vs `--spec mtp`) at `--concurrency 4 --rows distinct -pg 512,128 -r 2 --warmup 1` and `--concurrency 1 -pg 512,128 -r 2 --warmup 1` — require `>= +5%` net gain on `c=4 tg128` with `c=1 tg128` within `>= -2%`. *(For MTP fallback: `./build-native/bench/infernix_resident_moe_bench --columns 1,2,4 --codec q4` shows `>= 30%` speedup at `T=4`.)* Defer full `llama-benchy` and `tool-eval-bench` to **Gate C**.

### T10 (`P2`): Compact Screened Logits on Verification Trees (`D4` Continuation)
- **Files & Symbols:** [`src/models/qwen4_exp/program/program_impl.h#L5485-L5495`](../../src/models/qwen4_exp/program/program_impl.h#L5485-L5495) (`!is_tree`), [`#L5804`](../../src/models/qwen4_exp/program/program_impl.h#L5804) (`!tree_round`), [`#L5925`](../../src/models/qwen4_exp/program/program_impl.h#L5925), [`src/models/qwen4_exp/program/graph_set.cpp#L150`](../../src/models/qwen4_exp/program/graph_set.cpp#L150), [`src/ops/kernel/speculative_round.cuh#L1238-L1290`](../../src/ops/kernel/speculative_round.cuh#L1238-L1290) (`speculative_tree_warp_transport`), [`src/ops/launcher/speculative_round.cu#L358`](../../src/ops/launcher/speculative_round.cu#L358), [`src/ops/wrapper/speculative_round.cpp#L347`](../../src/ops/wrapper/speculative_round.cpp#L347), [`include/infernix/ops/speculative_tree.h#L156`](../../include/infernix/ops/speculative_tree.h#L156).
- **Measured Evidence:** Per [`20261123-gb10-decode-concurrency-perf-tasks.md`](20261123-gb10-decode-concurrency-perf-tasks.md#L59-L61) D4, root-fork verification trees recovered `~18.5%` side-branch hits at `c=2`, but `!is_tree` at `program_impl.h:5486` disabled `screened_head_compact`, spilling full `[248320, T]` FP32 logits (`~25–32 MB/round` DRAM write+read) and regressing `c=2` by `-4.1%` to `-5.3%`.
- **Concrete Code Change:** Add compact `(compact_ids, compact_logits, compact_counts)` lookup support to `speculative_accept_sparse_tree` (matching `speculative_accept_compact_drafts`), remove `!is_tree` / `!tree_round` gates in `program_impl.h`, and capture `batch <= 4` tree graphs in `graph_set.cpp` where `batch * W <= kScreenedHeadMaxColumns` (`64`).
- **Acceptance Criteria (Gate A + Quick Gate B):**
  - *Code Quality (Gate A):* `infernix_speculative_tree_test`, `infernix_speculative_tree_fp32_test`, `infernix_speculative_round_test`, and `infernix_qwen4_exp_speculative_test` pass (bit-exact tree transport against full-logit oracle); `infernix_cuda_stack_check` clean.
  - *Quick Benchmark (Gate B):* Quick `infernix_bench` sampled A/B (`--temperature 1.0 --concurrency 2` and `--concurrency 4`, `--rows distinct -pg 512,128 -r 3 --warmup 1`), enabling only where net `tg128` rises (`> +2%`) with zero full-logit fallback.

### T11 (`P2`): Long-Context & GDN State Bandwidth (`k8v4` KV + Optional FP8 GDN State)
- **Files & Symbols:** [`tools/run_server.sh#L52`](../../tools/run_server.sh#L52) (`KV_DTYPE`), [`src/ops/linear_attention/gated_delta_net/recurrent.cu#L25-L33`](../../src/ops/linear_attention/gated_delta_net/recurrent.cu#L25-L33) (`with_state_type`), [`docs/maintainer/model-quality.md`](../maintainer/model-quality.md).
- **Measured Evidence:** `recurrent_fold_kernel` (`78.55 ms`, `453 MB/rnd` R/W) + `recurrent_record_conv_kernel` (`67.12 ms`, `226.5 MB/rnd` read) total `145.67 ms` (`3.84 ms/round`, `5.6%` of decode) across 36 GDN layers storing FP16 `[128, 128]` states; at `d4096`/`d8192`, `int8` Value KV reads grow proportionally.
- **Concrete Code Change:**
  1. Evaluate `KV_DTYPE=k8v4` across long contexts (`4096+`) to halve Value-KV read bandwidth.
  2. Evaluate an optional **FP8/E4M3 GDN recurrent state** fast path (halving the `679 MB/round` GDN state R/W from `~3.84 ms` → `~1.8 ms/round`), keeping `--reproducible` bit-safety and qualifying purely under the model-level quality gate in `model-quality.md`.
- **Acceptance Criteria (Gate A + Quick Gate B):**
  - *Code Quality (Gate A):* `infernix_gated_delta_net_test`, `infernix_gdn_replay_fold_test`, and `infernix_cuda_stack_check` pass for FP8 GDN state.
  - *Quick Benchmark (Gate B):* Quick `infernix_bench` at `--concurrency 4 --rows distinct -pg 4096,128 -r 2 --warmup 1` comparing `--kv-dtype k8v4` vs `int8`, and `./build-native/bench/infernix_gdn_replay_bench --profile 35b --width 4 --batch 4` for FP8 vs FP16 state; defer full perplexity, [`model-quality.md`](../maintainer/model-quality.md), `tool-eval-bench`, and full `llama-benchy` to **Gate C**.

---

## 4. Cumulative Throughput Projection (Pending Per-Task A/B Verification)

*All projected tok/s figures below are roofline-informed estimates derived from Cluster 1 `66.5 ms/round` assuming ~43 rounds/512 tokens on standard MTP (`~12.0 tok/round`) unless noted for Tier 3.*

| Optimization Stage | Estimated `ms / round` | Projected `c=4` `tg128` (`d0..d8192`) | Risk to `c=1` / `pp2048` |
|---|---|---|---|
| **Baseline (`f65e1a776` / `HEAD`, MTP)** | `66.5 ms` | `~167–179 t/s` (`167.3 / 166.0 / 161.1`) | Baseline |
| **+ Tier 1 (`T1–T5`)** | `~61.0–62.5 ms` (`-4.0 to -5.5 ms`) | **`~182–193 t/s`** | None (all gated to `T >= 9` or graph split) |
| **+ Tier 2 (`T6–T8`)** | `~57.0–59.0 ms` (`-3.0 to -4.5 ms`) | **`~198–208 t/s`** (even on MTP) | None if `1..8`-col `REG <= 94` kept |
| **+ Tier 3 (`T9 DFlash2` + `T10 Trees`)** | `~53–56 ms` + `+10–15%` tok/rnd | **`~215–240+ t/s`** | Must verify `c=1` DFlash2 acceptance on `hc4` |

---

## 5. Verified Dead Ends & Guardrails (Do Not Repeat)

The following approaches were previously tested on GB10 and **regressed or deadlocked**; do not re-introduce them:

1. **3-Stage `MaxGroups=3` rewrite inside `gate_up_omma_persistent_kernel` / `down_omma_persistent_kernel`:** Raised register count to `REG=96` (`2 blk/SM`), grew `SHARED` to `16,160 B` and SASS (`4,296` vs `3,512` instructions), regressing MoE by `+467 ms` (`185 -> 172 t/s`). Keep `REG <= 94` (`__launch_bounds__(128, 5)`).
2. **Inline `c_base += 8` `omma_merge` loop inside existing 2-stage persistent OMMA (`HEAD`):** Caused GPU deadlock (`96%` util, `18 W`) on `Run 2/2` due to `mbarrier` phase/preload mismatch (`preloaded0`). Handle `9..24`-column experts in a dedicated kernel (**T8**) rather than mutating the `1..8`-col hot loop.
3. **K-baked `n2560_k6144` (`T<=8` / `T<=24`) & `RowTiles=1` `n13952_k2560` (`T<=16`):** K-baked `T=20..24` regressed `+50%` (`49 -> 73.4 us`), `T=8` regressed (`45.92 vs 45.22 us`), and `RowTiles=1` doubled activation L2 traffic.
4. **Tuning `TCap=24` / `TCap=32` or 2-slice `16+16` inside `hyper_connection_mix_fused.cu`:** Composed fallback is already `41.4–41.9 us` (`-10%` to `-64%` faster than `hyper_connection_mix_fused` at `T=20..32`); use **T1** (`kHcMixFusedMaxColumnsQ4 = 16`).
5. **MTP `kMultiRowWidthCost` `< 0.05` / `kMultiRowMaxDrafts` `> 5` and `NGRAM_CACHE_MIB=16384`:** `(0.02, 6)` (`165.6 t/s`) and `(0.015, 8)` (`169.4 t/s`) scored below `(0.05, 5)` (`170.7 t/s`) in 3-run `c=4` A/B; `NGRAM_CACHE_MIB=16384` left only `1.0 GiB` free page cache on `121 GiB` UMA (keep `2048` in [`run_server.sh#L55`](../../tools/run_server.sh#L55)).
6. **Unconditional final-state GDN write during `recurrent_record_conv_kernel`:** Writing `S_final` during speculation adds `+226 MB` of unconditional DRAM traffic and only breaks even when full-accept probability `p^K > 0.50`.

---

## 6. Step-by-Step Verification & Benchmark Acceptance Protocol

> **Strict Rule Against Premature Full `llama-benchy`:** A full `llama-benchy` matrix (`d0/d4096/d8192 × c=1/2/4`, ~6–10 min/run + server restarts) slows iteration by an order of magnitude. **Never run full `llama-benchy` after individual tasks.** Use **Gate A** + **Gate B** (seconds per check) for every task `T1`–`T11`, and run full `llama-benchy` **once in Gate C** after all planned changes are complete.

### 6.0 One-Time Pre-Campaign Quick Baseline

Before editing `T1`, record the quick native baseline once (using `INFERNIX_EXPERTS_ON_GPU=1` from [`run_server.sh`](../../tools/run_server.sh#L218)):

```bash
MODEL=/opt/etc/models/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix-HC4-W4Dense/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix-HC4-W4Dense.infernix
NGRAM=/opt/etc/models/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix.ngram

INFERNIX_EXPERTS_ON_GPU=1 ./build-native/bench/infernix_bench \
  --weights "$MODEL" --ngram-volume "$NGRAM" --ngram-residency volume \
  --kv-dtype int8 --expert-a4-activations \
  --spec mtp --draft-tokens auto --lm-head-draft --ngram-draft-tokens 31 --ngram-min-match 8 \
  --concurrency 4 --rows distinct -pg 512,128 -r 2 --warmup 1

INFERNIX_EXPERTS_ON_GPU=1 ./build-native/bench/infernix_bench \
  --weights "$MODEL" --ngram-volume "$NGRAM" --ngram-residency volume \
  --kv-dtype int8 --expert-a4-activations \
  --spec mtp --draft-tokens auto --lm-head-draft --ngram-draft-tokens 31 --ngram-min-match 8 \
  --concurrency 1 -pg 512,128 -r 2 --warmup 1
```

### 6.1 Gate A — Per-Task Code Quality & Correctness (Mandatory Before Benchmarking)

1. **Incremental Target Build:** Build only the affected tests, microbenches, and `infernix_bench` in a single Ninja call (skip `infernix-serve` relinks while iterating on Ops unless serving changed):
   ```bash
   ninja -C build-native -j "$(nproc)" <affected_tests> <affected_benches> infernix_bench
   ```
2. **Oracle & Contract CTests:** Run all affected CTest targets against their naive FP32 / exact oracles:
   ```bash
   ctest --test-dir build-native -R "<task_test_regex>" --output-on-failure
   ```
3. **CUDA Stack Spill & Register Ratchet:** Verify zero new `STACK > 0` / `LOCAL > 0` symbols and check modified kernel `REG`/SMEM/SASS (`PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt static --sass ...`):
   ```bash
   ctest --test-dir build-native -R infernix_cuda_stack_check --output-on-failure
   ```
4. **Diff Hygiene:** Confirm `git diff --check` is clean and superseded aliases/branches are removed.

### 6.2 Gate B — Per-Task Quick Benchmark Acceptance (`~20–45 s/task`, Repeat Per Step)

1. **Level 1 — Task Op Microbenchmark (`~5–15 s`, when applicable):** Run the exact Op bench listed in the task's acceptance block (`infernix_hyper_connection_bench`, `infernix_gdn_replay_bench`, `infernix_offloaded_moe_decode_bench`, `infernix_offloaded_moe_wide_bench`, `infernix_resident_moe_bench`) and confirm it meets the target delta without regressing narrow widths (`T = 1..8`).
2. **Level 2 — Quick Native E2E Smoke (`~20–40 s`, no HTTP server restart):** Run `infernix_bench` `c=4` and `c=1` `-pg 512,128 -r 2 --warmup 1` (plus `--profile-measured` under `nsys` only on tasks where kernel attribution matters: `T2`, `T3`, `T5`, `T8`).
3. **Keep / Revert Decision Rule:** Keep the task commit/diff and proceed to the next task **only if**:
   - **Gate A** is 100% green (`0` test failures, `0` new stack/local spills, `REG` within ceiling);
   - **Gate B** meets the task's Level 1/Level 2 improvement target **and** keeps `c=1` `tg128` and `pp512` within `>= -1.5%` of the prior step.
   - Otherwise debug or revert immediately before moving on.

### 6.3 Gate C — Final Full Campaign Sign-Off (Run ONCE After All Planned Tasks Land)

Only **after all selected tasks (`T1`–`T11`, or the completed campaign milestone) have passed Gate A + Gate B**:

1. **Full Binary Build & Full Affected CTest Sweep:**
   ```bash
   ninja -C build-native -j "$(nproc)" infernix-serve
   ctest --test-dir build-native -R "infernix_cuda_stack_check|hyper_connection|projection_fp32|screened_head|gated_delta_net|gdn_replay|offloaded_moe|speculative|qwen4_exp_fast_paths" --output-on-failure --parallel "$(nproc)"
   ```
2. **Single Final `llama-benchy` Matrix (`./tools/run_server.sh hc4`):**
   ```bash
   /opt/etc/Infernix/.venv-convert/bin/llama-benchy \
     --base-url http://127.0.0.1:8000/v1 \
     --no-cache --pp 2048 --tg 128 --depth 0 4096 8192 --concurrency 1 2 4 --runs 3
   ```
   - **Final Acceptance Thresholds:**
     - `c=4` `tg128`: **`>= 200.0 t/s`** sustained (`d0`, target `>= 200 t/s` across `d4096`/`d8192`), coherence PASS, `0` errors/hangs.
     - `c=1` `tg128` non-regression vs [`README.md`](../../README.md) (`f65e1a776`: `102.6 / 97.4 / 100.5 t/s` at `d0 / d4096 / d8192`; tolerance `>= -2.0%`).
     - `pp2048` prefill non-regression vs [`README.md`](../../README.md) (`f65e1a776`: `3,202 / 3,816 / 4,041 t/s` at `c=4`; TTFT tolerance `<= +2.0%`).
3. **Final Quality Parity (When `T9`/`T10`/`T11` Land):** Run `tool-eval-bench` (and [`model-quality.md`](../maintainer/model-quality.md) quality gate for `T11` / fast-path additions) **once** on the final combined server build before committing or updating `README.md`.

---

## 7. Gate C Verification Results (2026-11-24)

- **Kept Changes:** **T1** (`kHcMixFusedMaxColumnsQ4 = 16`), **T2** (head/tail split `verify_graph`), **T3** (`kRouterMmaMinColumns = 9` + decode HMMA schedule), **T5** (single-pass/4-stage `screen_kernel`), **T6** (already in base `098b2f4a3`), **T10** (`speculative_accept_compact_sparse_tree` + compact `tree_round` wiring).
- **Reverted / Deferred / Evaluated:** **T4** (reverted, slower `recurrent_record_conv`), **T7** (reverted, higher SMEM/reg pressure regressed), **T8** (Parts A/B reverted, `grouped_kernel` minor in decode & `xplane` regressed `c=1`), **T9** (`dflash2` regressed `c=4`; MTP `resident_moe_experts` at `228 GB/s` DRAM bound), **T11** (`k8v4` evaluated as optional; `int8` kept default; FP8 GDN deferred).
- **CTest Sweep (`54 passed, 1 skipped (real-artifact env unset), 0 failed / 55`):** `ctest --test-dir build-native -R "infernix_cuda_stack_check|hyper_connection|projection_fp32|screened_head|gated_delta_net|gdn_replay|offloaded_moe|speculative|qwen4_exp_fast_paths" --output-on-failure --parallel "$(nproc)"` (`infernix_qwen4_exp_speculative_sampling_real_test` skipped). `speculative_accept_compact_sparse_tree_kernel<false/true>`: `REG:79 STACK:0 SHARED:7256 LOCAL:0`.
- **Live Server `llama-benchy` (`tools/run_server.sh hc4`, Model ID `098b2f4a`, `--no-cache --pp 2048 --tg 128 --depth 0 4096 8192 --concurrency 1 2 4 --runs 3`, coherence `PASSED`):**

| Depth | `c=1` `tg128` (`peak`) | `c=1` `e2e_ttft` | `c=2` `pp2048` / `tg128` (`peak`) | `c=4` `pp2048` (`e2e_ttft`) | `c=4` `tg128` (`peak`) |
|---|---|---|---|---|---|
| `d0` | `86.93 ± 3.72` (`87.33`) | `625.09 ± 19.11 ms` | `3269.76` / `125.13 ± 9.92` (`129.33`) | `3171.43 ± 43.60` (`2584.78 ms`) | `174.98 ± 7.17` (`191.67`) |
| `d4096` | `93.78 ± 3.99` (`94.00`) | `1562.04 ± 9.59 ms` | `4008.42` / `119.23 ± 3.35` (`120.67`) | `3971.25 ± 38.00` (`6189.61 ms`) | `181.29 ± 7.39` (`199.33`) |
| `d8192` | `91.77 ± 7.44` (`92.33`) | `2451.29 ± 4.60 ms` | `4159.45` / `126.10 ± 5.84` (`129.33`) | `4175.72 ± 2.37` (`9809.79 ms`) | `179.39 ± 12.47` (`186.67`) |

- **Gap vs Acceptance Thresholds:** Compared with the user's pre-campaign `098b2f4a` `llama-benchy` baseline (`c=1` `87.5 / 90.6 / 89.2`, `c=4` `161.8 / 169.6 / 170.7`), `c=4` `tg128` improved by **`+8.1% / +6.9% / +5.1%`**, `c=1` was `-0.65% / +3.5% / +2.9%`, and `e2e_ttft` / `c=4` `pp2048` improved (`-3.7%` to `-7.5%` TTFT). However: (a) sustained `c=4` `tg128` (`174.98–181.29 t/s`, peak `199.33 t/s`) remains **`~19–25 t/s` short** of the **`200.0+ t/s`** target because `T4`/`T7`/`T8`/`T9` hit hardware/numerical limits or DRAM bandwidth ceilings, and (b) `c=1` `tg128` remains **`-15.3% / -3.7% / -8.7%`** below the older `README.md` (`f65e1a776`: `102.6 / 97.4 / 100.5`) figure (already lower on base `098b2f4a` before this campaign). Note `tool-eval-bench` was not run since `c=4` stayed below `200 t/s` and `KV_DTYPE=int8` remained default.
