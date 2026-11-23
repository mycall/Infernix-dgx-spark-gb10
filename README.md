# Infernix — Retuned for NVIDIA DGX Spark (`NVIDIA GB10`, `sm_121a`)

**Infernix** is a high-performance, from-scratch C++20/CUDA inference engine for the Qwen3.5, Qwen3.6, Qwen3.8, and **Qwen3.8-Flash-Next** (`Qwen4ExpForCausalLM`) model families.

For Windows on ARM64 with CUDA 13.5, see [N1X setup and serving](docs/serving.md#hc4-w4dense-with-dflash2-on-n1x-under-wslc) and [benchmark history](bench/n1x/README.md).

> **Project Lineage & DGX Spark (`GB10`) Retune:** This project is built on the **Infernix** inference engine (which originally began as a fork of [NInfer](https://github.com/Neroued/ninfer) by [Neroued](https://github.com/Neroued) targeting the desktop RTX 5090 `sm_120a`) and has been **architecturally retuned end-to-end for the NVIDIA DGX Spark (`NVIDIA GB10`, Blackwell `sm_121a`)** on 64-bit Linux `aarch64`. Every CUDA kernel family, `/shapes/` dispatch table (`103/103` shapes), TMA descriptor policy, L2 cache swizzle, unified-memory (UMA) expert placement, and multi-row Multi-Token Prediction (MTP) speculative scheduler has been calibrated specifically for the GB10's **48 SMs**, **24 MiB L2 cache**, **128 GB unified LPDDR5X memory**, and **20-core Grace ARMv9.2 CPU**.

---

## Key Architectural Highlights on NVIDIA DGX Spark (`GB10`)

### 1. Full-Model Unified Memory Residency for `Qwen3.8-Flash-Next` (`HC4-W4Dense`)
On **DGX Spark's 128 GB unified LPDDR5X**, Infernix pins all 48 layers of routed experts directly in device-accessible unified memory (`INFERNIX_EXPERTS_ON_GPU=1`, zero PCIe miss transfers) and streams the PLE n-gram volume with a 16 GiB GPU-direct UMA cache and 20-thread ARMv9.2 prefill workers.

### 2. Microarchitectural Retuning for Blackwell `sm_121a` (48 SMs, 24 MiB L2)
- **Native 2-Term Row-Scaled `NVFP4 A4` TMA (`OMMA.16864.F32.E2M1`, `501.6 TFLOP/s` pipe)**: Trunk and MoE projections dynamically quantize BF16 activations into two per-token row-scaled NVFP4 terms (`FastPath::Nvfp4A4Pair`) using 1-pass register-cached reciprocals and 2-CTA/SM `128×64` TMA tiles with `CU_TENSOR_MAP_L2_PROMOTION_L2_128B`.
- **24 MiB L2 Cache Thrashing Elimination**: Added 2D `Grouped, 8` CTA rasterization across all `BF16`, `FP8`, `Q8`, `NVFP4`, `Q6`, `Q5`, and `Q4` MMA kernels.
- **Direct FP32 Tensor Core Accumulation (`1:1` HMMA Parity on `sm_121a`)**: Replacing `mma_f16_f16acc` with direct FP32-accumulating `mma_f16` across dense causal attention (`fp8`, `k8v4`, `nvfp4`, `int8`, `vq2`) and Gated Delta Net (`chunked/recurrence.cu`) eliminated **100% of local-memory stack spills (`STACK = 0`)** while improving numerical precision.
- **Hardware `F2FP` + `HMUL2` Bit-Exact NVFP4 Codec & 2×/3× Occupancy Narrow MoE**: Hardware `__nv_cvt_fp4x2_to_halfraw2` + `__hmul2` (`0 ULP` exact in IEEE FP16), `48.0 KiB` static SMEM `Nvfp4SlicedInstance<16, 8, 2, 2>` (`2 CTAs/SM`), and register-quantized narrow MoE (`moe_layer.cu`, `STACK = 0`).

### 3. Production Serving & Agentic Capabilities
- **OpenAI & Anthropic APIs**: `/v1/chat/completions`, `/v1/responses`, `/v1/messages`, and `/v1/decide` with streaming, reasoning budgets (`--reasoning-effort`, `--restore-tool-reasoning`), tolerant XML/JSON tool-call recovery (`--tolerant-tool-calls`), and constrained decoding (JSON Schema, regex, GBNF, choice grammars).
- **Hybrid Prefix Cache**: Content-addressed 64-token KV blocks plus recurrent GDN/conv state snapshots across GPU, host RAM, and persistent disk tiers (`--prefix-cache-save-mins`).
- **Seven KV Cache Formats**: `bf16`, `fp8`, `int8`, `k8v4`, `nvfp4`, `k4v2`, and `vq2`.

---

## Supported Model (`HC4-W4Dense`)

| Model | Hugging Face Download | Description |
|---|---|---|
| **`Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix-HC4-W4Dense`** | [eugene-kamenev/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix-HC4-W4Dense](https://huggingface.co/eugene-kamenev/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix-HC4-W4Dense) | `Qwen3.8-Flash-Next` (`Qwen4ExpForCausalLM`) in `HC4-W4Dense` NVFP4 layout with vision support (`--vision`), `--expert-a4-activations`, MTP speculative decoding, and PLE n-gram volume. |

---

## Quick Start (DGX Spark / Linux `aarch64`)

### 1. Prerequisites
Builds and runs natively on **64-bit Linux `aarch64`** (Ubuntu 24.04 LTS, CUDA 13.0, GCC 13.3):

```bash
sudo apt-get update && sudo apt-get install -y \
  build-essential cmake ninja-build pkg-config \
  libavformat-dev libavcodec-dev libavutil-dev libswscale-dev \
  libcurl4-openssl-dev
```

### 2. Build
```bash
cmake -B build-native -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=121a
ninja -C build-native -j "$(nproc)" infernix-serve infernix_bench
```

> **WARNING:** The repository [`Dockerfile`](Dockerfile) / Docker image has not been well tested yet and should **not** be used for production runs. Build and run natively on the host (Ubuntu 24.04 LTS, CUDA 13.0) instead.

### 3. Run the Server (`infernix-serve`)

> **IMPORTANT:** The chat template [`tools/chat_templates/qwen3_8.jinja`](tools/chat_templates/qwen3_8.jinja) **must be used** (`--chat-template tools/chat_templates/qwen3_8.jinja`) for proper thinking, tool-calling, and structured JSON output behavior.

Use [`tools/run_server.sh`](tools/run_server.sh) (which builds `infernix-serve` and passes `--chat-template tools/chat_templates/qwen3_8.jinja` automatically):

```bash
./tools/run_server.sh hc4
```

Or launch `build-native/apps/infernix-serve` directly with the default `HC4-W4Dense` command:

```bash
INFERNIX_EXPERTS_ON_GPU=1 \
./build-native/apps/infernix-serve \
  /opt/etc/models/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix-HC4-W4Dense/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix-HC4-W4Dense.infernix \
  --expert-a4-activations \
  --vision \
  --spec mtp --draft-tokens auto \
  --ngram-volume /opt/etc/models/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix.ngram \
  --ngram-residency volume \
  --ngram-cache-mib 2048 \
  --kv-dtype int8 \
  --ngram-draft-tokens 31 \
  --ngram-min-match 8 \
  --temperature 1.0 \
  --top-p 0.9 \
  --min-p 0.1 \
  --top-k 20 \
  --lm-head-draft \
  --prefill-chunk 12288 \
  --kv-capacity 524288 \
  --default-max-tokens 32768 \
  --preserve-thinking \
  --chat-template tools/chat_templates/qwen3_8.jinja \
  --restore-tool-reasoning \
  --reasoning-effort medium \
  --reasoning-loop conclude \
  --host-context-mib 4096 \
  --tolerant-tool-calls \
  --max-context 262144 \
  --max-concurrency 8 \
  --host 0.0.0.0 \
  --port 8000
```

#### Handling Concurrent Bursts (Agents & Proxies)

`infernix-serve` runs at most `--max-concurrency` active requests at once (maximum `8`). Additional requests wait in a FIFO queue of size `--max-pending-requests` (default `16`, for `24` total in-flight requests at `--max-concurrency 8`; exceeding that returns HTTP `429` `server_overloaded`) and fail with **HTTP `503` (`request_queue_timeout`)** if they wait longer than `--pending-timeout-ms` (default **`30000` ms / 30 s**, covering preparation and queue wait before admission).

For example, a burst of 16 simultaneous requests (`C16`) against `--max-concurrency 8` queues 8 requests behind the active lanes; if those lanes take more than 30 s to free up, the queued requests time out with HTTP `503`. When running `infernix-serve` behind agent harnesses or a proxy where bursts above 8 are normal, raise `--pending-timeout-ms` (and `--max-pending-requests` if bursts exceed `24` total requests):

```bash
--pending-timeout-ms 300000 \
--max-pending-requests 32
```

Once a request is admitted to an active lane, `--pending-timeout-ms` no longer applies. See [HTTP Serving — Execution behavior](docs/serving.md#execution-behavior) for full queueing and scheduling details.

#### Tuning the N-gram Row Cache (`--ngram-cache-mib`)

When streaming the PLE n-gram table from NVMe (`--ngram-residency volume`), `--ngram-cache-mib` sets the budget for the 4-way set-associative host/UMA row cache (rounded down to the largest power-of-two slot count from `2^16` to `2^28` rows that fits in the budget; `0` uses the default `2^20` rows).

The example command above uses `--ngram-cache-mib 2048` as a conservative starting point, which may be too small for workloads with diverse prompts, long contexts, or high multi-session concurrency. Raising `--ngram-cache-mib` in power-of-two steps (for example `4096`, `8192`, or `16384`, or via `NGRAM_CACHE_MIB=8192 ./tools/run_server.sh hc4`) reduces NVMe read misses at the cost of unified memory shared with `--kv-capacity` and `--host-context-mib` — tune it to find the performance sweet spot for your workload. Run with `--log-level debug` to inspect per-request `n-gram rows` cache hit rates and NVMe read time (see [Qwen3.8-Flash-Next Guide — N-gram table in RAM](docs/qwen3_8-flash-next.md#n-gram-table-in-ram)).

---

## Benchmarks on NVIDIA DGX Spark (`NVIDIA GB10`)

### Benchmark Environment

| Component | Specification |
|---|---|
| **System / Device** | **NVIDIA DGX Spark** |
| **GPU** | **NVIDIA GB10** (Blackwell `sm_121a`, Compute Capability `12.1`, **48 SMs** @ up to `3,003 MHz`, **24 MiB L2 cache**) |
| **Memory Subsystem** | **128 GB Unified LPDDR5X** (`cudaDevAttrIntegrated = 1`, `~121 GiB` usable; `273.0 GB/s` theoretical peak, `247.6 GB/s` measured sustained read) |
| **CPU** | **20-core NVIDIA Grace ARMv9.2 (`aarch64`)** — `10× Cortex-X925` (`r0p1`) + `10× Cortex-A725` (`r0p1`), `48 MiB` L3 cache (`2 × 24 MiB`) |
| **Operating System** | **Ubuntu 24.04.5 LTS** (`Noble Numbat`, 64-bit `aarch64`) |
| **CUDA Toolkit** | **CUDA `13.0`** |
| **Host C++ Compiler** | **GCC `13.3.0`** |

### `llama-benchy` Results (`HC4-W4Dense`, `int8` KV Cache)

Model loading takes **8s**; full engine startup and graph initialization takes **14s**.

| Test | c | pp t/s | tg t/s | TTFT (ms) | Total (ms) |
|---|---:|---:|---:|---:|---:|
| pp2048 tg128 @ d0 | c1 | 3,058* | 102.6 | 670 | 1,907 |
| pp2048 tg128 @ d0 | c2 | 3,108 | 129.8 | 1,325 | 3,195 |
| pp2048 tg128 @ d0 | c4 | 3,202 | 167.3 | 2,571 | 5,409 |
| pp2048 tg128 @ d4096 | c1 | 3,823* | 97.4 | 1,607 | 2,911 |
| pp2048 tg128 @ d4096 | c2 | 3,823 | 123.4 | 3,223 | 5,258 |
| pp2048 tg128 @ d4096 | c4 | 3,816 | 166.0 | 6,455 | 9,399 |
| pp2048 tg128 @ d8192 | c1 | 4,035* | 100.5 | 2,538 | 3,802 |
| pp2048 tg128 @ d8192 | c2 | 4,036 | 116.9 | 5,082 | 7,205 |
| pp2048 tg128 @ d8192 | c4 | 4,041 | 161.1 | 10,151 | 13,144 |

\* `c = 1` prefill column is an estimator output, not wall-clock throughput — only `TTFT`-derived rates are meaningful at `c = 1`.

### `tool-eval-bench` Results (`HC4-W4Dense`, `tool-eval-bench v3.0.1.dev15+gea2a77499`)

| Metric | Value |
|---|---|
| **Score** | **96 / 100** (`133 / 138` points, ★★★★★ Excellent) |
| **Scenarios** | **65 passed**, **3 partial**, **1 failed** |
| **Quality** | **96 / 100** |
| **Responsiveness** | **60 / 100** (median turn: **2.3s**) |
| **Deployability** | **85 / 100** (`α = 0.7`) |

---

## Numerical Quality & Fast-Path Gate

Infernix provides two numerical execution routes:

- **The exact route (`--disable-fast-paths`)**: Produces bit-identical output from run to run and release to release. Where an artifact's formats allow it, it is also request-invariant (concurrent requests, prefix-cache state, and memory pressure do not alter a request's output).
- **Fast paths (enabled by default)**: Hardware-accelerated schedules and dynamic activation quantizations (`nvfp4-a4-pair`, `qsa-prepare-fused`, `gdn-f16-accumulate`, `sampled-mtp-drafts`, `adaptive-speculation`) that run only for the kernels they were qualified on, only after passing the numerical quality gate below, and only where measured faster on GB10.

### The Quality Gate
Every fast path is qualified against a high-precision reference (16-bit activations in every layer and a BF16 KV cache) over a 1.04M-token teacher-forced prefill corpus and 1,024 forced decoding segments (65,536 scored positions):
- **Shared budgets (at the 95% upper confidence bound)**: KL divergence $\le 1.5 \times 10^{-3}$, top-1 token agreement within $0.5$ percentage points, decode total variation (TV) $\le 4 \times 10^{-3}$, and 99.9th-percentile KL growth $\le 25\%$.
- Full specification: [Numerical Acceptance Policy (`docs/maintainer/model-quality.md`)](docs/maintainer/model-quality.md#6-numerical-acceptance-policy-and-fast-paths) and [GB10 Optimization Roadmap (`docs/gb10_roadmap.md`)](docs/gb10_roadmap.md).

---

## Documentation

- **DGX Spark (`GB10`) Engineering & Performance**:
  - [DGX Spark (`NVIDIA GB10`) Optimization Roadmap & Hardware Ceiling Audit](docs/gb10_roadmap.md)
  - [Performance Overview](docs/performance.md)
- **User & Product Guides**:
  - [CLI Guide](docs/cli.md)
  - [HTTP Serving & API Reference](docs/serving.md)
  - [Qwen3.8-Flash-Next Guide](docs/qwen3_8-flash-next.md)
  - [Perplexity & Quality Evaluation](docs/perplexity.md)
  - [Weight Conversion Guide](docs/weight-conversion.md)
- **Maintainer Architecture**:
  - [Documentation Map](docs/README.md)
  - [Engine Architecture](docs/maintainer/engine-architecture.md)
  - [Op Development & Qualification](docs/maintainer/op-development.md)

---

## Credits & Acknowledgments

- **Infernix** ([Wallawalla47](https://github.com/Wallawalla47)) provided the base engine, `.infernix` v3 container format, hybrid prefix cache, and `Qwen3.8-Flash-Next` implementation that this repository retuned for the **NVIDIA DGX Spark (`NVIDIA GB10`, `sm_121a`)**.
- **[NInfer](https://github.com/Neroued/ninfer)** by **[Neroued](https://github.com/Neroued)** ([Ko-fi](https://ko-fi.com/neroued)) created the original C++/CUDA engine and artifact architecture that Infernix grew from.
- **[Strata](https://github.com/Niko1221/Strata)** by **[Niko1221](https://github.com/Niko1221)** pioneered single-GPU `Qwen3.8-Flash-Next` execution and mathematical transcription.
- Thanks also to [IST-DASLab](https://github.com/IST-DASLab) (`llmq`), the authors of HyperQuant and Four Over Six, the Qwen team for the models, NVIDIA for the NVFP4 checkpoints, and all open-source contributors whose research and ideas shaped this engine.
