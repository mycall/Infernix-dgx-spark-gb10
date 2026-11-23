I might have found a way to get a 8% speed improvement but unsure if Infernix and delta-openshell-n1x will run as a 96GB dedicated VRAM setup (32GB shared RAM) on this system.

https://docs.nvidia.com/rtx-spark/rtx-spCark-porting-guide/latest/uma/umacuda.html

NVIDIA's documentation confirms that:

- `cudaMalloc` allocations go into dedicated GPU memory first, then spill into shared memory.
- Shared-segment allocations use smaller pages, potentially increasing GPU memory-access latency through additional address translation.
- Both segments use the same physical LPDDR5X memory, so this is **not the conventional PCIe shared-memory penalty** found with discrete GPUs.&#x20;

  [image](https://www.google.com/s2/favicons?domain=https://docs.nvidia.com\&sz=32)

  NVIDIA Windows on Arm Porting Guide



The important distinction is that smaller pages primarily affect **TLB coverage and memory-access patterns**, not necessarily raw DRAM bandwidth. Sequential operations may show little difference, while large, irregular memory accesses could suffer more.

## **1. Direct RTX N1X measurements — 4 GB versus 96 GB**

The strongest match is a November 15, 2026 benchmark by **maruhana**, using a 128 GB Surface Laptop Ultra with RTX Spark N1X, NVIDIA driver 621.24, and llama.cpp b12110.

The author changed dedicated GPU memory between 4, 64, 82, and 96 GB, rebooted between settings, and measured the same models.

Measured output generation speed, tokens/second — higher is better

GPT-OSS-120BNemotron 3.5 Bolt

0 tok/s20 tok/s40 tok/s60 tok/s80 tok/s4 GB64 GB82 GB96 GB

Source: maruhana, November 15, 2026. Three benchmark repetitions per configuration.

The GPT-OSS-120B model occupied approximately 63.4 GB, meaning most of its memory was outside the dedicated carveout in the 4 GB test.

# **8.0%**

Slower GPT-OSS-120B generation at 4 GB versus 64 GB

# **8.1%**

Slower GPT-OSS-120B prefill at 4 GB versus 64 GB

The 64–96 GB settings produced essentially the same generation speed. The author's three-run standard deviations were small enough to distinguish the approximately 8% gap from ordinary short-term variability, although each carveout configuration was tested during only one session.

Full results: [RTX Spark carveout benchmark, including commands and measurements](https://dev.to/maruhana/i-cut-dedicated-gpu-memory-to-4gb-and-a-local-llm-still-ran-at-90-speed-4h64?utm_source=chatgpt.com).&#x20;

[image](https://www.google.com/s2/favicons?domain=https://dev.to\&sz=32)

DEV Community



**This is the closest evidence to your proposed 8 GB versus 96 GB experiment.** It suggests that shared-segment CUDA allocations can be surprisingly competitive for LLM inference, although it does not isolate page size from other allocation differences.

## **2. Shared memory can be much slower for other GPU workloads**

The same researcher also tested ComfyUI video generation through WSL2, comparing 4 GB against 64 GB dedicated GPU memory.

| Video workload | 4 GB dedicated | 64 GB dedicated | Difference       |
| -------------- | -------------- | --------------- | ---------------- |
| Wan 2.2        | 271 s          | 143 s           | **1.90× longer** |
| MiniMax H3     | 33 s           | 30 s            | \~1.10× longer   |

These are averages of three videos per model. At 4 GB dedicated memory, shared GPU usage peaked at 47 GB. There was no observed WSL swapping, and SSD reads were similar between configurations.

That is significant: **the same hardware showed about an 8% LLM throughput penalty but an approximately 47% throughput reduction for Wan 2.2 video generation.** The author couldn't establish whether page size, cache behavior, allocation patterns, or another driver effect caused the difference.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://dev.to\&sz=32)

DEV Community



## **3. Actual page-size research on DGX Spark**

A separate October 2026 research paper, Characterizing CPU-GPU Memory Sharing over Unified Memory Architecture, provides page-size microbenchmarks in Appendix B, Figure B.1.

The authors compared 4 KB pages, 2 MB pages and Transparent Huge Pages on DGX Spark. For small memory regions, 4 KB mappings were faster to set up; for large regions, the curves converged. The paper also identifies TLB pressure as a mechanism through which small pages can degrade memory-access performance.

However, it **doesn't directly measure the execution-speed difference between the two Windows GPU memory segments**. Its other Spark measurements—up to 8% less GPU-read throughput and 27% less GPU-write throughput—compare cache/coherence policies, not page sizes.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv



[Read the research paper](https://arxiv.org/html/2610.13125?utm_source=chatgpt.com#S2.SS1) (see Appendix B for DGX Spark measurements).

## **What this means for your RTX N1X 128 GB**

The direct measurements support three conclusions:

- For the two tested LLMs, even **4 GB dedicated memory retained roughly 90–92% of generation speed** compared with larger carveouts.
- Some other workloads experienced much greater slowdowns, so that result cannot be generalized to all CUDA applications.
- There is still no published percentage I could verify that isolates the smaller-page penalty on Windows N1X from other factors.

For your proposed **8 GB versus 96 GB** setting, the most informative experiment would be to benchmark the same DeepSeek or Qwen model with the same runtime at 8, 64, and 96 GB, measuring both prefill and decode. The existing evidence makes 8 GB worth testing for LLM inference, while 64 GB looks like a reasonable compromise for mixed GPU workloads.

One correction to the earlier chart summary: GPT-OSS-120B prefill was **8.7% slower**, not 8.1%, at 4 GB versus 64 GB.