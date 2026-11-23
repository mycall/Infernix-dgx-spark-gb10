# Infernix at 8 and 96 GiB dedicated on RTX Spark N1X

The 96 GiB setting improved both decode throughput and cold prefill on this host. For six short inventory workflows with the same scored cache hits, median follow-up latency fell 7.9%. The measurements used the same CUDA 13.5 WSLC image (`a772ac9c0af1`), driver 621.24, HC4 model, eager DFlash2 K3, eight lanes, 4,096-token prefill chunks and shared 262,144-token INT8 KV pool. At 96 GiB the loaded Windows adapter showed 74.4 GiB dedicated and 2.7 GiB shared in use. The earlier 8 GiB snapshot showed 7.9 GiB dedicated and 66.5 GiB shared. Both snapshots include the desktop.

| Infernix workload | 8 GiB dedicated | 96 GiB dedicated |
| --- | ---: | ---: |
| Inventory, exact tool and answer, six sequential trials | 6/6, median 5.283 s (5.247–5.330) | 6/6, median 4.869 s (4.800–4.924) |
| Fixed 128-token counting output, six runs, median server decode | 77.92 tok/s (75.46–79.45) | 89.12 tok/s (83.47–90.14) |
| Cold 250,001-token tool selection, client time to first token | 97.98 s | 90.10 s |
| Two near-full clients, first-attempt tool tasks and total group wall time | 3/6 in 112.28 s | 3/6 in 109.41 s |

The short inventory timer starts after the setup reply. Each scored selection and final answer reused 8,192 tokens and generated 101 and 65 tokens respectively. Median setup on trials 2–6 was 3.434 versus 3.158 s. The first, uncached setup was 8.811 versus 6.252 s, one observation per partition. All six fixed-output pairs generated the same 128 tokens of text. Median decode throughput rose 14.4%.

For near-full work, both clients calibrated to 250,001 tokens. The admitted client generated 194 tokens on its cold selection at both settings. Cold prefill reached 2,552 versus 2,775 prompt tokens/s. Its first tool continuation was slower at 96 GiB. First-token latency rose from 0.625 to 6.902 s, and the engine reused 233,472 instead of 249,856 prompt tokens. The admitted client completed all three tasks. The second client received HTTP 503 under the default 30-second admission timeout in each run. Its remaining two tasks never ran. Raising the dedicated partition did not change the configured shared KV pool. These are runs on one machine, one partition setting per boot. Repetition on the short inputs measures within-run spread. Page size, clocks and reboot-to-reboot variation were uncontrolled.

## Two profiles at 96 GiB

Both profiles passed all six short inventory workflows. Infernix took 4.869 s on the median tool selection plus final answer, versus 5.590 s for `qwen38fn-best`. Including the initial setup reply reversed the order. The median complete workflow took 8.060 s with Infernix and 7.667 s with vLLM. Both reused 8,192 tokens on every scored request. Infernix generated 166 scored output tokens per trial, while vLLM generated 128–175. These timings measure task completion with each profile's own generation behavior.

The live `qwen38fn-best` profile completed all six near-full tasks on the first attempt in 201.65 s, with zero preemptions. Its configured tiers were 15 GiB GPU KV, 4 GiB LMCache host memory and 160 GiB LMCache disk. During the two-client group, vLLM's native prefix-cache hits increased by 2,498,560 tokens and its external prefix-cache hits increased by zero. This run demonstrates native reuse. Measuring the effect of enabling LMCache itself would require an on/off comparison. The local vLLM image was `ebfd91337c4d`. The committed historical profile pins a different image. Startup and model loading are excluded for both engines.

## Infernix cache capacity at 96 GiB

Each variant changed one launcher parameter on the same image. Each server started cold, then served the same six short inventory trials followed by the two near-full clients. The per-client context was 262,144 tokens and the admission deadline was 30 seconds throughout. The GPU KV pool holds live attention state across all clients. Infernix also keeps reusable prefix snapshots in host RAM.

| Infernix setting | Default | Double GPU KV | Quadruple host prefix cache |
| --- | ---: | ---: | ---: |
| Requested GPU KV pool, tokens | 262,144 | 524,288 | 262,144 |
| Host prefix cache, GiB | 1 | 1 | 4 |
| Short inventory median follow-up, six correct out of six | 4.869 s | 4.901 s | 4.872 s |
| Near-full tasks passed / planned | 3/6 | 3/6 | 3/6 |
| Near-full group wall time | 109.41 s | 110.07 s | 102.94 s |
| First tool continuation, cached tokens | 233,472 | 233,472 | 249,856 |
| First tool continuation, time to first token | 6.902 s | 6.857 s | 0.664 s |
| GPU prefix blocks evicted during group | 370 | 0 | 158 |
| Host snapshots evicted during group | 6 | 6 | 3 |

Doubling GPU KV eliminated device prefix-block evictions, with essentially the same short follow-up latency and the same near-full score. Raising the host cache to 4 GiB let the admitted client's first continuation reuse another 16,384 tokens. Its cold selection was similar at 89.94 s versus 90.10 s. The second client hit the queue timeout on every Infernix variant. Each near-full configuration was run once, so the host-cache result is evidence for repeated long prefixes, with more repetitions needed to establish its typical benefit. A token budget can govern which work fits in memory. The separate admission deadline bounds how long a caller waits.

The [measurement archive](20261123-partition-comparison.tar.gz) contains requests, responses, hardware readings, profile settings and cache counters. The six earlier short runs are under `8g-inventory/`. The [8 GiB comparison](20261123-native-context.md) identifies the matching earlier near-full and fixed-output records and the Infernix client adaptation. Both live engines used the same near-full task and first-attempt scoring from n1x-ai commit `38cd442`. vLLM used the stock client and Infernix used the adapted client for token counting and API compatibility.

## Longer repeat on the 96 GiB boot

| Measured workload | Earlier 8 GiB | Earlier 96 GiB | Fresh 96 GiB repeat |
| --- | ---: | ---: | ---: |
| Fixed 128-token decode, median tok/s | 77.92 (6 outputs) | 89.12 (6) | 88.92 (24, range 86.26–90.44) |
| Inventory tool follow-up, median seconds | 5.283 (6 tasks) | 4.869 (6) | 4.827 (12, range 4.777–4.939) |

The [repeat runner](compare-partitions.ps1) warmed up both workloads, then alternated their order across four blocks. Block medians were 88.00, 89.28, 89.15 and 89.02 tok/s. All 24 counting outputs matched the six archived 8 GiB outputs exactly and used zero cached prompt tokens. All 12 inventory tasks passed. Each tool selection and final answer reused 8,192 tokens and generated 101 and 65 tokens respectively. The median GPU temperature was 72.9 °C and the median SM clock was 2,385 MHz across 149 samples with at least 80% GPU utilization. The laptop was on AC with the Balanced power scheme. Windows reported 30.5 GiB visible RAM, versus about 119 GiB in the 8 GiB Task Manager snapshot.

The 96 GiB repeat reproduced the initial results on the same boot. A fresh 8 GiB boot with the same runner would test whether the difference reverses with the setting. A subsequent independent 96 GiB boot would give two boots at each setting. A 4 GiB comparison and desktop responsiveness under memory pressure remain unmeasured. [Raw repeat results and telemetry](20261123-partition-repeat-96g.tar.gz)

After changing the partition and starting the same daily Infernix profile, run the following from the engine checkout in PowerShell. The runner records the boot time, partition, image, launch arguments and power scheme in the output directory.

```powershell
.\bench\n1x\compare-partitions.ps1 -OutDir (Join-Path $env:LOCALAPPDATA ("infernix-bench/partition-" + (Get-Date -Format yyyyMMdd-HHmmss)))
```
