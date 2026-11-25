# GB10 upstream sync into N1X, November 24, 2026

The N1X parent is `018283f2` and the incoming GB10 parent is `b67e16f1`.
Their common base is the fork's `origin/main` at `f65e1a77`. Five upstream
commits changed 78 files (+3,489 / −515) from that base. The merge commit
has 109 changed files (+4,718 / −518) against `origin/main`. The integrated `cuda-kernel-toolkit/` is
excluded from the file breakdown.

| Subsystem | Action and mechanism |
| --- | --- |
| Core and runtime | M `src/core/pdl.cuh` documents the producer-written input contract. M `src/runtime/{contract/execution,engine/engine_core}.h` and `include/infernix/ops/sampling.h` carry the optional tool-branch token into constrained sampling. M the projection and speculative-tree headers for batched FP32 projection and sparse-tree acceptance. |
| Model execution | M Qwen3.5 frontend output sessions, text execution, decode, prefill and target verification propagate the JSON/tool branch choice. M its n-gram proposer and suffix automaton track candidate matches. M Qwen4Exp forward, graph construction, gating and program execution batch verification and calibrate sampled draft cutoffs. |
| Ops and quantization | M sampling and speculative-round kernels, wrappers and launchers implement branch selection and tree verification. M BF16, fused NVFP4 RMSNorm, projection, screened head, hyper-connection and expert kernels for numerical alignment and batched decode. M PDL consumers in GGML, BF16/FP8/NVFP4/Q4–Q8 linear, SwiGLU, MoE, QSA and sliding-window attention use coherent loads for producer-written activations. |
| Tools and serving | M `tools/chat_templates/qwen3_8.jinja` adopts upstream's operational rules and retains N1X's distinction between user instructions and tool-result data. The launcher, worker profile, converter and artifact format have no incoming changes. |
| Tests and benchmarks | M the Qwen3.5 frontend and suffix-automaton tests, Qwen4Exp speculation tests, and projection, sampling, screened-head and speculative-tree Op tests. M the README performance table. A [upstream GB10 campaign report](../tasks/20261123-gb10-c4-decode-200tps-tasks.md). |
| Maintenance | M `AGENTS.md` records the direct-merge policy for the N1X fork. A this changelog. Earlier N1X packaging, launcher, worker and measurement files are indexed in the [N1X branch history](../../bench/n1x/README.md). |

## Future conflict resolution

Keep upstream GB10's 48-SM wave sizing, sliced-K schedules, launch bounds,
1:1 FP32 tensor-core accumulation, packed `__vsub4` emulation and FMA
contraction handling. Its 128 GiB unified LPDDR5X permits zero-staging
device-resident experts, and its Grace 20-core big.LITTLE ARMv9.2 host informs
CPU thread defaults. Take upstream math and config fixes with the actual
N1X memory partition, WSLC artifact mounts and server configuration in view.
Bandwidth-bound decode motivates speculative batching. Validate DFlash2
depth and concurrency on N1X before transferring GB10 performance claims.

## N1X verification, November 24

Compared the pre-sync image `a772ac9c0af1` and its `018283f2` chat template
with the merged image `22d602a92207` and its resolved template. Both ran the
daily HC4-W4Dense recipe on the same RTX Spark N1X and 8 GiB partition under
the same underclock, using ARM64 WSLC, CUDA 13.5, eager DFlash2 K3, INT8 KV,
262,144 context and eight lanes. The production image built all 743 Ninja
steps. The two servers ran one at a time, baseline first.

Used the archived `20261123-n1x-drafter/grid.ps1` from
[`benchmark-history.tar.gz`](../../bench/n1x/benchmark-history.tar.gz) with
`-Cells '1,4'`, one discarded warm-up grid and two measured trials per
image. Client order reversed on the second trial. Compilation finished
before measurement. The fixed-length rate divides generated tokens by
total batch wall time over both trials, including prompt processing.
All 40 fixed-length replies produced 128 tokens. The counts matched the
exact expected string. Explanations produced two distinct texts across the
two builds. The results describe these runs under the owner's underclock.

| Workload | Clients | Before (tokens/s) | Merged (tokens/s) | Delta |
| --- | ---: | ---: | ---: | ---: |
| Counting | 1 | 62.9 | 61.0 | −3.0% |
| Counting | 4 | 125.8 | 110.4 | −12.3% |
| Explanation | 1 | 49.5 | 42.3 | −14.6% |
| Explanation | 4 | 98.7 | 90.0 | −8.8% |

All ten inventory sessions per image returned the expected setup reply,
tool name, arguments and final answer. Mean three-turn batch wall time rose
from 11.53 to 20.95 s at one client (+81.7%) and from 40.45 to 60.22 s at
four (+48.9%). The merged run generated more setup tokens, including
reasoning, with 940 versus 231 at one client and 2,783 versus 888 at four.
The mean one-client tool-selection turn fell from 3.21 to 3.13 s.
The comparison includes both engine and template changes. Server telemetry
sampled mean GPU clocks of 2,004 and 2,026 MHz and mean power of 32.5 and
28.9 W across the respective grids. The images ran in fixed order.

The merged endpoint passed a combined JSON-schema and optional-tool request
on both branches, including the final JSON answer after the tool result.
A real OpenCode parent dispatched the named lookup worker, which read the
fixture and returned its exact marker. Both worker turns used medium effort
and the configured 8,192-token cap.

Ten focused CTests passed, including the Python/C++ template comparison,
frontend, suffix automaton, speculation, sampling, screened head, projection
and both speculative-tree tests. The CUDA stack check failed on 138 entries
in both images. The violating symbols and byte counts matched exactly across
8,323 baseline and 8,340 merged kernels.

## CUDA Graph replay

Removing only `--no-cuda-graph` from the merged launcher arguments produced
24 successful 32-token responses at temperature 1 (seeds 0–11, twice).
The archived November 22 `patched-sampled.json` run returned four non-finite
logit errors in 24 requests on the same seeded workload.

With graphs enabled, the same two-trial grid delivered 71.4 and 123.5 tokens/s
on counting at one and four clients (+13.6% and −1.8% against today's pre-sync
eager image). Explanation delivered 46.6 and 98.2 tokens/s (−5.8% and −0.6%).
All fixed-length checks passed. Inventory took 24.80 and 56.69 s (+115.1% and
+40.1%). One four-client setup turn answered that an external instruction was
ignored when the fixture requested `READY`. All ten tool calls and final
answers passed. A separate sampled 64,030-token repetitive prompt returned
HTTP 200 but continued the repeated text for all 64 output tokens instead of
returning the requested marker. The harness stopped before the follow-up.
This prompt was tested only with graphs. Keep eager launches for the daily worker.
