# N1X parallel-worker tuning, November 23, 2026

Measured on one RTX Spark N1X, 8 GiB dedicated partition, driver 621.24, ARM64 WSLC and CUDA 13.5. The Infernix branch was at `4a04e725`. All three chunk settings used the same local image (`a772ac9c0af1`), HC4-W4Dense weights, DFlash2 K3, eager CUDA, 262,144-token context and KV pool, INT8 KV, eight lanes, preemption, 4 GiB of host prefix cache and a 300-second admission timeout.

## Worker effort

A real OpenCode 2.2.7 parent dispatched two Explore workers on a synthetic repository. One read a release marker. The other traced an entry point, a Python settings function and a TOML override through to the effective route and timeout. The scored OpenCode pairs used the 8,192-output-token worker alias. The parent retained its ordinary full allowance and template reasoning. Each row below compares identical file contents across different fresh directories. The worker columns time the successful child tool call. Parent wall time includes dispatch, failures, retries and the final summary.

| Fixture and setting | Lookup | Three-file trace | Parent wall | Correct final workers | Dispatch errors |
|---|---:|---:|---:|---:|---:|
| Coral, default | 13.74 s | 43.95 s | 64.50 s | 2/2 | 0 |
| Coral, low | 11.68 s | 35.87 s | 50.67 s | 2/2 | 0 |
| Birch, default | 14.41 s | 48.14 s | 63.03 s | 2/2 | 0 |
| Birch, low | 11.14 s | 43.46 s | 70.45 s | 2/2 | 2 |

Only runs 4/5 and 8/9 produced matched, wire-verified pairs. The initial agent-only effort setting appeared in OpenCode's debug configuration, but Infernix still logged its template default. One subsequent parent selected the full-output model for its Explore calls. Those timings are in the archive and excluded from the comparison. Both usable pairs ran default first. The template resolves an unspecified effort to `xhigh`.

A separate direct API test removed OpenCode dispatch and model-selection variability. It ran the same two workers concurrently, with three paired fixtures, alternating effort order, one file-reading tool, greedy sampling and an 8,192-token cap. Every answer had to parse as JSON and use the necessary source files.

| Task | Default median | Low median | Strict passes, default / low | Median reasoning tokens, default / low |
|---|---:|---:|---:|---:|
| Read one marker | 5.01 s | 4.24 s | 3/3 · 3/3 | 79 · 65 |
| Resolve the effective configuration | 8.73 s | 10.13 s | 3/3 · 2/3 | 197 · 229 |

All six configuration answers contained the right values. One low-effort answer wrapped its JSON in Markdown fences. Low is a candidate for explicit lookup workers. Keep template reasoning on general Explore workers and strict structured-output tasks. These timings do not establish a general speedup for multi-file reasoning.

For the verified OpenCode configuration, the worker alias declared reasoning support and set `settings.reasoningEffort` and `body.reasoning_effort` to `low`. The Explore agent referenced that alias with an 8,192-token cap. A loopback test adapter recorded the outgoing fields before forwarding. Across the two counted low runs, all 14 worker requests already contained `low` and the server logged it. The model definition in `fixture-5/opencode.json` in the archive shows the tested fields. Its loopback URL was for instrumentation.

The shipped lookup worker uses the user's preferred medium effort. Replaying the same three direct API fixture pairs at medium gave 3/3 exact marker lookups (median 4.43 s). It recovered the effective route and timeout in all three traces (median 9.52 s). All three missed the strict output contract. One used an array containing an object, and two wrapped the requested array in Markdown fences. Keep structured multi-file work on general Explore. A live OpenCode parent retrieved the correct marker through the named lookup worker after retrying one dispatch with a misspelled model ID. Infernix logged medium effort and an 8,192-token cap on both worker turns. [OpenCode profile](../../tools/infernix-parallel.jsonc)

## Stable-prefix breakpoint

Compared the ordinary API request with adding `prompt_cache_breakpoint: {"mode":"explicit"}` to the stable system text. Both arms used identical tools and instructions, plus independent early case IDs to prevent a warm hit crossing arms. Each of three pairs per size made an initial read request followed by two tool-result turns. Thinking was disabled for these timing controls.

| Initial prompt | Cached tokens per follow-up, ordinary / explicit | Median follow-up wall, ordinary / explicit | Median three-turn wall, ordinary / explicit |
|---|---:|---:|---:|
| ~3,260 tokens | 0 / 0 | 1.62 / 1.64 s | 4.97 / 5.01 s |
| ~10,410 tokens | 8,192 / 8,192 | 1.34 / 1.30 s | 6.98 / 6.97 s |

All 24 initial and intermediate tool reads matched the expected path. Eleven of twelve final turns reproduced the right marker with a label. One made another file request. None met the exact bare-marker contract. A four-loop control returning a bare marker from the tool also missed the exact final-answer contract. A final control returned the marker in a JSON field and set `tool_choice: "none"` on the answer turn. All four tool loops passed every exact-answer check, with the same 0 and 8,192 cached tokens on every follow-up at the respective prompt sizes. Keep the ordinary cache hints for this client pattern. This tested one system-text breakpoint placement on the daily profile.


## Prefill chunk size

One worker streamed a 512-token answer. On its first content event a second client submitted a cold ~19,850-token prompt with a short known answer. The pause is the longest interval between visible content updates on the first stream that intersects the incoming worker's prefill. DFlash2 can emit several tokens per update. The table gives medians from four measured pairs per row after one discarded warm-up. A separate pair of ~31,940-token cold prompts measured standalone time to first token. Each size had one service start. The daily size was also repeated after restoring the service.

| Chunk | Longest stream pause | Incoming first token | Both finished | Standalone first token |
|---|---:|---:|---:|---:|
| 2,048 | 7.89 s | 7.88 s | 14.34 s | 12.66 s |
| 4,096 | 7.23 s | 7.21 s | 13.88 s | 11.59 s |
| 8,192 | 7.04 s | 7.02 s | 13.47 s | 11.06 s |
| 4,096 after restore | 7.18 s | 7.16 s | 13.76 s | 11.38 s |

All 16 measured pairs overlapped, the short worker produced 512 tokens, every incoming worker returned the expected answer and each long prompt was cold. The smaller chunk increased the visible pause. The 8,192-token chunk finished the cold mixed pair about 0.4 s faster than the initial 4,096-token run. All three sizes left a multi-second stall.

To check the cost for agent conversations, a second test replayed fixed, identical two-read tool loops at both chunk sizes. Each size had two cold sequences per length, followed by two follow-up requests each. These are ordinary requests without a cache marker.

| Initial prompt | Cached on four follow-ups, 4,096 / 8,192 chunk | Median follow-up wall, 4,096 / 8,192 chunk |
|---|---:|---:|
| ~3,262 | 0 / 0 | 1.62 / 1.62 s |
| ~6,008 | 4,096 / 0 | 1.24 / 2.60 s |
| ~10,411 | 8,192 / 8,192 | 1.35 / 1.38 s |

Keep 4,096 for the daily OpenCode workload. The 8,192 setting is a candidate for a dedicated stream of long, fresh prompts. The 2,048 setting had the slowest cold-prefill and mixed-pair medians. All measurements were on one machine and GPU partition. This comparison covers two active workers and does not establish throughput at eight workers.

The [measurement archive](20261123-worker-tuning.tar.gz) includes the workloads, responses and timing fields, engine timing logs and OpenCode request-field checks. Extract into an empty directory. The harnesses append JSONL results. For the verified OpenCode replays start `wire.py` before `reason.py --verified` with the supplied worker profile installed. `direct.py` and `cache.py` run independently against port 8001. `mixed.py PORT CHUNK SERIES` targets an already-running server. The `chunks.ps1` driver covers the stop, temporary launches and restore on this N1X installation.
