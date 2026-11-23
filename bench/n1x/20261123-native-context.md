# Native-context comparison on RTX Spark N1X

On November 23, 2026, Infernix served its native 262,144-token context with the GPU partition set to 8,192 MiB (driver 621.24). The launcher was at `e3691012`, the n1x-ai client at `38cd442`, and the GPU used the local ARM64 CUDA 13.5 build. The HC4 artifact used DFlash2 K3, eager execution, eight lanes, a shared 262,144-token INT8 KV pool, 4,096-token prefill chunks and a 1 GiB host prefix cache. A boundary request completed with 262,143 input tokens and one output token. A separate 107,408-token prompt returned the requested answer.

## Scores

| Workload | Infernix, 8 GiB | Reference |
| --- | --- | --- |
| Single-client inventory, correct tool call and exact final answer | 2/2. Scored follow-ups took 8.315 and 5.194 s. | qwen38fn-best, 2/2 in 6.732 and 5.195 s. |
| Inventory setup, excluded from the scored times | 6.507 and 3.385 s. | qwen38fn-best, 5.970 and 6.109 s. |
| Two clients, 250K initial prompts, three successive inventory/price lookups each | 3/6 planned tasks correct in 112.28 s. The second session timed out in the admission queue before its first tool call. | qwen38fn-best at 96 GiB, 24/24 planned tasks correct in 1,046.20 s over two repeats of four histories. |
| Fixed 128-token counting output, six runs | Median server decode 77.92 tokens/s, range 75.46–79.45. All six outputs matched the archived text exactly. | Earlier selected Infernix eager K3 recipe, median 77.97 tokens/s, range 75.99–80.16. This control has no matched qwen38fn-best score. |

The short follow-ups generated 166 tokens on each Infernix trial and 129 and 128 tokens on the historical engine. Infernix's first tool selection had zero cache hits. Its other three scored requests reused 8,192 tokens each. The historical engine reused 8,192 tokens on all four scored requests. The first follow-up was slower on Infernix and the second tied to two decimal places. Including setup, total times were 14.82 and 8.58 s on Infernix and 12.70 and 11.30 s on qwen38fn-best.

Both long inputs calibrated to 250,001 tokens. The admitted client's three correct tasks took 103.95, 4.57 and 3.75 s. The second client's first request returned HTTP 503 after about 31 seconds under the 30-second pending-request timeout. The harness stops that history on the first error. One request failed at admission and two planned follow-ups never ran. Each turn of the admitted history selected the correct tool and SKU and returned the correct two archive codes and tool value. Comparing group wall times alone favors the run that dropped a history.

The historical 96 GiB campaign used W4A16 AutoRound, DFlash2 K3, a 15 GiB KV cache, an 8,192-token prefill batch and LMCache. The short baseline is from the archived profile results. Its per-trial partition telemetry is unavailable here. Engine, quantization, cache state and scheduling differ. Isolating the partition effect requires repeating the Infernix run at 96 GiB with the same image, settings and work. Record cache hits and first-attempt failures on both runs.

## Single-client inventory confirmation

At 16:21 UTC on November 23, GPU 0 reported 8,192 MiB. The live image was `a772ac9c0af1`. Six consecutive first-attempt trials alternated the two archived inputs using the replay script from the 8 GiB run archive. Every trial required the correct inventory tool and SKU, followed by the exact session code and returned count.

| Trial | Setup (s) | Scored follow-up (s) | Setup cached tokens |
| ---: | ---: | ---: | ---: |
| 1 | 8.811 | 5.288 | 0 |
| 2 | 3.407 | 5.293 | 8,192 |
| 3 | 3.422 | 5.256 | 8,192 |
| 4 | 3.434 | 5.247 | 8,192 |
| 5 | 3.453 | 5.279 | 8,192 |
| 6 | 3.447 | 5.330 | 8,192 |

Infernix passed 6/6, with a median scored time of 5.283 s. Each scored request reused 8,192 tokens. The tool selection and final answer together generated 166 tokens per trial. The two archived qwen38fn-best trials also reused 8,192 tokens on each scored request and passed 2/2, taking 5.195 and 6.732 s with 128 and 129 generated tokens respectively. These six repeats measure warm follow-up consistency on the same task. The stock qwen38fn-best launcher requires 98,304 MiB, so its scores here are historical.

## Measurement records

The [8 GiB run archive](20261123-native-context-8g.tar.gz) contains the requests, replies, scores, timings, hardware observation, server events and client adaptation. The short runner reuses the original inventory replay and its archived input messages. It supplies current assistant turns and tool-call IDs at runtime. The near-full runner uses n1x-ai's streaming client and original first-attempt scorer. Its adapter selects Infernix, omits the vLLM cache salt, calibrates with Responses input-token counts, verifies cold counts against actual usage, and uses the server-reported prompt length on continuations. The historical harness separately counted continuations before submitting them. vLLM-only scheduler counter fields in the adapted client do not measure Infernix preemptions.

The [historical short scores](20261122-n1x-best-results.csv) and n1x-ai's `wiki/data/campaign96-summary.json` (configuration `kv15-on-8k-retained`, concurrency 2) supply the cross-engine baselines. The six fixed-output controls come from the `dffasteager-*-fixed128.json` records in [the earlier run archive](benchmark-history.tar.gz).

For a partition repeat, unpack the 8 GiB archive into a new working directory. Run its `replay.ps1` with `-Variant repeat -Trial 0 -Repeat 0` and then trial/repeat 1, both with `-Root http://127.0.0.1:8001`. Copy n1x-ai's `benchmarks/Client` from commit `38cd442` to a scratch directory and apply `LongTools.patch` there with `git apply --ignore-space-change`. Supply a fresh host metadata JSON with the verified partition and a new output path. Run the client in Release with `near-full --url http://127.0.0.1:8001 --host <metadata> --out <new-output> --target 250000 --concurrency 2 --sessions 2 --turns 3 --repeats 1`. Retain its scores even on a nonzero qualification exit. Finally, run `fixed.ps1` for trials 0 and 1 with the same root and variant. Each invocation issues three requests.
