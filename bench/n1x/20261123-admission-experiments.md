# Admission and parallel worker budgets on N1X

Two clients submitted separate warehouse archives with 250,001 input tokens each. Each client had three tasks. Each task required the correct tool and SKU and an exact answer combining an early archive fact, the tool result and a late archive fact. The scorer stopped a session at its first failed task. All long-context cases used DFlash2 K3 and a maximum output of 8,192 tokens per request. The machine was on the 8 GiB dedicated partition with driver 621.24. The model and n-gram were loaded before timing.

## Full histories

At the daily 30-second deadline, the [earlier 8 GiB run](20261123-partition-comparison.md) completed 3/6 tasks in 112.28 s. One client received HTTP 503 and its remaining turns never ran. Raising only the pending deadline to 300 seconds gave 6/6 in 1,045.41 s. The first cold tool call reached its first token in 93.30 s and the second in 180.65 s. The second request spent 96.3 s queued before admission. All 375 decode rounds had one participating request. Every request after the first reused 24,576 tokens of its approximately 250K-token input. The server repeatedly processed the rest of each history as the clients alternated.

Each request receives the admission deadline independently. A multi-turn task can exceed 300 seconds overall even when every request is admitted on time.

## Carrying context into a worker

The harness rendered the same full input, then kept a suffix sized with the server tokenizer. It preserved the system message and tools. The early fact was at 10% of the original archive and the late fact at 90%. Both suffixes retained the late fact. In the pinned case the harness copied the original early fact verbatim ahead of the suffix. Subsequent tool calls and assistant turns were appended normally. This is a client-side handoff experiment. It does not modify the attention mechanism or identify arbitrary important facts automatically.

| Input per worker at the first turn | Correct / planned | Group wall time | First token, two clients | Decode rounds with two active rows |
| --- | ---: | ---: | --- | ---: |
| Full 250,001 tokens | 6/6 | 1,045.41 s | 93.30 / 180.65 s | 0 of 375 |
| Last 31,998 tokens | 0/6 | 93.02 s | 23.27 / 23.26 s | 173 of 918 |
| 31,997 tokens including the pinned early fact | 6/6 | 124.29 s | 23.26 / 23.26 s | 154 of 169 |

Each suffix-only session failed its first final answer, leaving four planned tasks unattempted. All three cases used a 300-second admission deadline and the same engine instance, in table order. Each case ran once. The 124-second result measures a task whose two required facts were deliberately preserved. It applies to workers that can operate on that smaller input.

## Incremental KV allocation on short workers

In the HC4 program, `binding_pages` calculates how many KV pages to reserve when admitting a request. With preemption off, the reservation covers the prompt and effective maximum output up to the context and pool limits. With preemption on, it covers the current prefill frontier and headroom for one round. As the request advances, the engine can grow that reservation. The explicit `--fast-path preemption` switch increased the number of selected paths from six to seven. The runtime identified this model configuration as NVFP4 dense, 16-bit expert activations and INT8 KV. [Reservation calculation](../../src/models/qwen4_exp/program/program_impl.h) · [Path selection](../../src/runtime/engine/qwen4_exp_instance.cpp)

Two simultaneous workers each requested a file, received a synthetic Python source file, and returned a verified ordered list of its 24 public functions. Each worker made two requests. Selection prompts occupied about 500 tokens and final prompts about 1,870. The workload alternated output allowances in the order 256K, 8K, 8K, 256K. Each paired run used identical inputs and completion token counts across the two server configurations. Both used the 300-second deadline; maximum pair wall time was under nine seconds. The preemption run also enabled a structured request log for inspection.

| Output allowance per worker | Default engine path, pair wall time | Explicit preemption, pair wall time | Two-row rounds, default / explicit |
| --- | --- | --- | --- |
| 256,000 | 7.439 s, 8.282 s | 5.976 s, 6.867 s | 0 / 45 in each trial |
| 8,192 | 5.914 s, 6.228 s | 5.956 s, 6.205 s | 45 / 45 in each trial |

All workers produced the exact expected lists. At the high limit both trials gained overlapping decode with explicit preemption and saved about 1.4 seconds per pair. At the smaller limit, both configurations already batched and had similar timings. The measured benefit came from admitting the pair with smaller initial reservations. The forced-path engine recorded zero actual pressure preemptions during the short trials and the following OpenCode test.

An OpenCode CLI parent then dispatched two built-in Explore subagents simultaneously using the ordinary Infernix model configuration, with only the endpoint redirected to the experimental server. Both workers requested up to 256,000 output tokens, read their respective local files and returned the correct markers on the first attempt. The server recorded 65 two-row decode rounds. These runs establish short-worker overlap. Correct recovery during forced eviction, long-output fairness, and the two 250K histories with preemption enabled were outside the test. The earlier 96 GiB vLLM long-context result and this 8 GiB experiment also used different GPU partitions.

The [measurement archive](20261123-admission-experiments.tar.gz) includes the server arguments, short-worker harness, an incremental patch for the [8 GiB near-full benchmark client](20261123-native-context.md), request traces, exact-answer scores, scheduler counters and OpenCode events. The 32K patch takes `BENCH_KEEP_SESSION_LINE=1` for the pinned case. Both shorter cases start from the fully calibrated 250K archive. The host-side script `parallel-output.py` takes the server URL and writes four trial results into its working directory.
