# AGENTS.md

These rules apply to the whole repository.

## Objective and scope

Complete the user's explicit deliverable within the applicable product and external contracts.
Choose a coherent solution with functional and numerical correctness, clear ownership, strong
architecture, and maximum performance at the requested scope. Do not sacrifice these goals to
reduce the diff or implementation effort. Evaluate complexity, maintenance cost, and verification
risk as engineering tradeoffs, not reasons to retain a known inferior design.

Before substantial work, identify the deliverable and its completion conditions. Work is relevant
when it completes that deliverable, preserves an applicable contract, resolves a material
uncertainty, or checks a realistic regression. A necessary redesign is in scope; unrelated cleanup,
hardening, compatibility, and benchmark campaigns are not. Address incidental findings when they
block the outcome or are inseparable from the selected implementation.

For implementation tasks, record the user's requirements, promised work, affected implementations
and routes, and the quality dimensions to be evaluated before judging results. For comparative
claims, also identify the baseline, workloads, metrics, aggregation method, and acceptance criteria.
Keep this accounting current as the task develops. Explain justified changes and preserve the
earlier findings; never narrow the scope or change the criteria after seeing unfavorable results
to make the task appear successful. This accounting may live in the working conversation or
existing task artifacts; it does not require a new permanent planning document.

For analysis or design, deliver the explanation or design. For diagnosis, establish the cause and
supporting evidence; implement a fix when requested. For implementation, complete the selected
design across its affected implementations, callers, tests, tools, and active documentation.

Requests for analysis or proposals for review do not authorize applying them;
implementation authorization must cover that same scope.

The current product and architecture govern ordinary work. An explicit task may change them;
update the affected contracts and implementation together instead of treating the current design
as an immutable prohibition. Skills provide task-specific methods, not additional deliverables or
approval requirements beyond the user's instructions and the actual execution environment.

## Product and architecture

Infernix is a from-scratch C++/CUDA inference engine for maximum single-GPU performance. It implements
`Qwen3_5ForCausalLM` and `Qwen3_5MoeForCausalLM`; official Qwen3.6/3.8 artifacts and user recipes
use the same architecture, binding and execution path. It also implements `Qwen4ExpForCausalLM`
(Qwen3.8-Flash-Next, model family `qwen4_exp`), whose routed experts are offloaded to host memory or
the SSD behind a VRAM expert cache. The implementation targets `sm_120a` and is tuned on NVIDIA
GeForce RTX 5090.

Generation uses one GPU, one resident model, one to eight resident execution lanes fixed at startup,
bounded FIFO ingress with finite bypass, resource-pressure preemption with Replay recovery,
one content-addressed prefix cache, and one compact decode batch per round.
Generation and offline CausalScoring use the same public `.infernix` Engine route. Delivered
capabilities and commands are documented in `README.md`, the product guides, and executable
`--help`. New mathematical architectures, execution platforms, large-scale continuous
batching, and priority/QoS require an explicit product change. Another training instance or mixture
of existing representations does not require a checkpoint-specific execution registration.

This is a local, single-owner project with trusted local models, generated artifacts, and
local workflow. Do not derive requirements from a different deployment or trust model.

Keep these ownership boundaries visible when selecting a design:

- The v3 container (`.infernix`, or `.ninfer` from NInfer) is the only C++ product artifact; CLI,
  serving, and inference benchmarks use the public Engine. Infernix has no Python model-inference
  route or installed/exported C++ SDK.
- Core owns physical primitives and raw transfers; artifact owns generic framing and
  materialization; Ops own closed mathematical and state-transition implementations.
- Models own fixed mathematics, config interpretation, logical parameter binding, frontend
  semantics and finite execution composition. Immutable Model data owns selected weights and
  resources; native Parameters supply the actual operands to planning and Program execution.
  Program owns mutable state, workspace, context stores and CUDA Graphs. Programs share no mutable
  state or device allocation.
- Converter recipes choose sources, formats, packing and per-input activation permissions. The
  loader validates, uploads and binds the stored representation. Native preparation, resource
  queries and execution enforce actual Op support; there is no whole-artifact capability registry.
- Runtime owns common execution contracts and Engine publication policy; product/serving own input
  acquisition and protocol translation. Model code does not acquire media or own transport.

Detailed model/runtime responsibilities and source ownership are defined in
[Engine architecture](docs/maintainer/engine-architecture.md). Read the relevant boundary before
changing it. Prefer explicit implementations for supported architectures. Do not introduce generic model
graphs, family base classes, plugin discovery, string-driven execution, hidden device allocation,
runtime weight repacking, or placeholders for hypothetical targets without a product requirement.

## Change consistency

Project-owned APIs, CLIs, Python tools, fixtures, reports, formats, and documentation do not preserve
backward compatibility. When replacing behavior, remove superseded aliases, fallbacks, transition
branches, and their tests within the affected contract. Leave unrelated paths alone.

Advertised OpenAI and Anthropic protocol behavior is an external contract. Changes update the
affected schema tests and serving documentation together.

Keep stable requirements in their existing active reference. Temporary plans are useful only for
active work; remove them when completed or abandoned. Maintain one current authority rather than
parallel `final`, `v2`, or `new-design` documents.

### Upstream syncs on the N1X fork

On the long-lived `n1x` branch, `origin` is the N1X fork and `upstream` is
`eugene-kamenev/Infernix-dgx-spark-gb10`. For a full sync, fetch `upstream/main` and merge it
directly into `n1x`. The merge records the GB10 tip as an ancestor for future syncs. Reserve
cherry-picks for a deliberately selective backport. Other branches and the fork's `main` can
stay at their own revisions. Verify a completed full sync with
`git merge-base --is-ancestor upstream/main n1x`.

Resolve conflicts against both the incoming engine changes and the N1X WSLC serving contract.
Use the [N1X refresh recipe](bench/n1x/README.md#refreshing-from-upstream) and check the affected
paths on N1X before publishing. Record both parent revisions in the dated GB10 changelog.

### GB10 branch changelog and rebase tracking (`docs/gb10/`)

To support the GB10 project's ongoing rebases onto its own upstream while preserving GB10 (`sm_121a`)
architectural adaptations, maintain dated changelog files in `docs/gb10/` following the pattern:

```text
docs/gb10/YYYYMMDD-changelog.md
```

(Example: `docs/gb10/20261028-changelog.md`). Whenever rebasing, completing a campaign milestone, or
introducing architectural changes:

1. **Commit range & accounting**: Record the base commit (`origin/main`), current HEAD, total files changed,
   insertions, and deletions. Exclude `cuda-kernel-toolkit/` from file-by-file enumeration (note it as an
   intentionally excluded integrated toolkit).
2. **Architectural rationale ("Why")**: Explicitly document the GB10 hardware invariants necessitating code
   divergence from upstream (48 SMs wave sizing, 128 GiB unified LPDDR5X UMA zero-staging / full device-resident
   MoE, Grace 20-core big.LITTLE ARMv9.2 CPU threading, 1:1 FP32 accumulation Tensor Core parity, `__vsub4`
   emulation / FMA contraction quirks, memory-bandwidth-bound decode amortization & speculative decoding parameters).
3. **Subsystem breakdown**: Group changed files by subsystem (`include/` & core runtime, ops & quantization,
   model & MoE offload, converter & tools, tests & benchmarks) with action codes (`A`/`M`/`D`) and specific
   mechanism explanations.
4. **Rebase conflict resolution policy**: State the default conflict resolution rules per subsystem (e.g.
   preserve GB10 unified memory defaults and device-resident MoE over upstream host-offloading; preserve GB10
   1:1 FP32 accumulation over FP16 accumulation; preserve GB10 sliced-K and launch bounds over upstream 170-SM
   constants; accept upstream math/config fixes with adaptations).

Task lists and roadmaps follow the companion naming pattern `docs/tasks/YYYYMMDD-<short-description>-tasks.md`
(e.g., `docs/tasks/20261029-gb10-post-rebase-perf-tasks.md`).

## Verification and completion

Select evidence to support the changed behavior and material claims. Tests should protect supported
observable behavior, mathematical or state semantics, and realistic regressions, including plausible
boundary failures that have not occurred yet. Avoid tests that merely mirror implementation,
freeze private file/class organization, or increase coverage numbers.

For numerical changes, identify represented public inputs, the independent mathematical oracle,
semantic cast/quantization/state boundaries, output criteria, and relevant real model shapes. Each
floating-point Op uses a naive FP32/FP64 oracle; exact transforms/codecs use an exact oracle. Packed
inputs are independently decoded with their stored scales. Qualify production routes directly
against that oracle, not another kernel or plausible model output. Private arithmetic need not
reproduce unfused materializations unless an intermediate is an observable semantic boundary.
[Op development](docs/maintainer/op-development.md) defines the full qualification contract.
The reproducible route (`--reproducible`) is deterministic and request-invariant on every artifact:
for a given engine build, artifact, options and GPU, a request (its input, sampling parameters and
seed) always gives the same output, whatever other requests, concurrency, cache state and resource
pressure. Its bits are not promised across engine versions: a change may alter them (a fix, a precision
change or a speed change) while the route stays deterministic and request-invariant and every changed Op
meets its criterion against its oracle; say so where it lands. A change that would make a request's output
depend on other requests or on scheduling, or that only a model-level quality gate can approve, ships as a
fast path, qualified under the
[numerical acceptance policy](docs/maintainer/model-quality.md#6-numerical-acceptance-policy-and-fast-paths)
for the kernels it changes (a configuration is the set of kernels an artifact runs, so artifacts
with the same kernels share a qualification) and registered only where it is measured faster.
The lossless route (`--lossless`, [definition](docs/maintainer/lossless-route.md)) computes every
operation of the model correctly rounded, so its output depends only on the request and the
artifact; each of its kernels is checked against an exact oracle, and no change may alter its bits.

Measure performance at the claimed scope. An Op microbenchmark establishes an Op result, not an
end-to-end improvement. Use whole-inference profiling when an in-scope end-to-end attribution is
unresolved; use kernel profiling when an identified kernel question can change the decision. Reuse
applicable evidence and stop collecting once the relevant alternatives can be distinguished.

Choose the affected checks, rather than running this table as a checklist:

| Change | Typical evidence |
|---|---|
| Documentation | affected links/references and `git diff --check` |
| C++ runtime/API | affected build targets and behavioral tests |
| Python tooling | Python 3.11 `py_compile` and affected tests |
| Artifact framing/binding/conversion | affected contract tests; real artifact when semantics require it |
| CUDA mathematics | independent oracle at relevant shapes and route boundaries |
| Memory or lifetime | affected execution; sanitizer for a concrete lifetime question |
| Performance | measurement at the claimed scope; profiling only for unresolved attribution |
| Serving | affected schema tests and observable request/stream behavior |

Record the target, relevant hardware/toolchain, workload or command, and summarized result needed
to interpret a material claim. Hashes, clean worktrees, full command transcripts, raw report
inventories, and exact probabilistic outputs are not default requirements. Use exact comparison for
exact outputs, and appropriate numerical or behavioral criteria otherwise. State checks that could
not run and their implications.

A throughput number is admissible only when the same work was verified to have been computed.
Tuning a capacity constant (tile width, row block, unit count, stage count) can reduce the work a
kernel performs instead of making it faster, which shows up as a large apparent speedup. Establish
the benchmark's units from its header, and re-derive the invariant that bounds the constant (warp
layout, fragment count, grid extent) before accepting such a result.

## Reporting and completion

Selective reporting and evidence gaming are prohibited, even when every disclosed
statement is individually true. For every implementation task:

1. Cover the entire agreed deliverable, its completion status, and all affected or
   evaluated dimensions: behavior, numerical semantics, interfaces, architecture,
   performance, resources, and maintenance. Distinguish completed, incomplete, and
   unverified work; never describe an unmeasured aspect as unchanged.

2. Put favorable and unfavorable findings in the final reply itself, including
   regressions, costs, rejected approaches, failures subsequently fixed, unresolved
   issues, and verification gaps. Explain their disposition. Group repetition
   without hiding distinct problems or exceptions. Small or unexplained adverse
   results must remain visible; attachments cannot substitute for disclosure.

3. Make comparisons representative and comparable. State the baseline, workload,
   conditions, metrics, coverage, outcome distribution, worst changes, and exceptions.
   Distinguish new capability, fallback replacement, and improvement to an optimized
   implementation. Keep claims within the measured scope; neither a best case nor
   an average may stand in for the full results.

4. Apply the same evidence standard to gains and regressions. Label uncertainty;
   do not dismiss slowdowns as noise without evidence. Explain changes to scope,
   baselines, methods, or acceptance criteria and preserve earlier adverse findings.
   Never change these choices to manufacture a favorable conclusion.

5. Reuse sufficient evidence. Additional or repeated checks must satisfy required
   verification, replace invalidated evidence, or resolve a concrete question that
   could change implementation or acceptance. Once the deliverable and acceptance
   conditions are satisfied, stop and report. Report review checks existing work
   and findings; it must not become a new audit, sweep, or reporting-tool project.
   Disclose remaining uncertainty without silently making it a new requirement.
   Disclosure does not excuse unmet completion conditions.

## Reference navigation

Read the authority relevant to the current decision; this is not a mandatory reading list.

| Decision | Entry point |
|---|---|
| Product capabilities and exact commands | `README.md`, executable `--help`; `docs/cli.md`, `docs/serving.md`, `docs/perplexity.md` |
| Execution, model/runtime ownership, scheduling, transactions, graphs | `docs/maintainer/engine-architecture.md` |
| Scheduling, permits, preemption; prefix cache; physical KV | `docs/maintainer/resource-scheduling-and-context-cache.md`; `docs/maintainer/hybrid-prefix-cache-spec.md`; `docs/maintainer/paged-kv-cache.md` |
| Artifact, layout, codec, conversion, or model mathematics | model/artifact references and conversion guide linked from `docs/README.md` |
| Op contracts, implementation ownership, numerical/performance qualification | `docs/maintainer/op-development.md` |
| Test/benchmark commands and published performance | `tests/README.md`, `bench/README.md`, `docs/performance.md`, `docs/gb10_roadmap.md` |
| GB10 branch changelogs and rebase tracking | `docs/gb10/YYYYMMDD-changelog.md` (e.g. `docs/gb10/20261028-changelog.md`) |
| Active task lists and optimization roadmaps | `docs/tasks/YYYYMMDD-<short-description>-tasks.md` |
| In-tree C++ interface | `include/infernix/engine.h`, `include/infernix/types.h` |

[Documentation map](docs/README.md) routes to narrower authorities when needed.

## Local operations

For the configured Ninja tree, the high-throughput default build command on this 20-logical-CPU
GB10 host is `cmake --build build-native --parallel "$(nproc)"` (or `ninja -C build-native -j "$(nproc)"`
directly to bypass CMake manifest checks). This rebuilds only stale outputs; do not clean or
reconfigure for ordinary source edits. For a focused change, build only its test or product target,
then run the matching test selection:

```bash
# 1. Fast compile check (~2.3s): validate C++/CUDA syntax/templates before linking
ninja -C build-native src/ops/CMakeFiles/infernix_ops.dir/linear/<qtype>/shapes/n<N>_k<K>.cu.o

# 2. Unified binary build: build bench, test, and infernix-serve in a single invocation to share device link
ninja -C build-native -j "$(nproc)" infernix_linear_bench infernix_linear_<qtype>_a16_test infernix-serve

# 3. Fast shape-targeted test (~2.1s vs 118s full suite): validate modified shape against oracle
./build-native/tests/infernix_linear_<qtype>_a16_test --shape <N> <K>

# 4. One-way CUDA stack-spill ratchet gate (verifies zero new STACK>0 or LOCAL>0 symbols on infernix-serve)
ctest --test-dir build-native -R infernix_cuda_stack_check --output-on-failure

# 5. Final end-to-end model A/B benchmark across KV dtypes and plain/MTP modes (after lower-level A/B passes)
./bench/run_kv_matrix_bench.sh /tmp/kvmatrix_after
```

Ninja compilation uses the requested parallelism, while the current CMake configuration also
allows up to `INFERNIX_LINK_JOBS=8` concurrent link/archive/device-link jobs. The default was
raised from strict serialization because a serialized link pool left the 20-core host about 63%
idle for 347 seconds during a measured 12-file change; the current CMake comment records 242
resulting link edges. Because `infernix_ops` is built with relocatable device code (`-rdc=true`),
`nvlink` (`cmake_device_link.o`) takes ~55.8s on this host; building multiple consumer targets
in a single Ninja invocation amortizes this device link so it runs only once. Do not mistake a
link phase with fewer processes for a stalled parallel build. If links exhaust memory, configure
`INFERNIX_LINK_JOBS=1`; lower build parallelism only when compilation causes actual memory pressure
or interferes with other work.

For the full configured test suite, let CTest use the host's available workers:

```bash
ctest --test-dir build-native --output-on-failure --parallel "$(nproc)"
```

CTest honors each registration's `RUN_SERIAL` property, running those tests alone while allowing
the other tests to overlap. In the current `build-native` configuration, 39 of 265 tests are
marked `RUN_SERIAL` (including real-artifact integrations and selected wide/resource-heavy GPU
tests). Real-artifact tests skip when their required artifact environment variables are unset.
Other GPU tests may still compete for device resources; if that is observed, reduce only CTest's
`--parallel` value and leave build parallelism unchanged. Inspect the active suite and scheduling
properties with `ctest --test-dir build-native --show-only=json-v1`.

Use a Python 3.11 interpreter explicitly (`python3` or the maintainer environment's
interpreter); the default `python`/`python3` may be a different version. Use it only after
selecting that environment or checking its version. Normal resources are the build trees
(`build/` from the CMake presets), `out/` for converted artifacts and their `.conversion.json`
reports, and `profiles/ncu/`, `profiles/nsys/`, `profiles/bench/`; the toolchain is CUDA 13 on
64-bit Linux aarch64, which is the only supported platform.
Select model artifacts by explicit path, never glob order, modification time, or unqualified
“latest”. Source checkpoints and large artifacts are prerequisites; download or regenerate them
only when that work is in scope. Install or upgrade dependencies only when the task needs it.

Inspect compiled kernels per function, never by dumping a whole binary. `cuobjdump -res-usage <linked-target>`
takes about 1.1 s and yields the exact stored symbol names together with `REG`/`STACK`/`SHARED`/`LOCAL`;
`cuobjdump -sass -fun <exact symbol> <linked-target>` then takes about 1.6 s for one kernel. An unfiltered
`cuobjdump -sass` on a 412 MB linked bench executable emitted 4.53 GB in 132.8 s. `--function` matches the
stored symbol exactly, so file-static kernels carry their `__nv_static_<hash>_<file>_<id>__` prefix, and a
short or mangled-only name returns exit code 0 with no instructions — verify a nonzero instruction count
before drawing any conclusion. This build's `cuobjdump` has no `regex:` function filter.

Create commits only when requested. Use Conventional Commit subjects with concise lowercase types
such as `feat`, `fix`, `perf`, `bench`, `test`, `build`, `refactor`, `docs`, or `chore`.

## GPU kernel engineering skills, `cuda-kernel-toolkit`, and multi-agent orchestration

For GPU kernel work (new kernels, kernel optimization, `/shapes/` tuning, profiling, or numerical
repair), read `.github/copilot-instructions.md`, `cuda-kernel-toolkit/README.md`,
`cuda-kernel-toolkit/docs/usage.md`, and the skills under `.github/skills/`: architecture discovery
for every supplied target GPU, kernel optimization for the end-to-end loop, kernel repair for
failure diagnosis, validation/benchmarking for correctness and honest timing, CUDA profiling for
NCU/NSYS evidence, experiment memory for candidate history, and the task-specific `cuda-*` design
playbooks. They provide methods, not additional deliverables, approval requirements, or ceilings
on the work; the product and architecture rules above still govern. In particular, profile NCU
(`ncu`) only on synthetic inputs and synthetic parameters — never load real model weights in an
NCU-instrumented process — while NSYS (`nsys`) may trace a real model invocation.

Always use the in-tree `cuda-kernel-toolkit` (`PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt`)
instead of writing ad-hoc profiling/timing scripts or doing mental arithmetic:

- **Zero hand math (`ckt math`)**: Never estimate occupancy, warp-register rounding steps, shared-memory
  bank conflicts, CTA wave efficiency, quantized weight bytes, L2 residency, or A/B statistics by hand.
  Run `ckt math {occupancy,waves,tile,smem,workload,l2,vector,roofline,mlp,ab}`.
- **Live hardware & 21-pipe precision ceilings (`ckt env`)**: Queries `cudaDeviceProp` and measures all
  21 SIMT and Tensor Core pipes (`bandwidth`, `shared_mem_bw`, `fp32`, `fp64`, `int32`, `fp16_simt`,
  `bf16_simt`, `tf32`, `fp16_f32acc`, `fp16_f16acc`, `bf16`, `int8`, `int4`, `fp8_e4m3`, `fp8_e5m2`,
  `mxfp8`, `mxfp6`, `mxfp4`, `nvfp4`), cached in `.ckt_cache/env.json`.
- **Per-symbol SASS & resource inspection (`ckt static --sass` & `ckt compare`)**: Extracts exact
  stored symbols (`REG`, `STACK`, `SHARED`, `LOCAL`, and SASS instruction mix including `IMMA`, `HMMA`,
  `QMMA`, `OMMA`, `LDG.128`, `LDS.128`, `STL`/`LDL` spills) in ~1.5 s without dumping whole binaries.
- **Fast `/shapes/` discovery and cold-L2 benchmarking (`ckt shapes {list,inspect,bench}`, `tools/gb10_shape_sweep.py`, `tools/gb10_nvfp4_a4_sweep.py`)**:
  Parses all `src/ops/linear/*/shapes/n<N>_k<K>.{cu,cpp}` files, audits spills and `__launch_bounds__`,
  and benchmarks them via `infernix_linear_bench` with a GB10-right-sized 32–48 MiB L2 flush (5x–8x faster
  than the 256 MiB RTX 5090 default) and route-boundary token probing (`[B, B+1]`), automatically flagging
  route-transition cliffs or sweeping new GB10-native tile schedules.
- **Interleaved A/B testing (`ckt ab`), SQLite hotspot ranking (`ckt hotspots`), & final E2E KV-matrix A/B (`bench/run_kv_matrix_bench.sh`)**:
  Runs round-robin interleaved GPU commands (`--cmd-a`, `--cmd-b`), env toggles (`--env-a`, `--env-b`), or
  shape CSV comparisons (`--csv-a`, `--csv-b`), ranks NSYS `.sqlite` traces with GB10 occupancy modeling,
  and concludes every multi-op or campaign-level optimization with `./bench/run_kv_matrix_bench.sh <out-dir>`
  to verify end-to-end `prefill t/s`, `decode t/s`, and MTP `acceptance_rate` on real Qwen3.8-Flash-Next
  artifacts (`iq3s`, `dense8`, `nvfp4`) across KV dtypes and `plain`/`mtp` modes.

When operating with `.github/agents/`, follow the Parent–Child hierarchy:

- **Parent Orchestrators (`cuda-kernel-orchestrator.agent.md` and `cuda-kernel-optimizer.agent.md`)**:
  Own goal tracking, dispatch child subagents, enforce lower-level op A/B gates followed by the final
  `./bench/run_kv_matrix_bench.sh` end-to-end A/B benchmark, and recover interrupted work.
- **Child Subagents (`agents: []`, `user-invocable: false` — cannot spawn nested subagents)**:
  - `cuda-source-auditor.agent.md` (CPU-only, parallelizable): source audit, `ckt static --sass`,
    `ckt compare`, `ckt math`, `ckt shapes inspect`, `ckt hotspots`.
  - `cuda-kernel-implementer.agent.md` (CPU-only, parallelizable): single-mechanism edits, fast `.cu.o`
    compile checks, unified Ninja build, post-build `ckt compare` and `ckt math` gatekeeper checks, and repair.
  - `cuda-gpu-benchmarker.agent.md` (Exclusive GPU Owner, strictly serialized — 1 instance at a time):
    `ctest` (including `infernix_cuda_stack_check`), `compute-sanitizer`, `ckt shapes bench`,
    op-level `infernix_*_bench` + `ckt ab`, `ckt sweep`, synthetic-only `ckt profile`, and final end-to-end
    `./bench/run_kv_matrix_bench.sh` A/B validation.
- **Subagent recovery protocol**: If a child subagent stops before completing its task, the Parent
  Orchestrator inspects the workspace and re-invokes the subagent with:
  ```text
  The previous subagent stopped before completing its task.

  Inspect the current workspace and determine:
  1. What it already implemented.
  2. What remains unfinished.
  3. Whether any tests are failing.

  Continue from the existing changes. Do not redo completed work.
  Finish the task, run the relevant tests, and report the result.
  ```

## Waiting for background tasks

Never use `sleep` or a busy-wait loop (fixed delays, `sleep N` polling loops, `while ...; do :; done`, or `sleep` inside retry loops) to wait for a
build, benchmark, test, or other background process. A fixed delay either wastes time or returns
before the work finished. Wait on the process itself and return the moment it exits.

Prefer waiting on the exact PID with `tail --pid` (blocks without spinning the CPU): several processes can share a name (parallel
`ninja`, `python3`, `bash`, other agents' runs), so name-based matching can wait on the wrong one
or never terminate.

```bash
# Preferred: start the task, capture its PID, wait for exactly that process
./bench/run_kv_matrix_bench.sh /tmp/out > /tmp/out.log 2>&1 &
PID=$!
tail --pid="$PID" -f /dev/null

# Already-running process: obtain its PID once, then wait on the PID
PID=$(pgrep -n -f '[r]un_kv_matrix_bench')    # -n newest, -o oldest; verify it is the right one
tail --pid="$PID" -f /dev/null

# Fallback only when the PID is unobtainable and the name is unique:
PID=$(pgrep -x ninja) && tail --pid="$PID" -f /dev/null
```

Rules:

- Use `tail --pid="$PID" -f /dev/null` by default. It blocks until the
  process exits, uses no CPU, and works for any PID (not only children of the shell).
- After the wait ends, check the result (exit code via `wait "$PID"` when it is a child of the
  same shell, log tail, expected output file); process disappearance alone does not prove success.
- When resolving a PID with `pgrep -f`, use a pattern that cannot match the waiting shell itself
  (e.g. the `[r]un_kv_matrix_bench` bracket trick) and confirm a single intended match
  (`pgrep -af`) before waiting.
- Use name-based `pgrep -x ...` only as a fallback, and only when no other process of that
  name can exist.
- Prefer the tool's native completion notification for background commands; use `tail --pid` only
  when a synchronous wait on an already-running process is required.
