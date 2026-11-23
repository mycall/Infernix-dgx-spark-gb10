# Windows ARM64 N1X serving, November 23

The first n1x release builds on origin/main at f65e1a776628ac427ccf46719d601211679e0062. The release commit is the commit introducing this file (`git log --diff-filter=A --format=%H -- docs/gb10/20261123-changelog.md`). The branch adds a WSLC serving recipe and preserves the earlier trials from ad1f9f94. Release diff against the base: 17 files changed, 591 insertions(+), 3 deletions(-). The integrated cuda-kernel-toolkit is outside this enumeration.

| Subsystem | Files and rationale |
| --- | --- |
| Build | A Dockerfile.n1x builds the ARM64 CUDA 13.5 server for sm_121a. |
| Tools | A tools/run_wslc_hc4_dflash2.ps1 and tools/prepare_wslc_hc4_ngram.ps1 select the HC4 artifacts and stage the external n-gram on the WSLC disk. M tools/chat_templates/qwen3_8.jinja corrects serialization of tool messages. |
| Benchmarks | A bench/n1x contains four summary CSVs, the raw benchmark and bring-up archives, the dual-drafter trial patch and its provenance. M .gitattributes preserves unified diff context. |
| Documentation | M README.md, docs/serving.md and docs/maintainer/deepseek_v4_1-flash-design.md. A this changelog records the branch boundary. |

The N1X recipe uses the existing GB10 implementation. The 48-SM launch sizing, unified-memory expert placement, ARM worker tuning, FP32 tensor accumulation, packed integer emulation, FMA handling and bandwidth-sensitive decode kernels remain owned by the GB10 upstream. The archived dual-drafter experiment applies only to its recorded base. Keep it out of the production build during refreshes. Merge upstream math, runtime and kernel fixes through the GB10 branch. Resolve N1X packaging, artifact layout, template and launch conflicts against the documented WSLC serving checks.
