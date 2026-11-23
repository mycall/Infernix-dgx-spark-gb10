# N1X branch and measurements

The production recipe builds the ARM64 CUDA 13.5 server with `Dockerfile.n1x`. See [serving on N1X](../../docs/serving.md#hc4-w4dense-with-dflash2-on-n1x-under-wslc) for artifact layout, n-gram preparation and the loopback endpoint. Model weights and the 29,127,258,112-byte n-gram are external inputs. The Docker build creates its Ninja tree inside the build stage and needs no previous host build, WSLC image, or temporary checkout.

## Refreshing from upstream

```powershell
git clone --branch n1x https://github.com/mycall/Infernix-dgx-spark-gb10.git
cd Infernix-dgx-spark-gb10
git remote add upstream https://github.com/eugene-kamenev/Infernix-dgx-spark-gb10.git
git fetch upstream main
git merge upstream/main
wslc build -f Dockerfile.n1x -t infernix:n1x .
```

Resolve any conflicting upstream changes on the branch and test before publishing. The first N1X branch base is `f65e1a77`. The November 22–23 measurements used `ad1f9f94` on an RTX Spark N1X.

## Preserved experiments

- [Explore worker effort, tool-prefix caching and mixed-load prefill sizes](20261123-worker-tuning.md)

- [Two-worker admission, context retention and preemption on N1X](20261123-admission-experiments.md)

- [Same-image 8 versus 96 GiB partition comparison](20261123-partition-comparison.md)

- [Native 262K context at 8 GiB versus qwen38fn-best history](20261123-native-context.md)

- [Inventory task history versus qwen38fn-best](20261122-n1x-best-results.csv)
- [Concurrency and prefix-cache results](20261123-n1x-drafter-grid-summary.csv)
- [Live drafter switch and lane-cap trials](20261123-n1x-mixed-live-measurements.csv)
- [DFlash2 K3 versus K5](20261123-n1x-dflash-k3-k5-results.csv)

The single-session historical inventory follow-ups took 5.17 and 5.30 s on the selected Infernix recipe, versus 5.19 and 6.73 s on qwen38fn-best. The engines used different quantization, cache settings and completion lengths. These results describe that task on the earlier revision.

Extract `tar -xzf bench/n1x/benchmark-history.tar.gz` at the repository root to restore the original ignored `profiles/bench` runs, including commands, prompts, HTTP responses, traces and intermediate failures. Historical scripts retain their original machine paths. The complete isolated experiment is preserved in `dual-trial-ad1f9f94.patch` against its original revision, including the alternate cache boundary. Dual residency remains an experiment. The serving launcher runs DFlash2 alone.

The original WSLC build recipe, GPU-memory probe source and output, build and launch logs, and first completion are in `wslc-bringup.tar.gz`.
