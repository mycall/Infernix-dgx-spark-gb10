# Start the HC4-W4Dense model on N1X with an existing CUDA 13.5 WSLC image
# and a Linux volume containing the model's .ngram file.
# Example: .\tools\run_wslc_hc4_dflash2.ps1 -NgramVolume infernix-hc4-ple-a78ba00a
[CmdletBinding()]
param(
    [string]$NgramVolume = 'infernix-hc4-ple-a78ba00a',
    [string]$ModelsDirectory = (Join-Path $env:USERPROFILE 'models'),
    [string]$Image = 'infernix:n1x',
    [string]$Name = 'infernix-n1x-hc4',
    [ValidateRange(1, 65535)][int]$Port = 8001,
    [ValidateRange(1, 8)][int]$MaxConcurrency = 8,
    [switch]$PrintOnly
)

$ErrorActionPreference = 'Stop'
$model = 'Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix-HC4-W4Dense'
$drafter = 'Qwen3.8-Flash-Next-DFlash2-Infernix'
$weights = Get-Item -LiteralPath (Join-Path $ModelsDirectory "$model/$model.infernix")
$draftWeights = Get-Item -LiteralPath (Join-Path $ModelsDirectory "$drafter/$drafter.infernix")
$template = Get-Item -LiteralPath (Join-Path $PSScriptRoot 'chat_templates/qwen3_8.jinja')
if (-not $PrintOnly) {
    & wslc volume inspect $NgramVolume | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Could not open WSLC n-gram volume '$NgramVolume'." }
}

# Captured DFlash2 rounds on N1X/WSLC produced non-finite logits in sampled
# generation. Eager CUDA keeps K3 draft verification and the model's fast paths.
$arguments = @(
    'run', '-d', '--name', $Name, '--gpus', 'all', '--pull', 'never',
    '--mount', "type=bind,source=$($weights.DirectoryName),target=/model,readonly",
    '--mount', "type=bind,source=$($draftWeights.DirectoryName),target=/draft,readonly",
    '--mount', "type=bind,source=$($template.DirectoryName),target=/templates,readonly",
    '--mount', "type=volume,source=$NgramVolume,target=/ple,readonly",
    '-p', "127.0.0.1:${Port}:8002",
    $Image, 'infernix-serve', "/model/$($weights.Name)", '--model-id', 'infernix-hc4',
    '--chat-template', "/templates/$($template.Name)",
    '--ngram-volume', '/ple/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix.ngram',
    '--ngram-residency', 'volume', '--ngram-cache-mib', '512',
    '--spec', 'dflash2', '--draft-model', "/draft/$($draftWeights.Name)",
    '--draft-tokens', '3', '--draft-tree-nodes', '0', '--no-cuda-graph',
    '--kv-dtype', 'int8', '--max-context', '16384', '--kv-capacity', [string][Math]::Min(65536, 16384 * $MaxConcurrency),
    '--host-context-mib', '1024', '--max-concurrency', [string]$MaxConcurrency, '--prefill-chunk', '4096',
    '--default-max-tokens', '128', '--host', '0.0.0.0', '--port', '8002'
)
if ($PrintOnly) { $arguments | ConvertTo-Json; return }
& wslc @arguments
if ($LASTEXITCODE -ne 0) { throw "Could not start WSLC container '$Name'." }
