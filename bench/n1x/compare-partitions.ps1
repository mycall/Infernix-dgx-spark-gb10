param(
    [Parameter(Mandatory)][string]$OutDir,
    [string]$Root = 'http://127.0.0.1:8001'
)
$ErrorActionPreference = 'Stop'
if (Test-Path -LiteralPath $OutDir) { throw "Choose a new output directory: $OutDir" }
$destination = (New-Item -ItemType Directory -Path $OutDir).FullName
$archive = Join-Path $PSScriptRoot '20261123-partition-comparison.tar.gz'
tar -xzf $archive -C $destination fixed.ps1 replay.ps1 responses.jsonl
if ($LASTEXITCODE) { throw 'Could not extract the reference workloads' }

$os = Get-CimInstance Win32_OperatingSystem
$gpu = nvidia-smi --id=0 --query-gpu=name,driver_version,memory.total --format=csv
if ($LASTEXITCODE) { throw 'Could not read GPU telemetry' }
$models = Invoke-RestMethod "$Root/v1/models"
if ($models.data.id -notcontains 'infernix-hc4') { throw 'Expected the infernix-hc4 profile' }
$metrics = (Invoke-RestMethod "$Root/metrics") -split "`n"
if ($metrics -match '^infernix_requests_(running|waiting|paused) [1-9]') { throw 'Inference server is busy' }
$metrics | Set-Content "$destination/start.prom"
$container = (wslc inspect infernix-n1x-hc4-256k -f json | ConvertFrom-Json)[0]
if ($LASTEXITCODE) { throw 'Could not identify the Infernix container' }
$container | Select-Object Image,@{Name='Command'; Expression={$_.Config.Cmd}} |
    ConvertTo-Json -Depth 5 | Set-Content "$destination/engine.json"
[ordered]@{
    started = (Get-Date).ToString('o')
    boot = $os.LastBootUpTime.ToString('o')
    windows_visible_kib = $os.TotalVisibleMemorySize
    windows_free_kib = $os.FreePhysicalMemory
    power_scheme = (powercfg /getactivescheme)
    ac = (Get-CimInstance -Namespace root/wmi -ClassName BatteryStatus -ErrorAction SilentlyContinue).PowerOnline
    gpu = $gpu
    root = $Root
    models = $models
    reference_archive_sha256 = (Get-FileHash $archive).Hash
} | ConvertTo-Json -Depth 10 | Set-Content "$destination/host.json"
(Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage','\GPU Adapter Memory(*)\Shared Usage').CounterSamples |
    Select-Object Path,CookedValue | Export-Csv "$destination/adapter.csv"

$telemetry = Start-Process nvidia-smi -ArgumentList '--id=0','--query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu,clocks.sm,clocks.mem,power.draw','--format=csv','-l','1' -WindowStyle Hidden -RedirectStandardOutput "$destination/gpu.csv" -PassThru
function Invoke-Inventory([string]$Variant, [int]$Trial) {
    & "$destination/replay.ps1" -Root $Root -Variant $Variant -Trial $Trial -Repeat ($Trial % 2) | Out-Null
    $result = Get-Content -Raw "$destination/$Variant-$Trial.json" | ConvertFrom-Json
    if (!$result.correct) { throw "Incorrect inventory result on $Variant trial $Trial" }
}
function Invoke-Counting([string]$Variant, [int]$Trial) {
    & "$destination/fixed.ps1" -Root $Root -Variant $Variant -Trial $Trial | Out-Null
    $result = Get-Content -Raw "$destination/$Variant-$Trial-fixed128.json" | ConvertFrom-Json
    if ($result.rows.Count -ne 3) { throw "Incomplete counting trial $Trial" }
    foreach ($row in $result.rows) {
        if ($row.response.timings.predicted_n -ne 128 -or $row.response.choices[0].finish_reason -ne 'length') {
            throw "Unexpected output length on $Variant counting trial $Trial"
        }
    }
}
try {
    Invoke-Inventory warmup 0
    Invoke-Inventory warmup 1
    Invoke-Counting warmup 0
    foreach ($block in 0..3) {
        $first = $block * 2
        if ($block % 2 -eq 0) { Invoke-Counting measured $first; Invoke-Counting measured ($first+1) }
        foreach ($trial in (($block*3)..($block*3+2))) { Invoke-Inventory measured $trial }
        if ($block % 2 -eq 1) { Invoke-Counting measured $first; Invoke-Counting measured ($first+1) }
        "Completed block $($block+1) of 4"
    }
}
finally {
    if (!$telemetry.HasExited) { $telemetry.Kill(); $telemetry.WaitForExit() }
}
(Invoke-RestMethod "$Root/metrics") | Set-Content "$destination/end.prom"
