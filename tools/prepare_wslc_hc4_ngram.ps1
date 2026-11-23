[CmdletBinding()]
param(
    [string]$NgramVolume='infernix-hc4-ple-a78ba00a',
    [string]$ModelsDirectory=(Join-Path $env:USERPROFILE 'models'),
    [string]$Image='infernix:n1x',
    [switch]$PrintOnly
)
$ErrorActionPreference='Stop'
$model='Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix-HC4-W4Dense'
$file='Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix.ngram'
$source=Get-Item -LiteralPath (Join-Path $ModelsDirectory "$model/$file")
$stage=@'
set -eu
src="/src/$1"
dst="/ple/$1"
if test -e "$dst"; then
    cmp -s "$src" "$dst" || { echo "Volume contains a different n-gram table" >&2; exit 1; }
else
    tmp=$(mktemp /ple/.ngram.XXXXXXXX)
    trap 'rm -f "$tmp"' EXIT
    cp "$src" "$tmp"
    chmod 644 "$tmp"
    ln "$tmp" "$dst" || cmp -s "$src" "$dst"
fi
'@
$arguments=@('run','--rm','--pull','never','--mount',"type=bind,source=$($source.DirectoryName),target=/src,readonly",'--mount',"type=volume,source=$NgramVolume,target=/ple",'--entrypoint','/bin/sh',$Image,'-c',$stage.Replace("`r`n","`n"),'stage',$file)
if($PrintOnly){$arguments|ConvertTo-Json;return}
& wslc volume inspect $NgramVolume 2>$null | Out-Null
if($LASTEXITCODE){
    & wslc volume create $NgramVolume
    if($LASTEXITCODE){
        & wslc volume inspect $NgramVolume | Out-Null
        if($LASTEXITCODE){throw "Could not create or open $NgramVolume"}
    }
}
& wslc @arguments
if($LASTEXITCODE){throw 'Could not stage the model n-gram'}
