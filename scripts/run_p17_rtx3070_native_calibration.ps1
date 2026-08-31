param(
  [string]$OutputRoot = "validation/p17/native_rtx3070",
  [int]$Warmup = 100,
  [int]$Iterations = 1000
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Nvcc = if ($env:HETEROSIM_NVCC) {
  $env:HETEROSIM_NVCC
} elseif ($env:CUDA_PATH) {
  Join-Path $env:CUDA_PATH "bin/nvcc.exe"
} else {
  "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin/nvcc.exe"
}
if (-not (Test-Path -LiteralPath $Nvcc)) {
  throw "nvcc was not found; set HETEROSIM_NVCC or CUDA_PATH"
}

if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
  $VsWhere = "C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe"
  if (-not (Test-Path -LiteralPath $VsWhere)) {
    throw "cl.exe was not found and vswhere.exe is unavailable"
  }
  $VsRoot = & $VsWhere -latest -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
  $VcVars = Join-Path $VsRoot "VC/Auxiliary/Build/vcvars64.bat"
  if (-not (Test-Path -LiteralPath $VcVars)) {
    throw "Visual Studio x64 build environment was not found"
  }
  cmd.exe /s /c "`"$VcVars`" >nul && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
      Set-Item -Path "Env:$($Matches[1])" -Value $Matches[2]
    }
  }
}

$OutputDirectory = Join-Path $ProjectRoot $OutputRoot
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$Source = Join-Path $ProjectRoot "workloads/cuda/p17_rtx3070_calibration.cu"
$Executable = Join-Path $OutputDirectory "p17_rtx3070_calibration.exe"
$Measurement = Join-Path $OutputDirectory "native_measurements.json"
$BuildLog = Join-Path $OutputDirectory "nvcc_build.txt"

& $Nvcc -std=c++17 -O3 -lineinfo -arch=sm_86 -allow-unsupported-compiler `
  -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH `
  $Source -o $Executable 2>&1 |
  Tee-Object -FilePath $BuildLog
if ($LASTEXITCODE -ne 0) {
  throw "P17 calibration workload compilation failed"
}
& $Executable --output $Measurement --warmup $Warmup --iterations $Iterations
if ($LASTEXITCODE -ne 0) {
  throw "P17 native calibration workload failed"
}

$GpuIdentity = & nvidia-smi --query-gpu=name,driver_version,memory.total `
  --format=csv,noheader
$Manifest = [ordered]@{
  schema_version = "hetero-p17-native-calibration-manifest/v1"
  measured_at_utc = (Get-Date).ToUniversalTime().ToString("o")
  source = "workloads/cuda/p17_rtx3070_calibration.cu"
  source_sha256 = (Get-FileHash -Algorithm SHA256 $Source).Hash.ToLower()
  executable_sha256 = (Get-FileHash -Algorithm SHA256 $Executable).Hash.ToLower()
  measurement_sha256 = (Get-FileHash -Algorithm SHA256 $Measurement).Hash.ToLower()
  nvcc = $Nvcc
  gpu_identity = $GpuIdentity
  warmup_iterations = $Warmup
  measured_iterations = $Iterations
  measurement_scope = "native_rtx3070_local_vram_not_external_3ddram"
  performance_eligible = $false
}
$Manifest | ConvertTo-Json -Depth 6 | Set-Content -Encoding utf8 `
  (Join-Path $OutputDirectory "measurement_manifest.json")
Write-Output "P17 native measurement completed: $Measurement"
