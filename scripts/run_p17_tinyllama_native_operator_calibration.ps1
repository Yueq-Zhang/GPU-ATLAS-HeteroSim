param(
  [Parameter(Mandatory = $true)]
  [string]$ModelPath,
  [string]$Python = "python",
  [string]$ProjectPython = "",
  [string]$OutputRoot = "validation/p17/gpu_operator_pairing",
  [int]$Warmup = 50,
  [int]$Iterations = 500,
  [switch]$SkipNativeMeasurement
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if (-not $ProjectPython) {
  $ProjectPython = Join-Path $ProjectRoot ".venv/Scripts/python.exe"
}
$OutputDirectory = Join-Path $ProjectRoot $OutputRoot
$Native = Join-Path $OutputDirectory "native_rtx3070_local_vram.json"
$Simulator = Join-Path $OutputDirectory "simulator_external_shared3d.json"
$Audit = Join-Path $OutputDirectory "pairing_audit.json"
$Capabilities = Join-Path $ProjectRoot `
  "configs/hetero/operator_capabilities/tinyllama_prefill_layer0_bs1_ctx16.json"
$SimpleMeasurements = Join-Path $ProjectRoot `
  "validation/p17/native_rtx3070/native_measurements.json"
$Benchmark = Join-Path $ProjectRoot `
  "workloads/python/tinyllama_prefill_native_benchmark.py"

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
if (-not $SkipNativeMeasurement) {
  & $Python $Benchmark `
    --model $ModelPath `
    --capabilities $Capabilities `
    --simple-kernel-measurements $SimpleMeasurements `
    --output $Native `
    --warmup $Warmup `
    --iterations $Iterations
  if ($LASTEXITCODE -ne 0) {
    throw "P17 native operator benchmark failed"
  }
} elseif (-not (Test-Path -LiteralPath $Native)) {
  throw "SkipNativeMeasurement requires an existing native catalog: $Native"
}

& $ProjectPython (Join-Path $ProjectRoot `
  "scripts/build_p17_gpu_simulator_catalog.py") --output $Simulator
if ($LASTEXITCODE -ne 0) {
  throw "P17 simulator catalog generation failed"
}
& $ProjectPython (Join-Path $ProjectRoot `
  "scripts/audit_p17_gpu_operator_pairing.py") `
  --native $Native `
  --simulator $Simulator `
  --output $Audit
if ($LASTEXITCODE -ne 0) {
  throw "P17 GPU operator pairing audit failed"
}

$ModelConfig = Join-Path $ModelPath "config.json"
$ModelWeights = Join-Path $ModelPath "model.safetensors"
if (-not (Test-Path -LiteralPath $ModelConfig) -or `
    -not (Test-Path -LiteralPath $ModelWeights)) {
  throw "The fixed TinyLlama config or safetensors file is missing"
}
$Manifest = [ordered]@{
  schema_version = "hetero-p17-gpu-operator-calibration-manifest/v1"
  measured_at_utc = (Get-Date).ToUniversalTime().ToString("o")
  model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  checkpoint_revision = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"
  model_config_sha256 = (Get-FileHash -Algorithm SHA256 $ModelConfig).Hash.ToLower()
  model_weights_sha256 = (Get-FileHash -Algorithm SHA256 $ModelWeights).Hash.ToLower()
  capability_catalog_sha256 = `
    (Get-FileHash -Algorithm SHA256 $Capabilities).Hash.ToLower()
  benchmark_source_sha256 = `
    (Get-FileHash -Algorithm SHA256 $Benchmark).Hash.ToLower()
  operator_builder_sha256 = (Get-FileHash -Algorithm SHA256 `
    (Join-Path $ProjectRoot `
      "workloads/python/tinyllama_prefill_operator.py")).Hash.ToLower()
  simple_kernel_measurement_sha256 = `
    (Get-FileHash -Algorithm SHA256 $SimpleMeasurements).Hash.ToLower()
  native_catalog_sha256 = (Get-FileHash -Algorithm SHA256 $Native).Hash.ToLower()
  simulator_catalog_sha256 = `
    (Get-FileHash -Algorithm SHA256 $Simulator).Hash.ToLower()
  pairing_audit_sha256 = (Get-FileHash -Algorithm SHA256 $Audit).Hash.ToLower()
  warmup_iterations = $Warmup
  measured_iterations = $Iterations
  native_memory_topology = "gpu_local_vram"
  simulator_memory_topology = "external_shared_3ddram"
  performance_eligible = $false
  blocking_reason = "native and simulator memory topologies do not match"
}
$Manifest | ConvertTo-Json -Depth 6 | Set-Content -Encoding utf8 `
  (Join-Path $OutputDirectory "measurement_manifest.json")
Write-Output "P17 exact-operator calibration evidence completed: $OutputDirectory"
