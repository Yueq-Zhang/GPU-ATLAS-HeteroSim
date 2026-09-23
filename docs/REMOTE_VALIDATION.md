# Remote validation policy

## Execution host

- Default long-running validation host: `yueqi@192.168.5.2`.
- Project path: `/opt/gpu-atlas/GPU-ATLAS-HeteroSim`, symlinked to the
  user-owned checkout under `/home/yueqi/gpu-atlas`.
- Dependency path: `/opt/gpu-atlas/dependencies`.
- Qualification output path: `/opt/gpu-atlas/qualification`.
- Authentication credentials must be entered interactively.  Passwords and
  private keys must not be committed to this repository, command files,
  experiment JSON, logs, or qualification records.

The remote RTX 4090 is not the simulated target of the P15d LM Head replay.
The pinned SM86 Trace still runs against the RTX 3070 Accel-Sim configuration;
the host GPU is only relevant to future native capture or functional checks.

## P30 and later execution policy

Starting with P30, all new tests, CUDA execution, SASS inspection and NVBit
Trace generation run on the remote RTX 4090 host. The local checkout is used
only for source editing and lightweight evidence archives. This policy changes
the execution location, not the simulated hardware identity.

Every capture record must keep these fields separate:

- `capture_device_sm`: the physical GPU that executed the framework request;
- `binary_versions`: the actual SASS versions observed in Trace headers;
- `replay_target_sm`: the Accel-Sim timing configuration.

P31 currently records SM89, `[86]` and SM86 respectively. A remote RTX 4090
capture must not be relabeled as a calibrated RTX 4090 simulation. Raw P31
Trace data remains in the remote evidence directory; only qualification JSON
and compact summaries are synchronized back to the repository.

P32-P34 follow the same location policy. P33 preserves the 40-kernel GPU state
inside one process per leg and the full ATLAS stream inside one Ramulator2 per
leg. P34 loads vLLM and TensorRT-LLM from the isolated P29 environments and
uses the pinned local Hugging Face cache; a Hub timeout is never part of a
qualification run. Runtime-random request IDs may be normalized for semantic
comparison, but raw event records remain archived.

The reproducible P30/P31 entry point is:

```bash
export P30_P31_PHASE=all
bash scripts/run_p30_p31_remote.sh
```

The script refuses non-RTX-4090/SM89 hosts and partial capture directories.
Credentials and private keys remain outside prompts, repository files and
logs.

## Codex orchestration policy

Routine deployment, monitoring, deterministic replay, artifact comparison and
documentation tasks for this validation flow should use `gpt-5.6-luna` with
`xhigh` reasoning.  This is an orchestration setting, not a simulator input,
and therefore is not part of the Simulation Key.  Architecture changes,
fidelity-boundary changes, and final performance-claim review still require an
explicit review rather than being silently delegated as routine validation.

## Parallel deterministic qualification

The two deterministic legs may execute concurrently because each process has
its own output directory and owns its own single Ramulator2 instance.  They
must never write to the same directory.  From the project root:

```bash
export PYTHONPATH=.
BACKEND=configs/hetero/backends/gpu_accelsim_rtx3070_ramulator2_hbdram_edge_16ch.json
MANIFEST=configs/hetero/operator_artifacts/p15d/tinyllama_prefill_bs1_ctx16_lm_head_sm86_trace.json
OUTPUT=/opt/gpu-atlas/qualification/p15d/coupled/accel-sim-rtx3070-lm-head-shared-hbdram-identity-remote

mkdir -p "$OUTPUT"
nohup taskset -c 2 python3 scripts/run_accel_sim_single.py \
  --backend-config "$BACKEND" \
  --trace-manifest "$MANIFEST" \
  --output "$OUTPUT/native_baseline" \
  >"$OUTPUT/native_launcher.log" 2>&1 &
nohup taskset -c 3 python3 scripts/run_accel_sim_single.py \
  --backend-config "$BACKEND" \
  --trace-manifest "$MANIFEST" \
  --output "$OUTPUT/adapter" \
  >"$OUTPUT/adapter_launcher.log" 2>&1 &
```

The 12-hour per-leg timeout accommodates the full LM Head Trace without
changing the simulated hardware.  A leg is reusable only after both
`command.json` and `stats.json` exist and match the current backend, command,
Simulation Key, frequency and schema.  Resume loading re-applies the same
Ramulator2 and ATLAS conservation checks used by a fresh run.  Starting a new
leg removes any stale `stats.json` before writing its command, and a partial
performance-counter file is not a completion record.

After both legs finish, generate the exact comparison record without replaying
them:

```bash
PYTHONPATH=. python3 -m frontend.hetero.cli qualify-gpu \
  --resume-completed-runs \
  --backend-config "$BACKEND" \
  --trace-manifest "$MANIFEST" \
  --output "$OUTPUT"
```

Only a `status=passed` record with equal GPU cycles, instructions and complete
external-memory statistics qualifies the operator.  Each leg must independently
report one Ramulator2 instance, nonzero traffic, complete Parent/Child and
durable completion conservation, and zero outstanding requests.  Independent
operator cycles must not be summed and reported as end-to-end Prefill latency.
