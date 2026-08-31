# Remote validation policy

## Execution host

- Default long-running validation host: `yueqi@192.168.0.197`.
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
