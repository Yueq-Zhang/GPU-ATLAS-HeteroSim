# P10b-B live request-cycle qualification

Date: 2026-08-28

## Scope

This stage connects the strict single-placement/versioned plan to one live Ramulator2 instance. GPU requests use the external link and Logic-Die Gateway; ATLAS requests use the internal Hybrid-Bond port. A Model 3 cross-device route waits for producer durability and issues a consumer acquire probe before the destination version becomes visible.

The active operator lifecycle is `sampled reads complete → explicit compute cycles → sampled writes complete → version commit`. It is not a Roofline run. Compute cycles come from an uncalibrated tiled schedule contract and memory traffic is bounded sampling, so performance claims remain disabled.

## Fixed case and result

- TinyLlama-1.1B FP16, BS=1, Context=16, one layer;
- `causal_attention` on `atlas0.compute`; all other tasks on `gpu0`;
- 20 device tasks, 4 synchronization routes, 100% cycle-contract coverage;
- 382 accepted/completed Parent requests: GPU 347, ATLAS 35;
- one Ramulator2 instance, 0 rejected, 0 outstanding;
- Global PA: 28 non-overlapping ranges, 351,615,554 B allocated;
- makespan: 36,175,000,000 fs, recorded only as an unqualified deployment value.

Simulation key: `004770e3f2b0bebce0ffa6a5f84142501e99de4102d8f5abd24fd709030eac41`.

Two independent runs have byte-identical metrics, memory statistics, request trace, Global PA map, artifact coverage, execution graph and residency outputs. The final qualification root is:

```text
/opt/gpu-atlas/qualification/prefill-p10b-to-p14-final
```

## Claim boundary

P10b-B qualifies request-cycle orchestration, live completion callbacks, initiator separation, one memory timing owner, fence/acquire version gating and deterministic replay for this fixed contract. It does not qualify full instruction traces for every operator, a calibrated GPU/ATLAS performance model, full unsampled memory traffic, virtual memory translation or configurable/XOR DRAM mapping.
