# P11 complete Prefill contract qualification

Date: 2026-08-28

P11 materializes the tensors and execution contracts needed by Prefill instead of multiplying a representative layer. The strict graph explicitly includes token IDs and embedding weight, every layer parameter, correct GQA packed QKV width, Q/K/V separation through KV append, two-input residuals, SwiGLU gate/up and activation widths, K/V versioned read-modify-write, final norm, LM head and first-token sampling.

For a one-layer TinyLlama graph the contract contains 20 tasks and 19 unique operators. Both `gpu0` and `atlas0.compute` entries in `tinyllama11b_prefill_fp16_v1.json` cover every operator; missing entries or analytical fallback fail the run.

For the 22-layer Context=1024 case static construction produces:

- 272 device tasks;
- 448 distinct values/Global PA ranges;
- 2,200,096,768 B of parameters;
- 1,734,412,800 B of activations;
- 23,068,672 B of KV storage;
- 3,957,580,290 B total allocated space including metadata and alignment.

Unit tests verify the 2,560-wide packed QKV output, 11,264-wide gate/up output, 5,632-wide SwiGLU activation, single-token final norm/LM head, KV read-modify-write, residual inputs, capacity and non-overlap. These are deployment contracts, not measured cycle artifacts.
