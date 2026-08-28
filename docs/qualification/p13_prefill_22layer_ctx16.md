# P13 22-layer Prefill scale qualification

Date: 2026-08-28

Fixed case: TinyLlama-1.1B FP16, BS=1, Context=16, all 22 layers, GPU-only Model 3 shared 3D-DRAM.

Result:

- 272 device tasks and 19 operator classes;
- 448 non-overlapping Global PA ranges using 2,227,624,514 B;
- 3,382 GPU Parent requests accepted and completed;
- 0 ATLAS Parent requests, one Ramulator2 instance, 0 outstanding;
- 100% cycle-contract coverage and 0 analytical fallback;
- unqualified makespan 476,967,500,000 fs.

Simulation key: `731280601f49bc7510bbea49dcc287b74d3ee5aa97f9870191ede4b4e53948f6`.

The stage validates all-layer dependency, parameter allocation and 22 K/V update pairs at small context. Two independent runs are byte-identical for all seven core artifacts.
