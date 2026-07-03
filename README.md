# ps_claude_opt — optimized PeriodSearch CUDA application

Heavily optimized CUDA build of the [AsteroidsAtHome PeriodSearch](https://github.com/AsteroidsAtHome/PeriodSearch)
BOINC application (asteroid lightcurve inversion, Kaasalainen–Ďurech convex
inversion). Derived from the `petri_cuda` lineage and rewritten for memory
locality and occupancy; all computation remains FP64 except where noted.

## Performance

Measured on Tesla V100-SXM2-16GB (CUDA 12.9), `input_gt_2000_test` suite:

| stage | input_239_1 |
|---|---|
| original petri build | 168 s |
| pole-merge (bit-exact) | 87 s |
| memory-layout rewrite | 44.5 s |
| gauss/batching/tuning | 36.6 s |
| row pairing + FP32 basis + arch gating | **~23 s (≈ 7×)** |

Also runs on Jetson Orin Nano (`Makefile_arm`) with arch-gated FP64-economy
code paths (`FAT_FP64` in `Start.cu`) so data-center (1:2 FP64) and
consumer/Jetson (1:32–1:64 FP64) GPUs each get their best variant.

## Key changes vs upstream

- One CUDA block per (trial frequency, pole) pair — the 10 pole trials run
  concurrently instead of serially (bit-exact, ~2×).
- Per-block `Dg` matrix eliminated via the rank-1 identity `Dg = Dsph · g`;
  derivative sums gather from one shared, transposed, cache-resident basis
  matrix (`CUDA_DsphT`, plus an FP32 mirror used on high-FP64-rate GPUs).
- Warp-cooperative brightness/derivative kernel; transposed per-point scratch
  (`dytempT`); rank-8 shared-memory tiles for the normal-equation updates;
  two-point row pairing halves the dominant cache stream.
- Gauss-Jordan solve in shared memory, one 128-thread block per pair.
- Solar-phase `acos` argument clamped to [-1,1] (opposition WUs come within
  1e-7 of the domain edge; an out-of-range rounding poisons every frequency).
- **Blackwell (RTX 50xx) workaround**: CUDA 12.9 ptxas miscompiles this code
  through the native `compute_120` + device-LTO pipeline (intermittent NaN
  chi-square, confirmed on a 5060 Ti). The Makefile therefore builds the
  sm_120 entry as `arch=compute_89,code=sm_120`, which is proven correct.

See [OPTIMIZATION_NOTES.md](OPTIMIZATION_NOTES.md) for the full engineering
log: profiling evidence, per-phase results, measured dead-ends, the
adversarial review findings, and the Blackwell diagnosis.

## Building

Linux x86_64: needs CUDA 12.9 at `/usr/local/cuda-12.9`, BOINC sources/libs at
`../../boinc` (prebuilt `libboinc_api.a` / `libboinc.a`), then:

```
make clean && make -j
```

(`make clean` matters: stale single-arch objects from development builds will
otherwise be relinked into the fat binary.)

Jetson (aarch64): `make -f Makefile_arm`.

The app reads `period_search_in` from the working directory and writes
`period_search_out`. Test inputs are the standard AsteroidsAtHome workunit
format; run standalone with `./period_search_BOINC_cuda12000 --device 0`.

## Validation

All changes validated against the CPU (`fma`) reference outputs on a 10-input
suite: most inputs line-for-line identical; the rest within ~1e-3 relative on
isolated lines (BOINC validator tolerance is 0.1/0.1/0.5 on period/rms/chi2).
Fit-quality neutrality of the FP32-basis path was verified with a mean-rms
oracle (per-line better/worse counts statistically balanced against the CPU
reference). Note: validator-pass alone is not a sufficient correctness check —
compare mean rms too (see notes for the incident that proved this).
