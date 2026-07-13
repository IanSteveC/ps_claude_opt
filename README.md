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

## Build matrix (one codebase, all targets)

A single source tree builds every platform/backend, and every one of them takes
`FP32=1` to select the df64 emulation build:

| target | command | `FP32=1` |
|---|---|---|
| Linux x64 CUDA | `make` | ✓ |
| Linux aarch64 CUDA (Jetson) | `make -f Makefile_arm` | ✓ |
| Windows x64 CUDA (MinGW cross) | `./build_win.sh` | ✓ |
| Linux x64 HIP/ROCm | `hip/build_hip.sh` | ✓ |
| Windows x64 HIP (MinGW cross) | `hip/build_win_hip.sh` | ✓ |

CUDA and HIP keep separate source copies (root vs `hip/`) as before — only the
stable pure-host-math `.c` routines and the FP32 abstraction `mreal.h` are
shared. `mreal == double` with no `FP32=1`, so the default builds are byte-for-
byte the prior FP64 app.

## FP32-only build

`make FP32=1` builds `period_search_BOINC_cuda12000_fp32`, in which **every
FP64 operation in device code is emulated with paired FP32 operations**
(double-float / "df64": Dekker/Knuth error-free transformations, ~46-48-bit
effective mantissa, see `mreal.h`). The host still computes in double and
converts at the copy/launch/readback boundary; `df64` and `double` are both
8 bytes, so all layouts are unchanged. `cuobjdump -sass` of the result
contains **zero FP64 instructions** — the build targets GPUs with no or
heavily throttled FP64 (Apple Silicon-class devices via future ports,
Intel Arc, 1:32-1:64 GeForce/Jetson).

Validated on the 10-input `input_gt_2000_test` suite against the CPU (`fma`)
reference: 0 validator violations (0.1/0.1/0.5 on period/rms/chi2);
max |dP| 1.4e-5, |dRMS| 4.9e-4, |dchi2| 0.34 across 4531 output lines —
the same divergence envelope as the FP64 GPU build (1.6e-5 / 4.7e-4 / 0.32);
mean rms within 2.2e-6 of reference with balanced per-line better/worse
counts (1497/1570 vs FP64's 1599/1527). The FP64 build (`make`) is
unaffected: mreal == double and outputs are bit-identical.

On FP64-rich hardware the emulation is a pessimization by design: ~9-10x
slower than the FP64 build on V100 (1:2 FP64). Each emulated op costs
~10-25 FP32 ops, so on 1:64-FP64 consumer parts the same arithmetic maps to
roughly 2.5-6x fewer issue slots than native FP64 — the build exists for
exactly those devices (unverified on real 1:64 hardware so far).

Notes for porters: the df64 `two_prod` must use `__fmul_rn` (nvcc's default
FMA contraction otherwise folds the error term to zero); the FP32 build must
not use `--use_fast_math`; inf/NaN must propagate IEEE-style through the
emulated ops or the LM reject-wild-steps recovery breaks (details in
`mreal.h` comments). The port also surfaced a latent upstream race in the
`Flags[]` helpers (`__ldg` read-modify-write of a location written in the
same kernel) that the FP64 build survives only by compiler luck — fixed
here for both builds.

### HIP/ROCm FP32 build (`hip/`, `FP32=1 ./build_hip.sh`)

The same df64 emulation is ported to the HIP tree (`mreal.h` is shared by both
trees; `hip/` keeps its own copies of the kernels per the CUDA/HIP isolation).
Produces `period_search_BOINC_hip_claude_fp32`. Correctness on gfx1030 is
identical to the CUDA FP32 build (same df64 code): 0 validator violations,
max |dP| 1.4e-5 / |dRMS| 4.9e-4 / |dchi2| 0.34 over the 10-input suite — the
same envelope as the FP64 build. Device code object: zero `v_*_f64`
instructions.

**Build flag that matters:** the FP32 HIP build uses `-ffp-contract=on`, NOT
the default `-ffp-contract=fast`. Under `=fast`, clang contracts `two_prod`'s
`a*b` into the following `fmaf(a,b,-p)` *across statements* and folds the error
term to zero — verified on gfx1030 to collapse multiply accuracy from 2^-46 to
2^-24 and wreck sincos/exp2/acos. `=on` only contracts within a single source
expression, so `two_prod` survives while every other multiply-add still fuses.

**Performance finding (the payoff test):** on a Radeon RX 6800 XT (gfx1030,
~1:16 FP64) the FP32 build is **~1.94× *slower*** than the FP64 HIP build
(mean 112.1 s vs 57.6 s over the suite). At a 1:16 FP64 ratio the emulation
does not pay off: each FP64 op becomes ~10-25 FP32 ops, which exceeds the 16×
FP64 penalty, and much of the kernel is memory/latency-bound rather than
FP64-arithmetic-bound. The emulation is therefore worthwhile only where FP64
is **absent** (Apple Silicon, Intel Arc — the FP32 build is the only way to
run at all) or **far more throttled than 1:16** (1:32-1:64 consumer GeForce and
Jetson). 1:16 is not deep enough; a 1:32+ part is needed to break even, and
that remains unverified for lack of such hardware here.

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
