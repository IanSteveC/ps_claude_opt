# PeriodSearch CUDA optimization — findings & plan (2026-07-01)

## Baseline (input_239_1, V100, CUDA 12.9, sm_70)
- Wall time: **168 s**; output matches reference `period_search_out_239_1_cuda`
  (372/373 lines identical; 1 marginal bin converged differently — normal FP jitter).

## Verified bottleneck (nsys + ncu)
1. **95.6% of GPU time in one kernel**: `CudaCalculateIter1Mrqcof2CurveM12I1IA0`
   (= `mrqcof_curve1` brightness/derivatives + `MrqcofCurve2I1IA0` normal equations),
   runs every LM iteration. Gauss (`Mrqmin1End`) is 1.9%. GPU timeline ~98% busy → no
   launch-gap problem; the waste is inside the kernel.
2. **Precalc phase = 41.7 s (25% of run) at grid = 10 blocks** (320 threads on 80 SMs).
3. **Main phase grid = 384 blocks × 32 threads** (one warp per frequency bin; 10 poles
   processed serially). ncu on a main-phase instance:
   - Achieved occupancy **7.2%** (4.6 warps/SM, waves/SM = 0.30 — grid smaller than GPU)
   - SM busy **6.6%**, IPC 0.29 — FP64 pipes idle
   - DRAM 30%, L1 hit 69%, L2 hit 57% — 8-byte scattered gathers from per-block `Dg`
     (~116 KB/block working set thrashes L1/L2), Petri's own `// ZZZ bad, strided read`
     comments mark the dytemp transposed reads.
4. Total useful FP64 work ≈ 5×10¹³ FLOP ≈ 6 s at V100 peak → **~25× theoretical headroom**.

So: "memory sparsity" = (a) far too little parallelism to hide any latency, ×
(b) scattered gathers/strided access in the hot kernel. Both must be fixed.

## Upstream comparison (GitHub survey)
This fork descends from `petri_cuda` (fastest upstream lineage, last touched 2024-07).
Upstream `dev` uses the same per-point rank-1 alpha updates and per-thread incl/dbr
lists, 128-thread blocks, no post-2024 kernel perf work. **Rebasing buys nothing;
keep the fork.** BOINC validator compares only per/rms/chi2 at ~10%/10%/50% relative
tolerance → ample numerical headroom for reordered reductions.

## Plan
**Phase A — pole/period merge (bit-exact math):** block = (frequency, pole) pair
instead of frequency; the 10-pole host loop disappears; host reduces best-over-poles
per frequency after each batch. Precalc: 10 periods × 10 poles = 100 bids at once.
Main: 373×10 = 3730 bids → fills the GPU. N_BLOCKS 2048→4096, batches sized by free
GPU memory.

**Phase B — memory-layout rewrite of the hot kernel family:**
- `Dg[coef][facet]` per block is rank-1: `Dg[k][f] = Dsph[k][f] · g_f(bid)`.
  Replace with **global shared `DsphT[facet][coef]`** (one 148 KB matrix, L1/L2-hot
  across all blocks) folding `g_f` into the per-point facet weight
  (`dbr' = Areag[bid][f]·s`, already computed). Eliminates the per-block 116 KB Dg
  array and its scattered gathers entirely.
- `mrqcof_curve1`: warp-cooperative per point — lanes=facets for visibility/br phase
  (chunked ×32 into shared `dbr`), lanes=coefs for the derivative matvec reading
  DsphT rows **coalesced**; writes `dytempT[point][coef]` (stride 64) **coalesced**.
- `MrqcofCurve2*`: 16-point tiles staged in shared from dytempT (coalesced), I1
  renormalization fused into staging (removes a full dytemp read+write pass), alpha
  updated **once per 16 points** (rank-16) instead of per point → 16× less L2 traffic.
  Index arithmetic of each I*/IA* variant kept verbatim.
- `conv`/`curv`: curv shrinks to Areag computation (no Dg write); conv reads DsphT
  with lanes=coefs (coalesced) and weight `Areag·Nor`.
- dytemp shrinks from (ndata+1)×(ma+1) to (max_lpoints+1)×64 per bid (it is per-curve
  scratch; ~2× smaller even with padding).

**Phase C:** occupancy/regs tuning, (32,4) blocks for hot kernels, re-profile;
gauss only if it becomes a significant fraction.

Expected: ~10–20× end-to-end.

## Results (2026-07-02)

Timeline on `input_239_1` (V100, sm_70 build):
- Baseline: **168 s**
- Phase A (pole merge, bit-exact): **87 s** (1.93×)
- Phase B (DsphT fold + warp-cooperative bright + dytempT + rank-8 alpha tiles): **44.5 s** (3.8×)
- Phase C (shared-memory Gauss-Jordan; branch-free matvec for tiny grids): **37.8 s** (**4.44×**)

Final production binary (all-arch fatbin, stock Makefile flags), measured baseline vs optimized:

| input     | baseline | optimized | speedup | output vs CPU reference |
|-----------|----------|-----------|---------|--------------------------|
| 239_1     | 168.3 s  | 37.8 s    | 4.45×   | 372/373 identical, rest ≤ 9e-5 rel |
| 248_1     | 151.1 s  | 36.8 s    | 4.11×   | 496/496 identical (2-batch path) |
| 320_1     | 167.6 s  | 44.0 s    | 3.81×   | 381/398 identical, rest ≤ 6e-4 rel |

All 10 test inputs pass the validator; final binary times across the whole
suite: 36.8–44.0 s (the even-batch fix also accelerated every >409-frequency
input from ~51–55 s to ~37–40 s). Final profile: 96% of GPU time in the one
remaining hot kernel (balanced ~32% SM/L1/L2 at 25% occupancy, register-bound),
gauss 1.5%, precalc ~4 s.

## Adversarial review (3 lenses, findings verified by skeptic agents)
Confirmed and fixed:
1. **NaN latch in host best-pole reduction** (major): a diverged pole writing
   dev_best = NaN would win the host argmin forever; baseline's serial device
   update recovered. Fixed NaN-safe (a NaN incumbent is replaced, a NaN
   candidate never displaces a good one — strictly more robust than baseline,
   same line counts).
2. **Memory-budget check vs rounded grid + unchecked cudaMalloc** (minor):
   budget was checked on the un-rounded bid count while allocations use the
   128-rounded grid; a failed malloc would hang the host loop silently. Fixed
   (rounded-grid check, NULL-initialized pointers, loud exit on failure).
3. **ia-gated tail rows missing their diagonal term** (minor, dead path for
   current WUs): compact column loop used m < l, leaving a structurally zero
   LM diagonal for non-contiguous ia[] configurations. Fixed to m <= l.
Rejected after verification: checkpoint double-count (resume is disabled in
shipped code, byte-identical to baseline) and the renorm tail formula (the new
uniform coef*(dy - coef1*dave) matches the CPU reference; baseline's unrolled
tail was self-inconsistent).

Caveats: IA1 (fitted pole beta) and ia-gap code paths are not exercised by any
available test input — they were ported to NR/CPU-reference-consistent form
rather than replicating baseline quirks (baseline had real bugs there). The
all-arch fatbin is validated on V100 (sm_70) only.

## Register-pressure restructure of the hot kernel (2026-07-02, second pass)

Changes: every warp-uniform value (phase-model scalars, Blmat, cl/cls, per-point
geometry ge/gde, scale/ff/d2/alpha) parked in per-warp shared instead of
per-thread registers; visible facets ballot-compacted per 32-facet chunk so the
derivative sweep is branch-free with independent (pipelineable) loads; curve1
and curve2 shared staging overlaid in a union (their lifetimes don't overlap in
the fused kernel), shrinking the block to 16.9 KB.

Results: 122 → 96 registers (no spills), theoretical occupancy 25 → 31.25%,
achieved 24 → 28.5%. Suite times 36.8–44.0 s → **34.6–38.9 s** (3–17% per
input; ~4.4–4.6× vs baseline overall). ncu after: L2 51% busy, "no eligible
warp" 72% — the kernel is now bound by the L2 round trips baked into the
algorithm (~3 KB/point of alpha read-modify-write at K=8, plus the 1 KB/point
dytempT write-then-read that the I1 renormalization forces). More occupancy no
longer helps; only cutting L2 bytes would (e.g. K=16 tiles shared by warp
pairs → not taken: breaks bit-reproducibility, adds block-sync complexity, est.
~10% for real risk).

Bug found in the process (fixed): the curve2 tile engine's unrolled sums were
hardcoded to 8 points while the arrays were sized CURVE2_K — any K ≠ 8 read out
of bounds into the neighboring warp's shared memory. A K=7 test build produced
validator-passing output whose rms was silently ~6% worse on every line (and
looked 35% "faster" via degraded convergence) — the engine now honors CURVE2_K
everywhere. Moral: validator-pass alone is not a correctness oracle here;
compare mean rms against the CPU reference too.

## Second campaign: toward 20 s (2026-07-02)

1. **Two-point DsphT row pairing** (`mrqcof_curve1_opt` processes points jp and
   jp+1 together; one row load feeds both points' accumulators; visible-facet
   compaction over the union of both points' visibility). Halves the dominant
   L2 stream. Registers 96 → 123 (no spills, 16 warps/SM) — a net big win,
   proving the L2-bound diagnosis. **Bit-identical output.** 36.6 → 25.8 s.
2. **FP32 mirror of DsphT** (`CUDA_DsphTf`) for the derivative sweep only; all
   accumulation stays FP64; the 74 KB float matrix is L1-resident. Outputs
   drift (~1e-3 worst line); the mean-rms oracle over all 10 inputs shows the
   drift is unbiased: 1599 lines better / 1527 worse vs the CPU reference,
   per-input mean-rms shifts ±8e-5 with no direction. 25.8 → 24.0 s.

Final suite: **23.0–27.2 s** (≈ 6.3–7.0× vs baseline; input_239_1: 168.3 → 24.1 s).
Post-change profile: SM 47%, L1 46%, L2 33% — compute/latency-balanced;
the memory-sparsity problem is gone.

## Jetson Orin port + V100 regression check (2026-07-02, later)

Ian's Orin work (`db7bf16`) added three FP64-economy changes plus a Makefile_arm.
V100 retest of that commit showed **+25%** (28.8–34.1 s suite). A/B decomposition
on input_239_1: FP64-table revert +12%, curve2 shuffle-broadcast +17% (shuffles
ride the MIO pipe the V100 build already saturates with shared-memory traffic),
geometry batching **−6% (a genuine V100 win too — kept unconditional)**.

Fix (`c36c9cc`): `FAT_FP64` gate on `__CUDA_ARCH__` — sm_60/70/80/90/100
(≥1:2 FP64) take the FP32 DsphT mirror + uniform-DP curve2 products; all other
architectures (GeForce, Jetson sm_87 via Makefile_arm) compile exactly the
committed Orin behavior. Each SASS target gets its branch at zero runtime cost.

Result: V100 suite **21.7–25.3 s — best yet** (~6.6–7.8× vs baseline), oracle
unchanged (1599 better / 1527 worse vs CPU references). Orin-side SASS is
identical to `db7bf16`; rebuild there once with Makefile_arm to confirm.

## Blackwell (RTX 50xx / sm_120) all-NaN bug (2026-07-03)

Symptom on 5060 Ti / 5070 (all app versions): valid trial periods, NaN rms/chi2,
pole frozen at (0,0). Diagnosed remotely with a 5-variant kit run by the
reporter. Findings:
- The unguarded `acos(ee.ee0)` domain edge was the prime suspect (real WUs come
  within 9e-8 of dot=1) — **disproved for this bug**: the unclamped diagnostic
  build's geometry NaN-trap never fired. The clamp (`1491627`) stays as
  correct hardening regardless.
- The chisq NaN-trap fired **intermittently at LM iteration 3** (one bid of
  1792, relative-curve pass) — data-dependent corruption, not a systematically
  broken instruction.
- Variant B — sm_120 SASS generated from **compute_89 PTX without device-LTO**
  — produces correct output. The compute_120 + LTO pipeline is miscompiled by
  ptxas (CUDA 12.9); historical versions failed the same way because the
  driver JIT lowers their compute_90 PTX through the same buggy path.
Fix (`115e8ad`): Makefile sm_120 entry is now `arch=compute_89,code=sm_120`.
Note: the FAT_FP64 gate sees __CUDA_ARCH__=890 for this entry → non-FAT branch,
which is correct for 1:64-FP64 GeForce parts. V100 output bit-identical.
Open questions: variant C (native compute_120, no LTO) would tell whether LTO
alone is the trigger — worth running for an NVIDIA bug report; sm_100 (GB100)
uses the same compute_100+LTO pipeline and is untested — watch for reports.
Also note the fatbin embeds NO plain PTX (only per-arch SASS + LTO-IR), so
future architectures beyond sm_120 cannot JIT this binary.

Not taken (the remaining ~4 s to reach ≈ 20–21 s):
- P=4 pairing variants compiled separately for the precalc grid (precalc is
  3.5 s of pure per-warp latency at 100-bid parallelism); est. −1.5–2 s,
  requires duplicating the 8 M12 wrappers with a P-templated curve1.
- Convergence-tail packing / batch overlap; est. −0.5–1 s.
- Warp-pair K=16 alpha tiles: obsolete — L2 is no longer the binder.

## Jetson Orin Nano port (2026-07-02, aarch64 / sm_87, Makefile_arm)

Orin economics invert the V100's: FP64 at 1:64 (~32 GFLOPS), 8 SMs, 68 GB/s
LPDDR5 shared with the OS → 18× more bandwidth-per-FLOP than V100. Memory
sparsity costs ~nothing here; every FP64 warp-instruction is precious. The
BOINC fleet's consumer GPUs (1:32–1:64) share these economics, so the DP-lean
choice is the correct single-code-path default.

Measured on input_239_1 (MAXN_SUPER, GPU 1.02 GHz; per-bid-iteration cost):
- baseline (85ea84f): 10.18 ms/bid, precalc 135 s → total ≈ 35.5 min
- optimized as arrived from V100 work: ≈ 23 ms/bid → ≈ 50+ min (user was right:
  net SLOWER than baseline, and the stuck-at-0% fraction made it look worse)
- + geometry batched one-point-per-lane (GEO_BATCH=16, bit-exact): 13.4 ms/bid.
  The all-lanes-redundant geometry (acos/sincos/exp2) was free on V100,
  ~32× the FP64 pipe cost here.
- + curve2 row weights & beta products lane-parallel + shuffle broadcast
  (INT pipe instead of FP64; bit-exact): 12.5 ms/bid
- + matvec back to the FP64 DsphT table: 10.5 ms/bid, precalc 62 s.
  The FP32 mirror's F32→F64 converts run ON the FP64 pipe at consumer ratios —
  +22% DP budget to save bandwidth Orin doesn't need. The float table remains
  built but unused (documented V100-only experiment; cost it ~7% there).
- SM-aware batch cap (52 bids/SM, floor 1024): measured fixed per-iteration
  overhead ≈ 0, so smaller batches are free — pure memory saving on the shared
  8 GB (1.1 GB instead of ~4 GB of scratch).
- Live progress line (iter/converged/percent) + monotone fraction blend: the
  old End-based fraction sat at 0% until convergence (all bids converge at the
  n_iter_max=50 wall simultaneously on these inputs).

Net on Orin: ≈ 34–38 min, at-or-slightly-better than baseline with the V100
structure retained; no arch-specific code paths (SM count and grid size are
runtime-adaptive). Expected V100 cost of the FP64-table revert: ~7%
(24.0 → ~26 s) — needs re-validation on the V100 (geometry/shuffle changes
predicted bit-exact there).

Correctness: all 10 test inputs pass the BOINC-validator comparison against the
CPU (`_fma`) references; 7 of 10 are line-for-line identical, the rest deviate
by ≤ 6×10⁻⁴ relative on isolated marginal lines (tolerance 0.1/0.1/0.5).

Tuning findings worth keeping (measured, not guessed):
- The hot kernel is register-bound at 122 regs; forcing it lower for occupancy
  causes hot-loop spills that cost far more than the extra warps gain
  (44.5 s → 52–67 s in experiments). 16 warps/SM with zero spills is the sweet spot.
- The `if (w != 0)` visible-facet skip in the derivative matvec is worth keeping
  on a full GPU (saves ~50% of DsphT reads/FMAs); an unconditional pipelined
  variant is faster only on tiny grids (precalc) and is selected at runtime.
- double2 vectorization of the DsphT rows was performance-neutral (latency-bound
  kernel); reverted for simplicity.

Remaining headroom (not taken): the main kernel now runs balanced at ~32% of
SM/L1/L2 with 25% occupancy; going further means restructuring for lower
register pressure per point (e.g. multi-point batching with shared staging of
per-point geometry), estimated ≤ 2× more. Precalc (~4 s) is inherently
latency-bound (only 100 (period, pole) pairs exist).

## 2026-07-14 — reduced-precision DsphT mirror removed

The FP32 mirror of `DsphT` and its arch gate (historically `FAT_FP64`,
default-on for sm_60/70/80/90/100 and gfx906/CDNA) are deleted; every
architecture now reads the full-precision table and uses the lane-parallel
shuffle forms. Rationale: the mirror's 2^-24-rounded basis values cost exact
pole (lambda/beta) agreement with the CPU reference on data-center cards
(V100 default build: 12-86 mismatched poles per input on the 10-input suite;
full-precision build: ZERO on all 10, most files 100% line-identical). The
speed it bought (~25% on V100, ~6% on Radeon VII, never measured on
A100/H100 whose larger L1 likely voids the cache argument) is not worth a
silent accuracy asymmetry between host classes. V100 full-precision runtime:
~30.5 s vs ~23 s on input_253_1-class WUs; GeForce/Jetson/RDNA unaffected
(they always used the full-precision path).

Two bit-identical optimizations then clawed most of the V100 loss back
(byte-compared on all 10 suite inputs):
- `26d74f8` curve2 rank-K tile product FORM gate (`PS_TILE_PRODUCTS_UNIFORM`):
  high-FP64 NVIDIA parts compute the K products redundantly per lane on the
  idle FP64 pipe instead of shuffle-broadcasting over the saturated MIO pipe.
  `__dmul_rn` fences the product so nvcc's FMA contraction cannot fuse it
  into the following add. V100 suite 296 → 254 s.
- `edf1262` split the fused M12 I1 curve kernels (port of the HIP tree's
  a3350cf split): dytemp already round-trips through global memory between
  the curve1/curve2 phases, so the split adds no traffic. V100 suite
  254 → 236 s (input_239_1 29.9 → 23.2 s vs the old reduced-precision
  build's 22.7 s — ~2% gap at full precision).

## 2026-07-14 — HIP port of the product-form gate: measured, LOSES on gfx906

Ported `PS_TILE_PRODUCTS_UNIFORM` to `hip/Start.cu` to test whether the V100's
+17% carries to high-FP64 AMD parts. Two HIP-specific findings:

- **`__dmul_rn` is NOT a contraction fence under hipcc.** The FP64 HIP build
  uses `-ffp-contract=fast` (build_hip.sh), and clang fuses
  `__dmul_rn(a,b)+c` into one `v_fma_f64` anyway (verified by gfx906
  disassembly) — that rounds once where the shuffle form rounds twice,
  breaking bit-identity. The working fence is a zero-cost inline-asm
  barrier after the multiply: `asm volatile("" : "+v"(p))`
  (`PS_TILE_FENCE`). ISA-checked: 8 standalone `v_mul_f64` + `v_add_f64`
  chain, zero fused FMAs in the guarded sites.
- **The uniform form is ~4% SLOWER on Radeon VII** (gfx906, 1:4 DP, wave64,
  2 waves/SIMD): 10-input suite 813.7 s uniform vs 781.3 s shuffle, slower
  on every input (outputs byte-identical to each other, pairwise-diffed).
  The V100 argument doesn't hold on GCN: 8 redundant `v_mul_f64` issues at
  1:4 rate cost more than the `ds_bpermute` broadcasts they replace, and
  with only 2 resident waves the extra VALU time sits on the critical path.

Decision: every AMD arch defaults to the shuffle form — the shipped HIP
binary is instruction-identical to the previous build (ISA-diffed for
gfx906 and gfx1030). The gate + fence stay in the source for CDNA
(gfx908/90a/940/942, 1:2 DP, untested — no hardware): build with
`PS_EXTRA_DEFS=-DPS_FORCE_TILE_UNIFORM` to trial it; the uniform form is
already hardware-validated byte-identical on the full suite.

Suite-reference note (matters for anyone re-running validation): the 9
`period_search_out_*_fma` references are CRLF (Windows CPU build) — compare
with CR stripped. HIP FP64 output differs from the CPU references on 4 of
10 inputs (239_1, 251_1, 279_1, 320_1) at a handful of frequency bins
(different LM endpoint, e.g. one line of 373 on 239_1) — pre-existing
ULP-level libm differences (ROCm device libs vs CPU/CUDA), byte-identical
to the earlier validated HIP FP64 campaign runs (fp32-test/runs/hip_fp64),
and unchanged by this work. The other 6 inputs are line-identical to the
CPU references.

## 2026-07-15 — gfx906 campaign: profile, LDS shave, wave64 curve1 (−18.3%)

Profiled the FP64 HIP build on the Radeon VII with classic `rocprof`
(`rocprofv3`'s preload breaks BOINC's timer thread: `pthread_create`
EINVAL → `boinc_init returned 22`; use `rocprof --stats`). Kernel time
on input_239_1: **`Mrqcof2Curve1Mid` 75.2%** (34.7 ms/launch),
`Mrqcof2Curve2MidI1IA0` 20.6%, gauss 2.1%; the `Mrqcof1*` twins are
~2% combined because the `isAlambda` gate early-exits them on every
iteration whose previous trial was rejected — the mrqcof2 (trial-
parameter) instances are the real hot path. GPU busy ≈ wall: no
launch-gap problem. Occupancy audit via the code-object metadata
(`llvm-objcopy --dump-section=.hip_fatbin` → `clang-offload-bundler`
→ `llvm-readelf --notes`; note RDC per-TU objects carry no final
numbers, dump the linked binary): Curve1Mid 127 VGPR / 16 224 B LDS,
Curve2Mid 90-93 VGPR / 16 896 B LDS, both 2 waves/SIMD — but Curve2Mid
was LDS-capped at 3 workgroups/CU = **6 waves/CU, below even its VGPR
limit of 8**.

Two changes, both validated byte-identical over the 10-input suite
(and byte-compared per input against the accepted baseline run — the
lambda/beta/pole bar from the 2026-07-14 incident):

1. **`c2share` LDS shave (`d696ec0`, −2.5%)**: the s2w/dws broadcast
   arrays (128 B/warp) pushed c2share to 4224 B; lane p now keeps point
   p's scalars in registers and the tile products broadcast them over
   the same `ds_bpermute` path they already ride. c2share = exactly
   4096 B/warp → 16 384 B/block → 4 workgroups/CU (8 waves) on gfx906;
   RDNA WGPs go 12 → 16 waves; the fused M12 kernels' union shrinks
   too and their 2-VGPR spill disappears. The uniform product form
   (CDNA trials) keeps the LDS arrays. Suite 783.3 → 763.6 s.

2. **Wave64 curve1 (`d468ba5`, −16.2%)**: on wave64 the (32,4) blocks
   pack two *bids'* logical warps into one physical wave, and
   `blockIdx()` makes those partners 48 frequency steps apart — the
   compact-gather loop runs max(cntA, cntB) iterations and a converged
   bid idles its half-wave for the partner's whole tail.
   `mrqcof_curve1_opt_w64` (gfx9 arch list; host selects by runtime
   warpSize, block (64,2), grid.x×2) gives the whole wave to ONE bid:
   lanes 0-31 run the unchanged paired-points body on (jp, jp+1),
   lanes 32-63 on (jp+2, jp+3), per-half wcA/wcB/fc. lave/dave stay in
   strict point order via a cross-half `__shfl_xor(v,32,64)` exchange —
   every value rounds identically. GEO_BATCH_W64=32 halves the staging
   rounds at the same per-point math. Suite 763.6 → **639.7 s**
   (783.3 → 639.7, −18.3% for the day; per-input 61-68 s vs 71-88 s).

   **Measured dead-end** (do not retry): the "obvious" W64 form — point
   A in the lower half, point B in the upper, union-compacted — is
   byte-identical but 9-14% *slower* than (32,4): both halves load the
   SAME DsphT row values (halving the gather's FMA:load ratio, the
   very thing two-point pairing bought), and every wave-uniform op
   runs once per bid instead of once per two bids. The winning form
   keeps the 32-lane body per half-wave verbatim and only changes who
   shares the physical wave. Also: ROCm 7 removed
   `__AMDGCN_WAVEFRONT_SIZE__` — the gate is the explicit gfx9 list,
   with a `__builtin_trap()` stub on wave32 so a future mismatch fails
   loudly instead of silently no-opping.

   RDNA (wave32) is untouched — same kernels and launch shapes as
   before. CDNA (gfx908/90a/94x, wave64) takes the W64 path: same wave
   structure as gfx906, hardware-untested here (no CDNA in the box);
   all-7-arch bundle compiles clean.

Rejected after analysis: alpha-in-LDS for curve2 (the 12.8 KB/bid
triangular working set caps residency at ~3 waves/CU — loses to the
current 16-bids-in-flight streaming form) and CURVE2_K=16 (halves the
alpha RMW traffic but changes the rank-K summation order → not
bit-identical). Curve2Mid (~26% post-change) is the next candidate,
but only via transforms that keep the per-tile accumulation order.
