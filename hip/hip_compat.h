/* hip_compat.h - shims to compile the optimized period_search device code
 * under HIP/ROCm. Included right after hip/hip_runtime.h in the .cu/.cuh.
 *
 * Three jobs:
 *   1. CUDA cache-hint load/store builtins (__stwb/__ldca/...) that HIP does
 *      not provide -> plain loads/stores (the hints are advisory; AMD ignores
 *      them anyway). cudamemasm.h's inline PTX helpers are dead code (0 call
 *      sites) and are guarded out entirely under HIP.
 *   2. Warp shuffles: the kernels use the CUDA _sync forms with an explicit
 *      32-bit mask. HIP's __shfl_* take no mask. Wrappers drop the mask and,
 *      crucially, pin the shuffle WIDTH to 32 so the 32-wide tree reductions
 *      stay correct on wave64 hardware (Radeon VII, CDNA) as well as wave32
 *      (RDNA). Phase-1 correctness relies on this width pin.
 *   3. FP64-economy arch gate (FAT_FP64): choose the branch by AMD DP:FP ratio
 *      instead of __CUDA_ARCH__. CDNA (gfx908/90a) and Radeon VII (gfx906) are
 *      >=1:2 / 1:4 FP64 -> take the data-center "fat FP64" path; RDNA consumer
 *      parts (~1:16..1:32) take the DP-pipe-economy path, same reasoning the
 *      CUDA build applies to data-center vs GeForce.
 */
#pragma once

#ifdef __HIP_PLATFORM_AMD__

/* host-side CUDA spellings the sources still use */
#ifndef CUDART_CB
#define CUDART_CB
#endif
#ifndef CUDA_VERSION
#define CUDA_VERSION HIP_VERSION
#endif
#ifndef CUDART_VERSION
#define CUDART_VERSION HIP_VERSION
#endif

/* ---- 1. cache-hint load/store builtins -> plain accesses ---- */
template <typename T> __device__ __forceinline__ T   ps_ld(const T* p)      { return *p; }
template <typename T, typename U> __device__ __forceinline__ void ps_st(T* p, U v) { *p = (T)v; }

#define __ldg(p)    ps_ld(p)
#define __ldca(p)   ps_ld(p)
#define __ldcs(p)   ps_ld(p)
#define __ldcg(p)   ps_ld(p)
#define __ldlu(p)   ps_ld(p)
#define __ldcv(p)   ps_ld(p)
#define __stwb(p,v) ps_st(p, v)
#define __stcs(p,v) ps_st(p, v)
#define __stcg(p,v) ps_st(p, v)
#define __stwt(p,v) ps_st(p, v)

/* ---- 2. warp shuffles: mask-dropping, width-pinned to 32 ---- */
/* HIP __shfl_* default width = warpSize (64 on wave64). Pinning width=32
   reproduces the CUDA 32-lane semantics on every AMD wave size. */
#ifndef PS_WARP
#define PS_WARP 32
#endif

template <typename T> __device__ __forceinline__
T ps_shfl_down(T v, unsigned d)     { return __shfl_down(v, d, PS_WARP); }
template <typename T> __device__ __forceinline__
T ps_shfl_xor(T v, int m)           { return __shfl_xor(v, m, PS_WARP); }
template <typename T> __device__ __forceinline__
T ps_shfl(T v, int s)               { return __shfl(v, s, PS_WARP); }

#define __shfl_down_sync(mask, v, d)  ps_shfl_down(v, d)
#define __shfl_xor_sync(mask, v, m)   ps_shfl_xor(v, m)
#define __shfl_sync(mask, v, s)       ps_shfl(v, s)

/* __ballot_sync: CUDA returns a 32-bit mask of the 32-lane warp. On AMD
   __ballot returns a 64-bit mask of the whole physical wave. On wave64 the
   two 32-thread logical warps (threadIdx.y groups) share one wave, so the raw
   64-bit ballot mixes them - fatal for the stream-compaction call sites, which
   derive a count and prefix-sum position from it (mixed -> count up to 64 ->
   the 32-entry fc[]/wc[] staging arrays overflow -> garbage facet index ->
   fault). ps_ballot32 returns only the 32 bits for THIS lane's half-wave; the
   code's `tid = threadIdx.x` equals the lane's index within that half, so the
   `(1u<<tid)` prefix mask and __popc stay correct. On wave32 the high half is
   empty and this is a no-op. */
__device__ __forceinline__ unsigned ps_ballot32(int pred)
{
    unsigned long long b = __ballot(pred);
    return (__lane_id() & 32u) ? (unsigned)(b >> 32) : (unsigned)(b & 0xffffffffu);
}
#define __ballot_sync(mask, pred)     ps_ballot32(pred)

/* CUDA __syncwarp() both synchronizes the warp AND orders its shared-memory
   accesses. The kernels use it as a producer/consumer barrier for LDS-staged
   arrays (fc[], wc[], geo[]): some lanes write, then all lanes read what other
   lanes wrote. A no-op is WRONG on AMD - a wavefront executes in lockstep, but
   cross-lane LDS writes are only guaranteed visible after an LDS wait, so the
   consumer reads garbage (garbage facet index -> garbage table pointer ->
   fault). __threadfence_block() emits the s_waitcnt lgkmcnt(0) that makes the
   staged writes visible; combined with lockstep execution that reproduces
   CUDA's warp-synchronous semantics, without the deadlock risk a full
   __syncthreads() carries in the divergent reduction tails. */
#ifndef __syncwarp
#define __syncwarp(...) __threadfence_block()
#endif

#endif /* __HIP_PLATFORM_AMD__ */
