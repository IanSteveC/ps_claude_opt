/* cuda_iface.h - dual-backend CUDA host interface.
 *
 * Backend 1 (nvcc, all Linux builds): the PS_LAUNCH / PS_SYMCPY* macros
 * expand to the exact <<<>>> and cudaMemcpyToSymbol* code this file's call
 * sites used before the macros were introduced. Zero behavioral change; the
 * runtime API and nvcc-generated registration glue are used as always.
 *
 * Backend 2 (PS_DRIVER_API, MinGW Windows cross-build): host code compiles
 * as plain C++ with no CUDA toolchain. Kernels live in an embedded fatbin
 * (built by nvcc on Linux; GPU code is host-OS independent) and are launched
 * through the driver API (nvcuda.dll) by NAME - PS_LAUNCH stringifies the
 * kernel identifier, PS_SYMCPY* stringify the __constant__/__device__ symbol
 * identifier for cuModuleGetGlobal lookup. A small cudart-lookalike layer
 * (streams, events, memory) maps 1:1 onto driver calls so the orchestration
 * code stays untouched. See build_win.sh; pattern proven by primegrid/ap27.
 */
#pragma once

#ifndef PS_DRIVER_API
/* ------------------------------------------------------------------ */
/* Backend 1: nvcc / CUDA runtime API (Linux release path, unchanged)  */
/* ------------------------------------------------------------------ */

#define PS_LAUNCH(kernel, grid, block, shmem, stream, ...) \
    kernel<<<grid, block, shmem, stream>>>(__VA_ARGS__)

#define PS_SYMCPY(sym, src, count) \
    cudaMemcpyToSymbol(sym, src, count)

#define PS_SYMCPY_ASYNC(sym, src, count, stream) \
    cudaMemcpyToSymbolAsync(sym, src, count, 0, cudaMemcpyHostToDevice, stream)

#define PS_SYMCPY_FROM_ASYNC(dst, sym, count, stream) \
    cudaMemcpyFromSymbolAsync(dst, sym, count, 0, cudaMemcpyDeviceToHost, stream)

#else
/* ------------------------------------------------------------------ */
/* Backend 2: driver API via nvcuda.dll (MinGW Windows cross-build)    */
/* ------------------------------------------------------------------ */

#include <cuda.h>      /* driver API only - no cudart, no nvcc */
#include <cstdio>
#include <cstring>
#include <cstdlib>

/* the code's CUDART_VERSION checks describe the toolkit that produced the
   fatbin, which is the same 12.9 toolkit that provides cuda.h */
#ifndef CUDART_VERSION
#define CUDART_VERSION CUDA_VERSION
#endif

/* ---- cudart-lookalike types ---- */
typedef CUresult cudaError_t;
typedef CUstream cudaStream_t;
typedef CUevent  cudaEvent_t;

#define cudaSuccess CUDA_SUCCESS
#define CUDART_CB CUDA_CB

struct dim3 {
    unsigned x, y, z;
    dim3(unsigned X = 1, unsigned Y = 1, unsigned Z = 1) : x(X), y(Y), z(Z) {}
};

enum cudaMemcpyKind {
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2
};

#define cudaStreamNonBlocking     CU_STREAM_NON_BLOCKING
#define cudaEventBlockingSync     CU_EVENT_BLOCKING_SYNC
#define cudaEventDisableTiming    CU_EVENT_DISABLE_TIMING
#define cudaDeviceScheduleBlockingSync 0 /* handled by cuCtxCreate flags */

struct cudaDeviceProp {
    char   name[256];
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    size_t sharedMemPerMultiprocessor;
    int    multiProcessorCount;
    int    major, minor;
};

/* ---- module / lookup state (inline: header-only, C++17) ---- */
extern "C" const char ps_fatbin[];   /* bin2c-embedded, see build_win.sh */

inline CUmodule ps_module = nullptr;
inline int ps_device_ordinal = 0;

/* host views of the __managed__ result arrays in the module (same names as
   the Linux build's host shadows; declared extern in globals_CUDA.h) */
inline int    *isReported = nullptr;
inline mreal  *dark_best = nullptr, *per_best = nullptr, *dev_best = nullptr,
              *la_best = nullptr, *be_best = nullptr;

inline void ps_check(CUresult r, const char* what)
{
    if (r != CUDA_SUCCESS) {
        const char* s = nullptr;
        cuGetErrorString(r, &s);
        fprintf(stderr, "CUDA driver error %d (%s) in %s\n", (int)r, s ? s : "?", what);
        exit(902);
    }
}

inline void ps_load_module_once(void)
{
    if (ps_module) return;
    ps_check(cuModuleLoadFatBinary(&ps_module, ps_fatbin), "cuModuleLoadFatBinary");
    CUdeviceptr p; size_t b;
    #define PS_MANAGED(var) \
        ps_check(cuModuleGetGlobal(&p, &b, ps_module, #var), "managed " #var); \
        var = (decltype(var))(uintptr_t)p;
    PS_MANAGED(isReported) PS_MANAGED(dark_best) PS_MANAGED(per_best)
    PS_MANAGED(dev_best)   PS_MANAGED(la_best)   PS_MANAGED(be_best)
    #undef PS_MANAGED
}

inline CUfunction ps_func(const char* name)
{
    struct Ent { const char* n; CUfunction f; };
    static Ent cache[64];
    static int ncache = 0;
    for (int i = 0; i < ncache; i++)
        if (!strcmp(cache[i].n, name)) return cache[i].f;
    ps_load_module_once();
    CUfunction f;
    ps_check(cuModuleGetFunction(&f, ps_module, name), name);
    if (ncache < 64) { cache[ncache].n = name; cache[ncache].f = f; ncache++; }
    return f;
}

inline CUdeviceptr ps_sym(const char* name, size_t* bytes = nullptr)
{
    struct Ent { const char* n; CUdeviceptr p; size_t b; };
    static Ent cache[64];
    static int ncache = 0;
    for (int i = 0; i < ncache; i++)
        if (!strcmp(cache[i].n, name)) { if (bytes) *bytes = cache[i].b; return cache[i].p; }
    ps_load_module_once();
    CUdeviceptr p; size_t b = 0;
    ps_check(cuModuleGetGlobal(&p, &b, ps_module, name), name);
    if (ncache < 64) { cache[ncache].n = name; cache[ncache].p = p; cache[ncache].b = b; ncache++; }
    if (bytes) *bytes = b;
    return p;
}

/* ---- kernel launch: variadic marshaling; arg types are the same ones
   <<<>>> type-checked on the Linux build of the identical call site ---- */
template <typename... Args>
inline cudaError_t ps_launch(const char* name, dim3 grid, dim3 block,
                             size_t shmem, CUstream stream, Args... args)
{
    void* params[] = { (void*)&args..., nullptr };
    return cuLaunchKernel(ps_func(name), grid.x, grid.y, grid.z,
                          block.x, block.y, block.z,
                          (unsigned)shmem, stream, params, nullptr);
}
inline cudaError_t ps_launch(const char* name, dim3 grid, dim3 block,
                             size_t shmem, CUstream stream)
{
    return cuLaunchKernel(ps_func(name), grid.x, grid.y, grid.z,
                          block.x, block.y, block.z,
                          (unsigned)shmem, stream, nullptr, nullptr);
}

#define PS_LAUNCH(kernel, grid, block, shmem, stream, ...) \
    ps_launch(#kernel, grid, block, shmem, stream, ##__VA_ARGS__)

#define PS_SYMCPY(sym, src, count) \
    cuMemcpyHtoD(ps_sym(#sym), src, count)

#define PS_SYMCPY_ASYNC(sym, src, count, stream) \
    cuMemcpyHtoDAsync(ps_sym(#sym), src, count, stream)

#define PS_SYMCPY_FROM_ASYNC(dst, sym, count, stream) \
    cuMemcpyDtoHAsync(dst, ps_sym(#sym), count, stream)

/* ---- cudart-lookalike functions over the driver API ---- */
inline cudaError_t cudaSetDevice(int dev) { ps_device_ordinal = dev; return CUDA_SUCCESS; }
inline cudaError_t cudaSetDeviceFlags(unsigned) { return CUDA_SUCCESS; } /* ctx flags set in cuCtxCreate */

inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p, int dev)
{
    CUdevice d;
    CUresult r = cuDeviceGet(&d, dev);
    if (r != CUDA_SUCCESS) return r;
    memset(p, 0, sizeof(*p));
    cuDeviceGetName(p->name, sizeof(p->name), d);
    cuDeviceTotalMem(&p->totalGlobalMem, d);
    int v = 0;
    cuDeviceGetAttribute(&v, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, d);      p->sharedMemPerBlock = v;
    cuDeviceGetAttribute(&v, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, d); p->sharedMemPerMultiprocessor = v;
    cuDeviceGetAttribute(&v, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, d);             p->multiProcessorCount = v;
    cuDeviceGetAttribute(&v, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, d);         p->major = v;
    cuDeviceGetAttribute(&v, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, d);         p->minor = v;
    return CUDA_SUCCESS;
}

inline cudaError_t cudaDeviceGetStreamPriorityRange(int* lo, int* hi) { return cuCtxGetStreamPriorityRange(lo, hi); }
inline cudaError_t cudaStreamCreateWithPriority(CUstream* s, unsigned flags, int prio) { return cuStreamCreateWithPriority(s, flags, prio); }
inline cudaError_t cudaStreamCreateWithFlags(CUstream* s, unsigned flags) { return cuStreamCreate(s, flags); }
inline cudaError_t cudaStreamQuery(CUstream s) { return cuStreamQuery(s); }
inline cudaError_t cudaStreamSynchronize(CUstream s) { return cuStreamSynchronize(s); }
inline cudaError_t cudaStreamDestroy(CUstream s) { return cuStreamDestroy(s); }
inline cudaError_t cudaStreamWaitEvent(CUstream s, CUevent e, unsigned flags = 0) { return cuStreamWaitEvent(s, e, flags); }
typedef void (CUDA_CB *ps_cb_t)(CUstream, CUresult, void*);
inline cudaError_t cudaStreamAddCallback(CUstream s, ps_cb_t cb, void* ud, unsigned) { return cuStreamAddCallback(s, cb, ud, 0); }
inline cudaError_t cudaEventCreateWithFlags(CUevent* e, unsigned flags) { return cuEventCreate(e, flags); }
inline cudaError_t cudaEventRecord(CUevent e, CUstream s) { return cuEventRecord(e, s); }
inline cudaError_t cudaEventDestroy(CUevent e) { return cuEventDestroy(e); }

inline cudaError_t cudaMalloc(void* p, size_t bytes) /* callers pass &ptr as void* via template below */
{ CUdeviceptr d; CUresult r = cuMemAlloc(&d, bytes); *(CUdeviceptr*)p = d; return r; }
template <typename T>
inline cudaError_t cudaMalloc(T** p, size_t bytes)
{ CUdeviceptr d = 0; CUresult r = cuMemAlloc(&d, bytes); *p = (T*)(uintptr_t)d; return r; }
inline cudaError_t cudaFree(void* p) { return p ? cuMemFree((CUdeviceptr)(uintptr_t)p) : CUDA_SUCCESS; }
template <typename T>
inline cudaError_t cudaMallocHost(T** p, size_t bytes) { return cuMemAllocHost((void**)p, bytes); }
inline cudaError_t cudaFreeHost(void* p) { return cuMemFreeHost(p); }
inline cudaError_t cudaMemGetInfo(size_t* freeB, size_t* totB) { return cuMemGetInfo(freeB, totB); }

inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t n, cudaMemcpyKind k, CUstream s)
{
    if (k == cudaMemcpyHostToDevice) return cuMemcpyHtoDAsync((CUdeviceptr)(uintptr_t)dst, src, n, s);
    return cuMemcpyDtoHAsync(dst, (CUdeviceptr)(uintptr_t)src, n, s);
}

inline const char* cudaGetErrorString(cudaError_t e)
{ const char* s = nullptr; cuGetErrorString(e, &s); return s ? s : "unknown CUDA driver error"; }

#endif /* PS_DRIVER_API */
