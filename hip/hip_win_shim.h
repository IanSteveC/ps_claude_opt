/* hip_win_shim.h
 *
 * MinGW HIP host build ONLY (PS_HIP_WIN). Hand-declared minimal HIP module +
 * runtime API so the host pass never includes any ROCm header - the same
 * discipline the CUDA Windows build uses with cuda.h. amdhip64.dll exports
 * these C symbols on Windows exactly as libamdhip64.so does on Linux (HIP is
 * one portable C ABI), so the same host object binds either.
 *
 * Enum integer values are HIP's own (read from ROCm 7.2 headers on this box
 * and validated against a live runtime via the Linux PS_HIP_WIN build before
 * the Windows exe ships).
 *
 * The functions are runtime-resolved pointers (LoadLibrary/dlopen +
 * GetProcAddress/dlsym, see hip_win_loader.cpp) so one exe binds whatever the
 * installed AMD stack names its HIP runtime and runs CPU-only if none.
 */
#ifndef PS_HIP_WIN_SHIM_H
#define PS_HIP_WIN_SHIM_H

#include <stddef.h>
#include <stdint.h>

/* the code's CUDART_VERSION / CUDA_VERSION checks + the callback macro */
#ifndef CUDA_VERSION
#define CUDA_VERSION 0
#endif
#ifndef CUDART_VERSION
#define CUDART_VERSION CUDA_VERSION
#endif
#ifndef CUDART_CB
#define CUDART_CB
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ---- handle + scalar types ---- */
typedef int   hipError_t;
typedef int   hipDevice_t;
typedef void* hipDeviceptr_t;
typedef void* hipCtx_t;
typedef void* hipModule_t;
typedef void* hipFunction_t;
typedef void* hipStream_t;
typedef void* hipEvent_t;
typedef int   hipMemcpyKind;
typedef int   hipDeviceAttribute_t;
typedef void (*hipStreamCallback_t)(hipStream_t, hipError_t, void*);

/* ---- constants (ROCm 7.2 values, ABI-stable) ---- */
#define hipSuccess                                          0
#define hipMemcpyHostToDevice                               1
#define hipMemcpyDeviceToHost                               2
#define hipStreamNonBlocking                                0x01
#define hipEventBlockingSync                                0x1
#define hipEventDisableTiming                               0x2
#define hipHostMallocDefault                                0x0
#define hipDeviceScheduleBlockingSync                       0x4
#define hipDeviceAttributeMultiprocessorCount               63
#define hipDeviceAttributeMaxSharedMemoryPerBlock           74
#define hipDeviceAttributeMaxSharedMemoryPerMultiprocessor  10002
#define hipDeviceAttributeComputeCapabilityMajor            23
#define hipDeviceAttributeComputeCapabilityMinor            61
#define hipDeviceAttributeWarpSize                          87

/* ---- the amdhip64 C ABI this host uses, one X-macro list ----
 * PS_HIP_API(_) expands _(ret, name, (paramtypes)) per function. */
#define PS_HIP_API(_) \
  _(hipError_t,  hipInit,                        (unsigned int)) \
  _(hipError_t,  hipDeviceGet,                   (hipDevice_t*, int)) \
  _(hipError_t,  hipDeviceGetName,               (char*, int, hipDevice_t)) \
  _(hipError_t,  hipDeviceGetAttribute,          (int*, hipDeviceAttribute_t, int)) \
  _(hipError_t,  hipCtxCreate,                   (hipCtx_t*, unsigned int, hipDevice_t)) \
  _(hipError_t,  hipSetDevice,                   (int)) \
  _(hipError_t,  hipSetDeviceFlags,              (unsigned int)) \
  _(hipError_t,  hipMemGetInfo,                  (size_t*, size_t*)) \
  _(hipError_t,  hipModuleLoadData,              (hipModule_t*, const void*)) \
  _(hipError_t,  hipModuleGetFunction,           (hipFunction_t*, hipModule_t, const char*)) \
  _(hipError_t,  hipModuleGetGlobal,             (hipDeviceptr_t*, size_t*, hipModule_t, const char*)) \
  _(hipError_t,  hipModuleLaunchKernel,          (hipFunction_t, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, hipStream_t, void**, void**)) \
  _(hipError_t,  hipMalloc,                      (void**, size_t)) \
  _(hipError_t,  hipFree,                        (void*)) \
  _(hipError_t,  hipHostMalloc,                  (void**, size_t, unsigned int)) \
  _(hipError_t,  hipHostFree,                    (void*)) \
  _(hipError_t,  hipMemcpy,                      (void*, const void*, size_t, hipMemcpyKind)) \
  _(hipError_t,  hipMemcpyHtoD,                  (hipDeviceptr_t, const void*, size_t)) \
  _(hipError_t,  hipMemcpyHtoDAsync,             (hipDeviceptr_t, const void*, size_t, hipStream_t)) \
  _(hipError_t,  hipMemcpyDtoHAsync,             (void*, hipDeviceptr_t, size_t, hipStream_t)) \
  _(hipError_t,  hipStreamCreateWithPriority,    (hipStream_t*, unsigned int, int)) \
  _(hipError_t,  hipStreamCreateWithFlags,       (hipStream_t*, unsigned int)) \
  _(hipError_t,  hipStreamQuery,                 (hipStream_t)) \
  _(hipError_t,  hipStreamSynchronize,           (hipStream_t)) \
  _(hipError_t,  hipStreamDestroy,               (hipStream_t)) \
  _(hipError_t,  hipStreamWaitEvent,             (hipStream_t, hipEvent_t, unsigned int)) \
  _(hipError_t,  hipStreamAddCallback,           (hipStream_t, hipStreamCallback_t, void*, unsigned int)) \
  _(hipError_t,  hipDeviceGetStreamPriorityRange,(int*, int*)) \
  _(hipError_t,  hipEventCreateWithFlags,        (hipEvent_t*, unsigned int)) \
  _(hipError_t,  hipEventRecord,                 (hipEvent_t, hipStream_t)) \
  _(hipError_t,  hipEventDestroy,                (hipEvent_t)) \
  _(const char*, hipGetErrorString,              (hipError_t))

/* runtime-resolved function pointers (defined in hip_win_loader.cpp) */
#define PS_HIP_DECL_PTR(ret, name, params) extern ret (*name) params;
PS_HIP_API(PS_HIP_DECL_PTR)
#undef PS_HIP_DECL_PTR

int psHipLoadRuntime(void);   /* 0 on success; LoadLibrary/dlopen + resolve */

#ifdef __cplusplus
}
#endif

/* dim3 (the real hip header supplies this on the non-win builds) */
struct dim3 { unsigned int x, y, z; dim3(unsigned int X = 1, unsigned int Y = 1, unsigned int Z = 1) : x(X), y(Y), z(Z) {} };

#endif /* PS_HIP_WIN_SHIM_H */
