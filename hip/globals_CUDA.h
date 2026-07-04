#include "hip/hip_runtime.h"
#pragma once

#ifndef PS_DRIVER_API
#include <hip/hip_runtime_api.h>
#endif

//  NOTE Fake declaration to satisfy intellisense. See https://stackoverflow.com/questions/39980645/enable-code-indexing-of-cuda-in-clion/39990500
#if !defined(__HIPCC__) && !defined(PS_DRIVER_API)
//#define __host__
//#define __device__
//#define __shared__
//#define __constant__
//#define __global__
//#define __host__
#include <hip/device_functions.h>
#include <hip/hip_vector_types.h>
#include <hip/driver_types.h>
#include <texture_types.h>
#include <hip/hip_texture_types.h>
//#define __HIPCC__
#define __CUDA__
inline void __syncthreads() {};
inline void atomicAdd(int*, int) {};

//template <class T>
//static __device__ T tex1Dfetch(texture<int2, 1> texObject, int x) { return {}; };

__device__ __device_builtin__ double __hiloint2double(int hi, int lo);

//template<class T, int texType = hipTextureType1D, enum hipTextureReadMode mode = hipReadModeElementType>
//struct texture {};
//	int                          norm;
//	enum hipTextureFilterMode   fMode;
//	enum hipTextureAddressMode  aMode;
//	struct hipChannelFormatDesc desc;
//};

//#include <__clang_cuda_builtin_vars.h>
//#include <__clang_cuda_intrinsics.h>
//#include <__clang_cuda_math_forward_declares.h>
//#include <__clang_cuda_complex_builtins.h>
//#include <../../../../../../2019/Professional/VC/Tools/Llvm/lib/clang/9.0.0/include/__clang_cuda_cmath.h>
#endif

//#ifdef __INTELLISENSE__
////#define __device__ \
////			__location__(device)
//#endif



#include "constants.h"
//NOTE: https://devtalk.nvidia.com/default/topic/517801/-34-texture-is-not-a-template-34-error-mvs-2010/

// One block ("bid") = one (frequency, pole) pair since the pole merge;
// must hold one full batch: (freqs per batch) * N_POLES, rounded up to 128.
#define N_BLOCKS 4096


#ifndef PS_DRIVER_API
//global to all freq
__constant__ extern int CUDA_Ncoef, CUDA_Numfac, CUDA_Numfac1, CUDA_Dg_block;
__constant__ extern int CUDA_ma, CUDA_mfit, /*CUDA_mfit1,*/ CUDA_lastone, CUDA_lastma, CUDA_ncoef0;
__constant__ extern double CUDA_cg_first[MAX_N_PAR + 1];
__constant__ extern int CUDA_n_iter_max, CUDA_n_iter_min, CUDA_ndata;
__constant__ extern double CUDA_iter_diff_max;
__constant__ extern double CUDA_conw_r;
__constant__ extern int CUDA_Lmax, CUDA_Mmax;
__constant__ extern double CUDA_lcl, CUDA_Alamda_start, CUDA_Alamda_incr, CUDA_Alamda_incrr;
__constant__ extern double CUDA_Phi_0;
__constant__ extern double CUDA_beta_pole[N_POLES + 1];
__constant__ extern double CUDA_lambda_pole[N_POLES + 1];

__device__ extern double CUDA_par[4];
__device__ extern int CUDA_ia[MAX_N_PAR + 1];
__device__ extern double CUDA_Nor[3][MAX_N_FAC + 1];
__device__ extern double CUDA_Fc[MAX_LM + 1][MAX_N_FAC + 1];
__device__ extern double CUDA_Fs[MAX_LM + 1][MAX_N_FAC + 1];

__device__ extern double CUDA_Pleg[MAX_LM + 1][MAX_LM + 1][MAX_N_FAC + 1];
__device__ extern double CUDA_Darea[MAX_N_FAC + 1]; 
__device__ extern double CUDA_Dsph[MAX_N_PAR + 1][MAX_N_FAC + 1];

__device__ extern int CUDA_End;
__device__ extern int CUDA_Is_Precalc;

__device__ extern double CUDA_tim[MAX_N_OBS + 1];
__device__ extern double CUDA_brightness[MAX_N_OBS+1];
__device__ extern double CUDA_sig[MAX_N_OBS+1];
__device__ extern double CUDA_sigr2[MAX_N_OBS+1]; // (1/CUDA_sig^2) /*[MAX_N_OBS+1]*/;
__device__ extern double CUDA_Weight[MAX_N_OBS+1];
__device__ extern double CUDA_ee[3][MAX_N_OBS+1];
__device__ extern double CUDA_ee0[3][MAX_N_OBS+1];
#endif /* !PS_DRIVER_API */


// dytemp is transposed since the 2026 rewrite: row = data point, column = parameter,
// row stride DYT_STRIDE doubles (ma <= 63 is asserted on the host).
#define DYT_STRIDE 64

//global to one thread
struct freq_context
{
  double *Dg;   // unused since the DsphT rewrite (kept for struct/copy layout)
  //double *alpha;
  double *covar;
  double *dytemp;
  double *ytemp;

  //double cg[MAX_N_PAR + 1];
  //double beta[MAX_N_PAR + 1];
  double da[MAX_N_PAR + 1];
};

#ifndef PS_DRIVER_API
extern __device__ double *CUDA_Dg;

__device__ extern freq_context *CUDA_CC;
#endif

/*
struct freq_result
{
	int isReported;
	double dark_best, per_best, dev_best, la_best, be_best;
};
*/

//__device__ extern freq_result *CUDA_FR;
//LFR
#ifndef PS_DRIVER_API
__managed__ extern int isReported[N_BLOCKS];
__managed__ extern double dark_best[N_BLOCKS];
__managed__ extern double per_best[N_BLOCKS];
__managed__ extern double dev_best[N_BLOCKS];
__managed__ extern double la_best[N_BLOCKS];
__managed__ extern double be_best[N_BLOCKS];
#else
/* driver-API build: the __managed__ module variables are reached through
   host pointers resolved once from the fatbin (hipModuleGetGlobal returns a
   host-dereferenceable pointer for managed variables); same names keep the
   indexing code identical. Defined/initialized in cuda_iface.h. */
extern int    *isReported;
extern double *dark_best, *per_best, *dev_best, *la_best, *be_best;
#endif
