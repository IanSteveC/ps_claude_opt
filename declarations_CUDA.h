#pragma once

#define BLOCKX4 4
#define BLOCKX8 8
#define BLOCKX16 16
#define BLOCKX32 32

#define blockIdx() (blockIdx.x + gridDim.x * threadIdx.y) 

#ifndef PS_DRIVER_API /* device declarations - meaningless to the mingw host build */
__device__ void curv(freq_context * __restrict__ CUDA_LCC,
		     mreal * __restrict__ cg,
		     int brtmpl, int brtmph,
		     int bid);
__device__ int mrqmin_1_end(freq_context * __restrict__ CUDA_LCC,
			    int ma, int mfit, int mfit1, int block);
__device__ void mrqmin_2_end(freq_context * __restrict__ CUDA_LCC,
			     int * __restrict__ ia, int ma);
__device__ void mrqcof_start(freq_context * __restrict__ CUDA_LCC,
			     mreal * __restrict__ a,
			     mreal * __restrict__ alpha,
			     mreal * __restrict__ beta,
			     int bid);
__device__ void mrqcof_matrix(freq_context * __restrict__ CUDA_LCC,
			      mreal * __restrict__ a,
			      int Lpoints);
__device__ void mrqcof_curve1(freq_context * __restrict__ CUDA_LCC,
			      mreal * __restrict__ a,
                              mreal * __restrict__ alpha,
			      mreal * __restrict__ beta,
			      int Inrel, int Lpoints);

__device__ void mrqcof_curve1_last(freq_context * __restrict__ CUDA_LCC,
				   mreal * __restrict__ a,
				   mreal * __restrict__ alpha,
				   mreal * __restrict__ beta,
				   int Inrel, int Lpoints);

__device__ void MrqcofCurve2(freq_context * __restrict__ CUDA_LCC,
			     mreal * __restrict__ alpha,
			     mreal * __restrict__ beta,
			     int inrel, int lpoints);

__device__ mreal mrqcof_end(freq_context * __restrict__ CUDA_LCC,
			     mreal * __restrict__ alpha);

__device__ mreal mrqcof(freq_context * __restrict__ CUDA_LCC,
			 mreal * __restrict__ a,
			 int * __restrict__ ia,
			 int ma,
                         mreal alpha[/*MAX_N_PAR+1*/][MAX_N_PAR+1],
			 mreal * __restrict__ beta,
			 int mfit, int lastone, int lastma);
//__device__ int gauss_errc(freq_context *CUDA_LCC,int n, mreal b[]);
__device__ int gauss_errc(freq_context * __restrict__ CUDA_LCC, int ma);
__device__ void blmatrix(freq_context * __restrict__ CUDA_LCC,
			 mreal bet, mreal lam);
__device__ mreal conv(freq_context * __restrict__ CUDA_LCC,
		       int nc,
		       mreal *dyda,
		       int bid);
__device__ mreal bright(freq_context * __restrict__ CUDA_LCC,
			 mreal * __restrict__ cg,
			 int jp, int Lpoints1, int Inrel);
__device__ void matrix_neo(freq_context * __restrict__ CUDA_LCC,
			   mreal const * __restrict__ cg,
			   int lnp1, int Lpoints);
extern "C" __global__ void CudaCalculateIter1Mrqcof2Curve2(int inrel, int lpoints);
extern "C" __global__ void CudaCalculateIter1Mrqcof1Curve2(int inrel, int lpoints);
#endif /* !PS_DRIVER_API */
