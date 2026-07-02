#pragma once

#define BLOCKX4 4
#define BLOCKX8 8
#define BLOCKX16 16
#define BLOCKX32 32

#define blockIdx() (blockIdx.x + gridDim.x * threadIdx.y) 

__device__ void curv(freq_context * __restrict__ CUDA_LCC,
		     double * __restrict__ cg,
		     int brtmpl, int brtmph,
		     int bid);
__device__ int mrqmin_1_end(freq_context * __restrict__ CUDA_LCC,
			    int ma, int mfit, int mfit1, int block);
__device__ void mrqmin_2_end(freq_context * __restrict__ CUDA_LCC,
			     int * __restrict__ ia, int ma);
__device__ void mrqcof_start(freq_context * __restrict__ CUDA_LCC,
			     double * __restrict__ a,
			     double * __restrict__ alpha,
			     double * __restrict__ beta,
			     int bid);
__device__ void mrqcof_matrix(freq_context * __restrict__ CUDA_LCC,
			      double * __restrict__ a,
			      int Lpoints);
__device__ void mrqcof_curve1(freq_context * __restrict__ CUDA_LCC,
			      double * __restrict__ a,
                              double * __restrict__ alpha,
			      double * __restrict__ beta,
			      int Inrel, int Lpoints);

__device__ void mrqcof_curve1_last(freq_context * __restrict__ CUDA_LCC,
				   double * __restrict__ a,
				   double * __restrict__ alpha,
				   double * __restrict__ beta,
				   int Inrel, int Lpoints);

__device__ void MrqcofCurve2(freq_context * __restrict__ CUDA_LCC,
			     double * __restrict__ alpha,
			     double * __restrict__ beta,
			     int inrel, int lpoints);

__device__ double mrqcof_end(freq_context * __restrict__ CUDA_LCC,
			     double * __restrict__ alpha);

__device__ double mrqcof(freq_context * __restrict__ CUDA_LCC,
			 double * __restrict__ a,
			 int * __restrict__ ia,
			 int ma,
                         double alpha[/*MAX_N_PAR+1*/][MAX_N_PAR+1],
			 double * __restrict__ beta,
			 int mfit, int lastone, int lastma);
//__device__ int gauss_errc(freq_context *CUDA_LCC,int n, double b[]);
__device__ int gauss_errc(freq_context * __restrict__ CUDA_LCC, int ma);
__device__ void blmatrix(freq_context * __restrict__ CUDA_LCC,
			 double bet, double lam);
__device__ double conv(freq_context * __restrict__ CUDA_LCC,
		       int nc,
		       double *dyda,
		       int bid);
__device__ double bright(freq_context * __restrict__ CUDA_LCC,
			 double * __restrict__ cg,
			 int jp, int Lpoints1, int Inrel);
__device__ void matrix_neo(freq_context * __restrict__ CUDA_LCC,
			   double const * __restrict__ cg,
			   int lnp1, int Lpoints);
__global__ void CudaCalculateIter1Mrqcof2Curve2(int inrel, int lpoints);
__global__ void CudaCalculateIter1Mrqcof1Curve2(int inrel, int lpoints);
