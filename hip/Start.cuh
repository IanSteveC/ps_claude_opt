#ifndef PS_HIP_WIN
#include "hip/hip_runtime.h"
#endif
#pragma once

extern "C" __global__ void CudaBuildDsphT(void);

extern "C" __global__ void CudaCalculatePrepare(int n_start, int n_max);

//__global__ void CudaCalculatePreparePole(int m, double freq_start, double freq_step, int n);
// Pole-merged: each bid = (freq, pole); pole/freq derived from tid, beta/lambda from
// __constant__ CUDA_beta_pole/CUDA_lambda_pole.
extern "C" __global__ void CudaCalculatePreparePole(mreal freq_start, mreal freq_step, int n);

extern "C" __global__ void CudaCalculateIter1Begin(int n_max);

extern "C" __global__ void CudaCalculateIter1Mrqmin1End(void);

extern "C" __global__ void CudaCalculateIter1Mrqmin2End(void);

extern "C" __global__ void CudaCalculateIter1Mrqcof1Start(void);

extern "C" __global__ void CudaCalculateIter1Mrqcof1Matrix(int lpoints);

extern "C" __global__ void CudaCalculateIter1Mrqcof1Curve1I0IA0(int lpoints);

extern "C" __global__ void CudaCalculateIter1Mrqcof1Curve1I0IA1(int lpoints);

extern "C" __global__ void CudaCalculateIter1Mrqcof1Curve1I1(int lpoints);

extern "C" __global__ void CudaCalculateIter1Mrqcof1Curve2I0IA0(void);

extern "C" __global__ void CudaCalculateIter1Mrqcof1Curve2I0IA1(void);

extern "C" __global__ void CudaCalculateIter1Mrqcof1Curve2I1IA0(void);

extern "C" __global__ void CudaCalculateIter1Mrqcof1Curve2I1IA1(void);

extern "C" __global__ void CudaCalculateIter1Mrqcof1CurveM1(int inrel, int lpoints);

extern "C" __global__ void CudaCalculateIter1Mrqcof1CurveM12I0IA0(int lpoints);

extern "C" __global__ void CudaCalculateIter1Mrqcof1CurveM12I0IA1(int lpoints);

extern "C" __global__ void CudaCalculateIter1Mrqcof1CurveM12I1IA0(int lpoints);

extern "C" __global__ void CudaCalculateIter1Mrqcof1CurveM12I1IA1(int lpoints);

extern "C" __global__ void CudaCalculateIter1Mrqcof1Curve1Mid(const int lpoints);
extern "C" __global__ void CudaCalculateIter1Mrqcof2Curve1Mid(const int lpoints);
/* wave64 forms: one wave per bid, block (64, PS_W64_BLOCKY); launched instead
   of the two above when the device warp size is 64 (gfx9/CDNA) */
extern "C" __global__ void CudaCalculateIter1Mrqcof1Curve1MidW64(const int lpoints);
extern "C" __global__ void CudaCalculateIter1Mrqcof2Curve1MidW64(const int lpoints);
extern "C" __global__ void CudaCalculateIter1Mrqcof1Curve2MidI1IA0(const int lpoints);
extern "C" __global__ void CudaCalculateIter1Mrqcof1Curve2MidI1IA1(const int lpoints);
extern "C" __global__ void CudaCalculateIter1Mrqcof2Curve2MidI1IA0(const int lpoints);
extern "C" __global__ void CudaCalculateIter1Mrqcof2Curve2MidI1IA1(const int lpoints);


extern "C" __global__ void CudaCalculateIter1Mrqcof1Curve1LastI0(void);

extern "C" __global__ void CudaCalculateIter1Mrqcof1Curve1LastI1(void);

extern "C" __global__ void CudaCalculateIter1Mrqcof1End(void);

extern "C" __global__ void CudaCalculateIter1Mrqcof2Start(void);

extern "C" __global__ void CudaCalculateIter1Mrqcof2Matrix(int lpoints);

extern "C" __global__ void CudaCalculateIter1Mrqcof1Curve1I0IA0(int lpoints);

extern "C" __global__ void CudaCalculateIter1Mrqcof1Curve1I0IA1(int lpoints);

extern "C" __global__ void CudaCalculateIter1Mrqcof1Curve1I1IA0(int lpoints);

extern "C" __global__ void CudaCalculateIter1Mrqcof1Curve1I1IA1(int lpoints);

extern "C" __global__ void CudaCalculateIter1Mrqcof2Curve1I0(int lpoints);

extern "C" __global__ void CudaCalculateIter1Mrqcof2Curve1I1(int lpoints);

extern "C" __global__ void CudaCalculateIter1Mrqcof2Curve2I0IA0(void);

extern "C" __global__ void CudaCalculateIter1Mrqcof2Curve2I0IA1(void);

extern "C" __global__ void CudaCalculateIter1Mrqcof2Curve2I1IA0(void);

extern "C" __global__ void CudaCalculateIter1Mrqcof2Curve2I1IA1(void);

extern "C" __global__ void CudaCalculateIter1Mrqcof2CurveM1(int inrel, int lpoints);

extern "C" __global__ void CudaCalculateIter1Mrqcof2CurveM12I0IA0(int lpoints);

extern "C" __global__ void CudaCalculateIter1Mrqcof2CurveM12I0IA1(int lpoints);

extern "C" __global__ void CudaCalculateIter1Mrqcof2CurveM12I1IA0(int lpoints);

extern "C" __global__ void CudaCalculateIter1Mrqcof2CurveM12I1IA1(int lpoints);

extern "C" __global__ void CudaCalculateIter1Mrqcof2Curve1LastI1(void);

extern "C" __global__ void CudaCalculateIter1Mrqcof2Curve1LastI0(void);

extern "C" __global__ void CudaCalculateIter1Mrqcof2End(void);

extern "C" __global__ void CudaCalculateIter2(void);

extern "C" __global__ void CudaCalculateFinishPole(void);

extern "C" __global__ void CudaCalculateFinish(void);