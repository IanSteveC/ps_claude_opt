#pragma once
#ifndef PS_DRIVER_API
#include <cuda_runtime_api.h>
#else
#include "cuda_iface.h"  /* provides the cudaDeviceProp lookalike */
#endif
class Cc
{
	int deviceCcMajor;
	int deviceCcMinor;

	int GetSmxBlockCuda12() const;
	int GetSmxBlockCuda11() const;
	int GetSmxBlockCuda10() const;
	int GetSmxBlockCc8() const;
	int GetSmxBlockCc7() const;
	int GetSmxBlockCc6() const;
	int GetSmxBlockCc5() const;
	int GetSmxBlockCc3() const;
	void Exit() const;
	
public:
	int cudaVersion;

	explicit Cc(const cudaDeviceProp deviceProp);
	~Cc();
	int GetSmxBlock() const;
};
