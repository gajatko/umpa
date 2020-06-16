#include "kernel.cuh"
#include <thrust/scan.h>

__global__ void expKernel(numeric_t* out, size_t N)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < N) {
		out[x] = exp(out[x]);
	}
}

__global__ void intKernel(numeric_t* out, numeric_t delta, size_t N)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < N) {
		out[x] = out[x] * delta;
		if (x % 1 == 0) {
			//printf("intKernel %d f(%f) = %f\n", x, f[x], out[x]);
		}
	}
}

__global__ void sumKernel(numeric_t* out, numeric_t* f, numeric_t* blockSum, size_t N)
{
	int tid = threadIdx.x;
	int off = blockIdx.x * blockDim.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	numeric_t sum = 0.0;
	for (int j = off; j < off + tid; j++)
	{
		sum += f[j];
	}
	out[i] == sum;

	if (threadIdx.x == blockDim.x - 1) { 
		blockSum[blockIdx.x] = sum;
	}
}


__global__ void sumBlocksKernel(numeric_t* sums, numeric_t* blockSum, size_t N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	numeric_t sum = 0;
	for (int j = 0; j < blockIdx.x; j++) {
		sum += blockSum[j];
	}
	sums[i] = sums[i] + sum;
}

