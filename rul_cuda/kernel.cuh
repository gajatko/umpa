#pragma once

#include "wrapper.cuh"
#include "stdio.h"
#include <functional>

typedef double numeric_t;

__global__ void sumKernel(numeric_t* out, numeric_t* f, numeric_t* blockSum, size_t N);

__global__ void intKernel(numeric_t* out, numeric_t* in, numeric_t delta, size_t N);

__global__ void expKernel(numeric_t* out, numeric_t* in, size_t N);

__global__ void sumBlocksKernel(numeric_t* sums, numeric_t* blockSum, size_t N);

__global__ void mulVec(numeric_t* out, numeric_t* in, numeric_t scalar, size_t N);
