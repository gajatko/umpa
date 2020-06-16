#pragma once

#include "cuda_runtime.h"
#include "cuda.h"
#include "stdio.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

const size_t THREADS_IN_BLOCK = 1024;

typedef double numeric_t;

__global__ void sumKernel(numeric_t* out, numeric_t* f, numeric_t* blockSum, size_t N);

__global__ void intKernel(numeric_t* out, numeric_t delta, size_t N);

__global__ void expKernel(numeric_t* out, size_t N);

__global__ void sumBlocksKernel(numeric_t* sums, numeric_t* blockSum, size_t N);

