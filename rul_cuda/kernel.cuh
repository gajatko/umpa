#pragma once

#include "cuda_runtime.h"
#include "cuda.h"
#include "stdio.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <functional>

	const size_t THREADS_IN_BLOCK = 1024;

	typedef double numeric_t;

	__global__ void sumKernel(numeric_t* out, numeric_t* f, numeric_t* blockSum, size_t N);

	__global__ void intKernel(numeric_t* out, numeric_t* in, numeric_t delta, size_t N);

	__global__ void expKernel(numeric_t* out, numeric_t* in, size_t N);

	__global__ void sumBlocksKernel(numeric_t* sums, numeric_t* blockSum, size_t N);

	__global__ void ss();

template <typename T1, typename T2, typename T3, typename T4>
void runKernel(void (*Functor)(T1, T2, T3, T4), int Blocks, T1 arg1, T2 arg2, T3 arg3, T4 arg4) {
	Functor << <Blocks, THREADS_IN_BLOCK >> > (arg1, arg2, arg3, arg4);
}

template <typename T1, typename T2, typename T3>
void runKernel(void (*fun)(T1, T2, T3), int Blocks, T1 arg1, T2 arg2, T3 arg3) {
	fun << <Blocks, THREADS_IN_BLOCK >> > (arg1, arg2, arg3);
}

template <typename T1, typename T2>
void runKernel(void (*fun)(T1, T2), int Blocks, T1 arg1, T2 arg2) {
	fun << <Blocks, THREADS_IN_BLOCK >> > (arg1, arg2);
}

template <typename T1>
void runKernel(void (*fun)(T1), int Blocks, T1 arg1) {
	fun << <Blocks, THREADS_IN_BLOCK >> > (arg1);
}

