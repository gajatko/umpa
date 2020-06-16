#pragma once

const size_t THREADS_IN_BLOCK = 128;
#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "stdio.h"


template <typename T1, typename T2, typename T3, typename T4>
void runKernel(void (*fun)(T1, T2, T3, T4), int Blocks, T1 arg1, T2 arg2, T3 arg3, T4 arg4) {
	printf("Starting kernel with %d blocks\n", Blocks);
	fun <<<Blocks, THREADS_IN_BLOCK>>> (arg1, arg2, arg3, arg4);
}

template <typename T1, typename T2, typename T3>
void runKernel(void (*fun)(T1, T2, T3), int Blocks, T1 arg1, T2 arg2, T3 arg3) {
	printf("Starting kernel with %d blocks\n", Blocks);
	fun <<<Blocks, THREADS_IN_BLOCK>>> (arg1, arg2, arg3);
}

template <typename T1, typename T2>
void runKernel(void (*fun)(T1, T2), int Blocks, T1 arg1, T2 arg2) {
	printf("Starting kernel with %d blocks\n", Blocks);
	fun <<<Blocks, THREADS_IN_BLOCK>>> (arg1, arg2);
}

template <typename T1>
void runKernel(void (*fun)(T1), int Blocks, T1 arg1) {
	printf("Starting kernel with %d blocks\n", Blocks);
	fun <<<Blocks, THREADS_IN_BLOCK>>> (arg1);
}
