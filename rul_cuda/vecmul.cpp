

#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"
#include "kernel.cuh"
#include "DeviceMemory.h"

#include <cstring>
#include <algorithm>
#include <string>
#include <iostream>
#include <chrono>

using std::cout;
using std::endl;



void vecmul() {
	const size_t SIZE = 1 << 28;
	cout << "N = " << SIZE << " ( " << (float)SIZE * sizeof(numeric_t) / 1024 / 1000 / 1000 << " GB)" << endl;

	auto vector = new DeviceMemory<numeric_t>(SIZE)->hostMalloc();
	auto output = new DeviceMemory<numeric_t>(SIZE)->hostMalloc(); 
	numeric_t scalar = 2;


	for (size_t i = 0; i < SIZE; i++) {
		vector->host()[i] = 10 * (double)i/ SIZE; 
	}
	auto start = std::chrono::high_resolution_clock::now();

	auto devVector = vector->deviceMalloc()->copyToDevice();
	auto devOutput = output->deviceMalloc()->copyToDevice();

	auto startKernel = std::chrono::high_resolution_clock::now();
	runKernel(mulVec, devVector->requiredBlockCount(), devOutput->device(), devVector->device(), scalar, (size_t)SIZE); 

	cudaDeviceSynchronize(); 

	cout << "GPU kernel: " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - startKernel).count() << " us" << endl;
	devOutput->copyToHost();

	auto end = std::chrono::high_resolution_clock::now(); 
	auto dur = end - start; 
	auto millis = std::chrono::duration_cast<std::chrono::microseconds>(dur).count();

	cout << "GPU computation: " << millis << " us" << endl;

	vector->freeDeviceMemory();
	output->freeDeviceMemory();


	// check
	//for (size_t i = 0; i < SIZE && i < 20; i++) {
		//cout << output[i] << endl;
	//}

	// impl in software
	start = std::chrono::high_resolution_clock::now();

	for (size_t i = 0; i < SIZE; i++) {
		output->host()[i] = exp(sin(sqrt(vector->host()[i])));
	}
	end = std::chrono::high_resolution_clock::now();
	dur = end - start;
	millis = std::chrono::duration_cast<std::chrono::microseconds>(dur).count();
	cout << "CPU computation: " << millis << " us" << endl;


	delete vector;
	delete output;
}



