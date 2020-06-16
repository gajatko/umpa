#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"
#include "kernel.cuh"

#include <cstring>
#include <algorithm>
#include <string>
#include <iostream>

using std::cout;
using std::endl;



void printDeviceInfo() { 
	cudaDeviceProp deviceProp;

	int devID = 0;

	auto error = cudaGetDeviceProperties(&deviceProp, devID);

	if (error != cudaSuccess)
	{
		printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
		exit(EXIT_FAILURE);
	}

	printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name,
		deviceProp.major, deviceProp.minor);
}


cudaError_t checkCudaError(cudaError_t cudaStatus, const char* message) {

	if (cudaStatus != cudaError_t::cudaSuccess) {
		printf( "Cuda error code: %d, '%s' msg: %s!\n", cudaStatus, message, cudaGetErrorString(cudaStatus));
		throw cudaStatus;
	}
	return cudaStatus;
}

cudaError_t checkCudaError(cudaError_t cudaStatus) {
	return checkCudaError(cudaStatus, "");
}


cudaError_t go(numeric_t** resultPtr, numeric_t** a, size_t size, int groups);



const char* concatStrs(const std::string desc, const std::string errStr) {
	return (desc + errStr).c_str();
}

const size_t CHUNK = 1000;
const numeric_t _min = -100;
const numeric_t _max = 100;
const numeric_t _delta = .000001;


int main(int argc, char** argv)
{

	printDeviceInfo();

	const size_t arraySize = ceil((_max - _min) / _delta);
	const size_t groupCount = arraySize % CHUNK == 0 ? arraySize / CHUNK : arraySize / CHUNK + 1;

	cout << "N = " << arraySize << "(" << arraySize / 1024.0 * sizeof(numeric_t) / 1000 / 1000 << " GB)" << endl;
	cout << "Allocating " << groupCount << " chunks of " << CHUNK * sizeof(numeric_t) << " bytes each " << endl;

	numeric_t** a = new numeric_t*[groupCount];
	numeric_t** result = new numeric_t*[groupCount];
	for (size_t i = 0; i < groupCount; i++) {
		a[i] = new numeric_t[CHUNK];
		result[i] = new numeric_t[CHUNK];
	}

	for (size_t i = 0; i < arraySize; i++) {
		size_t group = i / CHUNK;
		size_t x = i % CHUNK;
		a[group][x] = _delta * i + _min;
	}
	cudaError_t cudaStatus;

	go(result, a, arraySize, groupCount);

	cout << endl << endl;
	cout << "INPUT || OUTPUT" << endl;
	//for (int r = 0; r < arraySize; r++) {
		//if (r%10==0)
		//cout << r << ". f(" << a[0][r] << ") = " << exp(a[0][r]) << " | " << result[0][r] << endl;
	//}
	cout << endl << endl;
	float rows = 100;
	for (int r = 0; r < rows; r++) {
		int i = r / rows * arraySize;
		size_t group = i / CHUNK;
		size_t x = i % CHUNK;
		cout << r << ". f(" << a[group][x] << ") = " << exp(a[group][x]) << " | " << result[group][x] << endl;
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaDeviceReset();

	for (int i = 0; i < groupCount; i++) {
		delete[] a[i];
		delete[] result[i];
	} 
	delete[] a;
	delete[] result;

	getchar();

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t go(numeric_t** resultPtr, numeric_t** a, size_t size, int groups)
{
	if (size == 0) {
		return cudaError::cudaSuccess;
	}
	numeric_t* input = 0;
	numeric_t* output = 0;
	numeric_t* blockSum = 0;
	cudaError_t status = cudaError_t::cudaSuccess;

	try {
		cudaSetDevice(0);

		cout << "Allocating memory on device. ";
		checkCudaError(cudaMalloc((void**)&output, size * sizeof(numeric_t)), "cudaMalloc output");
		checkCudaError(cudaMalloc((void**)&input, size * sizeof(numeric_t)), "cudaMalloc input");


		cout << "Copying memory to device. ";
		for (int i = 0; i < groups; i++) {
			numeric_t* addr = input + i * CHUNK;
			checkCudaError(cudaMemcpy(addr, a[i], CHUNK * sizeof(numeric_t), cudaMemcpyHostToDevice), "copying memory");
		}
		//cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
		int B = std::max((size_t)1, (size_t)ceil((float)size / THREADS_IN_BLOCK));
		int SH = 0;// THREADS_IN_BLOCK * sizeof(numeric_t);

		checkCudaError(cudaMalloc((void**)&blockSum, B), "cudaMalloc cache");
		cout << "Blocks: " << B << " Threads: " << THREADS_IN_BLOCK << " Shared: " << SH << endl;

		// Launch a kernel on the GPU with one thread for each element.
		
		cout << "Calculating paramater values. " << endl;
		expKernel <<<B, THREADS_IN_BLOCK >>> (input, size);
		checkCudaError(cudaGetLastError(), "expKernel launch failed: ");
		cout << "Calculating inf areas. " << endl;
		intKernel <<<B, THREADS_IN_BLOCK >>> (input, _delta, size); 
		checkCudaError(cudaGetLastError(), "intKernel launch failed: ");
		cout << "Calculating partial sums. " << endl;
		sumKernel <<<B, THREADS_IN_BLOCK >>> (output, input, blockSum, size);
		checkCudaError(cudaGetLastError(), "sumKernel launch failed: ");
		

		B = std::max((size_t)1, (size_t)ceil(B / THREADS_IN_BLOCK));

		cout << "Merging results. " << endl;
		sumBlocksKernel <<<B, THREADS_IN_BLOCK >>> (output, blockSum, size);
		checkCudaError(cudaGetLastError(), "sumBlocksKernel launch failed: ");

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.

		cudaDeviceSynchronize();
		// Copy output vector from GPU buffer to host memory.

		cout << "Retrieving results from device. " << endl;
		for (int i = 0; i < groups; i++) {
			cudaMemcpy(resultPtr[i], output + i * CHUNK, CHUNK * sizeof(numeric_t), cudaMemcpyDeviceToHost);
		}

	}
	catch (cudaError_t e) {
		printf("cudaGetLastError: %s\n", cudaGetErrorString(e));
		status = e;
	}
	cudaFree(input);
	cudaFree(output);
	cudaFree(blockSum);

	return status;
}
