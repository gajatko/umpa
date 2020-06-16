#pragma once
#include "wrapper.cuh"
#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <iostream>
#include <string>

using std::cout;
using std::endl;

template<typename T>
class DeviceMemory
{
private:
	bool externalHostMemory, externalDeviceMemory;
	T* hostPtr;
	T* devicePtr;
	size_t size;
	size_t sizeInBytes;
	static size_t* totalAvailibleMemory = 0;

	cudaError_t checkCudaError(cudaError_t cudaStatus, const char* message) { 
		if (cudaStatus != cudaError_t::cudaSuccess) {
			printf( "Cuda error code: %d, '%s' msg: %s!\n", cudaStatus, message, cudaGetErrorString(cudaStatus));
			throw std::string(cudaGetErrorString(cudaStatus));
		}
		return cudaStatus;
	}

	void checkMemoryInitialized() {
		if (hostPtr == nullptr) {
			cout << "Host memory uninitialized.";
			throw 1;
		}
		if (devicePtr == nullptr) {
			cout << "Device memory uninitialized.";
			throw 1;
		}
	}

public:

	DeviceMemory(size_t size, T* hostPtr = nullptr, T* devicePtr = nullptr) {
		this->size = size;
		this->sizeInBytes = size * sizeof(numeric_t);
		externalDeviceMemory = devicePtr != nullptr;
		this->devicePtr = devicePtr;
		externalHostMemory = hostPtr != nullptr;
		this->hostPtr = hostPtr;
		if (externalHostMemory) {
			cudaHostRegister(hostPtr, size, cudaHostRegisterDefault);
		}
	}

	~DeviceMemory() {
		cout << "Destroying DeviceMemory object...";
		freeDeviceMemory();
		freeHostMemory();
		cout << endl;
	}

	int requiredBlockCount() { return size % THREADS_IN_BLOCK == 0 ? size / THREADS_IN_BLOCK : size / THREADS_IN_BLOCK + 1; }
	T* device() { return devicePtr; }
	T* host() { return hostPtr; }

	DeviceMemory* hostMalloc() {
		if (hostPtr != nullptr) {
			cout << "Host memory already allocated. Create another DeviceMemory object or call freeHostMemory to allocate again.";
			throw 1;
		}
		cout << "Allocating " << sizeInBytes << " bytes on host..." << endl;
		
		checkCudaError(cudaMallocHost((void**)&hostPtr, sizeInBytes), "cudaHostAlloc");
		externalHostMemory = false;
		return this;
	}

	DeviceMemory* deviceMalloc() {
		if (devicePtr != nullptr) {
			cout << "Device memory already allocated. Create another DeviceMemory object or call freeDeviceMemory to allocate again.";
			throw 1;
		}
		cout << "Allocating " << sizeInBytes << " bytes on device..." << endl;
		//checkCudaError(cudaMalloc((void**)&devicePtr, sizeInBytes), "cudaMalloc");
		checkCudaError(cudaMallocManaged((void**)&devicePtr, sizeInBytes), "cudaMallocManaged");
		externalDeviceMemory = false;
		return this;
	}


	DeviceMemory* copyToDevice() {
		checkMemoryInitialized();
		cout << "Copying " << sizeInBytes << " to device..." << endl;
		checkCudaError(cudaMemcpy(devicePtr, hostPtr, sizeInBytes, cudaMemcpyKind::cudaMemcpyHostToDevice), "copyToDevice");
		return this;
	}

	DeviceMemory* copyToHost() {
		checkMemoryInitialized();
		cout << "Copying " << sizeInBytes << " to host..." << endl;
		checkCudaError(cudaMemcpy(hostPtr, devicePtr, sizeInBytes, cudaMemcpyKind::cudaMemcpyDeviceToHost), "copyToHost");
		return this;
	} 

	DeviceMemory* freeHostMemory() {
		if (hostPtr != nullptr && !externalHostMemory) {
			cudaFreeHost(hostPtr);
			hostPtr = nullptr;
		}
		else if (hostPtr != nullptr && externalHostMemory) {
			cudaHostUnregister(hostPtr);
			cout << " Host memory won't be freed because it wasn't allocated by this DeviceMemory.";
		}
		return this;
	}

	DeviceMemory* freeDeviceMemory() {
		if (devicePtr != nullptr && !externalDeviceMemory) {
			cudaFree(devicePtr);
			devicePtr = nullptr;
		} else if (devicePtr != nullptr && externalDeviceMemory) {
			cout << " Device memory won't be freed because it wasn't allocated by this DeviceMemory.";
		}
		return this;
	}
};


