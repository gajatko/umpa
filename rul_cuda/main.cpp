#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

int integrate();
void vecmul();

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

int main() {
	printDeviceInfo();
	vecmul();
	//integrate();
}
