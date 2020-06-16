

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

#define SIZE 100

void vecmul() {
	double* vector = new double[SIZE];
}
