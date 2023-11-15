#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 4096
#define TYPE double

__global__ void gpuVectorAdd(TYPE* bufferIn1, TYPE* bufferIn2, TYPE* bufferOut)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	bufferOut[tid] = bufferIn1[tid] + bufferIn2[tid];
}

int main()
{
	TYPE* cpuBufferIn1;
	TYPE* cpuBufferIn2;
	TYPE* cpuBufferOut;

	TYPE* gpuBufferIn1;
	TYPE* gpuBufferIn2;
	TYPE* gpuBufferOut;

	srand(time(NULL));

	cpuBufferIn1 = (TYPE*)malloc(SIZE * sizeof(TYPE));
	cpuBufferIn2 = (TYPE*)malloc(SIZE * sizeof(TYPE));
	cpuBufferOut = (TYPE*)malloc(SIZE * sizeof(TYPE));

	cudaMalloc((void**)&gpuBufferIn1, SIZE * sizeof(TYPE));
	cudaMalloc((void**)&gpuBufferIn2, SIZE * sizeof(TYPE));
	cudaMalloc((void**)&gpuBufferOut, SIZE * sizeof(TYPE));

	for (int i = 0; i != SIZE; ++i)
	{
		cpuBufferIn1[i] = (TYPE)rand() / (TYPE)RAND_MAX;
		cpuBufferIn2[i] = (TYPE)rand() / (TYPE)RAND_MAX;
	}

	cudaMemcpy(gpuBufferIn1, cpuBufferIn1, SIZE * sizeof(TYPE), cudaMemcpyHostToDevice);
	cudaMemcpy(gpuBufferIn2, cpuBufferIn2, SIZE * sizeof(TYPE), cudaMemcpyHostToDevice);

	int threads = 32;
	int blocks  = (SIZE + threads - 1) / threads;

	gpuVectorAdd<<<blocks, threads>>>(gpuBufferIn1, gpuBufferIn2, gpuBufferOut);

	cudaDeviceSynchronize();

	cudaMemcpy(cpuBufferOut, gpuBufferOut, SIZE * sizeof(TYPE), cudaMemcpyDeviceToHost);

	int errorCount = 0;

	for (int i = 0; i != SIZE; ++i)
	{
		errorCount += ((cpuBufferIn1[i] + cpuBufferIn2[i]) != cpuBufferOut[i]);
	}

	printf("%d\n", errorCount);

	free(cpuBufferIn1);
	free(cpuBufferIn2);
	free(cpuBufferOut);

	cudaFree(gpuBufferIn1);
	cudaFree(gpuBufferIn2);
	cudaFree(gpuBufferOut);

	return 0;
}