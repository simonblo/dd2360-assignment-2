#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 16777216
#define TYPE double

__global__ void gpuVectorAdd(TYPE* bufferIn1, TYPE* bufferIn2, TYPE* bufferOut, int bufferSize)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < bufferSize) bufferOut[tid] = bufferIn1[tid] + bufferIn2[tid];
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

	clock_t time0 = clock();

	cudaMemcpy(gpuBufferIn1, cpuBufferIn1, SIZE * sizeof(TYPE), cudaMemcpyHostToDevice);
	cudaMemcpy(gpuBufferIn2, cpuBufferIn2, SIZE * sizeof(TYPE), cudaMemcpyHostToDevice);

	clock_t time1 = clock();

	int threads = 64;
	int blocks = (SIZE + threads - 1) / threads;

	gpuVectorAdd<<<blocks, threads>>>(gpuBufferIn1, gpuBufferIn2, gpuBufferOut, SIZE);
	cudaDeviceSynchronize();

	clock_t time2 = clock();

	cudaMemcpy(cpuBufferOut, gpuBufferOut, SIZE * sizeof(TYPE), cudaMemcpyDeviceToHost);

	clock_t time3 = clock();

	int errorCount = 0;

	for (int i = 0; i != SIZE; ++i)
	{
		errorCount += ((cpuBufferIn1[i] + cpuBufferIn2[i]) != cpuBufferOut[i]);
	}

	printf("Elements:      %d\n", SIZE);
	printf("Errors:        %d\n", errorCount);
	printf("CPU-GPU Copy:  %.3f seconds\n", (double)(time1 - time0) / (double)CLOCKS_PER_SEC);
	printf("GPU Execution: %.3f seconds\n", (double)(time2 - time1) / (double)CLOCKS_PER_SEC);
	printf("GPU-CPU Copy:  %.3f seconds\n", (double)(time3 - time2) / (double)CLOCKS_PER_SEC);

	cudaFree(gpuBufferIn1);
	cudaFree(gpuBufferIn2);
	cudaFree(gpuBufferOut);

	free(cpuBufferIn1);
	free(cpuBufferIn2);
	free(cpuBufferOut);

	return 0;
}