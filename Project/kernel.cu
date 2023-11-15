#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 1024
#define TYPE double

__global__ void gpuVectorAdd(TYPE* bufferIn1, TYPE* bufferIn2, TYPE* bufferOut)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	bufferOut[tid] = bufferIn1[tid] + bufferIn2[tid];
}

int main(int argc, char** argv)
{
	TYPE* cpuBuffer1;
	TYPE* cpuBuffer2;
	TYPE* cpuBuffer3;

	TYPE* gpuBuffer1;
	TYPE* gpuBuffer2;
	TYPE* gpuBuffer3;

	srand(time(NULL));

	cpuBuffer1 = (TYPE*)malloc(SIZE * sizeof(TYPE));
	cpuBuffer2 = (TYPE*)malloc(SIZE * sizeof(TYPE));
	cpuBuffer3 = (TYPE*)malloc(SIZE * sizeof(TYPE));

	cudaMalloc((void**)&gpuBuffer1, SIZE * sizeof(TYPE));
	cudaMalloc((void**)&gpuBuffer2, SIZE * sizeof(TYPE));
	cudaMalloc((void**)&gpuBuffer3, SIZE * sizeof(TYPE));

	for (int i = 0; i != SIZE; ++i)
	{
		cpuBuffer1[i] = (TYPE)rand() / (TYPE)RAND_MAX;
		cpuBuffer2[i] = (TYPE)rand() / (TYPE)RAND_MAX;
	}

	cudaMemcpy(gpuBuffer1, cpuBuffer1, SIZE * sizeof(TYPE), cudaMemcpyHostToDevice);
	cudaMemcpy(gpuBuffer2, cpuBuffer2, SIZE * sizeof(TYPE), cudaMemcpyHostToDevice);

	gpuVectorAdd<<<32, 32>>>(gpuBuffer1, gpuBuffer2, gpuBuffer3);

	cudaDeviceSynchronize();

	cudaMemcpy(cpuBuffer3, gpuBuffer3, SIZE * sizeof(TYPE), cudaMemcpyDeviceToHost);

	int errorCount = 0;

	for (int i = 0; i != SIZE; ++i)
	{
		errorCount += ((cpuBuffer1[i] + cpuBuffer2[i]) != cpuBuffer3[i]);
	}

	printf("%d\n", errorCount);

	free(cpuBuffer1);
	free(cpuBuffer2);
	free(cpuBuffer3);

	cudaFree(gpuBuffer1);
	cudaFree(gpuBuffer2);
	cudaFree(gpuBuffer3);

	return 0;
}