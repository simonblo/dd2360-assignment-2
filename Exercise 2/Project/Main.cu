#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DIMU 511
#define DIMV 1023
#define DIMW 4094
#define TYPE double

__global__ void gpuVectorAdd(TYPE* matrixA, TYPE* matrixB, TYPE* matrixC)
{
	uint2 tid;
	tid.x = threadIdx.x + blockIdx.x * blockDim.x;
	tid.y = threadIdx.y + blockIdx.y * blockDim.y;

	if (tid.x >= DIMW) return;
	if (tid.y >= DIMU) return;

	TYPE value = (TYPE)0;

	for (int v = 0; v != DIMV; ++v)
	{
		value += matrixA[v + tid.y * DIMV] * matrixB[tid.x + v * DIMW];
	}

	matrixC[tid.x + tid.y * DIMW] = value;
}

int main()
{
	TYPE* cpuMatrixA;
	TYPE* cpuMatrixB;
	TYPE* cpuMatrixC;
	TYPE* gpuMatrixA;
	TYPE* gpuMatrixB;
	TYPE* gpuMatrixC;

	srand(time(NULL));

	cpuMatrixA = (TYPE*)malloc(DIMU * DIMV * sizeof(TYPE));
	cpuMatrixB = (TYPE*)malloc(DIMV * DIMW * sizeof(TYPE));
	cpuMatrixC = (TYPE*)malloc(DIMU * DIMW * sizeof(TYPE));

	cudaMalloc((void**)&gpuMatrixA, DIMU * DIMV * sizeof(TYPE));
	cudaMalloc((void**)&gpuMatrixB, DIMV * DIMW * sizeof(TYPE));
	cudaMalloc((void**)&gpuMatrixC, DIMU * DIMW * sizeof(TYPE));

	for (int i = 0; i != DIMU * DIMV; ++i) cpuMatrixA[i] = (TYPE)rand() / (TYPE)RAND_MAX;
	for (int i = 0; i != DIMV * DIMW; ++i) cpuMatrixB[i] = (TYPE)rand() / (TYPE)RAND_MAX;

	cudaMemcpy(gpuMatrixA, cpuMatrixA, DIMU * DIMV * sizeof(TYPE), cudaMemcpyHostToDevice);
	cudaMemcpy(gpuMatrixB, cpuMatrixB, DIMV * DIMW * sizeof(TYPE), cudaMemcpyHostToDevice);

	dim3 threads;
	threads.x = 8;
	threads.y = 8;
	threads.z = 1;

	dim3 blocks;
	blocks.x = (DIMW + threads.x - 1) / threads.x;
	blocks.y = (DIMU + threads.y - 1) / threads.y;
	blocks.z = 1;

	gpuVectorAdd<<<blocks, threads>>>(gpuMatrixA, gpuMatrixB, gpuMatrixC);
	cudaDeviceSynchronize();

	cudaMemcpy(cpuMatrixC, gpuMatrixC, DIMU * DIMW * sizeof(TYPE), cudaMemcpyDeviceToHost);

	int errorCount = 0;

	for (int i = 0; i != DIMU * DIMW; ++i)
	{
		int u = i / DIMW;
		int w = i % DIMW;

		TYPE value = (TYPE)0;

		for (int v = 0; v != DIMV; ++v)
		{
			value += cpuMatrixA[v + u * DIMV] * cpuMatrixB[w + v * DIMW];
		}

		errorCount += (value != cpuMatrixC[i]);
	}

	printf("Matrix A: (%d, %d)\n", DIMU, DIMV);
	printf("Matrix B: (%d, %d)\n", DIMV, DIMW);
	printf("Matrix C: (%d, %d)\n", DIMU, DIMW);
	printf("Errors:   %d\n", errorCount);

	cudaFree(gpuMatrixA);
	cudaFree(gpuMatrixB);
	cudaFree(gpuMatrixC);

	free(cpuMatrixA);
	free(cpuMatrixB);
	free(cpuMatrixC);

	return 0;
}