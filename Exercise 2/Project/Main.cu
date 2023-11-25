#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DIMU 1024
#define DIMV 1024
#define DIMW 1024
#define TYPE double

__global__ void gpuMatrixMultiply(TYPE* matrixA, TYPE* matrixB, TYPE* matrixC, int dimU, int dimV, int dimW)
{
	uint2 tid;
	tid.x = threadIdx.x + blockIdx.x * blockDim.x;
	tid.y = threadIdx.y + blockIdx.y * blockDim.y;

	if (tid.x >= dimW) return;
	if (tid.y >= dimU) return;

	TYPE element = (TYPE)0;

	for (int v = 0; v != dimV; ++v)
	{
		element += matrixA[v + tid.y * dimV] * matrixB[tid.x + v * dimW];
	}

	matrixC[tid.x + tid.y * dimW] = element;
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

	gpuMatrixMultiply<<<blocks, threads>>>(gpuMatrixA, gpuMatrixB, gpuMatrixC, DIMU, DIMV, DIMW);
	cudaDeviceSynchronize();

	cudaMemcpy(cpuMatrixC, gpuMatrixC, DIMU * DIMW * sizeof(TYPE), cudaMemcpyDeviceToHost);

	int errorCount = 0;

	for (int i = 0; i != DIMU * DIMW; ++i)
	{
		int u = i / DIMW;
		int w = i % DIMW;

		TYPE value = cpuMatrixC[i];

		for (int v = 0; v != DIMV; ++v)
		{
			value -= cpuMatrixA[v + u * DIMV] * cpuMatrixB[w + v * DIMW];
		}

		errorCount += (fabs(value) > 0.001f);
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