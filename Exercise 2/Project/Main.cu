#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DIMU 4
#define DIMV 4
#define DIMW 4
#define TYPE double

__global__ void gpuVectorAdd(TYPE* bufferIn1, TYPE* bufferIn2, TYPE* bufferOut, int bufferSize)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < bufferSize) bufferOut[tid] = bufferIn1[tid] + bufferIn2[tid];
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

	//int threads = 64;
	//int blocks  = (SIZE + threads - 1) / threads; // TODO

	//gpuVectorAdd<<<blocks, threads>>>(gpuBufferIn1, gpuBufferIn2, gpuBufferOut, SIZE); // TODO
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
			value += cpuMatrixA[v + u * DIMV] * cpuMatrixB[w + u * DIMW];
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