
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <time.h>
#include <stdio.h>
#include<curand.h>
#include <curand_kernel.h>

#define BLOCK_SIZE  16         
#define N           1024     
#define MAX         1000
__global__ void piCalk(double*a, double*b, int n, int* circle_points)
{
    int   bx = blockIdx.x;    
    int   by = blockIdx.y;
    int   tx = threadIdx.x;     
    int   ty = threadIdx.y;   
    int   ia = BLOCK_SIZE * by + ty;  
    int   ib = BLOCK_SIZE * bx + tx;
    double V = 0.0;
    for (int k = 0 ; k < n; k++)
    {
        V = pow(a[ia + k], 2) + pow(b[ib + k], 2);
        if (V < 1)
            *circle_points++;
    }
}

__global__ void random(unsigned int seed, int* result) {

    curandState_t state;
    curand_init(seed, 0, 0, &state);
    *result = curand(&state) % MAX;
}

int randGPU()
{
    int* gpu_x;
    cudaMalloc((void**)&gpu_x, sizeof(int));

    random <<<1, 1 >>> (time(NULL), gpu_x);
    int x;
    cudaMemcpy(&x, gpu_x, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(gpu_x);
    return x;
}
int main()
{
    int numBytes = N * sizeof(float);

    double* X = new double[N];
    double* Y = new double[N];
    int* circle_points = new int(0);
    double* xdev = NULL;
    double* ydev = NULL;
    for (int i = 0; i < N; i++)
    {   
        double xi = double(randGPU()) / MAX;
        X[i] = xi;
        double yi = double(randGPU()) / MAX;
        Y[i] = yi;
    }
    cudaMalloc((void**)&xdev, numBytes);
    cudaMalloc((void**)&ydev, numBytes);

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(N / threads.x, N / threads.y);
    cudaEvent_t start, stop;
    float gpuTime = 0.0f;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    cudaMemcpy(xdev, X, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(ydev, Y, numBytes, cudaMemcpyHostToDevice);
    for (int i = 0; i < N; i++)
    {
        printf("%d., %d.\n", X[i], Y[i]);
    }
    piCalk <<<blocks, threads >>> (xdev, ydev, N, circle_points);
    double pi = double(4 * *circle_points) / N;

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);
    printf("time spent executing by the GPU: %.2f millseconds\n %d", gpuTime, pi);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(xdev);
    cudaFree(ydev);

    delete X;
    delete Y;
    return 0;
}
