
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib> 
#include <ctime>
#include <chrono>

#define R 5
#define N 10000
#define BS 8
#define K 1

cudaError_t sumMatrix(float *mat, float *out, dim3 threads, dim3 blocksMatrix, int size);

void sequencial(float *mat, float *out) {
    for (int i = R; i < N - R; i++) {
        for (int j = R; j < N - R; j++) {
            float sum = 0;
            for (int x = i - R; x <= i + R; x++) {
                for (int y = j - R; y <= j + R; y++) {
                    sum += mat[x * N + y];
                }
            }
            out[(i - R) * (N - 2 * R) + j - R] = sum;
        }
    }
}

void addRandomData(float* mat) {
    srand(time(NULL));
    for (int i = 0;i < N;i++) {
        for (int j = 0;j < N;j++) {
            mat[i * N + j] = (float)(rand()) / (float)(rand());
        }
    }
}

void display(float* mat, int size) {
    for (int i = 0;i < size;i++) {
        for (int j = 0;j < size;j++) {
            printf("%f ", mat[i * size + j]);
        }
        printf("\n");
    }
}

__global__ void calculateMatrixGlobal(float* mat, float* out, int outSize)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = (threadIdx.y + blockIdx.y * blockDim.y) * K;

    for (int k = 0; k < K; k++)
    {
        float sum = 0;
        if (i < outSize && j + k < outSize) {
            for (int ry = -R; ry <= R; ry++) {
                for (int rx = -R; rx <= R; rx++) {
                    sum += mat[(j + k + R + ry) * N + (i + R + rx)];
                }
            }
            out[(j + k) * outSize + i] = sum;
        }
    }
}

int main()
{  
    int outputSize = (N - R * 2);
    float* basicMatrix = (float*)malloc(N * N * sizeof(float));
    float* outputMatrix = (float*)malloc(outputSize * outputSize * sizeof(float));


    addRandomData(basicMatrix);
    //display(basicMatix, N);

    auto startSeq = std::chrono::high_resolution_clock::now();
    sequencial(basicMatrix, outputMatrix);
    auto stopSeq = std::chrono::high_resolution_clock::now();
    auto timeDifSeq = std::chrono::duration_cast<std::chrono::milliseconds>(stopSeq - startSeq);
    printf("Time in seconds: %f\n", timeDifSeq.count() / 1000.0f);
    
    //display(outputMatrix, outputSize);


    dim3 threads(BS, BS);
    dim3 blocksMatrix(ceil(outputSize / (float)BS), ceil(outputSize / (float)BS / K));

    float* outputMatrix2 = (float*)malloc(outputSize * outputSize * sizeof(float));
    // Add vectors in parallel.
    cudaError_t cudaStatus = sumMatrix(basicMatrix, outputMatrix2, threads, blocksMatrix, outputSize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "sumMatrix failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t sumMatrix(float *mat, float *out, dim3 threads, dim3 blocks, int outSize)
{
    cudaError_t cudaStatus;
    cudaEvent_t start, stop;

    float* dev_mat;
    float* dev_out;
    float time = 0;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_mat, N * N * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_out, outSize * outSize * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_mat, mat, N * N * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    
    cudaEventRecord(start, nullptr);
    // Launch a kernel on the GPU with one thread for each element.
    calculateMatrixGlobal <<<blocks, threads>>> (dev_mat, dev_out, outSize);

    cudaEventRecord(stop, nullptr);
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(out, dev_out, outSize * outSize * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaEventElapsedTime(&time, start, stop);
    printf("Time GPU %f\n", time / 1000);

Error:
    cudaFree(dev_mat);
    cudaFree(dev_out);
    
    return cudaStatus;
}
