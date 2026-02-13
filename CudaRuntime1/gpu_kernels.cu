/*#include "config.h"
#include <device_launch_parameters.h> // For __syncthreads

// The ONLY place where c_Mask is defined
__constant__ float c_Mask[MAX_MASK_SIZE];

__global__ void convolution_kernel(float* input, float* output, int w, int h, int n) {
    extern __shared__ float s_tile[];
    int maskSize = 2 * n + 1;
    int sharedDim = blockDim.x + 2 * n;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;

    for (int i = ty; i < sharedDim; i += blockDim.y) {
        for (int j = tx; j < sharedDim; j += blockDim.x) {
            int r_idx = blockIdx.y * blockDim.y - n + i;
            int c_idx = blockIdx.x * blockDim.x - n + j;
            if (r_idx >= 0 && r_idx < h && c_idx >= 0 && c_idx < w)
                s_tile[i * sharedDim + j] = input[r_idx * w + c_idx];
            else
                s_tile[i * sharedDim + j] = 0.0f;
        }
    }
    __syncthreads(); // Now it will be recognized by NVCC

    if (row < h && col < w) {
        float sum = 0.0f;
        for (int r = 0; r < maskSize; r++) {
            for (int c = 0; c < maskSize; c++) {
                sum += s_tile[(ty + r) * sharedDim + (tx + c)] * c_Mask[r * maskSize + c];
            }
        }
        output[row * w + col] = sum;
    }
}

void run_gpu_shared(float* d_in, float* d_out, float* h_mask, int w, int h, int n) {
    size_t maskBytes = (2 * n + 1) * (2 * n + 1) * sizeof(float);
    cudaMemcpyToSymbol(c_Mask, h_mask, maskBytes);

    int sharedDim = TILE_WIDTH + 2 * n;
    size_t sharedMemSize = sharedDim * sharedDim * sizeof(float);
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((w + TILE_WIDTH - 1) / TILE_WIDTH, (h + TILE_WIDTH - 1) / TILE_WIDTH);
    convolution_kernel << <grid, block, sharedMemSize >> > (d_in, d_out, w, h, n);
}
*/
#include "config.h"
#include <device_launch_parameters.h> // For __syncthreads
#include <stdio.h>

// Definition of Constant Memory for the mask
__constant__ float c_Mask[MAX_MASK_SIZE];

/**
 * Step 2, 3, & 5: Basic Convolution Kernel
 * Uses Global Memory for image and Constant Memory for mask.
 * Includes 'restrict' keyword for pointer aliasing optimization.
 */
__global__ void kernel_basic(const float* __restrict__ input, float* __restrict__ output, int w, int h, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < h && col < w) {
        float sum = 0.0f;
        int mSize = 2 * n + 1;
        // Iterate through the mask dimensions
        for (int i = -n; i <= n; i++) {
            for (int j = -n; j <= n; j++) {
                int r = row + i;
                int c = col + j;
                // Boundary check for input pixels
                if (r >= 0 && r < h && c >= 0 && c < w)
                    sum += input[r * w + c] * c_Mask[(i + n) * mSize + (j + n)];
            }
        }
        output[row * w + col] = sum;
    }
}

/**
 * Step 4: Tiled Convolution using Shared Memory
 * Loads image tiles and halo cells into shared memory to reduce global memory traffic.
 */
__global__ void kernel_shared(const float* __restrict__ input, float* __restrict__ output, int w, int h, int n) {
    extern __shared__ float s_tile[];
    int mSize = 2 * n + 1;
    int sDim = blockDim.x + 2 * n; // Dimension of the tile including halo cells
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;

    // Collaborative loading of tile data into Shared Memory
    for (int i = ty; i < sDim; i += blockDim.y) {
        for (int j = tx; j < sDim; j += blockDim.x) {
            int r = blockIdx.y * blockDim.y - n + i;
            int c = blockIdx.x * blockDim.x - n + j;
            // Load pixel or zero-pad if outside boundaries
            s_tile[i * sDim + j] = (r >= 0 && r < h && c >= 0 && c < w) ? input[r * w + c] : 0.0f;
        }
    }
    __syncthreads(); // Synchronize threads before starting computation

    if (row < h && col < w) {
        float sum = 0.0f;
        for (int r = 0; r < mSize; r++) {
            for (int c = 0; c < mSize; c++)
                sum += s_tile[(ty + r) * sDim + (tx + c)] * c_Mask[r * mSize + c];
        }
        output[row * w + col] = sum;
    }
}

float run_gpu_steps(int step, float* d_in, float* d_out, float* h_mask, int w, int h, int n) {
    int mW = 2 * n + 1;
    // Copy mask from Host to Constant Memory on Device [cite: 70]
    cudaMemcpyToSymbol(c_Mask, h_mask, mW * mW * sizeof(float));

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((w + TILE_WIDTH - 1) / TILE_WIDTH, (h + TILE_WIDTH - 1) / TILE_WIDTH);

    // Timing using CUDA Events [cite: 67, 79]
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    if (step == 4) {
        // Dynamic shared memory allocation size [cite: 78]
        int sMem = (TILE_WIDTH + 2 * n) * (TILE_WIDTH + 2 * n) * sizeof(float);
        kernel_shared <<<grid, block, sMem >>> (d_in, d_out, w, h, n);
    }
    else {
        kernel_basic <<<grid, block >>> (d_in, d_out, w, h, n);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}