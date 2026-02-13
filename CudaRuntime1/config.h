#ifndef CONFIG_H
#define CONFIG_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define TILE_WIDTH 16
#define MAX_MASK_SIZE 121

// Function Prototype
void run_cpu_serial(float* h_in, float* h_out, float* h_mask, int w, int h, int n);
float run_gpu_steps(int step, float* d_in, float* d_out, float* h_mask, int w, int h, int n);

#endif