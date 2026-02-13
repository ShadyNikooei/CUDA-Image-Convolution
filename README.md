# High-Performance Image Convolution with CUDA

## Project Overview

This repository contains a high-performance C++/CUDA implementation of
2D image convolution, designed to demonstrate the progressive
optimization of GPU memory hierarchies. The project explores the
transition from sequential CPU processing to advanced GPU techniques,
including Constant Memory, Shared Memory Tiling, and Hardware Cache
utilization.

The implementation supports various spatial filters, ranging from simple
3x3 kernels to complex 9x9 Laplacian of Gaussian (LoG) operators,
emphasizing the Speedup achieved through parallel architectural
optimizations.

------------------------------------------------------------------------

## Supported Filters

The system implements the following convolution masks as defined in
standard image processing pipelines:

### Averaging Filter (3x3)

Low-pass filtering for simple noise reduction.

### Gaussian Blur (5x5)

Non-uniform low-pass filter for high-quality smoothing.

### Sobel Operator (3x3)

Gradient-based edge detection (Horizontal/Vertical).

### Laplacian of Gaussian (9x9)

Advanced edge detection combining Gaussian smoothing with the Laplacian
operator to mitigate noise sensitivity.

------------------------------------------------------------------------

## Optimization Roadmap (The Five Stages)

The project is structured into five distinct evolutionary steps,
following the standard CUDA optimization workflow.

### Step 1: Sequential Host Implementation

A baseline CPU implementation using nested loops. This stage serves as
the reference point for calculating the Speedup ratio and verifying the
correctness of the convolution output.

### Step 2: Basic Parallel GPU Kernel

The initial migration to CUDA. This version utilizes Global Memory for
both the input image and the convolution mask, demonstrating the raw
throughput of parallel threads before memory optimization.

### Step 3: Constant Memory Cache

Optimization of the convolution mask. By utilizing `__constant__` memory
and `cudaMemcpyToSymbol`, the mask is stored in the Constant Cache. This
reduces DRAM traffic and takes advantage of the broadcast mechanism
available when all threads in a warp access the same memory address.

### Step 4: Shared Memory Tiling and Halo Cells

The most significant optimization phase. This step implements Tiling to
load image blocks into `__shared__` memory. It includes logic for "Halo
Cells" (boundary pixels) to ensure correct convolution at the edges of
each tile. This minimizes Global Memory transactions by a factor
proportional to the mask size.

### Step 5: Hardware L2 Cache Utilization

A simplified parallel version designed to test the efficiency of the
GPU's automatic L2 Cache. This stage compares manual memory management
(Step 4) against hardware-managed caching systems in modern NVIDIA
architectures.

------------------------------------------------------------------------

## Technical Specifications

-   Language: C++17 / CUDA C
-   API: CUDA Toolkit 12.x
-   External Library: OpenCV 4.x (used for Image I/O and Pre-processing)
-   Memory Management: Static and Dynamic Shared Memory Allocation
-   Timing: High-precision measurement using `cudaEvent_t` for GPU and
    `std::chrono` for CPU
-   Data Precision: 32-bit floating point (FP32)

------------------------------------------------------------------------

## Performance Analysis

The primary metric for this project is Speedup (S), defined as:

    Speedup = Execution Time (Step 1) / Execution Time (Step X)

Detailed performance logs are generated during execution, allowing for a
granular comparison of how each memory optimization (Constant vs. Shared
vs. Cache) affects the total processing latency.

------------------------------------------------------------------------

## Installation and Requirements

### Prerequisites

-   NVIDIA GPU: Architecture Maxwell (SM 5.2) or higher recommended
-   CUDA Toolkit: Version 11.0 or newer
-   OpenCV: Version 4.0 or newer (ensure the OPENCV_DIR environment
    variable is set)
-   IDE: Visual Studio 2022 with Desktop development with C++ and NVIDIA
    Nsight Integration

------------------------------------------------------------------------

## Building the Project

1.  Clone the repository to your local machine.
2.  Open the .sln file in Visual Studio.
3.  Configure the Additional Include Directories to point to your OpenCV
    build path.
4.  Configure the Linker Input to include: opencv_world\[version\].lib
5.  Set the Build Configuration to Release / x64.
6.  Build the solution.

------------------------------------------------------------------------

## Usage

Upon execution, the program provides an interactive command-line
interface:

-   Select Filter: Choose between Average, Gaussian, Sobel, or LoG.
-   Select Step: Choose the optimization level (1-5).

Output:

-   The program displays the execution time and the calculated Speedup.
-   The processed image is saved with a prefix indicating the filter and
    step used.

------------------------------------------------------------------------

## Implementation Details

### Thread Mapping

Blocks are configured as 16x16 or 32x32 to maximize occupancy.

### Boundary Handling

Zero-padding is applied dynamically within the kernels to manage edge
cases without requiring image padding in host memory.

### Data Precision

All calculations are performed in 32-bit floating point (FP32) to
prevent overflow during large kernel accumulations.

------------------------------------------------------------------------

## Developer

Shady Nikooei
