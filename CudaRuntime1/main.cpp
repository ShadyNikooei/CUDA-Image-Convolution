/*#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "config.h"

using namespace cv; // Namespace MUST be here
int main() {
    // 1. Load Image
    cv::Mat input = cv::imread("input.png", cv::IMREAD_GRAYSCALE);
    if (input.empty()) {
        std::cout << "Error: 'input.png' not found!" << std::endl;
        return -1;
    }

    int W = input.cols;
    int H = input.rows;
    int n = 4; // 9x9 mask radius

    // 2. LoG Mask (81 values)
    float h_LoG[81] = {
        0,0,3,2,2,2,3,0,0, 0,2,3,5,5,5,3,2,0, 3,3,5,3,0,3,5,3,3,
        2,5,3,-12,-23,-12,3,5,2, 2,5,0,-23,-40,-23,0,5,2, 2,5,3,-12,-23,-12,3,5,2,
        3,3,5,3,0,3,5,3,3, 0,2,3,5,5,5,3,2,0, 0,0,3,2,2,2,3,0,0
    };

    // 3. Prepare Memory
    cv::Mat input_f;
    input.convertTo(input_f, CV_32F);
    float* d_in, * d_out;
    cudaMalloc(&d_in, W * H * sizeof(float));
    cudaMalloc(&d_out, W * H * sizeof(float));
    cudaMemcpy(d_in, input_f.ptr<float>(), W * H * sizeof(float), cudaMemcpyHostToDevice);

    // 4. Run GPU Kernel (Step 4)
    run_gpu_shared(d_in, d_out, h_LoG, W, H, n);
    cudaDeviceSynchronize();

    // 5. Results
    cv::Mat res_f(H, W, CV_32F);
    cudaMemcpy(res_f.ptr<float>(), d_out, W * H * sizeof(float), cudaMemcpyDeviceToHost);

    cv::Mat final_res;
    res_f.convertTo(final_res, CV_8U);
    cv::imwrite("output_result.png", final_res);

    std::cout << "Done! Result saved in 'output_result.png'" << std::endl;

    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <map>
#include "config.h"

void displayMenu() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "   GPU Parallel Convolution Project     " << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Select Filter Types:" << std::endl;
    std::cout << "1. Averaging (3x3)" << std::endl;
    std::cout << "2. Gaussian (5x5)" << std::endl;
    std::cout << "3. Sobel Edge Detection (3x3)" << std::endl;
    std::cout << "4. LoG - Laplacian of Gaussian (9x9)" << std::endl;
    std::cout << "0. Exit" << std::endl;
    std::cout << "Choice: ";
}

int main() {
    // Dictionary to store CPU times for each filter to calculate Speedup later
    std::map<int, float> cpu_times;

    // Load input image in grayscale mode
    cv::Mat img = cv::imread("input.png", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Error: Image 'input.png' not found in working directory!" << std::endl;
        return -1;
    }

    int W = img.cols, H = img.rows;
    cv::Mat input_f, res_f(H, W, CV_32F);
    img.convertTo(input_f, CV_32F);

    while (true) {
        displayMenu();
        int f_choice; std::cin >> f_choice;
        if (f_choice == 0) break;

        float* h_mask; int n; std::string f_name;
        // Masks defined according to the assignment PDF
        float m_avg[] = { 1 / 9.f,1 / 9.f,1 / 9.f, 1 / 9.f,1 / 9.f,1 / 9.f, 1 / 9.f,1 / 9.f,1 / 9.f };
        float m_gauss[] = { 2 / 159.f,4 / 159.f,5 / 159.f,4 / 159.f,2 / 159.f, 4 / 159.f,9 / 159.f,12 / 159.f,9 / 159.f,4 / 159.f, 5 / 159.f,12 / 159.f,15 / 159.f,12 / 159.f,5 / 159.f, 4 / 159.f,9 / 159.f,12 / 159.f,9 / 159.f,4 / 159.f, 2 / 159.f,4 / 159.f,5 / 159.f,4 / 159.f,2 / 159.f };
        float m_sobelX[] = { -1,0,1, -2,0,2, -1,0,1 };
        float m_log[] = { 0,0,3,2,2,2,3,0,0, 0,2,3,5,5,5,3,2,0, 3,3,5,3,0,3,5,3,3, 2,5,3,-12,-23,-12,3,5,2, 2,5,0,-23,-40,-23,0,5,2, 2,5,3,-12,-23,-12,3,5,2, 3,3,5,3,0,3,5,3,3, 0,2,3,5,5,5,3,2,0, 0,0,3,2,2,2,3,0,0 };

        if (f_choice == 1) { h_mask = m_avg;    n = 1; f_name = "Average"; }
        else if (f_choice == 2) { h_mask = m_gauss;  n = 2; f_name = "Gaussian"; }
        else if (f_choice == 3) { h_mask = m_sobelX; n = 1; f_name = "SobelX"; }
        else { h_mask = m_log;    n = 4; f_name = "LoG"; }

        std::cout << "Enter Optimization Step (1-5): ";
        int step; std::cin >> step;

        float current_execution_time = 0.0f;

        if (step == 1) {
            // STEP 1: HOST SERIAL IMPLEMENTATION
            auto start = std::chrono::high_resolution_clock::now();
            run_cpu_serial((float*)input_f.data, (float*)res_f.data, h_mask, W, H, n);
            auto end = std::chrono::high_resolution_clock::now();

            current_execution_time = std::chrono::duration<float, std::milli>(end - start).count();
            cpu_times[f_choice] = current_execution_time; // Store as reference for this filter
            std::cout << ">>> Step 1 (CPU) Execution Time: " << current_execution_time << " ms" << std::endl;
        }
        else {
            // STEPS 2-5: CUDA PARALLEL IMPLEMENTATIONS
            float* d_in, * d_out;
            cudaMalloc(&d_in, W * H * sizeof(float));
            cudaMalloc(&d_out, W * H * sizeof(float));
            cudaMemcpy(d_in, input_f.data, W * H * sizeof(float), cudaMemcpyHostToDevice);

            // Time measurement inside run_gpu_steps using CUDA Events
            current_execution_time = run_gpu_steps(step, d_in, d_out, h_mask, W, H, n);
            std::cout << ">>> Step " << step << " (GPU) Execution Time: " << current_execution_time << " ms" << std::endl;

            // Documenting Speedup
            if (cpu_times.count(f_choice) > 0) {
                float speedup = cpu_times[f_choice] / current_execution_time;
                std::cout << "------------------------------------------------" << std::endl;
                std::cout << "[ANALYSIS] Speedup (vs CPU): " << speedup << "x" << std::endl;
                std::cout << "------------------------------------------------" << std::endl;
            }
            else {
                std::cout << "[HINT] Run Step 1 for this filter first to see Speedup calculation!" << std::endl;
            }

            cudaMemcpy(res_f.data, d_out, W * H * sizeof(float), cudaMemcpyDeviceToHost);
            cudaFree(d_in); cudaFree(d_out);
        }

        // Save result
        cv::Mat final_img;
        res_f.convertTo(final_img, CV_8U);
        std::string fileName = f_name + "_Step" + std::to_string(step) + ".png";
        cv::imwrite(fileName, final_img);
        std::cout << "SUCCESS: Saved to " << fileName << std::endl;
    }

    return 0;
}