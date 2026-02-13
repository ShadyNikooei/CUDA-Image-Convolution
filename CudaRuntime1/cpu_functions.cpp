#include "config.h"

void run_cpu_serial(float* h_in, float* h_out, float* h_mask, int w, int h, int n) {
    int maskSize = 2 * n + 1;
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            float sum = 0.0f;
            for (int i = -n; i <= n; i++) {
                for (int j = -n; j <= n; j++) {
                    int curr_r = r + i;
                    int curr_c = c + j;
                    if (curr_r >= 0 && curr_r < h && curr_c >= 0 && curr_c < w) {
                        sum += h_in[curr_r * w + curr_c] * h_mask[(i + n) * maskSize + (j + n)];
                    }
                }
            }
            h_out[r * w + c] = sum;
        }
    }
}