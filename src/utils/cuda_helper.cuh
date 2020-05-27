//
// Created by ynki9 on 5/27/20.
//

#ifndef PROJECT_EDGE_CUDA_HELPER_H
#define PROJECT_EDGE_CUDA_HELPER_H

#include <opencv2/opencv.hpp>
// #include "utils/cuda_defs.cu"
#include <helper_functions.h>
#include <helper_cuda.h>       // CUDA device initialization helper functions
#include <math.h>

__host__
void setupDepthMap(int width, int height, cv::Mat depth_map, cudaTextureObject_t& dev_depth_tex);

#endif //PROJECT_EDGE_CUDA_HELPER_H
