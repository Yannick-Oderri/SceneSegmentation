//
// Created by ynki9 on 5/25/20.
//

#ifndef PROJECT_EDGE_CUDA_DEPTH_IMG_OPR_H
#define PROJECT_EDGE_CUDA_DEPTH_IMG_OPR_H

#include <opencv2/opencv.hpp>

extern "C" cv::Mat cuCurveDiscOperation(cv::Mat& depth_map);
extern "C" cv::Mat cleanDiscontinuityOpr(cv::Mat& disc_img);
#endif //PROJECT_EDGE_CUDA_DEPTH_IMG_OPR_H
