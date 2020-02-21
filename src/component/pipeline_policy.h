//
// Created by ynki9 on 2/5/20.
//

#ifndef PROJECT_EDGE_PIPELINEPOLICY_H
#define PROJECT_EDGE_PIPELINEPOLICY_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "frame.h"

using namespace std;

using Contours = vector<vector<cv::Point>>;
using Contour = vector<cv::Point>;

/**
 * Policies used to perform operations
 */
class PipelinePolicy {
    virtual void executePolicy() = 0;
};


#endif //PROJECT_EDGE_PIPELINEPOLICY_H
