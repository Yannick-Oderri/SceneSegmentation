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
using LineSegments = vector<vector<cv::Line>>;

/**
 * Policies used to perform operations
 */
class PipelinePolicy {
    virtual void executePolicy() = 0;
};

/**
 * Abstract policy for processing contour data
 */
class ContourPolicy: public PipelinePolicy{
public:
    /**
     * Constructor
     */
    ContourPolicy(){}
    virtual void setContourData(ContourAttributes * const contour_data) = 0;
    virtual void executePolicy() = 0;
};

/**
 *
 */
class LineSegmentContourPolicy: public ContourPolicy{
private:
    ContourAttributes * current_contour_data_;

public:
    LineSegmentContourPolicy():
            ContourPolicy(){

    }

    void setContourData(ContourAttributes * const contour_data);
    void executePolicy();
};


#endif //PROJECT_EDGE_PIPELINEPOLICY_H
