//
// Created by ynki9 on 4/1/20.
//

#ifndef PROJECT_EDGE_CU_CONTOUR_OPR_H
#define PROJECT_EDGE_CU_CONTOUR_OPR_H

#include <draw/line_sgmnt.h>
#include <vector>


struct ContourResult{
    float p_region_mean;
    float p_region_count;
    float n_region_mean;
    float n_region_count;
    float edge_mean;
    float contour_len;

    int tval[12];
};

extern "C"
ContourResult* cu_determineROIMean(std::vector<std::vector<LineSegment>>& contour_segments, cv::Mat& depth_map, int window_size);

#endif //PROJECT_EDGE_CU_CONTOUR_OPR_H
