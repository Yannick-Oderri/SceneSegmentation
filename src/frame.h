//
// Created by ynki9 on 12/25/19.
//

#ifndef PROJECT_EDGE_FRAME_H
#define PROJECT_EDGE_FRAME_H

#include <opencv2/opencv.hpp>

typedef unsigned int FrameID;

class ImageFrame {
public:
    ImageFrame(const cv::Mat &img, const FrameID id, const unsigned long elapse_time);

private:
    cv::Mat image;
    const FrameID id;
    const unsigned long elapse_time;
};

#endif //PROJECT_EDGE_FRAME_H
