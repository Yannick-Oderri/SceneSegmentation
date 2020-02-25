//
// Created by ynk on 2/24/20.
//

#ifndef PROJECT_EDGE_DRAWABLE_H
#define PROJECT_EDGE_DRAWABLE_H

#include <opencv2/opencv.hpp>

using DrawableFrameBuffer = cv::Mat;

class Drawable {
public:

    virtual cv::Scalar getForegroundColor() = 0;
    virtual cv::Scalar getBackgroundColor() = 0;
    virtual void draw2D(DrawableFrameBuffer&) = 0;
    virtual void draw3D() = 0;
};


#endif //PROJECT_EDGE_DRAWABLE_H
