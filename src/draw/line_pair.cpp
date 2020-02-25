//
// Created by ynk on 2/24/20.
//

#include "line_pair.h"

LineSegment &LinePair::operator[](int index) {
    if(index == 0)
        return this->line_segments_.first;
    else if(index == 1)
        return this->line_segments_.second;
    else
        throw std::out_of_range("index out of range");
}

cv::Scalar LinePair::getBackgroundColor() {
    return this->background_color_;
}

cv::Scalar LinePair::getForegroundColor() {
    return this->foreground_color_;
}

void LinePair::draw3D() {
    // TODO
    throw "Function not implemented";
}

void LinePair::draw2D(DrawableFrameBuffer& framebuffer) {
    // TODO
    throw "Funciton not implemented";
}
