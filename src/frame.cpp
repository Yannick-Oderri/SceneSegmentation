//
// Created by ynki9 on 12/25/19.
//

#include "frame.h"

ImageFrame::ImageFrame(const cv::Mat &img, const FrameID id, const unsigned long elapse_time):
image(img), id(id), elapse_time(elapse_time)
{

}