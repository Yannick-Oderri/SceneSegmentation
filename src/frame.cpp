//
// Created by ynki9 on 12/25/19.
//

#include "frame.h"


cv::Mat toProjectionMatrix(DepthCameraParams const *const camera_params){
    cv::Mat proj_mat(4, 4, cv::DataType<float>::type);

    const float bad_point = std::numeric_limits<float>::quiet_NaN();
    const float cx(camera_params->cx), cy(camera_params->cy);
    const float fx(1/camera_params->fx), fy(1/camera_params->fy);


    const float depth_val = undistorted_data[512*r+c]/1000.0f; //scaling factor, so that value of 1 is one meter.
    if (isnan(depth_val) || depth_val <= 0.001)
    {
        //depth value is not valid
        x = y = z = bad_point;
    }
    else
    {
        x = (c + 0.5 - cx) * fx * depth_val;
        y = (r + 0.5 - cy) * fy * depth_val;
        z = depth_val;
    }

}

DepthFrameElement::DepthFrameElement(int width, int height, int channel_size, float const *const data,
                                     DepthCameraParams const *const camera_params):
                                     width_(width),
                                     height_(height),
                                     bits_per_channel_(channel_size),
                                     data_(data){
    camera_projection_ = toProjectionMatrix(camera_params);
}