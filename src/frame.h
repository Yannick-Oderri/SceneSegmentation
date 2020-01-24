//
// Created by ynki9 on 12/25/19.
//

#ifndef PROJECT_EDGE_FRAME_H
#define PROJECT_EDGE_FRAME_H

#include <opencv2/opencv.hpp>
#include <libfreenect2/libfreenect2.hpp>

using DepthCameraParams = libfreenect2::Freenect2Device::IrCameraParams;


class DepthFrameElement {
    /**
     * Depth Frame constructure used to store frame data
     * @param width
     * @param height
     * @param channel_size
     * @param data
     * @param camera_params
     */
    DepthFrameElement(int width, int height, int channel_size,
            float const* const data,
            DepthCameraParams const* const camera_params):
            width_(width),
            height_(height),
            bits_per_channel_(channel_size),
            data_(data),
            depth_camera_params_(camera_params){}

    DepthFrameElement(const DepthFrameElement& cpy):
            width_(cpy.width_),
            height_(cpy.height_),
            bits_per_channel_(cpy.bits_per_channel_),
            data_(cpy.data_),
            depth_camera_params_(cpy.depth_camera_params_)
            {}

    float const* const data_;
    int width_;
    int height_;
    int bits_per_channel_;
    DepthCameraParams const* const depth_camera_params_;

    cv::Point3f getXYZPoint(int r, int c, double& x, double& y, double&z) const {
        const float bad_point = std::numeric_limits<float>::quiet_NaN();
        const float cx(depth_camera_params_->cx), cy(depth_camera_params_->cy);
        const float fx(1/depth_camera_params_->fx), fy(1/depth_camera_params_->fy);
        float* undistorted_data = (float *)data_;
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

};



#endif //PROJECT_EDGE_FRAME_H
