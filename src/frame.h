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
private:
    float const* const data_;
    int width_;
    int height_;
    int bits_per_channel_;
//    DepthCameraParams const* const depth_camera_params_;
    cv::Mat camera_projection_;

};


class ContourElement {
public:
    class ContourElementFactory;
private:
    const std::vector<cv::Point2d> points_;
    ContourElement(std::vector<cv::Point2d> contour_points):
            points_(contour_points){}

};

class ContourElement::ContourElementFactory {
    enum{};
    std::vector<ContourElement> GenerateContourElements(std::vector<std::vector<cv::Point2d>> contours);
    ContourElement GenerateContourElement(std::vector<cv::Point2d> contour_points);
};


#endif //PROJECT_EDGE_FRAME_H
