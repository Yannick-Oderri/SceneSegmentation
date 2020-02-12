 //
// Created by ynki9 on 12/25/19.
//

#ifndef PROJECT_EDGE_FRAME_H
#define PROJECT_EDGE_FRAME_H

#include <opencv2/opencv.hpp>
#include <libfreenect2/libfreenect2.hpp>
#include <vector>

using namespace std;

/// Define temporary color frame element
using ColorFrameElement = cv::Mat;

/**
 * Camera Paramters for Depth camera
 */
using DepthCameraParams = libfreenect2::Freenect2Device::IrCameraParams;

/**
 * Holding class for depth frame data
 */
class DepthFrameElement {

public:
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
            const DepthCameraParams * camera_params):
            width_(width),
            height_(height),
            bits_per_channel_(channel_size),
            data_(data),
            depth_camera_params_(camera_params)
            {}

    /**
     * Default Copy Constructor
     * @param cpy
     */
    DepthFrameElement(const DepthFrameElement& cpy):
            width_(cpy.width_),
            height_(cpy.height_),
            bits_per_channel_(cpy.bits_per_channel_),
            data_(cpy.data_),
            depth_camera_params_(cpy.depth_camera_params_)
            {}
    /**
     * Default Constructor
     */
    DepthFrameElement():
            width_(0),
            height_(0),
            bits_per_channel_(0),
            data_(nullptr){}

    float const* const getData(){
        return this->data_;
    }

    cv::Mat getcvMat(){
        cv::Mat mat(this->height_, this->width_, CV_32F, (char* const)this->data_);
        return mat;
    }

private:
    /// Private Fields
    float const* const data_; // Frame Data element
    int width_; // Frame width
    int height_; // frame height
    int bits_per_channel_; // bits per pixel
    DepthCameraParams const* depth_camera_params_; // camera parameters

    /**
     * Provides the XYZ point cloud element for provide row and column coordinate
     * @param r The vertical coordinate for the frame
     * @param c Horizontal Coordiante of the pixel
     * @param x Output world coordinate
     * @param y Output world coordinate
     * @param z Output world coordinate
     * @return  XYZ World Coordinate
     */
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

        return cv::Point3f(x, y, z);
    }

};


/**
 * Frame Elements
 */
class FrameElement {
    /// Private Fields
    ColorFrameElement color_frame_element_;
    DepthFrameElement depth_frame_element_;

public:
    /// Constructor
    FrameElement(ColorFrameElement color_frame_element, DepthFrameElement depth_frame_element):
            color_frame_element_(color_frame_element),
            depth_frame_element_(depth_frame_element){}

    /// Copy Contructor
    FrameElement(const FrameElement& cpy):
    color_frame_element_(cpy.color_frame_element_),
    depth_frame_element_(cpy.depth_frame_element_){}


    /**
     * Depth Frame Element
     * @return
     */
    inline DepthFrameElement *const getDepthFrameData() {
        return static_cast<DepthFrameElement *>(&this->depth_frame_element_);
    }

    /**
     * Return color Frame Element
     * @return
     */
    inline ColorFrameElement  const* getColorFrameElement(){
        return &this->color_frame_element_;
    }
};


/**
 * Stores output of ContourExtractor filter
 */
 class ContourAttributes {
 public:
     ContourAttributes(ContourAttributes& rhs):
     frame_element(rhs.frame_element),
     contours(rhs.contours){

     }

     /// Constructor
     ContourAttributes(ColorFrameElement color_data, DepthFrameElement depth_data, vector<vector<cv::Point2d>> contour_data):
     frame_element(color_data, depth_data),
     contours(contour_data){}

     /// Copy Constructor
     ContourAttributes(FrameElement frame_data, vector<vector<cv::Point2d>> contour_data):
     frame_element(frame_data),
     contours(contour_data){}

     FrameElement frame_element;
     vector<vector<cv::Point2d>> contours;
 };



 /**
  * REndearable element to showcase to output renderer
  *
  */
 class RenderableElement {
 public:
     RenderableElement(ContourAttributes contour_data):
     contour_attributes_(contour_data){

     }

     ContourAttributes contour_attributes_;
 };



#endif //PROJECT_EDGE_FRAME_H
