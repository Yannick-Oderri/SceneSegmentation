 //
// Created by ynki9 on 12/25/19.
//

#ifndef PROJECT_EDGE_FRAME_H
#define PROJECT_EDGE_FRAME_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <stdlib.h>

#ifdef WITH_AZURE_KINECT
#include <k4a/k4a.hpp>
#endif


using namespace std;

/// Define temporary color frame element
using ColorFrameElement = cv::Mat;

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
            const float* data,
            const float* ndata = nullptr,
            const unsigned char* rdata = nullptr):
            width_(width),
            height_(height),
            bits_per_channel_(channel_size),
            data_(data),
            ndata_(ndata),
            rdata_(rdata)
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
            ndata_(cpy.ndata_),
            rdata_(cpy.rdata_)
            {}
    /**
     * Default Constructor
     */
    DepthFrameElement():
            width_(0),
            height_(0),
            bits_per_channel_(0),
            data_(nullptr),
            ndata_(nullptr),
            rdata_(nullptr){}

    ~DepthFrameElement(){
        free((void *) data_);
        free((void *) ndata_);
        free((void *) rdata_);
    }

    float const* const getData(){
        return this->data_;
    }

    cv::Mat getcvMat(){
        cv::Mat mat(this->height_, this->width_, CV_32F, (char* const)this->data_);
        return mat;
    }

    /**
     * Returns cv::mat object of depth data
     * @return
     */
    cv::Mat getDepthImage(){
        cv::Mat mat(this->height_, this->width_, CV_32F, (char* const)this->data_);
        return mat;
    }

    /**
     * Returns cv::mat object of normalized depth data
     * @return
     */
    cv::Mat getNDepthImage(){
        cv::Mat mat(this->height_, this->width_, CV_32F, (char* const)this->ndata_);
        return mat;
    }

    /**
    * Returns cv::mat object of raw depth data
    * @return
    */
    cv::Mat getRawDepthImage(){
        cv::Mat mat(this->height_, this->width_, CV_16U, (char* const)this->rdata_);
        return mat;
    }

    /**
     * REturn Depth frame width
     * @return
     */
    int getWidth(){
        return this->width_;
    }

    /**
     * Return Depth frame height
     * @return
     */
    int getHeight(){
        return this->height_;
    }


    /**
     * Provides the XYZ point cloud element for provide row and column coordinate
     * @param r The vertical coordinate for the frame
     * @param c Horizontal Coordiante of the pixel
     * @param x Output world coordinate
     * @param y Output world coordinate
     * @param z Output world coordinate
     * @return  XYZ World Coordinate
     */
    inline cv::Point3f getXYZPoint(int r, int c, float& x, float& y, float&z) const {
//        const float bad_point = std::numeric_limits<float>::quiet_NaN();
//        const float cx(depth_camera_params_->cx), cy(depth_camera_params_->cy);
//        const float fx(1/depth_camera_params_->fx), fy(1/depth_camera_params_->fy);
//        float* undistorted_data = (float *)data_;
//        const float depth_val = undistorted_data[this->height_*r+c]/(400.0f); //scaling factor, so that value of 1 is one meter.
//        if (isnan(depth_val) || depth_val <= 0.001)
//        {
//            //depth value is not valid
//            x = y = z = bad_point  ;
//        }
//        else
//        {
//            x = (c - cx) * fx * depth_val;
//            y = (r - cy) * fy * depth_val;
//            z = depth_val;
//        }

        return cv::Point3f(x, y, z);
    }

private:
    /// Private Fields
    unsigned char const* const rdata_; // raw depth data
    float const* const data_; // Depth Data element
    float const* const ndata_; // Normal Depth data element
    int width_; // Frame width
    int height_; // frame height
    int bits_per_channel_; // bits per pixel
};


/**
 * Frame Elements
 */
class FrameElement {
    /// Private Fields
    ColorFrameElement color_frame_element_;
    DepthFrameElement depth_frame_element_;
    cv::Mat ddiscontinuity_data_;
    cv::Mat cdiscontinuity_data_;
    cv::Mat contour_data_;
    const int frame_id_;
    const long frame_time_;

public:
    /// Constructor
    FrameElement(int frame_id, ColorFrameElement color_frame_element, DepthFrameElement* depth_frame_element, long frame_time):
            color_frame_element_(color_frame_element),
            depth_frame_element_(*depth_frame_element),
            ddiscontinuity_data_(),
            cdiscontinuity_data_(),
            contour_data_(),
            frame_id_(frame_id),
            frame_time_(frame_time){}

#ifdef WITH_AZURE_KINECT
#endif
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

    /**
     * Set Edge information
     * @param depth_disc
     * @param curve_disc
     * @param contour_data
     */
    inline void setEdgeData(cv::Mat& depth_disc, cv::Mat& curve_disc, cv::Mat& contour_data){
        this->ddiscontinuity_data_ = depth_disc;
        this->cdiscontinuity_data_ = curve_disc;
        this->contour_data_ = contour_data;
    }

    /**
     * Get depth discontinuty frame
     * @return
     */
    inline cv::Mat getDepthDiscontinuity(){
        return this->ddiscontinuity_data_;
    }

    /**
     * get curve discontinuity frame
     * @return
     */
    inline cv::Mat getCurveDiscontinuity(){
        return this->cdiscontinuity_data_;
    }

    /**
     * get Contour Frame
     * @return
     */
    inline cv::Mat getContourFrame(){
        return this->contour_data_;
    }

    /**
     * get Frame Spawn Time
     * @return
     */
    double getFrameTime() const {
        return this->frame_time_;
    }

    /**
     * get Frame ID
     * @return
     */
    int getFrameID() const {
        return this->frame_id_;
    }

private:
     /// Copy Contructor
     FrameElement(const FrameElement& cpy) = delete;

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

     ~ContourAttributes(){

     }

     /// Constructor
//     ContourAttributes(ColorFrameElement color_data, DepthFrameElement depth_data, vector<vector<cv::Point>> contour_data):
//     frame_element(color_data, depth_data),
//     contours(contour_data){}

     /// Copy Constructor
     ContourAttributes(FrameElement* frame_data, vector<vector<cv::Point>> contour_data):
     frame_element(frame_data),
     contours(contour_data){}

     FrameElement* frame_element;
     vector<vector<cv::Point>> contours;
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
