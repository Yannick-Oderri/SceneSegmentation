//
// Created by ynki9 on 12/30/19.
//

#ifndef PROJECT_EDGE_SIMPLE_IMAGE_PRODUCER_H
#define PROJECT_EDGE_SIMPLE_IMAGE_PRODUCER_H

// #include <format.h>
#include <libfreenect2/libfreenect2.hpp>
#include <opencv2/opencv.hpp>
#include <boost/log/trivial.hpp>

#include "dataflow/pipeline_filter.h"
#include "res/resource_mgr.h"
#include "frame.h"

class SimpleImageProducer: public ProducerPipeFilter<FrameElement* > {
    libfreenect2::Frame* t_image_;
    cv::Mat depth_image_;
    cv::Mat ndepth_image_;
    cv::Mat color_image_;

    const int image_idx_;

    /** IR camera intrinsic calibration parameters.
   * Kinect v2 includes factory preset values for these parameters. They are used in depth image decoding, and Registration.
   */
    struct IrCameraParams
    {
        float fx; ///< Focal length x (pixel)
        float fy; ///< Focal length y (pixel)
        float cx; ///< Principal point x (pixel)
        float cy; ///< Principal point y (pixel)
        float k1; ///< Radial distortion coefficient, 1st-order
        float k2; ///< Radial distortion coefficient, 2nd-order
        float k3; ///< Radial distortion coefficient, 3rd-order
        float p1; ///< Tangential distortion coefficient
        float p2; ///< Tangential distortion coefficient
    };

public:
    SimpleImageProducer(ResMgr* const res_mgr, int img_idx):
    ProducerPipeFilter(new QueueClient<FrameElement* >(), res_mgr),
    image_idx_(img_idx){}

    DepthCameraParams getDepthCameraParams(){
        DepthCameraParams camera_params;

        camera_params.fx = 550;
        camera_params.fy = 550;
        camera_params.cx = 640 / 2;
        camera_params.cy = 480 / 2;

        return camera_params;
    }

    void initialize(){
        if(this->res_mgr_ == nullptr){
            BOOST_LOG_TRIVIAL(error) << "Resource Manager not initialized";
            return;
        }

        if(this->image_idx_ < 0){
            BOOST_LOG_TRIVIAL(error) << "Invalid Image ID";
            return;
        }

        char name_buff[20];
        std::sprintf(name_buff, "ctest%d.png", image_idx_);
        color_image_ = this->res_mgr_->loadColorImage(string(name_buff));

        std::sprintf(name_buff, "test%d.png", image_idx_);
        cv::Mat t_depth_img = this->res_mgr_->loadDepthImage(string(name_buff));


        double min_val, max_val;
        cv::minMaxLoc(t_depth_img, &min_val, &max_val);
        t_depth_img.convertTo(depth_image_, CV_32F, 1.0, 0);
        t_depth_img.convertTo(ndepth_image_, CV_32F, 1.0/max_val, 0);

        t_image_ = new libfreenect2::Frame(depth_image_.cols, depth_image_.rows, sizeof(float), depth_image_.ptr());
    }

    void start(){
        int depth_buffer_size = depth_image_.cols * depth_image_.rows * sizeof(float);
        // int color_buffer_size = color_image_.cols * color_image_.rows * sizeof(CV_8UC3)

        int frame_count = 0;
        const DepthCameraParams camera_params = this->getDepthCameraParams();
        while(this->close_pipe_ == false){
            BOOST_LOG_TRIVIAL(info) << "Receiving Simple frame: " << frame_count;

            // Copy image to new buffer to feed down pipeline
            unsigned char* new_depth_buffer = (unsigned char*)malloc(depth_buffer_size);
            unsigned char* new_ndepth_buffer = (unsigned char*)malloc(depth_buffer_size);
            memcpy(new_depth_buffer, depth_image_.data, depth_buffer_size);
            memcpy(new_ndepth_buffer, ndepth_image_.data, depth_buffer_size);


            DepthFrameElement* depth_content  = new DepthFrameElement(
                    t_image_->width,
                    t_image_->height,
                    sizeof(float),
                    &camera_params,
                    (float*)new_depth_buffer,
                    (float*)new_ndepth_buffer);

            // Generate Color Image mat
            cv::Mat t_col;
            color_image_.copyTo(t_col);

            out_queue_->push(new FrameElement(t_col, *depth_content));
            frame_count++;

            std::this_thread::sleep_for(std::chrono::milliseconds(200000));
        }
    }

    /**
     * End Pipeline loop
     */
    void signalEnd(){
        this->close_pipe_ = true;
    }

};



#endif //PROJECT_EDGE_SIMPLE_IMAGE_PRODUCER_H
