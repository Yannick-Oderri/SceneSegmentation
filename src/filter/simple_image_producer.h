//
// Created by ynki9 on 12/30/19.
//

#ifndef PROJECT_EDGE_SIMPLE_IMAGE_PRODUCER_H
#define PROJECT_EDGE_SIMPLE_IMAGE_PRODUCER_H

// #include <format.h>
#include <opencv2/opencv.hpp>
#include <boost/log/trivial.hpp>
#include <thread>
#include "dataflow/pipeline_filter.h"
#include "dataflow/observer.h"
#include "res/resource_mgr.h"
#include "frame.h"

struct SimpleImageProducerConfig{
    bool isDirty_;
    int imageID_;
    int updateRate_;
};

class SimpleImageProducer: public ProducerPipeFilter<FrameElement* >, public Observer<SimpleImageProducerConfig> {
    libfreenect2::Frame* t_image_;
    cv::Mat depth_image_;
    cv::Mat ndepth_image_;
    cv::Mat color_image_;
    SimpleImageProducerConfig config_;
    SimpleImageProducerConfig new_config_;
    bool dirty_config_;
    ResMgr* const res_mgr_;

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
    SimpleImageProducer(ResMgr* const res_mgr, int img_idx, int frame_delay=33):
        ProducerPipeFilter(new QueueClient<FrameElement* >()),
        res_mgr_(res_mgr),
        config_({false, img_idx, frame_delay}),
        dirty_config_(true),
        new_config_(config_){}

    SimpleImageProducer(ResMgr* const res_mgr, SimpleImageProducerConfig &config):
            ProducerPipeFilter(new QueueClient<FrameElement* >()),
            res_mgr_(res_mgr),
            dirty_config_(true){
        config_ = config;
        new_config_ = config;
    }

    void initialize(){
        if(this->res_mgr_ == nullptr){
            BOOST_LOG_TRIVIAL(error) << "Resource Manager not initialized";
            return;
        }

        if(this->config_.imageID_ < 0){
            BOOST_LOG_TRIVIAL(error) << "Invalid Image ID";
            return;
        }

        // Load Color Image
        char name_buff[20];
        std::sprintf(name_buff, "ctest%d.png", config_.imageID_);
        color_image_ = this->res_mgr_->loadColorImage(string(name_buff), cv::IMREAD_COLOR);
        // Load Depth Image
        std::sprintf(name_buff, "test%d.png", config_.imageID_);
        cv::Mat t_depth_img = this->res_mgr_->loadDepthImage(string(name_buff), cv::IMREAD_UNCHANGED);

        // Extract ROI of depth image
        cv::Mat t_mask = (t_depth_img < 300) | (t_depth_img > 1700);
        t_depth_img.setTo(0, t_mask);

        // Cast to float and perform normalization
        double min_val, max_val;
        cv::minMaxLoc(t_depth_img, &min_val, &max_val);
        t_depth_img.convertTo(depth_image_, CV_32F, 1.0, 0);
        t_depth_img.convertTo(ndepth_image_, CV_32F, 1.0/max_val, 0);

        t_image_ = new libfreenect2::Frame(depth_image_.cols, depth_image_.rows, sizeof(float), depth_image_.ptr());
    }



    FrameElement* generateCurrentFrame(int frame_idx = 0) {
        int depth_buffer_size = depth_image_.cols * depth_image_.rows * sizeof(float);
        // int color_buffer_size = color_image_.cols * color_image_.rows * sizeof(CV_8UC3)

        // Copy image to new buffer to feed down pipeline
        unsigned char *new_depth_buffer = (unsigned char *) malloc(depth_buffer_size);
        unsigned char *new_ndepth_buffer = (unsigned char *) malloc(depth_buffer_size);
//        memcpy(new_depth_buffer, depth_image_.data, depth_buffer_size);
//        memcpy(new_ndepth_buffer, ndepth_image_.data, depth_buffer_size);

        new_depth_buffer = depth_image_.data;
        new_ndepth_buffer = ndepth_image_.data;

        DepthFrameElement *depth_content = new DepthFrameElement(
                t_image_->width,
                t_image_->height,
                sizeof(float),
                (float *) new_depth_buffer,
                (float *) new_ndepth_buffer);

        // Generate Color Image mat
        cv::Mat t_col;
        color_image_.copyTo(t_col);
        double currentTime = glfwGetTime();
        return new FrameElement(frame_idx, t_col, depth_content, currentTime);
    }

    void start(){
        int frame_count = 0;
        while(this->close_pipe_ == false){
//            BOOST_LOG_TRIVIAL(info) << "Receiving Simple frame: " << frame_count;
            if (dirty_config_) {
                config_ = new_config_;
                this->initialize();
                dirty_config_ = false;
            }


            FrameElement* frame_element = this->generateCurrentFrame(frame_count);
            out_queue_->push(frame_element);

            frame_count++;
            std::this_thread::sleep_for(std::chrono::milliseconds(1000/this->config_.updateRate_));
        }
    }

    SimpleImageProducerConfig getCurrentConfig(){
        return config_;
    }

    void NotifyTermination() {
        this->close_pipe_ = true;
    }
    void ClearData() {

    }
    void NotifyData(const SimpleImageProducerConfig &data){
        new_config_ = data;
        dirty_config_ = true; // Use Queue??
    }

    bool getTerminationStatus() {
        return this->close_pipe_;
    }

    /**
     * End Pipeline loop
     */
    void signalEnd(){
        this->close_pipe_ = true;
    }
};



#endif //PROJECT_EDGE_SIMPLE_IMAGE_PRODUCER_H
